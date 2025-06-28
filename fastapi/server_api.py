import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd
import awswrangler as wr
from io import BytesIO

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Carga el modelo entrenado y el diccionario de datos asociado.

    Esta función intenta cargar un modelo entrenado especificado por su nombre y alias. Si el modelo no se encuentra en el
    registro de MLflow, carga el modelo predeterminado desde un archivo. Además, carga información sobre el pipeline de ETL
    desde un bucket de S3. Si el diccionario de datos no se encuentra en el bucket de S3, lo carga desde un archivo local.

    :param model_name: El nombre del modelo.
    :param alias: El alias de la versión del modelo.
    :return: Una tupla que contiene el modelo cargado, su versión y el diccionario de datos.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except Exception as e:
        print("Model not found in MLflow, loading default model... ")
        # If there is no registry in MLflow, open the default model
        # file_ml = open('/opt/fastapi/model.pkl', 'rb')
        # model_ml = pickle.load(file_ml)
        # file_ml.close()
        # version_model_ml = 0

    # try:
    #     # Load information of the ETL pipeline from S3
    #     s3 = boto3.client('s3')

    #     s3.head_object(Bucket='data', Key='data_info/data.json')
    #     result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
    #     text_s3 = result_s3["Body"].read().decode()
    #     data_dictionary = json.loads(text_s3)
    #     print(f"Data dictionary loaded from S3: {data_dictionary.keys()}")

    #     data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
    #     data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    # except:
    #     # If data dictionary is not found in S3, load it from local file
    #     print("Data dictionary not found in S3, loading default data dictionary...")
    #     # file_s3 = open('/app/files/data.json', 'r')
    #     # data_dictionary = json.load(file_s3)
    #     # file_s3.close()

    return model_ml, version_model_ml


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "heart_disease_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass

def load_label_encoders(path: str = "s3://data/data_info/label_encoders.pkl"):
    """
    Load label encoders from a file.

    This function attempts to load label encoders from a file named 'label_encoders.pkl'. If the file does not exist,
    it returns an empty dictionary.

    :return: A dictionary containing the label encoders.
    """

    # Extraer bucket y clave del path
    if path.startswith("s3://"):
        s3_path = path[5:]  # Eliminar "s3://"
        bucket, key = s3_path.split("/", 1)
    else:
        raise ValueError("El path debe comenzar con 's3://'")

    # Crear cliente de S3
    s3_client = boto3.client("s3")
    
    # Descargar el archivo como un objeto binario
    response = s3_client.get_object(Bucket=bucket, Key=key)
    pickle_data = response["Body"].read()
    
    # Cargar el archivo pickle desde los datos binarios
    label_encoders = pickle.load(BytesIO(pickle_data))
    
    return label_encoders
     
    # try:
    #     path = "s3://data/data_info/label_encoders.pkl"
        
    #     label_encoders = pickle.load(label_encoders_pickle)

    #     return label_encoders
    # except FileNotFoundError:
    #     print(f"Label encoders file not found at {path}. Returning empty dictionary.")
    #     return {}

class ModelInput(BaseModel):
    """
    Esquema de entrada para el modelo con parámetros personalizados.

    Esta clase define los campos de entrada requeridos para el modelo, con sus descripciones
    y restricciones de validación. Todos los campos son obligatorios y de tipo entero (long).

    :param dur_mat: Duración del matrimonio en años.
    :param hijos_2: Número de hijos de la segunda persona.
    :param edad_2: Edad de la segunda persona.
    :param edad_1: Edad de la primera persona.
    :param area_1: Área de residencia de la primera persona (código numérico).
    :param area_2: Área de residencia de la segunda persona (código numérico).
    :param hijos_rec: Número de hijos reconocidos.
    :param mcap_bie: Indicador de manejo de bienes (código numérico).
    :param p_etnica1: Pertenencia étnica de la primera persona (código numérico).
    :param cant_insc: Cantidad de inscripciones.
    :param p_etnica2: Pertenencia étnica de la segunda persona (código numérico).
    :param cant_hab1: Cantidad de habitantes en el hogar de la primera persona.
    :param cant_hab2: Cantidad de habitantes en el hogar de la segunda persona.
    :param sabe_leer2: Indicador de alfabetismo de la segunda persona (1: sí, 0: no).
    :param sabe_leer1: Indicador de alfabetismo de la primera persona (1: sí, 0: no).
    :param mes_nac1: Mes de nacimiento de la primera persona (1 a 12).
    :param niv_inst2: Nivel de instrucción de la segunda persona (código numérico).
    :param prov_insc: Provincia de inscripción (código numérico).
    :param mes_nac2: Mes de nacimiento de la segunda persona (1 a 12).
    """

    dur_mat: int = Field(
        description="Duración del matrimonio en años",
        ge=0,
        le=65
    )
    hijos_2: int = Field(
        description="Número de hijos de la segunda persona",
        ge=0,
        le=150
    )
    edad_2: int = Field(
        description="Edad de la segunda persona",
        ge=0,
        le=120
    )
    edad_1: int = Field(
        description="Edad de la primera persona",
        ge=0,
        le=120
    )
    area_1: str = Field( #['Urbana', 'Rural']
        description="Área de residencia de la primera persona (código numérico)",
    )
    area_2: str = Field( #['Urbana', 'Rural']
        description="Área de residencia de la segunda persona (código numérico)",
    )
    hijos_rec: int = Field(
        description="Número de hijos reconocidos",
        ge=0,
        le=12
    )
    mcap_bie: str = Field( #['No', 'Si']
        description="Indicador de manejo de bienes (código numérico)",
    )
    p_etnica1: str = Field( #['Mestizo', 'Indígena']
        description="Pertenencia étnica de la primera persona (código numérico)",
    )
    cant_insc: str = Field( # ['Guayaquil', 'Quito']
        description="Cantidad de inscripciones",
    )
    p_etnica2: str = Field( #['Mestizo', 'Indígena']
        description="Pertenencia étnica de la segunda persona (código numérico)",
    )
    cant_hab1: str = Field( #['Quito', 'Guayaquil']
        description="Cantidad de habitantes en el hogar de la primera persona",
    )
    cant_hab2: str = Field( #['Quito', 'Guayaquil']
        description="Cantidad de habitantes en el hogar de la segunda persona",
    )
    sabe_leer2: str = Field( #['No', 'Si']
        description="Indicador de alfabetismo de la segunda persona (1: sí, 0: no)",
    )
    sabe_leer1: str = Field( #['No', 'Si']
        description="Indicador de alfabetismo de la primera persona (1: sí, 0: no)",
    )
    mes_nac1: str = Field( # ['Agosto', 'Septiembre']
        description="Mes de nacimiento de la primera persona (1 a 12)",
    )
    niv_inst2: str = Field( #['Educación media / Bachillerato', 'Superior Universitaria']
        description="Nivel de instrucción de la segunda persona (código numérico)",
    )
    prov_insc: str = Field( #['Guayas', 'Pichincha']
        description="Provincia de inscripción (código numérico)",
    )
    mes_nac2: str = Field( # ['Agosto', 'Septiembre']
        description="Mes de nacimiento de la segunda persona (1 a 12)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                # ejemplo de entrada con prediccion de Divorcio
                {
                    "dur_mat": 10,
                    "hijos_2": 2,
                    "edad_2": 35,
                    "edad_1": 38,
                    "area_1": "Urbana",
                    "area_2": "Rural",
                    "hijos_rec": 2,
                    "mcap_bie": "No",
                    "p_etnica1": "Mestizo",
                    "cant_insc": "Guayaquil",
                    "p_etnica2": "Mestizo",
                    "cant_hab1": "Quito",
                    "cant_hab2": "Guayaquil",
                    "sabe_leer2": "Si",
                    "sabe_leer1": "Si",
                    "mes_nac1": "Junio",
                    "niv_inst2": "Superior Universitaria",
                    "prov_insc": "Pichincha",
                    "mes_nac2": "Agosto"
                },
                # ejemplo de entrada con prediccion de No divorcio
                {
                    "dur_mat": 0,
                    "hijos_2": 0,
                    "edad_2": 28,
                    "edad_1": 23,
                    "area_1": "Urbana",
                    "area_2": "Urbana",
                    "hijos_rec": 2,
                    "mcap_bie": "No",
                    "p_etnica1": "Mestizo",
                    "cant_insc": "Guayaquil",
                    "p_etnica2": "Mestizo",
                    "cant_hab1": "Quito",
                    "cant_hab2": "Guayaquil",
                    "sabe_leer2": "Si",
                    "sabe_leer1": "Si",
                    "mes_nac1": "Junio",
                    "niv_inst2": "Superior Universitaria",
                    "prov_insc": "Pichincha",
                    "mes_nac2": "Agosto"
                }
            ]
        }
    }
class ModelOutput(BaseModel):
    """
    Esquema de salida para el modelo de predicción de enfermedades cardíacas.

    Esta clase define los campos de salida del modelo de predicción de enfermedades cardíacas,
    junto con sus descripciones y valores posibles.

    :param int_output: Salida del modelo entre con un valor entre 0 o 1, donde indica si representa o no un divorcio.
    :param str_output: Salida del modelo en forma de cadena. Puede ser "No divorcio" o "Divorcio".
    """

    int_output: int = Field(
        description="",
    )
    str_output: Literal["No divorcio", "Divorcio"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": 0,
                    "str_output": "No divorcio",
                }
            ]
        }
    }


# Load the model before start
model, version_model = load_model("divorcios_ecuador_modelo_prod", "champion")

label_encoders = load_label_encoders("s3://data/data_info/label_encoders.pkl")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Ruta raíz de la API de prediccion de un posible divorcio para parejas ecuatorianas.

    Este endpoint devuelve una respuesta JSON con un mensaje de bienvenida para indicar que la API está en funcionamiento.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Bienvenido a la API de predicción de posibles divorcios para parejas ecuatorianas"}))


@app.post("/predict/", response_model=[])
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint para predecir un divoricio en parejas ecuatorianas.

    Este endpoint recibe características relacionadas a un matrimonio enlistado en el registro civil ecuatoriano 
    y predice si la pareja tiene una Divorcio de divorciarse o no utilizando un modelo entrenado.
    Devuelve un porcentaje de probabilidad de divorcio y un mensaje indicando si la pareja tiene una Divorcio
    de divorcio o no.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]


    # Codificar los datos usando los label_encoders
    encoded_features = []
    for key, value in zip(features_key, features_list):
        if key in label_encoders:
            encoder = label_encoders[key]
            print(f"Encoding feature '{key}' with value '{value}' using encoder: {encoder}")
            encoded_value = encoder.transform([value])[0]
            print(f"Encoded value for feature '{key}': {encoded_value}")
            encoded_features.append(encoded_value)
        else:
            encoded_features.append(value)

    

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(encoded_features).reshape([1, -1]), columns=features_key)


    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "No divorcio"
    if prediction[0] > 0:
        str_pred = "Divorcio"

    # # Check if the model has changed asynchronously
    # background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=prediction[0], str_output=str_pred)
