import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

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


class ModelInput(BaseModel):
    """
    Input schema for the heart disease prediction model.

    This class defines the input fields required by the heart disease prediction model along with their descriptions
    and validation constraints.

    :param age: Age of the patient (0 to 150).
    :param sex: Sex of the patient. 1: male; 0: female.
    :param cp: Chest pain type. 1: typical angina; 2: atypical angina; 3: non-anginal pain; 4: asymptomatic.
    :param trestbps: Resting blood pressure in mm Hg on admission to the hospital (90 to 220).
    :param chol: Serum cholestoral in mg/dl (110 to 600).
    :param fbs: Fasting blood sugar. 1: >120 mg/dl; 0: <120 mg/dl.
    :param restecg: Resting electrocardiographic results. 0: normal; 1: having ST-T wave abnormality; 2: showing
                    probable or definite left ventricular hypertrophy.
    :param thalach: Maximum heart rate achieved (beats per minute) (50 to 210).
    :param exang: Exercise induced angina. 1: yes; 0: no.
    :param oldpeak: ST depression induced by exercise relative to rest (0.0 to 7.0).
    :param slope: The slope of the peak exercise ST segment. 1: upsloping; 2: flat; 3: downsloping.
    :param ca: Number of major vessels colored by flourosopy (0 to 3).
    :param thal: Thalassemia disease. 3: normal; 6: fixed defect; 7: reversable defect.
    """

    age: int = Field(
        description="Age of the patient",
        ge=0,
        le=150,
    )
    sex: int = Field(
        description="Sex of the patient. 1: male; 0: female",
        ge=0,
        le=1,
    )
    cp: int = Field(
        description="Chest pain type. 1: typical angina; 2: atypical angina, 3: non-anginal pain; 4: asymptomatic",
        ge=1,
        le=4,
    )
    trestbps: float = Field(
        description="Resting blood pressure in mm Hg on admission to the hospital",
        ge=90,
        le=220,
    )
    chol: float = Field(
        description="Serum cholestoral in mg/dl",
        ge=110,
        le=600,
    )
    fbs: int = Field(
        description="Fasting blood sugar. 1: >120 mg/dl; 0: <120 mg/dl",
        ge=0,
        le=1,
    )
    restecg: int = Field(
        description="Resting electrocardiographic results. 0: normal; 1:  having ST-T wave abnormality (T wave "
                    "inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite "
                    "left ventricular hypertrophy by Estes' criteria",
        ge=0,
        le=2,
    )
    thalach: float = Field(
        description="Maximum heart rate achieved (beats per minute)",
        ge=50,
        le=210,
    )
    exang: int = Field(
        description="Exercise induced angina. 1: yes; 0: no",
        ge=0,
        le=1,
    )
    oldpeak: float = Field(
        description="ST depression induced by exercise relative to rest",
        ge=0.0,
        le=7.0,
    )
    slope: int = Field(
        description="The slope of the peak exercise ST segment .1: upsloping; 2: flat, 3: downsloping",
        ge=1,
        le=3,
    )
    ca: int = Field(
        description="Number of major vessels colored by flourosopy",
        ge=0,
        le=3,
    )
    thal: Literal[3, 6, 7] = Field(
        description="Thalassemia disease. 3: normal; 6: fixed defect; 7: reversable defect",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 67,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 160.0,
                    "chol": 286.0,
                    "fbs": 0,
                    "restecg": 2,
                    "thalach": 108.0,
                    "exang": 1,
                    "oldpeak": 1.5,
                    "slope": 2,
                    "ca": 3,
                    "thal": 3,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the heart disease prediction model.

    This class defines the output fields returned by the heart disease prediction model along with their descriptions
    and possible values.

    :param int_output: Output of the model. True if the patient has a heart disease.
    :param str_output: Output of the model in string form. Can be "Healthy patient" or "Heart disease detected".
    """

    int_output: bool = Field(
        description="Output of the model. True if the patient has a heart disease",
    )
    str_output: Literal["Healthy patient", "Heart disease detected"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "Heart disease detected",
                }
            ]
        }
    }


# Load the model before start
model, version_model = load_model("divorcios_ecuador_modelo_prod", "champion")

print(f"Model loaded: {model.__str__()}")
print(f"Model version: {version_model}")
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
    # features: Annotated[
    #     ModelInput,
    #     Body(embed=True),
    # ],
    # background_tasks: BackgroundTasks
):
    """
    Endpoint para predecir un divoricio en parejas ecuatorianas.

    Este endpoint recibe características relacionadas a un matrimonio enlistado en el registro civil ecuatoriano 
    y predice si la pareja tiene una alta probabilidad de divorciarse o no utilizando un modelo entrenado.
    Devuelve un porcentaje de probabilidad de divorcio y un mensaje indicando si la pareja tiene una alta probabilidad
    de divorcio o no.
    """

    # # Extract features from the request and convert them into a list and dictionary
    # features_list = [*features.dict().values()]
    # features_key = [*features.dict().keys()]

    # # Convert features into a pandas DataFrame
    # features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # # Process categorical features
    # for categorical_col in data_dict["categorical_columns"]:
    #     features_df[categorical_col] = features_df[categorical_col].astype(int)
    #     categories = data_dict["categories_values_per_categorical"][categorical_col]
    #     features_df[categorical_col] = pd.Categorical(features_df[categorical_col], categories=categories)

    # # Convert categorical features into dummy variables
    # features_df = pd.get_dummies(data=features_df,
    #                              columns=data_dict["categorical_columns"],
    #                              drop_first=True)

    # # Reorder DataFrame columns
    # features_df = features_df[data_dict["columns_after_dummy"]]

    # # Scale the data using standard scaler
    # features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    # # Make the prediction using the trained model
    # prediction = model.predict(features_df)

    # # Convert prediction result into string format
    # str_pred = "Healthy patient"
    # if prediction[0] > 0:
    #     str_pred = "Heart disease detected"

    # # Check if the model has changed asynchronously
    # background_tasks.add_task(check_model)

    # Return the prediction result
    # return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
    return [model.__str__(), version_model.__str__()]
