import mlflow.sklearn
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
    try:
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        s3 = boto3.client('s3')
        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)
        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    global model
    global data_dict
    global version_model

    try:
        model_name = "divorcios_ecuador_modelo_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        if new_version_model != version_model:
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        pass


class ModelInput(BaseModel):
    Edad: float
    Tiempo_de_relacion: float
    Tiene_hijos_si: int
    Nivel_educativo_Superior: int
    Nivel_educativo_Tecnico: int
    Ocupacion_Independiente: int
    Ocupacion_Publico: int
    Satisfaccion_Matrimonial_Baja: int
    Satisfaccion_Matrimonial_Media: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Edad": 35,
                    "Tiempo_de_relacion": 10,
                    "Tiene_hijos_si": 1,
                    "Nivel_educativo_Superior": 1,
                    "Nivel_educativo_Tecnico": 0,
                    "Ocupacion_Independiente": 0,
                    "Ocupacion_Publico": 1,
                    "Satisfaccion_Matrimonial_Baja": 0,
                    "Satisfaccion_Matrimonial_Media": 1
                }
            ]
        }
    }



class ModelOutput(BaseModel):
    int_output: bool = Field(description="Predicción del modelo. 1 = divorcio probable")
    str_output: Literal["No se predice divorcio", "Divorcio probable"]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "Divorcio probable"
                }
            ]
        }
    }


model, version_model, data_dict = load_model("divorcios_ecuador_modelo_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    return JSONResponse(content=jsonable_encoder({"message": "API predictor de divorcios activo"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[ModelInput, Body(embed=True)],
    background_tasks: BackgroundTasks
):
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    if "categorical_columns" in data_dict:
        for categorical_col in data_dict["categorical_columns"]:
            features_df[categorical_col] = features_df[categorical_col].astype(int)
            categories = data_dict["categories_values_per_categorical"][categorical_col]
            features_df[categorical_col] = pd.Categorical(features_df[categorical_col], categories=categories)
        features_df = pd.get_dummies(data=features_df,
                                     columns=data_dict["categorical_columns"],
                                     drop_first=True)
        features_df = features_df[data_dict["columns_after_dummy"]]

    if "standard_scaler_mean" in data_dict:
        features_df = (features_df - data_dict["standard_scaler_mean"]) / data_dict["standard_scaler_std"]

    prediction = model.predict(features_df)

    str_pred = "No se predice divorcio"
    if prediction[0] > 0:
        str_pred = "Divorcio probable"

    background_tasks.add_task(check_model)

    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
