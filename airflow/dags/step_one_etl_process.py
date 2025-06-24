import datetime
import pandas as pd
from airflow.decorators import dag, task

markdown_text = """
### Proceso de TEL para el dataset de matrimonimos/divorcios Ecuador 2023

Divide el archivo dataset_combinado.csv en un conjunto de entrenamiento y otro de prueba,
con una proporción de 70/30, y los guarda en un bucket de S3.
Ademas se almacena el dataset original en un bucket de S3.

"""


default_args = {
    'owner': "Yandri Uchuari y Marck Murillo",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=15),
    'dagrun_timeout': datetime.timedelta(minutes=30)
}


@dag(
    dag_id="process_etl_divorcios_ecuador_2023",
    description="Proceso de ETL para el dataset de matrimonios/divorcios Ecuador 2023",
    doc_md=markdown_text,
    tags=["ETL", "Matrimonios", "Divorcios", "Ecuador"],
    default_args=default_args,
    catchup=False,
)
def process_etl_dataset_div_mat():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["ucimlrepo==0.0.3",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def obtener_datos():
        """
        Carga los datos limpios desde un archivo CSV
        """
        import awswrangler as wr
        from ucimlrepo import fetch_ucirepo
        from airflow.models import Variable
        import pandas as pd

        # fetch dataset

        data_path = "s3://data/raw/dataset_combinado.csv"
        dataframe = pd.read_csv("/opt/airflow/dataset_combinado.csv", low_memory=True)

        # formatear columna objetivo
        dataframe['es_divorcio'] = dataframe['es_divorcio'].replace({True: 1, False: 0})

        wr.s3.to_csv(df=dataframe,
                     path=data_path,
                     index=False)


    @task.virtualenv(
        task_id="limpieza_dataset",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
        execution_timeout=datetime.timedelta(minutes=60)
    )
    def limpieza_dataset():
        """
        Limpa el dataset de matrimonios/divorcios Ecuador 2023
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow
        import joblib
        from io import BytesIO
        import pickle

        import awswrangler as wr
        import pandas as pd
        import numpy as np

        from sklearn.preprocessing import LabelEncoder
        from airflow.models import Variable



        data_original_path = "s3://data/raw/dataset_combinado.csv"
        data_end_path = "s3://data/raw/dataset_limpio.csv"
        dataset = wr.s3.read_csv(data_original_path)

        # formateo de los tipos de datos 
        tipos_de_datos = {
            "dur_mat": np.int64,
            "hijos_2": np.int64,
            "edad_2": np.int64,
            "edad_1": np.int64,
            "area_1": str,
            "area_2": str,
            "hijos_rec": np.int64,
            "mcap_bie": str,
            "p_etnica1": str,
            "cant_insc": str,
            "p_etnica2": str,
            "cant_hab1": str,
            "cant_hab2": str,
            "sabe_leer1": str,
            "sabe_leer2": str,
            "mes_nac1": str,
            "niv_inst2": str,
            "prov_insc": str,
            "mes_nac2": str,
            "es_divorcio": bool
        }

        # Convertir los tipos de datos
        df_clean = dataset.astype(tipos_de_datos)

        # Definicion de las columnas a limpiar
        df_clean = df_clean.loc[:,[
            'dur_mat',
            'hijos_2',
            'edad_2',
            'edad_1',
            'area_1',
            'area_2',
            'hijos_rec',
            'mcap_bie',
            'p_etnica1',
            'cant_insc',
            'p_etnica2',
            'cant_hab1',
            'cant_hab2',
            'sabe_leer2',
            'sabe_leer1',
            'mes_nac1',
            'niv_inst2',
            'prov_insc',
            'mes_nac2',
            'es_divorcio'
            ]]

        # codificar las variables categoricas
        df_clean = df_clean.replace({True: 1, False: 0})

        categorical_cols = df_clean.select_dtypes(include=['object']).columns

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            label_encoders[col] = le


        # no se realiza limpieza de datos nulos ni duplicados, 
        # ya que el dataset_combinado original no tiene.
        # se espera agregar otro DAG para la limpieza de datos crudos.


        wr.s3.to_csv(df=df_clean,
                     path=data_end_path,
                     index=False)

        # Save information of the dataset
        client = boto3.client('s3')


        # # Guardar los label_encoders en un archivo local
        pickle_buffer = BytesIO()
        pickle.dump(label_encoders, pickle_buffer)
        pickle_buffer.seek(0)  # Volver al inicio del buffer
        
        # Extraer bucket y clave del path
        s3_path = "s3://data/data_info/label_encoders.pkl"
        s3_path = s3_path[5:]  # Eliminar "s3://"
        bucket, key = s3_path.split("/", 1)
        
        # Subir el archivo a S3
        client.put_object(Bucket=bucket, Key=key, Body=pickle_buffer.getvalue())

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e

        # para usar esta variable, se debe declarar en el documento 
        # variables.yaml de Airflow
        target_col = Variable.get("target_col_divorcios")
        dataset_log = dataset.drop(columns=[target_col])
        dataset_cleaned_log = df_clean.drop(columns=[target_col])


        categories_list = ["area_1", "area_2", "mcap_bie", "p_etnica1", "cant_insc", "p_etnica2"]

        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_after_clean'] = dataset_cleaned_log.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = categories_list
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_clean'] = {k: str(v) for k, v in dataset_cleaned_log.dtypes
                                                                                                 .to_dict()
                                                                                                 .items()}

        category_dummies_dict = {}
        for category in categories_list:
            category_dummies_dict[category] = np.sort(dataset_log[category].unique()).tolist()

        data_dict['categories_values_per_categorical'] = category_dummies_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Divorcios Ecuador")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "matrimonios_divorcios_ecuador_2023"},
                         log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(dataset.sample(n=100, random_state=42),
                                                 source="https://www.ecuadorencifras.gob.ec/matrimonios-divorcios/",
                                                 targets=target_col,
                                                 name="divorcios_matromonios_ecuador_2023")
        mlflow_dataset_cleaned = mlflow.data.from_pandas(df_clean.sample(n=100, random_state=42),
                                                         source="https://www.ecuadorencifras.gob.ec/matrimonios-divorcios/",
                                                         targets=target_col,
                                                         name="divorcios_matromonios_ecuador_2023_limpio")
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_cleaned, context="Dataset")

        

    @task.virtualenv(
        task_id="dividir_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def dividir_dataset():
        """
        Crea un conjunto de datos dividido en una parte de entrenamiento y una parte de prueba
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        data_original_path = "s3://data/raw/dataset_limpio.csv"
        dataset = wr.s3.read_csv(data_original_path)

        target_col = Variable.get("target_col_divorcios")

        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        # Clean duplicates
        # dataset.drop_duplicates(inplace=True, ignore_index=True)

        save_to_csv(X_train, "s3://data/final/train/divorcios_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/divorcios_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/divorcios_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/divorcios_y_test.csv")

    obtener_datos() >> limpieza_dataset() >> dividir_dataset() 


dag = process_etl_dataset_div_mat()