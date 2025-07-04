import datetime

from airflow.decorators import dag, task

markdown_text = """
### Entrenar el modelo para el predictor de divorcios para Ecuador con datos de 2023

Este DAG entrena el modelo basado en los datos iniciales cargados en minio y pone en producción el modelo
"""

default_args = {
    'owner': "Yandri Uchuari y Marck Murillo",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="entrenamiento_inicial_modelo",
    description="re-entrena el modelo basado en nuevos datos, prueba el modelo anterior y pone en producción el nuevo",
    doc_md=markdown_text,
    tags=["Train", "Entrenar", "Divorcios", "Ecuador"],
    default_args=default_args,
    catchup=False,
)
def dag_reentrenamiento_modelo():

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def train_the_challenger_model():
        import datetime
        import mlflow
        import awswrangler as wr

        from sklearn.base import clone
        from sklearn.metrics import f1_score
        from mlflow.models import infer_signature
        from sklearn.ensemble import RandomForestClassifier
        from mlflow.exceptions import MlflowException


        import joblib
        import os
        import json


        mlflow.set_tracking_uri('http://mlflow:5000')

        def load_the_initial_model():

            model_data = RandomForestClassifier( n_estimators=100,
                                        min_samples_split=10,
                                        min_samples_leaf=2,
                                        max_features='sqrt',
                                        max_depth=10,
                                        bootstrap=False,
                                        random_state=42)
            

            return model_data

        def load_the_train_test_data():
            X_train = wr.s3.read_csv("s3://data/final/train/divorcios_X_train.csv")
            y_train = wr.s3.read_csv("s3://data/final/train/divorcios_y_train.csv")
            X_test = wr.s3.read_csv("s3://data/final/test/divorcios_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/divorcios_y_test.csv")

            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):

            # Track the experiment
            experiment = mlflow.set_experiment("Divorcios Ecuador")

            mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                             experiment_id=experiment.experiment_id,
                             tags={"experiment": "challenger models", "dataset": "Divorcios Ecuador"},
                             log_system_metrics=True)

            params = model.get_params()
            params["model"] = type(model).__name__

            mlflow.log_params(params)

            # Save the artifact of the challenger model
            artifact_path = "model"
            signature_path = "signature.json"


            # Infer and save the model signature
            signature = infer_signature(X, model.predict(X))
            
            # Save the signature to a JSON file
            signature_path = "model_signature.json"
            with open(signature_path, "w") as f:
                json.dump(signature.to_dict(), f, indent=2)

            # Save the compressed model locally
            compressed_model_path = "compressed_model.pkl.gz"
            joblib.dump(model, compressed_model_path, compress=3)

            # Log the compressed model and signature as artifacts
            mlflow.log_artifact(compressed_model_path, artifact_path)
            mlflow.log_artifact(signature_path, artifact_path)

            # Optionally, clean up local files
            os.remove(compressed_model_path)
            os.remove(signature_path)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format='pickle',
                registered_model_name="divorcios_ecuador_model_dev",
                metadata={"model_data_version": 1}
            )

            # Obtain the model URI
            return mlflow.get_artifact_uri(artifact_path)

        def register_challenger(model, f1_score, model_uri):

            client = mlflow.MlflowClient()
            name = "divorcios_ecuador_modelo_prod"

            # Intenta crear el modelo registrado si no existe
            try:
                client.get_registered_model(name)
            except MlflowException:
                client.create_registered_model(name)

            # Save the model params as tags
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["f1-score"] = f1_score

            # Save the version of the model
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )

            # Save the alias as challenger
            client.set_registered_model_alias(name, "champion", result.version)

        # Load the champion model
        initial_model = load_the_initial_model()


        # Load the dataset
        X_train, y_train, X_test, y_test = load_the_train_test_data()

        # Fit the training model
        initial_model.fit(X_train, y_train.to_numpy().ravel())

        # Obtain the metric of the model
        y_pred = initial_model.predict(X_test)
        f1_score = f1_score(y_test.to_numpy().ravel(), y_pred)

        # Track the experiment
        artifact_uri = mlflow_track_experiment(initial_model, X_train)

        # Record the model
        register_challenger(initial_model, f1_score, artifact_uri)


    train_the_challenger_model() 


my_dag = dag_reentrenamiento_modelo()
