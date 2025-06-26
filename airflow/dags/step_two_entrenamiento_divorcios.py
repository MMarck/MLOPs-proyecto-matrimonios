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
    description="Entrena el modelo con nuevos datos y lo registra en MLflow",
    doc_md=markdown_text,
    tags=["Train", "Divorcios", "Ecuador"],
    default_args=default_args,
    catchup=False,
)
def dag_entrenamiento_modelo():

    @task.virtualenv(
        task_id="entrenar_y_registrar_modelo",
        requirements=["scikit-learn==1.3.2", "mlflow==2.10.2", "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def entrenar_y_registrar_modelo():
        import datetime
        import mlflow
        import awswrangler as wr
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        from mlflow.models import infer_signature
        from mlflow.exceptions import MlflowException
        import joblib
        import json
        import os

        mlflow.set_tracking_uri('http://mlflow:5000')

        # Cargar datos
        X_train = wr.s3.read_csv("s3://data/final/train/divorcios_X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/train/divorcios_y_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/divorcios_X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/divorcios_y_test.csv")

        # Entrenar modelo
        model = RandomForestClassifier(
            n_estimators=700,
            min_samples_split=10,
            min_samples_leaf=2,
            max_features='sqrt',
            max_depth=30,
            bootstrap=False,
            random_state=42
        )
        model.fit(X_train, y_train.to_numpy().ravel())

        # Evaluar modelo
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test.to_numpy().ravel(), y_pred)
        accuracy = accuracy_score(y_test.to_numpy().ravel(), y_pred)
        precision = precision_score(y_test.to_numpy().ravel(), y_pred)
        recall = recall_score(y_test.to_numpy().ravel(), y_pred)

        # Registrar en MLflow
        experiment = mlflow.set_experiment("Divorcios Ecuador_2023")
        with mlflow.start_run(run_name='modelo_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) as run:
            mlflow.log_params(model.get_params())
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            # Firma del modelo
            signature = infer_signature(X_train, model.predict(X_train))
            joblib.dump(model, "model.pkl")

            mlflow.log_artifact("model.pkl", artifact_path="model")
            result = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="divorcios_ecuador_modelo_prod"
            )
            os.remove("model.pkl")

            # Establecer alias como 'champion'
            client = mlflow.MlflowClient()
            try:
                client.get_registered_model("divorcios_ecuador_modelo_prod")
            except MlflowException:
                client.create_registered_model("divorcios_ecuador_modelo_prod")

            version = client.get_latest_versions("divorcios_ecuador_modelo_prod", stages=[])[-1].version
            client.set_registered_model_alias("divorcios_ecuador_modelo_prod", "champion", version)

    entrenar_y_registrar_modelo()

my_dag = dag_entrenamiento_modelo()
