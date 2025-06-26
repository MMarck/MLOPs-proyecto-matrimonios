import datetime
from airflow.decorators import dag, task

markdown_text = """
### Reentrenamiento automático del modelo de divorcios

Este DAG entrena un nuevo modelo con datos actualizados, evalúa contra el modelo actual en producción (`champion`) y lo reemplaza solo si el nuevo obtiene mejor `accuracy`.
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
    dag_id="reentrenamiento_modelo_divorcios",
    description="Reentrena el modelo y reemplaza a champion si el nuevo tiene mejor accuracy",
    doc_md=markdown_text,
    tags=["Reentrenamiento", "Divorcios", "MLflow"],
    default_args=default_args,
    catchup=False,
)
def dag_reentrenamiento():

    @task.virtualenv(
        task_id="train_challenger_model",
        requirements=["scikit-learn==1.3.2", "mlflow==2.10.2", "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def train_challenger():
        import datetime
        import mlflow
        import awswrangler as wr
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri("http://mlflow:5000")

        X_train = wr.s3.read_csv("s3://data/final/train/divorcios_X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/train/divorcios_y_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/divorcios_X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/divorcios_y_test.csv")

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

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test.to_numpy().ravel(), y_pred)

        mlflow.set_experiment("Divorcios Ecuador_2023")
        with mlflow.start_run(run_name='Challenger_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) as run:
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="divorcios_ecuador_modelo_prod"
            )

            run_id = run.info.run_id
            uri = f"runs:/{run_id}/model"
        
        return {"accuracy": accuracy, "model_uri": uri}

    @task.virtualenv(
        task_id="compare_and_promote_model",
        requirements=["mlflow==2.10.2"],
        system_site_packages=True
    )
    def compare_and_promote(challenger_info: dict):
        import mlflow
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import awswrangler as wr

        mlflow.set_tracking_uri("http://mlflow:5000")

        client = mlflow.MlflowClient()
        model_name = "divorcios_ecuador_modelo_prod"

        # Accuracy del challenger
        challenger_accuracy = challenger_info["accuracy"]
        challenger_uri = challenger_info["model_uri"]

        # Cargar champion actual
        try:
            version = client.get_model_version_by_alias(model_name, "champion")
            champion_model = mlflow.sklearn.load_model(version.source)
        except:
            print("No hay modelo 'champion'. Promocionando challenger directamente.")
            client.set_registered_model_alias(model_name, "champion", client.get_latest_versions(model_name, stages=["None"])[0].version)
            return

        # Cargar datos de prueba
        X_test = wr.s3.read_csv("s3://data/final/test/divorcios_X_test.csv")
        y_test = wr.s3.read_csv("s3://data/final/test/divorcios_y_test.csv")

        y_pred_champion = champion_model.predict(X_test)
        champion_accuracy = accuracy_score(y_test.to_numpy().ravel(), y_pred_champion)

        if challenger_accuracy > champion_accuracy:
            print(f"Nuevo modelo mejor: {challenger_accuracy:.4f} > {champion_accuracy:.4f}. Promoviendo...")
            run_id = challenger_uri.split("/")[1]
            version = client.get_model_version(model_name, run_id=run_id).version
            client.set_registered_model_alias(model_name, "champion", version)
        else:
            print(f"Champion sigue siendo mejor: {champion_accuracy:.4f} ≥ {challenger_accuracy:.4f}")

    # Dependencia entre tareas
    challenger_info = train_challenger()
    compare_and_promote(challenger_info)

my_dag = dag_reentrenamiento()
