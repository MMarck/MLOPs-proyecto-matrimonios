from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.s3_key_sensor import S3KeySensor
import os
import sys

# Agregar directorio src al path
sys.path.append('/opt/airflow/src')

# Configuración del DAG
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

dag = DAG(
    'divorce_prediction_training',
    default_args=default_args,
    description='Entrenamiento automático del modelo de predicción de divorcios',
    schedule_interval='@weekly',  # Ejecutar semanalmente
    max_active_runs=1,
    tags=['machine-learning', 'divorce-prediction', 'ecuador']
)

def check_data_quality():
    """Verificar calidad de los datos"""
    import pandas as pd
    import logging
    
    logging.info("🔍 Verificando calidad de los datos...")
    
    # Cargar dataset
    data_path = '/opt/airflow/data/dataset_combinado_edit.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset no encontrado en {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Verificaciones básicas
    assert len(df) > 1000, f"Dataset muy pequeño: {len(df)} registros"
    assert 'es_divorcio' in df.columns, "Columna target 'es_divorcio' faltante"
    
    # Verificar distribución de clases
    target_dist = df['es_divorcio'].value_counts()
    minority_pct = min(target_dist.values) / len(df) * 100
    
    if minority_pct < 1:
        raise ValueError(f"Clase minoritaria muy pequeña: {minority_pct:.2f}%")
    
    # Verificar columnas críticas
    critical_cols = ['edad_1', 'edad_2', 'sexo_1', 'sexo_2', 'prov_insc']
    missing_critical = [col for col in critical_cols if col not in df.columns]
    
    if missing_critical:
        raise ValueError(f"Columnas críticas faltantes: {missing_critical}")
    
    logging.info(f"✅ Datos válidos: {len(df):,} registros, {len(df.columns)} columnas")
    return f"Dataset válido: {len(df):,} registros"

def train_model():
    """Ejecutar entrenamiento del modelo"""
    import subprocess
    import logging
    
    logging.info("🤖 Iniciando entrenamiento del modelo...")
    
    # Configurar variables de entorno
    env = os.environ.copy()
    env.update({
        'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
        'AWS_ACCESS_KEY_ID': 'minioadmin',
        'AWS_SECRET_ACCESS_KEY': 'minioadmin123',
        'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000'
    })
    
    # Ejecutar script de entrenamiento
    result = subprocess.run(
        ['python', '/opt/airflow/src/train_model.py'],
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        logging.error(f"Error en entrenamiento: {result.stderr}")
        raise RuntimeError(f"Entrenamiento falló: {result.stderr}")
    
    logging.info("✅ Entrenamiento completado exitosamente")
    return "Modelo entrenado exitosamente"

def validate_model():
    """Validar modelo entrenado"""
    import mlflow
    from mlflow.tracking import MlflowClient
    import logging
    
    logging.info("🧪 Validando modelo entrenado...")
    
    # Configurar MLflow
    mlflow.set_tracking_uri('http://mlflow:5000')
    client = MlflowClient()
    
    # Verificar que existe el modelo
    try:
        models = client.search_registered_models()
        divorce_models = [m for m in models if m.name == "DivorcePredictor"]
        
        if not divorce_models:
            raise ValueError("Modelo 'DivorcePredictor' no encontrado")
        
        # Verificar métricas mínimas
        latest_versions = client.get_latest_versions("DivorcePredictor")
        if not latest_versions:
            raise ValueError("No hay versiones del modelo")
        
        latest_version = latest_versions[0]
        run = client.get_run(latest_version.run_id)
        
        accuracy = run.data.metrics.get('test_accuracy', 0)
        f1_score = run.data.metrics.get('test_f1_score', 0)
        
        # Umbrales mínimos de calidad
        if accuracy < 0.7:
            raise ValueError(f"Accuracy muy baja: {accuracy:.3f} < 0.7")
        
        if f1_score < 0.5:
            raise ValueError(f"F1 Score muy bajo: {f1_score:.3f} < 0.5")
        
        logging.info(f"✅ Modelo válido - Accuracy: {accuracy:.3f}, F1: {f1_score:.3f}")
        return f"Modelo válido: Accuracy={accuracy:.3f}, F1={f1_score:.3f}"
        
    except Exception as e:
        logging.error(f"Error validando modelo: {e}")
        raise

def promote_to_production():
    """Promover modelo a producción"""
    import mlflow
    from mlflow.tracking import MlflowClient
    import logging
    
    logging.info("🚀 Promoviendo modelo a producción...")
    
    mlflow.set_tracking_uri('http://mlflow:5000')
    client = MlflowClient()
    
    try:
        # Obtener última versión
        latest_versions = client.get_latest_versions("DivorcePredictor")
        if not latest_versions:
            raise ValueError("No hay versiones para promover")
        
        latest_version = latest_versions[0]
        
        # Promover a Production
        client.transition_model_version_stage(
            name="DivorcePredictor",
            version=latest_version.version,
            stage="Production"
        )
        
        logging.info(f"✅ Modelo v{latest_version.version} promovido a Production")
        return f"Modelo v{latest_version.version} en Production"
        
    except Exception as e:
        logging.error(f"Error promoviendo modelo: {e}")
        raise

def notify_completion():
    """Notificar completion del pipeline"""
    import requests
    import logging
    
    logging.info("📧 Enviando notificación de completion...")
    
    # Aquí puedes agregar notificaciones (Slack, email, etc.)
    # Por ahora solo log
    message = {
        "pipeline": "divorce_prediction_training",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "message": "Modelo de predicción de divorcios entrenado y desplegado exitosamente"
    }
    
    logging.info(f"✅ Pipeline completado: {message}")
    return "Notificación enviada"

# Definir tareas
check_data_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

promote_model_task = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_to_production,
    dag=dag
)

# Reiniciar API para cargar nuevo modelo
restart_api_task = BashOperator(
    task_id='restart_api',
    bash_command='curl -X POST http://divorce-api:8000/reload-model',
    dag=dag
)

notify_task = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag
)

# Definir dependencias
check_data_task >> train_model_task >> validate_model_task >> promote_model_task >> restart_api_task >> notify_task