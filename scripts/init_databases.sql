-- Crear bases de datos para MLflow y Airflow
CREATE DATABASE mlflow_db;
CREATE DATABASE airflow_db;

-- Crear usuario para MLflow
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

-- Crear usuario para Airflow
CREATE USER airflow_user WITH PASSWORD 'airflow_password';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

-- Configurar permisos
\c mlflow_db;
GRANT ALL ON SCHEMA public TO mlflow_user;

\c airflow_db;
GRANT ALL ON SCHEMA public TO airflow_user;