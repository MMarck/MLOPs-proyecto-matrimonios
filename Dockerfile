# Dockerfile para API de Predicción de Divorcios - Ecuador
FROM python:3.9-slim

# Metadatos
LABEL maintainer="divorce-prediction-ecuador"
LABEL description="API de predicción de divorcios basada en datos del Registro Civil de Ecuador"
LABEL version="2.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear usuario no-root para seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiar y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/

# Crear directorios necesarios
RUN mkdir -p /app/data \
    && mkdir -p /app/models \
    && mkdir -p /app/logs \
    && mkdir -p /app/temp \
    && chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Variables de entorno por defecto
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_NAME=DivorcePredictor
ENV MODEL_STAGE=Production
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin123
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_DEFAULT_REGION=us-east-1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]