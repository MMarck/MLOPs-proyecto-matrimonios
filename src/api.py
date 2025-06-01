from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import pickle
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import asyncio

from models import (
    DivorceInputEcuador, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthResponse, ModelInfoResponse
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Divorcios - Ecuador",
    description="API de Machine Learning para predecir divorcios basada en datos del Registro Civil de Ecuador",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "General", "description": "Endpoints generales"},
        {"name": "Predicción", "description": "Endpoints de predicción"},
        {"name": "Modelo", "description": "Información del modelo"},
        {"name": "Monitoreo", "description": "Health checks y métricas"}
    ]
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
scaler = None
label_encoders = {}
feature_names = []
model_info = {
    "loaded": False,
    "model_name": "DivorcePredictor",
    "model_stage": "Production",
    "version": None,
    "accuracy": 0.0,
    "f1_score": 0.0,
    "roc_auc": 0.0,
    "run_id": None,
    "last_update": None,
    "error_message": None,
    "total_features": 0,
    "prediction_count": 0
}

# Configuración MLflow
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
mlflow.set_tracking_uri(mlflow_uri)

def setup_s3_credentials():
    """Configurar credenciales S3 para MLflow"""
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')

def load_model_from_mlflow():
    """Cargar modelo y artefactos desde MLflow"""
    global model, scaler, label_encoders, feature_names, model_info
    
    try:
        logger.info("🤖 Cargando modelo desde MLflow...")
        setup_s3_credentials()
        
        client = MlflowClient()
        
        # Intentar cargar modelo en Production
        try:
            model_uri = f"models:/DivorcePredictor/Production"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Obtener información del modelo en Production
            model_versions = client.get_latest_versions("DivorcePredictor", stages=["Production"])
            if model_versions:
                model_version = model_versions[0]
                model_info.update({
                    "version": model_version.version,
                    "run_id": model_version.run_id,
                    "stage": "Production"
                })
                logger.info(f"✅ Modelo Production v{model_version.version} cargado")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar modelo Production: {e}")
            
            # Intentar cargar última versión disponible
            latest_versions = client.get_latest_versions("DivorcePredictor")
            if latest_versions:
                latest_version = latest_versions[0]
                model_uri = f"models:/DivorcePredictor/{latest_version.version}"
                model = mlflow.sklearn.load_model(model_uri)
                
                model_info.update({
                    "version": latest_version.version,
                    "run_id": latest_version.run_id,
                    "stage": latest_version.current_stage
                })
                logger.info(f"✅ Modelo v{latest_version.version} cargado")
            else:
                logger.error("❌ No se encontraron versiones del modelo")
                return False
        
        # Cargar scaler
        try:
            if model_info["run_id"]:
                scaler_path = mlflow.artifacts.download_artifacts(
                    run_id=model_info["run_id"],
                    artifact_path="scaler.pkl"
                )
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info("✅ Scaler cargado")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar scaler: {e}")
        
        # Cargar label encoders
        try:
            if model_info["run_id"]:
                encoders_path = mlflow.artifacts.download_artifacts(
                    run_id=model_info["run_id"],
                    artifact_path="label_encoders.pkl"
                )
                with open(encoders_path, 'rb') as f:
                    label_encoders = pickle.load(f)
                logger.info(f"✅ Label encoders cargados: {len(label_encoders)}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron cargar label encoders: {e}")
        
        # Obtener métricas del modelo
        try:
            if model_info["run_id"]:
                run = client.get_run(model_info["run_id"])
                metrics = run.data.metrics
                model_info.update({
                    "accuracy": metrics.get("test_accuracy", 0.0),
                    "f1_score": metrics.get("test_f1_score", 0.0),
                    "roc_auc": metrics.get("test_roc_auc", 0.0),
                    "total_features": int(metrics.get("total_features", 0))
                })
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron cargar métricas: {e}")
        
        # Obtener nombres de características
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        elif hasattr(model, 'n_features_in_'):
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
        
        model_info.update({
            "loaded": True,
            "last_update": datetime.now().isoformat(),
            "error_message": None
        })
        
        logger.info("🎉 Modelo cargado exitosamente")
        return True
        
    except Exception as e:
        error_msg = f"Error cargando modelo: {e}"
        logger.error(error_msg)
        model_info.update({
            "loaded": False,
            "error_message": error_msg
        })
        return False

def preprocess_input(data: DivorceInputEcuador) -> np.ndarray:
    """Preprocesar entrada para predicción"""
    try:
        # Convertir a DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Manejar valores None/faltantes
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('Desconocido')
        
        # Aplicar label encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                value = str(df[col].iloc[0])
                
                if value in le.classes_:
                    df[col] = le.transform([value])[0]
                else:
                    # Usar primera clase como fallback
                    df[col] = le.transform([le.classes_[0]])[0]
                    logger.warning(f"Valor desconocido '{value}' para columna '{col}', usando '{le.classes_[0]}'")
            else:
                # Encoding numérico simple si no hay label encoder
                unique_vals = [value]
                df[col] = 0  # Default value
        
        # Aplicar scaling si está disponible
        if scaler:
            processed_data = scaler.transform(df)
        else:
            processed_data = df.values
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}")
        raise e

def calculate_risk_category(probability: float) -> str:
    """Calcular categoría de riesgo"""
    if probability >= 0.8:
        return "Muy Alto"
    elif probability >= 0.6:
        return "Alto"
    elif probability >= 0.4:
        return "Moderado"
    elif probability >= 0.2:
        return "Bajo"
    else:
        return "Muy Bajo"

def calculate_confidence_level(probability: float) -> str:
    """Calcular nivel de confianza"""
    confidence = max(probability, 1 - probability)
    if confidence >= 0.9:
        return "Muy Alta"
    elif confidence >= 0.8:
        return "Alta"
    elif confidence >= 0.7:
        return "Moderada"
    else:
        return "Baja"

def generate_interpretation_message(prediction: bool, probability: float, data: DivorceInputEcuador) -> str:
    """Generar mensaje interpretativo personalizado"""
    age_diff = abs(data.edad_1 - data.edad_2)
    years_married = getattr(data, 'anios_matrimonio', 0) or 0
    
    if prediction:
        message = f"Alto riesgo de divorcio ({probability:.1%}). "
        
        # Factores de riesgo
        risk_factors = []
        if age_diff > 10:
            risk_factors.append(f"gran diferencia de edad ({age_diff} años)")
        if years_married > 15:
            risk_factors.append(f"matrimonio de larga duración ({years_married} años)")
        if data.niv_inst1 != data.niv_inst2:
            risk_factors.append("diferencias en nivel educativo")
        
        if risk_factors:
            message += f"Factores influyentes: {', '.join(risk_factors)}."
    else:
        stability = 1 - probability
        message = f"Matrimonio estable ({stability:.1%} de estabilidad). "
        
        # Factores protectores
        protective_factors = []
        if age_diff <= 5:
            protective_factors.append("edades similares")
        if 5 <= years_married <= 10:
            protective_factors.append("duración matrimonial estable")
        if data.niv_inst1 == data.niv_inst2:
            protective_factors.append("nivel educativo similar")
        
        if protective_factors:
            message += f"Factores protectores: {', '.join(protective_factors)}."
    
    return message

# Eventos de startup y shutdown
@app.on_event("startup")
async def startup_event():
    """Inicializar aplicación"""
    logger.info("🚀 Iniciando API de Predicción de Divorcios - Ecuador")
    
    # Cargar modelo en background para no bloquear startup
    asyncio.create_task(load_model_async())

async def load_model_async():
    """Cargar modelo de forma asíncrona"""
    await asyncio.sleep(5)  # Esperar que otros servicios estén listos
    success = load_model_from_mlflow()
    if success:
        logger.info("✅ Modelo cargado en background")
    else:
        logger.error("❌ Error cargando modelo en background")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos"""
    logger.info("🛑 Cerrando API de Predicción de Divorcios")

# Endpoints principales
@app.get("/", response_class=HTMLResponse, tags=["General"])
async def root():
    """Página principal con información de la API"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Predicción Divorcios Ecuador</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            .header {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
            .status {{ padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .status.ok {{ background: #d4edda; color: #155724; }}
            .status.error {{ background: #f8d7da; color: #721c24; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric {{ background: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }}
            .links {{ margin-top: 30px; }}
            .links a {{ display: inline-block; margin: 5px 10px; padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 API de Predicción de Divorcios</h1>
                <h2>República del Ecuador</h2>
                <p>Basado en datos del Registro Civil</p>
            </div>
            
            <div class="status {'ok' if model_info['loaded'] else 'error'}">
                {'✅ Modelo cargado y listo' if model_info['loaded'] else '❌ Modelo no disponible'}
                {f" - Versión: {model_info['version']}" if model_info['version'] else ""}
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Accuracy</h3>
                    <p>{model_info['accuracy']:.1%}</p>
                </div>
                <div class="metric">
                    <h3>F1 Score</h3>
                    <p>{model_info['f1_score']:.3f}</p>
                </div>
                <div class="metric">
                    <h3>ROC AUC</h3>
                    <p>{model_info['roc_auc']:.3f}</p>
                </div>
                <div class="metric">
                    <h3>Predicciones</h3>
                    <p>{model_info['prediction_count']:,}</p>
                </div>
            </div>
            
            <div class="links">
                <h3>🔗 Enlaces Útiles:</h3>
                <a href="/docs">📚 Documentación API</a>
                <a href="/health">🏥 Health Check</a>
                <a href="/model-info">📊 Info del Modelo</a>
                <a href="{mlflow_uri}">🔬 MLflow UI</a>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #e7f3ff; border-radius: 5px;">
                <h3>📋 Información del Dataset</h3>
                <p><strong>Fuente:</strong> Registro Civil del Ecuador</p>
                <p><strong>Registros:</strong> ~70,000 matrimonios</p>
                <p><strong>Características:</strong> {model_info['total_features']} variables demográficas</p>
                <p><strong>Última actualización:</strong> {model_info.get('last_update', 'N/A')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse, tags=["Monitoreo"])
async def health_check():
    """Health check completo"""
    mlflow_connected = False
    
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        mlflow_connected = True
    except:
        mlflow_connected = False
    
    return HealthResponse(
        status="healthy" if model_info["loaded"] else "degraded",
        model_loaded=model_info["loaded"],
        mlflow_connected=mlflow_connected,
        model_info=model_info,
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Modelo"])
async def get_model_info():
    """Información detallada del modelo"""
    if not model_info["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible"
        )
    
    try:
        additional_info = {}
        
        if model_info["run_id"]:
            client = MlflowClient()
            run = client.get_run(model_info["run_id"])
            additional_info = {
                "training_metrics": dict(run.data.metrics),
                "training_params": dict(run.data.params),
                "training_date": run.info.start_time
            }
        
        return ModelInfoResponse(
            model_name=model_info["model_name"],
            model_stage=model_info["stage"],
            model_version=model_info["version"],
            accuracy=model_info["accuracy"],
            f1_score=model_info["f1_score"],
            roc_auc=model_info["roc_auc"],
            total_features=model_info["total_features"],
            run_id=model_info["run_id"],
            last_updated=model_info["last_update"],
            additional_info=additional_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo información: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict_divorce(data: DivorceInputEcuador):
    """
    Predecir probabilidad de divorcio para una pareja ecuatoriana
    
    Basado en características demográficas y socioeconómicas del Registro Civil
    """
    if not model_info["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verifica el health check."
        )
    
    try:
        # Preprocesar datos
        processed_data = preprocess_input(data)
        
        # Realizar predicción
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Calcular métricas adicionales
        risk_category = calculate_risk_category(probability)
        confidence_level = calculate_confidence_level(probability)
        interpretation = generate_interpretation_message(bool(prediction), probability, data)
        
        # Incrementar contador
        model_info["prediction_count"] += 1
        
        # Log para monitoreo
        logger.info(f"Predicción: {prediction} (prob: {probability:.3f}, riesgo: {risk_category})")
        
        return PredictionResponse(
            prediccion=bool(prediction),
            probabilidad=float(probability),
            categoria_riesgo=risk_category,
            nivel_confianza=confidence_level,
            interpretacion=interpretation,
            modelo_version=model_info["version"],
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predicción en lote para múltiples parejas (máximo 100)
    """
    if not model_info["loaded"]:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if len(request.predictions) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo 100 predicciones por lote. Recibidas: {len(request.predictions)}"
        )
    
    results = []
    successful = 0
    failed = 0
    batch_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    for i, data in enumerate(request.predictions):
        try:
            # Procesar individualmente
            processed_data = preprocess_input(data)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            risk_category = calculate_risk_category(probability)
            confidence_level = calculate_confidence_level(probability)
            interpretation = generate_interpretation_message(bool(prediction), probability, data)
            
            result = PredictionResponse(
                prediccion=bool(prediction),
                probabilidad=float(probability),
                categoria_riesgo=risk_category,
                nivel_confianza=confidence_level,
                interpretacion=interpretation,
                modelo_version=model_info["version"],
                prediction_id=f"{batch_id}_{i}",
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
            successful += 1
            
        except Exception as e:
            logger.error(f"Error en predicción {i}: {e}")
            
            # Crear resultado de error
            error_result = PredictionResponse(
                prediccion=False,
                probabilidad=0.0,
                categoria_riesgo="Error",
                nivel_confianza="Error",
                interpretacion=f"Error en procesamiento: {str(e)}",
                modelo_version=model_info["version"],
                prediction_id=f"{batch_id}_{i}_error",
                timestamp=datetime.now().isoformat()
            )
            results.append(error_result)
            failed += 1
    
    # Calcular tiempo de procesamiento
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Actualizar contador
    model_info["prediction_count"] += successful
    
    logger.info(f"Lote procesado: {successful} exitosas, {failed} fallidas")
    
    return BatchPredictionResponse(
        results=results,
        total_predictions=len(request.predictions),
        successful_predictions=successful,
        failed_predictions=failed,
        batch_id=batch_id,
        processing_time_seconds=processing_time,
        timestamp=datetime.now().isoformat()
    )

@app.get("/test", tags=["General"])
async def test_prediction():
    """Test rápido con datos de ejemplo de Ecuador"""
    if not model_info["loaded"]:
        return {
            "status": "error",
            "message": "Modelo no disponible",
            "modelo_info": model_info
        }
    
    # Datos de ejemplo realistas para Ecuador
    test_data = DivorceInputEcuador(
        prov_insc="Guayas",
        cant_insc="Guayaquil",
        edad_1=32,
        edad_2=29,
        niv_inst1="Superior Universitaria",
        niv_inst2="Educación media / Bachillerato",
        est_civi1="Soltero",
        est_civi2="Soltero",
        sexo_1="Hombre",
        sexo_2="Mujer",
        p_etnica1="Mestizo",
        p_etnica2="Mestizo",
        hijos_rec=1.0,
        anios_matrimonio=5
    )
    
    try:
        result = await predict_divorce(test_data)
        
        return {
            "status": "success",
            "message": "API funcionando correctamente",
            "datos_prueba": test_data.dict(),
            "resultado": result.dict(),
            "modelo_info": {
                "version": model_info["version"],
                "accuracy": f"{model_info['accuracy']:.1%}",
                "stage": model_info["stage"],
                "total_predictions": model_info["prediction_count"]
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error en test: {str(e)}",
            "modelo_info": model_info
        }

@app.post("/reload-model", tags=["Modelo"])
async def reload_model(background_tasks: BackgroundTasks):
    """Recargar modelo desde MLflow"""
    def reload_task():
        logger.info("🔄 Recargando modelo...")
        success = load_model_from_mlflow()
        if success:
            logger.info("✅ Modelo recargado exitosamente")
        else:
            logger.error("❌ Error recargando modelo")
    
    background_tasks.add_task(reload_task)
    
    return {
        "message": "Recarga de modelo iniciada en background",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", tags=["Monitoreo"])
async def get_prediction_stats():
    """Estadísticas de uso de la API"""
    return {
        "api_version": "2.0.0",
        "modelo": {
            "nombre": model_info["model_name"],
            "version": model_info["version"],
            "stage": model_info["stage"],
            "accuracy": model_info["accuracy"],
            "f1_score": model_info["f1_score"],
            "roc_auc": model_info["roc_auc"]
        },
        "estadisticas": {
            "total_predicciones": model_info["prediction_count"],
            "modelo_cargado": model_info["loaded"],
            "ultima_actualizacion": model_info["last_update"]
        },
        "sistema": {
            "mlflow_uri": mlflow_uri,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.get("/features/info", tags=["Modelo"])
async def get_features_info():
    """Información sobre las características del modelo"""
    if not model_info["loaded"]:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Información sobre las características esperadas
    feature_info = {
        "total_features": model_info["total_features"],
        "feature_categories": {
            "demograficas": [
                "edad_1", "edad_2", "sexo_1", "sexo_2", 
                "p_etnica1", "p_etnica2"
            ],
            "geograficas": [
                "prov_insc", "cant_insc", "parr_insc",
                "prov_hab1", "cant_hab1", "parr_hab1",
                "prov_hab2", "cant_hab2", "parr_hab2"
            ],
            "educativas": [
                "niv_inst1", "niv_inst2", "sabe_leer1", "sabe_leer2"
            ],
            "matrimoniales": [
                "est_civi1", "est_civi2", "mcap_bie",
                "anio_insc", "mes_insc", "dia_insc"
            ],
            "familiares": [
                "hijos_rec", "nmatant1", "nmatant2"
            ]
        },
        "categorical_encoders": list(label_encoders.keys()) if label_encoders else [],
        "model_features": feature_names[:20] if feature_names else []  # Primeras 20
    }
    
    return feature_info

@app.get("/provinces", tags=["General"])
async def get_provinces():
    """Lista de provincias de Ecuador para el formulario"""
    provinces = [
        "Azuay", "Bolívar", "Cañar", "Carchi", "Chimborazo", "Cotopaxi",
        "El Oro", "Esmeraldas", "Galápagos", "Guayas", "Imbabura", "Loja",
        "Los Ríos", "Manabí", "Morona Santiago", "Napo", "Orellana",
        "Pastaza", "Pichincha", "Santa Elena", "Santo Domingo de los Tsáchilas",
        "Sucumbíos", "Tungurahua", "Zamora Chinchipe"
    ]
    
    return {
        "provinces": provinces,
        "total": len(provinces),
        "note": "Provincias de Ecuador según división política administrativa"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Manejador global de excepciones"""
    logger.error(f"Error no manejado: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "detail": "Ha ocurrido un error inesperado",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Punto de entrada principal
if __name__ == "__main__":
    import uvicorn
    
    # Configuración del servidor
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    workers = int(os.getenv("API_WORKERS", 1))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"🚀 Iniciando servidor en {host}:{port}")
    logger.info(f"📊 MLflow URI: {mlflow_uri}")
    logger.info(f"🔧 Workers: {workers}, Reload: {reload}")
    
    try:
        uvicorn.run(
            "main:app",  # Módulo:aplicación
            host=host,
            port=port,
            workers=workers if not reload else 1,  # Solo 1 worker en modo reload
            reload=reload,
            log_level="info",
            access_log=True,
            server_header=False,
            date_header=False
        )
    except KeyboardInterrupt:
        logger.info("🛑 Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error iniciando servidor: {e}")
        raise