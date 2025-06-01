from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class DivorceInputEcuador(BaseModel):
    """Modelo de entrada para predicción de divorcios en Ecuador"""
    
    # Información geográfica de inscripción
    prov_insc: str = Field(..., description="Provincia de inscripción")
    cant_insc: str = Field(..., description="Cantón de inscripción")
    parr_insc: Optional[str] = Field(None, description="Parroquia de inscripción")
    
    # Información demográfica persona 1
    edad_1: int = Field(..., ge=18, le=100, description="Edad de la persona 1")
    sexo_1: str = Field(..., description="Sexo de la persona 1")
    niv_inst1: str = Field(..., description="Nivel de instrucción persona 1")
    est_civi1: str = Field(..., description="Estado civil anterior persona 1")
    p_etnica1: str = Field(..., description="Pertenencia étnica persona 1")
    sabe_leer1: Optional[str] = Field(None, description="Sabe leer persona 1")
    
    # Información demográfica persona 2
    edad_2: int = Field(..., ge=18, le=100, description="Edad de la persona 2")
    sexo_2: str = Field(..., description="Sexo de la persona 2")
    niv_inst2: str = Field(..., description="Nivel de instrucción persona 2")
    est_civi2: str = Field(..., description="Estado civil anterior persona 2")
    p_etnica2: str = Field(..., description="Pertenencia étnica persona 2")
    sabe_leer2: Optional[str] = Field(None, description="Sabe leer persona 2")
    
    # Información de residencia persona 1
    prov_hab1: Optional[str] = Field(None, description="Provincia de residencia persona 1")
    cant_hab1: Optional[str] = Field(None, description="Cantón de residencia persona 1")
    parr_hab1: Optional[str] = Field(None, description="Parroquia de residencia persona 1")
    
    # Información de residencia persona 2
    prov_hab2: Optional[str] = Field(None, description="Provincia de residencia persona 2")
    cant_hab2: Optional[str] = Field(None, description="Cantón de residencia persona 2")
    parr_hab2: Optional[str] = Field(None, description="Parroquia de residencia persona 2")
    
    # Información matrimonial
    mcap_bie: Optional[str] = Field(None, description="Régimen de bienes")
    anio_insc: Optional[int] = Field(None, ge=1950, le=2030, description="Año de inscripción del matrimonio")
    mes_insc: Optional[int] = Field(None, ge=1, le=12, description="Mes de inscripción del matrimonio")
    dia_insc: Optional[int] = Field(None, ge=1, le=31, description="Día de inscripción del matrimonio")
    
    # Información familiar
    hijos_rec: Optional[float] = Field(None, ge=0, description="Número de hijos reconocidos")
    nmatant1: Optional[int] = Field(None, ge=0, description="Número de matrimonios anteriores persona 1")
    nmatant2: Optional[int] = Field(None, ge=0, description="Número de matrimonios anteriores persona 2")
    
    # Campo calculado
    anios_matrimonio: Optional[int] = Field(None, ge=0, description="Años de matrimonio (calculado)")
    
    @validator('sexo_1', 'sexo_2')
    def validate_gender(cls, v):
        valid_genders = ['Hombre', 'Mujer']
        if v not in valid_genders:
            raise ValueError(f'Sexo debe ser uno de: {valid_genders}')
        return v
    
    @validator('niv_inst1', 'niv_inst2')
    def validate_education(cls, v):
        valid_levels = [
            'Sin estudios',
            'Educación básica',
            'Educación media / Bachillerato',
            'Superior no universitaria',
            'Superior Universitaria',
            'Postgrado'
        ]
        if v not in valid_levels:
            raise ValueError(f'Nivel de instrucción debe ser uno de: {valid_levels}')
        return v
    
    @validator('est_civi1', 'est_civi2')
    def validate_civil_status(cls, v):
        valid_status = ['Soltero', 'Viudo', 'Divorciado', 'Separado']
        if v not in valid_status:
            raise ValueError(f'Estado civil debe ser uno de: {valid_status}')
        return v
    
    @validator('p_etnica1', 'p_etnica2')
    def validate_ethnicity(cls, v):
        valid_ethnicities = [
            'Mestizo', 'Blanco', 'Indígena', 'Afroecuatoriano',
            'Montubio', 'Mulato', 'Negro', 'Otro'
        ]
        if v not in valid_ethnicities:
            raise ValueError(f'Pertenencia étnica debe ser una de: {valid_ethnicities}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "prov_insc": "Pichincha",
                "cant_insc": "Quito",
                "parr_insc": "La Mariscal",
                "edad_1": 28,
                "sexo_1": "Hombre",
                "niv_inst1": "Superior Universitaria",
                "est_civi1": "Soltero",
                "p_etnica1": "Mestizo",
                "sabe_leer1": "Si",
                "edad_2": 26,
                "sexo_2": "Mujer",
                "niv_inst2": "Superior Universitaria",
                "est_civi2": "Soltero",
                "p_etnica2": "Mestizo",
                "sabe_leer2": "Si",
                "prov_hab1": "Pichincha",
                "cant_hab1": "Quito",
                "prov_hab2": "Pichincha",
                "cant_hab2": "Quito",
                "mcap_bie": "Sociedad de bienes",
                "anio_insc": 2020,
                "mes_insc": 6,
                "dia_insc": 15,
                "hijos_rec": 0.0,
                "nmatant1": 0,
                "nmatant2": 0,
                "anios_matrimonio": 3
            }
        }

class PredictionResponse(BaseModel):
    """Respuesta de predicción individual"""
    
    prediccion: bool = Field(..., description="Predicción de divorcio (True = divorcio probable)")
    probabilidad: float = Field(..., ge=0, le=1, description="Probabilidad de divorcio [0-1]")
    categoria_riesgo: str = Field(..., description="Categoría de riesgo")
    nivel_confianza: str = Field(..., description="Nivel de confianza de la predicción")
    interpretacion: str = Field(..., description="Interpretación de la predicción")
    modelo_version: Optional[str] = Field(None, description="Versión del modelo usado")
    prediction_id: str = Field(..., description="ID único de la predicción")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    
    class Config:
        schema_extra = {
            "example": {
                "prediccion": False,
                "probabilidad": 0.23,
                "categoria_riesgo": "Bajo",
                "nivel_confianza": "Alta",
                "interpretacion": "Matrimonio estable (77.0% de estabilidad). Factores protectores: edades similares, nivel educativo similar.",
                "modelo_version": "1",
                "prediction_id": "pred_12345",
                "timestamp": "2024-01-15T10:30:00"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Solicitud de predicción en lote"""
    
    predictions: List[DivorceInputEcuador] = Field(
        ..., 
        max_items=100,
        description="Lista de datos para predicción (máximo 100)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prov_insc": "Guayas",
                        "cant_insc": "Guayaquil",
                        "edad_1": 32,
                        "edad_2": 29,
                        "niv_inst1": "Superior Universitaria",
                        "niv_inst2": "Educación media / Bachillerato",
                        "est_civi1": "Soltero",
                        "est_civi2": "Soltero",
                        "sexo_1": "Hombre",
                        "sexo_2": "Mujer",
                        "p_etnica1": "Mestizo",
                        "p_etnica2": "Mestizo",
                        "hijos_rec": 1.0,
                        "anios_matrimonio": 5
                    }
                ]
            }
        }

class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción en lote"""
    
    results: List[PredictionResponse] = Field(..., description="Lista de resultados")
    total_predictions: int = Field(..., description="Total de predicciones solicitadas")
    successful_predictions: int = Field(..., description="Predicciones exitosas")
    failed_predictions: int = Field(..., description="Predicciones fallidas")
    batch_id: str = Field(..., description="ID del lote")
    processing_time_seconds: float = Field(..., description="Tiempo de procesamiento en segundos")
    timestamp: str = Field(..., description="Timestamp del procesamiento")

class HealthResponse(BaseModel):
    """Respuesta del health check"""
    
    status: str = Field(..., description="Estado general: healthy, degraded, unhealthy")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    mlflow_connected: bool = Field(..., description="Si hay conexión con MLflow")
    model_info: Dict[str, Any] = Field(..., description="Información del modelo")
    timestamp: str = Field(..., description="Timestamp del check")
    version: str = Field(..., description="Versión de la API")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "mlflow_connected": True,
                "model_info": {
                    "loaded": True,
                    "model_name": "DivorcePredictor",
                    "version": "1",
                    "accuracy": 0.85,
                    "prediction_count": 150
                },
                "timestamp": "2024-01-15T10:30:00",
                "version": "2.0.0"
            }
        }

class ModelInfoResponse(BaseModel):
    """Información detallada del modelo"""
    
    model_name: str = Field(..., description="Nombre del modelo")
    model_stage: str = Field(..., description="Etapa del modelo (Production, Staging, etc.)")
    model_version: Optional[str] = Field(None, description="Versión del modelo")
    accuracy: float = Field(..., description="Accuracy del modelo")
    f1_score: float = Field(..., description="F1 Score del modelo")
    roc_auc: float = Field(..., description="ROC AUC del modelo")
    total_features: int = Field(..., description="Número total de características")
    run_id: Optional[str] = Field(None, description="ID del run de MLflow")
    last_updated: Optional[str] = Field(None, description="Última actualización")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Información adicional")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "DivorcePredictor",
                "model_stage": "Production",
                "model_version": "1",
                "accuracy": 0.847,
                "f1_score": 0.832,
                "roc_auc": 0.891,
                "total_features": 25,
                "run_id": "abc123def456",
                "last_updated": "2024-01-15T08:00:00",
                "additional_info": {
                    "training_metrics": {},
                    "training_params": {},
                    "training_date": 1704960000000
                }
            }
        }

class ErrorResponse(BaseModel):
    """Respuesta de error estándar"""
    
    error: str = Field(..., description="Tipo de error")
    detail: str = Field(..., description="Detalle del error")
    timestamp: str = Field(..., description="Timestamp del error")
    path: Optional[str] = Field(None, description="Path del endpoint que causó el error")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "La edad debe estar entre 18 y 100 años",
                "timestamp": "2024-01-15T10:30:00",
                "path": "/predict"
            }
        }