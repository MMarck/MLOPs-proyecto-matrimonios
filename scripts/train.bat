@echo off
echo 🤖 ENTRENAMIENTO DEL MODELO DE PREDICCIÓN DE DIVORCIOS
echo =====================================================

REM Verificar que los servicios estén corriendo
echo 🔍 Verificando servicios necesarios...

curl -s http://localhost:5000/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ MLflow no está disponible
    echo 💡 Ejecuta: docker-compose up -d mlflow
    pause
    exit /b 1
)

curl -s http://localhost:9000/minio/health/live >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ MinIO no está disponible
    echo 💡 Ejecuta: docker-compose up -d minio
    pause
    exit /b 1
)

echo ✅ Servicios verificados

REM Verificar dataset
if not exist "data\dataset_combinado_edit.csv" (
    echo ❌ Dataset no encontrado: data\dataset_combinado_edit.csv
    pause
    exit /b 1
)

echo ✅ Dataset encontrado

REM Configurar variables de entorno
set MLFLOW_TRACKING_URI=http://localhost:5000
set AWS_ACCESS_KEY_ID=minioadmin
set AWS_SECRET_ACCESS_KEY=minioadmin123
set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

echo 🚀 Iniciando entrenamiento...
echo    MLflow: %MLFLOW_TRACKING_URI%
echo    S3 Endpoint: %MLFLOW_S3_ENDPOINT_URL%
echo.

REM Ejecutar entrenamiento
python src\train_model.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo 🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!
    echo ==========================================
    echo.
    echo 📊 Verifica los resultados en:
    echo    • MLflow UI: http://localhost:5000
    echo    • MinIO Console: http://localhost:9001
    echo.
    echo 🔄 Reiniciando API para cargar nuevo modelo...
    curl -X POST http://localhost:8000/reload-model >nul 2>&1
    
    timeout /t 5 /nobreak >nul
    
    echo ✅ API notificada
    echo.
    echo 🧪 Prueba el modelo:
    echo    curl http://localhost:8000/test
    echo.
) else (
    echo.
    echo ❌ ERROR EN EL ENTRENAMIENTO
    echo ============================
    echo 💡 Revisa los logs anteriores para identificar el problema
    echo.
)

pause