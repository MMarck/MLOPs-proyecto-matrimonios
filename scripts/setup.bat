@echo off
echo 🇪🇨 CONFIGURACIÓN COMPLETA - PREDICCIÓN DE DIVORCIOS ECUADOR
echo ============================================================

REM Verificar Docker y Docker Compose
echo 🐳 Verificando requisitos del sistema...
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker no está instalado
    echo 💡 Descarga Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker Compose no disponible
    pause
    exit /b 1
)

echo ✅ Docker y Docker Compose disponibles

REM Verificar estructura del proyecto
echo.
echo 📁 Verificando estructura del proyecto...

set "missing_files="

if not exist "docker-compose.yml" (
    echo ❌ docker-compose.yml faltante
    set "missing_files=true"
)

if not exist "Dockerfile" (
    echo ❌ Dockerfile faltante
    set "missing_files=true"
)

if not exist "requirements.txt" (
    echo ❌ requirements.txt faltante
    set "missing_files=true"
)

if not exist "src" (
    echo ❌ Directorio src/ faltante
    set "missing_files=true"
)

if not exist "data" (
    echo ⚠️ Directorio data/ no existe, creándolo...
    mkdir data
)

if not exist "data\dataset_combinado_edit.csv" (
    echo ❌ Dataset principal faltante: data\dataset_combinado_edit.csv
    echo 💡 Coloca tu dataset en data\dataset_combinado_edit.csv
    set "missing_files=true"
)

if defined missing_files (
    echo.
    echo ❌ ARCHIVOS FALTANTES DETECTADOS
    echo --------------------------------
    echo 💡 Asegúrate de tener todos los archivos necesarios
    echo.
    echo 📋 Estructura requerida:
    echo    divorce-prediction-ecuador/
    echo    ├── docker-compose.yml
    echo    ├── Dockerfile
    echo    ├── requirements.txt
    echo    ├── src/
    echo    │   ├── api.py
    echo    │   ├── models.py
    echo    │   ├── train_model.py
    echo    │   └── utils.py
    echo    ├── data/
    echo    │   └── dataset_combinado_edit.csv
    echo    ├── dags/
    echo    └── scripts/
    echo.
    pause
    exit /b 1
)

echo ✅ Estructura del proyecto verificada

REM Crear directorios adicionales
echo.
echo 📂 Creando directorios adicionales...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "notebooks" mkdir notebooks
if not exist "tests" mkdir tests
if not exist "dags" mkdir dags

echo ✅ Directorios creados

REM Verificar dataset
echo.
echo 📊 Verificando dataset...
python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('data/dataset_combinado_edit.csv')
    print(f'✅ Dataset cargado: {len(df):,} registros, {len(df.columns)} columnas')
    
    if 'es_divorcio' not in df.columns:
        print('❌ Columna target \"es_divorcio\" no encontrada')
        sys.exit(1)
    
    target_dist = df['es_divorcio'].value_counts()
    print(f'🎯 Distribución target:')
    for val, count in target_dist.items():
        pct = count/len(df)*100
        print(f'   {val}: {count:,} ({pct:.1f}%%)')
        
except Exception as e:
    print(f'❌ Error verificando dataset: {e}')
    sys.exit(1)
" 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Problemas con el dataset
    echo 💡 Verifica que el archivo data\dataset_combinado_edit.csv sea válido
    pause
    exit /b 1
)

REM Detener servicios anteriores
echo.
echo 🛑 Deteniendo servicios anteriores...
docker-compose down >nul 2>&1

REM Limpiar volúmenes si es necesario
echo.
echo 🧹 ¿Limpiar datos anteriores? (esto borrará experimentos y modelos previos)
set /p clean_volumes="Escribe 'si' para limpiar o Enter para continuar: "
if /i "%clean_volumes%"=="si" (
    echo Limpiando volúmenes...
    docker-compose down -v
    docker volume prune -f
    echo ✅ Volúmenes limpiados
)

REM Construir imágenes
echo.
echo 🏗️ Construyendo imágenes Docker...
docker-compose build --no-cache

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error construyendo imágenes
    echo 💡 Revisa los logs anteriores
    pause
    exit /b 1
)

echo ✅ Imágenes construidas exitosamente

REM Iniciar servicios en orden
echo.
echo 🚀 INICIANDO SERVICIOS PASO A PASO
echo =================================

echo 🐘 Iniciando PostgreSQL...
docker-compose up -d postgres
timeout /t 20 /nobreak >nul

echo 🔍 Verificando PostgreSQL...
for /l %%i in (1,1,30) do (
    docker-compose exec -T postgres pg_isready -U postgres >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo ✅ PostgreSQL listo
        goto :postgres_ready
    )
    timeout /t 2 /nobreak >nul
)
echo ❌ PostgreSQL no se inicializó
goto :error

:postgres_ready

echo.
echo 🗄️ Iniciando MinIO...
docker-compose up -d minio
timeout /t 15 /nobreak >nul

echo 🔍 Verificando MinIO...
for /l %%i in (1,1,30) do (
    curl -s http://localhost:9000/minio/health/live >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo ✅ MinIO listo
        goto :minio_ready
    )
    timeout /t 2 /nobreak >nul
)
echo ❌ MinIO no se inicializó
goto :error

:minio_ready

echo.
echo 🪣 Configurando buckets MinIO...
docker-compose up -d minio_setup
timeout /t 10 /nobreak >nul

echo.
echo 📊 Iniciando MLflow...
docker-compose up -d mlflow
timeout /t 45 /nobreak >nul

echo 🔍 Verificando MLflow...
for /l %%i in (1,1,60) do (
    curl -s http://localhost:5000/health >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo ✅ MLflow listo
        goto :mlflow_ready
    )
    timeout /t 2 /nobreak >nul
)
echo ❌ MLflow no se inicializó en 2 minutos
echo 📋 Revisando logs...
docker-compose logs --tail=10 mlflow
goto :error

:mlflow_ready

echo.
echo ✈️ Iniciando Airflow...
docker-compose up -d airflow-init
timeout /t 30 /nobreak >nul

docker-compose up -d airflow-scheduler airflow-webserver
timeout /t 30 /nobreak >nul

echo.
echo 🚀 Iniciando API de Predicción...
docker-compose up -d divorce-api
timeout /t 20 /nobreak >nul

echo 🔍 Verificando API...
for /l %%i in (1,1,30) do (
    curl -s http://localhost:8000/health >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo ✅ API lista
        goto :api_ready
    )
    timeout /t 2 /nobreak >nul
)
echo ⚠️ API no responde aún, puede necesitar más tiempo

:api_ready

echo.
echo 📊 ESTADO FINAL DE SERVICIOS
echo ============================
docker-compose ps

echo.
echo 🧪 VERIFICACIÓN DE CONECTIVIDAD
echo ==============================

REM Verificar cada servicio
curl -s http://localhost:5432 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ PostgreSQL: Puerto 5432 abierto
) else (
    echo ⚠️ PostgreSQL: Puerto 5432 no responde
)

curl -s http://localhost:9000/minio/health/live >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ MinIO: Servicio funcionando
) else (
    echo ❌ MinIO: No responde
)

curl -s http://localhost:5000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ MLflow: Servicio funcionando
) else (
    echo ❌ MLflow: No responde
)

curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Airflow: Servicio funcionando
) else (
    echo ⚠️ Airflow: Puede estar iniciando
)

curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ API: Servicio funcionando
) else (
    echo ❌ API: No responde
)

echo.
echo 🎉 ¡CONFIGURACIÓN COMPLETADA!
echo ============================
echo.
echo 🌐 SERVICIOS DISPONIBLES:
echo    • API de Predicción:  http://localhost:8000
echo    • Documentación API:  http://localhost:8000/docs
echo    • MLflow UI:          http://localhost:5000
echo    • Airflow UI:         http://localhost:8080 (admin/admin)
echo    • MinIO Console:      http://localhost:9001 (minioadmin/minioadmin123)
echo.
echo 📊 INFORMACIÓN DEL DATASET:
echo    • Fuente: Registro Civil del Ecuador
echo    • Ubicación: data\dataset_combinado_edit.csv
echo    • Variables: ~56 características demográficas
echo.
echo 🚀 PRÓXIMOS PASOS:
echo ==================
echo.
echo 1. 🤖 ENTRENAR PRIMER MODELO:
echo    python src\train_model.py
echo.
echo 2. 🧪 PROBAR LA API:
echo    curl http://localhost:8000/test
echo    
echo 3. 📚 EXPLORAR DOCUMENTACIÓN:
echo    Abre: http://localhost:8000/docs
echo.
echo 4. 📊 VER EXPERIMENTOS:
echo    Abre: http://localhost:5000
echo.
echo 5. ✈️ CONFIGURAR AIRFLOW:
echo    Abre: http://localhost:8080
echo    Usuario: admin / Contraseña: admin
echo.
echo 💡 COMANDOS ÚTILES:
echo    docker-compose logs [servicio]     # Ver logs
echo    docker-compose restart [servicio]  # Reiniciar
echo    docker-compose down               # Detener todo
echo.
echo ⚠️ IMPORTANTE:
echo   La API estará en "modo sin modelo" hasta que ejecutes el entrenamiento.
echo   Esto es normal y esperado.
echo.

goto :end

:error
echo.
echo ❌ ERROR EN LA CONFIGURACIÓN
echo ============================
echo.
echo 🔍 Para diagnosticar:
echo    docker-compose logs [servicio]
echo.
echo 🔧 Para reintentar:
echo    docker-compose down
echo    scripts\setup.bat
echo.

:end
echo Presiona cualquier tecla para continuar...
pause >nul