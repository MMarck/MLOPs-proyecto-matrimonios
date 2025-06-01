import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, confusion_matrix
import pickle
import os
import json
import tempfile
import warnings
from datetime import datetime
import boto3
warnings.filterwarnings('ignore')

def setup_mlflow():
    """Configurar MLflow y S3"""
    print("🔧 CONFIGURANDO MLFLOW Y S3")
    print("-" * 50)
    
    # Configurar variables de entorno para S3
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minio')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment_name = os.getenv('EXPERIMENT_NAME', 'divorce_prediction_ecuador')
    
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass  # Experimento ya existe
    
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ MLflow configurado")
    print(f"   URI: {mlflow_uri}")
    print(f"   Experimento: {experiment_name}")
    print(f"   S3 Endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
    
    # Verificar conexión con S3
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )
        s3_client.list_buckets()
        print("✅ Conexión S3 verificada")
    except Exception as e:
        print(f"⚠️ Warning: Problemas con S3 - {e}")

def load_and_explore_data():
    """Cargar y explorar el dataset de divorcios de Ecuador"""
    print("\n📊 CARGANDO DATASET DE DIVORCIOS DE ECUADOR")
    print("-" * 50)
    
    # Buscar el dataset en diferentes ubicaciones
    data_paths = [
    "./dataset/dataset_combinado.csv"
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            print(f"📁 Cargando desde: {path}")
            df = pd.read_csv(path, low_memory=False)
            print(f"✅ Dataset cargado exitosamente")
            break
    
    if df is None:
        raise FileNotFoundError(
            "No se encontró el dataset. Colócalo en una de estas ubicaciones:\n" +
            "\n".join(f"  - {path}" for path in data_paths)
        )
    
    print(f"📈 Dimensiones del dataset: {df.shape}")
    print(f"📋 Columnas disponibles: {len(df.columns)}")
    
    # Verificar columna target
    if 'es_divorcio' not in df.columns:
        raise ValueError("Columna 'es_divorcio' no encontrada en el dataset")
    
    # Convertir target a numérico si es necesario
    if df['es_divorcio'].dtype == 'object' or df['es_divorcio'].dtype == 'bool':
        df['es_divorcio'] = df['es_divorcio'].astype(str).map({
            'True': 1, 'False': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
        })
    
    # Estadísticas del target
    target_dist = df['es_divorcio'].value_counts()
    print(f"\n🎯 DISTRIBUCIÓN DEL TARGET:")
    print(f"   No divorcio (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   Divorcio (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"   Total registros: {len(df):,}")
    
    # Información sobre valores faltantes
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0].sort_values(ascending=False)
    print(f"\n❓ VALORES FALTANTES:")
    print(f"   Total valores faltantes: {missing_info.sum():,}")
    print(f"   Columnas con faltantes: {len(missing_cols)}")
    
    if len(missing_cols) > 0:
        print("   Top 10 columnas con más faltantes:")
        for col, missing in missing_cols.head(10).items():
            pct = (missing / len(df)) * 100
            print(f"     {col}: {missing:,} ({pct:.1f}%)")
    
    # Verificar balanceo de clases
    if len(target_dist) < 2:
        print("⚠️  ADVERTENCIA: Solo se encontró una clase en el target")
        raise ValueError("Dataset desbalanceado - necesita al menos 2 clases")
    
    minority_class_pct = min(target_dist.values) / len(df) * 100
    if minority_class_pct < 1:
        print(f"⚠️  ADVERTENCIA: Clase minoritaria muy pequeña ({minority_class_pct:.1f}%)")
    
    return df

def preprocess_data(df):
    """Preprocesar el dataset de divorcios"""
    print("\n🔧 PREPROCESANDO DATASET")
    print("-" * 50)
    
    # Separar características y target
    X = df.drop("es_divorcio", axis=1)
    y = df["es_divorcio"]
    
    print(f"📊 Características originales: {X.shape[1]}")
    print(f"📊 Registros: {X.shape[0]:,}")
    
    # Identificar columnas por tipo
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"🔢 Columnas numéricas: {len(numeric_cols)}")
    print(f"📝 Columnas categóricas: {len(categorical_cols)}")
    
    # Identificar columnas específicas de divorcio (que pueden tener muchos NaN)
    divorce_related_cols = [
        'anio_div', 'mes_div', 'dia_div', 'fecha_div', 
        'anio_mat', 'mes_mat', 'dia_mat', 'fecha_mat', 
        'dur_mat', 'cau_div', 'hijos_1', 'hijos_2'
    ]
    
    # Manejar columnas relacionadas con divorcio
    for col in divorce_related_cols:
        if col in X.columns:
            if col in numeric_cols:
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna('No aplica')
    
    # Llenar valores faltantes en columnas numéricas
    for col in numeric_cols:
        if col not in divorce_related_cols:  # Ya manejadas arriba
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    # Llenar valores faltantes en columnas categóricas
    for col in categorical_cols:
        if col not in divorce_related_cols:  # Ya manejadas arriba
            mode_val = X[col].mode()
            if len(mode_val) > 0:
                X[col] = X[col].fillna(mode_val[0])
            else:
                X[col] = X[col].fillna('Desconocido')
    
    # Aplicar Label Encoding a columnas categóricas
    label_encoders = {}
    print(f"\n🏷️ Aplicando Label Encoding...")
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Convertir a string para manejar valores mixtos
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
        # Log información del encoder
        n_categories = len(le.classes_)
        print(f"   {col}: {n_categories} categorías únicas")
    
    print(f"✅ Label encoders creados: {len(label_encoders)}")
    
    # Verificar que no hay valores faltantes
    remaining_missing = X.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"⚠️ Valores faltantes restantes: {remaining_missing}")
        # Llenar cualquier NaN restante con 0
        X = X.fillna(0)
    else:
        print(f"✅ Sin valores faltantes restantes")
    
    print(f"✅ Preprocesamiento completado: {X.shape}")
    
    return X, y, label_encoders, numeric_cols, categorical_cols

def train_model_with_mlflow(X, y, label_encoders, numeric_cols, categorical_cols, df):
    """Entrenar modelo con tracking completo en MLflow"""
    print("\n🤖 ENTRENANDO MODELO CON MLFLOW")
    print("-" * 50)
    
    with mlflow.start_run(run_name=f"divorce_model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log información del dataset
        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("total_features", X.shape[1])
        mlflow.log_param("numeric_features", len(numeric_cols))
        mlflow.log_param("categorical_features", len(categorical_cols))
        
        # División estratificada de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 División de datos:")
        print(f"   Entrenamiento: {X_train.shape} ({len(y_train):,} registros)")
        print(f"   Prueba: {X_test.shape} ({len(y_test):,} registros)")
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("test_ratio", 0.2)
        
        # Escalado de características
        print(f"⚖️ Aplicando StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Parámetros del modelo optimizados para dataset grande
        model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1  # Usar todos los cores disponibles
        }
        
        # Log parámetros del modelo
        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Entrenar modelo
        print(f"🚀 Entrenando RandomForestClassifier...")
        model = RandomForestClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        print(f"✅ Modelo entrenado: {type(model).__name__}")
        
        # Predicciones
        print(f"🔮 Realizando predicciones...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Verificar predicciones
        unique_predictions = np.unique(y_pred)
        unique_actual = np.unique(y_test)
        
        print(f"📊 Clases en test: {unique_actual}")
        print(f"📊 Clases predichas: {unique_predictions}")
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        print(f"🔄 Realizando cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"\n📈 MÉTRICAS DEL MODELO:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   CV F1 Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Log métricas en MLflow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_metric("cv_f1_mean", cv_mean)
        mlflow.log_metric("cv_f1_std", cv_std)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # Reporte de clasificación detallado
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log métricas por clase
        if '0' in class_report:
            mlflow.log_metric("precision_no_divorce", class_report['0']['precision'])
            mlflow.log_metric("recall_no_divorce", class_report['0']['recall'])
            mlflow.log_metric("f1_no_divorce", class_report['0']['f1-score'])
        
        if '1' in class_report:
            mlflow.log_metric("precision_divorce", class_report['1']['precision'])
            mlflow.log_metric("recall_divorce", class_report['1']['recall'])
            mlflow.log_metric("f1_divorce", class_report['1']['f1-score'])
        
        # Log métricas macro y weighted
        if 'macro avg' in class_report:
            mlflow.log_metric("macro_precision", class_report['macro avg']['precision'])
            mlflow.log_metric("macro_recall", class_report['macro avg']['recall'])
            mlflow.log_metric("macro_f1", class_report['macro avg']['f1-score'])
        
        if 'weighted avg' in class_report:
            mlflow.log_metric("weighted_precision", class_report['weighted avg']['precision'])
            mlflow.log_metric("weighted_recall", class_report['weighted avg']['recall'])
            mlflow.log_metric("weighted_f1", class_report['weighted avg']['f1-score'])
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log top características más importantes
        top_features = feature_importance.head(20)
        for idx, row in top_features.iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
        
        print(f"\n🔝 TOP 15 CARACTERÍSTICAS MÁS IMPORTANTES:")
        for idx, row in feature_importance.head(15).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Guardar artefactos en S3 via MLflow
        print(f"\n💾 GUARDANDO ARTEFACTOS EN S3...")
        
        # 1. Modelo principal
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="DivorcePredictor"
        )
        print("   ✅ Modelo guardado")
        
        # 2. Scaler
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(scaler, f)
            mlflow.log_artifact(f.name, "scaler.pkl")
        print("   ✅ Scaler guardado")
        
        # 3. Label encoders
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(label_encoders, f)
            mlflow.log_artifact(f.name, "label_encoders.pkl")
        print("   ✅ Label encoders guardados")
        
        # 4. Feature importance
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        print("   ✅ Feature importance guardado")
        
        # 5. Reporte de clasificación
        with open("classification_report.txt", "w", encoding='utf-8') as f:
            f.write("REPORTE DE CLASIFICACIÓN - PREDICCIÓN DE DIVORCIOS ECUADOR\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {df.shape[0]:,} registros, {df.shape[1]} características\n")
            f.write(f"Fecha de entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("MÉTRICAS PRINCIPALES:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write(f"CV F1 Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})\n\n")
            f.write("REPORTE DETALLADO:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\n\nTOP 20 CARACTERÍSTICAS MÁS IMPORTANTES:\n")
            for idx, row in feature_importance.head(20).iterrows():
                f.write(f"{row['feature']}: {row['importance']:.4f}\n")
        
        mlflow.log_artifact("classification_report.txt")
        print("   ✅ Reporte de clasificación guardado")
        
        # 6. Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        cm_info = {
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1])
        }
        
        with open("confusion_matrix.json", "w") as f:
            json.dump(cm_info, f, indent=2)
        mlflow.log_artifact("confusion_matrix.json")
        print("   ✅ Matriz de confusión guardada")
        
        # 7. Información del dataset
        target_dist = y.value_counts()
        dataset_info = {
            'total_records': len(df),
            'features': X.shape[1],
            'target_distribution': {
                'no_divorce': int(target_dist.get(0, 0)),
                'divorce': int(target_dist.get(1, 0))
            },
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'training_date': datetime.now().isoformat(),
            'model_params': model_params,
            'feature_names': list(X.columns)[:50]  # Primeras 50 para evitar archivos muy grandes
        }
        
        with open("dataset_info.json", "w", encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact("dataset_info.json")
        print("   ✅ Información del dataset guardada")
        
        # 8. Distribución de probabilidades
        prob_stats = {
            "min_prob": float(y_pred_proba.min()),
            "max_prob": float(y_pred_proba.max()),
            "mean_prob": float(y_pred_proba.mean()),
            "median_prob": float(np.median(y_pred_proba)),
            "std_prob": float(y_pred_proba.std())
        }
        
        with open("probability_stats.json", "w") as f:
            json.dump(prob_stats, f, indent=2)
        mlflow.log_artifact("probability_stats.json")
        print("   ✅ Estadísticas de probabilidades guardadas")
        
        # Obtener información del run actual
        current_run = mlflow.active_run()
        run_id = current_run.info.run_id
        artifact_uri = current_run.info.artifact_uri
        
        print(f"\n✅ Modelo y artefactos guardados en MLflow")
        print(f"🔗 Run ID: {run_id}")
        print(f"📊 Artifact URI: {artifact_uri}")
        
        return model, scaler, label_encoders, feature_importance, run_id

def promote_model_to_production(run_id):
    """Promover modelo a Production"""
    print(f"\n🚀 PROMOVIENDO MODELO A PRODUCTION")
    print("-" * 50)
    
    try:
        client = mlflow.MlflowClient()
        model_name = "DivorcePredictor"
        
        # Buscar la versión del modelo asociada con este run
        model_versions = client.search_model_versions(f"name='{model_name}'")
        target_version = None
        
        for version in model_versions:
            if version.run_id == run_id:
                target_version = version
                break
        
        if target_version:
            # Promover a Production
            client.transition_model_version_stage(
                name=model_name,
                version=target_version.version,
                stage="Production"
            )
            
            print(f"✅ Modelo v{target_version.version} promovido a Production")
            print(f"   Run ID: {run_id}")
            print(f"   Nombre del modelo: {model_name}")
            return target_version.version
        else:
            print(f"❌ No se encontró versión del modelo para el run {run_id}")
            return None
            
    except Exception as e:
        print(f"⚠️  Error promoviendo modelo: {e}")
        print("💡 Puedes promoverlo manualmente desde la UI de MLflow")
        return None

def validate_model_deployment():
    """Validar que el modelo se desplegó correctamente"""
    print(f"\n🧪 VALIDANDO DESPLIEGUE DEL MODELO")
    print("-" * 50)
    
    try:
        client = mlflow.MlflowClient()
        
        # Verificar modelos registrados
        registered_models = client.search_registered_models()
        divorce_models = [m for m in registered_models if m.name == "DivorcePredictor"]
        
        if not divorce_models:
            print("❌ Modelo 'DivorcePredictor' no encontrado en el registro")
            return False
        
        print(f"✅ Modelo 'DivorcePredictor' encontrado")
        
        # Verificar versiones en Production
        model = divorce_models[0]
        production_versions = [v for v in model.latest_versions if v.current_stage == "Production"]
        
        if production_versions:
            version = production_versions[0]
            print(f"✅ Versión en Production: v{version.version}")
            print(f"   Run ID: {version.run_id}")
            
            # Intentar cargar el modelo
            try:
                model_uri = f"models:/DivorcePredictor/Production"
                loaded_model = mlflow.sklearn.load_model(model_uri)
                print(f"✅ Modelo cargado exitosamente desde Production")
                return True
            except Exception as e:
                print(f"❌ Error cargando modelo desde Production: {e}")
                return False
        else:
            print("⚠️  No hay versiones en Production")
            return False
            
    except Exception as e:
        print(f"❌ Error validando despliegue: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ENTRENAMIENTO DE MODELO DE PREDICCIÓN DE DIVORCIOS - ECUADOR")
    print("=" * 80)
    print(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = datetime.now()
    
    try:
        # 1. Configurar MLflow y S3
        setup_mlflow()
        
        # 2. Cargar y explorar datos
        df = load_and_explore_data()
        
        # 3. Preprocesar datos
        X, y, label_encoders, numeric_cols, categorical_cols = preprocess_data(df)
        
        # 4. Entrenar modelo con MLflow
        model, scaler, label_encoders, feature_importance, run_id = train_model_with_mlflow(
            X, y, label_encoders, numeric_cols, categorical_cols, df
        )
        
        # 5. Promover modelo a Production
        model_version = promote_model_to_production(run_id)
        
        # 6. Validar despliegue
        deployment_success = validate_model_deployment()
        
        # Calcular tiempo total
        end_time = datetime.now()
        total_time = end_time - start_time
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("=" * 80)
        print(f"⏱️  Tiempo total: {total_time}")
        print(f"📊 Registros procesados: {len(df):,}")
        print(f"🔗 Run ID: {run_id}")
        
        if model_version:
            print(f"🚀 Modelo v{model_version} en Production")
        
        print(f"\n🌐 RECURSOS DISPONIBLES:")
        print(f"   • MLflow UI: http://localhost:5000")
        print(f"   • MinIO Console: http://localhost:9001")
        print(f"   • API Docs: http://localhost:8000/docs")
        print(f"   • Airflow: http://localhost:8080")
        
        print(f"\n📝 PRÓXIMOS PASOS:")
        print(f"   1. Verificar experimento en MLflow: http://localhost:5000")
        print(f"   2. Revisar artefactos en MinIO: http://localhost:9001")
        print(f"   3. Reiniciar API: docker-compose restart divorce-api")
        print(f"   4. Probar predicciones: curl http://localhost:8000/test")
        
        if deployment_success:
            print(f"\n✅ El modelo está listo para servir predicciones!")
        else:
            print(f"\n⚠️  Revisa el despliegue del modelo en MLflow")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR DURANTE EL ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)