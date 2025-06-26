# Ejemplo de Implementación de un Modelo de Predicción de Divorcios en Ecuador
### MLOPS - CEIA - FIUBA

En este proyecto mostramos una implementación completa y automatizada de un modelo predictivo de divorcios en Ecuador utilizando herramientas de MLOps. El sistema sigue buenas prácticas de producción, incluyendo almacenamiento de datos, entrenamiento de modelos, experimentación, despliegue de una API de predicción y reentrenamiento automático.

Los datos utilizados provienen del **Instituto Nacional de Estadística y Censos (INEC)** de **Ecuador**, asegurando así la relevancia local del modelo.

## Autores

Este proyecto fue desarrollado por **Yandri Uchuari** y **Marck Murillo** como parte del curso de MLOps del CEIA - FIUBA.

## Componentes del Proyecto

- En **Apache Airflow**, un DAG que descarga y procesa los datos, realiza limpieza, transforma los features relevantes y guarda conjuntos separados para entrenamiento y prueba en el bucket `s3://divorcios-data`. Todo el proceso se rastrea con **MLflow**.
- Un **servicio API (FastAPI)** que sirve el modelo registrado, exponiendo un endpoint REST para realizar predicciones a partir de nuevos datos.
- Otro DAG en **Airflow** que permite reentrenar el modelo con un nuevo dataset. Si el nuevo modelo supera en precisión al modelo `champion`, es promovido automáticamente a producción.

![Diagrama de arquitectura](docs/diagrama_arquitectura.png)

## Pasos para Levantar el Proyecto

1. Clona el repositorio con la rama correspondiente:
   ```bash
   git clone -b main --single-branch https://github.com/MMarck/MLOPs-proyecto-matrimonios.git
   cd MLOPs-proyecto-matrimonios
   ```

2. Asegúrate de tener `docker` y `docker-compose` instalados.

3. Crea los archivos `.env` necesarios (si no existen) para configurar tus credenciales y rutas.

4. Ejecuta el entorno multi-contenedor:
   ```bash
   docker-compose --profile all up --build
   ```

5. Ingresa a la interfaz de Airflow en `http://localhost:8080` y ejecuta el DAG `etl_divorcios_data`.

6. Ejecuta la notebook en `notebook_example/` o el script de entrenamiento para entrenar el mejor modelo y registrarlo en MLflow (`http://localhost:5000`).

7. Accede a la API de predicción en `http://localhost:8800/docs`.

8. Para probar reentrenamientos automáticos, ejecuta el DAG `retrain_model_if_better`.

## Ejemplo de Uso de la API

```bash
curl -X POST 'http://localhost:8800/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "edad": 32,
      "duracion_matrimonio": 5,
      "numero_hijos": 2,
      "nivel_educativo": "universitario",
      "ocupacion": "empleado_privado",
      "tipo_union": "civil"
    }
  }'
```

Respuesta esperada:

```json
{
  "int_output": 1,
  "str_output": "Alta probabilidad de divorcio"
}
```

## Nota Final

Este proyecto puede ser utilizado como base para otros modelos sociales predictivos en Ecuador o Latinoamérica. Se aceptan Pull Requests con mejoras o contribuciones.

---
