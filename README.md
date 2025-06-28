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

![Diagrama de arquitectura](/final_assign.png)

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
   docker-compose --profile all up
   ```

5. Ingresa a la interfaz de Airflow en `http://localhost:8080` y ejecuta el DAG `etl_divorcios_data`.

6. Ejecuta el DAG `entrenamiento_inicial_modelo`.


7. Accede a la API de predicción en `http://localhost:8800/docs`.

8. Para probar reentrenamientos automáticos, ejecuta el DAG `retrain_model_if_better`.

## Ejemplo de Uso de la API

```bash
curl -X POST 'http://localhost:8800/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "dur_mat": 0,
      "hijos_2": 0,
      "edad_2": 18,
      "edad_1": 18,
      "area_1": "Urbana",
      "area_2": "Urbana",
      "hijos_rec": 2,
      "mcap_bie": "No",
      "p_etnica1": "Mestizo",
      "cant_insc": "Guayaquil",
      "p_etnica2": "Mestizo",
      "cant_hab1": "Quito",
      "cant_hab2": "Guayaquil",
      "sabe_leer2": "Si",
      "sabe_leer1": "Si",
      "mes_nac1": "Junio",
      "niv_inst2": "Superior Universitaria",
      "prov_insc": "Pichincha",
      "mes_nac2": "Agosto"
    }
  }'
```

Respuesta esperada:

```json
{
  "int_output": 0,
  "str_output": "No divorcio"
}
```

## Nota Final

Este proyecto puede ser utilizado como base para otros modelos sociales predictivos en Ecuador o Latinoamérica. Se aceptan Pull Requests con mejoras o contribuciones.

---
