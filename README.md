# API de Evaluación de Modelos de Detección de Objetos

## Descripción

Este proyecto implementa una API para la evaluación de modelos de detección de objetos utilizando modelos preentrenados de visión por computadora.

Los modelos evaluados son:

* YOLO
* Faster R-CNN 
* SSD

El objetivo principal es analizar el comportamiento de estos modelos en un entorno controlado, evaluando sus salidas y su rendimiento para determinar cuál es más adecuado para su integración en el proyecto de grado.


## Contexto (Actividad A6)

Este desarrollo corresponde a la actividad A6 del proyecto de grado:

**"Analizar salidas del modelo de detección de objetos"**

Se busca:

* ejecutar modelos con imágenes de prueba
* analizar su salida en formato JSON
* estudiar bounding boxes, clases y niveles de confianza
* comparar modelos en términos de rendimiento y comportamiento



## Objetivo

Evaluar modelos de detección de objetos considerando:

* precisión de detección
* confianza de las predicciones
* tiempo de respuesta
* consumo de recursos del sistema (CPU y memoria)


## Estructura del proyecto

app/
├── routes/
│   ├── detect.py        # Endpoints de detección
│   └── batch.py         # Ejecución de pruebas masivas
│
├── services/
│   ├── yolo_service.py
│   ├── fasterrcnn_service.py
│   └── ssd_service.py
│
├── main.py                # Registro de rutas
├── run.py                 # Punto de entrada

test_images/               # Imágenes de prueba
test_results_json/         # Resultados JSON por prueba
charts/                    # Gráficas generadas
test_results_summary.xlsx  # Resultados en Excel


## Instalación

1. Crear entorno virtual (opcional pero recomendado):
    python -m venv venv
2. Activar entorno:
    Windows:
    venv\Scripts\activate
3. Instalar dependencias:
    pip install -r requirements.txt

## Ejecución

Iniciar la API:
python run.py


La API estará disponible en:
http://127.0.0.1:8000

Documentación automática:
http://127.0.0.1:8000/docs

## Endpoints disponibles

### 1. Detección individual

POST `/api/detect`

Permite ejecutar un modelo específico.

Parámetros:

* model: yolo | fasterrcnn | ssd
* file: imagen
* confidence_threshold: valor entre 0 y 1

### 2. Detección con todos los modelos

POST `/api/detect-all`

Ejecuta los tres modelos sobre la misma imagen.

### 3. Ejecución batch (pruebas completas)

POST `/api/run-batch`

Este endpoint:

* procesa todas las imágenes en la carpeta `test_images`
* ejecuta los tres modelos
* evalúa múltiples thresholds (0.3, 0.5, 0.7)
* genera resultados automáticos

## Métricas evaluadas

### Métricas de detección

* num_detections: número de objetos detectados
* avg_confidence: confianza promedio
* max_confidence: confianza máxima
* min_confidence: confianza mínima
* unique_labels: número de clases distintas
* labels: clases detectadas

### Métricas de rendimiento

* response_time_ms: tiempo de respuesta
* cpu_avg_percent: uso promedio de CPU
* cpu_max_percent: uso máximo de CPU
* mem_avg_mb: memoria promedio utilizada
* mem_peak_mb: memoria máxima utilizada
* mem_before_mb / mem_after_mb: memoria antes y después

## Resultados generados

Después de ejecutar `/api/run-batch` se generan:

* Archivo Excel:
  * `test_results_summary.xlsx`
* Resultados JSON:
  * carpeta `test_results_json`
* Gráficas:
  * carpeta `charts`

## Gráficas generadas

Se generan automáticamente:

* tiempo promedio por modelo
* uso de CPU por modelo
* uso de memoria por modelo
* confianza promedio por modelo

## Metodología

El análisis se realiza bajo condiciones controladas:

* mismo conjunto de imágenes
* mismos niveles de confianza
* evaluación individual por modelo
* medición de recursos por ejecución

## Notas

* Los modelos utilizados son preentrenados (COCO dataset)
* Los resultados dependen de la calidad de las imágenes de prueba
* El análisis incluye tanto métricas cuantitativas como observación manual