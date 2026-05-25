# API de Generación de Descripciones Narrativas Egocéntricas

Sistema backend que procesa imágenes de entornos Web 3D y genera
descripciones auditivas accesibles para personas con ceguera total.

**Modelo de detección:** YOLO26s (Ultralytics 2026) — seleccionado tras
evaluación comparativa contra Faster R-CNN, Mask R-CNN y SSD (ver rama
`comparativa/multi-modelo`).

## Estructura del proyecto

```
vision-api-project/
├── app/
│   ├── main.py                      # FastAPI app + registro de routers
│   ├── routes/
│   │   ├── detect.py                # /detect, /debug-detect, /health
│   │   └── evaluation.py            # /dataset/*, /metrics/*, /test/*, /finetune/*
│   ├── services/
│   │   ├── yolo_service.py          # Detección YOLO26s
│   │   ├── spatial_analyzer.py      # Cuadrícula 3×3 + categorías + prioridad
│   │   ├── step_estimator.py        # Estimación de pasos (heurística monocular)
│   │   ├── free_space_analyzer.py   # Zonas navegables libres
│   │   ├── risk_engine.py           # Decisión de movimiento
│   │   ├── llm_enhancer.py          # Descripción egocéntrica (Groq/Llama)
│   │   ├── scene_classifier.py      # Clasificación de escenario (Groq/Llama)
│   │   └── tts_service.py           # Síntesis de voz (Google Cloud TTS)
│   └── utils/
│       ├── translator.py            # Traducción EN→ES dinámica con caché
│       └── groq_client.py           # Singleton cliente Groq
├── app/experimental/                # Modelos comparativos (no se cargan en prod)
│   ├── fasterrcnn_service.py
│   ├── maskrcnn_service.py
│   ├── ssd_service.py
│   ├── batch.py
│   └── diagnostico_yolo.py
├── dataset/                         # Generado en producción — excluido de Git
│   ├── images/                      # Imágenes subidas con /api/dataset/upload
│   ├── labels/                      # Etiquetas YOLO auto-generadas (class cx cy bw bh)
│   ├── metadata/                    # JSON de metadatos por imagen
│   └── finetune/                    # Dataset preparado para yolo train
│       ├── images/train/
│       ├── images/val/
│       ├── labels/train/
│       ├── labels/val/
│       └── data.yaml
├── metrics/                         # Generado en producción — excluido de Git
│   └── production_metrics.jsonl     # Métricas de cada solicitud a /api/detect
├── test_results/                    # Resultados de pruebas — excluido de Git
│   └── test_history.jsonl
├── audio_output/                    # Archivos MP3 generados por TTS
├── test_images/                     # Imágenes de prueba
├── run.py                           # Punto de entrada
├── requirements.txt
└── .env                             # Variables de entorno (excluido de Git)
```

## Pipeline de procesamiento

```
Imagen (JPEG/PNG)
  │
  ├─ resize_image()           Máx 800px, ratio preservado
  │
  ├─ run_yolo()               YOLO26s — detección con umbrales por clase
  │
  ├─ analyze_spatial()        Cuadrícula 3×3 — posición egocéntrica + categoría
  │
  ├─ estimate_steps()         Heurística monocular — pasos por objeto
  │
  ├─ calculate_free_space()   Fracción bloqueada por columna (izq/centro/der)
  │
  ├─ decide_movement()        Instrucción: avanzar / desviar / detenerse
  │
  ├─ classify_scene()         LLM → tipo de escenario (sala, cocina, calle...)
  │
  ├─ generate_description()   LLM → descripción egocéntrica con pasos
  │
  ├─ build_narrative()        Escenario + descripción + instrucción
  │
  └─ log_metric()             Registra métricas en production_metrics.jsonl
```

## Instalación

```bash
# 1. Clonar y crear entorno virtual
git clone <repo>
cd vision-api-project
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus claves
```

### Variables de entorno (`.env`)

```
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
YOLO_WEIGHTS=yolo26s.pt
YOLO_IMGSZ=1280
YOLO_IOU=0.45
TTS_VOICE_NAME=es-ES-Neural2-A
TTS_SPEAKING_RATE=0.95
```

## Ejecución

```bash
python run.py
```

API disponible en: `http://127.0.0.1:8000`
Documentación Swagger: `http://127.0.0.1:8000/docs`

---

## Endpoints

### Producción

#### `POST /api/detect`
Detección + narrativa completa.

```
form-data:
  file                  JPEG/PNG
  confidence_threshold  float 0.0–1.0  (default: 0.35)
  debug                 bool           (default: false)
  audio                 bool           (default: false)
```

Respuesta JSON:
```json
{
  "status": "success",
  "narrativa_final": "Parece que estás en una sala de estar. Sofá a tu derecha...",
  "escenario": { "tipo": "sala de estar", "confianza": "alta" },
  "audio": { "disponible": true, "data_uri": "data:audio/mpeg;base64,..." },
  "metricas": {
    "total_ms": 2317,
    "deteccion_ms": 1.2,
    "objetos_detectados": 7,
    "confianza_prom": 0.758
  }
}
```

#### `POST /api/debug-detect`
Pipeline paso a paso — diagnóstico y validación.

#### `GET /api/health`
Estado del servicio, modelos activos y conteo del dataset.

---

### Dataset y Fine-Tuning

#### `POST /api/dataset/upload`
Almacena una imagen y la etiqueta automáticamente con YOLO26s.

```
form-data:
  file        JPEG/PNG
  scene_type  str   (default: "unknown")
  source      str   (default: "web3d")
  auto_label  bool  (default: true)
```

#### `GET /api/dataset/stats`
Estadísticas del dataset: total, etiquetadas, distribución por escena, top clases.

#### `POST /api/finetune/prepare`
Organiza el dataset en formato YOLO y genera `data.yaml`.

```
form-data:
  train_split  float  (default: 0.8)
  min_images   int    (default: 10)
```

Respuesta incluye el comando completo para ejecutar `yolo train`.

#### `GET /api/finetune/status`
Estado del dataset preparado y comando de entrenamiento.

---

### Métricas

#### `GET /api/metrics/summary?limit=500`
Promedio, p50, p90, p95, p99 de tiempos de respuesta en producción.

#### `GET /api/metrics/latency?limit=100`
Historial de latencias para graficar en frontend.

---

### Pruebas

#### `POST /api/test/functional`
Suite de 7 pruebas funcionales automáticas. Requiere servidor activo.

```
form-data:
  base_url  str  (default: "http://127.0.0.1:8000")
```

#### `POST /api/test/load`
Prueba de carga parametrizable.

```
form-data:
  n_requests   int    (default: 10)
  concurrency  int    (default: 3)
  base_url     str    (default: "http://127.0.0.1:8000")
  image_path   str    (default: "test_images/sala.jpg")
```

#### `GET /api/test/results?limit=20`
Historial de ejecuciones de pruebas (funcionales y carga), más recientes primero.

---

## Flujo de fine-tuning

```
1. Usar el sistema en producción (cliente Web 3D envía imágenes a /api/detect)
2. Cada imagen interesante → POST /api/dataset/upload  (se etiqueta automáticamente)
3. GET /api/dataset/stats  → verificar que finetune_ready = true (≥50 imágenes)
4. POST /api/finetune/prepare  → genera dataset/finetune/data.yaml
5. Ejecutar el comando retornado:
   yolo train model=yolo26s.pt data=dataset/finetune/data.yaml epochs=50 imgsz=640 batch=8
6. Reemplazar yolo26s.pt con los nuevos pesos (runs/detect/train/weights/best.pt)
```

## Narrativa de ejemplo

```
Parece que estás en una sala de estar.
Sofá a tu derecha a aproximadamente 2 pasos.
3 sillas frente a ti a aproximadamente 5 pasos.
Televisor al fondo a tu izquierda.
Puedes avanzar hacia el frente.
Tienes aproximadamente 4 pasos libres antes del primer obstáculo.
```

## Ramas del repositorio

| Rama | Descripción |
|---|---|
| `main` | Producción — YOLO26s + pipeline completo + endpoints de evaluación |
| `comparativa/multi-modelo` | Investigación — 4 modelos + batch (A9–A11) |