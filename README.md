# Vision API — Navegación Egocéntrica para Personas con Ceguera Total

API backend que convierte imágenes de entornos Web 3D en descripciones
auditivas, pensada como apoyo de navegación para personas con ceguera
total. Detecta objetos con un modelo YOLO, los ubica en el espacio relativo
al usuario, estima la distancia en pasos y genera una narrativa hablada en
español — todo en una sola petición HTTP.

Este proyecto es el backend de un trabajo de grado orientado a
accesibilidad digital.

**Modelo de detección:** YOLO26s (Ultralytics 2026) — seleccionado tras una
evaluación comparativa contra Faster R-CNN, Mask R-CNN y SSD (ver rama
`comparativa/multi-modelo`).

---

## Tabla de contenidos

- [Cómo funciona](#cómo-funciona)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Endpoints](#endpoints)
- [Flujo de fine-tuning](#flujo-de-fine-tuning)
- [Despliegue](#despliegue)
- [Ramas del repositorio](#ramas-del-repositorio)

---

## Cómo funciona

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
  ├─ save_annotated_image()   Imagen con bounding boxes (debug/visualización)
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

**Narrativa de ejemplo:**

```
Parece que estás en una sala de estar.
Sofá a tu derecha a aproximadamente 2 pasos.
3 sillas frente a ti a aproximadamente 5 pasos.
Televisor al fondo a tu izquierda.
Puedes avanzar hacia el frente.
Tienes aproximadamente 4 pasos libres antes del primer obstáculo.
```

---

## Estructura del proyecto

```
vision-api-project/
├── app/
│   ├── main.py                        # FastAPI app, CORS, routers, eventos de ciclo de vida
│   ├── routes/
│   │   ├── detect.py                  # /detect, /debug-detect, /health
│   │   ├── evaluation.py              # /dataset/*, /metrics/*, /test/*, /finetune/*
│   │   └── metrics.py                 # /metrics, /feedback
│   ├── services/
│   │   ├── yolo_service.py            # Detección YOLO26s
│   │   ├── spatial_analyzer.py        # Cuadrícula 3×3 + categorías + prioridad
│   │   ├── step_estimator.py          # Estimación de pasos (heurística monocular)
│   │   ├── free_space_analyzer.py     # Zonas navegables libres
│   │   ├── risk_engine.py             # Decisión de movimiento
│   │   ├── llm_enhancer.py            # Descripción egocéntrica (Groq/Llama)
│   │   ├── scene_classifier.py        # Clasificación de escenario (Groq/Llama)
│   │   ├── detection_visualizer.py    # Imagen anotada con bounding boxes
│   │   └── tts_service.py             # Síntesis de voz (Google Cloud TTS)
│   └── utils/
│       ├── translator.py              # Traducción EN→ES dinámica con caché
│       └── groq_client.py             # Singleton cliente Groq
├── app/experimental/                  # Modelos comparativos (no se cargan en prod)
│   ├── fasterrcnn_service.py
│   ├── maskrcnn_service.py
│   ├── ssd_service.py
│   ├── batch.py
│   └── diagnostico_yolo.py
├── dataset/                           # Generado en producción — excluido de Git
│   ├── images/                        # Imágenes subidas con /api/dataset/upload
│   ├── labels/                        # Etiquetas YOLO auto-generadas (class cx cy bw bh)
│   ├── metadata/                      # JSON de metadatos por imagen
│   └── finetune/                      # Dataset preparado para yolo train
├── metrics/                           # Generado en producción — excluido de Git
├── test_results/                      # Resultados de pruebas — excluido de Git
├── audio_output/                      # MP3 generados por TTS — excluido de Git
├── detections_output/                 # Imágenes anotadas — excluido de Git
├── test_images/                       # Imágenes de prueba
├── run.py                             # Punto de entrada local
├── startup.sh                         # Comando de arranque para Azure App Service
├── requirements.txt                   # Dependencias de producción (lo que despliega Azure)
├── requirements-dev.txt               # + modelos comparativos y herramientas de análisis
└── .env                                # Variables de entorno (excluido de Git)
```

---

## Instalación

```bash
# 1. Clonar y crear entorno virtual
git clone <repo>
cd vision-api-project
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias de producción
pip install -r requirements.txt

# Si además vas a entrenar o comparar contra Faster R-CNN / Mask R-CNN / SSD:
pip install -r requirements-dev.txt

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
CORS_ORIGINS=https://tu-cliente.vercel.app
```

`CORS_ORIGINS` acepta varios orígenes separados por coma. En desarrollo,
`localhost:3000`/`3001` ya están permitidos por defecto.

---

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
  "imagen_anotada": { "disponible": true, "url": "/detections/detection_xxx.jpg" },
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

### Métricas y evaluación de usuarios

#### `GET /api/metrics/summary?limit=500`
Promedio, p50, p90, p95, p99 de tiempos de respuesta en producción.

#### `GET /api/metrics/latency?limit=100`
Historial de latencias para graficar en frontend.

#### `POST /api/feedback` / `GET /api/feedback`
Registro y consulta de evaluación de usuarios (escala Likert).

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

---

## Despliegue

El backend está listo para Azure App Service (Linux, Python). Resumen del
proceso:

1. **Plan de App Service:** se recomienda **B1** (1.75 GB RAM) — el modelo
   YOLO26s sobre `torch` necesita más memoria de la que ofrece el plan
   gratuito F1.
2. **Build:** Azure instala automáticamente `requirements.txt` (solo
   dependencias de producción; los modelos comparativos quedan fuera).
3. **Comando de inicio:** configurar `bash startup.sh` en
   *Configuración → General → Comando de inicio*. Internamente usa
   `gunicorn` con worker de `uvicorn` para servir la app ASGI de FastAPI.
4. **Variables de entorno:** configurar en *Configuración → Variables de
   entorno* las mismas claves del `.env` local (`GROQ_API_KEY`,
   `GOOGLE_API_KEY`/credenciales de Google Cloud, `CORS_ORIGINS` con el
   dominio del cliente desplegado en Vercel, etc.).
5. Los pesos de YOLO26s no se versionan en Git; si no están presentes en el
   contenedor, Ultralytics los descarga automáticamente en el primer
   arranque.

---

## Ramas del repositorio

| Rama | Descripción |
|---|---|
| `main` | Producción — YOLO26s + pipeline completo + endpoints de evaluación |
| `comparativa/multi-modelo` | Investigación — 4 modelos + batch (A9–A11) |
