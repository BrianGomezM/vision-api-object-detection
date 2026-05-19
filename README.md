# API de Generación de Descripciones Narrativas Egocéntricas

Sistema backend que procesa imágenes de entornos Web 3D y genera
descripciones auditivas accesibles para personas con ceguera total.

**Modelo de detección:** YOLO26s (Ultralytics 2026) — seleccionado tras
evaluación comparativa contra Faster R-CNN, Mask R-CNN y SSD (ver rama
`comparativa/multi-modelo`).

## Estructura del proyecto


vision-api-project/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── routes/
│   │   └── detect.py              # /detect, /debug-detect, /health
│   ├── services/
│   │   ├── yolo_service.py        # Detección YOLO26s
│   │   ├── spatial_analyzer.py    # Cuadrícula 3×3 + categorías + prioridad
│   │   ├── step_estimator.py      # Estimación de pasos (heurística monocular)
│   │   ├── free_space_analyzer.py # Zonas navegables libres
│   │   ├── risk_engine.py         # Decisión de movimiento
│   │   ├── llm_enhancer.py        # Descripción egocéntrica (Groq/Llama)
│   │   └── scene_classifier.py    # Clasificación de escenario (Groq/Llama)
│   └── utils/
│       └── translator.py          # Traducción EN→ES dinámica con caché
├── app/experimental/              # Modelos comparativos (no se cargan en producción)
│   ├── fasterrcnn_service.py
│   ├── maskrcnn_service.py
│   ├── ssd_service.py
│   ├── batch.py
│   └── diagnostico_yolo.py
├── test_images/                   # Imágenes de prueba
├── run.py                         # Punto de entrada
├── requirements.txt
└── .env                           # Variables de entorno (excluido de Git)

## Pipeline de procesamiento


Imagen (JPEG/PNG)
  │
  ├─ resize_image()          Máx 800px, ratio preservado
  │
  ├─ run_yolo()              YOLO26s — detección con umbrales por clase
  │
  ├─ analyze_spatial()       Cuadrícula 3×3 — posición egocéntrica + categoría
  │
  ├─ estimate_steps()        Heurística monocular — pasos por objeto
  │
  ├─ calculate_free_space()  Fracción bloqueada por columna (izq/centro/der)
  │
  ├─ decide_movement()       Instrucción: avanzar / desviar / detenerse
  │
  ├─ classify_scene()        LLM → tipo de escenario (sala, cocina, calle...)
  │
  ├─ generate_description()  LLM → descripción egocéntrica con pasos
  │
  └─ build_narrative()       Escenario + descripción + instrucción

## Instalación

bash
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


### Variables de entorno (`.env`)

GROQ_API_KEY=gsk_...
YOLO_WEIGHTS=yolo26s.pt
YOLO_IMGSZ=1280
YOLO_IOU=0.45



## Ejecución

# Desarrollo (reload automático)
python run.py

# Producción
python run.py --prod

API disponible en: `http://127.0.0.1:8000`
Documentación Swagger: `http://127.0.0.1:8000/docs`


## Endpoints

### `POST /api/detect`
Detección + narrativa completa.

form-data:
  file                  JPEG/PNG
  confidence_threshold  float 0.0–1.0  (default: 0.35)
  debug                 bool           (default: false)


Respuesta:
json
{
  "status": "success",
  "narrativa_final": "Parece que estás en una sala de estar. Sofá a tu derecha...",
  "escenario": { "tipo": "sala de estar", "confianza": "alta" },
  "metricas": { "total_ms": 2317, "deteccion_ms": 1228, "objetos_detectados": 14 }
}


### `POST /api/debug-detect`
Pipeline paso a paso — útil para diagnóstico y validación.

### `GET /api/health`
Estado del servicio y configuración activa del modelo.

## Ejemplo de narrativa

Parece que estás en una sala de estar.
Sofá a tu derecha a aproximadamente 2 pasos.
3 sillas frente a ti a aproximadamente 5 pasos.
Televisor al fondo a tu izquierda.
Puedes avanzar hacia el frente.
Tienes aproximadamente 4 pasos libres antes del primer obstáculo.

## Ramas del repositorio

| Rama | Descripción |
|---|---|
| `main` | Producción — YOLO26s + pipeline completo |
| `comparativa/multi-modelo` | Investigación — 4 modelos + batch (A9–A11) |
