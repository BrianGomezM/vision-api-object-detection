# API de Generación de Descripciones Narrativas Egocéntricas

Sistema backend que procesa imágenes y genera descripciones auditivas
accesibles para personas con ceguera total, orientadas a la navegación
en entornos Web 3D.

---

## Modelos de detección disponibles

| Modelo | Versión | Framework | mAP COCO |
|--------|---------|-----------|----------|
| `yolo` | YOLO26-X (Ultralytics 2026) | PyTorch | Estado del arte |
| `fasterrcnn` | ResNet-101-FPN 3x (Detectron2) | PyTorch | 42.0 |
| `maskrcnn` | ResNet-101-FPN 3x + segmentación (Detectron2) | PyTorch | 42.9 |
| `ssd` | SSD-MobileNet V2 (TF Hub / TF OD API) | TensorFlow | 21.3 |

---

## Estructura del proyecto

```
app/
├── main.py                       # FastAPI app + registro de routers
├── routes/
│   ├── detect.py                 # Endpoints: /detect, /detect-all, /debug-detect, /health
│   └── batch.py                  # Endpoint: /run-batch (pruebas masivas)
├── services/
│   ├── yolo_service.py           # YOLO26 — detección principal
│   ├── fasterrcnn_service.py     # Faster R-CNN — Detectron2
│   ├── maskrcnn_service.py       # Mask R-CNN — Detectron2 + máscaras
│   ├── ssd_service.py            # SSD-MobileNet V2 — TF Hub
│   ├── spatial_analyzer.py       # Cuadrícula 3×3 + categorías + prioridad
│   ├── step_estimator.py         # Estimación de pasos hasta cada objeto
│   ├── free_space_analyzer.py    # Zonas navegables libres
│   ├── risk_engine.py            # Decisión de movimiento con pasos libres
│   ├── llm_enhancer.py           # Descripción egocéntrica vía Groq/Llama
│   └── scene_classifier.py       # Clasificación de escenario vía LLM
└── utils/
    └── translator.py             # Traducción EN→ES dinámica con caché

run.py                            # Punto de entrada
diagnostico_yolo.py               # Script de diagnóstico de detección
test_images/                      # Imágenes de prueba
test_results_json/                 # JSON por combinación imagen/modelo/threshold
charts/                           # Gráficas generadas por batch
test_results_summary.xlsx         # Resultados consolidados

ELIMINADO (no usar):
  app/services/narrative_service.py   — reemplazado por llm_enhancer
  app/routes/detect_old.py            — versión obsoleta
```

---

## Instalación

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Detectron2 (Faster R-CNN y Mask R-CNN) — instalación especial:
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Variables de entorno (`.env`):
```
GROQ_API_KEY=gsk_...
YOLO_WEIGHTS=yolo26x.pt
YOLO_IMGSZ=1280
```

---

## Ejecución

```bash
python run.py
```

API disponible en `http://127.0.0.1:8000`
Documentación automática en `http://127.0.0.1:8000/docs`

---

## Endpoints

### POST `/api/detect`
Detección individual + narrativa completa.

Parámetros form-data:
- `model`: `yolo` | `fasterrcnn` | `maskrcnn` | `ssd`
- `file`: imagen
- `confidence_threshold`: 0.0 – 1.0 (default 0.35)
- `debug`: true | false

Respuesta incluye:
- `narrativa_final`: escenario + objetos con pasos + instrucción
- `escenario`: tipo y confianza del escenario detectado
- `metricas`: tiempos desglosados de cada etapa del pipeline

### POST `/api/detect-all`
Ejecuta los 4 modelos sobre la misma imagen. Usado para comparativa A10.

### POST `/api/run-batch`
Procesa todas las imágenes en `test_images/` con todos los modelos
y thresholds (0.3, 0.5, 0.7). Genera Excel y 6 gráficas comparativas.

### GET `/api/health`
Estado del servicio y versiones de modelos cargados.

---

## Pipeline de procesamiento

```
Imagen → Modelo (YOLO26 / FasterRCNN / MaskRCNN / SSD)
       → Análisis espacial 3×3 (lateral + profundidad + categoría)
       → Estimación de pasos (heurística por tamaño + posición vertical)
       → Análisis de espacio libre (3 columnas, sin small_objects)
       → Decisión de movimiento (pasos libres al frente o desvío)
       → Clasificación de escenario (LLM → sala, cocina, pasillo...)
       → Descripción LLM egocéntrica (objetos + pasos)
       → Narrativa final = escenario + descripción + instrucción
```

---

## Ejemplo de narrativa de salida

```
Parece que estás en una sala de estar.
Sofá a tu derecha a aproximadamente 2 pasos.
3 sillas frente a ti a aproximadamente 5 pasos.
2 sillas a tu izquierda a aproximadamente 5 pasos.
Televisor al fondo a tu izquierda.
Puedes avanzar hacia el frente.
Tienes aproximadamente 4 pasos libres antes del primer obstáculo.
```

---

## Diagnóstico de detección

```bash
python diagnostico_yolo.py test_images/prueba.jpg 0.35
```

Muestra qué detecta YOLO26, qué pasa/no pasa los filtros, y la
distribución en la cuadrícula 3×3.