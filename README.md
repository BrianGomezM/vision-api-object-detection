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
- [Qué se necesita antes de empezar](#qué-se-necesita-antes-de-empezar)
- [Cómo instalar el proyecto](#cómo-instalar-el-proyecto)
- [Cómo poner en marcha el proyecto](#cómo-poner-en-marcha-el-proyecto)
- [Cómo comprobar que funciona](#cómo-comprobar-que-funciona)
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
│   │   └── tts_service.py             # Síntesis de voz (edge-tts, sin costo)
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

## Qué se necesita antes de empezar

Para usar este proyecto en una computadora hacen falta tres cosas:

1. **Python** (versión 3.11), que es el programa que ejecuta el código.
2. Una **clave de Groq**, gratuita, que se obtiene creando una cuenta en
   [console.groq.com](https://console.groq.com). Esta clave permite generar
   las descripciones en español; sin ella el proyecto no puede iniciar.
3. Opcionalmente, **Docker Desktop**, si se prefiere ejecutar el proyecto
   de la misma forma en que corre en el servidor de producción, en vez de
   instalar cada programa por separado.

No hace falta tarjeta de crédito ni pagar nada para obtener la clave de
Groq ni para ejecutar el proyecto.

---

## Cómo instalar el proyecto

1. Descargar el proyecto a la computadora (clonar el repositorio o
   descargarlo como archivo comprimido y extraerlo).
2. Abrir una terminal dentro de la carpeta del proyecto.
3. Crear un espacio separado para instalar los programas que necesita el
   proyecto, sin mezclarlos con el resto de la computadora:

   ```bash
   python -m venv venv
   venv\Scripts\activate          # en Windows
   source venv/bin/activate       # en Mac o Linux
   ```

4. Instalar todo lo que el proyecto necesita para funcionar:

   ```bash
   pip install -r requirements.txt
   ```

   Este paso puede tardar varios minutos la primera vez, porque descarga
   el modelo de inteligencia artificial y sus componentes.

5. Crear un archivo de configuración llamado `.env` en la carpeta principal
   del proyecto, con este contenido:

   ```
   GROQ_API_KEY=la_clave_obtenida_en_groq
   YOLO_WEIGHTS=yolo26s.pt
   YOLO_IMGSZ=1280
   YOLO_IOU=0.45
   TTS_VOICE_NAME=es-ES-AlvaroNeural
   TTS_SPEAKING_RATE=0.95
   CORS_ORIGINS=https://direccion-del-sitio-web-que-lo-va-a-usar.com
   ```

   Solo `GROQ_API_KEY` es obligatoria. Las demás ya tienen un valor por
   defecto y pueden dejarse como están. `CORS_ORIGINS` solo es necesaria
   si otra página web (por ejemplo, la interfaz visual del proyecto) va a
   consumir este servicio; puede tener varias direcciones separadas por
   coma.

---

## Cómo poner en marcha el proyecto

Hay dos formas de hacerlo. Cualquiera de las dos deja el proyecto
funcionando en la misma dirección.

### Opción 1: directamente con Python

```bash
python run.py
```

### Opción 2: con Docker

Esta opción usa exactamente la misma configuración con la que el proyecto
corre en el servidor de producción.

```bash
docker build -t vision-api .
docker run -p 8000:8000 --env-file .env vision-api
```

En ambos casos, después de unos segundos el proyecto queda disponible en
la propia computadora, en esta dirección:

```
http://127.0.0.1:8000
```

---

## Cómo comprobar que funciona

1. Abrir en el navegador la dirección `http://127.0.0.1:8000/docs`.
   Se muestra una página con la lista de todas las funciones disponibles
   del proyecto, y permite probarlas sin necesidad de escribir código.

2. Comprobar el estado general: abrir
   `http://127.0.0.1:8000/api/health`. Si el proyecto está funcionando,
   se muestra un mensaje indicando que el servicio está activo, junto con
   el estado de cada componente (detección de objetos, generación de
   texto y de voz).

3. Probar la función principal: en la página `/docs`, buscar
   `POST /api/detect`, presionar "Try it out", y subir una de las
   imágenes de ejemplo incluidas en la carpeta `test_images`. El
   resultado incluye una descripción en español de lo que aparece en la
   imagen.

4. Probar automáticamente que todo funciona correctamente: en la misma
   página `/docs`, buscar `POST /api/test/functional` y ejecutarlo. Este
   paso revisa por sí solo varias funciones del proyecto y devuelve un
   resumen de cuáles pasaron y cuáles no.

5. Probar que el proyecto soporta varias solicitudes al mismo tiempo: en
   `/docs`, buscar `POST /api/test/load` y ejecutarlo. Simula varias
   personas usando el servicio a la vez y muestra cuánto tiempo tarda en
   responder.

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

El backend se despliega en Azure App Service (Linux) como **contenedor
Docker** — ver [Dockerfile](Dockerfile). Se eligió este enfoque en vez del
despliegue por código (Oryx) porque Oryx reinstalaba `torch`/`ultralytics`
y las libs de sistema (`libxcb1`, etc. — requeridas por
`opencv-python-headless`) en **cada arranque del contenedor**, no solo en
cada deploy, haciendo cualquier reinicio lento y dependiente de la
disponibilidad de los mirrors de Debian en ese momento. Con Docker, todo
eso queda horneado en la imagen en build time.

1. **Plan de App Service:** se recomienda **B1** (1.75 GB RAM) — el modelo
   YOLO26s sobre `torch` necesita más memoria de la que ofrece el plan
   gratuito F1.
2. **CI/CD:** `.github/workflows/main_visionnav-api.yml` construye la
   imagen en cada push a `main`, la publica en GitHub Container Registry
   (`ghcr.io/<owner>/vision-api-object-detection`) y actualiza el App
   Service para que la use.
3. **Configuración de la pila (una sola vez, manual en el Portal):**
   *Configuración → Configuración general → Pila* → cambiar a **Contenedor
   Docker** → Imagen única → Origen: *Otros registros de contenedores* →
   URL del registro `https://ghcr.io` → Imagen y etiqueta
   `ghcr.io/<owner>/vision-api-object-detection:latest`. El paquete en
   GitHub debe estar en visibilidad **pública** (Settings del paquete en
   GitHub) para que Azure pueda descargarlo sin credenciales.
4. **Variables de entorno:** configurar en *Configuración → Variables de
   entorno* las mismas claves del `.env` local (`GROQ_API_KEY`,
   `CORS_ORIGINS` con el dominio del cliente desplegado en Vercel, etc.)
   — estas nunca van dentro de la imagen. No se requiere ninguna clave de
   pago: `GROQ_API_KEY` es gratuita y el audio se genera con `edge-tts`,
   que no necesita clave ni tarjeta.
5. Los pesos de YOLO26s no se versionan en Git; si no están presentes en el
   contenedor, Ultralytics los descarga automáticamente en el primer
   arranque.

---

## Ramas del repositorio

| Rama | Descripción |
|---|---|
| `main` | Producción — YOLO26s + pipeline completo + endpoints de evaluación |
| `comparativa/multi-modelo` | Investigación — 4 modelos + batch (A9–A11) |
