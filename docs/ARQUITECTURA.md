# Arquitectura

VisionNav API es un monolito modular: un solo servicio y un solo repositorio, con módulos de responsabilidades separadas. El núcleo procesa la imagen y no depende del framework web; las demás capas lo exponen por HTTP o lo usan para evaluarlo.

## Capas

| Capa | Ubicación | Responsabilidad |
|---|---|---|
| Núcleo | `app/core/`, `app/services/`, `app/utils/` | Detección, análisis espacial, pasos, espacio libre, decisión de movimiento, escenario, narrativa y voz |
| Adaptadores HTTP | `app/routes/`, `app/errors.py`, `app/security.py` | Endpoints, validación de entradas, autenticación y traducción de errores |
| Plataforma | `app/profiles.py`, `app/ratelimit.py`, `app/observability.py`, `app/telemetry.py`, `app/storage.py`, `app/deploy_identity.py`, `app/experiment.py` | Perfil de ejecución, límite de solicitudes, identificador de solicitud, telemetría, rutas de datos e identidad del despliegue |
| Evaluación | `app/experimental/`, `app/catalog/`, `app/study_export.py`, `scripts/`, `evaluation/` | Comparación de modelos, catálogo de pruebas, sesiones con usuarios y resultados |

La prueba `tests/test_architecture.py` verifica que el núcleo no importe FastAPI ni los módulos de evaluación.

## Flujo de una solicitud

`POST /api/detect` delega en `app/core/pipeline.run()`, la misma función que usan los scripts de evaluación:

1. **Validación** (`routes/detect.py`): archivo no vacío, de hasta 10 MB, JPEG o PNG identificado por su contenido.
2. **Preprocesamiento** (`core/pipeline.py`): conversión a RGB y reducción a 800 px en el lado mayor si hace falta.
3. **Detección** (`services/yolo_service.py`): YOLO26s con umbral interno de 0,15, filtro de clases de navegación y umbral efectivo por clase (el menor entre el de la clase y el de la solicitud).
4. **Análisis espacial** (`services/spatial_analyzer.py`): columna (izquierda, centro, derecha), banda de cercanía, categoría y prioridad de cada objeto.
5. **Pasos** (`services/step_estimator.py`): estimación ordinal de 1 a 7 pasos.
6. **Espacio libre y decisión** (`services/free_space_analyzer.py`, `services/risk_engine.py`): fracción bloqueada por columna e instrucción de movimiento.
7. **Escenario** (`services/scene_classifier.py`): tipo de lugar a partir de los objetos detectados.
8. **Narrativa** (`services/llm_enhancer.py`): descripción en segunda persona redactada por el modelo de lenguaje; si no responde, se usa una plantilla.
9. **Voz** (`services/tts_service.py`): audio MP3 con Gemini TTS o Azure AI Speech.

## Perfiles de ejecución

La variable `APP_PROFILE` decide qué endpoints se publican.

| Perfil | Uso | Endpoints |
|---|---|---|
| `development` | Desarrollo local (valor por defecto) | Todos, incluida la documentación `/docs` y las herramientas internas |
| `study` | Sesiones de evaluación con usuarios | Detección, estado, voces, sesiones y catálogo; exige `API_KEYS` |
| `production` | Servicio desplegado (perfil del Dockerfile) | Detección, estado, voces y, con clave, métricas y rutas del investigador |

## Configuración registrada

`experimental_config.yaml` guarda la configuración con la que se evaluó el sistema: pesos y su SHA-256, versiones de las librerías, parámetros y huella de los archivos del núcleo. En el perfil `production`, `app/deploy_identity.py` compara el entorno con ese registro al arrancar y detiene el servicio si no coincide. Ver [REPRODUCIBILIDAD.md](REPRODUCIBILIDAD.md).
