# Revisión arquitectónica: propuesta de monolito modular

**Fecha:** 2026-09-26. **Estado:** diseño; **implementado parcialmente el 2026-09-27** (ver abajo).

> **Estado de implementación (2026-09-27, rama `fase3/infraestructura`).** Se aplicó el refactor aprobado priorizando la **separación de dependencias** y **sin mover `services/` a `core/`**, porque ese movimiento sería solo estético.
>
> **Hecho:**
> - línea base de regresión y contratos (etapa 0);
> - `app/core/pipeline.py` como orquestador único (1);
> - `app/telemetry.py` y `app/profiles.py`, más la prueba de arquitectura (2);
> - traductor con el diccionario fijo como fuente de verdad (3);
> - perfil `production` y `routes/health.py` (4);
> - eliminación de `GET /api/metrics` y traslado del script de diagnóstico (5).
>
> **No hecho, deliberadamente:**
> - paquete `app/evaluation/` (catálogo y estudio siguen en `app/catalog` y `app/routes`, y se montan por perfil);
> - división de `routes/evaluation.py`, que en su conjunto es solo de development;
> - `request_id` y manifests (se harán con F4);
> - alias por `sys.modules`, innecesarios porque no se movieron módulos del núcleo.
>
> **Detalle:** checkpoint del refactor en la conversación del 2026-09-27; `docs/DATOS_PERSISTENCIA.md`; `docs/AUDITORIA_REGLA_UMBRAL_FASE10.md`.

**Base:** backend `fase3/infraestructura` @ `d016557`; cliente `fase3/infraestructura` @ `0cee9ac`.

**Qué no se hizo:** no se movió código, no se hicieron commits, no se ejecutó YOLO, el LLM ni el TTS, y el Dataset 1 no se tocó.

**Premisa.** El producto del trabajo de grado es el pipeline *imagen → detección → análisis espacial → narrativa → TTS → respuesta*. Todo lo demás es infraestructura para evaluarlo. Se propone separarlos **dentro del mismo repositorio y del mismo proceso** (monolito modular), sin microservicios y sin cambiar los contratos públicos.

---

## 1. Arquitectura actual

```
app/
  main.py            create_app(): perfiles development | study; CORS; montaje de routers
  security.py        API keys, rate limit, APP_PROFILE                         [plataforma]
  storage.py         DATA_ROOT y rutas de datos                               [plataforma]
  routes/
    detect.py   673  /detect, /debug-detect, /health, /tts/models  +  ORQUESTACIÓN DEL PIPELINE
                     (_run_full_pipeline, resize_image, build_narrative, _build_annotated_info)
    evaluation.py 965 dataset/upload+stats, finetune/prepare+status, metrics/summary+latency,
                     test/functional+load+results, log_metric() (telemetría usada por /detect)
    metrics.py  246  GET /metrics (en memoria) + POST/GET /feedback (Likert)
    study.py    247  /study/sessions*  (participantes y respuestas)
    catalog.py   42  /catalog, /catalog/stimuli/{id}/image
  services/          yolo_service, spatial_analyzer, step_estimator, free_space_analyzer,
                     risk_engine, scene_classifier, llm_enhancer, tts_service,
                     detection_visualizer                                    [NÚCLEO]
                     diagnostico_yolo.py  (script CLI; duplicado en experimental/)
  utils/             groq_client, translator  [NÚCLEO]  ·  uploads  [plataforma]
  catalog/           catalog.yaml + loader.py                                 [EVALUACIÓN]
  experimental/      SSD / Faster / Mask R-CNN + batch (selección de modelo del Cap. 3; no montado)
scripts/             22 scripts de las fases 2A–10 (importan app.services.*, app.utils.*, app.routes.detect)
evaluation/          imágenes Web3D, metadatos, resultados (evidencia; no es código)
stimuli/dataset1/    18 PNG congelados + manifest
tests/               64 pruebas
```

### Flujo real del núcleo

Es `routes/detect.py::_run_full_pipeline`, con estas etapas:

1. `resize_image`;
2. `run_yolo`;
3. `analyze_spatial`;
4. `estimate_steps`;
5. `save_annotated_image` (escribe en disco);
6. `calculate_free_space`;
7. `decide_movement`;
8. `classify_scene` y `generate_description`, en paralelo (Groq);
9. `build_narrative`.

Después, en el endpoint, van `synthesize_and_save` (TTS, escribe en disco) y `log_metric`.

### Dependencias verificadas

Se obtuvieron con `grep` de los imports:

- **El núcleo no depende de la evaluación, salvo en un punto:** `/detect` importa `log_metric` desde `app.routes.evaluation`.
- **13 scripts de evidencia** importan módulos del núcleo **por su ruta** (`app.services.*`, `app.utils.*`, `app.routes.detect.resize_image`).
- **Las pruebas** importan `app.routes.study._STUDY_DIR`, `app.routes.evaluation._resolve_test_image`, `app.routes.catalog.get_catalog`, `app.services.yolo_service`, `app.security` y `app.storage`, y parchean atributos de los módulos (`monkeypatch.setattr`).
- **El cliente** consume 20 rutas. Además de detect, health, tts/models, study y catalog, usa dataset, finetune, metrics, feedback, test y debug.

## 2. Problemas de separación encontrados

| # | Problema | Evidencia | Consecuencia |
|---|---|---|---|
| P1 | La lógica del producto vive dentro de un archivo de rutas HTTP | `routes/detect.py` mezcla el pipeline, el ensamblado de la narrativa, el preprocesado, `/health`, `/tts/models` y `/debug-detect` | La evaluación (F4) tendría que llamar a una función "privada" de un router, y los scripts ya importan `resize_image` desde ahí |
| P2 | El núcleo depende de un router de evaluación | `detect.py` → `from app.routes.evaluation import log_metric` | La telemetría de producción vive en el módulo de dataset y pruebas |
| P3 | `routes/evaluation.py` (965 líneas) junta 4 responsabilidades | fine-tuning, telemetría (lectura), banco de pruebas de la API (auto-HTTP) y logging | Imposible montar solo una parte por perfil |
| P4 | `routes/metrics.py` mezcla métricas en memoria con valoraciones de usuarios | `record_request()` **nunca se llama** (grep), así que `GET /api/metrics` siempre devuelve contadores vacíos. `/feedback` es una encuesta Likert, es decir, evaluación con usuarios. | Código muerto y una responsabilidad mal ubicada |
| P5 | `/health` informa sobre datos de evaluación | Cuenta imágenes del dataset de fine-tuning y métricas, y lista endpoints de evaluación | El health del producto depende de datos experimentales |
| P6 | Efectos en disco dentro del núcleo | `save_annotated_image` (en el pipeline) y `synthesize_and_save` (en el endpoint), cada uno con su propia rotación | La política de persistencia está repartida entre los servicios; no hay un identificador de procesamiento |
| P7 | Script CLI dentro de `services/` | `services/diagnostico_yolo.py` (con un hack de `sys.path`), casi duplicado en `experimental/` | Confunde qué es el núcleo |
| P8 | Configuración dispersa | Cada módulo lee `os.getenv` al importarse | No existe una "foto" única de la configuración efectiva para el manifest del experimento |
| P9 | Perfil por defecto inseguro para el despliegue | `APP_PROFILE` vale `development` por defecto y el Dockerfile no lo fija | Un despliegue expone debug, dataset, fine-tuning y pruebas |
| P10 | La reproducibilidad depende del estado de la máquina | `translate_label` consulta **primero la caché en disco** y después el diccionario estático, y como último recurso Google Translate en línea | Una caché antigua puede cambiar las etiquetas de la narrativa entre máquinas |
| P11 | No hay integración continua de pruebas | El workflow de GitHub solo construye la imagen Docker y despliega | Una refactorización no queda protegida |
| P12 | *(Evidencia, fuera de la arquitectura)* El script de la fase 10 filtra con `min(...)` y producción con `max(...)` | `scripts/evaluation/phase10_regression_test_images.py:48` frente a `yolo_service.py:325` | El README de la fase 10 dice "igual que producción". **Hay que revisarlo antes de citar esa fase en la tesis.** |

## 3. Arquitectura propuesta

Un proceso y un repositorio, con **cuatro zonas** y reglas de dependencia comprobadas por pruebas:

```
            ┌──────────────── app.main:create_app(profile) ────────────────┐
            │                                                               │
   PRODUCTO │  app/api (detect, health)  ──►  app/core  (pipeline puro)     │
            │                                   ▲                           │
 EVALUACIÓN │  app/evaluation (catalog, study, devtools, runner) ─┘         │
            │                                                               │
 PLATAFORMA │  app/platform-lite: security.py, storage.py, telemetry.py,    │
            │                     uploads.py   (usados por todos)           │
            └───────────────────────────────────────────────────────────────┘
   FUERA DE LA IMAGEN: scripts/ (evidencia de fases), evaluation/ (datos y evidencia),
                       tests/, docs/, generador de escenas (repo aparte)
```

**Reglas de dependencia**, que una prueba de arquitectura hará cumplir:

1. `app/core` **no importa** FastAPI, `app.api`, `app.evaluation` ni `app.routes`.
2. `app/api` solo importa `app.core` y la plataforma.
3. `app/evaluation` puede importar `app.core` (para evaluarlo) y la plataforma; nunca al revés.
4. Los perfiles deciden qué routers se montan. El núcleo no conoce los perfiles.

**Por qué ayuda a F4.** El ejecutor de estímulos congelados llama **a la misma función** `core.pipeline.run()` que usa `/detect`. El endpoint pasa a ser un adaptador fino: HTTP a pipeline a JSON. Así la evaluación mide exactamente el producto, sin pasar por HTTP ni por la autenticación.

## 4. Árbol de carpetas propuesto

Se conservan los **nombres de archivo** del núcleo. La tesis cita `yolo_service.py`, `spatial_analyzer.py`, etc., y esas referencias seguirán siendo válidas.

```
app/
  main.py                         (sin cambio de ruta: app.main:app)
  security.py                     (sin mover)            plataforma
  storage.py                      (sin mover; layout v2) plataforma
  telemetry.py                    NUEVO ← log_metric/_read_metrics de routes/evaluation.py
  core/                           PRODUCTO — sin FastAPI
    pipeline.py                   NUEVO ← _run_full_pipeline, resize_image, build_narrative,
                                          normalize_threshold (de routes/detect.py)
    config_snapshot.py            NUEVO — solo lectura: foto de la config efectiva (manifest)
    detector/yolo_service.py      ← services/yolo_service.py
    spatial/spatial_analyzer.py   ← services/spatial_analyzer.py
    spatial/step_estimator.py     ← services/step_estimator.py
    spatial/free_space_analyzer.py← services/free_space_analyzer.py
    spatial/risk_engine.py        ← services/risk_engine.py
    narrative/scene_classifier.py ← services/scene_classifier.py
    narrative/llm_enhancer.py     ← services/llm_enhancer.py
    narrative/groq_client.py      ← utils/groq_client.py
    narrative/translator.py       ← utils/translator.py
    tts/tts_service.py            ← services/tts_service.py
    output/detection_visualizer.py← services/detection_visualizer.py
  api/                            PRODUCTO — adaptadores HTTP
    detect.py                     ← parte /detect y /tts/models de routes/detect.py (fino)
    health.py                     ← /health de routes/detect.py (básico + detallado por perfil)
  evaluation/                     INFRAESTRUCTURA DE EVALUACIÓN (mismo proceso)
    catalog/                      ← app/catalog/ (catalog.yaml, loader.py) + routes/catalog.py
    study/                        ← routes/study.py (+ feedback Likert de routes/metrics.py)
    devtools/                     solo perfil development
      debug.py                    ← /debug-detect de routes/detect.py
      dataset.py                  ← dataset/* y finetune/* de routes/evaluation.py
      api_tests.py                ← test/functional, test/load, test/results
      telemetry_report.py         ← metrics/summary, metrics/latency
    metrics/                      FUTURO (tras aprobar CP3B): emparejamiento, IoU, Wilson, monotonía
    runner/                       FUTURO (F4): ejecutor de estímulos congelados + manifests
  experimental/                   SIN CAMBIOS (evidencia del Cap. 3; no montado)
  routes/  services/  utils/  catalog/   ALIAS DE COMPATIBILIDAD (ver §6)
scripts/                          SIN CAMBIOS (+ scripts/diagnostics/diagnostico_yolo.py)
stimuli/dataset1/                 SIN CAMBIOS
evaluation/                       SIN CAMBIOS (datos y evidencia; no es código de la app)
tests/
  core/  api/  evaluation/  platform/  architecture/   (reorganización de las 64 pruebas + nuevas)
```

**Conflicto de nombres.** El paquete `app/evaluation/` (código) convive con la carpeta `evaluation/` de la raíz (evidencia). Son rutas distintas; se documenta. Otro nombre posible para el paquete sería `app/research/`. **[Brian decide]**

### 4.1 Datos: estructura en `DATA_ROOT` (v2)

La v2 sustituye al layout F1, que todavía no se ha usado porque `DATA_ROOT` no está definido en el `.env` local.

```
DATA_ROOT/
  processed/<request_id>/            un procesamiento de /detect (perfil y retención configurables)
      input.<ext>                    imagen recibida (opcional; ver privacidad)
      detection.json                 salida cruda (≥ conf interna) + filtrada
      spatial.json                   analyzed, free_space, decision
      narrative.json                 texto, escenario y metadatos del LLM
      audio.mp3                      si hubo TTS
      annotated.jpg                  imagen con cajas
      manifest.json
  stimuli_frozen/<dataset>/<stimulus_id>/<package_version>/   F4 (inmutable)
      detection.json  spatial.json  narrative_k{1..3}.json  audio.mp3  manifest.json
  evaluations/
      runs/<run_id>/                 una ejecución del protocolo CP3B
          experimental_config.yaml   copia + sha256
          raw/  results/  run_manifest.json
      api_tests/                     historial de /test/* (development)
  study/
      sessions/<pseudonym_id>/participant.json, responses.jsonl
      feedback/                      Likert (antes responses/feedback)
  telemetry/production_metrics.jsonl
  logs/
  dataset_finetune/                  solo development
  cache/translation_cache.json
```

- **Dentro del repositorio solo quedan datos versionados y congelados:** `stimuli/dataset1` (las 18 PNG y el manifest, necesarios para las pruebas). El GT completo sigue en el repositorio del generador, referenciado por su hash; el catálogo nunca lo sirve. El futuro GT visual (CP3B, D1) se guardaría junto al del generador, igual de protegido.
- **Retención (default = comportamiento actual):**
  - `processed/` conserva los últimos N procesamientos: `DETECTION_MAX_SAVED`=10 y `TTS_MAX_SAVED_FILES`=5 siguen significando lo mismo.
  - `input.<ext>` **no se guarda por defecto** (privacidad; hoy tampoco se guarda).
  - Los procesamientos de `stimuli_frozen/` y `evaluations/runs/` nunca se rotan.

## 5. Responsabilidades de cada módulo

| Módulo | Responsabilidad única | Perfiles |
|---|---|---|
| `core/pipeline.py` | Orquestar las etapas y medir los tiempos; devuelve un resultado estructurado. No hace HTTP, no escribe métricas y no conoce perfiles. | todos |
| `core/detector` | Inferencia YOLO + filtros (lista blanca, umbral efectivo) + política de pesos | todos |
| `core/spatial` | Columna/profundidad, pasos, espacio libre, decisión de movimiento | todos |
| `core/narrative` | Clasificación de escena, descripción, cliente Groq, traducción de etiquetas | todos |
| `core/tts` | Síntesis y codificación MP3; la persistencia se delega en `storage` | todos |
| `core/output` | Imagen anotada | todos |
| `core/config_snapshot.py` | Leer, **sin cambiar**, la configuración efectiva: pesos + hash, imgsz, iou, conf, reglas, modelo LLM y temperaturas, hash del prompt, modelo/voz TTS, hash del estilo, versiones y commit | todos |
| `api/detect.py` | Validar la subida → `pipeline.run` → TTS → respuesta con el **mismo esquema JSON**; añadir `request_id` | todos |
| `api/health.py` | Estado básico (production) y detallado (development/study) | todos |
| `telemetry.py` | Registrar métricas de producción (latencias, número de objetos) | todos |
| `security.py` / `storage.py` | Autenticación, rate limit, perfiles / rutas de datos | todos |
| `evaluation/catalog` | Catálogo único de pruebas y estímulos, sin GT | study, development |
| `evaluation/study` | Participantes, sesiones, respuestas, feedback Likert y (futuro) reproducción de estímulos congelados | study, development |
| `evaluation/devtools` | Debug, dataset/fine-tuning, banco de pruebas de la API, informes de telemetría | development |
| `evaluation/metrics`, `evaluation/runner` | Métricas CP3B y ejecución de F4 (futuro); se usan desde la CLI, **sin endpoint** | CLI |

## 6. Qué se puede mover

Todos los movimientos se hacen con `git mv` para conservar el historial. Donde otro código importa la ruta antigua, se deja un **alias por identidad de módulo**:

```python
# app/services/yolo_service.py  (alias; sin lógica)
import sys
from app.core.detector import yolo_service as _m
sys.modules[__name__] = _m
```

Así `app.services.yolo_service` y `app.core.detector.yolo_service` son **el mismo objeto**:

- un solo singleton del modelo, una sola caché y un solo `_last_tts_error`;
- `monkeypatch.setattr` y los scripts de evidencia siguen funcionando igual.

**No se usan alias `from … import *`**, porque duplicarían ese estado.

| Movimiento | Razón de responsabilidad |
|---|---|
| Pipeline de `routes/detect.py` → `core/pipeline.py` | P1: la lógica del producto no pertenece a la capa HTTP, y F4 debe llamar a la misma función que `/detect` |
| `log_metric` → `app/telemetry.py` | P2: el núcleo deja de depender de un router de evaluación |
| `routes/evaluation.py` → `evaluation/devtools/{dataset,api_tests,telemetry_report}.py` | P3: montaje por perfil |
| `/debug-detect` → `evaluation/devtools/debug.py` | Es una herramienta interna, no el producto |
| `routes/study.py`, `routes/catalog.py`, `app/catalog/` → `evaluation/study`, `evaluation/catalog` | Pertenecen a la evaluación con usuarios |
| `/feedback` → `evaluation/study` | Es una encuesta a usuarios (P4) |
| `services/*`, `utils/{groq_client,translator}` → `core/*` (con el mismo nombre de archivo) | Frontera del producto verificable por prueba. **Es el movimiento más "estético" de todos: opcional (ver §12, paso 6).** |
| `services/diagnostico_yolo.py` → `scripts/diagnostics/` | P7: es un script, no un servicio |

## 7. Qué NO se debe mover

| Elemento | Motivo |
|---|---|
| `app/main.py` y el objeto `app` | Los usan `Dockerfile` CMD, `startup.sh`, el workflow de CI y `run.py` |
| Rutas públicas (`/api/...`) y esquema JSON de `/detect` | Contrato con el cliente desplegado |
| `stimuli/dataset1/` | Hashes y rutas congelados; `import_dataset1.py --check` y las pruebas dependen de ellos |
| `evaluation/images`, `evaluation/metadata`, `evaluation/results` | Evidencia citada en los informes de las fases y en la tesis, con rutas relativas en los README |
| `scripts/evaluation/*` | Reproducen evidencia ya reportada; siguen funcionando gracias a los alias |
| `app/experimental/` | Evidencia de la selección de modelo (Cap. 3); no está montado y está excluido de Docker |
| `yolo26s.pt` y `test_images/` | `YOLO_WEIGHTS` es relativo; `/test/*` y los scripts usan esas rutas |
| Nombres de las variables de entorno y **sus valores** | Configuración experimental (CP3B) |
| Contenido de las funciones del núcleo | Solo se mueven; **no se toca la lógica** (se comprueba con golden master) |
| El generador de escenas | Repositorio aparte y congelado |

## 8. Impacto en endpoints

### 8.1 Matriz de rutas por perfil (propuesta)

| Ruta | production | study | development |
|---|---|---|---|
| `POST /api/detect` | ✔ | ✔ (con clave) | ✔ |
| `GET /api/health` | ✔ básico | ✔ detallado | ✔ detallado |
| `GET /api/tts/models` | **[decidir]**: el panel de ajustes del cliente la usa | ✔ | ✔ |
| `GET /api/catalog`, `/api/catalog/stimuli/{id}/image` | ✘ | ✔ | ✔ |
| `/api/study/sessions*` | ✘ | ✔ | ✔ |
| `POST/GET /api/feedback` | ✘ | **[decidir]**: ¿forma parte del estudio? | ✔ |
| `POST /api/debug-detect` | ✘ | ✘ | ✔ |
| `/api/dataset/*`, `/api/finetune/*` | ✘ | ✘ | ✔ |
| `/api/test/*` | ✘ | ✘ | ✔ |
| `/api/metrics/summary`, `/latency` | ✘ | ✘ | ✔ |
| `GET /api/metrics` (en memoria) | ✘ | ✘ | ✔ **[decidir: eliminar; es código muerto, P4]** |
| `/detections/*` (estático) | **[decidir]** | ✘ | ✔ |
| `/docs`, `/openapi.json` | **[decidir: desactivar]** | ✔ | ✔ |

**Sobre `/detections/*`.** El cliente muestra la imagen anotada desde `data_uri`, que va dentro del JSON; `url` solo se usa en un enlace secundario.

### 8.2 Cambios en las respuestas

- **Aditivos, sin romper nada:**
  - `request_id` en el JSON de `/detect`;
  - cabecera `X-Request-ID`, añadida a `expose_headers`.
- **`/api/health` en production:** solo `status`, `version`, `perfil` y la disponibilidad de YOLO, LLM y TTS. Sin rutas, sin datos de dataset y sin `ultimo_error`.
- **Mismo `/api/health` en development:** la respuesta actual, igual que hoy.
- **Ruta `/health`:** el enunciado menciona `/health`. Hoy la ruta es `/api/health`, y el cliente y Azure usan esa. Se propone **mantener `/api/health`**; un alias `/health` sin prefijo es opcional. **[decidir]**

### 8.3 Autenticación en production **[decidir]**

`/detect` es hoy el endpoint público de la demo (Vercel). Una clave dentro del navegador no es un secreto. Hay dos opciones:

- **(a)** `/detect` público con un rate limit **por IP** (hoy solo existe por clave);
- **(b)** clave obligatoria, aceptando que el cliente público deja de funcionar sin ella.

**Recomendación: (a).**

### 8.4 Cliente

En production y en study, las pestañas de dataset, pruebas, métricas, debug y feedback recibirían 404. Propuesta, dentro de F5: el cliente lee `perfil` en `/api/health` y oculta las pestañas que no están disponibles.

## 9. Impacto en tests

| Aspecto | Impacto |
|---|---|
| 64 pruebas actuales | Siguen pasando **sin cambios** gracias a los alias. Después se migran sus imports a las rutas nuevas, una por una. |
| **Nuevas pruebas previas a cualquier movimiento** (red de seguridad) | Ver la lista siguiente |

**Red de seguridad, antes de mover nada:**

1. **Golden master de las etapas deterministas.** Con las detecciones ya registradas en `evaluation/results/phase2a/detection/raw_predictions.jsonl` (41 imágenes, que no son del Dataset 1), se guardan las salidas de:
   - `analyze_spatial`, `estimate_steps`, `calculate_free_space` y `decide_movement`;
   - el prompt exacto que construye `llm_enhancer`, con Groq simulado.

   Tras cada paso, la salida debe ser idéntica byte a byte. No hace falta ejecutar YOLO, el LLM ni el TTS.
2. **Contrato de `/detect`:** una instantánea del **esquema** de la respuesta (claves y tipos) con el pipeline simulado.
3. **Rutas por perfil:** una instantánea del conjunto de rutas de cada perfil.
4. **Arquitectura:** `app/core` no importa FastAPI, `app.api`, `app.evaluation` ni `app.routes` (se comprueba recorriendo el AST).
5. **Alias:** `app.services.X is app.core.….X` para cada alias.

**CI:** se añade un paso `pytest` al workflow (P11), antes del build de Docker.

## 10. Impacto en Docker

- **Sin cambios en:** `CMD` (`app.main:app`), la imagen base, las dependencias ni `COPY . .`.
- **`.dockerignore`:** ya excluye `scripts/`, `tests/`, `evaluation/` y `docs/`. Se añadiría `app/evaluation/runner/`, que es solo para CLI; `catalog` y `study` deben quedarse porque el perfil study los necesita.
- **Perfil por defecto de la imagen (P9) — [decidir, CRÍTICO]:** hoy el despliegue arranca en `development` y expone todo. Propuesta: `ENV APP_PROFILE=production` en el Dockerfile, sobrescribible.
  - **Consecuencia:** el cliente de Vercel apuntando a Azure perdería las pestañas internas. Es lo deseable, pero es un cambio de comportamiento del despliegue.
- **Una sola imagen para todos los perfiles.** Los estímulos (unos 0,5 MB) y el catálogo van dentro aunque production no los monte. No compensa mantener dos imágenes.

## 11. Impacto en la tesis

La separación permite escribir, sin ambigüedad:

> **Producto (Cap. 4).** "El producto desarrollado es una API REST que recibe una imagen de un entorno Web 3D y devuelve una descripción narrativa egocéntrica en texto y audio. Su núcleo (`app/core`) implementa el pipeline detección → análisis espacial → narrativa → síntesis de voz, y se expone mediante `POST /api/detect` y `GET /api/health` (perfil de producción)."

> **Evaluación (Cap. 5).** "Para evaluar el producto se implementó una infraestructura experimental separada (`app/evaluation`, scripts y el generador de escenas): catálogo de pruebas, estímulos congelados con verificación de hash, ejecución reproducible del protocolo CP3B y registro de sesiones con participantes. Esta infraestructura no forma parte del alcance funcional del producto y no se despliega en el perfil de producción."

Consecuencias en el documento:

- **Tabla 15:** se divide en *endpoints del producto* y *endpoints de evaluación y desarrollo por perfil* (matriz del §8.1). `/run-batch` desaparece de la tabla.
- **Figura de arquitectura del Cap. 4:** muestra solo el núcleo. La infraestructura de evaluación va en el Cap. 5.
- **Defensa ante el jurado:**
  - el sistema evaluado es **el mismo código** que se despliega, porque el runner llama a `core.pipeline.run`;
  - las herramientas internas no están expuestas en producción.
- **No debe presentarse como alcance del producto:** fine-tuning, dataset, pruebas de carga, catálogo, sesiones ni el generador.
- **Hallazgo P12:** revisar cómo se cita la fase 10 (regla `min` frente a `max`).

## 12. Plan de migración por pasos

Cada paso cumple lo siguiente:

- un commit;
- las 64 pruebas actuales y la red de seguridad pasan en verde;
- **no cambia el comportamiento**;
- se puede revertir con `git revert`.

| Paso | Contenido | Riesgo | ¿Antes de F4? |
|---|---|---|---|
| 0 | Red de seguridad (§9: golden master, contrato, rutas, arquitectura) + pytest en CI | nulo | **sí** |
| 1 | Extraer `core/pipeline.py` (y `resize_image`, `build_narrative`) de `routes/detect.py`; `routes/detect.py` los reexporta | bajo | **sí** |
| 2 | `app/telemetry.py` (P2); `/detect` deja de importar el router de evaluación | bajo | **sí** |
| 3 | `api/health.py` + dividir `routes/evaluation.py` en `evaluation/devtools/*`; `debug` a devtools; `routes/evaluation.py` queda como alias | bajo | **sí** |
| 4 | Mover `study`, `catalog` y `feedback` a `app/evaluation/`, con alias | bajo | **sí** |
| 5 | Perfil `production` y matriz de rutas; `/health` básico; decisiones del §8 | medio (despliegue) | **sí** |
| 6 | *(Opcional)* `services/*` y `utils/*` → `core/*` con alias por identidad; script de diagnóstico a `scripts/` | medio | sí, o nunca: **no a mitad de la evaluación** |
| 7 | `storage` v2 + `request_id` + `manifest.json` + `config_snapshot` (aditivo) | medio | **sí** (F4 necesita los manifests) |
| 8 | Cliente: ocultar pestañas según el perfil (F5) | bajo | no |
| 9 | Documentación (arquitectura, REPRODUCIBILIDAD, tabla de rutas) | nulo | sí |

**Restricción temporal clave.** El CP3B congelará el commit de la app y los hashes del código del núcleo. Toda la refactorización debe **terminar antes de congelar** (antes de F4) o **esperar a que termine toda la evaluación**. **Nunca debe hacerse a mitad del experimento.**

## 13. Riesgos

### CRÍTICO

- **Cambiar la semántica del pipeline al extraerlo:** orden de etapas, `ThreadPoolExecutor`, `resize_image`, redondeos. Rompería la equivalencia con lo que se congele. *Mitigación:* golden master byte a byte (paso 0) y ningún cambio de lógica.
- **Alias con estado duplicado** (dos singletons YOLO, dos cachés, `monkeypatch` sin efecto). *Mitigación:* alias por `sys.modules` y prueba de identidad.
- **Perfil por defecto del despliegue** (P9). Si se cambia sin avisar, el despliegue pierde endpoints; si no se cambia, sigue expuesto. *Mitigación:* decisión explícita + variable en Azure.
- **Refactorizar después de congelar F4:** invalidaría los hashes del experimento. *Mitigación:* la restricción temporal del §12.

### IMPORTANTE

- **P10, traductor:** una caché en disco con prioridad sobre el diccionario estático introduce dependencia del estado de la máquina. *Propuesta (decisión aparte):* en la evaluación, usar una `TRANSLATION_CACHE_PATH` limpia y registrar el hash de la caché en el manifest.
- **`input.<ext>` en `processed/`:** guardar las imágenes de los usuarios afecta a su privacidad. Por defecto no se guardan.
- **Rate limit por IP** si `/detect` es público en production (hoy no existe).
- **`GET /api/metrics` es código muerto** y `/feedback` está mal ubicado: eliminar o mover afecta a pestañas del cliente.
- **P12:** evidencia de la fase 10 con la regla `min`.
- **CI sin pruebas** hasta el paso 0.

### MENOR

- Nombre `app/evaluation` frente a la carpeta `evaluation/`.
- Scripts que importan `app.routes.detect.resize_image` (cubiertos por el alias).
- Etiquetas de OpenAPI.
- Duplicado de `diagnostico_yolo.py`: eliminar uno de los dos requiere tu aprobación.
- Referencias de ruta en documentos de auditoría antiguos (los nombres de archivo se conservan).

### Decisiones pendientes de Brian

1. Nombre del paquete (`app/evaluation` o `app/research`).
2. `/api/tts/models` en production.
3. `/api/feedback`: ¿estudio o solo development?
4. `/detections` en production.
5. `/docs` en production.
6. Alias `/health`.
7. Autenticación de `/detect` en production.
8. `ENV APP_PROFILE=production` en Docker.
9. Eliminar `GET /api/metrics`.
10. Hacer o no el paso 6 (mover `services/` a `core/`).
11. Traductor en evaluación (P10).
12. Revisión de la fase 10 (P12).
