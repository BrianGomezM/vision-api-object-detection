# Clasificación de las pruebas

**Fecha:** 2026-09-27. **Commit:** `63a3917`. Resultados obtenidos con informes JUnit, archivo por archivo.

## 1. Qué es y qué no es "311"

- `pytest` recoge **315 pruebas** en 24 archivos.
- En el entorno local de referencia (Windows, Python 3.13.15) **pasan 311** y **se omiten 4**: los smoke reales, que solo se ejecutan con `LIVE_SMOKE_TEST=1`.
- **Ninguna de las 311 ejecuta YOLO, el LLM ni el TTS reales.** El modelo, Groq y Gemini se sustituyen por dobles simulados (`tests/conftest.py`, fixture `sim`). Verifican el **comportamiento del software**: contrato de errores, límites, seguridad, limpieza, concurrencia, políticas, identidad y configuración. **No miden la calidad de las detecciones ni de la narrativa.**
- La evidencia sobre inferencia real está en otro lugar (§3). No debe sumarse a 311.

Cifra recomendada para la tesis:

> 311 pruebas automatizadas de software (proveedores simulados; 315 recogidas, 4 omitidas por requerir servicios reales), más 4 smoke tests con servicios reales ejecutados aparte (4/4) y la verificación de inferencia descrita en §3.

## 2. Suites

Columnas:
- **Local:** Windows, entorno de referencia.
- **Docker:** CI reproducido dentro de la imagen de producción.
- **CI:** GitHub Actions, `-m "not entorno_referencia and not live_smoke"`.
- **Resultado:** local / Docker-CI.

| Suite (archivo) | N | Tipo | Servicios externos | Local | Docker | CI | Smoke real | Resultado |
|---|---|---|---|---|---|---|---|---|
| `live/test_live_smoke.py` | 4 | Smoke real (YOLO, Groq, Gemini TTS, `/api/detect`) | **Sí** | Manual (`run_live_smoke.py`) | No | No | **Sí** | 4/4 PASS en `63a3917` (omitidas en la corrida normal) |
| `regression/test_pipeline_regression.py` | 2 | Regresión byte a byte (41 imágenes, cajas de YOLO registradas) | No | Sí | Sí (manual) | No (`entorno_referencia`) | No | 2/2 · 2/2 |
| `regression/test_contracts.py` | 1 | Contrato de rutas y entradas por perfil | No | Sí | Sí | Sí | No | 1/1 · 1/1 |
| `test_error_contract.py` | 41 | Contrato de errores (integración API) | No | Sí | Sí | Sí | No | 41 · 41 |
| `test_limits_and_security.py` | 43 | Seguridad y límites | No | Sí | Sí | Sí | No | 43 · 43 |
| `test_cleanup.py` | 39 | Limpieza de archivos temporales | No | Sí | Sí | Sí | No | 39 · 39 |
| `test_policy_matrix.py` | 27 | Matriz de políticas (perfil × degradación) | No | Sí | Sí | Sí | No | 27 · 27 |
| `test_security_profiles.py` | 24 | Seguridad por perfil y autenticación | No | Sí | Sí | Sí | No | 24 · 24 |
| `test_deploy_identity.py` | 16 | Identidad del despliegue (1 de `entorno_referencia`) | No | Sí | 15 | 15 | No | 16 · 15 |
| `test_ratelimit_and_request_id.py` | 15 | Límite por IP y `request_id` | No | Sí | Sí | Sí | No | 15 · 15 |
| `test_catalog.py` | 14 | Catálogo y hashes de estímulos | No | Sí | Sí | Sí | No | 14 · 14 |
| `test_ssrf_uploads.py` | 14 | Seguridad de subidas (SSRF, tamaño) | No | Sí | Sí | Sí | No | 14 · 14 |
| `test_experiment.py` | 13 | Configuración congelada y preflight de F4 (2 de `entorno_referencia`) | No | Sí | 11 | 11 | No | 13 · 11 |
| `test_storage.py` | 10 | Almacenamiento y rotación | No | Sí | Sí | Sí | No | 10 · 10 |
| `test_e2e_simulated.py` | 9 | E2E de la API con proveedores simulados | No | Sí | Sí | Sí | No | 9 · 9 |
| `test_core_pipeline.py` | 8 | Núcleo del pipeline (unidad/integración) | No | Sí | Sí | Sí | No | 8 · 8 |
| `test_yolo_weights.py` | 7 | Política y hash de pesos (1 requiere `yolo26s.pt`) | No | Sí | 6 + 1 omitida | 6 + 1 omitida | No | 7 · 6 |
| `test_health_during_detect.py` | 6 | Health durante detecciones y serialización | No | Sí | Sí | Sí | No | 6 · 6 |
| `test_import_dataset1.py` | 4 | Integridad del Dataset 1 frente al generador (requiere el repositorio hermano) | No | Sí | Omitidas | Omitidas | No | 4 · 0 (4 omitidas) |
| `test_scene_cache.py` | 4 | Caché de escenario | No | Sí | Sí | Sí | No | 4 · 4 |
| `test_translator.py` | 4 | Traducción (unidad) | No | Sí | Sí | Sí | No | 4 · 4 |
| `test_yolo_rule.py` | 4 | Regla del umbral (unidad) | No | Sí | Sí | Sí | No | 4 · 4 |
| `test_architecture.py` | 3 | Arquitectura (imports y perfiles) | No | Sí | Sí | Sí | No | 3 · 3 |
| `test_study_sessions.py` | 3 | Sesiones del estudio | No | Sí | Sí | Sí | No | 3 · 3 |

**Totales:**

| Entorno | Recogidas | Pasan | Omitidas | Fallan |
|---|---|---|---|---|
| Local | 315 | **311** | 4 | 0 |
| Docker / CI | 306 (se excluyen las 5 de `entorno_referencia` y las 4 live) | **301** | 5 | 0 |

Las 5 omitidas en Docker/CI se deben a que faltan el repositorio del generador (4) y el archivo `.pt` en el checkout (1).

## 3. Evidencia de inferencia real (fuera de las 311)

| Evidencia | Qué ejecuta | Resultado |
|---|---|---|
| `evaluation/results/hardening/live_smoke_63a3917.json` | YOLO + Groq + Gemini reales, commit limpio | 4/4 PASS |
| `evaluation/results/hardening/docker/smoke_real_63a3917.json` | `/api/detect` real en la imagen de producción (1 vCPU / 1,75 GB) | 200, sin degradaciones |
| `evaluation/results/hardening/torch/*.json` | YOLO real en 41 imágenes frente a las cajas de la fase 2A, en 3 entornos | §4 de `REPRODUCIBILIDAD.md` |
| `evaluation/results/hardening/docker/docker_63a3917.json` | Imagen: pesos, fallos controlados, endpoint con YOLO real, CORS, límites | 15/15 |
