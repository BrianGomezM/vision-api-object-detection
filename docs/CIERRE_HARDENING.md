# Cierre del hardening: evidencias

**Fecha:** 2026-09-27. **Base:** backend `022b325` (código) + commits de evidencia; cliente `c69ca54`.

**Qué no se hizo:** no se desplegó, no se ejecutó F4 y el Dataset 1 no se modificó.

## 1. EXIF: decisión técnica

**La orientación EXIF no se aplica y se mantiene así.**

- **No afecta al Dataset 1.** Las 18 PNG son RGB 800×450, sin EXIF ni metadatos (`info` vacío), verificado al cerrar el hardening.
- **Afecta a fotografías de móvil (JPEG con `Orientation` ≠ 1).** El detector ve los píxeles almacenados, no la imagen girada. Es un comportamiento documentado del producto (`tests/test_limits_and_security.py::test_exif_orientacion_no_se_aplica`).
- **Motivo para no cambiarlo:**
  - aplicarlo cambiaría el preprocesamiento congelado (`experimental_config.yaml` → `imagen`);
  - el caso de uso del producto son capturas de entornos Web 3D (PNG o JPEG sin orientación).

## 2. Concurrencia con worker único: medición y decisión

- **Configuración medida** (`scripts/hardening/concurrency_bench.py` → `evaluation/results/hardening/concurrencia.json`):
  - servidor real (uvicorn, 1 worker) con YOLO real en CPU;
  - LLM y TTS simulados con las latencias medidas en los smoke tests reales: LLM 0,5 s, TTS 16,5 s.

| Solicitudes simultáneas | Tiempo total | Latencia mín / media / máx | Errores | `/api/health` máx | RSS máx |
|---|---|---|---|---|---|
| 1 | 18,3 s | 18,3 / 18,3 / 18,3 s | 0 | 17,7 s | 739 MB |
| 2 | 36,1 s | 18,1 / 27,1 / 36,1 s | 0 | 17,6 s | 780 MB |
| 3 | 53,9 s | 17,9 / 35,9 / 53,9 s | 0 | 17,6 s | 797 MB |
| 5 | 89,9 s | 18,0 / 61,1 / 89,8 s | 0 | 35,5 s | 827 MB |
| 8, production (límite 6/60 s) | 26,8 s | 6 × 200; 2 × 429 en **104 ms** | 2 × 429 (esperados) | 8,2 s | 863 MB |

- **RSS en reposo:** 726 MB.
- **Estado compartido:** la misma imagen produce la misma narrativa en todos los escenarios, y los `request_id` son únicos.
- **Interpretación:**
  - las solicitudes se **serializan**: la n-ésima espera unos n × 18 s. No hay errores ni mezcla de estado;
  - el límite por IP responde en unos 100 ms aunque el worker esté ocupado;
  - la memoria crece poco con la cola (+100 MB con 5).
- **Hallazgo:** mientras se procesa una detección, **`/api/health` también espera** (hasta 35 s): el handler `async` ejecuta código bloqueante.
- **Decisión:** el worker único es **aceptable para el alcance actual** (demo pública con límite por IP y estudio con un investigador que lanza una solicitud a la vez). Motivos:
  - es seguro frente al estado compartido;
  - el plan de Azure (1,75 GB) no admite dos workers de unos 750 MB cada uno con margen.
- **Resuelto (2026-09-27, commit 3799b23):** el pipeline se ejecuta en un único hilo dedicado (`ThreadPoolExecutor(max_workers=1)`) y bajo un candado, así que se mantiene la serialización y `/api/health` queda libre.
  - Medido: `/api/health` responde en ≤ 139 ms durante 1, 2, 3 y 5 detecciones simultáneas (`concurrencia.json`).
  - Pruebas: `tests/test_health_during_detect.py`.
  - En Docker con 1 detección: ≤ 76 ms (`docker/docker_<commit>.json`).

## 3. Tabla de requisitos y evidencias

| ID | Requisito | Evidencia | Estado | Impacto |
|---|---|---|---|---|
| H1 | Política HTTP formal: sin fallos funcionales ocultos tras un 200 | `tests/test_policy_matrix.py` (27 casos); `CONTRATO_ERRORES.md` §8.1 | PASS | Contrato claro para el cliente y la tesis |
| H2 | Limpieza de archivos en éxito y en cualquier excepción | `tests/test_cleanup.py` (39; con la limpieza anulada fallan 23) | PASS | Production no conserva derivados del usuario |
| H3 | Límite por IP en production con 429, sin confiar en XFF | `tests/test_ratelimit_and_request_id.py` (15); UI: 429 real con `Retry-After: 57` | PASS (local) | El proxy de Azure no se ha verificado (§4) |
| H4 | `Retry-After` solo con un valor real; categorías de proveedor | `test_LMN_tts` (7 casos), `test_llm_limite_y_no_disponible` | PASS | Se elimina el `Retry-After: 60` inventado |
| H5 | `request_id` generado por el servidor, en respuesta, log y errores | `test_request_id_lo_genera_el_servidor` (5), `test_request_id_tambien_en_404_405_y_429` | PASS | Trazabilidad |
| H6 | Caché: A1–A9 no reutilizan la narrativa; documentada | `tests/test_scene_cache.py` (4, incluido HTTP); `CONTRATO_ERRORES.md` §5 | PASS | Requisito de F4 |
| H7 | Smoke tests con proveedores reales | `evaluation/results/hardening/live_smoke.jsonl` (4/4) | PASS | YOLO, Groq y Gemini operativos |
| H8 | Cliente: 200/400/413/415/422/429/500/502/503/504/timeout/sin audio/study/degradación/sin voz del navegador | `evaluation/results/hardening/ui/` (16/16 + capturas) | PASS | Interfaz verificada en navegador real |
| H9 | E2E real local (cliente → API → YOLO → LLM → TTS → reproducción) y fallos reales de LLM/TTS | Escenarios de UI "E2E real" (éxito: audio de 27,7 s) y "claves inválidas" | PASS | Flujo completo verificado |
| H10 | Concurrencia con worker único medida | `concurrencia.json` | PASS | Health bloqueado: resuelto (§2) |
| H11 | EXIF analizado | §1 | PASS (decisión documentada) | Sin impacto en el Dataset 1 |
| H12 | Regresión del pipeline idéntica | `tests/regression` (41/41) | PASS | Núcleo sin cambios de salida |
| H13 | Configuración congelada coherente | preflight en commit limpio; diff: solo hashes del código | PASS | Parámetros congelados intactos |
| H14 | Proxy real (XFF / hops), CORS con la URL real, timeouts del proxy, logs en la plataforma | — | **BLOCKED** (requiere despliegue) | Verificar en la fase de deployment |

**Defectos reales encontrados y corregidos en el cierre:**

- archivos residuales cuando fallaba el pipeline;
- `Retry-After: 60` inventado;
- faltaba `X-Degradacion` en la respuesta MP3 de `audio=true`;
- carrera en la rotación de archivos con solicitudes simultáneas;
- el cliente no tenía timeout: esperaba indefinidamente.

## 4. Lo que requiere el entorno desplegado

- **Valor de `TRUSTED_PROXY_HOPS`** en Azure: confirmar que el frontend añade exactamente una entrada a `X-Forwarded-For`.
- **CORS** con la URL real de Vercel.
- **Timeouts:** el del proxy de Azure (~230 s) frente al cliente (180 s) y a gunicorn (600 s).
- **Logs:** recolección de los logs JSON (Log Stream).
- **Health check:** activarlo en `/api/health` (ya no se bloquea, §2).
- **Arranque:** medido localmente con la imagen final (Python 3.13, torch CPU, pesos incluidos y verificados): 12–15 s con 1 vCPU. Queda pendiente la descarga de la imagen en Azure. Ver `docs/DEPLOYMENT.md`.
