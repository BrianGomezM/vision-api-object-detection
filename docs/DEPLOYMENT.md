# Plan de despliegue reproducible — Vision API (backend) + cliente (Vercel)

**Fecha:** 2026-09-27. **Estado:** PLAN. **No se ha ejecutado ningún despliegue.** Todo lo marcado *(medido)* sale de evidencia local sobre la imagen Docker final; lo marcado *(pendiente)* solo puede comprobarse en la nube.

> Restricciones vigentes: no desplegar sin aprobación; no ejecutar F4; no iniciar el estudio con usuarios; no modificar el Dataset 1; no generar resultados experimentales oficiales.

Evidencia citada (en `evaluation/results/hardening/`):

| Archivo | Contenido |
|---|---|
| `docker/docker_<commit>.json` | Pruebas de la imagen: pesos, fallos controlados, endpoint, CORS, límites, limpieza y logs |
| `docker/limites_recursos_<commit>.json` | La imagen bajo los límites de CPU y memoria de cada plan candidato |
| `live_smoke_<commit>.json` | Smoke tests reales (YOLO, Groq, Gemini TTS, endpoint) sobre commit limpio |
| `concurrencia.json` | Serialización, `/api/health` durante detecciones y memoria bajo concurrencia |

---

## A. Perfil del backend (lo que decide la plataforma)

| Aspecto | Valor | Fuente |
|---|---|---|
| Stack | FastAPI + gunicorn/uvicorn, **1 worker**, pipeline serializado (1 hilo + lock); `/api/health` responde durante una detección | `Dockerfile`, `app/routes/detect.py` |
| Imagen | `python:3.13-slim`, torch **2.13.0+cpu**, ultralytics 8.4.123, ~530 MB | *(medido)* |
| Pesos | `yolo26s.pt` **dentro de la imagen**, SHA-256 `646f8bc3…4a1b`, verificado en el build y en el arranque | §C |
| Memoria | ~490–560 MiB en reposo; meseta de ~827 MiB tras 10 detecciones seguidas, **sin crecimiento hasta 40** (sin fugas); **pico máximo observado 918 MiB** | *(medido)* `limites_recursos`, prueba de 40 solicitudes |
| Arranque | 4–5 s con 8 hilos; **13–14 s con 1 vCPU**, ~9 s con 2 vCPU (carga de YOLO + warm-up) | *(medido)* |
| Inferencia YOLO | ~0,5–1,4 s con 8 hilos; **3,1–4,9 s con 1 vCPU**; 1,8–2,6 s con 2 vCPU | *(medido)* |
| LLM (Groq) | ~0,4–0,5 s por llamada (2 llamadas por solicitud); timeout de 15 s con 2 reintentos | `live_smoke` |
| TTS (Gemini) | ~2 s para una frase; ~16 s para una narrativa completa; timeout de 60 s | `live_smoke` |
| Solicitud completa | ~19 s en local con proveedores reales; en 1 vCPU, ~+4 s | *(medido / estimado)* |
| Almacenamiento | Solo temporal (`DATA_ROOT=/tmp/visionnav`). La imagen anotada y el audio se borran tras responder; queda la telemetría sin imagen ni texto | `docker_<commit>.json` |
| Salida a Internet | Groq, Gemini y el traductor (deep-translator). **Ninguna descarga de pesos** | — |

La **memoria** manda en la elección: todo plan de 512 MB queda descartado por evidencia. Con 0,5 CPU y 512 MB el worker muere por OOM durante el warm-up y gunicorn lo reinicia en bucle; el contenedor nunca queda sano.

---

## B. Azure App Service vs Render (para ESTE backend)

Especificaciones y precios consultados el 2026-09-27; **confirmar en la calculadora oficial antes de contratar**:

- **Azure (Linux):** B1 = 1 núcleo, 1,75 GB y 10 GB; B2 = 2 núcleos y 3,5 GB. La página oficial no mostró precios. Las fuentes secundarias dan ≈ 0,018 USD/h para B1 (≈ 13 USD/mes).
- **Render:** Starter = 0,5 CPU y 512 MB por 7 USD/mes; Standard = 1 CPU y 2 GB por 25 USD/mes. El plan Free se suspende tras 15 min sin tráfico (≈ 1 min para despertar) y da 750 h/mes por workspace.

| Criterio | Azure App Service B1 (Linux, contenedor) | Render |
|---|---|---|
| FastAPI / Python / Docker | Contenedor desde GHCR (ya configurado). `WEBSITES_PORT=8000` | Contenedor desde Dockerfile o registro. Usa `$PORT` (el `CMD` ya lo respeta) |
| PyTorch CPU + YOLO: memoria | **1,75 GB: margen ≈ 1,9× sobre el pico máximo medido (918 MiB)** | Starter 512 MB: **no arranca** (14 reinicios por OOM en 150 s, *medido*). Standard 2 GB: sí, a 25 USD/mes. Free: RAM no documentada en la página consultada y se suspende, no apto |
| CPU / inferencia | 1 núcleo: YOLO ≈ 3,1–4,9 s *(medido con `--cpus=1`)* | Standard 1 CPU: igual. Starter 0,5 CPU: no aplica (OOM) |
| Arranque | 12–15 s de la app (1 vCPU) + descarga de ~530 MB de imagen en cada nuevo host. **Always On** (disponible en Basic) evita descargas por inactividad | Similar en planes de pago. Free: ~1 min tras cada suspensión |
| **Timeout HTTP** | **Balanceador: ~230–240 s, NO configurable.** El cliente corta a 180 s, antes que la plataforma | Máx. 100 min (sin riesgo) |
| LLM / TTS externos | Salida a Internet sin restricción; latencia similar | Igual |
| Variables / secretos | App settings (cifrados en reposo); Key Vault opcional | Environment / Secret files |
| CORS | En la app (`CORS_ORIGINS` / `CORS_ORIGIN_REGEX`); **no activar el CORS de la plataforma**, que duplicaría cabeceras | En la app |
| Almacenamiento temporal | `/tmp` del contenedor, efímero (correcto: no se persiste nada del usuario) | Disco efímero, igual |
| Logs | Log stream / `az webapp log tail`; Log Analytics opcional | Logs en el panel, con retención limitada según el plan |
| Rate limit por IP | Detrás del front-end de Azure: `TRUSTED_PROXY_HOPS` debe ajustarse *(pendiente de verificar)* | Detrás del proxy de Render: igual *(pendiente de verificar)* |
| Health check | Health check de App Service sobre `/api/health`. El HEALTHCHECK del Dockerfile no lo usa Azure | Health check path configurable |
| CI/CD existente | **Ya existe:** GitHub Actions → GHCR → `azure/webapps-deploy` (OIDC). App `visionnav-api`, rg `rg-visionnav`, plan B1 | Habría que crearlo |
| Coste | ≈ 13 USD/mes estimado (verificar; posible crédito académico de la suscripción) | Standard: 25 USD/mes para una RAM equivalente |
| Estabilidad | SLA en planes de pago; reinicios de plataforma ocasionales | Planes de pago estables; Free se suspende |
| Reproducibilidad | Despliegue por **imagen inmutable `:<sha>`**. Rollback = volver a apuntar al tag anterior | Igual si se despliega por imagen; si Render construye desde el repo, el build ocurre en su infraestructura |
| Tesis | La documentación y el memory del proyecto declaran Azure | Requeriría cambiar la documentación |

**Recomendación: Azure App Service B1 (Linux, contenedor), con Always On y Health check en `/api/health`.**

Por qué:

1. Es el único plan barato que cabe con margen: 1,75 GB frente a un pico medido de ~0,8 GB. En Render, el equivalente en RAM es Standard, que cuesta el doble.
2. La infraestructura, el pipeline OIDC y la documentación de la tesis ya son Azure; cambiar de plataforma añade riesgo sin beneficio medible.
3. El despliegue por imagen `:<sha>` más la verificación de identidad al arrancar (§C) garantizan que lo desplegado es lo evaluado.

**Riesgo propio de Azure:** el timeout no configurable de ~240 s. Con el pipeline serializado (~20–25 s por solicitud en 1 vCPU), una cola de más de 7–9 solicitudes simultáneas superaría los 180 s del cliente. Queda acotado por el límite por IP (6/60 s) y por el uso previsto, que es un solo participante a la vez en el estudio. Si hiciera falta más capacidad, pasar a B2 (2 vCPU: YOLO ~1,8–2,6 s) sin cambiar nada más.

**Render** queda como alternativa documentada solo con **Standard (2 GB)**. Starter y Free quedan descartados por evidencia.

---

## C. Imagen Docker: estrategia de pesos

**Decisión: Opción 1, pesos incluidos en la imagen y verificados.**

- **Build:** `ADD --checksum=sha256:646f8bc3…` desde la release **fija** `ultralytics/assets v8.4.0` (nunca "latest"). Si el archivo remoto cambia, **el build falla** (`digest mismatch`, *medido*). Su hash coincide byte a byte con el congelado en `experimental_config.yaml`.
- **Arranque:** `YOLO_WEIGHTS_SHA256` + `YOLO_ALLOW_DOWNLOAD=false`. Si el archivo no coincide o no existe, **el worker no arranca** (salida 3, mensaje `RuntimeError: [YOLO]…` o `[Identidad]…`) y nunca se descarga nada.
- **Identidad completa** (`app/deploy_identity.py`): en production, además, se compara el entorno efectivo con `experimental_config.yaml`: pesos, código del núcleo, parámetros, modelos LLM/TTS, voz, estilo y versiones. Si hay diferencias, no arranca. Tolera solo:
  - la ruta de los pesos (se exige el SHA-256);
  - el sufijo `+cpu` frente a `+cu126` de torch y torchvision (misma versión, dispositivo congelado en CPU).

  `/api/health` publica `identidad.estado`, el `commit` y las diferencias toleradas.
- **Por qué no la Opción 2** (descargar al arrancar):
  - añade una dependencia de red en cada arranque o escalado;
  - hace el arranque más lento;
  - abre una ventana en la que la app existe sin pesos;
  - dos instancias podrían no ser idénticas.

  La Opción 1 verifica una sola vez en el build y produce una imagen inmutable.
- **Otras propiedades de la imagen** *(medido)*:
  - usuario sin privilegios (uid 10001);
  - pesos `0444` de solo lectura; `/app` no escribible;
  - `.dockerignore` excluye `*.pt` locales, `.env` y datos;
  - dependencias fijadas en `requirements-docker.lock.txt` y torch en el `Dockerfile`;
  - `GROQ_MODEL` fijado al congelado, porque el valor por defecto del código no lo es;
  - `HEALTHCHECK` y `PORT` configurable.

Pruebas de la imagen (`scripts/hardening/docker_evidence.py`):

| Prueba | Resultado |
|---|---|
| Build | OK |
| Arranque → health 200 | OK |
| Pesos existen, SHA-256 correcto, solo lectura, sin `.pt` locales copiados | OK |
| Hash incorrecto en el build | Build falla |
| `YOLO_WEIGHTS_SHA256` incorrecto / pesos ausentes / pesos alterados en 1 byte | No arranca (exit 3) |
| `GROQ_MODEL` o `TTS_VOICE` distintos del congelado | No arranca |
| `/api/detect` con YOLO real y sin claves | 200 + `X-Degradacion: LLM_UNAVAILABLE,TTS_UNAVAILABLE` |

---

## D. Matriz de variables de entorno

Ningún valor secreto aparece en este documento. "Congelado" significa que la verificación de identidad **impide arrancar** si se cambia.

| Variable | Obligatoria | Producción | Study | Sensibilidad | Secreto | Comportamiento si falta / es incorrecta |
|---|---|---|---|---|---|---|
| `GROQ_API_KEY` | Prod: recomendada. Study: sí | Definir (app setting) | Definir | Alta (cuota y coste) | **Sí** | Prod: 200 con `X-Degradacion: LLM_UNAVAILABLE` (narrativa por reglas). Study: error 503 |
| `GOOGLE_API_KEY` | Prod: recomendada. Study: sí | Definir | Definir | Alta (cuota ≈ 15.000 COP) | **Sí** | Prod: 200 sin audio (`TTS_UNAVAILABLE`); con `audio=true`, 503. Study: error |
| `API_KEYS` | Study: sí | **Vacía** (endpoint público, `POLITICA_API_DETECT.md`) | Definir | Alta | **Sí** | Prod: sin autenticación (previsto). Study: **no arranca** |
| `RESEARCHER_API_KEYS` | Prod: sí, para el estudio | Definir (clave del investigador; se escribe en el cliente en tiempo de ejecución, nunca en Vercel) | Opcional (se suma a `API_KEYS`) | Alta (datos de participantes) | **Sí** | Prod: `/api/study/*`, `/api/catalog*` y `/api/metrics/*` responden 401 (fallo cerrado). **No** afecta a `/api/detect` |
| `APP_PROFILE` | Sí | `production` (imagen) | `study` | Baja | No | Sin definir: `development`, que expone todos los endpoints. La imagen fija `production` |
| `APP_COMMIT` | Sí (trazabilidad) | Build-arg = SHA del commit (CI) | Igual | Baja | No | `desconocido`: se pierde la correlación entre logs y commit |
| `GROQ_MODEL` | Congelado | `qwen/qwen3.8-27b` (imagen) | Igual | Media | No | Distinto: **no arranca**. No definirlo en la plataforma |
| `TTS_MODEL` / `TTS_VOICE` / `TTS_STYLE_INSTRUCTIONS` | Congelados | Valor por defecto = congelado | Igual | Media | No | Distinto: **no arranca** |
| `YOLO_WEIGHTS` / `YOLO_WEIGHTS_SHA256` / `YOLO_ALLOW_DOWNLOAD` | Sí | Imagen: `/app/weights/yolo26s.pt`, hash congelado, `false` | Igual | Media (integridad) | No | Hash distinto o archivo ausente: **no arranca**; nunca se descargan pesos |
| `YOLO_IMGSZ`, `YOLO_IOU`, `YOLO_CONF_INTERNAL`, `YOLO_AUGMENT`, `API_DEFAULT_CONF`, `API_MAX_IMAGE_DIM`, `SPATIAL_*`, `STEP_*`, `FREE_SPACE_*`, `RISK_*`, `LLM_*`, `GROQ_TIMEOUT`, `GROQ_MAX_RETRIES`, `TTS_TIMEOUT_S`, `TTS_MAX_CHARS` | Congelados | **No definir** (valor por defecto = congelado) | Igual | Media | No | Distinto: **no arranca** |
| `DATA_ROOT` | Sí | `/tmp/visionnav` (imagen, efímero). **Con sesiones de estudio en production: `/home/visionnav-data` + `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true`**, o las sesiones se pierden al reiniciar | Obligatorio (preflight) | Media (telemetría) / Alta (sesiones) | No | Prod: la imagen lo fija; sin él escribiría en `/app`, que no es escribible, y daría errores |
| `MAX_UPLOAD_MB` | No | 10 (defecto) | 10 | Baja | No | Defecto 10 MB → 413 por encima |
| `CORS_ORIGINS` | Solo si el dominio no encaja con la regex | Dominio de producción de Vercel si es propio | — | Media | No | Solo localhost y la regex |
| `CORS_ORIGIN_REGEX` | No | **Recomendado acotar al equipo:** `^https://visionnav-client(-[a-z0-9]+)*-<equipo>\.vercel\.app$` más el dominio principal en `CORS_ORIGINS` | — | Media | No | Defecto: `visionnav-client(-…)*.vercel.app`. Cualquiera que cree un proyecto con ese prefijo en Vercel pasaría el CORS; no es autenticación, pero amplía la superficie |
| `RATE_LIMIT_IP_ENABLED` | No | Activo por defecto en production | — | Media | No | Defecto: activo en production |
| `RATE_LIMIT_IP_REQUESTS` / `RATE_LIMIT_IP_WINDOW_S` | No | 6 / 60 (defecto) | — | Media | No | Defectos |
| `TRUSTED_PROXY_HOPS` | **Sí detrás de proxy** | Valor verificado tras desplegar (esperado 1 en Azure) *(pendiente)* | — | Media | No | 0 detrás de proxy: todos los usuarios comparten la IP del proxy y el límite de 6/min se vuelve **global**. Un valor mayor que el real permite falsificar la IP con `X-Forwarded-For` |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | No | No aplica (sin claves) | Ajustar a la sesión | Baja | No | 10/60 s por clave |
| `IDENTIDAD_EXPERIMENTAL` | No | **No definir** | No definir | Alta (integridad) | No | `omitir` desactiva la verificación y queda declarado en `/api/health` |
| `EVALUATION_DISABLE_TTS` | No | **No definir** | No definir | Media | No | `true` desactiva el TTS |
| `PORT` (Render) / `WEBSITES_PORT` (Azure) | Según plataforma | Azure: `WEBSITES_PORT=8000` | — | Baja | No | Puerto incorrecto: la plataforma no alcanza la app y el health check falla |
| `WEBSITES_CONTAINER_START_TIME_LIMIT` (Azure) | No | Defecto 230 s (arranque medido 12–15 s) | — | Baja | No | — |
| Timeouts fijos | — | Cliente 180 s · Azure ~240 s · gunicorn 600 s · Groq 15 s × 3 · TTS 60 s | — | — | — | El cliente corta antes que la plataforma |
| `NEXT_PUBLIC_API_URL` (Vercel) | Sí | URL HTTPS del backend | — | Baja (pública) | **No** (va al bundle) | Sin ella: `http://127.0.0.1:8000`, y el cliente desplegado no funciona |

---

## E. Plan de despliegue (NO ejecutado)

Precondiciones:

- aprobación explícita;
- rama fusionada a `main` **solo cuando se apruebe**: un push a `main` **dispara el despliegue** (workflow `main_visionnav-api.yml`);
- la suite local en verde; `run_live_smoke.py` en PASS sobre el commit a desplegar; `docker_evidence.py` en PASS.

| # | Paso | Comando / acción | Verificación |
|---|---|---|---|
| 1 | **Build** | CI: job `test` (Python 3.13 y versiones de la imagen) → `docker/build-push-action` con `APP_COMMIT=${{ github.sha }}`. Local equivalente: `docker build --build-arg APP_COMMIT=$(git rev-parse HEAD) -t visionnav-api:<sha7> .` | Build verde; el paso `ADD --checksum` pasa |
| 2 | **Push** | CI publica `ghcr.io/briangomezm/vision-api-object-detection:<sha>` (y `:latest`, que **no** se usa para desplegar) | Tag `<sha>` presente en GHCR; anotar el digest |
| 3 | **Deploy** | CI: `azure/webapps-deploy` con `images: …:<sha>`. Manual: `az webapp config container set -g rg-visionnav -n visionnav-api --container-image-name ghcr.io/…:<sha>`; si estaba detenida, `az webapp start …` | `az webapp show … --query state` = Running; la imagen configurada es `:<sha>` |
| 4 | **Variables** | App settings: `GROQ_API_KEY`, `GOOGLE_API_KEY` (secretos), `WEBSITES_PORT=8000`, `TRUSTED_PROXY_HOPS` (tras el paso 10), `CORS_ORIGINS` / `CORS_ORIGIN_REGEX` si procede. **No** definir las congeladas, `API_KEYS`, `IDENTIDAD_EXPERIMENTAL` ni `EVALUATION_DISABLE_TTS`. Plataforma: **Always On** = on, **Health check** = `/api/health`, CORS de la plataforma **desactivado** | `az webapp config appsettings list` (solo nombres); ningún valor secreto en logs |
| 5 | **Health** | `curl -s https://<app>.azurewebsites.net/api/health` | 200; `perfil=production`; `identidad.estado=verificada`; `identidad.commit=<sha>`; `llm.modelo=qwen/qwen3.8-27b`; `llm.activo` y `tts.activo` = true |
| 6 | **Logs** | `az webapp log tail -g rg-visionnav -n visionnav-api` | `[Identidad] Despliegue verificado…`, `[YOLO] Modelo listo`, líneas JSON con `request_id` y `app_commit=<sha>`, sin claves |
| 7 | **CORS** | Preflight `OPTIONS /api/detect` con el origen de producción de Vercel y con un origen ajeno | Producción: `access-control-allow-origin` presente. Ajeno: 400 y sin cabecera |
| 8 | **Smoke** | `/api/health` + un `POST /api/detect` con `test_images/05_sala_muebles.jpg` (**nunca** estímulos del Dataset 1) | 200, **sin** `X-Degradacion`, `audio.disponible=true`, `imagen_anotada.disponible=true` |
| 9 | **/api/detect** | Tabla F (casos de éxito) | Tiempos por etapa en `metricas`, dentro de lo esperado (§A) |
| 10 | **Errores** | Tabla F (413/415/422/400/429). **Verificación de hops:** enviar 7 solicitudes con `X-Forwarded-For` falso y variable | Con `TRUSTED_PROXY_HOPS` correcto, el límite se aplica a la IP real aunque cambie el XFF falso. Si con 0 todos comparten bucket, fijar 1 y repetir |
| 11 | **Vercel** | Proyecto `visionnav-client`: `NEXT_PUBLIC_API_URL=https://<app>.azurewebsites.net` → redeploy | El bundle apunta a la URL HTTPS; el panel de ajustes muestra "configurada via NEXT_PUBLIC_API_URL" |
| 12 | **E2E navegador** | Chrome: abrir la app de Vercel → subir la imagen → ver la imagen anotada, leer la narrativa y reproducir el audio. Sin errores CORS en la consola | Captura + `request_id` de la respuesta = línea del log |
| 13 | **Rollback** | Volver a apuntar el contenedor al tag anterior: `az webapp config container set … --container-image-name ghcr.io/…:<sha_anterior>` + restart (o re-ejecutar el workflow en el commit anterior). Vercel: "Promote" del despliegue anterior. Criterio: cualquier fallo de los pasos 5–12 que no se corrija en minutos | `/api/health` → `identidad.commit=<sha_anterior>` |

Registrar en `evaluation/results/hardening/deploy_<sha7>.json`: digest de la imagen, commit, salidas de los pasos 5–12 (sin secretos), fecha y quién aprobó.

---

## F. Pruebas post-despliegue

`BASE=https://<app>.azurewebsites.net`. Imágenes de `test_images/` (nunca del Dataset 1). Cada fila se registra con su `request_id`.

| Prueba | Objetivo | Entrada | Resultado esperado | Evidencia |
|---|---|---|---|---|
| /health | Servicio vivo y con la identidad correcta | `GET /api/health` | 200; `identidad.estado=verificada`; `commit=<sha>`; LLM y TTS activos | JSON de respuesta |
| /api/detect éxito | Pipeline completo | `05_sala_muebles.jpg` | 200, objetos > 0, narrativa, `audio.disponible=true`, imagen anotada, sin `X-Degradacion` | Cabeceras + métricas por etapa |
| Imagen inválida | Validación | Archivo de texto como `file` | 422 `INVALID_IMAGE`, `stage=validacion` | Cuerpo de error + `request_id` |
| Payload grande | Límite de subida | 12 MB | 413 `PAYLOAD_TOO_LARGE` | Ídem |
| Formato no soportado | Formato por contenido | GIF (también con extensión `.jpg`) | 415 `UNSUPPORTED_IMAGE` | Ídem |
| YOLO | Detector y pesos | Health + detect | Health: SHA en identidad. Detect: objetos coherentes con la ejecución local de la misma imagen | Comparar etiquetas con `docker_<commit>.json` |
| LLM | Narrativa del modelo congelado | Detect éxito | Sin `LLM_*` en `X-Degradacion`; `llm.modelo=qwen/qwen3.8-27b` | Narrativa + tiempo `llm_ms` |
| TTS | Audio | Detect éxito | `audio.disponible=true`, MP3 reproducible | Tamaño + duración |
| Timeout | El cliente corta antes que la plataforma | No provocable sin tocar proveedores; revisar la configuración | Cliente 180 s < Azure 240 s; errores de proveedor → 504 con código propio | Revisión documental *(no se fuerza en producción)* |
| 429 | Límite por IP | 7 detect seguidos desde la misma IP | 6 × 200/4xx y luego 429 `RATE_LIMITED` con `Retry-After` | Secuencia de status |
| CORS permitido | Cliente Vercel | Preflight con el origen de producción | 200 + `access-control-allow-origin` | Cabeceras |
| CORS rechazado | Orígenes ajenos | Preflight con `https://evil.example.com` y `…vercel.app.evil.com` | 400 sin `allow-origin` | Cabeceras |
| request_id | Trazabilidad | Cualquier respuesta | `X-Request-ID` único = `error.request_id` = línea del log | Log stream |
| Limpieza | Sin datos del usuario | Tras las pruebas: `az webapp ssh` → `find /tmp/visionnav -type f` | Solo `telemetry/production_metrics.jsonl` | Listado |
| Logs | Observabilidad sin secretos | Log stream durante las pruebas | JSON por solicitud con `app_commit`; ninguna clave (`gsk_`, `AIza`) | Extracto |
| Concurrencia | Serialización + health | 2 detect simultáneos + health en bucle | Ambos 200; el segundo tarda ≈ el doble; health < 1 s | Tiempos |
| Frontend Vercel | Integración | Abrir la app y subir la imagen | Resultado completo sin errores de consola | Captura |
| Audio en navegador | Reproducción | Botón de reproducir | Se oye el audio de Gemini (no la voz del navegador) | Captura / nota |
| Imagen anotada | Visualización | Resultado en la UI | Cajas y etiquetas visibles | Captura |

---

## G. Estado real de Azure (auditoría del 2026-09-27, solo lectura)

| Elemento | Valor encontrado | Requerido | Estado |
|---|---|---|---|
| App | `visionnav-api` (rg `rg-visionnav`, Canada Central), **Stopped** | — | Sin cambios |
| Plan | `ASP-rgvisionnav-aff9`, B1 Linux, 1 instancia | B1, 1 instancia | OK |
| Imagen configurada | `ghcr.io/…:3a1ddb6…` (versión antigua) | `:<sha>` del commit aprobado | La cambia el paso 3 |
| `YOLO_IMGSZ` | **640** | sin definir (congelado: 1280) | **P1 pendiente** |
| `YOLO_WEIGHTS` | `yolo26s.pt` (relativa, no existe en la imagen) | sin definir | **P1 pendiente** |
| `YOLO_IOU` | 0.45 | sin definir | **P1 pendiente** |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | true (resto del despliegue por código) | sin definir | **P1 pendiente** |
| `WEBSITES_PORT` | ausente | 8000 | **P1 pendiente** |
| Health check | sin configurar | `/api/health` | **P1 pendiente** |
| Always On / HTTPS only / TLS / FTPS | on / on / 1.2 / FtpsOnly | = | OK |
| CORS de la plataforma | desactivado | desactivado | OK |
| Paquete GHCR | público (pull anónimo verificado) | = | OK |

Con la configuración actual, **la imagen nueva no arranca** (reproducido localmente: `deteccion.imgsz: esperado 1280, encontrado 640` → exit 3). Es la verificación de identidad actuando como se diseñó. P1 **debe aplicarse antes o junto con el paso 3**:

```
az webapp config appsettings delete -g rg-visionnav -n visionnav-api --setting-names YOLO_IMGSZ YOLO_IOU YOLO_WEIGHTS SCM_DO_BUILD_DURING_DEPLOYMENT -o none
az webapp config appsettings set    -g rg-visionnav -n visionnav-api --settings WEBSITES_PORT=8000 -o none
az webapp config set                -g rg-visionnav -n visionnav-api --generic-configurations '{"healthCheckPath": "/api/health"}' -o none
```

(`-o none` evita que la CLI imprima los valores de las demás variables, que incluyen las claves). Valores anteriores, para revertir: `YOLO_IMGSZ=640`, `YOLO_IOU=0.45`, `YOLO_WEIGHTS=yolo26s.pt`, `SCM_DO_BUILD_DURING_DEPLOYMENT=true`.

**Consecuencia para la tesis:** la demo desplegada anteriormente (`3a1ddb6`) corrió con `imgsz=640`, no con la configuración experimental (1280). Ningún resultado obtenido de esa URL debe presentarse como del sistema evaluado.

## H. Riesgos abiertos (verificables solo en la nube)

1. `TRUSTED_PROXY_HOPS` real de Azure (paso 10).
2. Latencia real de la CPU de B1 frente a la simulación con `--cpus=1`.
3. Tiempo de descarga de la imagen (~530 MB) en el primer arranque.
4. Timeout de 240 s de Azure con colas largas (acotado por el límite por IP).
5. Precio vigente de B1 en la región elegida.
