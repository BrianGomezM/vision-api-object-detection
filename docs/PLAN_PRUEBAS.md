# Plan de pruebas funcionales y no funcionales: trazabilidad y resultados

- **Plan original:** 33 casos (14 funcionales, 8 no funcionales y 11 de integración), definidos en la
  fase de cierre metodológico (documento 09 de la carpeta `claude`). En septiembre de 2026 solo 8
  estaban codificados.
- **Este documento:** relaciona cada caso con la prueba que hoy lo verifica y con su resultado.
  Verificado el **11 de octubre de 2026** sobre el commit `fcf80e9` (versión desplegada).
- **Suites:**
  - Suite automatizada (`pytest -m "not entorno_referencia and not live_smoke"`): 442 pruebas,
    438 aprobadas y 4 omitidas por requerir el entorno de referencia. Proveedores simulados: no
    consume servicios externos. Se ejecuta en GitHub Actions antes de cada despliegue.
  - Regresión del procesamiento (`tests/regression`, entorno de referencia): 3/3 aprobadas; las
    41 imágenes de la fase 2A producen una salida idéntica a la línea base.
  - Mediciones con los proveedores reales: `scripts/evaluation/plan_pruebas_nf.py`
    (`evaluation/results/plan_pruebas/nf_fcf80e9*.json`) y `scripts/hardening/concurrency_bench.py`
    (`evaluation/results/plan_pruebas/concurrencia_fcf80e9.json`).

## Funcionales (FUN)

| ID | Requisito | Qué se verifica | Prueba que lo cubre | Resultado |
|---|---|---|---|---|
| FUN-01 | RF-02, RF-06 | Detección y narrativa en una escena con muebles | `test_E2E01_imagen_valida_flujo_completo`, `test_E2E03_multiples_objetos`; con proveedores reales: `plan_pruebas_nf.py` (6 escenas del estudio) | Aprobado; 6/6 escenas reales con narrativa y audio |
| FUN-02 | RF-02 | Imagen sin objetos relevantes | `test_E2E02_sin_objetos_relevantes` | Aprobado (200, sin objetos, sin degradación) |
| FUN-03 | RF-08 | `/api/health` informa el estado | `test_health_y_tts_models_publicos`, `test_health_sin_carga`, `test_production_health_basico_sin_datos_internos` | Aprobado |
| FUN-04 | RF-02 | Un umbral menor no detecta menos objetos | `test_FUN04_umbral_bajo_detecta_igual_o_mas_objetos`, `tests/test_yolo_rule.py` (4) | Aprobado; confirma la regla min(mínimo de clase, umbral) |
| FUN-05 | RF-01 | Archivo vacío | `test_archivo_vacio` | Aprobado (400, `EMPTY_FILE`) |
| FUN-06 | Herramienta de desarrollo | Subida al conjunto de datos | `test_FUN06_subida_al_conjunto_de_datos`, `test_FUN06_07_no_existen_en_production` | Aprobado; no se publica en producción |
| FUN-07 | Herramienta de desarrollo | Estadísticas del conjunto de datos | `test_FUN07_estadisticas_del_conjunto_de_datos` | Aprobado |
| FUN-08 | RF-01 | Formato inválido: nunca 500 ni 200 con error | `test_A_imagen_corrupta`, `test_A_jpeg_truncado`, `test_B_formato_de_imagen_no_soportado`, `test_B_archivo_que_no_es_imagen`, `test_C_archivo_demasiado_grande`, `test_no_hay_http_200_con_error_en_el_codigo` | Aprobado (422, 415 y 413 con mensaje uniforme) |
| FUN-09 | RF-01 | Umbral fuera de rango | `test_FUN09_umbral_fuera_de_rango` (1,5 y −0,1) | Aprobado. El plan esperaba 422; la API responde 400 `INVALID_REQUEST`, según el contrato de errores (`docs/CONTRATO_ERRORES.md`) |
| FUN-10 | RF-07 | Respuesta en audio | `test_E2E05b_audio_true_devuelve_mp3`, `test_E2E05c_cabeceras_de_texto_codificadas` | Aprobado (audio/mpeg y cabeceras) |
| FUN-11 | Diagnóstico | Modo `debug` con objetos y entrada del modelo de lenguaje | `test_debug_detect_usa_core_pipeline_run`, `test_core_pipeline.py` | Aprobado |
| FUN-12 | Diagnóstico | Las ocho etapas del procesamiento | `test_FUN12_diagnostico_con_las_ocho_etapas` | Aprobado |
| FUN-13 | Hallazgo H19 | `/api/metrics` en memoria | `test_get_api_metrics_eliminado`; sustituto: `/api/metrics/summary` y `/api/metrics/latency` (`test_production_routes.py`) | Sustituido: el endpoint con el defecto se eliminó y sus métricas pasaron a la telemetría persistente |
| FUN-14 | Infraestructura del objetivo 3 | Registro de la valoración de una narrativa | `test_FUN14_valoracion_persistida` | Aprobado. Para el estudio con usuarios se usa el módulo de sesiones (`tests/test_study_sessions.py`, 63 pruebas) |

## No funcionales (NF)

| ID | Requisito | Qué se mide | Evidencia | Resultado |
|---|---|---|---|---|
| NF-01 | RNF-02 | Tiempo total de respuesta | `plan_pruebas_nf.py`, 6 escenas, CPU | Procesamiento sin voz: 1,2 s de media (cumple < 5 s). Con la voz por defecto (Gemini TTS): 17,0 s de media (12,7–20,6 s); la síntesis ocupa 15,7 s. Con la voz de Azure (es-CO-SalomeNeural, `nf_302db7e_es-CO-SalomeNeural.json`): 3,7 s de media (2,6–5,8 s), síntesis 2,3 s de media; 5 de 6 solicitudes por debajo de 5 s (la primera tardó 5,8 s porque su procesamiento fue más lento, 3,0 s) |
| NF-02 | RNF-02 | Tiempo por etapa | `plan_pruebas_nf.py` | Detección 600 ms; escenario y narrativa 561 ms (en paralelo); análisis espacial, pasos, espacio libre y decisión < 1 ms cada uno; imagen anotada 20 ms; voz 15 686 ms |
| NF-03 | RNF-01, RNF-02 | Comportamiento con solicitudes simultáneas | `concurrency_bench.py` (1 worker, YOLO real en CPU, voz simulada con 16,5 s) | Sin errores ni caídas; `/api/health` responde en 34–124 ms durante la carga. En el perfil de desarrollo las solicitudes se atienden en serie: 1, 2, 3 y 5 simultáneas terminan en 18, 36, 54 y 89 s. En el perfil de producción, una ráfaga de 8 atiende 6 (4,7–26,6 s) y rechaza 2 con 429 `RATE_LIMITED` por el límite por cliente, como está diseñado. Memoria máxima 739 MB; sin mezcla de datos entre solicitudes |
| NF-04 | RNF-06 | Tasa de solicitudes exitosas | `plan_pruebas_nf.py` y `concurrency_bench.py` | 6/6 con proveedores reales; con carga, 0 errores del servidor (los únicos rechazos son los 429 del límite de solicitudes) |
| NF-05 | RNF-01 | Uso de CPU | `plan_pruebas_nf.py` | 31 % de media por solicitud (máx. 57 %), sin GPU |
| NF-06 | RNF-01 | Memoria del proceso | `plan_pruebas_nf.py` y `concurrency_bench.py` | 280 MB al iniciar y 496 MB como máximo tras cargar el modelo; sin crecimiento sostenido |
| NF-07 | RNF-07 | Tamaño de la respuesta JSON | `plan_pruebas_nf.py` | 694 KB de media (máx. 834 KB), con el audio y la imagen anotada en base64 |
| NF-08 | RNF-03 | Estabilidad ante el umbral y repetibilidad | `plan_pruebas_nf.py` | Mismos objetos y misma confianza media con 0,25, 0,35 y 0,5 en las 6 escenas; 3 repeticiones idénticas por escena |

## Integración (INT)

| ID | Requisito | Qué se verifica | Prueba que lo cubre | Resultado |
|---|---|---|---|---|
| INT-01 | RF-01 | Imagen mayor de 800 px se reduce | `test_INT01_imagen_mayor_de_800_px_se_reduce`, `test_dimension_extrema_procesable` | Aprobado |
| INT-02 | RF-01 | Imagen de 800 px o menos no cambia | `test_INT02_imagen_de_800_px_o_menos_no_cambia`, `test_dimensiones_validas` | Aprobado |
| INT-03 | RF-03 | Columna izquierda, centro o derecha | `test_INT03_columna_por_tercios` (7 posiciones); regresión de 41 imágenes | Aprobado |
| INT-04 | RF-03 | Categoría de cercanía | Regresión de 41 imágenes (`tests/regression`) | Aprobado (salida idéntica a la línea base) |
| INT-05 | RF-04 | Estimación de pasos | Regresión de 41 imágenes | Aprobado (salida idéntica a la línea base) |
| INT-06 | RF-05 | Todo bloqueado | `test_INT06_todo_bloqueado`; regresión | Aprobado (indica la salida menos bloqueada) |
| INT-07 | RF-05 | Centro libre | `test_INT07_centro_libre`; regresión | Aprobado («avanzar hacia el frente») |
| INT-08 | RF-05 | Solo un lateral libre | `test_INT08_solo_un_lateral_libre`; regresión | Aprobado («el paso al frente está bloqueado… izquierda») |
| INT-09 | RF-02 a RF-06 | Coherencia de extremo a extremo | `test_E2E01`, `test_E2E03`, `test_detect_usa_core_pipeline_run`, `test_FUN12_diagnostico_con_las_ocho_etapas` | Aprobado |
| INT-10 | RNF-06 | Degradación si falla el modelo de lenguaje | `test_E2E04_error_de_llm`, `test_IJK_llm_en_development_se_degrada_y_se_declara`, `test_O_llm_no_configurado` | Aprobado (narrativa de respaldo y cabecera `X-Degradacion`) |
| INT-11 | RNF-06 | Degradación si falla la voz | `test_E2E05_error_de_tts`, `test_LMN_tts`, `test_errores_de_azure_usan_el_contrato_de_degradaciones` | Aprobado (narrativa sin audio, declarada) |

## Resumen

- **Ejecutados y aprobados:** 32 de 33 casos.
- **Sustituido:** FUN-13, porque el endpoint con el defecto se eliminó.
- **RNF-02 (menos de 5 s):** se cumple en el procesamiento (1,2 s) y con la voz de Azure (3,7 s de
  media; 5 de 6 solicitudes por debajo de 5 s), pero no con la voz por defecto de Gemini (17,0 s).
- **Cambios de expectativa documentados:** FUN-09 responde 400 en lugar de 422, y FUN-06/07/14 son
  herramientas de desarrollo que no se publican en producción.
