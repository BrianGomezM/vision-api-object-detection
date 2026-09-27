# Auditoría de la regla del umbral por clase (`min()` frente a `max()`) y de la fase 10

- **Fecha:** 2026-09-27.
- **Método:** historial de git, código, resultados guardados y el texto de la tesis. **No se ejecutó YOLO ni el LLM.**
- **Reproducción:** `python scripts/audits/audit_phase10_regla_umbral.py <salida.json>`.
- **Cambios de código:** ninguno. La regla **no se modificó**; la decisión queda para el CP3B (D4).

## 1. Hechos verificados

| # | Hecho | Fuente |
|---|---|---|
| H1 | Desde `de2cff4` (19-05-2026) hasta el 21-09-2026, producción usaba `effective = min(class_min, umbral)` | `git log -S` sobre `yolo_service.py` |
| H2 | El commit `2c29448` (21-09-2026 01:14, "ajustes") la cambió a `max(class_min, umbral)`. Motivo que da el código: con `min()`, "subir el slider no filtraba estas ~30 clases". Es un motivo de interfaz, **sin evaluación metodológica documentada**. | diff de `2c29448` |
| H3 | La fase 9 (20-09, 02:46) y la fase 10 (20-09, 03:00) se ejecutaron **antes** del cambio. Sus scripts replican la regla `min()`, que era la de producción en ese momento. | fecha de modificación de los resultados; `phase10_*.py:48/91` |
| H4 | **La tesis documenta `min()`.** La tabla de la función `run_yolo` dice "Aplica umbral efectivo = min(_CLASS_MIN_CONF[clase], umbral_cliente)". La Tabla 9 justifica umbrales bajos ("el umbral bajo evita omisiones críticas", mesa 0,10). | `_CORREGIDO.docx`, línea 855 del texto extraído; Tabla 9 |
| H5 | **Hoy el código usa `max()`.** Con el umbral por defecto (0,35), los mínimos por debajo de 0,35 no tienen efecto: mesa, planta, botella, silla, sofá, persona, etc. | `yolo_service.py:325` |
| H6 | **Segunda diferencia de la fase 10 respecto a producción: el preprocesado.** Los scripts pasan la imagen **original** a YOLO. Producción la redimensiona antes (máximo 800 px y re-codificación JPEG q90, `resize_image`). Por eso el README de la fase 10 ("igual que producción") era exacto en la regla, pero **no en el preprocesado**. | `phase10_*.py:65/113`; `core/pipeline.py` |
| H7 | En la fase 2A (41 imágenes, 1280 px, preprocesado de producción), `min()` añadió 50 detecciones con confianza entre 0,157 y 0,349, sobre todo `dining table` (24). En el GT manual de presencia de 5 imágenes: `min` dio VP 16 / FN 1 / no confirmadas 3; `max` dio VP 15 / FN 2 / no confirmadas 2. **La muestra es exploratoria y muy pequeña.** | `phase2a/detection/analysis_summary.txt` |

## 2. ¿Afecta a los resultados de la fase 10?

La fase 10 solo guardó las clases resultantes, sin sus confianzas. Para recalcular se usaron las cajas crudas de la fase 2A, que tienen la misma configuración de YOLO pero **el preprocesado de producción**. Como la fase 10 no redimensionaba, la reproducción exacta solo es posible en algunas imágenes:

| Conjunto | `min()` reproduce exactamente las clases de la fase 10 | De esas, cambiarían con `max()` |
|---|---|---|
| Web3D (29) | 15 imágenes (variantes A y D) | 3: JC3D-03 (pierde `bottle`), SKF-05 (pierde `dining table` y `vase`), **SKF-14** (pierde `dining table`) |
| test_images (12) | 7 imágenes | 2: 01_persona_bolso (pierde `suitcase`), 11_escritorio (pierde `dining table`) |

- **Imágenes no reproducibles.** En 14 imágenes Web3D y 5 de `test_images` la reproducción no es exacta, por la diferencia de preprocesado. Para ellas **no puede determinarse** el efecto exacto sin volver a ejecutar YOLO, y además el LLM, que no es determinista.
- **Indicio en dos imágenes con GT de escenario.** Con los datos de la fase 2A, Universidad (`bench`) y SKF-07 (`dining table`) también perderían clases.

**Sobre las afirmaciones de la tesis (§5.2.3 y conclusiones de H41):**

- **La observación central de H41-B se mantiene:** la escena de oficina (SKF-09) pasa de "sala de estar" a "oficina". SKF-09 se reproduce exactamente y su entrada al clasificador **es idéntica con `min()` y con `max()`**. Queda la salvedad, ya declarada en la tesis, de que el LLM no es determinista.
- **Los recuentos "de 2 a 3 aciertos sobre 8 casos evaluables" corresponden a la configuración del 20-09** (regla `min()` y sin redimensionar). Con la configuración actual podrían variar:
  - **SKF-14** tiene una entrada distinta. Hoy se clasifica como "comedor" (incorrecto), probablemente por la mesa de baja confianza;
  - **SKF-07 y Universidad** probablemente también.

  **No se puede afirmar** que los recuentos sean los mismos en el sistema actual, **ni que sean distintos**, sin volver a ejecutar.
- **La afirmación "sin regresiones" en los 8 casos** también está ligada a esa configuración.
- **La tesis es coherente consigo misma** (describe `min()` y evaluó con `min()`), pero **no con el código actual** (`max()`).

## 3. ¿Qué regla es la metodológicamente correcta?

Los datos no permiten elegirla por rendimiento. Elegir la regla que dé mejores resultados en el Dataset 1 sería ajustar sobre el conjunto de evaluación, y la fase 2A es exploratoria (n = 5). La elección debe hacerse **antes** de F4 y justificarse por **coherencia entre el diseño documentado y el sistema evaluado**.

| Regla | Semántica | Coherencia con la tesis | Consecuencias |
|---|---|---|---|
| **`min(class_min, T)`** (hasta el 20-09) | El mínimo de clase solo puede **rebajar** el umbral | **Alta.** La tesis la documenta (línea 855), cumple la justificación de la Tabla 9 para las clases de riesgo (mesa, planta, botella, puerta…) y es la regla con la que se hicieron las fases 9 y 10. | El umbral del cliente no puede endurecer las clases con mínimo propio (el problema de interfaz de H2). `tv` queda en 0,35, no en 0,40: contradice la Tabla 9 en las clases "informativas". |
| **`max(class_min, T)`** (actual) | El mínimo de clase es un **piso** | **Baja.** Con el valor por defecto, la Tabla 9 no tiene efecto en casi todas las clases, y la tesis diría algo falso (línea 855). | Hay que reescribir la Tabla 9 y la línea 855, y declarar que las fases 9 y 10 describen una configuración anterior. |
| **Sustitución** (`class_min` si existe, `T` si no) | Aplica la Tabla 9 literalmente, en los dos sentidos | Coherente con la Tabla 9, pero **no está documentada ni se ha evaluado nunca** | Sería una tercera configuración, introducida justo antes de la evaluación |

**Recomendación (para aprobar, no aplicada):**

- **Restaurar `min()` como regla oficial del sistema evaluado.** Es la que documenta la tesis, la que justifica la Tabla 9 y la que usaron las fases 9 y 10. El cambio a `max()` fue una decisión de interfaz sin evaluación.
- **Declarar la limitación:** con `min()`, el control deslizante del cliente no endurece las clases con mínimo propio. La `tv` a 0,40 de la Tabla 9 no se alcanza con el umbral por defecto: la Tabla 9 debería decir "0,35 efectivo".
- **Si se prefiere mantener `max()`**, hay que actualizar la tesis (línea 855 y Tabla 9) y rotular la evidencia de las fases 9 y 10 como "configuración anterior".

**Cómo se aplicaría cualquiera de las dos opciones:**

- **Cuándo:** antes de F4, en un commit propio.
- **Pruebas:** la regresión byte a byte cambiará a propósito, porque la línea base se capturó con `max()`. Se regenera y el diff se revisa, igual que con el contrato.
- **Congelación:** la regla elegida se congela en `experimental_config.yaml` (CP3B, D4).

## 4. Qué debe corregirse en los documentos (sin tocar la tesis todavía)

- **README de la fase 10:** decir que usó la regla `min()` (la de producción el 20-09) y la **imagen original, sin el redimensionado de producción**.
- **Tesis §5.2.3:** indicar la configuración exacta (regla y preprocesado) con la que se obtuvieron los recuentos. Evitar sugerir que representan el sistema desplegado hoy.
- **Tesis, línea 855 y Tabla 9:** alinearlas con la regla que se apruebe.
