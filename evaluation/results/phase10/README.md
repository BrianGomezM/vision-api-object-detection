# Fase 10 — Evaluación experimental de H41: separación entre navegación y contexto de escena

**Estado: experimental. Ningún archivo de producción fue modificado.**

**Nota de versión (segunda revisión):** este documento fue auditado dos veces contra los archivos reales (`comparison.json`, `all_variants_raw.json`, `test_images_regression.json`, y el código de producción). La primera revisión corrigió atribuciones causales y el uso de "generalización"; esta segunda revisión corrige un punto que la primera todavía dejaba demasiado fuerte: la atribución a `bowl` en `04_cocina_utensilios.jpg` seguía llamándose "atribución causal limpia", cuando en realidad la variabilidad estocástica del LLM (temperatura 0.1, una sola ejecución por variante, sin repeticiones) impide descartarla como factor de confusión incluso en ese caso de una sola variable. Se corrige aquí. Ningún dato experimental fue alterado — todos los números se reverificaron directamente contra los JSON y coinciden exactamente con lo ya reportado.

## 1. Objetivo

Evaluar experimentalmente si existe evidencia que respalde separar, conceptualmente, las clases usadas para navegación (`_NAV_CLASSES`) de las clases usadas como evidencia contextual para `scene_classifier.py`, y si esa separación puede aplicarse sin producir efectos adversos observables en los componentes de navegación evaluados.

**Distinción explícita de tres situaciones, que no deben confundirse en ningún punto de este documento:**

| Situación | Estado real |
|---|---|
| **Implementación existente** | `_NAV_CLASSES` sigue siendo, hoy, el único filtro compartido por navegación y `scene_classifier.py`. No cambió. |
| **Experimentación realizada** | Se probaron 4 variantes de filtrado **fuera de producción**, en scripts aislados (`scripts/evaluation/phase10_*.py`), reutilizando el código real de `analyze_spatial()`/`classify_scene()` sin modificarlo. |
| **Implementación futura propuesta** | La separación estructural (variante D) **no está implementada** — es una hipótesis con evidencia preliminar, pendiente de diseño, pruebas unitarias, de integración, de regresión y end-to-end antes de considerarse para producción. |

## 2. Hipótesis H41-A y H41-B (separadas explícitamente)

- **H41-A — Hipótesis arquitectónica:** la separación entre las clases utilizadas para navegación (`_NAV_CLASSES`) y las clases utilizadas como evidencia contextual para `scene_classifier.py` constituye una decisión arquitectónica técnicamente justificable.
- **H41-B — Hipótesis experimental:** ampliar las clases disponibles para la clasificación de escenas puede modificar algunas clasificaciones sin producir efectos adversos observables en la navegación, dentro de la implementación evaluada.

**Veredicto por hipótesis (desarrollado en las secciones 14-15):**

| Hipótesis | Estado |
|---|---|
| H41-A (arquitectónica) | **Razonablemente respaldada** — evidencia convergente de código, documentación y experimento (sección 14). |
| H41-B (experimental) | **Evidencia preliminar, acotada a un caso (oficina)** — no rechazada, pero tampoco generalizada; persisten amenazas a la validez no resueltas (sección 13). |

## 3. Configuración experimental

`yolo26s.pt`, `imgsz=1280`, `iou=0.45`, `conf_interno=0.15`, `confidence_threshold` cliente `=0.35` (igual que producción), CPU, `augment=False`. LLM: Groq, `qwen/qwen3.8-27b` (activo desde la Fase 8, sin cambios en esta fase). **Una sola ejecución por combinación (imagen, conjunto de objetos)** — no se repitieron corridas para medir estabilidad del LLM (ver sección 13).

## 4. Dataset

29 imágenes de `evaluation/images/web3d/` (verificado: ninguna agregada ni eliminada) + 12 imágenes de `test_images/` (Capítulo 3, dataset independiente, usado solo para prueba de regresión). Ground truth de escenario: `evaluation/results/phase6/ground_truth.json`, **10 imágenes con `expected_scene` registrado**.

**Verificado directamente contra `comparison.json` (10 filas exactas):** de las 10, 1 está marcada `no_evaluable` (imagen promocional de un asset pack de KayKit, no representa una escena navegable única) y 1 está marcada `no_evaluable_ambiguo` (studio2, escena híbrida sin categoría clara). **Quedan 8 filas evaluables**, no 10.

## 5. Variantes A/B/C/D

- **A — Baseline**: `_NAV_CLASSES` actual, sin cambios.
- **B — Táctico**: `_NAV_CLASSES + {book, keyboard, mouse}`.
- **C — Táctico ampliado**: `_NAV_CLASSES + {book, keyboard, mouse, bowl, cup}`.
- **D — Estructural**: `_NAV_CLASSES` ∪ `{book, keyboard, mouse, bowl, cup, fork, microwave, oven, toaster, remote, toothbrush}` (11 clases). Estas clases nuevas se usaron únicamente como evidencia para `classify_scene()` — nunca se pasaron a `calculate_free_space()`/`decide_movement()`.

**Origen de la lista de D, verificado:** las 11 clases fueron derivadas cruzando las clases reales de COCO-80 contra las palabras clave que `_CONTEXT_GROUPS` (dentro de `app/services/scene_classifier.py`, código de producción sin modificar) ya usa para cada categoría — regla reproducible: *COCO-80 ∩ keywords de `_CONTEXT_GROUPS` que `_NAV_CLASSES` excluye hoy*. Esta lista quedó documentada en el informe de la Fase 9 (`evaluation/results/phase9/README.md`, sección 5), generado antes de ejecutarse el experimento de esta fase, según el orden de trabajo de esta sesión. **No existe un sistema de control de versiones con marcas de tiempo verificables de forma independiente (por ejemplo, commits firmados) que lo confirme criptográficamente** — la trazabilidad se apoya en el orden de generación de los documentos dentro de esta sesión de trabajo, no en un mecanismo externo auditable. Con esa salvedad explícita, no se dispone de evidencia de que la lista haya sido ajustada *después* de observar los resultados de esta fase.

## 6. Resultados — verificados numeral por numeral contra `comparison.json`

| Variante | Correctos | Incorrectos | No evaluables | Fracción | Porcentaje |
|---|---|---|---|---|---|
| A — Baseline | 2 | 6 | 2 | 2/8 | 25% |
| B — Táctico | 3 | 5 | 2 | 3/8 | 37.5% |
| C — Táctico ampliado | 3 | 5 | 2 | 3/8 | 37.5% |
| D — Estructural | 3 | 5 | 2 | 3/8 | 37.5% |

Verificado directamente: los 8 casos evaluables son `biblioteca(F), universidad(F), bathroom(F), sofa(T), living-room(T), low-poly-city(F), office(F→T), waiting-room(F)`. **El único caso cuyo veredicto cambia entre A y B/C/D es `oficina`.** No hay discrepancia entre estas cifras y las de `comparison.json`.

## 7. Comparación — B, C y D están empatadas en desempeño

**Hecho verificado:** B = C = D = 3/8 exactamente. No existe ninguna diferencia de desempeño entre las tres variantes en el conjunto evaluado.

**No se afirma que "D fue la mejor variante"** — sería incorrecto: D no obtuvo mejor resultado que B o C. La preferencia por D (si se preferiera) debe justificarse por razones **arquitectónicas, metodológicas y de mantenibilidad** (la lista de D se deriva de una regla reproducible sobre el código existente, no de una selección manual de 3-5 clases), **nunca por una supuesta superioridad de precisión, porque esa superioridad no existe en los datos.**

**Sobre 25% → 37.5%:** esto significa exactamente **2 casos correctos → 3 casos correctos**, es decir, un caso evaluable adicional dentro de un conjunto de 8. No se presenta, y no debe presentarse, como una mejora porcentual general del sistema ni como evidencia estadística de mejora — el tamaño de muestra no lo permite.

## 8. Transferencia a `test_images/` (no "generalización")

Prueba de regresión A vs. D sobre las 12 imágenes de `test_images/` (`test_images_regression.json`, verificado directamente):

| Imagen | Clases nuevas en D | Cambio de escena | Ground truth disponible |
|---|---|---|---|
| `03_escritorio_objetos.jpg` | `cup, keyboard, mouse` (conjuntas) | espacio interior → oficina | No hay ground truth de escenario para este conjunto; se interpreta por coincidencia con el patrón de oficina, no se confirma formalmente |
| `04_cocina_utensilios.jpg` | `bowl` (única) | espacio interior → comedor | Ídem |
| `09_objetos_pequenos.jpg` | `book, cup, keyboard, mouse` (conjuntas) | comedor → espacio interior | Ídem — imagen curada para pruebas de detección, no diseñada como escena navegable representativa |
| 9 imágenes restantes | ninguna o sin cambio de escena | sin cambio | — |

**Formulación correcta (no "generalización"):** se observó **evidencia preliminar de transferencia** del patrón "incorrecto → oficina" a una segunda imagen (`03_escritorio_objetos.jpg`) en un dataset independiente. Esto es **evidencia exploratoria de comportamiento consistente en 2 casos**, no una demostración de que el sistema generaliza a escenas Web3D en general. El conjunto es pequeño, parcialmente curado, y ninguna de estas 12 imágenes tiene ground truth de escenario formal — la lectura de "mejora" en `03` y `04` es una interpretación razonable por coincidencia de patrón, no una verificación contra una referencia etiquetada independientemente.

## 9. Impacto sobre navegación y análisis espacial

**Verificado directamente en el código de producción (sin modificar), en esta segunda revisión:**

- `app/services/spatial_analyzer.py`, diccionario `OBJECT_TAXONOMY`: `book, keyboard, mouse, cup, bowl, fork, remote, toothbrush` están listados bajo la clave `"small_object"`, con el comentario en el propio código: *"No bloquean el paso. Baja prioridad en narrativa."* `microwave, oven` están bajo `"informative"`.
- `app/services/free_space_analyzer.py`, conjunto `_NON_BLOCKING = frozenset({"small_object", "informative", "exit", "other"})`: ambas categorías anteriores quedan excluidas del cálculo de bloqueo.

**Resultado experimental (ejecución real, no solo lectura de código):** se simuló un `book` ocupando el 80% del cuadro, a mínima distancia, junto con `keyboard`/`mouse`/`cup`, sobre una imagen de prueba de 800×600 píxeles. `calculate_free_space()` devolvió `center: 0.0` (libre) y `situation: "clear"`.

**Conclusión, con el alcance correcto:** *"En la implementación evaluada no se observó impacto adverso de estas clases sobre la decisión espacial."* Esto es válido únicamente para el código de `spatial_analyzer.py`/`free_space_analyzer.py` tal como existe hoy. **No se afirma, ni puede afirmarse, que "estas clases nunca afectarán la navegación"** — si una implementación futura modifica `OBJECT_TAXONOMY`, `_NON_BLOCKING`, o la forma en que estas clases se integran al pipeline, esta prueba deberá repetirse; no es una garantía permanente.

## 10. Caso biblioteca

**HECHO:** en la imagen de biblioteca, la única clase nueva incorporada entre A y B/C/D es `book` (verificado en `all_variants_raw.json`: A tiene 8 clases distintas, B/C/D tienen las mismas 8 más `book` — ninguna otra clase de la lista de 11 aparece en esta imagen).

**RESULTADO:** la clasificación de escena no cambió en ninguna variante — se mantuvo `comedor`.

**INTERPRETACIÓN:** la incorporación de `book` no fue suficiente, en este caso, para desplazar la clasificación hacia una categoría distinta. Esto es compatible con que la representación contextual actual (que no incluye una categoría "biblioteca" en `_CONTEXT_GROUPS` ni en el prompt del LLM) constituya una limitación relevante.

**LO QUE NO ESTÁ DEMOSTRADO:** que la ausencia de categoría sea la *única* causa. El experimento no aisló el peso relativo de `dining table`+`chair` (ya presentes en A, y semánticamente asociados a "comedor") frente al de `book`; tampoco se probó si una categoría "biblioteca" de control habría cambiado el resultado. **No se afirma "causa raíz confirmada."**

**EXPERIMENTO QUE SÍ permitiría confirmarlo:** agregar temporalmente una categoría "biblioteca" de prueba (sin implementarla en producción) y verificar si, con `book` ya disponible, el resultado cambia. No ejecutado en esta fase (fuera del alcance autorizado: "no implementar nuevas categorías").

## 11. Causalidad de `book`, `keyboard`, `mouse`, `bowl`, `cup` — revisado con el máximo rigor

Para cada clase, se distingue explícitamente:

| Clase | Diferencia observada | Asociación | Influencia plausible | Causalidad demostrada |
|---|---|---|---|---|
| `keyboard` + `mouse` (oficina Web3D y `test_images/03`) | Sí — ambas se incorporan siempre juntas en B/C/D | Sí, coincide con el cambio incorrecto→oficina en 2 imágenes | Sí, es plausible semánticamente | **No** — nunca se aislaron una de otra; no se puede saber si basta una sola |
| `bowl` (`test_images/04_cocina_utensilios.jpg`) | Sí — es la única clase nueva en esta imagen específica | Sí, coincide con el cambio interior→comedor | Sí, es la variable de evidencia más directamente asociada al cambio en este caso | **No** — aunque es la única clase que cambió, el LLM no fue ejecutado más de una vez sobre la misma entrada; no puede descartarse que la variabilidad estocástica del modelo (temperatura 0.1, sin repeticiones) contribuyera al resultado observado, incluso con una sola variable de diferencia |
| `book` (biblioteca Web3D) | Sí — única clase nueva en esa imagen | No — el resultado no cambió | No aplica (sin cambio que explicar) | No aplica |

**Formulación correcta para el caso más fuerte del experimento (`bowl`):** *"`bowl` fue la única clase nueva incorporada como evidencia contextual en este caso, y su incorporación es compatible con una posible influencia sobre la clasificación. No se demostró causalidad, porque no se ejecutaron repeticiones que permitan descartar la variabilidad del LLM como factor de confusión, ni una condición de ablación que aislara `bowl` de forma independiente en más de una imagen."* Esta es la redacción más precisa que los datos permiten — es un nivel más cauto que la formulación de la primera revisión de este documento, y se adopta aquí como la definitiva.

## 12. Scores del clasificador

**Verificado en el código:** `classify_scene()` (`app/services/scene_classifier.py`) retorna únicamente `scene_type`, una `confidence` categórica autorreportada (`alta`/`media`/`baja`, no numérica), y `scene_intro`. **No existe, en el código de producción ni en los resultados generados por esta fase, ningún score numérico por categoría candidata, probabilidad, ni desglose de por qué una categoría "ganó" sobre otra en el camino LLM.** No se dispone de logs con las respuestas completas del LLM más allá de los campos ya reportados (`scene_type`, `confidence`, `scene_intro`, `llm_error`). **No se inventan estos valores.** Como trabajo futuro: instrumentar `scene_classifier.py` (o el prompt del LLM) para que devuelva una puntuación o justificación por categoría candidata permitiría estudiar la estabilidad de las decisiones y sustentar interpretaciones causales con mayor rigor.

## 13. Variabilidad del LLM y amenazas a la validez

- **Una sola ejecución por combinación (imagen, conjunto de objetos)** en todo el experimento — no se realizaron repeticiones para medir estabilidad. Esta es una limitación explícita, no una omisión oculta.
- **Variabilidad ya observada entre fases distintas:** la escena de calle low-poly fue clasificada como "exterior"/"calle urbana" en las Fases 7-8 y como "espacio interior" en esta fase, con idéntico código, configuración y (hasta donde se verificó) el mismo modelo. `temperature=0.1` no garantiza determinismo exacto entre llamadas.
- **Amenaza a la validez de todas las atribuciones de esta fase:** dado que no hay repeticiones, cualquier cambio de clasificación observado podría, en principio, deberse en parte a la variabilidad estocástica del LLM y no exclusivamente a la evidencia añadida. Esta amenaza no se puede descartar con los datos disponibles — se declara explícitamente en vez de ignorarse.

## 14. Interpretación de H41-A (arquitectónica)

Evidencia convergente de tres fuentes independientes:
1. **Código:** `OBJECT_TAXONOMY` (Nivel 4) ya anticipa y categoriza objetos (`small_object`, `informative`) que `_NAV_CLASSES` (Nivel 2) nunca deja pasar hoy.
2. **Documentación:** la actividad A13 y el documento principal de la tesis describen `_NAV_CLASSES` únicamente en términos de navegación/accesibilidad física (Fase 9).
3. **Experimento:** agregar clases de contexto no afectó el cálculo de espacio libre bajo la implementación evaluada (sección 9).

**Conclusión razonable:** H41-A cuenta con un respaldo razonable a partir de evidencia convergente. Esto es un argumento arquitectónico bien evidenciado, no una prueba formal exhaustiva de que la separación sea la única solución correcta posible.

## 15. Interpretación de H41-B (experimental)

- Se observó un cambio de clasificación favorable en un tipo de escena (oficina), con un comportamiento consistente observado en una imagen independiente de un segundo dataset, pero sin aislar la contribución individual de `keyboard`/`mouse` (sección 11).
- No se observaron cambios adversos en ningún caso con ground truth disponible (secciones 6-8).
- No se puede descartar la variabilidad del LLM como factor de confusión en ninguno de los cambios observados (sección 13).

**Conclusión:** H41-B cuenta con **evidencia preliminar, limitada a un tipo de escena, no confirmatoria**. No se rechaza, pero tampoco se declara demostrada.

## 16. Conclusión general

La Fase 10 aporta evidencia razonable a favor de la hipótesis arquitectónica (H41-A) y evidencia preliminar, acotada y no concluyente a favor de la hipótesis experimental (H41-B). No demuestra generalización a escenas Web3D en general, no demuestra causalidad individual de ninguna clase, y no garantiza ausencia de impacto en navegación para cualquier implementación futura — solo para la evaluada aquí.

## 17. Limitaciones

1. Solo 8 casos evaluables con ground truth en el conjunto principal.
2. Ninguna repetición de ejecución del LLM — no hay medida de estabilidad.
3. `keyboard`/`mouse` y la mayoría de las demás clases nunca se aislaron individualmente.
4. `test_images/` no tiene ground truth de escenario formal para los casos usados en la transferencia.
5. No existen scores numéricos del clasificador para sustentar mecánicamente las interpretaciones.
6. La trazabilidad temporal de la lista de D se apoya en el orden de trabajo de la sesión, no en un mecanismo de verificación externo.
7. El caso biblioteca no permite distinguir entre "falta de categoría" y "peso relativo de la evidencia existente" como causa dominante.

## 18. Trabajo futuro (no ejecutado — no se inventan resultados)

**Ablación propuesta para oficina** (Web3D y `test_images/03`): `A`, `A+book`, `A+keyboard`, `A+mouse`, `A+book+keyboard`, `A+book+mouse`, `A+keyboard+mouse`, `A+book+keyboard+mouse` — con **repeticiones** para poder distinguir variabilidad del LLM de efecto real de cada clase.

**Ablación propuesta para cocina/comedor:** `A`, `A+bowl`, `A+cup`, `A+bowl+cup`, igualmente con repeticiones, sobre más de una imagen si se identifican casos adicionales.

**Otros pendientes:**
- Prueba de categoría "biblioteca" de control (sección 10).
- Instrumentación de scores/justificación por categoría en el clasificador (sección 12).
- Ampliación del dataset con ground truth formal antes de citar cualquier cifra en el Capítulo 5.
- Si se aprueba una implementación futura de la separación estructural: diseño concreto en `yolo_service.py`, pruebas unitarias, pruebas de integración, prueba de regresión completa, y validación end-to-end real (como la Fase 8) antes de considerarla parte de producción.

## Entregables e integridad de producción

Mismos archivos de la entrega original (`baseline_results.json`, `tactical_results.json`, `tactical_extended_results.json`, `structural_results.json`, `comparison.json`, `all_variants_raw.json`, `test_images_regression.json`) — ningún dato fue alterado en esta revisión, solo `README.md` y `analysis.md`. `git status` idéntico antes y después: `D test_images/cap1.png` (preexistente, ajeno) + `evaluation/`, `scripts/compare_checkpoints.py`, `scripts/evaluation/` sin trackear. `.env` sin cambios desde la Fase 8. Ningún archivo de `app/` fue modificado. Sin commits, sin pushes.
