# Fase 9 — Auditoría de H41: filtrado de objetos relevantes para navegación

**Estado: auditoría. Ningún archivo de producción fue modificado. `_NAV_CLASSES`, `.env`, `scene_classifier.py`, `yolo_service.py` intactos.**

## 1. Resumen ejecutivo

`_NAV_CLASSES` **está documentado explícitamente en el propio texto de la tesis y en la actividad A13** como un filtro diseñado exclusivamente para "clases relevantes para la navegación" (obstáculos, salidas, peligros, objetos informativos) — **no** para clasificación de escena ni para narrativa. El problema no es que se haya "olvidado" incluir `keyboard`/`mouse`: es que `scene_classifier.py`, un módulo distinto y más reciente, reutiliza sin adaptación un filtro diseñado para un propósito diferente. Un experimento aislado confirmó que agregar `keyboard`+`mouse` **sí corrige** la clasificación de oficina, pero agregar `book` **no corrige** la de biblioteca — la solución no generaliza. Además, `book` (no solo `keyboard`/`mouse`) es la clase excluida con mayor presencia real en el dataset Web3D del proyecto (10 de 29 imágenes). **Conclusión: H41 requiere rediseño metodológico, no un parche de dos clases.**

## 2. Descripción de H41

Ver Fase 8. `_NAV_CLASSES` excluye `keyboard` y `mouse` pese a ser clases reales de COCO-80 detectadas con alta confianza, lo que contribuyó a que la escena de oficina llegara al clasificador con evidencia insuficiente.

## 3. Funcionamiento actual de `_NAV_CLASSES`

`app/services/yolo_service.py`, líneas 132-144. Es la unión de las 32 claves de `_CLASS_MIN_CONF` más 8 clases de vehículos/movilidad sin umbral propio (`bicycle, motorcycle, car, bus, truck, sports ball, skateboard, umbrella`). Se aplica dentro de `run_yolo()`, como **primer filtro** (`if label not in _NAV_CLASSES: continue`), **antes** del filtro por umbral de confianza por clase. Una detección cuya clase no esté en el conjunto se descarta **sin importar su confianza** — no hay excepción ni registro de lo descartado (no se conserva un log de "clases rechazadas por allowlist").

**Motivo documentado en código:** el docstring de `yolo_service.py` dice *"Conjunto de todas las clases relevantes para navegación."* — coincide con la documentación externa encontrada (ver sección 9).

## 4. Flujo del pipeline (5 niveles)

| Nivel | Módulo | Rol en H41 |
|---|---|---|
| 1. Detección | `yolo_service.py` → `model.predict()` | YOLO detecta correctamente `keyboard`(0.95) y `mouse`(0.88) — sin error aquí |
| 2. Filtrado | `yolo_service.py` → `_NAV_CLASSES` | **Aquí ocurre H41.** Las detecciones correctas de nivel 1 se descartan antes de continuar |
| 3. Interpretación de escena | `scene_classifier.py` | Recibe una lista ya empobrecida — no tiene forma de saber que existían `keyboard`/`mouse`; su comportamiento con la evidencia que sí recibe es razonable dado lo que tiene |
| 4. Análisis espacial | `spatial_analyzer.py` | No interviene en H41 |
| 5. Narrativa | `llm_enhancer.py` | Hereda la misma lista empobrecida que el Nivel 3 |

**H41 ocurre en el Nivel 2. No se le atribuye al LLM ni al clasificador de escena — ambos están, en este caso concreto, funcionando correctamente sobre datos incompletos que ya llegaron filtrados.**

## 5. Clases incluidas/excluidas

De las 80 clases de COCO-80, **31 son realmente alcanzables** a través de `_NAV_CLASSES` (23 de `_CLASS_MIN_CONF` que coinciden con nombres reales de COCO + 8 de vehículos/movilidad; las 9 entradas restantes de `_CLASS_MIN_CONF` — `table, desk, sofa, stool, stairs, door, bag, box, monitor` — ya se documentaron en la Fase 3 como código inerte, no corresponden a clases reales del modelo). **49 clases reales de COCO quedan excluidas.**

### Cruce contra las 10 categorías del heurístico de escenario — hallazgo central de esta fase

Se verificó, keyword por keyword, cuántas de las palabras que `scene_classifier.py` usa para reconocer cada categoría corresponden a clases que **de hecho pueden llegar** al clasificador (es decir, que están en `_NAV_CLASSES` Y son clases reales de COCO):

| Categoría heurística | Keywords totales | Keywords realmente alcanzables | Keywords excluidas por `_NAV_CLASSES` o inexistentes |
|---|---|---|---|
| exterior | 6 | **6/6 (100%)** | ninguna |
| sala de cine | 4 | 3/4 (75%) | control remoto (`remote`) |
| entrada o pasillo | 4 | 3/4 (75%) | puerta (`door`, ni siquiera es clase real) |
| sala de estar | 5 | 4/5 (80%) | control remoto (`remote`) |
| baño | 3 | 2/3 (67%) | cepillo de dientes (`toothbrush`) |
| comedor | 6 | 3/6 (50%) | taza (`cup`), cuenco (`bowl`), tenedor (`fork`) |
| dormitorio | 5 | 2/5 (40%) | almohada, lámpara, armario (ni siquiera clases reales de COCO) |
| **oficina** | 7 | **2/7 (29%)** | teclado (`keyboard`), ratón (`mouse`), libro (`book`) — 3 de las 7; escritorio/monitor ni siquiera son clases reales |
| **cocina** | 6 | **2/6 (33%)** | microondas (`microwave`), horno (`oven`), tostadora (`toaster`), cuenco (`bowl`) — 4 de las 6 |
| tienda | 5 | 4/5 (80%) | libro (`book`) |

**Esto explica, de forma sistemática y no anecdótica, por qué "oficina" y "cocina" son las categorías más frágiles del sistema**: son las que más dependen de objetos pequeños de escritorio/electrodomésticos que `_NAV_CLASSES` excluye casi por completo — mientras que "exterior" (100% alcanzable) es consistentemente la categoría mejor clasificada en las Fases 6-8. Esto **no se limita a keyboard/mouse** — es un patrón estructural.

## 6. Análisis de keyboard y mouse

1. **¿Clases válidas del modelo?** Sí, ambas son clases reales de COCO-80 (ids 64 y 66), confirmado por `model.names`.
2. **¿Aparecen en el dataset Web3D?** Sí, pero **solo en 1 de las 29 imágenes** (la oficina), con confianza alta (`keyboard` 0.95, `mouse` 0.88).
3. **Utilidad contextual:** alta y específica — son los objetos más discriminativos posibles para "oficina" (nadie tiene teclado y ratón de escritorio en una sala de estar).
4. **¿Riesgo de falso contexto?** Bajo — son objetos muy específicos de escritorio, sin ambigüedad de contexto significativa observada.
5. **Utilidad separada por tarea:**
   - Detección: ya funciona correctamente (no hay problema en Nivel 1).
   - Clasificación de escena: **alta**, confirmada por el experimento (sección 11).
   - Narrativa: baja/nula — mencionar "hay un teclado" no aporta información de navegación egocéntrica relevante.
   - Navegación física: **ninguna** — un teclado sobre un escritorio no es un obstáculo de paso, consistente con por qué el diseño original (orientado a navegación) los excluyó deliberadamente.

**Conclusión de esta sección:** keyboard/mouse son útiles específicamente para clasificación de escena, no para navegación ni for narrativa — confirma que se trata de un desajuste de propósito, no de un descuido.

## 7. Otras clases potencialmente relevantes (no solo keyboard/mouse)

Con base en la frecuencia real en el dataset Web3D (sección 8) y el cruce de la sección 5, las clases excluidas con mayor justificación potencial, en orden de evidencia:

| Clase | Frecuencia en dataset Web3D | Categoría heurística que beneficiaría | Prioridad |
|---|---|---|---|
| **book** | **10/29 imágenes (34%)** | oficina, tienda (biblioteca no existe como categoría) | **Alta** — mayor evidencia real del dataset, aunque el experimento (sección 11) mostró que no basta por sí sola |
| **bowl** | 7/29 (24%) | comedor, cocina | Media |
| **cup** | 5/29 (17%) | comedor, cocina | Media |
| **keyboard** | 1/29 (3%) | oficina | Alta (específica, confirmada por experimento) |
| **mouse** | 1/29 (3%) | oficina | Alta (específica, confirmada por experimento) |
| microwave | 1/29 (3%) | cocina | Baja (evidencia escasa en este dataset) |
| traffic light, fire hydrant | 3/29, 1/29 | exterior (ya funciona sin ellas) | Baja (no se necesitan, "exterior" ya es 100% alcanzable) |

**No se propone agregar las 49 clases excluidas.** Solo `book`, `bowl`, `cup`, `keyboard`, `mouse` tienen evidencia real de aparición en el dataset del proyecto y relación directa con una categoría heurística existente.

## 8. Evidencia del dataset Web3D

Extraída de `evaluation/results/checkpoint_comparison/results.json` (detecciones YOLO reales ya generadas, modelo COCO, sin filtro de producción). Conteo completo por clase disponible en el log de ejecución de esta fase; resumen en la tabla de la sección 7.

## 9. Evidencia documental — GAP parcialmente cerrado

**No es un gap total.** Se encontró evidencia documental explícita:

- **Actividad `A13`** (Desarrollo de Actividades): *"Conjunto de clases de navegación (_NAV_CLASSES), limitando las detecciones retornadas a las clases relevantes para la accesibilidad... El conjunto _NAV_CLASSES define las clases cuyas detecciones son relevantes para la navegación del usuario."*
- **Documento principal de la tesis** (texto ya extraído en fases anteriores): incluye una tabla de pipeline que documenta *"Filtro por clase NAV | yolo_service.py | Descarta clases no pertenecientes a _NAV_CLASSES"* como paso 5 de 8, y la Tabla 2/9 de umbrales por clase, con columna **"Categoría"** cuyos valores son *obstáculo, Salida, peligro, Informativo, pequeño* — el mismo vocabulario de `OBJECT_TAXONOMY` en `spatial_analyzer.py`.

**Lo que SÍ es un gap real:** ningún documento del proyecto (ni A13, ni el documento principal, ni ninguna actividad revisada) discute o justifica el efecto de `_NAV_CLASSES` sobre `scene_classifier.py` — porque, por cronología de commits (ya establecida en fases previas), `_NAV_CLASSES` es anterior y `scene_classifier.py` es un módulo posterior que consume su salida sin que exista documentación que haya reevaluado la idoneidad del filtro para este nuevo consumidor.

## 10. Evidencia de Trello

Ninguna tarjeta (T09, T10, T12, T17, T18, T20, T21, T22, T24, T26, T27, T28, ni ninguna otra) menciona `_NAV_CLASSES`, filtrado de clases, ni la decisión de incluir/excluir `keyboard`/`mouse`/`book`. La única coincidencia de búsqueda por palabra clave fue T23 ("accesibilidad mediante teclado y lector de pantalla"), que se refiere a **accesibilidad de teclado del frontend** (navegación por Tab/lector de pantalla), un concepto completamente distinto — **no es evidencia relacionada, se descarta explícitamente para no confundir conceptos homónimos**. **GAP de trazabilidad confirmado**: no existe ninguna tarjeta que documente esta decisión de diseño ni su reutilización posterior.

## 11. Experimento aislado

Se construyó un script aislado (`scripts/evaluation/test_nav_classes_experiment.py`) que **no modifica `yolo_service.py`**: vuelve a correr YOLO real (conf=0.15, igual que producción) sobre 5 imágenes y aplica DOS filtros en el propio script — el actual (`_NAV_CLASSES`) y uno hipotético (`_NAV_CLASSES ∪ {book, keyboard, mouse}`), después llamando al `analyze_spatial()`/`classify_scene()` reales de producción sobre cada resultado.

| Imagen | A (actual) | B (hipótesis) | ¿Cambió? |
|---|---|---|---|
| oficina | sala de estar (media) — **incorrecto** | **oficina (alta) — correcto** | **Sí, se corrigió** |
| biblioteca | comedor (alta) — incorrecto | comedor (alta) — **sigue incorrecto** | No cambió |
| studio2 (ambiguo) | espacio interior (baja) | espacio interior (baja) | No cambió |
| living room (control, ya correcto) | sala de estar (alta) — correcto | sala de estar (alta) — correcto | **Sin regresión** |
| universidad | espacio interior (baja) | espacio interior (baja) | No cambió |

**Interpretación honesta (Paso 8):** el cambio **sí funciona** para el caso que lo motivó (oficina) y **no introduce regresión** en un caso de control ya correcto (living room) — pero **no generaliza** a biblioteca, studio2 ni universidad. Agregar `book` por sí solo no es suficiente para biblioteca porque "comedor" (mesa de comedor + silla) sigue teniendo una coincidencia de keywords más fuerte que cualquier cosa que "libro" pueda aportar por sí solo al conteo heurístico/LLM actual — el problema de fondo de biblioteca no es de vocabulario de detección, es de ausencia de categoría, ya documentado en las Fases 6-7.

## 12. Riesgos de modificar `_NAV_CLASSES`

- **Riesgo bajo de falsos positivos nuevos** para `keyboard`/`mouse`/`book` específicamente (son clases con semántica poco ambigua), pero no se probó en un conjunto grande — solo 5 imágenes.
- **Riesgo de generar una falsa sensación de "problema resuelto"**: agregar 2-3 clases y ver mejorar 1 de 5 casos podría interpretarse erróneamente como una solución general, cuando el experimento demuestra que no lo es.
- **Riesgo de mezclar responsabilidades**: si se amplía `_NAV_CLASSES` para servir mejor a `scene_classifier.py`, se corre el riesgo de que futuras decisiones de navegación (Nivel 1-2, seguridad física) se contaminen con criterios de clasificación de escena (Nivel 3) dentro del mismo filtro — exactamente el problema conceptual ya diagnosticado.
- **No se identificó ningún riesgo de regresión sobre TTS, análisis espacial o formato de respuesta** — el experimento reutiliza el mismo pipeline real sin tocar esos módulos.

## 13. Clasificación final de H41

### **H41 requiere rediseño metodológico**

Evidencia que lo sustenta:
1. `_NAV_CLASSES` tiene un propósito documentado, explícito y coherente (navegación/accesibilidad física), confirmado en la tesis y en A13 — **no es un descuido**.
2. `scene_classifier.py` reutiliza esa misma lista sin que exista ninguna decisión ni documento que haya evaluado su idoneidad para ese propósito distinto.
3. El cruce sistemático (sección 5) muestra que el problema afecta a **múltiples categorías** (oficina 29%, cocina 33% de keywords alcanzables), no solo a la oficina.
4. El experimento aislado (sección 11) demuestra que un parche de 2-3 clases **corrige un caso pero no generaliza** — la causa raíz de al menos el caso biblioteca es la ausencia de categoría, no el vocabulario de detección.

**No se descarta que, además del rediseño, un cambio táctico acotado (agregar `book`, `keyboard`, `mouse`, `bowl`, `cup` específicamente, con evidencia real del dataset) sea una mejora válida de corto plazo** — pero no debe presentarse ni implementarse como "la solución a H41", sino como una mejora parcial dentro de un rediseño mayor que también debe abordar las categorías faltantes (ya identificadas en la Fase 6-7).

## 14. Cambio propuesto, SIN implementarlo

**No se aplica ningún cambio.** Si en una fase futura se aprobara actuar:

| Opción | Alcance | Resuelve |
|---|---|---|
| Táctica mínima | Agregar `book`, `keyboard`, `mouse` a `_NAV_CLASSES` (sin tocar `_CLASS_MIN_CONF`, usarían el umbral del cliente por defecto) | Corrige oficina; NO corrige biblioteca por sí sola (confirmado por experimento) |
| Táctica ampliada | Además, agregar `bowl`, `cup` (evidencia moderada en dataset) | Podría reforzar comedor/cocina; no probado en esta fase |
| Estructural (recomendada para evaluar después) | Separar conceptualmente el filtro de navegación (Nivel 2, seguridad física) del conjunto de evidencia que recibe `scene_classifier.py` (Nivel 3) — dos listas con propósitos distintos en vez de una compartida | Ataca la causa raíz conceptual, no solo el síntoma |

## 15. Pruebas necesarias para una eventual Fase 10

1. Repetir el experimento de la sección 11 sobre las 29 imágenes completas, no solo 5, para medir generalización real.
2. Probar la opción "táctica ampliada" (agregar `bowl`/`cup`) sobre las escenas de comedor/cocina disponibles.
3. Prueba de regresión completa sobre `test_images/` si se decide aplicar cualquier cambio a `_NAV_CLASSES`.
4. Verificar si agregar estas clases afecta negativamente el análisis espacial/espacio libre (`free_space_analyzer.py`) al introducir más "objetos pequeños" en la escena — no evaluado en esta fase.
5. Decidir explícitamente, antes de tocar código, si se opta por la vía táctica o la estructural.

## 16. Limitaciones

- El experimento aislado usó solo 5 de las 29 imágenes (3 casos de interés + 1 control + 1 adicional).
- No se probó el efecto de agregar `bowl`/`cup` (solo se investigó su frecuencia, no se ejecutó el experimento con ellas).
- No se evaluó el impacto de estas clases en `free_space_analyzer.py`/`risk_engine.py` (decisión de movimiento), solo en clasificación de escena.
- La evidencia de frecuencia en el dataset (sección 8) proviene de un conjunto de 29 imágenes curado para este proyecto, no es una muestra representativa de todos los escenarios Web3D posibles.

## Integridad de producción

`git status` antes y después de esta fase es idéntico: `D test_images/cap1.png` (preexistente, ajeno) + `evaluation/`, `scripts/compare_checkpoints.py`, `scripts/evaluation/` sin trackear. `.env` no fue tocado (sigue con el único cambio de la Fase 8). Ningún archivo de `app/` fue modificado. Sin commits, sin pushes.

**Archivos nuevos de esta fase:**
- `scripts/evaluation/test_nav_classes_experiment.py`
- `evaluation/results/phase9/README.md`
- `evaluation/results/phase9/nav_classes_experiment.json`
