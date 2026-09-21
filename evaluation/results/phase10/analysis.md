# Fase 10 — Análisis detallado (segunda revisión)

**Este archivo es complementario de `README.md`. Donde exista cualquier diferencia de matiz, `README.md` (secciones 10-11) contiene la formulación más reciente y correcta — este archivo se actualizó para ser consistente con ella.**

## Biblioteca — HECHO / RESULTADO / INTERPRETACIÓN / LIMITACIÓN

- **HECHO** (verificado en `all_variants_raw.json`): la única clase nueva que se incorpora entre A y B/C/D para la imagen de biblioteca es `book`. Ninguna de las otras 10 clases de la lista D (`keyboard, mouse, bowl, cup, fork, microwave, oven, toaster, remote, toothbrush`) aparece en esta imagen.
- **RESULTADO:** la clasificación se mantuvo en `comedor` en las 4 variantes.
- **INTERPRETACIÓN:** el resultado es compatible con que la ausencia de una categoría "biblioteca" explícita (en `_CONTEXT_GROUPS` y en el prompt del LLM) sea una limitación relevante del sistema actual.
- **LIMITACIÓN:** no se aisló si el resultado se debe a esa ausencia de categoría, al peso relativo de `dining table`+`chair` (ya presentes antes del experimento y semánticamente asociados a "comedor"), o a ambos factores combinados. **No se afirma "causa raíz confirmada."**
- **TRABAJO FUTURO:** una prueba de control con una categoría "biblioteca" temporal (no implementada en producción) permitiría distinguir estas causas.

## Oficina — HECHO / RESULTADO / INTERPRETACIÓN / LIMITACIÓN

- **HECHO:** `keyboard` y `mouse` se incorporan siempre juntos, tanto en la oficina Web3D como en `test_images/03_escritorio_objetos.jpg` — ninguna variante los aisló entre sí.
- **RESULTADO:** la clasificación cambió de incorrecta a "oficina" en ambas imágenes, en las variantes B/C/D.
- **INTERPRETACIÓN:** el patrón es semánticamente razonable (objetos de escritorio → oficina) y se repite en un segundo conjunto de imágenes — evidencia exploratoria de comportamiento consistente en 2 casos.
- **LIMITACIÓN:** no puede determinarse si `keyboard` sola, `mouse` sola, o ambas juntas son necesarias para el cambio. Tampoco se descarta la variabilidad del LLM (sección 13 de `README.md`) como factor que contribuye al resultado observado en una ejecución única.
- **TRABAJO FUTURO:** ablación `A+keyboard` / `A+mouse` por separado, con repeticiones.

## Cocina/comedor (`test_images/04_cocina_utensilios.jpg`) — el caso de mayor aislamiento del experimento, con su límite explícito

- **HECHO:** `bowl` es la única clase nueva incorporada entre A (`[dining table]`) y D (`[bowl, dining table]`) para esta imagen — verificado directamente en `test_images_regression.json`.
- **RESULTADO:** la clasificación cambió de "espacio interior" a "comedor".
- **INTERPRETACIÓN:** esta es la diferencia de evidencia más directamente asociada a un cambio de clasificación en todo el experimento, porque es la única instancia donde solo una clase nueva está presente.
- **LIMITACIÓN — explícita y no debe omitirse:** esto es una **asociación de una sola observación**, no una causalidad demostrada. El experimento ejecutó una sola vez esta combinación; no hay repetición que permita descartar que la misma entrada, ejecutada de nuevo, produjera un resultado distinto por variabilidad del LLM. **No se afirma que "bowl causó el cambio."** Se afirma únicamente que `bowl` fue la única clase nueva presente cuando el cambio ocurrió.
- **TRABAJO FUTURO:** repetir esta combinación varias veces (mínimo 3-5 ejecuciones) para estimar si el resultado es estable, antes de considerar esta asociación como evidencia sólida.

## Impacto en navegación — resumen de la verificación de código

Ver `README.md` sección 9 para el detalle completo verificado línea por línea contra `app/services/spatial_analyzer.py` (`OBJECT_TAXONOMY`) y `app/services/free_space_analyzer.py` (`_NON_BLOCKING`). La conclusión se limita explícitamente a la implementación evaluada, no a cualquier implementación futura.

## No sobreajuste de la lista D

La regla de construcción (`COCO-80 ∩ keywords de _CONTEXT_GROUPS excluidas hoy por _NAV_CLASSES`) es reproducible y está documentada en la Fase 9, generada antes de este experimento según el orden de trabajo de la sesión. No existe verificación externa criptográfica de esa secuencia temporal — se declara así explícitamente en vez de presentarse como un hecho incuestionable. Como evidencia indirecta adicional: la lista no resuelve biblioteca, universidad, baño ni sala de espera, lo cual es inconsistente con la hipótesis de que se haya ajustado post-hoc para maximizar aciertos en este dataset.
