# Fase 6 — Análisis

## Separación de niveles (Sección 17)

Aplicado al caso biblioteca (el más ilustrativo):

| Nivel | Contenido real observado | ¿Dónde aparece el error? |
|---|---|---|
| 1. RAW DETECTIONS | YOLO (ambos modelos) devolvió candidatos razonables: person, laptop, book, dining table, chair, potted plant, bottle, backpack, couch, vase (COCO); más lamp, cabinet/shelf, trolley, head phone, tv (Objects365, este último un falso positivo espurio sobre un laptop, ya documentado en Fase 5) | Sin error relevante aquí — el detector identificó correctamente los objetos físicos principales |
| 2. FILTERED DETECTIONS | En esta prueba no se aplicó el post-filtro `_CLASS_MIN_CONF` (ver Fase 5/6, conf único 0.15 para no favorecer a ningún modelo) | No evaluado en esta fase con el filtrado real de producción — pendiente si se decide investigar más |
| 3. SCENE INTERPRETATION | `classify_scene()` heurístico produjo `tienda` (COCO) / `oficina` (Objects365) — **ambos incorrectos** | **Aquí está el error principal**, con causa mecánica exacta documentada en `scene_evaluation.json` (conteo de keywords genéricos + desempate por orden de inserción del diccionario) |
| 4. SPATIAL INTERPRETATION | No evaluado en profundidad en esta fase (no es la causa del problema de escenario — `analyze_spatial()` no interviene en `classify_scene()`) | Sin evidencia de error en este nivel para este caso |
| 5. EGOCENTRIC NARRATIVE | Al estar actualmente en modo fallback manual (H40: LLM roto), la narrativa no incorpora razonamiento contextual adicional que pudiera corregir el escenario | El error de escenario se propagaría tal cual a la narrativa si `confidence` no fuera `baja` (en los casos `alta`/`media` de esta prueba, sí se propagaría) |

**Conclusión de separación de niveles:** el error del caso biblioteca es inequívocamente del **Nivel 3 (interpretación de escenario)**, no del detector (Nivel 1) ni del análisis espacial (Nivel 4). No se le atribuye al modelo de detección.

## Hallazgo crítico no buscado: H40 (LLM inactivo por modelo deprecado)

Ver detalle completo en `scene_evaluation.json`. Resumen: `GROQ_MODEL=llama-3.3-70b-versatile` devuelve error 404 (`model_not_found`) en el 100% de las llamadas reales realizadas durante esta fase, tanto para `classify_scene()` como para `generate_description()`. `is_llm_active()` reporta `True` de forma engañosa (solo valida que el cliente se pueda construir). **Esto significa que, en el estado actual del entorno, tanto la clasificación de escenario como la narrativa generada dependen exclusivamente de las rutas de respaldo (heurística y plantilla manual), no del LLM**, contradiciendo la premisa de diseño documentada en el propio código (LLM como estrategia primaria, heurística como respaldo).

## Regresión sobre `test_images/` — hallazgo que requiere seguimiento

Dos observaciones no triviales, más allá del recuento agregado:

1. **`08_mercado_productos.jpg`: COCO detecta 2 `person`, Objects365 detecta 0 `person`.** `person` existe de forma idéntica en ambos vocabularios (misma clase 0 en ambos), por lo que esta diferencia **no se explica por vocabulario** — es una diferencia real de comportamiento del checkpoint Objects365 en esta imagen específica, y debe investigarse antes de cualquier consideración de integración.
2. **`10_escena_compleja.jpg`: COCO detecta 24 `car` + 1 `truck` (25 vehículos); Objects365 detecta solo 6 `car` + 2 `van` (8 vehículos-relacionados).** Parte de la diferencia es explicable por una taxonomía más granular en Objects365 (`car`/`van`/`suv`/`truck`/`pickup truck` son clases separadas, mientras COCO solo tiene `car`/`truck`), pero incluso sumando las categorías relacionadas (8 vs. 25) la diferencia es demasiado grande para explicarse solo por redistribución de clases — sugiere una posible pérdida real de sensibilidad en escenas con muchos vehículos pequeños/lejanos.

Ninguna de estas dos observaciones fue investigada más a fondo en esta fase (fuera del alcance de "prueba de regresión", que es exploratoria); ambas quedan documentadas como **pendientes concretos antes de cualquier integración**.

## Limitación reconocida del ground truth de objetos (autocrítica)

Al revisar los resultados de `detector_metrics.json`, se detectó que el ground truth de la imagen de biblioteca **no incluyó `backpack` ni `couch`** como objetos relevantes presentes, pese a que ambos son visibles en la imagen (mochilas bajo las mesas de lectura, un sofá/sillón verde al fondo derecho) — un error de omisión en la anotación, no un falso positivo real del detector. Esto infla artificialmente el conteo de falsos positivos de esa imagen en `detector_metrics.json` para las clases `backpack` y `couch`. Se documenta aquí de forma transparente en lugar de corregir el ground truth retroactivamente sin dejar rastro; una iteración futura del ground truth debería corregir esta omisión antes de usar estas métricas de forma definitiva.

**Recall=1.0 en `detector_metrics.json`:** debe leerse con la misma cautela — el ground truth se construyó priorizando objetos visualmente evidentes, lo cual está correlacionado con objetos que un detector maduro detecta fácilmente. Un recall perfecto en esta prueba no debe interpretarse como "el detector nunca pierde objetos relevantes" en general, sino como un artefacto parcial del método de construcción del ground truth sobre esta muestra pequeña.
