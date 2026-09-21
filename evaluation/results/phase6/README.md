# Fase 6 — Evaluación cuantitativa del detector y robustez de interpretación de escenarios

**Estado: exploratoria/diagnóstica. NO es resultado formal del Capítulo 5. NO se modificó producción.**

## Objetivo

Dos problemas relacionados:
1. Evaluar cuantitativamente COCO-80 vs. Objects365 sobre un subconjunto controlado con ground truth propio.
2. Investigar la robustez del mecanismo actual de interpretación de escenarios (`scene_classifier.py`), específicamente el problema reportado "biblioteca → sala de estar / oficina".

## Dataset

`evaluation/images/web3d/` — 29 imágenes (verificadas, ninguna eliminada ni añadida). Subconjunto de 10 imágenes seleccionado para reproducción de escenario (las 5 priorizadas por el usuario + 5 adicionales para probar generalización/casos ambiguos/sin categoría). Subconjunto de 5 imágenes (las priorizadas) para ground truth de objetos y métricas de detección.

## Modelos

- `yolo26s.pt` (COCO-80, producción)
- `yolo26s-objv1-150.pt` (Objects365, 365 clases, ya descargado en Fase 5)

## Configuración

Idéntica a la Fase 5: `imgsz=1280`, `iou=0.45`, `device=cpu`, `augment=False`, `conf=0.15` único (sin post-filtro por clase). Las detecciones reutilizan `evaluation/results/checkpoint_comparison/results.json` de la Fase 5 cuando corresponde (no se re-ejecutó YOLO innecesariamente sobre las mismas 29 imágenes).

## Ground truth

`ground_truth.json` — construido por inspección visual directa (no derivado de predicciones de ningún modelo):
- **Escenarios** (10 imágenes): `expected_scene` + certeza + notas. Incluye un caso marcado explícitamente como "no aplica" (imagen promocional de un asset pack, no es una escena navegable única) en vez de forzar una categoría.
- **Objetos relevantes** (5 imágenes priorizadas): lista de clases relevantes presentes, registradas por separado para el vocabulario COCO y el de Objects365 (nunca se convierte artificialmente `tv` en `monitor`).

## Métricas del detector (`detector_metrics.json`)

Calculadas a **nivel de presencia de clase por imagen**, no a nivel de bounding box/IoU por instancia (ver limitación abajo).

| Modelo | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| yolo26s (COCO) | 22 | 5 | 0 | 0.815 | 1.0 | 0.898 |
| yolo26s-objv1-150 (Objects365) | 29 | 10 | 0 | 0.744 | 1.0 | 0.853 |

**Lectura honesta de estos números:** el Recall=1.0 es parcialmente un artefacto de cómo se construyó el ground truth (se priorizaron objetos visualmente evidentes, correlacionados con objetos fáciles de detectar) — ver `analysis.md`. Además, se detectó una omisión real en el ground truth de la imagen de biblioteca (`backpack` y `couch` sí están presentes pero no se anotaron), que infla el conteo de falsos positivos de Objects365 en esa imagen específica. Estos números **no deben citarse en el Capítulo 5 sin antes corregir esa omisión y, si se desea mayor rigor, añadir bounding boxes reales para IoU**.

## Escenarios (`scene_evaluation.json`)

- **Hallazgo crítico no buscado (H40):** el LLM (Groq, modelo `llama-3.3-70b-versatile`) devuelve error 404 en el 100% de las llamadas reales realizadas en esta fase — tanto para clasificación de escenario como para generación de narrativa. El sistema está funcionando **enteramente con las rutas de respaldo (heurística + plantilla manual)**, no con el LLM, contrario al diseño documentado en el propio código.
- **Reproducción exacta del problema biblioteca:** aritmética completa del algoritmo de conteo de keywords, mostrando que el resultado depende de coincidencias genéricas (persona/botella/libro/mochila → "tienda") y, en el caso de Objects365, de un empate resuelto por **orden de inserción del diccionario** (`oficina` antes que `tienda`), no por ninguna lógica de desempate diseñada.
- **Tasa de acierto en el subconjunto de 9 imágenes evaluables:** COCO 3/9 (33%), Objects365 2/9 (22%) — muestra pequeña, no representativa, solo para diagnóstico del mecanismo, **no para reportarse como precisión general**.
- **El sistema siempre fuerza una categoría** de un conjunto cerrado de 10 (o "espacio interior"); nunca retorna `unknown`/`ambiguous`. La única mitigación existente es que `confidence="baja"` suprime el texto de escenario en la narrativa, pero el campo JSON `escenario.tipo` siempre contiene un valor forzado.
- **7 estrategias evaluadas (A-G)** sin implementar ninguna — ver `scene_evaluation.json`. La evidencia recogida sugiere que evidencia ponderada por especificidad (C) + co-ocurrencia (D) + estado `unknown`/`ambiguous` (G) atacarían directamente las causas identificadas, pero esto es una hipótesis a validar, no una decisión tomada.
- **LLM como "parche":** no se recomienda porque ya existe en el código como estrategia primaria — lo que hace falta es corregir H40 para que ese camino ya diseñado vuelva a funcionar, y solo entonces evaluar si resuelve el problema mejor que la heurística.

## Regresión sobre `test_images/` (`regression_results.json`)

12 imágenes del Capítulo 3, misma configuración. Dos hallazgos que requieren seguimiento antes de cualquier integración:
1. `08_mercado_productos.jpg`: COCO detecta 2 `person`, Objects365 detecta 0 — diferencia no explicable por vocabulario (la clase es idéntica en ambos).
2. `10_escena_compleja.jpg`: caída de 25 a 8 detecciones vehiculares equivalentes, solo parcialmente explicable por una taxonomía más granular en Objects365.

## Trazabilidad Trello

Cartas relevantes identificadas: T09 (métricas de detección respaldables), T10 (configuración de YOLO26s), T12 (justificación de selección del modelo), T17/T18 (pruebas funcionales), T20 (cálculo espacial), T21 (fidelidad de narrativas), T22/T24 (evaluación con usuarios), T26/T27/T28 (Capítulo 5). **Ninguna carta de Trello aborda explícitamente "robustez de la clasificación de escenario/ambiente"** como tarea propia — es un gap de trazabilidad detectado en esta fase, no cubierto hoy por ninguna tarjeta. Trello se usa aquí solo como trazabilidad de planificación, nunca como evidencia de ejecución.

## Limitaciones

1. Métricas de detección a nivel de presencia de clase, no de bounding box/IoU por instancia.
2. Ground truth de objetos con al menos una omisión conocida (backpack/couch en biblioteca), documentada, no corregida retroactivamente.
3. Muestra de 9-10 imágenes para escenario — no aleatoria, no representativa, solo diagnóstica.
4. Hallazgo H40 (LLM roto) impide evaluar en esta fase cómo se comportaría el camino LLM diseñado originalmente.
5. Regresión sobre `test_images/` es exploratoria; los dos hallazgos de posible pérdida de sensibilidad (person, vehículos) no fueron investigados en profundidad.
6. No se calculó mAP ni se hizo barrido de thresholds en esta fase.

## Decisiones pendientes (no tomadas en esta fase)

- Corregir `GROQ_MODEL` (H40) — requiere verificar qué modelo está disponible actualmente en la cuenta de Groq del proyecto.
- Investigar la pérdida de detecciones de `person` y vehículos con Objects365 en `test_images/` antes de considerar cualquier integración.
- Decidir si se amplía el ground truth (bounding boxes reales, más imágenes, corrección de la omisión de biblioteca) antes de usar estas métricas en el Capítulo 5.
- Decidir si se investiga una estrategia de escenario ponderada + `unknown`/`ambiguous` (no implementado, solo evaluado conceptualmente).

## Integridad de producción

Ver verificación de `git status` en el informe final entregado en chat. Ningún archivo de `app/`, `requirements.txt` ni `.env` fue modificado en esta fase.
