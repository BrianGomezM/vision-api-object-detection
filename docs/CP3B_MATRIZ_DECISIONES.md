# CHECKPOINT 3B — Matriz de decisiones metodológicas

**Fecha:** 2026-09-26

**Estado:** PROPUESTA para que decida Brian. Nada de lo que contiene está aprobado ni ejecutado.

**Qué no se hizo al preparar este documento:**
- no se ejecutó YOLO, ni el LLM, ni el TTS;
- no se modificó código ni el Dataset 1;
- no se usó ningún resultado de detección sobre el Dataset 1, porque no existe.

**Relación con otros documentos:**
- sustituye a la tabla de decisiones de `docs/CP3B_PROTOCOLO_EVALUACION.md` (borrador v1);
- la §6 contiene la versión propuesta v2 del protocolo.

---

## Estado de las decisiones (actualizado en el checkpoint final pre-F4, 2026-09-27)

La configuración aprobada está congelada en `experimental_config.yaml` y la verifica `app.experiment.preflight()`.

| # | Decisión | Estado | Resolución / qué falta |
|---|---|---|---|
| 1 | Caja de referencia del IoU | **PENDIENTE** | Brian debe aprobar el GT visual modal (opción B). Hace falta código nuevo en el generador, **sin modificar las escenas** |
| 2 | Profundidad (3 condiciones frente a 4 categorías) | **PENDIENTE** | Aprobar la métrica (monotonía + tabla 3×4) |
| 3 | Dispositivo | **CERRADA: cpu** | Evidencia E11/E12 (diferencias CPU/GPU ≤ 9·10⁻⁵, repeticiones idénticas). Se fuerza con `CUDA_VISIBLE_DEVICES=-1` y el preflight lo verifica |
| 4 | Umbral y regla de aceptación | **CERRADA: regla `min(class_min, umbral)`, umbral 0,35** | `docs/AUDITORIA_REGLA_UMBRAL_FASE10.md` §5. Los análisis de sensibilidad (regla alternativa; 0,3/0,5/0,7) siguen **PENDIENTES de aprobación** |
| 4b | Preprocesamiento de imagen | **CERRADA: ruta del producto** (`core.pipeline.run` → `resize_image` → `run_yolo` → letterbox de Ultralytics) | `experimental_config.yaml` → `imagen`. La fase 10 (imagen original) queda documentada como configuración distinta |
| 4c | Modelo y pesos | **CERRADA: `yolo26s.pt`, sha256 `646f8bc3…4a1b`** | El preflight detiene la ejecución si no coinciden; no hay descarga automática en F4 |
| 5 | Número de corridas | **PENDIENTE** | Propuesta: 3 corridas con igualdad por hash |
| 6 | OBJ-04 / C2-espejo | **PENDIENTE** | No generado |
| 7 | Definición de VP/FP/FN | **PENDIENTE** | Depende de la decisión 1 |
| 8 | Posición horizontal | **PENDIENTE** | — |
| 9 | Relaciones | **PENDIENTE** | — |
| 10 | Narrativa (codificación, k generaciones) | **PENDIENTE** | — |
| 11 | Evaluación con usuarios | **PENDIENTE** | — |
| 12 | Agregación e IC | **PENDIENTE** | — |
| 13 | Dataset 2 | **PENDIENTE** | Procedencia sin verificar |
| — | Regla `max()` como configuración oficial | **DESCARTADA** | Cambio de interfaz sin evaluación y nunca desplegado |
| — | Caja proyectada como referencia de IoU (1-A) | **DESCARTADA** (recomendación) | Contradice la convención del GT; se confirmará al aprobar la decisión 1 |

**Categorías de métricas que no deben mezclarse.** Cada una tiene su capa y su fuente:

| Categoría | Capa | Fuente | Decisiones |
|---|---|---|---|
| **Detección** (VP/FP/FN, IoU) | 4 | salida de `run_yolo` (y salida cruda como diagnóstico) | 1, 4, 7 |
| **Espacial** (columna, profundidad, sobre superficie) | 5 | `analyzed` del pipeline | 2, 8, 9 |
| **Narrativa** (fidelidad y extremo a extremo) | 6 | texto de la narrativa | 9, 10 |
| **Evaluación con usuarios** (tareas y valoraciones) | 7 | respuestas de los participantes | 11 |
| **Latencia** | — | tiempos (`tiempos_ms`, telemetría) | **no forma parte de las métricas del CP3B.** Si se reporta, será aparte y declarando que es CPU local, distinto del despliegue |

**Hallazgo que afecta a F4 (resuelto en el runner, no es una decisión metodológica).** La caché de escenario (10 s, con la lista de objetos como clave) haría que estímulos consecutivos con los mismos objetos (A1–A9: una silla) reutilizaran la respuesta del LLM. El runner debe llamar a `app.experiment.reset_request_state()` antes de cada estímulo, lo que equivale a una solicitud de producción que llega pasados 10 s.

## 0. Evidencia usada

Todo lo que sigue se obtuvo leyendo código, datos o documentos existentes.

| # | Hecho verificado | Fuente |
|---|---|---|
| E1 | El Dataset 1 tiene 18 escenas y 32 objetos de diseño. **Ninguna escena repite clase COCO.** | `stimuli/dataset1/manifest.yaml` |
| E2 | Solo 3 de los 32 objetos tienen oclusión: mesa C1 (O = 0,236), mesa C3 (0,203) y mesa D2 (0,132). Los otros 29 tienen O = 0. Ningún objeto sale del encuadre. | GT del generador (`occlusion`, `projected_bbox_2d_truncated`) |
| E3 | Al generar, el generador ya calculó un mapa de píxeles visibles por objeto con dos métodos independientes: el render de IDs de VTK (sin antialiasing) y el z-buffer de NumPy. Su concordancia mínima es IoU = 0,9995 en los 32 objetos. | GT `render_consistency.visible_iou_vtk_vs_zbuffer`; `generator/render.py`, `generator/raster.py` |
| E4 | El GT declara que `projected_bbox_2d` "is NOT the box a detector such as YOLO should output and must not be used as such". También declara que no aplica ninguna correspondencia 4→3 en profundidad. | GT `conventions` |
| E5 | Cajas más pequeñas: botella D2 de 8×31 px, botella B3 de 10×32 px y sillas lejanas de 27–36 × 63–71 px. | GT `projected_bbox_2d_px` |
| E6 | La sala solo tiene suelo, paredes y techo lisos: **no hay puerta, ventana ni otros elementos**. (El borrador v1 del CP3B mencionaba "ventanas y puerta": era un error.) | `config/environment.yaml` del generador |
| E7 | `run_yolo`: inferencia con conf = 0,15. Luego pasan dos filtros: (1) la lista blanca `_NAV_CLASSES` y (2) el **umbral efectivo = max(`_CLASS_MIN_CONF[clase]`, umbral de la petición)**. El umbral de la petición es 0,35 por defecto desde el commit `de2cff4` (19-05-2026), cuatro meses antes de que existiera el Dataset 1. | `app/services/yolo_service.py`, `app/routes/detect.py`, historial git |
| E8 | Las 7 clases del Dataset 1 están en `_NAV_CLASSES`. Con 0,35 y la regla `max`, el umbral efectivo es **0,35 para las 7**: los mínimos por clase (mesa 0,10; planta y botella 0,25; silla, sofá y persona 0,30) no tienen efecto. | E7 |
| E9 | **Discrepancia entre la tesis y el código.** La Tabla 9 de la tesis presenta los umbrales por clase como "la adaptación más determinante". En la configuración por defecto no actúan para ninguna clase del Dataset 1. | Tesis `_CORREGIDO.docx`, §3.5.3; E8 |
| E10 | En la fase 2A (41 imágenes que no son del Dataset 1), cambiar la regla de `max` a `min` modificó las detecciones en 22 de 41 imágenes. Añadió 50 detecciones con confianza entre 0,157 y 0,349; la clase más afectada fue `dining table` (24). La mesa aparece en 6 escenas del Dataset 1. | `evaluation/results/phase2a/detection/analysis_summary.txt` |
| E11 | En la fase 2A, las 3 repeticiones fueron idénticas en 41/41 imágenes tanto en CPU como en CUDA. Entre CPU y GPU, a 1280 px: las mismas clases en 41/41 imágenes, diferencia máxima de confianza 9·10⁻⁵ y diferencia máxima de caja 0,01 px. | `phase2a/detection/timing_resources.csv`, `raw_predictions.jsonl` (reanalizado, sin ejecutar YOLO) |
| E12 | Tiempo en CPU a 1280 px: mediana de 572 ms por imagen, es decir unos 11 s para 18 imágenes. | `timing_resources.csv` |
| E13 | La tesis (Tabla 3) define como métricas de detección: mAP, IoU y precisión. Durante la selección del modelo se usaron los umbrales 0,3, 0,5 y 0,7 como prueba de robustez. En el Capítulo 5 no se calculó IoU por falta de cajas anotadas. | Tesis, Tabla 3, Tabla 7 y §5.2 |
| E14 | La tesis registra "3 ejecuciones idénticas sobre la misma imagen" (detección y narrativa), con la salvedad de que la temperatura 0,1 "no garantiza determinismo". En la fase 10 se observó que la clasificación de escena varió entre fases con el mismo código. | Tesis, Tabla 5.x; `phase10/README.md` |
| E15 | La API calcula, entre objetos, solo la relación "objeto sobre superficie" (`_merge_surfaces`). Además fusiona duplicados (`_deduplicate`). La profundidad se deriva del área de la caja y de y2/yc, no de una distancia métrica. | `app/services/spatial_analyzer.py` |
| E16 | OBJ-03 pide una escena con obstáculos "al centro y a un lado" y el paso libre en el lado contrario. OBJ-04 pide "la segunda imagen de una secuencia (misma escena, objetos reubicados)", reproducida inmediatamente después de OBJ-03. | `app/catalog/catalog.yaml` (texto migrado sin cambios) |
| E17 | Escenas del Dataset 1 con la estructura de OBJ-03: **C2** (sofá al centro, planta a la derecha; izquierda libre) y **B3** (persona al centro, botella a la derecha; izquierda libre, aunque la botella es un obstáculo débil). **No existe ningún par "misma escena con objetos reubicados"** con esa estructura. Los pares del bloque A tienen un único objeto. | `manifest.yaml` |
| E18 | Dataset 2: ninguna de las 29 imágenes tiene procedencia y licencia verificadas. | `evaluation/metadata/inventario_dataset2_verificacion.csv` |

**Marcas usadas en las fichas:**
- **[ANTES]:** decidido antes de observar resultados, y verificable como tal por la fecha del commit del protocolo, anterior a cualquier ejecución.
- **[APROBAR]:** requiere la aprobación de Brian.

---

## 1. Matriz de decisiones

### Decisión 1 — Caja de referencia para la localización (IoU)

**1. Problema.** Para medir si YOLO localiza bien un objeto hace falta una caja de referencia que represente **lo mismo que el detector aprendió a producir**. YOLO se entrenó con COCO, cuyas cajas rodean la parte **visible** del objeto (caja modal). El GT geométrico 3D no es esa caja (E4).

**2–6. Alternativas**

| | A. Caja proyectada (amodal) | B. Caja visible derivada del mapa de IDs | C. Sin IoU |
|---|---|---|---|
| Qué es | Proyección de todos los vértices, sin tratar la oclusión (ya existe en el GT) | Rectángulo mínimo que contiene los píxeles donde el objeto es visible en la escena completa: capa nueva "GT visual" | Correspondencia solo por clase; no se mide la localización |
| Ventajas | Ya existe, con hash y congelada. Para 29 de 32 objetos coincide con la visible (E2). | Misma convención que COCO (modal). Se deriva de forma determinista de las mismas escenas congeladas, sin tocar las imágenes. Dos métodos independientes ya concuerdan (IoU ≥ 0,9995, E3). Permite reportar IoU, que la Tabla 3 de la tesis prometía (E13). | Simple. Sin ambigüedad en el Dataset 1, porque no hay clases repetidas (E1). Comparable con el método del Cap. 5 (presencia de clase). |
| Desventajas / riesgos | **Contradice la convención declarada en el propio GT (E4).** Usarla obligaría a reinterpretar el GT *a posteriori*, algo difícil de defender. En C1, C3 y D2 penaliza una detección correcta de la parte visible. | Es **GT nuevo**: exige una versión nueva del GT, generada y congelada **antes** de ejecutar YOLO. El mapa de IDs no usa antialiasing y la imagen RGB sí: diferencias de 1 px en los bordes. Las botellas (8–10 px de ancho, E5) tienen un IoU muy sensible: 1 px de desplazamiento equivale a unos −0,1 a −0,2 de IoU. | **Pierde la afirmación sobre localización.** Una detección de la clase correcta en el lugar equivocado contaría como acierto. No cumple la Tabla 3 de la tesis. |
| Reproducibilidad | Alta (ya existe) | Alta si se congela el script, el commit del generador y el hash del GT visual. Se puede comprobar re-renderizando la imagen RGB y verificando que su hash coincide con la congelada (prueba de que es la misma escena). | Máxima |
| Validez | Baja para los objetos ocluidos; construcción inconsistente con el detector | Alta: la referencia coincide con la tarea del detector | Válida solo para "presencia", no para "detección + localización" |

**7. Afirmaciones que permite:**
- **A:** "Las cajas coinciden con la proyección geométrica", pero en contra del propio GT.
- **B:** "YOLO localiza los objetos visibles con un IoU mediano de X (k/32 con IoU ≥ 0,5) frente a una referencia visible derivada geométricamente."
- **C:** "YOLO identifica la presencia de k/32 objetos."

**8. Afirmaciones que no permite:**
- **A:** afirmar que la caja es "correcta" en objetos ocluidos.
- **B:** afirmar que coincide con una anotación humana. Es una referencia geométrica, no una anotación manual.
- **C:** afirmar nada sobre localización ni sobre mAP.

**9. Recomendación: B para la localización + correspondencia por clase para la detección**, como dos métricas separadas:
- **Detección:** un objeto cuenta como detectado si hay una detección de su clase. Es inequívoco porque ninguna escena repite clase (E1).
- **Localización:** IoU de esa detección frente a la caja visible. Se reporta la distribución completa, la proporción k/n con IoU ≥ 0,5 y los objetos pequeños por separado.
- **A no se usa.**
- **C queda como métrica secundaria de presencia**, para comparar con el Cap. 5.

**Por qué:** es la única opción coherente con E4 y con la convención de COCO, y se puede construir sin modificar las 18 escenas.

**10. Qué se congela antes de ejecutar:**
- el script que genera el GT visual, su commit y el commit del generador (`ae85f90`, que no cambia);
- el campo nuevo `visible_bbox_2d_px` por objeto, en un archivo aparte `ground_truth_visual/<scene>.json` con su propio hash y `schema_version`;
- el método (mapa de IDs de VTK; el z-buffer como verificación, IoU ≥ 0,99);
- la comprobación de que el re-render RGB reproduce el hash congelado.

Todo esto se genera y se somete a commit **antes** de la primera ejecución de YOLO. **[APROBAR]**

**Ficha de defensa:**
- **Qué:** referencia visible (modal).
- **Por qué:** coincide con lo que YOLO aprendió (COCO) y el GT prohíbe usar la caja proyectada.
- **Cuándo:** [ANTES].
- **Evidencia:** E2–E5.
- **Reproducción:** script + commits + hashes.
- **Limitación:** la referencia es geométrica, no una anotación humana; el IoU de las botellas es muy sensible.

---

### Decisión 2 — Profundidad (3 condiciones de diseño frente a 4 categorías de la API)

**1. Problema.** El diseño manipula la distancia en 3 condiciones (near 3,02 m, medium 4,64 m y far 7,14 m; solo en el bloque A, n = 9). La API devuelve 4 categorías calculadas con el área y la posición vertical de la caja (E15). No existe ninguna correspondencia, y el GT prohíbe inventarla (E4).

**2–6. Alternativas: qué mide exactamente cada una**

| Alternativa | Qué mide | Ventajas | Riesgos | Reproducibilidad | Validez |
|---|---|---|---|---|---|
| **Tabla de contingencia 3×4** | La distribución conjunta completa: cuántos objetos de cada condición reciben cada categoría | No asume nada; muestra todo | No da una cifra resumen. Con n = 9 la tabla es muy dispersa. | Total | Alta (descriptiva) |
| **Monotonía por columna** (ordinal, a priori) | En cada posición horizontal (izquierda, centro, derecha) se comprueba si la categoría de la API se aleja al pasar de near a medium y a far. Resultado: k/3 tripletes estrictamente monótonos y k/3 débilmente monótonos. | Mide exactamente lo que importa: si la API **ordena** bien la distancia. No necesita correspondencia. | Solo 3 tripletes. No informa de la calibración absoluta. | Total | Alta para "ordenamiento" |
| **Kendall τ-b** (condición ordinal × categoría ordinal) | Asociación ordinal global, corrigiendo los empates (habrá muchos: 3 × 4 niveles) | Estándar y adecuado para empates | Con n = 9, el valor p no tiene potencia; el τ-b solo sirve como descriptor | Total | Media: resume lo mismo que la monotonía con menos transparencia |
| **Correspondencia predefinida** (por ejemplo, `muy_cerca`/`cerca` → near) | Una "exactitud" de profundidad | Da un porcentaje fácil de comunicar | Los umbrales de la API no se diseñaron para 3,02/4,64/7,14 m. **Cualquier correspondencia es arbitraria, y el jurado preguntará por qué esa.** Si se elige después de ver resultados, invalida la métrica. | Total si se fija antes | Baja: mide la correspondencia, no la API |
| **Efecto del tamaño a distancia fija** (bloques B–D, todos en *medium*) | Distribución de las categorías de la API para objetos a la **misma** distancia y de distinto tamaño (botella frente a sofá) | Revela si la API confunde tamaño con distancia (cabe esperarlo, porque usa el área, E15). Aprovecha 23 objetos. | Es una hipótesis diagnóstica, no una exactitud | Total | Alta como diagnóstico |

**7. Afirmación que permite (recomendada):** "En el bloque A, la categoría de profundidad de la API fue monótona con la distancia geométrica en k/3 posiciones horizontales (τ-b = X, descriptivo). A distancia constante, la categoría varió con el tamaño del objeto (tabla)."

**8. Afirmaciones que no permite:**
- "La API estima la profundidad con un X % de exactitud."
- Cualquier afirmación en metros.

**9. Recomendación:**
- **principal:** monotonía por columna + tabla 3×4;
- **secundario (descriptivo):** τ-b;
- **diagnóstico:** efecto del tamaño en B–D;
- **no usar** ninguna correspondencia 3↔4.

**10. Qué se congela:**
- el orden de cercanía de las categorías: `muy_cerca` > `cerca` > `medio` > `lejos`;
- las definiciones de monotonía estricta y débil;
- que τ-b es solo descriptivo;
- que no se usa correspondencia.

**[APROBAR]**

**Ficha de defensa:** se compara el **orden**, no la etiqueta. [ANTES]. Evidencia: E4 y E15. Limitación: n = 9 (3 tripletes); no permite afirmaciones métricas.

---

### Decisión 3 — Dispositivo de inferencia

**1. Problema.** `run_yolo` no fija el dispositivo: en el entorno local usa la GPU (GTX 1050) y en Docker o Azure, la CPU. La evaluación oficial necesita un único entorno declarado.

| | CPU | GPU (CUDA) |
|---|---|---|
| Disponibilidad | Cualquier máquina: jurado, Docker, Azure (el despliegue de la tesis es CPU) | Solo máquinas con NVIDIA + CUDA 12.6 |
| Determinismo | Repeticiones idénticas 41/41 (E11) | Repeticiones idénticas 41/41 (E11). cuDNN puede ser no determinista en otras versiones o GPU. |
| Diferencia con el otro dispositivo | — | Confianza: ≤ 9·10⁻⁵; cajas: ≤ 0,01 px; las mismas clases en 41/41 (E11) |
| Tiempo | ~0,57 s por imagen, ~11 s para el Dataset 1 (E12) | ~0,07 s por imagen |
| Posibilidad de repetir | Máxima | Depende del hardware |

**4. Riesgos:**
- **CPU:** ninguno relevante.
- **GPU:** repetir el experimento exige un hardware equivalente. Una detección con confianza a menos de 10⁻⁴ del umbral podría cambiar de lado.

**5–6. Impacto:** la diferencia práctica es despreciable (E11). La elección afecta a la **repetibilidad por terceros**, no a la validez.

**7. Afirmación que permite:** "Los resultados se obtuvieron en CPU (versiones X), en el mismo entorno de ejecución que el despliegue, y cualquiera puede reproducirlos sin GPU."

**8. Afirmación que no permite:** afirmar que los resultados son "idénticos en cualquier hardware". Solo se verificó el equipo propio.

**9. Recomendación: CPU oficial**, forzada **sin cambiar código** con `CUDA_VISIBLE_DEVICES=-1` en el proceso de evaluación. Se registra `torch.cuda.is_available() == False` en los metadatos de la ejecución. Una pasada en GPU se reporta solo como verificación cruzada, nunca como resultado principal.

**10. Qué se congela:**
- dispositivo = cpu;
- versiones de torch/torchvision/ultralytics/numpy/pillow (`requirements.lock.txt`);
- `torch.get_num_threads()`, que se registra;
- Python 3.13.15;
- SO y CPU, que se registran.

**[APROBAR]**

**Ficha de defensa:** CPU [ANTES]. Evidencia: E11 y E12. Reproducción: variable de entorno + lock. Limitación: la reproducción exacta requiere las mismas versiones de las bibliotecas.

---

### Decisión 4 — Umbral de confianza

**1. Problema.** Hay que fijar qué detecciones "cuentan" **sin mirar el Dataset 1**. Conviene distinguir cuatro niveles:

| Nivel | Valor actual | Naturaleza | ¿Se decide aquí? |
|---|---|---|---|
| Confianza interna de inferencia | 0,15 | Configuración interna del modelo, documentada en la Tabla 10 | No: se congela tal cual |
| IoU de NMS | 0,45 | Configuración interna, Tabla 10 | No: se congela |
| Lista blanca `_NAV_CLASSES` | 40 clases (31 alcanzables) | Filtro del sistema, documentado | No: se congela; incluye las 7 clases (E8) |
| **Criterio de aceptación** = regla y umbral de la petición | `max(mín_clase, 0,35)`, que da 0,35 efectivo (E8) | **Decisión experimental** | **Sí** |

**Actualización del 2026-09-27** (`docs/AUDITORIA_REGLA_UMBRAL_FASE10.md`):
- la tesis documenta **`min()`** de forma explícita (tabla de `run_yolo`, "umbral efectivo = min(...)");
- las fases 9 y 10 se ejecutaron con `min()`;
- el código pasó a `max()` el 21-09-2026 por un motivo de interfaz.

Por eso la alternativa "B" incluye además la opción **restaurar `min()`**, que es la recomendada en esa auditoría.

**Hallazgo que hay que resolver primero (E9).** La tesis describe el sistema con umbrales por clase "determinantes", pero el sistema tal como se ejecuta hoy aplica un 0,35 uniforme a las clases del Dataset 1. Según E10, la diferencia es material: afecta sobre todo a `dining table`.

**2–6. Alternativas** (todas definidas antes de que existiera el Dataset 1, así que ninguna es ajuste posterior):

| | A. Sistema tal como está desplegado | B. Sistema tal como lo documenta la Tabla 9 | C. Detector sin umbral | D. Ajustar el umbral con el Dataset 1 |
|---|---|---|---|---|
| Qué es | Petición a 0,35, regla `max` → 0,35 efectivo | Petición a ≤ 0,10, regla `max` → umbral efectivo = mínimo de cada clase (mesa 0,10; planta y botella 0,25; silla, sofá y persona 0,30; portátil 0,35). **No requiere cambiar código.** | Salida cruda ≥ 0,15 con curva de precisión y exhaustividad (recall) y AP50 | Buscar el mejor umbral |
| Ventajas | Evalúa lo que existe y lo que oirán los participantes. Umbral documentado desde el 19-05-2026. | Coincide con el texto de la tesis | Independiente del umbral; responde a la Tabla 3 (mAP) | — |
| Riesgos | La Tabla 9 de la tesis debe corregirse, porque describe un comportamiento que no está activo | El sistema desplegado no se comporta así: se evaluaría una configuración que nadie usa | AP con 32 objetos (≈ 1 instancia por clase y escena) es ruidoso; es un complemento, no la cifra principal | **Prohibido: sobreajuste al conjunto de evaluación** |
| Reproducibilidad | Total | Total | Total | — |
| Validez | Alta para "el sistema" | Alta para "el sistema de la tesis", **si** se cambia el valor por defecto | Alta para "el detector" | Nula |

**7. Afirmaciones que permite:**
- **A:** "Con la configuración operativa (umbral efectivo 0,35), el sistema…"
- **C:** "El detector alcanzó un AP50 de X frente a la referencia visible (n = 32; muestra pequeña)."

**8. Afirmaciones que no permite:**
- con A, afirmar que "los umbrales por clase mejoraron…" (no estaban activos);
- con C, afirmar que es comparable con el mAP de COCO.

**9. Recomendación:**
- **Principal: A** (el sistema tal como está desplegado), porque los estímulos para usuarios deben reflejar el sistema real;
- **corregir la Tabla 9 de la tesis** para que diga la regla real;
- **sensibilidad declarada de antemano:**
  - (i) B;
  - (ii) la rejilla 0,30 / 0,50 / 0,70, ya usada en la tesis para medir robustez (E13);
- **complemento a nivel de detector:** C, a partir de la salida cruda;
- **nunca D.**

Si Brian prefiere que el sistema oficial sea B, eso es un **cambio de configuración del sistema** (el valor por defecto de la petición). Debería decidirse ahora, documentarse con su propio commit y aplicarse también a los estímulos de usuario.

**10. Qué se congela:**
- conf interna 0,15; NMS 0,45; imgsz 1280; augment false;
- regla `max`; umbral de la petición 0,35;
- `_CLASS_MIN_CONF` y `_NAV_CLASSES` (hash de `yolo_service.py`);
- la lista de análisis de sensibilidad;
- que la salida cruda (≥ 0,15, antes de los filtros) **también se guarda** para C y para diagnosticar falsos positivos.

**[APROBAR — decisión de configuración experimental]**

**Ficha de defensa:**
- **Qué:** el valor por defecto documentado.
- **Por qué:** es el sistema que usarán los participantes.
- **Cuándo:** el umbral es anterior al Dataset 1 (commit del 19-05-2026).
- **Evidencia:** E7–E10.
- **Limitación:** los umbrales por clase no están activos en esta configuración; la tesis debe decirlo.

---

### Decisión 5 — Número de corridas

**1. Problema.** ¿Basta una ejecución por imagen?

**Evidencia:** con configuración y dispositivo fijos, YOLO fue determinista en 41/41 imágenes con 3 repeticiones (E11). El análisis espacial (capa 5) es una función determinista de las detecciones. El LLM **no** es determinista (E14).

| | 1 corrida | 1 oficial + 2 de verificación | N corridas promediadas |
|---|---|---|---|
| Ventajas | Suficiente si el sistema es determinista | **Demuestra** el determinismo en este conjunto; coste de unos 30 s en CPU | — |
| Riesgos | El determinismo queda supuesto, no demostrado | Ninguno | Promediar salidas idénticas no aporta nada y sugiere una variabilidad que no existe |
| Reproducibilidad | Media | Alta: se compara el sha256 del JSON canónico de las tres | — |

**7. Afirmación que permite (recomendada):** "Tres ejecuciones produjeron salidas idénticas (hash), así que el resultado de la capa 4 y la capa 5 no depende de la corrida."

**8. Afirmación que no permite:** sostener eso mismo para el LLM.

**9. Recomendación:**
- **capas 4 y 5:** 1 corrida oficial + 2 de verificación. Si algún hash difiere, se **detiene** la ejecución y se reporta; no se promedia ni se elige.
- **LLM:** ver la Decisión 10.

**10. Qué se congela:** el número de corridas (3), el criterio (igualdad del hash del JSON canónico: claves ordenadas y floats redondeados a 6 decimales) y la acción ante una diferencia (detener). **[APROBAR]**

---

### Decisión 6 — OBJ-04 (actualización de contexto en secuencia)

**1. Problema.**
- **Qué exige OBJ-04 (E16):** una **segunda imagen de la misma escena de OBJ-03 con los objetos reubicados**, de modo que la dirección libre cambie. Se reproduce justo después de OBJ-03.
- **Qué ofrece el Dataset 1 (E17):** ningún par así. **No puede afirmarse que las 18 escenas cubran OBJ-04.**

**2–6. Alternativas:**

| | a. Extensión versionada "Dataset 1-S" (secuencias) | b. Redefinir OBJ-04 con pares existentes del bloque A | c. Eliminar OBJ-04 |
|---|---|---|---|
| Qué es | Generar con el **mismo generador y la misma configuración congelada** (sin cambiar ningún hash de `configs_sha256`) una escena nueva, declarada antes de generarla. Por ejemplo, **C2-espejo**: planta a la izquierda y sofá "al lado" de ella hacia el centro, con la derecha libre. Va en un directorio y un manifest aparte (`stimuli/dataset1s/`). **Las 18 escenas no se tocan.** | Usar, por ejemplo, A4 → A1 (una silla que pasa del centro a la izquierda) | Retirar la tarea |
| Ventajas | Cumple la tarea tal como está definida. Mismos criterios geométricos verificados. | No hay estímulos nuevos | No hay estímulos nuevos |
| Riesgos | Hay un paso de generación nuevo. Si la escena no cumple los criterios, necesita alternativas declaradas de antemano (como se hizo con D3). Necesita su propio checkpoint de congelación. | **Cambia la tarea:** un solo obstáculo, sin la estructura de OBJ-03. Rompe la adyacencia con OBJ-03, o exige redefinir también OBJ-03. | Se pierde la medida de actualización de contexto; limitación declarada |
| Reproducibilidad | Alta (mismo pipeline, hashes) | Alta | — |
| Validez | Alta | Media: mide algo más simple | — |

**7. Afirmaciones que permite:**
- **a:** "El participante detectó el cambio de disposición en k/n casos."
- **b:** solo "detectó el cambio de posición de un obstáculo único".

**8. Afirmaciones que no permite:**
- **b:** afirmar nada sobre escenas con varios obstáculos.
- **c:** afirmar nada sobre la actualización de contexto.

**9. Recomendación: a.** Una sola escena nueva (C2-espejo) con 2–3 alternativas ordenadas y declaradas antes de generar, igual que en D3. Se genera en una fase aparte **después** de aprobar este CP3B y **antes** de F4, con su propio checkpoint y sin modificar el Dataset 1.

Si Brian prefiere no generar nada, b es aceptable **solo** si también se reescribe el texto de OBJ-03 y OBJ-04 y la tesis lo describe como tarea simplificada.

**10. Qué se congela:** la especificación de C2-espejo y sus alternativas, el criterio de aceptación (los mismos criterios del generador) y el par (C2 → C2-espejo). **[APROBAR — decisión sobre estímulos]**

---

### Decisión 7 — Detección: definición de VP, FP y FN

**1. Problema.** Definir de forma inequívoca qué es un acierto, una omisión y una detección espuria, y en qué punto del pipeline se mide.

**Punto de medición:**
- **(i) salida cruda ≥ 0,15**, antes de los filtros: evalúa el modelo;
- **(ii) salida de `run_yolo`**, después de la lista blanca y el umbral: evalúa el detector integrado.

Recomendación: **(ii) como principal y (i) como diagnóstico**. Motivo: la lista blanca descarta en silencio las clases que no son de navegación (E7), así que ciertos falsos positivos solo son visibles en (i).

**Definiciones recomendadas** (válidas porque ninguna escena repite clase, E1):
- **VP (verdadero positivo):** objeto de diseño con al menos una detección de **su misma clase COCO** cuya IoU con la caja visible (Decisión 1) sea ≥ 0,5.
- **VP de presencia** (secundario, comparable con el Cap. 5): la misma clase, sin exigir IoU.
- **FN (falso negativo):** objeto de diseño sin VP.
- **FP (falso positivo)**, en cuatro tipos que se reportan por separado:
  - (a) **duplicado:** otra detección de una clase ya emparejada;
  - (b) **confusión:** detección con IoU ≥ 0,5 sobre un objeto de diseño, pero de otra clase; el objeto cuenta además como FN;
  - (c) **alucinación de clase:** una clase ausente de la escena, sin superposición con ningún objeto;
  - (d) **mala localización:** la clase correcta con IoU < 0,5; también cuenta como FN si no hay otra.
- **Elementos de la sala:** la sala solo tiene suelo, paredes y techo lisos (E6). Cualquier detección sobre ellos es FP de tipo (c).
- **Sinónimos:** no se acepta ninguno. Por ejemplo, "chair" detectada como "couch" es una confusión.

**Alternativas descartadas y sus riesgos:**
- aceptar sinónimos (inflaría el acierto con una regla *ad hoc*);
- no contar los duplicados (ocultaría un problema de la narrativa).

**7. Afirmación que permite:** "El detector integrado detectó k/32 objetos (IoU ≥ 0,5), con f falsos positivos (desglosados por tipo) en 18 escenas."

**8. Afirmación que no permite:** hablar de "precisión en entornos Web3D en general", porque es un solo entorno controlado.

**9–10. Qué se congela:** las definiciones anteriores, la IoU de emparejamiento de 0,5, el emparejamiento voraz por confianza descendente y los dos puntos de medición. **[APROBAR]**

---

### Decisión 8 — Posición horizontal

**1. Problema.** Comparar la columna de la API con la posición de diseño.

**Hecho:** los dos usan los tercios 1/3 y 2/3. El diseño sitúa los objetos en u = 0,25 / 0,5 / 0,75 con un margen ≥ 0,05 respecto a las fronteras. La API usa el centro x de la caja de YOLO.

**Alternativas:**
- **(a)** columna de la API frente a la etiqueta de diseño, condicionada a los VP;
- **(b)** lo mismo, de extremo a extremo, contando los FN como fallo;
- **(c)** **control previo sin YOLO:** columna del centro de la caja **visible** (GT visual) frente a la etiqueta de diseño.

(c) comprueba, **antes de ejecutar**, que la etiqueta de diseño se puede recuperar a partir de una caja ideal. Si en algún objeto no se puede (por ejemplo, un objeto ocluido cuyo centro visible cae en otro tercio), se detecta antes de ver resultados y se documenta.

**Riesgos:**
- sin (c), un fallo "de la API" podría deberse a la definición del criterio;
- solo condicionada, se ocultan los FN.

**7. Afirmación que permite:** "La API asignó la columna correcta en k/n objetos detectados (y en k'/32 de extremo a extremo)." **8. Afirmación que no permite:** hablar de posición angular continua.

**9. Recomendación:** (a) + (b) + matriz de confusión 3×3; (c) como control previo obligatorio.

**10. Qué se congela:** el criterio de tercios, los dos modos de reporte y el resultado de (c), sometido a commit antes de YOLO. **[APROBAR]**

---

### Decisión 9 — Relaciones espaciales

**1. Problema.**
- **Relaciones de diseño:** `in_front_of` (C1, C3), `beside` (C2, D3) y `on_top_of` (D2 × 2). Son 6 pares.
- **Relaciones que calcula la API:** solo "sobre superficie" (E15).

**Alternativas:**
- **(a)** `on_top_of` en la capa 5 y las demás solo en la narrativa (capa 6);
- **(b)** todas solo en la narrativa;
- **(c)** no evaluar las relaciones de forma cuantitativa.

**Riesgos:**
- con n = 6 (y n = 2 para `on_top_of`), cualquier porcentaje es engañoso;
- (b) ignora una capacidad que la API sí tiene.

**7. Afirmación que permite:** una tabla caso por caso: "la relación X se expresó / no se expresó". **8. Afirmación que no permite:** "La API describe correctamente las relaciones espaciales en un X %".

**9. Recomendación:** (a), con un reporte **descriptivo caso por caso** y sin porcentajes como resultado principal. En B y D se añade, también de forma descriptiva, si el orden izquierda/derecha implícito es correcto.

**10. Qué se congela:** la lista de los 6 pares, la capa en que se evalúa cada uno y el formato de reporte. **[APROBAR]**

---

### Decisión 10 — Narrativa: codificación y número de generaciones

**1. Problema.**
- La narrativa (capa 6) no es determinista (E14).
- Evaluarla exige un procedimiento de codificación.
- El estímulo que oirán los participantes debe ser **uno solo y congelado**.

**Codificación:**

| Alternativa | Ventajas | Riesgos |
|---|---|---|
| Rúbrica manual con 1 codificador (Brian) | Viable | Sesgo del autor; sin medida de fiabilidad |
| **Rúbrica manual con 2 codificadores + κ de Cohen** | Estándar; fiabilidad medible | Necesita un segundo codificador |
| Reglas automáticas por palabras clave (lista cerrada de sinónimos en español por clase y lado) | 100 % reproducible | Falla con paráfrasis |
| LLM como juez | — | **No recomendado:** circular (un LLM evalúa a otro LLM) y no reproducible |

**Rúbrica propuesta, por objeto de diseño:**
- ¿se menciona el objeto? (sí/no);
- ¿el lado mencionado coincide con el diseño?;
- ¿la cercanía mencionada es coherente con la categoría de la API?;
- por escena: objetos inexistentes mencionados y relaciones.

Cada punto se codifica **contra dos referencias por separado**:
- (a) la salida de YOLO de esa corrida, que mide la fidelidad del LLM;
- (b) el diseño, que mide el resultado de extremo a extremo.

**Número de generaciones:**

| | 1 generación | k = 3 generaciones por estímulo |
|---|---|---|
| Ventajas | Coste mínimo | Permite estimar la variabilidad del LLM, la limitación que la propia tesis reconoce (E14) |
| Riesgos | La variabilidad no se mide | Hay que llamar más veces a la API de Groq; hay que decidir cuál se usa como estímulo |

**Recomendación:**
- **k = 3** generaciones para la evaluación de la capa 6;
- el **estímulo para usuarios es la generación n.º 1**, por una regla declarada de antemano, **sin selección manual**;
- el TTS se genera **solo** para esa (18 audios + la referencia de OBJ-05), dentro del presupuesto de unos 15.000 COP;
- codificación: reglas automáticas **y** 2 codificadores humanos con κ. Si no hay segundo codificador, se usa Brian con doble codificación ciega separada por al menos 7 días (fiabilidad intra-codificador), declarada como limitación.

**7. Afirmación que permite:** "La narrativa mencionó k/n objetos detectados (fidelidad) y k'/32 objetos de diseño (extremo a extremo), con acuerdo entre codificadores κ = X; en 3 generaciones, la mención varió en v casos."

**8. Afirmación que no permite:** que la narrativa escuchada por los participantes sea "la mejor" ni "representativa", si no se midió la variabilidad.

**10. Qué se congela:**
- la rúbrica y la lista de sinónimos;
- k = 3 y la regla "estímulo = generación 1";
- el prompt (con su hash) y el modelo (`qwen/qwen3.8-27b`) con temperatura 0,1;
- la configuración del TTS (modelo, voz, hash de `TTS_STYLE_INSTRUCTIONS`).

**[APROBAR — decisión sobre estímulos]**

---

### Decisión 11 — Evaluación con usuarios: estímulos, criterio de acierto y orden

**1. Problema.**
- Hoy `estimulos: POR_DEFINIR` en todas las tareas OBJ.
- Las métricas del catálogo (por ejemplo "% identificación correcta") no tienen una definición operativa.
- El guion de OBJ-03 compara la respuesta del participante con "la dirección que reportó el sistema".

**Asignación propuesta** (orientativa; la decisión es de Brian):

| Tarea | Estímulo propuesto | Razón |
|---|---|---|
| OBJ-01 identificación | D2 y D3 | Varios objetos: permite medir la omisión |
| OBJ-02 relación simple | 2–3 escenas del bloque A (una por columna, a la misma distancia) | Una sola variable manipulada |
| OBJ-03 ruta libre | **C2** (B3 como reserva) | Estructura exacta de la tarea (E17) |
| OBJ-04 secuencia | C2-espejo (Decisión 6) | — |
| OBJ-05 voz | Un texto neutro fijo, sintetizado con la misma configuración de TTS | Separa la voz del contenido |
| OBJ-06 / OBJ-07 | Tras las tareas anteriores, sin estímulo propio | Valoraciones subjetivas |

**Criterio de acierto de OBJ-03 y OBJ-04.** Hay dos referencias posibles:
- (a) **la dirección libre según el diseño**, conocida a priori;
- (b) **lo que dijo la narrativa**.

Recomendación: registrar **las dos**:
- (b) mide la **comprensión** de la narrativa;
- (a) mide el **éxito de la tarea de extremo a extremo**.

Si la narrativa se equivoca, un participante que la entendió bien "falla" según (a). Mezclarlas confunde un error del sistema con un error de comprensión. Para ello, el guion de OBJ-03 se ajusta de modo que registre las dos cosas.

**Orden:**
- **fijo** para todos los participantes, con OBJ-03 → OBJ-04 contiguos;
- el efecto de orden se declara como limitación;
- con una N pequeña no se recomienda contrabalancear.

**Pistas:** la pista piloto nunca se reporta como evidencia de accesibilidad (ya consta en el catálogo).

**7. Afirmación que permite:** "k/n participantes identificaron la dirección libre coherente con la narrativa."
**8. Afirmaciones que no permite:**
- generalizar a la población con ceguera;
- comparar la pista piloto con la pista objetivo como si fuera un efecto.

**9. Recomendación:** la tabla anterior + los criterios (a) y (b) + el orden fijo.

**10. Qué se congela:**
- la asignación tarea → `stimulus_id`, escrita en `catalog.yaml`;
- la definición operativa de cada métrica;
- el guion ajustado de OBJ-03;
- el texto neutro de OBJ-05;
- el orden.

**[APROBAR — decisión metodológica]**

*Fuera del alcance de este checkpoint, pero bloquean la evaluación (no F4):* el N de participantes, el reclutamiento y la pseudonimización de sesiones.

---

### Decisión 12 — Agregación, incertidumbre y criterio de éxito

**1. Problema.**
- Las muestras son pequeñas: 32 objetos, 18 escenas, 9 en el bloque A y pocos participantes.
- La tesis no define umbrales de éxito para el rendimiento (E13 solo contiene criterios de selección de modelo).

| Alternativa | Ventajas | Riesgos |
|---|---|---|
| Umbrales de éxito predefinidos (por ejemplo, recall ≥ 0,8) | Veredicto claro | Sin respaldo en la tesis ni en la literatura, serían arbitrarios; fijados después de ver resultados, inválidos |
| **Descriptivo: k/n + IC de Wilson al 95 % + estratificación** | Honesto con muestras pequeñas; estándar para proporciones | No da un "aprobado/suspenso" |
| Pruebas de hipótesis | — | Sin potencia con estos n |

**7. Afirmación que permite:** "Recall = 27/32 (IC 95 % de Wilson: a–b)."
**8. Afirmaciones que no permite:**
- "El sistema cumple el requisito" sin un requisito previo;
- significancia estadística.

**9. Recomendación:**
- descriptivo con IC de Wilson;
- tablas por bloque, clase, columna y condición (esta última solo en el bloque A);
- datos por imagen en un anexo;
- **sin criterio de éxito numérico**, salvo que Brian aporte ahora uno con respaldo (un requisito de la tesis o de la literatura). En ese caso se congela antes de ejecutar.

**10. Qué se congela:** el método de los IC, la lista de estratos y la ausencia (o el valor) de un criterio de éxito. **[APROBAR]**

---

### Decisión 13 — Dataset 2 (robustez / variabilidad visual)

**1. Problema.** Ninguna de las 29 imágenes tiene licencia y procedencia verificadas (E18). Además, YOLO ya se ejecutó sobre las 29 en fases anteriores, lo que crea riesgo de sesgo de selección.

| Alternativa | Ventajas | Riesgos |
|---|---|---|
| a. Usar solo las que Brian pueda documentar (JC3D-01 si es un render propio; G-04 si la herramienta y sus términos lo permiten) | Aprovecha el material existente | Brian probablemente ya vio resultados de YOLO de esas imágenes (fases 5–6): sesgo que hay que declarar |
| b. Crear escenas nuevas con assets CC0 verificados (Poly Haven) y renderizarlas uno mismo | Procedencia completa | Son estímulos nuevos: más trabajo y otro checkpoint |
| c. Omitir el Dataset 2 y declararlo como limitación | Cero riesgo de licencia | Se pierde la evidencia de robustez |

**7. Afirmación que permite:** con a o b, "en k imágenes de estilo visual distinto, el sistema…" (descriptivo). **8. Afirmaciones que no permite:** generalizar a "entornos Web3D", o evaluar profundidad y posición contra una verdad geométrica.

**9. Recomendación:** **a** si Brian documenta la procedencia de al menos 2 imágenes; si no, **c**, declarando la limitación. **b** solo si hay tiempo.

En los casos a y b, el GT de presencia se define **antes** de ejecutar, por inspección, y sin consultar las salidas anteriores. El conocimiento previo de esas salidas se declara.

**10. Qué se congela:** la lista de imágenes, sus hashes y licencias, y el GT de presencia de cada una. **[APROBAR — decisión sobre estímulos]**

---

## 2. Recomendación de Claude, en una línea por decisión

| # | Recomendación |
|---|---|
| 1 | GT visual modal (B) para la localización; emparejamiento por clase; no usar la caja proyectada |
| 2 | Monotonía por columna + tabla 3×4; τ-b descriptivo; efecto del tamaño en B–D; sin correspondencia |
| 3 | CPU oficial (`CUDA_VISIBLE_DEVICES=-1`); GPU solo como verificación cruzada |
| 4 | Sistema desplegado (regla `max`, 0,35) como principal; corregir la Tabla 9; sensibilidad a B y a 0,3/0,5/0,7; AP50 como complemento; nunca ajustar con el Dataset 1 |
| 5 | 3 corridas; igualdad por hash; detenerse si difieren |
| 6 | Extensión versionada con C2-espejo; no tocar el Dataset 1 |
| 7 | VP = clase + IoU ≥ 0,5 frente a la caja visible; FP en 4 tipos; puntos de medición (ii) y (i) |
| 8 | Condicionada + extremo a extremo + control previo sin YOLO |
| 9 | `on_top_of` en la capa 5; el resto en la capa 6; descriptivo caso por caso |
| 10 | k = 3 generaciones; estímulo = generación 1; rúbrica + 2 codificadores (κ) |
| 11 | Asignación propuesta; doble criterio en OBJ-03/04; orden fijo |
| 12 | k/n + IC de Wilson; sin umbral de éxito salvo que haya un requisito previo |
| 13 | Solo imágenes documentables; si no las hay, omitir y declararlo |

## 3. Riesgo metodológico por alternativa (resumen)

| Nivel | Alternativas |
|---|---|
| **Riesgo alto** (no defendibles) | 1-A (contradice el GT); 2 con correspondencia a posteriori; 4-D (ajuste con los datos de evaluación); 10 con LLM como juez o con elección manual de la narrativa; 12 con umbral de éxito fijado a posteriori |
| **Riesgo medio** | 1-C (renuncia a la localización); 4-B como principal sin cambiar el sistema; 6-b (cambia la tarea); 13-a sin declarar la exposición previa a las salidas de YOLO |
| **Riesgo bajo** | Las alternativas recomendadas, siempre que se congelen **antes** de ejecutar |

## 4. Parámetros que quedarían congelados

Irían en un único archivo `experimental_config.yaml`, con su hash registrado en cada ejecución.

- **Estímulos:**
  - Dataset 1: `ae85f90`, manifest y hashes;
  - GT visual: commit y hashes;
  - Dataset 1-S: si se aprueba.
- **Detector:**
  - `yolo26s.pt` con sha256 `646f8bc3…4a1b`;
  - imgsz 1280, NMS 0,45, conf interna 0,15, augment false;
  - regla `max` con umbral de petición 0,35;
  - hash de `yolo_service.py` (`_CLASS_MIN_CONF` y `_NAV_CLASSES`).
- **Entorno:**
  - CPU; Python 3.13.15; `requirements.lock.txt`;
  - hilos de torch registrados; commit de la app.
- **Métricas:**
  - emparejamiento por clase con IoU 0,5 y emparejamiento voraz;
  - tipos de FP; puntos de medición (i) y (ii);
  - monotonía y tabla 3×4; τ-b descriptivo;
  - tercios; pares de relaciones;
  - IC de Wilson; estratos.
- **Corridas:** 3, con igualdad por hash.
- **Narrativa:**
  - modelo, temperatura y hash del prompt;
  - k = 3; estímulo = generación 1;
  - rúbrica y sinónimos;
  - configuración del TTS y hash de la instrucción de estilo.
- **Usuario:**
  - asignación tarea → estímulo; criterios (a) y (b);
  - orden; texto de OBJ-05.

## 5. Decisiones que requieren la aprobación de Brian

Hay que aprobar las 13. Por su naturaleza:

- **Ground truth:** 1 (crear el GT visual).
- **Configuración experimental:** 3 (CPU), 4 (qué sistema es el oficial: A o B), 5 (corridas).
- **Métricas:** 2, 7, 8, 9, 12 (incluido si existe un criterio de éxito con respaldo).
- **Estímulos:** 6 (generar C2-espejo o redefinir OBJ-04), 10 (k = 3 y la regla de la generación 1), 13 (qué imágenes del Dataset 2, y aportar su procedencia).
- **Metodología de usuario:** 11 (asignación, doble criterio, orden).
- **Documento de la tesis** (sin modificarlo ahora):
  - la Tabla 9 debe describir la regla `max`;
  - el borrador v1 del CP3B tenía un error (ventanas y puerta), ya corregido en este documento.

## 6. Versión propuesta del CP3B (v2), sin ejecutar

1. **Objetivo.** Evaluar el sistema en capas separadas. Cada métrica pertenece a una sola capa.

| Capa | Qué contiene |
|---|---|
| 1–2 | Diseño |
| 3 | Imagen |
| 4 | Detector |
| 5 | Interpretación espacial |
| 6 | Narrativa |
| 7 | Usuario |

2. **Estímulos.**
   - Dataset 1: 18 escenas, 32 objetos, congelado en `ae85f90`.
   - GT visual modal: nuevo, versionado, **congelado antes de YOLO** (D1).
   - Extensión Dataset 1-S con C2-espejo, si se aprueba (D6).
   - Dataset 2 según D13.
3. **Configuración.** La del §4, en `experimental_config.yaml` con su hash. CPU (D3). Sistema desplegado: regla `max` con 0,35 (D4). Se guarda también la salida cruda ≥ 0,15.
4. **Ejecución.**
   - 3 corridas; si los hashes difieren, se detiene (D5).
   - Controles previos sin YOLO:
     - re-render RGB = hash congelado;
     - columna del centro visible = etiqueta de diseño (D8).
5. **Capa 4 (detección):**
   - VP, FN y FP de 4 tipos (D7);
   - recall k/32 y precisión, con IC de Wilson;
   - IoU frente a la caja visible: distribución, k con IoU ≥ 0,5, objetos pequeños aparte (D1);
   - AP50 como complemento;
   - sensibilidad: la configuración B y la rejilla 0,3/0,5/0,7 (D4).
6. **Capa 5 (interpretación espacial):**
   - columna: condicionada, extremo a extremo y matriz 3×3 (D8);
   - profundidad: monotonía, tabla 3×4, τ-b descriptivo y efecto del tamaño (D2);
   - `on_top_of`, descriptivo (D9).
7. **Capa 6 (narrativa):**
   - k = 3 generaciones; rúbrica con doble referencia (YOLO y diseño);
   - 2 codificadores con κ; relaciones caso por caso (D9, D10).
8. **Paquete congelado (F4):**
   - generación 1 + TTS;
   - campos definidos en el borrador v1, §5;
   - asignación a tareas (D11).
9. **Capa 7 (usuario):**
   - criterios (a) diseño y (b) narrativa;
   - orden fijo;
   - la pista piloto no es evidencia (D11).
10. **Reporte:**
    - k/n + IC de Wilson; estratos; anexo con los datos por imagen;
    - sin criterio de éxito, salvo requisito previo (D12);
    - los resultados exploratorios anteriores no se mezclan con los formales.

## 7. Qué impide todavía iniciar F4

1. **Aprobar las 13 decisiones** (§5), en especial la 4 (qué sistema es el oficial), porque determina las narrativas y los audios que se congelan.
2. **Generar y congelar el GT visual** (D1) y pasar los controles previos (D8). Implica un cambio de código en el generador, que necesita aprobación.
3. **Decidir OBJ-04** (D6). Si se elige la opción a: generar C2-espejo y congelarla en su propio checkpoint.
4. **Crear `experimental_config.yaml`** con el §4 y hacer su commit **antes** de la primera ejecución.
5. **Congelar el catálogo:** asignación tarea → estímulo (D11), texto de OBJ-05, guion ajustado de OBJ-03.
6. **Congelar la rúbrica** de la narrativa y la lista de sinónimos (D10).
7. **Script de evaluación de F4:**
   - llama al pipeline real en el perfil `development` con `CUDA_VISIBLE_DEVICES=-1`;
   - guarda la salida cruda y la filtrada, y los hashes.
   - Es código nuevo que necesita aprobación y pruebas.
8. **Presupuesto y cuota del TTS:** confirmar el coste por audio o si se usa el nivel gratuito (unas 19 síntesis) antes de llamar al TTS.
9. **Dataset 2:** no bloquea F4 del Dataset 1, pero sí la evaluación de robustez (D13).
10. **Antes de la evaluación con usuarios** (no bloquea F4):
    - `API_KEYS` configuradas;
    - límite de peticiones ajustado para el perfil `study`;
    - pseudonimización;
    - clasificar las 2 sesiones existentes;
    - N y reclutamiento.
