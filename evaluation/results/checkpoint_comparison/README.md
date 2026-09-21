# Prueba exploratoria — YOLO26s COCO-80 vs. YOLO26s Objects365

**Estado: exploratoria. NO es resultado formal del Capítulo 5. NO reemplaza ni modifica el modelo de producción.**

## Qué se comparó

- **A. `yolo26s.pt`** — checkpoint de producción actual (COCO-80, 80 clases).
- **B. `yolo26s-objv1-150.pt`** — checkpoint oficial de Ultralytics preentrenado en Objects365v1 (365 clases). Descargado automáticamente desde `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-objv1-150.pt` (19.9 MB) para esta prueba. No fue afinado ni modificado por este proyecto.

Script: `scripts/compare_checkpoints.py` (aislado, no importa ni modifica ningún módulo de `app/`).

## Configuración usada (idéntica para ambos modelos, salvo la excepción explicada abajo)

| Parámetro | Valor |
|---|---|
| `imgsz` | 1280 (misma resolución que producción) |
| `iou` (NMS) | 0.45 (mismo valor que producción) |
| `device` | `cpu` (única opción disponible en este entorno; `torch.cuda.is_available()` = False) |
| `augment` | False (mismo valor por defecto que producción) |
| `conf` | 0.15 — **única diferencia respecto a producción**, ver nota siguiente |

**Nota sobre `conf`:** producción aplica un umbral de confianza interno bajo (0.15) y luego un post-filtro por clase (`_CLASS_MIN_CONF`, definido solo para las 80 clases de COCO). Esa tabla no tiene equivalente definido para las 365 clases de Objects365, así que aplicarla habría sido arbitrario y hubiera favorecido a COCO. Para no favorecer a ningún modelo, esta prueba usa **conf=0.15 como único umbral final para ambos**, sin post-filtro por clase, y registra la confianza exacta de cada detección para que pueda filtrarse después si se desea.

## Imágenes utilizadas

**29 imágenes**, exactamente las que existen hoy en `evaluation/images/web3d/` (verificado antes de ejecutar, ninguna imagen fue eliminada ni añadida). No se usó `test_images/` porque ya participó en la selección del modelo del Capítulo 3.

Nota heredada de la Fase 3: el inventario CSV (`evaluation/metadata/inventario_imagenes.csv`) solo cataloga 26 de las 29 imágenes; faltan `W3D-G-04-Biblioteca.png`, `W3D-G-05-Ciudad.png`, `W3D-G-06-Universidad.png`. Esta prueba las usó igualmente (están presentes en disco), pero el hallazgo del CSV sigue pendiente de corrección documental, no se tocó aquí.

## Resultados agregados

| | A. yolo26s (COCO-80) | B. yolo26s-objv1-150 (Objects365) |
|---|---|---|
| Total de detecciones (29 imágenes, conf≥0.15) | 542 | 834 |
| Promedio por imagen | 18.69 | 28.76 |
| Clases distintas detectadas al menos una vez | 35 de 80 | 72 de 365 |
| Tiempo de inferencia (ms) — promedio / mínimo / máximo | 601.24 / 476.86 / 918.31 | 664.69 / 502.67 / 950.84 |

**Advertencia sobre el tiempo:** medido en CPU, en este entorno de desarrollo, sin aislar otros procesos del sistema. No es comparable a los benchmarks oficiales de Ultralytics (87.2 ms CPU ONNX para YOLO26s), que usan ONNX Runtime optimizado, imgsz=640 y condiciones controladas de laboratorio — esta medición usa PyTorch directo, imgsz=1280 y el hardware real de este equipo. Objects365 fue en promedio ~10% más lento que COCO en esta medición, consistente con tener una cabeza de clasificación de 365 salidas en vez de 80.

### Clases detectadas solo por un modelo (en las 29 imágenes)

**Solo por Objects365 (41 clases nuevas observadas):** air conditioner, banana, barrel/bucket, basket, bathtub, blackboard/whiteboard, cabinet/shelf, candle, carpet, chicken, coffee table, deer, **desk**, faucet, fire extinguisher, flower, hat, head phone, ladder, **lamp**, leather shoes, machinery vehicle, mirror, picture/frame, pillow, plate, pot/pan, power outlet, printer, sneakers, speaker, stool, storage box, street lights, suv, towel/napkin, traffic sign, trash bin/can, trolley, van, wild bird.

**Solo por COCO (4 clases, todas irrelevantes para navegación egocéntrica interior):** bird, boat, cake, orange.

## Corrección importante respecto a la Fase 4 (documental) — verificado ahora empíricamente

En la Fase 4, una fuente documental (el archivo `Objects365.yaml` de Ultralytics, leído vía fetch resumido) indicó que la clase 37 era `Monitor/TV` (combinada). **Al cargar el checkpoint real y leer `model.names` directamente, la clase 23 es literalmente `'tv'`, igual que en COCO — no existe ninguna clase "Monitor/TV" combinada ni ninguna clase "monitor" independiente.** Esto corrige la Fase 4: la fuente documental no coincidía con el vocabulario real horneado en el checkpoint. **La conclusión de fondo de la Fase 4 se mantiene igual (ningún checkpoint separa monitor de TV), pero el detalle del nombre de clase queda corregido con evidencia directa, que prevalece sobre la documentación.**

## TV / Monitor — observado en las imágenes

| Modelo | Imagen | Confianza |
|---|---|---|
| yolo26s | `W3D-I-JC3D-04-school.png` | 0.80, 0.71, 0.37 |
| yolo26s | `W3D-I-SKF-07-living-room.png` | 0.90 |
| yolo26s | `W3D-I-SKF-09-office.png` | 0.70 |
| yolo26s-objv1-150 | `W3D-G-04-Biblioteca.png` | 0.25 |
| yolo26s-objv1-150 | `W3D-I-SKF-07-living-room.png` | 0.84 |
| yolo26s-objv1-150 | `W3D-I-SKF-09-office.png` | 0.89, 0.20, 0.16 |

**Inspección visual directa (recortando la región exacta del bounding box):**

- `W3D-I-SKF-09-office.png`: confirmado visualmente — hay **monitores de computador reales** sobre los escritorios de los cubículos. Tanto COCO como Objects365 los etiquetan `tv`. **Ningún checkpoint distingue monitor de TV; ambos cometen la misma "confusión" porque, para ambos, es la clase correcta y única disponible, no un error de inferencia.**
- `W3D-I-SKF-07-living-room.png`: confirmado visualmente — hay un **televisor de pared real**. Aquí la etiqueta `tv` es correcta en ambos modelos.
- `W3D-G-04-Biblioteca.png` (Objects365, conf=0.25): se recortó el bounding box exacto (641,504)-(690,557) y corresponde a la **pantalla de un laptop abierto** en primer plano — el mismo laptop que Objects365 también detectó correctamente como `laptop` con confianza alta (0.95) en otra caja. Es una **detección duplicada/espuria de baja confianza sobre el mismo objeto**, no una confusión monitor/TV real.

**Conclusión de esta sección: ni COCO-80 ni Objects365-365 distinguen monitor de TV — ambos usan la misma clase `tv` para cualquier pantalla. Objects365 no resuelve este problema, confirmado con evidencia visual directa, no solo documental.**

## Escenas prioritarias — inspección visual detallada

### Oficina (`W3D-I-SKF-09-office.png`)
Escena real: oficina tipo call-center con cubículos, monitores, teclados, mouse, sillas, una planta, un mueble/estantería de madera con un objeto que parece impresora/escáner.

- **COCO** (14 detecciones): keyboard, mouse, chair(×7), potted plant, tv — correcto pero genérico; **no tiene forma de representar los escritorios ni el mueble/estantería** (no existen esas clases en COCO-80).
- **Objects365** (23 detecciones): además de lo anterior, añade **`desk`(×2, conf 0.34/0.24)** y **`cabinet/shelf`(×2, conf 0.69/0.25)**, ambos visualmente correctos, más `printer`(×2) y `book`(0.28) plausibles. **Esta es la evidencia más clara de valor añadido real**: la escena es estructuralmente una oficina de escritorios, y solo Objects365 puede representar ese mobiliario.

### Biblioteca (`W3D-G-04-Biblioteca.png`)
Escena real: biblioteca moderna de dos niveles, personas trabajando en mesas con laptops, lámparas de mesa, estanterías, un carrito de libros, una escalera, plantas.

- **COCO** (64 detecciones): person(×9), laptop(×4), chair(×9), potted plant(×9), book(×15), dining table(×4), couch, bottle, backpack(×2), vase — ya es una lista razonablemente rica, pero confunde las mesas de biblioteca con `dining table` (única clase de mesa disponible en COCO) y no puede nombrar las estanterías ni el carrito.
- **Objects365** (84 detecciones): añade correctamente **`lamp`(×7)** (las lámparas de mesa visibles), **`cabinet/shelf`(×6)** (las estanterías), **`trolley`(0.34)** (el carrito de libros, visible en la imagen), `desk`(×4), `head phone`(0.31, plausible — hay auriculares visibles en una persona), `carpet`(0.19). También produce el falso positivo de `tv` ya analizado arriba.

### Universidad (`W3D-G-06-Universidad.png`)
- **COCO** (46 detecciones): person(×27), backpack(×5), bench(×5), book, laptop, cup, handbag(×2) — razonable para una escena de campus con estudiantes.
- **Objects365** (68 detecciones): añade `sneakers`(×9), `trash bin/can`(×4), `street lights`, `lamp` — granularidad mayor pero de utilidad marginal para la narrativa egocéntrica (saber que alguien lleva "sneakers" no aporta información de navegación). No se identificó mobiliario académico distintivo (pizarras, escritorios) porque la escena es un espacio abierto de campus, no un aula — **no inspeccionado en el mismo nivel de detalle visual que oficina/biblioteca por alcance de tiempo de esta prueba exploratoria**.

### Sala de estar (`W3D-I-SKF-07-living-room.png`)
Escena real: TV de pared, dos sofás, mesa de centro negra, alfombra, lámpara de pie, cuadro enmarcado.

- **COCO** (5 detecciones): couch(×2), tv, bowl, potted plant — limpio y correcto.
- **Objects365** (16 detecciones): añade correctamente `carpet`(0.93), `lamp`(0.89), `coffee table`(×3), `picture/frame`(0.60) — todo verificado visualmente como presente y correcto, refuerza la evidencia de "sala de estar" con más detalle contextual real. También produjo 2 detecciones de baja confianza (`desk`, 0.24/0.23) que no se lograron ubicar visualmente con certeza en esta inspección — posible falso positivo sobre la mesa de centro o una mesa auxiliar, marcado como **no verificado con certeza**, no se afirma que sea correcto ni incorrecto.

### Estudio con librero (`W3D-I-PHV-03-studio2.png`)
Escena real: librero lleno de libros, escritorio pequeño con jarrón y flores, silla, ventana, cajas apiladas, cuadro, un globo terráqueo en el piso.

- **COCO** (28 detecciones): chair, vase(×2), **book(×25)** — sobre-segmenta el librero en muchas detecciones individuales de lomos de libros (plausible dada la densidad del librero, pero no se puede confirmar sin contar manualmente cada lomo).
- **Objects365** (16 detecciones): **book(×3)** — muchas menos, pero añade `storage box`(0.38, correcto — las cajas visibles), `desk`(0.20, correcto — el escritorio pequeño), `cabinet/shelf`(0.15, correcto — el librero como mueble), `pillow`(0.19, correcto — el cojín de la silla), `flower`(0.57, correcto), `mirror`(0.23, **no confirmado visualmente, posible falso positivo**), `bench`/`stool`(ambiguos, posiblemente la misma silla).

**Este caso ilustra un trade-off real, no una superioridad de un modelo sobre otro**: COCO captura mejor la cantidad de libros individuales; Objects365 captura mejor el mobiliario y objetos contextuales del cuarto, a costa de sub-contar los libros. Cuál es más útil depende de si la narrativa necesita "hay muchos libros" (ambos lo permiten, con distinto detalle) o "hay un escritorio con flores junto a una ventana" (solo Objects365 lo permite).

## Falsos positivos observados (inspección visual, no exhaustiva)

- Objects365, `W3D-G-04-Biblioteca.png`: `tv`(0.25) — duplicado espurio de un `laptop` ya detectado correctamente (ver arriba).
- Objects365, `W3D-I-SKF-07-living-room.png`: `desk`(0.24, 0.23) — no verificado con certeza, posible falso positivo de baja confianza.
- Objects365, `W3D-I-PHV-03-studio2.png`: `mirror`(0.23) — no confirmado visualmente.

No se realizó inspección visual exhaustiva de las 29 imágenes — se priorizaron oficina, biblioteca, sala de estar y estudio por ser las escenas más relevantes para la pregunta concreta del proyecto (desambiguar oficina/biblioteca de sala de estar). Universidad, calle, ciudad, parque, granja e interiores restantes solo se revisaron a partir de los datos JSON, sin inspección visual detallada — **esto debe declararse como limitación de esta prueba exploratoria si se usa como base de una decisión futura.**

## Objetos importantes que siguen sin detectarse (ambos modelos)

Ninguna de las 29 imágenes produjo detecciones de escaleras, puertas, ventanas, pasamanos ni ascensores como clases propias, en ningún modelo — consistente con el diagnóstico de la Fase 4 (esas clases no existen en el vocabulario de ninguno de los dos checkpoints). La escalera visible en `W3D-G-04-Biblioteca.png` no fue detectada como tal por ningún modelo.

## Comparación final (dimensional, sin declarar un "ganador")

| Dimensión | Observación |
|---|---|
| **Cobertura de objetos** | Objects365 detecta 72 clases distintas de las 365 disponibles vs. 35 de 80 en COCO, y añade 41 clases nuevas observadas en este conjunto, varias directamente relevantes (`desk`, `lamp`, `cabinet/shelf`, `printer`, `coffee table`, `picture/frame`, `storage box`, `trolley`) |
| **Corrección semántica aparente** | En los casos inspeccionados visualmente (oficina, biblioteca, sala de estar), las detecciones nuevas de Objects365 fueron mayormente correctas; se identificaron 2-3 posibles falsos positivos de baja confianza, ninguno grave |
| **Información contextual** | Claramente mayor en Objects365 para escenas de oficina y biblioteca — es la evidencia más directa a favor de estudiar esta opción con más profundidad |
| **Falsos positivos** | Presentes en ambos modelos; los de Objects365 observados en esta prueba fueron de confianza baja (<0.25) y no sistemáticos |
| **Objetos no detectados** | Ninguno de los dos modelos detecta elementos arquitectónicos de navegación (escaleras, puertas, ventanas, pasamanos, ascensores) — limitación compartida, no resuelta por el vocabulario mayor |
| **Tiempo de inferencia** | Objects365 ~10% más lento en esta medición de CPU local; ambos del orden de 0.5-0.9 s por imagen en este entorno, muy por encima del benchmark oficial en GPU |
| **Impacto potencial en narrativa** | Las clases nuevas de Objects365 (`desk`, `cabinet/shelf`, `lamp`, `printer`, `coffee table`) son del tipo que ya se usa hoy como evidencia heurística/LLM en `scene_classifier.py` — su incorporación podría reforzar directamente la desambiguación oficina/biblioteca vs. sala de estar que motivó esta fase, condicionado a definir umbrales y categorías nuevas antes de cualquier integración |

**No se declara un modelo "ganador".** La evidencia recogida es suficiente para considerar que **vale la pena una evaluación más formal de Objects365** enfocada específicamente en las clases de mobiliario de oficina/biblioteca, pero no es suficiente (ni pretende serlo) para tomar una decisión de integración — eso requeriría ground truth, métricas formales y una prueba de regresión contra `test_images/` (Fase 3/regla 28), ninguna de las cuales se ejecutó aquí.

## Limitaciones de esta prueba

1. Exploratoria — sin ground truth, sin Precision/Recall/F1/mAP.
2. Inspección visual no exhaustiva (5 de 29 imágenes revisadas en profundidad).
3. `conf=0.15` sin post-filtro por clase — no reproduce exactamente el comportamiento de producción, que sí aplica `_CLASS_MIN_CONF`.
4. Medición de tiempo en CPU local, no comparable a benchmarks oficiales ni a un entorno de producción con GPU.
5. No se ejecutó ninguna prueba de regresión contra `test_images/` — pendiente si se decide avanzar (regla 28 del proyecto).
6. No se probó YOLOE-26 en esta fase (explícitamente fuera de alcance, según instrucción).

## Posibles siguientes pasos (no autorizados en esta fase, requieren aprobación explícita)

1. Inspección visual del resto de las 29 imágenes (calle, ciudad, parque, granja, interiores restantes) para completar la evidencia exploratoria.
2. Definir manualmente un pequeño ground truth solo para las escenas de oficina/biblioteca/sala de estar, para poder calcular Precision/Recall en ese subconjunto específico antes de decidir integración.
3. Si el resultado sigue siendo favorable: diseñar la matriz de `_CLASS_MIN_CONF` equivalente para las clases de Objects365 relevantes (no las 365 completas) antes de proponer cualquier cambio a `yolo_service.py`.
4. Prueba de regresión contra `test_images/` comparando ambos checkpoints, no solo Objects365.

**Ningún archivo de producción fue modificado en esta fase.** El checkpoint `yolo26s.pt` sigue siendo el único usado por `app/services/yolo_service.py`.
