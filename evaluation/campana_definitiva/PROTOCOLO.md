# Protocolo de la campaña definitiva (Capítulo 3) y de la prueba R2 (RNF-02)

- **Versión:** 5, del 27-oct-2026. Sustituye a las versiones 1 a 4, que nunca tuvieron commit. Los cambios frente a la versión 4 están en `Matriz_cambios_protocolo_v5_2026-10-27.md`.
- **Rama:** `evaluacion/campana-definitiva`. Sin commits en `main` y sin despliegues.
- **Estado:** **en revisión. Nada se ha ejecutado ni descargado.** Las decisiones marcadas **[APROBACIÓN]** dependen del autor.
- **Regla:** este documento, la rúbrica, el código, el inventario y las huellas de los pesos se registran en un commit de la rama **antes** de cualquier medición. El código **se niega a medir** si hay cambios sin commit (sección 7.2).

---

## 0. Pregunta y alcance

**Pregunta.** ¿Cómo se comparan YOLO26s, Faster R-CNN, Mask R-CNN y SSD-MobileNet V2 en estos aspectos?
- velocidad de inferencia;
- uso de CPU y memoria;
- sensibilidad al umbral de confianza;
- concordancia de las clases detectadas con las presentes en la escena.

Se miden sobre las 12 imágenes de la Tabla 4, con las 80 clases de COCO, umbral aplicado directamente y en el mismo equipo y sesión.

**Uso.** Esta campaña es la fuente de las Tablas 5–7 y de la selección del modelo.

**Qué no es.** No es una evaluación de exactitud de detección al estilo COCO (sección 2).

---

## 1. Inventario de presencia

### 1.1 Archivos y valores

- `inventario_12_imagenes.csv`: 80 clases de COCO (inglés y español) × 12 imágenes, más `observaciones`.
- `inventario_observaciones_por_imagen.csv`: SHA-256 de cada imagen, descripción, casos dudosos con el criterio aplicado y fecha.

| Valor | Significado |
|---|---|
| `1` | Al menos un objeto **físico** de la clase está visible, aunque sea parcialmente, y se identifica sin ambigüedad. |
| `0` | No hay ningún objeto de esa clase. |
| `?` | Dudoso: objeto muy pequeño o borroso; objeto representado y no físico (foto, cuadro, reflejo, contenido de una pantalla); o una clase que se confunde con otra (sofá y sillón). La razón se anota en `observaciones`. |

### 1.2 Cómo se construye (se declara tal cual en la tesis)

1. **Plantilla.** Se generó automáticamente con la lista de las 80 clases de COCO del traductor del servicio y los nombres de archivo de `test_images/`. Todas las celdas están vacías; la plantilla no contiene ninguna predicción.
2. **Llenado.** Lo hace el autor, mirando cada imagen, con los valores de la sección 1.1. Anota en `observaciones` el criterio de cada `?`.
3. **Sin revisión independiente.** Lo llena una sola persona y no hay segundo anotador.
4. **No es ciego.** El autor revisa las clases que nombraron los modelos, así que conoce sus detecciones. **La tesis no afirmará que la evaluación fue ciega** ni que la revisión es independiente de esas detecciones.

### 1.3 Versionado obligatorio antes de la campaña

El inventario completo se registra en un **commit identificable** de la rama antes de iniciar la campaña. El código lo exige:

1. `campana_definitiva.py orquestar` exige `--inventario` y, **antes de crear la carpeta de salida o ejecutar cualquier modelo**, comprueba (`congelar_inventario()`):
   - que está completo: 80 clases × 12 imágenes, solo `1`, `0` o `?`;
   - que está **rastreado por Git** y tiene un commit (`git log -1 -- <ruta>`);
   - que el archivo actual **coincide con el del commit**. Se comparan los identificadores de objeto de Git (`git hash-object` del archivo frente a `git rev-parse <commit>:<ruta>`), porque el repositorio usa `core.autocrlf=true` y una huella SHA-256 del contenido de `git show` no coincidiría con la del archivo en disco.
   Si algo falla, no arranca.
2. Registra en `orquestacion.json` el commit, los dos identificadores de objeto y el **SHA-256 del archivo**.
3. `campana_definitiva_resumen.py` repite la comprobación. **La campaña no es válida y no se selecciona ningún modelo** si:
   - no hay commit registrado al iniciar;
   - el inventario actual está en otro commit;
   - tiene cambios respecto a su commit;
   - su SHA-256 no coincide con el registrado.

**Qué garantiza y qué no.** Garantiza la **integridad**: el inventario que se usa para puntuar es el mismo que estaba en el commit antes de la campaña, y no se puede cambiar después sin que se note. **No garantiza la independencia ni la ceguera** (sección 1.2).

### 1.4 Qué permite calcular el inventario y qué no

**Permite calcular** (por modelo y umbral, sobre pares imagen–clase):
- VP, FP y FN de presencia;
- precisión, recall y F1 de presencia;
- número de imágenes con al menos un FP de presencia;
- la lista de las clases nombradas sin estar y de las omitidas;
- el recall de presencia por clase, solo como dato descriptivo, porque hay pocas imágenes por clase;
- todo lo anterior con 0,3, 0,5 y 0,7.

**No permite calcular**, porque no hay recuadros ni conteos de instancias:
- precisión o recall de detección por recuadro;
- IoU, AP, mAP@0,5 ni mAP@[0,5:0,95];
- curvas de precisión–recall de detección;
- exactitud de localización o del número de objetos por clase;
- falsos positivos por instancia;
- calibración de la confianza.

Ninguna métrica permitida se llamará «mAP», «AP» ni «precisión de detección» sin el calificativo «de presencia».

---

## 2. Qué mide cada métrica

| Métrica | Unidad | Procedimiento | Requisito de datos |
|---|---|---|---|
| **Precisión convencional de detección** | Recuadro | Con un umbral de confianza `s` y de IoU `t`, una detección es VP si su clase coincide y su IoU con un objeto anotado aún no emparejado es ≥ `t`, y se empareja una sola vez; si no, es FP. Los objetos sin emparejar son FN | Recuadros anotados |
| **AP y mAP** | Recuadro | AP de una clase = área bajo la curva precisión–recall, ordenando todas las detecciones por confianza, con un IoU `t`. mAP de COCO = media del AP sobre clases y diez umbrales de IoU (0,50 a 0,95) | Recuadros anotados y todas las detecciones con su confianza |
| **Precisión, recall y F1 de presencia** (esta campaña) | Par (imagen, clase) | Sección 3.2 | Inventario de presencia |
| **Sensibilidad al umbral (retención)** | Conteo | `R = Σ n(0,7) / Σ n(0,3)` | Ninguno adicional |

Las métricas de presencia **no comprueban la ubicación ni el número de instancias, no integran una curva y no son mAP**. Tampoco son comparables con los valores publicados para estos modelos. Responden si la narrativa nombraría clases que no están en la escena o si omitiría clases que sí están. Es el mismo procedimiento de la Tabla 22 de la tesis.

---

## 3. Definiciones y fórmulas

Notación: `m` es el modelo, `i` la imagen (12), `u` el umbral (0,3, 0,5 y 0,7) y `k` la repetición (1, 2 y 3). Una **llamada** es una ejecución de la función de servicio con los bytes ya preparados; cada modelo tiene 12 × 3 × 3 = **108**. Una **corrida** es la combinación `(m, i, u)`; cada modelo tiene **36**.

### 3.1 Tiempos, recursos y conteos

| Métrica | Definición |
|---|---|
| `n(m,i,u)` | Número de detecciones de la **repetición 1**. |
| Repeticiones distintas | Las 3 repeticiones se comparan (clase, confianza con 3 decimales y recuadro). Toda diferencia es un **aviso** y no invalida la campaña. Conteos, confianza y presencia usan siempre la repetición 1. |
| Corrida sin detecciones | Cuenta como 0 en los objetos promedio y en la retención. Su confianza media **no está definida**: no entra en los promedios y se informa como «corridas sin detecciones». En presencia: sección 3.2. |
| Objetos por imagen | `media_i n(m,i,u)` por umbral (Tabla 7) y `media_{i,u} n(m,i,u)` en total (Tabla 5). |
| Confianza media | Media de las confianzas de la corrida; para el modelo, media de las corridas con `n > 0`. |
| Tiempo de una corrida `t(m,i,u)` | `mediana_k` del tiempo de reloj (`perf_counter`) de la llamada. Incluye la decodificación de la imagen, el preprocesamiento interno, la inferencia y el postproceso del servicio. |
| Tiempo de inferencia del modelo | `mediana` de los 36 `t(m,i,u)` (criterio de velocidad). También se informan la media y los dos p95 (sección 3.3). |
| CPU por inferencia | `mediana_k` de los segundos de CPU del proceso (usuario + sistema) durante la llamada; para el modelo, la mediana de las 36 corridas. También se informa el % de un núcleo = 100 × CPU / tiempo. |
| Memoria | Sección 5.4. En la Tabla 5: RSS con el modelo listo, tras la carga y el calentamiento. |
| Retención | `R(m) = Σ_i n(m,i,0,7) / Σ_i n(m,i,0,3)`. Si el denominador es 0, no está definida. También se informa la variación absoluta `(Σ_i n(0,3) − Σ_i n(0,7)) / 12`. |
| Imágenes con falsos positivos | `#{i : FP_i ≥ 1}` con 0,5. |

**Límites de la retención:** las confianzas de arquitecturas distintas no están en la misma escala; no distingue si lo que se pierde al subir el umbral son aciertos o errores; con 12 imágenes, pocos objetos mueven el cociente. En la tesis se llamará **«sensibilidad al umbral (retención de detecciones)»**, no «robustez».

### 3.2 Presencia: VP, FP, FN, precisión, recall y F1

Para cada imagen `i`, con el umbral 0,5 y la repetición 1:
- `D_i` = conjunto de clases detectadas (una clase cuenta una vez aunque tenga varias detecciones);
- `P_i` = clases marcadas `1`;
- `Q_i` = clases marcadas `?`.

| Magnitud | Fórmula | Notas |
|---|---|---|
| VP | `Σ_i \|D_i ∩ P_i\|` | |
| FP | `Σ_i \|D_i − P_i − Q_i\|` | Una clase detectada que el inventario marca `0`. |
| FN | `Σ_i \|P_i − D_i\|` | Una clase presente que no se detectó. |
| Precisión | `VP / (VP + FP)` | **No definida** si `VP + FP = 0`. |
| Recall | `VP / (VP + FN)` | **No definida** si `VP + FN = 0`. |
| **F1** (criterio principal) | `2·VP / (2·VP + FP + FN)` | Equivale a la media armónica de precisión y recall cuando ambas están definidas. **No definida** solo si `2·VP + FP + FN = 0`. |

**Clases dudosas (`?`).** Son **neutras**: si se detectan, no suman VP ni FP; si no se detectan, no suman FN. Así, una clase dudosa no premia ni castiga a ningún modelo. El número de celdas `?` y sus motivos se informan.

**Casos sin detecciones.**
- Una imagen sin detecciones aporta `D_i = ∅`: ningún VP ni FP y tantos FN como clases presentes.
- Un modelo sin ninguna detección no dudosa en las 12 imágenes tiene `VP + FP = 0`: su **precisión no está definida**, pero su **F1 sí está definida y vale 0** si hay clases presentes. Por eso F1 no deja el criterio incompleto en ese caso, y ese modelo queda último en el criterio, como corresponde.
- F1 solo queda indefinida si el inventario no tiene ninguna clase presente y el modelo no nombra ninguna clase no dudosa. Con las 12 imágenes del proyecto eso no ocurre, pero si ocurriera el criterio queda incompleto y no hay selección automática (sección 4.3).

**Por qué F1 y no la precisión.** Con la precisión, un detector que solo nombra la clase más evidente obtiene 1,0 aunque omita casi todo. Ejemplo verificado en las pruebas: el modelo A detecta solo «persona» en 12 imágenes con 5 clases presentes cada una (VP = 12, FP = 0, FN = 48: precisión 1,0, recall 0,2, F1 = 0,33); el modelo B detecta las 5 clases y nombra un gato inexistente en 3 imágenes (VP = 60, FP = 3, FN = 0: precisión 0,95, recall 1,0, F1 = 0,98). Con la precisión gana A; con F1 gana B. Para una narrativa destinada a una persona con discapacidad visual, omitir objetos presentes también es un error. F1 pondera por igual ambos errores.

**Qué se conserva en el registro.** `metricas.json` guarda, por modelo y umbral (0,3, 0,5 y 0,7): VP, FP, FN, precisión, recall, F1, imágenes con FP y el detalle de las clases nombradas sin estar. Precisión y recall no puntúan, pero entran en la sensibilidad (sección 4.6).

### 3.3 Percentil 95: definición

Hay dos p95 posibles y miden cosas distintas:

| | p95 de las **108 llamadas** | p95 de las **36 medianas por corrida** |
|---|---|---|
| Conjunto | Cada llamada de la campaña principal (12 imágenes × 3 umbrales × 3 repeticiones) | `t(m,i,u)`: la mediana de las 3 repeticiones de cada corrida |
| Posición (rango más cercano) | `⌈0,95 × 108⌉ = 103`: el **sexto** valor mayor | `⌈0,95 × 36⌉ = 35`: el **segundo** valor mayor |
| Qué responde | ¿Con qué frecuencia una llamada concreta al detector es lenta? | ¿Hay imágenes o umbrales que son lentos de forma sistemática? |
| Sensible a picos aislados | **Sí**: una pausa del sistema durante una llamada aparece | No: la mediana de 3 la oculta |
| Campo | `tiempo_p95_llamadas_ms` | `tiempo_p95_medianas_ms` |

**Decisión: la elegibilidad usa el p95 de las 108 llamadas.**
- **Por qué.** El RNF-02 se aplica **a cada solicitud**: un usuario sufre la llamada lenta, no la mediana de tres. El p95 de medianas oculta precisamente las colas que el requisito castiga.
- **El p95 de las 36 medianas se informa** y se usa en la variante `elegibilidad_p95_36_medianas` de la sensibilidad.
- **Exclusiones** de ambos: carga, calentamiento, control de deriva y corrida de YOLO26s a 640 px.

**Cálculo en el código** (`campana_definitiva_resumen.py`):

```python
def pct(valores, p):
    v = sorted(valores)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]      # rango más cercano, sin interpolación

llamadas = [r["tiempo_ms"] for r in c["medicion"]]          # en metricas_modelo()
mt["tiempo_p95_llamadas_ms"] = pct(llamadas, 95)            # 108 valores → posición 103
```

`p95 = t_(⌈0,95·n⌉)`, donde `t_(1) ≤ … ≤ t_(n)` son los tiempos ordenados. Sin interpolación: el resultado siempre es un tiempo realmente observado.

**Qué observaciones entran y cómo se excluyen las demás:**
1. `medir_modelo()` escribe tipos de registro distintos: `carga` (1), `calentamiento` (3), `memoria_tras_carga` y, después, `medicion` (una por llamada). La carga y el calentamiento **se ejecutan antes** de la primera `medicion` y **no se escriben como `medicion`**.
2. `metricas_modelo()` solo lee `c["medicion"]`. La carga y el calentamiento se leen aparte para `carga_ms` y `calentamiento_ms`, que no intervienen en ningún tiempo de inferencia.
3. El control de deriva y la corrida a 640 px están en **archivos distintos** (`crudo_deriva_yolo.jsonl` y `crudo_yolo_640.jsonl`) y no se leen al calcular las métricas principales.
4. `validar()` exige que `c["medicion"]` tenga **exactamente** las 108 combinaciones imagen × umbral × repetición, sin duplicados. Si falta o sobra alguna, la campaña no es válida y el p95 no se usa.

Lo comprueban `carga_y_calentamiento_fuera_de_la_medicion` (con un doble cuya carga y primeras llamadas son 20–30 veces más lentas; ninguna `medicion` las contiene), `p95_de_108_llamadas_y_de_36_medianas` y `percentil_rango_mas_cercano`.

**Separación del RNF-02 del sistema completo.** Este p95 mide **solo la llamada al detector**, en un proceso aislado y sin servidor. No incluye la red, la narrativa (Groq), la voz (Azure) ni el transporte HTTP. Por eso es una **condición necesaria** para seleccionar un modelo, no una prueba del RNF-02. El RNF-02 se evalúa **solo** con R2 (sección 6), sobre la solicitud completa y con criterio por solicitud. La tesis no usará este p95 para afirmar que el sistema cumple el RNF-02.

**Limitaciones declaradas.**
- Las 108 llamadas **no son observaciones independientes**: hay 3 repeticiones de la misma imagen y el mismo umbral, y los tres umbrales comparten la imagen. El p95 describe esta muestra; no es una estimación con intervalo de la distribución de tiempos.
- Con 108 valores, el p95 depende de los 6 mayores; con 36, de los 2 mayores. Ambos son inestables.
- Es una **condición necesaria** sobre la detección aislada. El cumplimiento del RNF-02 de extremo a extremo se comprueba solo con R2 (sección 6).

---

## 4. Validación, rúbrica y selección

### 4.0 Validación obligatoria antes de puntuar

`validar()` produce `{"valida", "errores", "avisos"}`. **Si `valida` es falso, `evaluar()` no puntúa ni selecciona ningún modelo**: devuelve `seleccionado = None` con el motivo y los errores, y no calcula la sensibilidad.

**Errores (cualquiera invalida la campaña):**
- falta el archivo crudo de algún modelo o alguno de sus registros (`entorno`, `pesos`, `carga`, `configuracion`, `calentamiento`, `memoria_tras_carga`, `medicion`, `fin`);
- algún archivo contiene un registro `error`;
- algún archivo es de una **prueba de humo** (`prueba_de_humo = true`);
- el archivo pertenece a otro modelo;
- faltan, sobran o se repiten combinaciones: deben ser exactamente las 12 imágenes del inventario × {0,3, 0,5, 0,7} × {1, 2, 3};
- hay etiquetas fuera de las 80 clases de COCO;
- los hilos efectivos al final son distintos de 4 (intra) y 1 (inter);
- YOLO26s principal no corrió a 1280 px;
- **algún archivo se midió con cambios sin commit** fuera de `evaluation/results/`;
- **los modelos (o la deriva o la corrida a 640) se midieron en commits distintos**, o falta el commit;
- falta el control de deriva, o la deriva es mayor del 10 %;
- el inventario no tenía commit al iniciar, está en otro commit, tiene cambios o su SHA-256 cambió (sección 1.3);
- **la configuración efectiva** de algún archivo (también la deriva y la corrida a 640 px) **difiere de `CONFIG_ESPERADA`**, o falta su registro;
- **las huellas de los pesos registradas** difieren de `huellas_esperadas.json`, faltan o la huella esperada es `null`;
- **la orquestación no terminó en estado `completa`** (detenida, en curso o sin `orquestacion.json`).

**Configuración fijada** (`CONFIG_ESPERADA`, leída del objeto cargado, no de las variables):

| Archivo | Valores exigidos |
|---|---|
| YOLO26s principal y deriva | `imgsz = 1280`, `iou_pasado = 0,45`, `end2end = True`, dispositivo `cpu` |
| YOLO26s a 640 px | Igual, con `imgsz = 640` |
| Faster R-CNN | `box_score_thresh = 0,05`, `box_nms_thresh = 0,5`, `box_detections_per_img = 100`, `min_size = [800]`, `max_size = 1333`, sin máscaras, dispositivo `cpu` (valores por defecto de torchvision; el servicio solo pasa los pesos) |
| Mask R-CNN | Igual, con máscaras |
| SSD | URL `https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2` y ninguna GPU visible |

**Qué ocurre en cada caso:**

| Situación | Qué hace el código | Resultado | Prueba |
|---|---|---|---|
| Falta una medición, una repetición o un registro | `validar()`: «incompletos o de más» o «falta el registro» | **Inválida**, sin selección | `campana_invalida_no_produce_recomendacion` (falta una repetición; faltan todas; falta un modelo; sin configuración) |
| Mediciones duplicadas o de otro umbral | `validar()` | **Inválida** | ídem (umbral distinto) |
| Inventario incompleto, sin commit o con cambios **antes** de empezar | `congelar_inventario()` detiene la orquestación | **No se mide** | `inventario_incompleto_no_se_puede_congelar`, `inventario_completo_sin_commit_o_alterado_no_se_congela` |
| Inventario alterado **después** de empezar | `validar()` compara commit, objeto de Git y SHA-256 | **Inválida** | 4 casos de inventario en `campana_invalida_no_produce_recomendacion` |
| Código con cambios sin commit | `exigir_commit()` detiene; `validar()` invalida si de todos modos aparecen | **No se mide** / **inválida** | `exigir_commit_se_niega_con_cambios`, casos «cambios sin commit» y «commits distintos» |
| Cambia una configuración (NMS, umbral interno, `end2end`, resolución, dispositivo, URL, hilos) | `validar()` compara con `CONFIG_ESPERADA` y los hilos con 4/1 | **Inválida** | casos de configuración, hilos, «YOLO principal a 640» y «deriva con otra configuración» |
| Pesos distintos o sin huella esperada | `modo_modelo()` no carga el modelo; `validar()` invalida si el registro no coincide | **No se mide** / **inválida** | `verificacion_de_huellas`, casos «pesos con otra huella», «huella esperada sin registrar», «sin huellas esperadas» |
| Error durante la carga, el calentamiento o una llamada | Registro `error` con traza; el proceso termina con código ≠ 0; la orquestación pasa a `detenida` | **Inválida**; se repite el modelo completo tras corregir y hacer commit | `error_del_servicio_queda_registrado_y_se_relanza`, casos «registro de error» y «orquestación detenida» |
| Archivo de prueba de humo mezclado | `validar()` | **Inválida** | caso «archivo de prueba de humo» |
| Deriva > 10 % o ausente | `validar()` | **Inválida** | casos de deriva |
| Repeticiones con detecciones distintas; conteos en el máximo | Aviso | Válida, se declara | `repeticiones_distintas_son_aviso_no_error` |

Una campaña inválida **no se corrige editando archivos**: se repite completa en una carpeta nueva, después de corregir la causa y hacer commit. La carpeta inválida se conserva y se informa.

**Además, el código se niega a medir:**
- si hay cambios sin commit (sección 7.2);
- si el inventario no está completo y versionado (sección 1.3);
- un modelo cuyos pesos no coinciden con `huellas_esperadas.json` o no tienen huella esperada (sección 7.1).

**Avisos (se informan, no invalidan):** repeticiones con detecciones distintas; corridas que alcanzan el máximo de detecciones (conteo truncado).

### 4.1 Procedimiento de selección — [APROBACIÓN]

1. **Elegibilidad: condición necesaria del RNF-02.** `p95 de las 108 llamadas(m) < 5000 ms` (sección 3.3). Si la detección sola dura 5 s o más en más del 5 % de las llamadas, en esas llamadas ninguna solicitud puede cumplir. No usa supuestos sobre las demás etapas. **Es necesaria, no suficiente**, y está separada del criterio de extremo a extremo de R2.
2. **Total:**
   ```
   Total(m) = Σ_{c ∈ C} puntos_c(m)
   C = {F1 de presencia, velocidad, recursos, sensibilidad al umbral,
        ausencia de falsos positivos, información espacial}
   Máximo = 6 × 4 = 24.  Todos los pesos son 1.
   ```
3. **Selección.** El elegible con mayor total. **Desempate:** la menor mediana del tiempo de inferencia. Si persiste el empate, **no hay ganador automático**: se declaran los empatados y la tesis argumenta la elección.
4. **Sin selección automática** si: la campaña no es válida (4.0), algún criterio queda incompleto (4.3), no hay ningún elegible o hay un empate persistente.

### 4.2 Criterios

| Criterio | Métrica | Dirección | Regla de empate (4.3) | Origen | ¿En el total? |
|---|---|---|---|---|---|
| F1 de presencia | F1 de presencia con 0,5 | Mayor es mejor | Exacta | Sustituye de forma declarada a la «precisión» de A3 (decisión P) | Sí |
| Velocidad | Mediana del tiempo de inferencia | Menor es mejor | Relativa, 5 % | A3 | Sí |
| Consumo de recursos | Mediana de los s de CPU por inferencia | Menor es mejor | Relativa, 5 % | A3 | Sí |
| Sensibilidad al umbral | Retención | Mayor es mejor | Exacta | Añadido en A10 y A11 | Sí |
| Ausencia de falsos positivos | Imágenes con al menos un FP de presencia con 0,5 | Menor es mejor | Exacta | Añadido en A10 y A11 (RNF-03) | Sí |
| Información espacial | Cualitativa (A11): Mask 3; YOLO, Faster y SSD 2 | — | — | A3 | Sí |
| Facilidad de integración | Cualitativa (A11): 3 / 3 / 3 / 3 | — | — | A3 | **No** (decisión I) |

**Por qué la integración sale del total.** Da 3 a los cuatro modelos: sumarla agrega una constante y **no cambia ninguna clasificación**; solo infla el máximo (28 en lugar de 24). Se conserva en la tabla como descripción. El código lo comprueba: con la integración, todos los totales suben exactamente 3. Si en la versión final de A11 dejara de ser idéntica, volvería a entrar en el total y habría que declararlo.

**Por qué la información espacial sí entra.** Distingue una capacidad real (máscaras frente a solo recuadros) y viene de A3 con importancia «Media». Es cualitativa y A3 dice que la segmentación «no es indispensable para el alcance»; por eso la sensibilidad la quita.

**La tesis debe declarar** que la sensibilidad al umbral y los falsos positivos no estaban en A3, y que F1 de presencia sustituye a la precisión con mAP e IoU de A3 (sección 4.5).

### 4.3 Puntos, reglas de empate y valores faltantes

**Fórmula:** `puntos_c(m) = 4 − |{o ≠ m : o es estrictamente mejor que m en c}|`.

`o` es **estrictamente mejor** que `m` si `(v_o − v_m)·d_c > 0` (con `d_c = +1` si mayor es mejor y `−1` si menor es mejor) **y la diferencia no es un empate según la regla del criterio**.

#### Alternativas comparadas para decidir qué es un empate

| Alternativa | Qué hace | Problema | Uso en v5 |
|---|---|---|---|
| **A. Relativa (τ = 5 %)** | Empate si `\|Δ\| / max(\|v_o\|, \|v_m\|) < 5 %` | En proporciones, la tolerancia depende del nivel: casi nula cerca de 0 y de varios pares cerca de 1 | **Principal para tiempo y CPU**, que tienen ruido de medición |
| **B. Relativa + absoluta fija (p. ej. 0,02)** | Empate si la diferencia es menor que 5 % o que 0,02 | 0,02 no corresponde a ninguna unidad de las métricas ni puede justificarse antes de ver los datos | **Descartada** |
| **C. Tolerancia absoluta común** | Empate si `\|Δ\| < ε`, con ε igual para los cuatro modelos | En métricas deterministas no absorbe ruido: iguala diferencias reales. Es una tolerancia práctica, no una resolución | **Solo sensibilidad** (F1, retención, imágenes con FP) |
| **D. Bootstrap pareado por imagen** | Remuestrear las 12 imágenes | Con 12 imágenes casi todo sería empate; requiere fijar remuestreos, semilla y nivel | **Descartada** como regla de puntuación |
| **E. Exacta** | Empate solo si los valores son iguales | Una diferencia de un solo par o una sola detección ya da un punto | **Principal para F1, retención e imágenes con FP**; su fragilidad se examina con las variantes C |

#### Reglas principales (aprobadas el 27-oct-2026, antes de medir)

| Criterio | Regla principal | Variantes de sensibilidad |
|---|---|---|
| Velocidad | Relativa, **τ = 5 %** | τ = 0 % y 10 % |
| Recursos (CPU) | Relativa, **τ = 5 %** | τ = 0 % y 10 % |
| F1 de presencia | **Exacta** | Tolerancia común `1/|P|` (`f1_tolerancia_1_par`) y `2/|P|` (`f1_tolerancia_2_pares`) |
| Retención | **Exacta** | Unidad común `max_m 1/Σ_i n(m,i,0,3)` (`retencion_unidad_comun`) |
| Imágenes con FP | **Exacta** | Tolerancia de una imagen (`fp_tolerancia_1_imagen`: empate si difieren en una imagen o menos) |

#### Justificación

**F1 de presencia: por qué la regla exacta.**
- **Cálculo:** `F1 = 2·VP / (2·VP + FP + FN)`. `|P| = Σ_i |P_i|` es el número de celdas `1` del inventario. Como cada par presente es acierto u omisión, `VP + FN = |P|` para todos los modelos, y por tanto `F1 = 2·VP / (VP + |P| + FP)`.
- **Efecto del inventario:** `|P|` lo fija el inventario, registrado en un commit antes de medir. Las celdas `?` no entran en `|P|`.
- **El paso de un par no es constante.** Sea `S = VP + |P| + FP`:
  - si una omisión pasa a acierto, `ΔF1 = 2·(|P| + FP) / (S·(S+1))`;
  - si hay un falso positivo menos, `ΔF1 = 2·VP / (S·(S+1))`.
  Con `|P| = 60`, el cambio de un par va de 0,0002 a `2/(|P|+1)` = 0,033. En la zona esperable (F1 entre 0,6 y 0,97) va de 0,006 a 0,014, **menos que** `1/|P|` = 0,0167. El cálculo exacto está en `Revision_reglas_empate_y_rubrica_v5_2026-10-27.md`.
- **`1/|P|` no es la granularidad de F1.** Equivale aproximadamente a una tolerancia de **uno o dos pares**, y `2/|P|` a dos a cuatro pares. Además, dos modelos pueden diferir en muchos pares con F1 casi iguales: una distancia en F1 no se traduce en un número de pares.
- **Por qué es exacta:** las detecciones son deterministas en CPU (las repeticiones lo comprueban) y el inventario está fijado, así que F1 no tiene ruido de medición. **Toda diferencia distinta de cero corresponde al menos a un par imagen–clase distinto según el inventario.**
- **Incertidumbre que no se modela:** un solo anotador, casos dudosos y 12 imágenes. Ninguna tolerancia fija la representa. Se examina con las variantes de 1 y 2 pares: si la selección cambia, la sección 4.6 la declara no estable.
- **Informe:** VP, FP y FN de cada modelo junto a F1 (`metricas.json`), para que la diferencia en pares sea visible.

**Retención: por qué la regla exacta.**
- **Escala:** `R(m) = A_m / B_m`, con `A_m = Σ_i n(m,i,0,7)` y `B_m = Σ_i n(m,i,0,3)` enteros. Es adimensional y está entre 0 y 1 en la práctica. Una detección más con 0,7 cambia `R(m)` exactamente en `1/B_m`.
- **Por qué es exacta:** los conteos son deterministas, así que una diferencia menor que `1/B_m` sigue siendo real.
- **Unidad común (solo sensibilidad):** `ε_R = max_m 1/B_m = 1/min_m B_m`, una detección del modelo que menos detecta con 0,3. Ejemplo: `B` = 200, 150, 120 y 300 → `ε_R` = 0,0083; 0,483 frente a 0,489 es empate en la variante y diferencia en la regla principal. La fórmula está fijada antes de medir, pero su valor depende de los conteos.
- **Límite que ninguna regla corrige:** la retención compara confianzas de arquitecturas distintas, que no están en la misma escala (§3.1). La variante `sin_retencion` muestra su peso.

**Tiempo y CPU: denominador de la regla relativa.**
- **Fórmula:** `empate ⇔ |v_o − v_m| / max(v_o, v_m) < 0,05 ⇔ min/max > 0,95 ⇔ |ln(v_o/v_m)| < 0,0513`.
- **Denominador:** el **mayor** de los dos valores. La regla es simétrica (no depende del orden), está acotada entre 0 y 1 y no depende de la unidad. Con el menor como denominador el resultado sería casi idéntico (`max/min < 1,05`).
- **Ejemplos:** 812 frente a 845 ms → 3,9 % → empate; 812 frente a 900 ms → 9,8 % → diferencia; 1,20 frente a 1,25 s de CPU → 4,0 % → empate.
- **Por qué aquí sí hay tolerancia:** el tiempo tiene **ruido de medición** (procesos de fondo, frecuencia de la CPU, temperatura). **τ = 5 % es una convención fijada a priori**; estimar el ruido con la propia campaña sería decidir después. Variantes: 0 y 10 %.
- **Riesgo sin verificar:** en Windows, el tiempo de CPU del proceso puede acumularse en pasos de unos 15,6 ms por hilo. Se comprobará en la prueba de humo; si existe, se declara en la tesis sin cambiar la regla.

**Imágenes con FP: limitaciones de la igualdad exacta.** Es un conteo entero de 0 a 12, así que la unidad mínima ya es una imagen. Limitaciones:
- una sola imagen da un punto;
- una celda discutible del inventario (`0` en lugar de `?`) puede cambiarlo;
- no distingue una imagen con un FP de una con cinco;
- los FP ya penalizan F1, así que el mismo error cuenta dos veces.

La variante de una imagen y `sin_falsos_positivos` muestran su peso.

**Cómo se garantiza que no se eligen después.**
- τ, la regla de cada criterio (`CRITERIOS`) y las tolerancias de las variantes (`_tolerancia_absoluta`) están en el código que se registra en un commit antes de medir.
- `validar()` invalida la campaña si algún archivo se midió con cambios sin commit o en commits distintos (secciones 4.0 y 7.2).
- El resumen no acepta tolerancias por línea de comandos.
- **Las reglas no se cambian según el modelo que resulte ganador.**

**Efecto en el total.** Cada empate da a los dos modelos los mismos puntos; un cambio de regla mueve como mucho 1 punto por criterio y modelo. La selección solo es estable si **todas** las variantes de la sección 4.6 eligen el mismo modelo.

**Coherencia.**
- **Monotonía:** la regla relativa, la exacta y las tolerancias absolutas comunes son monótonas: si `m` es mejor que `m'` en valor, `puntos(m) ≥ puntos(m')`.
  - Relativa: si `o < m < m'` (menor es mejor) y `(m − o)/m ≥ τ`, entonces `(m' − o)/m' ≥ τ`.
  - Tolerancia común: si `|v_o − v_m| ≥ ε` y `m'` está más lejos de `o`, también `|v_o − v_m'| ≥ ε`.
  - Una tolerancia distinta para cada par de modelos rompería la monotonía (lo muestra `regla_de_una_unidad_con_resoluciones_distintas_sigue_siendo_monotona`); por eso siempre es común.
  - `test_rubrica_es_monotona` recorre todas las combinaciones de 4 valores de una lista.
- **Intransitividad:** 100 ≈ 104 y 104 ≈ 108, pero 100 es mejor que 108: da 4, 4 y 3, coherente con el orden.

**Valores faltantes:** si a un modelo le falta la métrica de un criterio (o, en una variante de tolerancia absoluta, la tolerancia), el criterio queda **incompleto para todos** y no hay selección automática. Un modelo que falle se repite **completo**, nunca de forma parcial, y se informa.

### 4.4 Limitaciones del total

- **Criterios que pesan doble:** velocidad y CPU miden el mismo costo de cómputo; F1 y falsos positivos usan el mismo inventario.
- **La puntuación por posición pierde la magnitud.** Las tablas muestran los valores medidos junto a los puntos.
- **Un criterio cualitativo no se mide.**
- **Son 12 imágenes.**

### 4.5 Criterio de precisión y mAP oficial — [APROBACIÓN: decisión P]

**Situación:**
- A3 define la precisión con «mAP, IoU».
- La Tabla 3 de la tesis la mide hoy con «confianza promedio y revisión visual». **El criterio ya cambió y la tesis no lo declara.**
- Sin recuadros anotados, ninguna métrica de esta campaña es mAP.

**mAP oficial** (verificado el 27-oct-2026):

| Modelo | Valor publicado | Condiciones | ¿Corresponde a la campaña? |
|---|---|---|---|
| Faster R-CNN, `…coco-dd69338a.pth` | box mAP 46,7 (COCO val2017) | Metadatos de los pesos en torchvision | Sí: mismos pesos y transformación |
| Mask R-CNN, `…coco-73cbd019.pth` | box mAP 47,4; mask mAP 41,8 (COCO val2017) | Ídem | Sí |
| YOLO26s | 48,6 (cabeza con NMS) y 47,8 (de extremo a extremo), mAP val 50–95 | **640 px** | No del todo: la campaña usa 1280 px. El aplicable es 47,8, medido a 640 |
| SSD-MobileNet V2 (TF Hub `/2`) | COCO mAP 20,2 | Zoo de TF2; **no se indican** la partición ni el protocolo | Probablemente (320 × 320), sin condiciones documentadas |

**No son comparables para puntuar** y **no puntúan**. Se informan como referencia citada (P2′).

**Opciones:**
- **P1. Anotar recuadros** de las 12 imágenes y medir AP@0,5 y mAP@[0,5:0,95] con `pycocotools`. Única opción que cumple A3 con medición propia. Costo alto; con 12 imágenes el mAP es inestable; difícil antes del 30-oct.
- **P3. Cambio formal y declarado del criterio** a **F1 de presencia**. **No equivale al criterio de A3 y no se presentará como equivalente.** La tesis debe decir en 3.1 que A3 definía la precisión con mAP e IoU, que sin recuadros no se puede calcular, que se sustituye por F1 de presencia con sus límites (sección 1.4), corregir la Tabla 3 y añadir la tabla de mAP oficial como referencia (P2′).

**Recomendación: P3 + P2′**, por el plazo; **P1** si hay tiempo. El código implementa P3.

#### Qué se podrá afirmar con F1 de presencia (si se aprueba P3)

| Se puede sostener | No se puede sostener |
|---|---|
| «En las 12 imágenes del proyecto, con umbral 0,5, el modelo X nombró las clases presentes según el inventario con un F1 de presencia de …» | Que un modelo tiene mayor **precisión de detección**, mAP o AP que otro |
| Cuántas clases presentes omitió cada modelo (FN) y cuántas nombró sin estar (FP), y en cuántas imágenes | Que localiza mejor los objetos (no hay recuadros ni IoU) |
| Qué clases concretas se nombraron sin estar o se omitieron | Que cuenta correctamente cuántos objetos hay de cada clase |
| La comparación entre los cuatro modelos **en este conjunto**, con las mismas imágenes, umbral y equipo | Que el resultado se generaliza a otras escenas o al uso real (12 imágenes de una sola persona) |
| Que la narrativa, al usar ese detector, nombraría o no ciertas clases en estas escenas | Que las diferencias son estadísticamente significativas (no hay prueba ni intervalo) |
| Que el inventario no cambió desde su commit | Que el inventario es independiente o ciego (sección 1.2) |
| | Que F1 de presencia equivale, aproxima o sustituye al mAP de A3 |

#### Uso del mAP oficial solo como referencia (P2′)

**Por qué no puede compararse con los resultados de la campaña:**
- **Otro conjunto:** COCO val2017 (5000 imágenes anotadas por terceros), no las 12 del proyecto.
- **Otra métrica:** área bajo la curva precisión–recall con emparejamiento por IoU, promediada sobre clases y umbrales de IoU; F1 de presencia es un cociente de pares imagen–clase con un único umbral de confianza.
- **Otras condiciones:** YOLO26s se midió a 640 px (la campaña usa 1280); para SSD no se publican la partición ni el protocolo.
- **No es una medición de este trabajo:** no puede verificarse con el código del repositorio.

**Cómo se evita la comparación indebida:**
1. El código **no lee ni usa** el mAP oficial: no entra en `CRITERIOS`, en el total, en la elegibilidad ni en la sensibilidad.
2. En la tesis va en una **tabla separada**, titulada como valores publicados por los autores de cada modelo, con la fuente, el conjunto, la resolución y las condiciones de cada valor, y con la nota de que no se midieron en este trabajo.
3. **No comparte tabla ni columna** con F1 de presencia ni con ningún resultado de la campaña.
4. **No se ordena ni se puntúa** a los modelos por el mAP oficial, ni se combina con resultados propios.
5. La frase de referencia es del tipo «según sus autores, en COCO val2017 a 640 px, YOLO26s alcanza …», nunca «YOLO26s es más preciso que …».
6. La corrida complementaria a 640 px solo sirve para describir el efecto de la resolución en los tiempos y las detecciones de este conjunto; **no reproduce ni valida** el mAP publicado.

### 4.6 Análisis de sensibilidad (fijado antes de ejecutar)

`sensibilidad()` recalcula la selección en estas variantes:

| Variante | Qué cambia |
|---|---|
| `tolerancia_continuas_0%`, `_5%` (principal), `_10%` | τ de velocidad y recursos |
| `f1_tolerancia_1_par`, `f1_tolerancia_2_pares` | F1 con tolerancia común `1/|P|` y `2/|P|` (≈ 1–2 y 2–4 pares) |
| `retencion_unidad_comun` | Retención con tolerancia común `max_m 1/Σ_i n(m,i,0,3)` |
| `fp_tolerancia_1_imagen` | Imágenes con FP: empate si difieren en una imagen o menos |
| `precision_en_lugar_de_f1` | El criterio usa la precisión de presencia (versión 4) |
| `recall_en_lugar_de_f1` | El criterio usa el recall de presencia |
| `elegibilidad_p95_36_medianas` | Elegibilidad con el p95 de las 36 medianas |
| `pesos_A3` | Importancia de A3 como peso: Alta = 2 (F1, velocidad, recursos), Media = 1 (información espacial, integración); los dos criterios ajenos a A3 = 1. Incluye la integración |
| `con_integracion` | Suma el criterio descriptivo (debe dar el mismo resultado) |
| `sin_cualitativos` | Sin información espacial |
| `sin_<criterio>` | Quita, uno a la vez, cada uno de los 6 criterios del total |

**Regla de estabilidad:** la selección es **estable solo si todas las variantes seleccionan un modelo y es el mismo en todas**. Si alguna no selecciona o cambia el ganador, la tesis lo declara, indica de qué decisión depende y argumenta la elección con los criterios que deciden, sin presentar el total como prueba.

**Cómo se evita que una decisión arbitraria cambie la selección:** las reglas, τ, las unidades, la elegibilidad, los pesos y las variantes quedan en un commit antes de medir; las puntuaciones cualitativas son las de A11 sin cambios; todas las variantes se informan; las métricas medidas se publican junto a las puntuaciones.

---

## 5. Comparabilidad de los cuatro modelos

### 5.1 Condiciones comunes

- **Imágenes:** las 12, con su huella. `resize_image()` del servicio las reduce **una vez** al lado mayor de 800 px en JPEG 90, y todos los modelos leen **los mismos bytes**.
- **Clases y umbral:** todas las clases de COCO, sin filtro de navegación, sin umbrales por clase y sin la regla del mínimo.
- **Dispositivo:** CPU (`CUDA_VISIBLE_DEVICES=-1`).
- **Hilos:** intra 4, inter 1. Se fijan después de importar las bibliotecas, se registran al final y se validan.
- **Aislamiento:** un proceso por modelo, con la carga y 3 llamadas de calentamiento fuera de la medición.
- **Repeticiones:** 3 por corrida.
- **Orden:** YOLO26s → Faster → Mask → SSD, con 60 s de pausa; al final, el control de deriva (YOLO26s con 0,5) y, si se pide, YOLO26s a 640 px.
- **Límite de la medición:** la función de servicio de cada modelo.

### 5.2 Diferencias propias de cada arquitectura (se declaran en la tesis)

| Aspecto | YOLO26s (principal) | Faster R-CNN | Mask R-CNN | SSD-MobileNet V2 |
|---|---|---|---|---|
| Pesos | `yolo26s.pt` de producción (SHA-256 646f8bc3…) | torchvision `…FPN_V2…DEFAULT` | ídem, Mask | TF Hub `ssd_mobilenet_v2/2` |
| Resolución | **1280, la de producción**, fijada por el orquestador y validada. La nativa de entrenamiento es 640 | Lado menor 800 y mayor 1333 | Ídem | 320 × 320 |
| Supresión de duplicados | **Ninguna: extremo a extremo** (`end2end = True` en los pesos). `ultralytics/utils/nms.py` (línea 66) omite el NMS. El `iou = 0,45` de producción **no tiene efecto**; la campaña lo pasa igual que producción y lo registra | NMS de torchvision (IoU 0,5) | Ídem | NMS dentro del grafo |
| Umbral | `conf = umbral`, sin la confianza interna de 0,15 ni el filtro por clase de producción | Filtro del servicio | Ídem | Ídem |
| Máximo de detecciones | 300 | 100 | 100 | 100 |
| Trabajo adicional | — | — | Calcula máscaras | — |
| Entorno | `.venv` de referencia (Python 3.13.15) | Ídem | Ídem | Entorno aparte de TensorFlow |

Todo lo anterior se **registra como configuración efectiva** leída del objeto cargado (sección 7.3) y **se valida** contra `CONFIG_ESPERADA` (sección 4.0): cualquier diferencia invalida la campaña.

**Análisis complementario de YOLO26s a 640 px.** La configuración de producción (1280 px) es la **principal y la única que puntúa**. La corrida a 640 px (`--yolo-640`, etiqueta `yolo_640`) usa el mismo commit y entorno, se valida en cuanto a commit, se guarda en `metricas.json` bajo `complementario_yolo_640_no_puntuado` y **no entra en el total, en la elegibilidad ni en la sensibilidad**. Sirve para relacionar los resultados con el mAP publicado, que se midió a 640.

**Otros riesgos:**
- **Máximos de detecciones.** Si alguna corrida llega al máximo, queda un aviso y la tesis debe declararlo.
- **SSD en TensorFlow.** Se compara el detector tal como se usaría, no a igualdad de código.
- **Memoria base.** La de TensorFlow es mayor y se informa aparte.

### 5.3 Exclusión de la carga y el calentamiento

`medir_modelo()` registra, en este orden: `carga`, `configuracion`, 3 de `calentamiento`, `memoria_tras_carga`, una `medicion` por llamada y `fin`. Si algo falla, escribe un registro `error` y se detiene. Ninguna métrica usa la carga ni el calentamiento; se informan aparte. La prueba simulada lo verifica; **no demuestra el comportamiento de los modelos reales**.

### 5.4 Memoria en Windows

| Valor | Qué es | Cuándo se toma |
|---|---|---|
| `rss` | Conjunto de trabajo actual | Tras importar, tras la carga y el calentamiento, y tras cada llamada |
| `peak_wset` | Máximo del conjunto de trabajo en **toda la vida del proceso**, incluida la carga | Al final |
| `peak_pagefile` | Máximo de memoria privada comprometida en toda la vida del proceso | Al final |

**Ninguno es el pico durante la inferencia.** En la Tabla 5 se presenta «memoria del proceso con el modelo listo» (RSS tras la carga y el calentamiento). El máximo de RSS tras inferencia y `peak_wset` se informan como complementarios.

---

## 6. Definición operativa del RNF-02 y prueba R2

**Requisito** (Tabla 13): «El sistema debe responder en un tiempo total de procesamiento inferior a 5 segundos por solicitud operando sobre CPU».

**Definición operativa:**
- **Qué se mide.** El tiempo de cada solicitud `POST /api/detect`, medido en el cliente desde el instante anterior al envío hasta tener la respuesta completa (narrativa y audio de Azure AI Speech, es-CO-SalomeNeural).
- **Condiciones:** equipo de referencia (i5-8300H, 4 núcleos y 8 hilos, 32 GB, Windows 11, solo CPU); **perfil `production`** y **entorno de referencia** (`.venv`, Python 3.13.15), en uvicorn con 1 proceso; la medición empieza después del arranque normal (verificación de identidad, carga y calentamiento de YOLO26s); servicios externos reales (Groq y Azure).
- **Criterio de aceptación, por solicitud.** Cada solicitud debe ser **exitosa** (HTTP 200, `status = success`, audio disponible, sin degradación y voz `azure:es-CO-SalomeNeural`) y durar **menos de 5,0 s**. El RNF-02 se declara cumplido en la muestra **solo si se registraron las 30 y las 30 cumplen** (`cumple_por_solicitud`). La media, la mediana y el p95 son descriptivos.
- **Muestra:** **una sola ejecución de 30 solicitudes** (6 escenas × 5 rondas intercaladas). **No hay segunda ejecución.** Las 30, incluidas las fallidas y las lentas, son el resultado. Si fallan los proveedores, se informa así y el RNF-02 no se declara cumplido. Aunque las 30 cumplan, la tesis dirá «en las 30 solicitudes medidas».

**El cronómetro cubre el intervalo que exige el requisito** (`medir_solicitud()`): `t0` se toma después de leer la imagen y antes de `cli.post(...)`; `httpx.Client.post` sin streaming devuelve después de leer el cuerpo completo. El transporte es local, así que la medida es conservadora. La latencia de Groq y Azure por Internet está incluida y es fuente de variabilidad.

### 6.1 Contabilidad de las 30 solicitudes

`medir_solicitud()` **nunca lanza una excepción**: cada solicitud produce una fila, con su `ts_envio` (UTC con milisegundos).

| Caso | Registro |
|---|---|
| Error de transporte (`httpx.HTTPError`) | Fallida, `status = None`, motivo con el tipo de error y tiempo hasta el error |
| Tiempo de espera del cliente (120 s) | Igual, con motivo `ReadTimeout` u otro `Timeout*` |
| Respuesta no JSON (por ejemplo, 502 de texto) | Fallida, motivo `respuesta_no_json:<content-type>` |
| JSON inválido | Fallida, motivo `json_invalido` |
| Código ≠ 200, `status ≠ success`, sin audio, degradación o voz distinta | Fallida, un motivo por cada condición |
| ≥ 5 s y exitosa | Aparece en `solicitudes_lentas` |

`resumir_rnf02(filas, n_esperadas=30)` añade `n_esperadas` y `registro_completo` (`n = 30`). **Si faltara alguna fila, `cumple_por_solicitud` es falso.**

### 6.2 Verificación previa y registro del servidor

- **Árbol limpio:** el script se niega a medir si hay cambios sin commit fuera de `evaluation/results/`.
- **Registro del servidor:** la salida estándar y de error de uvicorn (nivel `info`) se guarda en `evaluation/results/plan_pruebas/nf_r2_<fecha>_servidor.log`, y su ruta queda en el JSON de resultados. Al terminar, `contar_log_servidor()` cuenta las líneas de acceso `POST /api/detect` por código HTTP y las compara con los códigos de las filas del cliente; registra también las solicitudes sin respuesta (que no llegan al servidor) y el número de trazas de excepción del registro. Si no coinciden (`coincide = false`), la tesis debe explicarlo antes de usar el resultado.
- **Verificación previa** (`verificar_salud()`), **antes de enviar cualquier solicitud de medición**, sobre `GET /api/health`: perfil `production`, identidad `verificada`, narrativa activa, voz activa y modelo de voz `azure:es-CO-SalomeNeural`. Si algo falla, el script termina sin medir.
- **`--solo-verificar`:** arranca el servidor, hace la verificación previa y termina **sin enviar solicitudes de medición**. El arranque carga YOLO26s y lee la configuración de los proveedores.

**Configuración controlada sin tocar producción.** Servidor local con variables propias: `APP_PROFILE=production`; `RATE_LIMIT_IP_ENABLED=false` y `RATE_LIMIT_REQUESTS=1000` (30 solicitudes en unos 2 minutos superan el límite de 6 por minuto; no cambian el procesamiento); `YOLO_ALLOW_DOWNLOAD=false` y `CUDA_VISIBLE_DEVICES=-1`. Las variables de la terminal prevalecen sobre el `.env`. No se modifica el `.env`, la configuración del repositorio ni el despliegue.

**Evidencia anterior que se conserva sin cambios:** `nf_302db7e_es-CO-SalomeNeural.json` (Azure, 12-oct, n = 6, perfil development) y `nf_fcf80e9.json` (Gemini, 11-oct). La evidencia nueva va en `nf_r2_<commit>_<fecha>.json`.

---

## 7. Seguridad, registro y reproducibilidad

### 7.1 Huellas de los pesos

- `huellas_esperadas.json`: YOLO26s tiene la suya (`646f8bc3…a384a1b`); Faster R-CNN, Mask R-CNN y SSD están en `null` hasta la descarga aprobada (decisión S5).
- **Mientras una huella sea `null` o no coincida, el código se niega a medir ese modelo**, antes de cargarlo.
- En torchvision, los 8 caracteres del nombre del archivo son el prefijo del SHA-256 que publica PyTorch y la descarga lo comprueba.
- **Sin descargas inesperadas:** torchvision y TF Hub usan la caché llenada en el paso aprobado; en R2, `YOLO_ALLOW_DOWNLOAD=false`.

### 7.2 Ningún cambio sin commit

- `modelo` (salvo `--humo`) y `orquestar` ejecutan `git status --porcelain --untracked-files=all` y **se niegan a medir** si hay cambios fuera de `evaluation/results/` (la carpeta que la propia ejecución escribe). Incluye archivos nuevos sin rastrear.
- Cada registro `entorno` guarda el commit y los cambios sin commit; `validar()` invalida la campaña si hay cambios o si los modelos se midieron en commits distintos (sección 4.0).
- `plan_pruebas_nf_r2.py` aplica la misma regla.

### 7.3 Qué se registra en cada ejecución

| Registro | Contenido | Dónde |
|---|---|---|
| Fechas y horas | `ts` (UTC, milisegundos) en **cada** registro; inicio y fin de cada paso y de la campaña | `crudo_*.jsonl`, `orquestacion.json` |
| Estado de la campaña | `en_curso`, `detenida` (con el motivo y el paso) o `completa` | `orquestacion.json` |
| Errores | Registro `error` con la fase, la imagen, el umbral, la repetición, la excepción y la traza | `crudo_*.jsonl` |
| Salida de cada paso | Salida estándar y de error de cada proceso | `logs/<etiqueta>.log` |
| Configuración efectiva | Leída del objeto cargado: YOLO (`imgsz`, `iou` pasado, `end2end`, `max_det` de la cabeza, `imgsz` de entrenamiento, dispositivo); torchvision (`score_thresh`, `nms_thresh`, `detections_per_img`, tamaños mínimo y máximo, máscaras, dispositivo); SSD (URL, firma de entrada, dispositivos visibles) | registro `configuracion` |
| Equipo y sistema | CPU, núcleos, RAM, sistema, plan de energía, conexión a la corriente | registro `entorno` |
| Python y dependencias | Versión y ejecutable; versiones de los paquetes principales; **lista completa de paquetes instalados** | registro `entorno` |
| Variables | Hilos, dispositivo, pesos, `YOLO_IMGSZ`, `YOLO_IOU`, perfil, etc. Hilos efectivos al final | `entorno`, `fin` |
| Código | Commit, rama, cambios sin commit y SHA-256 de los archivos de código | registro `entorno` |
| Datos | SHA-256 de cada imagen preparada (`imagenes.json`); commit, objetos de Git y SHA-256 del inventario | `imagenes.json`, `orquestacion.json` |
| Pesos | SHA-256 de cada archivo de pesos | registro `pesos` |
| R2 | `ts_envio` por solicitud, registro del servidor, verificación previa, entorno y commit | `nf_r2_*.json`, `nf_r2_*_servidor.log` |

**Entornos:**

| Entorno | Uso | Versiones conocidas hoy |
|---|---|---|
| `.venv` de referencia | YOLO26s, Faster, Mask y R2 | Python 3.13.15, torch 2.13.0+cu126, torchvision 0.28.0+cu126, ultralytics 8.4.123 (comprobado) |
| `.venv-ssd` | SSD | **Por crear.** Versiones exactas tras la instalación aprobada |

**Qué no garantiza la reproducibilidad.** Que el código compile y pase pruebas simuladas no demuestra que la medición sea reproducible. Lo reproducible es el **procedimiento** y las **detecciones** (deterministas en CPU, lo que se comprueba con las repeticiones). Los **tiempos** dependen del equipo, de su estado y de la red: son reproducibles en distribución, no exactamente.

---

## 8. Prueba de humo real (separada de la campaña)

**Propósito.** Comprobar, con los modelos reales, que cada paso carga, se configura y produce registros con la forma esperada **antes** de la campaña. **No es evidencia**: sus tiempos no se usan ni se informan, y sus archivos no pueden pasar por campaña (`prueba_de_humo = true` invalida cualquier resumen).

**Estado:** **no ejecutada.** No se ejecutará hasta que se aprueben las descargas.

**Requisitos previos:**
1. Descargas aprobadas y realizadas (sección 10).
2. `.venv-ssd` creado.
3. Huellas de Faster, Mask y SSD registradas en `huellas_esperadas.json` (sin ellas el código se niega a cargar esos modelos).
4. Equipo conectado a la corriente.
5. **No** requiere el inventario ni un commit (es la única excepción a la regla 7.2, y por eso sus resultados no son válidos como campaña).

**Comando** (la carpeta de salida **debe estar fuera de `evaluation/`**; el código lo exige):

```powershell
$env:YOLO_WEIGHTS = "D:\University\T2\Programacion\vision-api-object-detection\yolo26s.pt"
$env:YOLO_WEIGHTS_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
$env:YOLO_ALLOW_DOWNLOAD = "false"
D:\University\T2\Programacion\vision-api-object-detection\.venv\Scripts\python.exe scripts\evaluation\campana_definitiva.py humo --salida D:\University\T2\Programacion\humo_campana_AAAAMMDD --python-ssd D:\University\T2\Programacion\.venv-ssd\Scripts\python.exe
```

Ejecuta, por modelo, 1 imagen, umbral 0,5 y 1 repetición (tras la carga y las 3 llamadas de calentamiento), y escribe `humo.json` y `logs/`.

**Criterios de aprobación (todos):**

| # | Criterio | Dónde se comprueba |
|---|---|---|
| 1 | Los cuatro pasos terminan con código 0 | `humo.json` |
| 2 | Ningún registro `error` | `crudo_*.jsonl` |
| 3 | Registro `pesos` con huellas que coinciden con las esperadas | `crudo_*.jsonl` |
| 4 | Registros `carga`, `configuracion`, 3 `calentamiento`, `memoria_tras_carga`, `medicion` y `fin` presentes y en ese orden | `crudo_*.jsonl` |
| 5 | Hilos efectivos 4 / 1 | registro `fin` |
| 6 | YOLO26s con `imgsz = 1280` y `end2end = True` | registro `configuracion` |
| 7 | Configuración de torchvision igual a la de la sección 5.2 | registro `configuracion` |
| 8 | Todas las etiquetas dentro de las 80 clases de COCO | registro `medicion` |
| 9 | Ninguna descarga durante el paso | `logs/*.log` |
| 10 | Registrar (no es criterio de rechazo) si `cpu_s` toma valores en múltiplos de un paso fijo (cuantización del tiempo de CPU en Windows, §4.3); si ocurre, se declara en la tesis sin cambiar la regla | registros `calentamiento` y `medicion` |

Si alguno falla, se corrige el código, se hace commit y se repite la prueba de humo; **la campaña no se inicia** hasta aprobarla. La carpeta de la prueba de humo no se añade al repositorio.

---

## 9. Comandos de la campaña y de R2

Desde `D:\University\T2\Programacion\vision-api-campana-definitiva`.

### 9.1 Preparación — [APROBACIÓN de las descargas y del plan de energía]

1. Plan de energía en «Alto rendimiento» (hoy está en «Equilibrado») y equipo conectado. Lo hace el autor.
2. Entorno de SSD:
   ```powershell
   py -3.13 -m venv D:\University\T2\Programacion\.venv-ssd
   D:\University\T2\Programacion\.venv-ssd\Scripts\python.exe -m pip install -r requirements-experimental-ssd.txt
   ```
   Si TensorFlow no tiene versión para 3.13, se usa `py -3.12` y se declara.
3. Descarga de los pesos (no mide nada):
   ```powershell
   D:\University\T2\Programacion\vision-api-object-detection\.venv\Scripts\python.exe -c "from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as F, MaskRCNN_ResNet50_FPN_V2_Weights as M; F.DEFAULT.get_state_dict(progress=True, check_hash=True); M.DEFAULT.get_state_dict(progress=True, check_hash=True)"
   D:\University\T2\Programacion\.venv-ssd\Scripts\python.exe -c "import tensorflow_hub as hub; print(hub.resolve('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'))"
   ```
4. Registrar las huellas en `huellas_esperadas.json`.
5. Prueba de humo (sección 8) y su aprobación.
6. Inventario completo (sección 1).
7. **Commit** del protocolo, el código, las huellas y el inventario.

### 9.2 Pruebas del protocolo (simuladas)

```powershell
D:\University\T2\Programacion\.venv-py312\Scripts\python.exe -m pytest tests\test_campana_definitiva.py -q
```

### 9.3 Campaña definitiva — [APROBACIÓN] (40–60 min)

```powershell
$env:YOLO_WEIGHTS = "D:\University\T2\Programacion\vision-api-object-detection\yolo26s.pt"
$env:YOLO_WEIGHTS_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
$env:YOLO_ALLOW_DOWNLOAD = "false"
D:\University\T2\Programacion\vision-api-object-detection\.venv\Scripts\python.exe scripts\evaluation\campana_definitiva.py orquestar --salida evaluation\results\campana_definitiva_AAAAMMDD --inventario evaluation\campana_definitiva\inventario_12_imagenes.csv --python-ssd D:\University\T2\Programacion\.venv-ssd\Scripts\python.exe --yolo-640
D:\University\T2\Programacion\.venv-py312\Scripts\python.exe scripts\evaluation\campana_definitiva_resumen.py --salida evaluation\results\campana_definitiva_AAAAMMDD --inventario evaluation\campana_definitiva\inventario_12_imagenes.csv
```

### 9.4 Prueba R2 — [APROBACIÓN] (30 solicitudes, una sola ejecución)

```powershell
$env:YOLO_WEIGHTS = "D:\University\T2\Programacion\vision-api-object-detection\yolo26s.pt"
$env:YOLO_WEIGHTS_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
D:\University\T2\Programacion\vision-api-object-detection\.venv\Scripts\python.exe scripts\evaluation\plan_pruebas_nf_r2.py --env-file D:\University\T2\Programacion\vision-api-object-detection\.env --solo-verificar
D:\University\T2\Programacion\vision-api-object-detection\.venv\Scripts\python.exe scripts\evaluation\plan_pruebas_nf_r2.py --env-file D:\University\T2\Programacion\vision-api-object-detection\.env
```

El primer comando solo verifica; el segundo mide.

---

## 10. Descargas necesarias — [APROBACIÓN]

| Archivo | Origen | Tamaño aproximado |
|---|---|---|
| `tensorflow` (con dependencias) y `tensorflow-hub` | PyPI | 300–500 MB |
| `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` | download.pytorch.org | ≈ 167 MB |
| `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` | download.pytorch.org | ≈ 177 MB |
| SSD-MobileNet V2 (COCO 2017) | tfhub.dev (redirige a Kaggle Models) | ≈ 25–30 MB |

---

## 11. Decisiones

| ID | Decisión | Estado |
|---|---|---|
| F1 | F1 de presencia sustituye a la precisión de presencia en el total; precisión y recall se registran y entran en la sensibilidad | Autorizada (v5) |
| INV | Inventario versionado en un commit antes de la campaña; integridad comprobada por Git y SHA-256; sin afirmar ceguera | Autorizada (v5) |
| T | Empates: relativa 5 % para tiempo y CPU (variantes 0 y 10 %); exacta para F1 (variantes 1/\|P\| y 2/\|P\|), retención (variante unidad común) e imágenes con FP (variante una imagen) | Aprobada provisionalmente el 27-oct-2026 |
| S1 | Elegibilidad = p95 de las 108 llamadas < 5000 ms; el de las 36 medianas en la sensibilidad | Propuesta en v5; [APROBACIÓN] |
| REG | Registro de fechas, errores, configuración efectiva, dependencias, huellas y commit; ninguna medición con cambios sin commit | Autorizada (v5) |
| R | R2: una sola ejecución de 30 solicitudes, criterio por solicitud, registro del servidor, verificación previa y contabilidad completa | Autorizada (v5) |
| H | Prueba de humo separada, documentada y no ejecutada | Autorizada (v5); ejecución pendiente |
| I | Integración fuera del total mientras sea idéntica; información espacial dentro | Autorizada (v5) |
| S3 | YOLO26s a 640 px complementario, sin puntuar; principal a 1280 | Autorizada (v5) |
| P | Criterio de precisión: P1 o P3 + P2′ | [APROBACIÓN]; recomendación P3 + P2′ |
| S4 | Memoria de la Tabla 5 = RSS con el modelo listo | [APROBACIÓN] |
| S5 | Huellas esperadas registradas tras la descarga | [APROBACIÓN] |
| E1 | Entorno de referencia para PyTorch y R2; uno aparte para SSD | [APROBACIÓN] |
| D | Descargas y cambio del plan de energía | [APROBACIÓN] |
| C | Commit en la rama (sin push) antes de la prueba de humo definitiva y de la campaña | [APROBACIÓN] |
