# CHECKPOINT 3B — Protocolo de evaluación (borrador para revisión)

- **Estado:** BORRADOR. **Nada de este documento está aprobado.**
- **Tipos de contenido.** Cada punto está marcado de una de tres formas:
  - **HECHO:** verificado en el código o en los datos;
  - **PROPUESTA:** requiere aprobación;
  - **POR DEFINIR:** hay que tomar una decisión. Se indica qué decisión falta.
- **Fecha:** 2026-09-26.
- **Base:**
  - backend en la rama `fase3/infraestructura`;
  - Dataset 1 congelado (generador, commit `ae85f90`, importado en `stimuli/dataset1`).
- **Ejecuciones.** No se ha ejecutado YOLO, ni el LLM, ni el TTS sobre el Dataset 1. **No existe ningún resultado.**

---

## 0. Principio: capas separadas

El ground truth del generador ya define 6 capas (HECHO, `conventions.layers` de cada GT). Cada métrica de este protocolo se asigna a **una sola** capa. Una métrica nunca mezcla capas.

| Capa | Contenido | Fuente | Qué se evalúa |
|---|---|---|---|
| 1–2 | Condición geométrica de diseño: clase, posición horizontal, condición de profundidad, relaciones y oclusión O | GT del generador (solo servidor) | Nada: es la referencia |
| 3 | Imagen renderizada | `stimuli/dataset1/*.png` (hash verificado) | Nada: es el estímulo |
| 4 | Salida del detector: clases, cajas y confianzas | YOLO | Detección (§2) |
| 5 | Interpretación espacial de la API: columna y profundidad en 4 categorías | `analyze_spatial` | Posición, profundidad y relaciones (§3) |
| 6 | Narrativa (texto) y audio | LLM + TTS | Fidelidad narrativa (§4) |
| 7 | Evaluación subjetiva y de tarea | Participantes | Estudio con usuarios (§6) |

- **Arrastre de errores.** Un error en la capa 4 se arrastra a las capas 5 y 6. Por eso las métricas de las capas 5 y 6 se reportan dos veces:
  - **condicionadas**, solo sobre los objetos que la capa anterior acertó;
  - **de extremo a extremo**, sobre todos los objetos de diseño.
- **Estado.** PROPUESTA.

---

## 1. Unidad de análisis

- **HECHO.** Composición del Dataset 1:
  - 18 escenas: A1–A9, B1–B3, C1–C3 y D1–D3;
  - **32 objetos de diseño** en total: A = 9 × 1, B = 3 × 2, C = 3 × 2, D = 3 + 4 + 4;
  - clases: silla, mesa, sofá, planta, persona, botella y portátil.
- **PROPUESTA.** Tres unidades de análisis:
  - **objeto de diseño** (n = 32), para la detección, la posición y la profundidad;
  - **par con relación declarada**, para las relaciones:
    - `in_front_of`: C1 y C3;
    - `beside`: C2 y D3;
    - `on_top_of`: D2 × 2;
    - en total, n = 6 pares con relación explícita;
  - **escena** (n = 18), para la completitud narrativa.
- **Advertencia (HECHO).** Los tamaños de muestra son pequeños. Hay que reportar conteos (k/n) y no solo porcentajes. Los intervalos de confianza son **POR DEFINIR**: falta decidir si se reportan intervalos de Wilson o solo conteos.

---

## 2. Detección (capa 4)

### 2.1 Configuración fija de YOLO

HECHO: valores actuales, según `docs/REPRODUCIBILIDAD.md`.

- **Modelo:** `yolo26s.pt`, sha256 `646f8bc3…4a1b`.
- **Parámetros:**
  - `imgsz` 1280;
  - IoU de NMS 0.45;
  - confianza interna 0.15;
  - umbral de salida por defecto 0.35;
  - `augment` false.
- **Preprocesado:** las imágenes del Dataset 1 (800×450) **no se redimensionan ni se re-codifican**, porque `resize_image` solo actúa por encima de 800 px. YOLO recibe el PNG original.

**POR DEFINIR:**

- **Umbral de confianza para la evaluación formal.**
  - Opciones:
    - 0.35, el valor por defecto de la app;
    - reportar además una curva de umbrales;
    - otro valor fijado antes de ver resultados.
  - Decisión que falta: fijarlo **antes** de ejecutar.
- **Dispositivo.**
  - Hoy `run_yolo` no fija `device`: el entorno local usa la GPU (GTX 1050) y Docker usa la CPU. Los resultados pueden diferir en decimales y, cerca del umbral, en detecciones.
  - PROPUESTA: `device=cpu` para la evaluación formal, por reproducibilidad en cualquier máquina.
  - Decisión que falta: aprobarlo.
- **Número de corridas.**
  - Con los pesos y el dispositivo fijos, la inferencia debería ser determinista.
  - PROPUESTA:
    - 3 corridas completas;
    - verificar que las salidas son idénticas, comparando el hash de la salida JSON;
    - si difieren, reportar la variación.
  - Decisión que falta: el número de corridas y qué hacer si difieren.

### 2.2 Correspondencia detección ↔ objeto de diseño

- **Hecho crítico (HECHO).** El GT **no contiene una caja 2D que sirva de objetivo al detector**. `projected_bbox_2d_px` es la proyección geométrica de *todos* los vértices, sin tratar la oclusión, y el propio GT declara: *"NOT the box a detector such as YOLO should output and must not be used as such"*.
- **Oclusión.** En C1 y C3 hay oclusión parcial (O = 0.236 y 0.203). En esos casos la caja proyectada es más grande que la parte visible.
- **POR DEFINIR (decisión de ground truth). ¿Contra qué caja se calcula el IoU?** Opciones:
  1. **Caja proyectada** (existente). Se puede usar solo si se reinterpreta explícitamente como caja **amodal**. Contradice la advertencia del GT, así que requiere una decisión documentada.
  2. **Caja visible (modal).** Se derivaría de la máscara de visibilidad del z-buffer del generador. Es **nuevo GT**: exige una versión nueva del GT, sin tocar las imágenes, y aprobación.
  3. **Sin IoU.** La correspondencia se hace por clase + columna del centro de la caja: la detección de la clase correcta cuyo centro cae en el tercio de diseño. En A y B basta, porque hay una sola instancia por clase. En C y D puede ser ambiguo.
- **PROPUESTA provisional**, pendiente de la decisión anterior:
  - correspondencia por clase + IoU ≥ **0.5**, emparejamiento voraz por confianza descendente;
  - una detección solo puede emparejarse con un objeto;
  - en C1 y C3 se reporta aparte el objeto ocluido.
- **Umbral de IoU.** El 0.5 es PROPUESTA (convención PASCAL VOC). También podría reportarse 0.5:0.95 al estilo COCO. Decisión que falta: el umbral.

### 2.3 Definiciones

PROPUESTA, una vez resuelta la correspondencia:

- **VP (verdadero positivo):** objeto de diseño con una detección emparejada de la **misma clase COCO**.
- **FN (falso negativo):** objeto de diseño sin detección emparejada.
- **FP (falso positivo):** detección que no se empareja con ningún objeto de diseño.
- **Tipos de FP.** Se reportan por separado:
  - clase ausente de la escena (alucinación);
  - duplicado de un objeto ya emparejado;
  - clase confundida en la posición de un objeto de diseño.
  - Decisión que falta (**POR DEFINIR**): la sala contiene paredes, suelo, ventanas y puerta, que no son objetos de diseño. Una detección de una clase COCO sobre esos elementos (por ejemplo, "tv" sobre una ventana) ¿cuenta como FP?
    - PROPUESTA: sí, cuenta como FP;
    - se revisa la categoría de cada caso sin cambiar el conteo.
- **Métricas:**
  - recall por objeto (VP / 32);
  - precisión (VP / (VP + FP));
  - FP por escena;
  - todo desglosado por bloque, clase, posición horizontal y condición de profundidad.
- **Clases de sinónimo (POR DEFINIR).** Una "chair" de diseño detectada como "couch" ¿es un VP? PROPUESTA: no; se registra como confusión.

---

## 3. Interpretación espacial (capa 5)

### 3.1 Posición horizontal

- **HECHO.** El criterio es consistente entre el sistema y el GT:
  - el GT sitúa el punto de referencia en u = 0.25 / 0.5 / 0.75, con fronteras de tercio en 1/3 y 2/3 y un margen mínimo de 0.05;
  - la API clasifica por el centro X de la caja de YOLO con los mismos tercios.
- **PROPUESTA:**
  - acierto = columna de la API igual a la `horizontal` de diseño;
  - solo sobre los VP (condicionada), y de extremo a extremo contando los FN como fallo;
  - métrica: exactitud y matriz de confusión 3×3.

### 3.2 Profundidad

- **HECHO.** Las escalas no coinciden:
  - el diseño tiene **3** condiciones geométricas: near 3.02 m, medium 4.64 m y far 7.14 m, en bandas de igual razón logarítmica;
  - la API produce **4** categorías (`muy_cerca`, `cerca`, `medio`, `lejos`) a partir del área de la caja y de las posiciones y2/yc;
  - el GT dice expresamente que **no se aplica ninguna correspondencia 4→3**.
- **POR DEFINIR (decisión de métrica).** Opciones:
  1. **Ordinal.** Evaluar solo si el orden es monótono: near < medium < far ⇒ la categoría de la API no aumenta con la distancia. No exige correspondencia de categorías.
  2. **Correspondencia fijada a priori**, antes de ver resultados. Por ejemplo: `muy_cerca`/`cerca` → near, `medio` → medium, `lejos` → far. Exige justificarla.
  3. Reportar solo la **tabla de contingencia 3×4** sin métrica de acierto.
- **PROPUESTA:** 1 + 3. Son las únicas opciones que no inventan una correspondencia.
- **Alcance.** El bloque A (3×3) es el único que varía la profundidad. En B, C y D todos los objetos están en *medium*, así que no aportan a esta métrica salvo como control.

### 3.3 Relaciones

- **HECHO.** Las relaciones de diseño son `in_front_of` (C1, C3), `beside` (C2, D3) y `on_top_of` (D2, dos objetos).
- **Qué calcula la API (HECHO).** En la capa 5, la API calcula una sola relación entre objetos: "objeto sobre superficie" (`_merge_surfaces`, por ejemplo un portátil sobre una mesa). **No calcula** "delante de" ni "al lado de": para esos objetos solo da columna y profundidad.
  - Además, `_deduplicate` fusiona detecciones duplicadas antes de la narrativa. La capa 5 puede tener menos objetos que la capa 4.
- **POR DEFINIR:**
  - **`on_top_of` en la capa 5.** ¿Se evalúa ahí? Solo afecta a D2. PROPUESTA: sí.
  - **`in_front_of` y `beside`.** ¿Se evalúan solo en la narrativa (capa 6)? PROPUESTA: sí.
  - En la capa 5 se reporta además el orden izquierda/derecha implícito en las columnas de B y D.

---

## 4. Narrativa (capa 6)

**POR DEFINIR** (decisión de métrica y de procedimiento):

- **Cómo se codifica la narrativa.** Opciones:
  - codificación manual con una rúbrica;
  - dos codificadores y su acuerdo (kappa);
  - reglas automáticas por palabras clave.
- **Qué se codifica** (PROPUESTA de rúbrica por objeto de diseño):
  - ¿se menciona el objeto?
  - ¿la posición mencionada coincide con el diseño?
  - ¿la distancia mencionada es coherente?
  - ¿se menciona algún objeto inexistente?
  - por escena: la completitud y las relaciones (C y D).
- **Referencias separadas.** La narrativa se compara con dos referencias distintas, reportadas por separado:
  - **(a)** con la **salida de YOLO** de esa misma corrida, para medir la fidelidad del LLM;
  - **(b)** con el **diseño**, para medir el extremo a extremo.
- **Temperatura del LLM (HECHO):** 0.1. La salida no es determinista.
  - PROPUESTA: la narrativa que se evalúa y se presenta a los participantes es la del **paquete de estímulo congelado** (F4), no una nueva generación.

---

## 5. Paquete de estímulo congelado (definición de campos; se genera en F4 tras aprobación)

Para cada estímulo, un directorio en `DATA_ROOT/stimuli_frozen/<stimulus_id>/`:

- **Archivos:**
  - `package.json`;
  - `narrative.txt`;
  - `audio.mp3`;
  - `detections.json`.
- **Campos de `package.json`:**
  - `stimulus_id`, `dataset`, `image_sha256`;
  - `detector`: `{weights, weights_sha256, ultralytics, torch, device, imgsz, iou_nms, conf_internal, conf_output, augment}`;
  - `detections_sha256`, `spatial_analysis` (salida de la capa 5);
  - `narrative`: `{text_sha256, llm_provider, llm_model, temperature, prompt_version/sha256}`;
  - `tts`: `{provider, model, voice, style_instructions_sha256, config}`, `audio_sha256`, `audio_duration_s`;
  - `created_at_utc`, `app_commit`, `experimental_config_sha256`.
- **POR DEFINIR:**
  - si se genera **una** narrativa por estímulo, o varias para elegir;
  - si se eligen, el criterio de elección. Elegir a mano introduce sesgo. PROPUESTA: una sola generación, sin selección.
  - presupuesto del TTS: alrededor de 15.000 COP (regla de trabajo).

---

## 6. Estudio con usuarios (capa 7)

**POR DEFINIR:**

- **Estímulos de cada tarea.** Qué estímulo congelado usa cada tarea OBJ-01…OBJ-07. En `catalog.yaml` está como `estimulos: POR_DEFINIR`.
  - Por ejemplo, OBJ-03 ("ruta libre") exige una escena con obstáculo al centro y a un lado. Hay que decidir qué escena del Dataset 1 cumple eso, o si ninguna lo cumple.
- **OBJ-04 ("secuencia").** Requiere dos imágenes de la *misma escena con objetos reubicados*. **El Dataset 1 no tiene pares así.** Habría que decidir entre crear un estímulo nuevo (lo que modificaría el dataset congelado, y no se debe hacer) o redefinir la tarea.
- **OBJ-05.** Requiere un "audio de referencia neutro". Aún no está definido.
- **Métricas.** Las `metricas_texto` del catálogo (por ejemplo "% identificación correcta") necesitan una definición operativa: qué respuesta cuenta como correcta y quién la codifica.
- **Separación de pistas.** Los resultados de la pista piloto **no** son evidencia de accesibilidad. Esto ya consta en el catálogo.

---

## 7. Dataset 2 (robustez / variabilidad visual)

- **HECHO.** Estado actual:
  - sin GT 3D;
  - **ninguna imagen tiene hoy procedencia y licencia verificadas** (`evaluation/metadata/inventario_dataset2_verificacion.csv`);
  - PHV-01 y JC3D-03, que eran candidatas, quedaron excluidas.
- **Qué se puede evaluar**, cuando haya imágenes aprobadas: presencia de clases, con un GT manual por inspección definido **antes** de ejecutar, y coherencia narrativa. No se evalúan la profundidad métrica ni la posición contra una verdad geométrica.
- **Preprocesado (HECHO).** Las imágenes de más de 800 px se redimensionan y se re-codifican a JPEG q90 antes de YOLO. Esto hay que declararlo.
- **POR DEFINIR:** la selección final, una vez verificadas las licencias; ver el checkpoint.

---

## 8. Agregación y reporte

PROPUESTA:

- **Tablas:**
  - por bloque (A, B, C, D) y total;
  - por clase;
  - por posición horizontal;
  - por condición de profundidad (solo el bloque A).
- **Conteos:** siempre k/n junto al porcentaje.
- **Resultados por imagen:** se publican también (anexo) en crudo, junto con el hash de la configuración.
- **Tablas 15, 16 y 17 de la tesis:** se rehacen con este protocolo (ver `IMPACTO_TESIS_CAMBIOS_ACTUALES.md`). Los resultados exploratorios anteriores no se mezclan con los formales.

## 9. Resumen de decisiones pendientes

| # | Decisión | Tipo | Propuesta |
|---|---|---|---|
| 1 | Caja de referencia para el IoU (proyectada/amodal, visible/modal o sin IoU) | ground truth | proyectada como amodal **o** GT modal nuevo versionado |
| 2 | Umbral de IoU | métrica | 0.5 |
| 3 | Umbral de confianza de salida | configuración experimental | 0.35, fijado antes de ejecutar |
| 4 | Dispositivo (CPU o GPU) | configuración experimental | CPU |
| 5 | Número de corridas y criterio de igualdad | configuración experimental | 3 corridas, comparación por hash |
| 6 | ¿Cuentan como FP las detecciones sobre elementos de la sala? | métrica | sí, con categoría aparte |
| 7 | Profundidad 3 condiciones frente a 4 categorías | métrica | ordinal + contingencia, sin correspondencia |
| 8 | `on_top_of` en la capa 5; `in_front_of`/`beside` solo en la narrativa | métrica | sí |
| 9 | Rúbrica y procedimiento de codificación de la narrativa | métrica | rúbrica por objeto; 2 codificadores |
| 10 | Una o varias narrativas por estímulo | estímulos | una, sin selección |
| 11 | Estímulo de cada tarea OBJ; OBJ-04 sin pares en el Dataset 1 | estímulos / metodología | por decidir |
| 12 | Intervalos de confianza | métrica | Wilson + conteos |
| 13 | Dataset 2: selección tras verificar licencias | estímulos | ver checkpoint |
