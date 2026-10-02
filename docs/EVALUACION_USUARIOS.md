# Instrumento de evaluación con usuarios (Objetivo Específico 3)

- **Fecha:** 2026-09-28.
- **Fuente metodológica:** "27 - Preparación evaluación final Objetivo 3" (diseño exploratorio, una sesión por participante, sin navegación 3D; el participante escucha el audio de la API y responde verbalmente).
- **Estado:** el instrumento está implementado y probado solo con datos técnicos (PTEST01, PTEST02). **La evaluación con usuarios objetivo NO se ha ejecutado.**
- **Código:** backend `app/routes/study.py`, `app/catalog/loader.py`; cliente `components/study/*`, `lib/study-protocol.ts`, `hooks/use-study.ts`.
- **Revisión del 2026-10-01:** ver §0. Prevalece sobre lo que digan las secciones siguientes.

## 0. Revisión del diseño de pruebas (2026-10-01, doc. 34 de la carpeta `claude`)

Se hizo antes de cualquier sesión con participantes reales. La detección, el pipeline, la narrativa, el TTS y el contrato de `POST /api/detect` **no cambian**; solo cambia el instrumento de evaluación.

| # | Problema | Cambio |
|---|---|---|
| 1 | Cada participante podía escuchar una narrativa distinta (el LLM se llamaba en vivo; "Volver a generar") | **Audio congelado**: `scripts/study/freeze_study_audio.py` genera una vez narrativa y audio de cada escena asignada y de la práctica, con sha256, en `stimuli/dataset1/estudio/` (`congelados.yaml`, `intentos.jsonl`). Una respuesta formal solo se acepta si su sha256 y su narrativa coinciden con las congeladas (409 en otro caso). Nueva ruta `GET /api/study/stimuli/{id}/audio`. La generación en vivo queda solo para ensayos. |
| 2 | El % de objetos identificados se calculaba también en OBJ-03/04, donde no se pregunta por objetos | Campo `codificacion` en `catalog.yaml`: OBJ-01 `[objetos, relaciones]`, OBJ-02 `[ubicacion, distancia]`, OBJ-04 `[cambio]`, el resto `[]`. El servidor solo cuenta lo que la prueba codifica y rechaza codificación en pruebas que no la usan. Resumen **por prueba** (`resumen.por_prueba`). |
| 3 | La relación de OBJ-01 no se preguntaba | Guion de OBJ-01 con la segunda pregunta "¿Cómo están ubicados esos objetos entre sí?". |
| 4 | El mismo clic iniciaba la grabación y marcaba el tiempo de respuesta | Botón propio "Respuesta iniciada"; la grabación ya no marca el tiempo. |
| 5 | El participante objetivo no tenía práctica | `asignaciones.yaml → practica: [DS1-A2]`. En sesiones objetivo, el modo ensayo solo se admite con esa escena. |
| 6 | El piloto no dejaba datos consolidados | El piloto registra como **formales** las actividades OBJ; van al grupo `piloto` del consolidado, nunca al de objetivo. |
| 7 | OBJ-03: en DS1-C2 la narrativa congelada indica **"frente"** (el sofá está a ~5 pasos y no bloquea el espacio inmediato) y el diseño de la escena espera **"izquierda"** | La decisión registra **tres capas**: `coincide_con_narrativa` (comprensión; métrica principal), `correcto` (frente al diseño de la escena) y `narrativa_coincide_con_diseno` (resultado técnico del sistema). La dirección narrada la extrae el servidor (`decisions.narrated_direction`). Se elimina el juicio manual del investigador como métrica (queda como `juicio_investigador`). |
| 8 | OBJ-04 juntaba dos preguntas y el cambio no tenía campo | Dos preguntas separadas y campo `percepcion_cambio` (`menciona_cambio_real`, `no_menciona_cambio`, `menciona_cambio_inexistente`, `no_responde`), obligatorio en formales. |
| 9 | Escalas repetidas en el cierre; anclas provisionales | Anclas únicas **1 nada · 2 poco · 3 moderadamente · 4 bastante · 5 muy**, leídas completas. El cuestionario de cierre solo pregunta claridad de las descripciones y esfuerzo (lo demás ya se pregunta en OBJ-05/06/07). En OBJ-05 `claridad` pasa a `inteligibilidad` (ITU-T P.85). |
| 10 | Aclaraciones sin registrar | `aclaraciones[]` (pregunta / escala / otra) y métrica `aclaraciones`. |
| 11 | OBJ-02 no evaluaba la distancia narrada | Segunda pregunta de distancia; `distancia_reportada` y `distancia_correcta` frente a los pasos que dijo la narrativa. |
| 12 | PIL-01 decía "antes de comenzar" y se hacía al final; PIL-03 duplicaba el registro técnico; faltaba validar las escalas | PIL-01 en retrospectiva; PIL-02 compara con la duración real (`resumen.duracion_sesion_min`); **PIL-03 ahora es "Comprensión de las escalas y las preguntas"** (el registro técnico va en el cierre). |
| 13 | Ficha sin rango de edad ni audición | `ficha.rango_edad` (rangos, nunca la edad exacta) y `ficha.audicion_autodeclarada`; obligatorios en el cliente, opcionales en el servidor por compatibilidad. |

**Carpetas separadas (2026-10-02).** Las sesiones se guardan por grupo y, dentro de cada sesión, las respuestas por modo. Las sesiones anteriores, que están en la raíz, se siguen leyendo:

```
DATA_ROOT/study/sessions/
  objetivo/<session_id>/          participantes reales (evidencia del OE3)
  piloto/<session_id>/            validación del instrumento
  pruebas_tecnicas/<session_id>/  códigos PTEST (nunca evidencia)
    sesion.json · responses.jsonl · grabacion_consentimiento.<ext>
    respuestas/formales/R00N/     audio_narrativa_api.mp3, respuesta_participante.<ext>
    respuestas/ensayos/R00N/      práctica y ensayos
```

**Usabilidad para el investigador (2026-10-02).**
- El audio congelado se carga solo al abrir la prueba.
- La codificación se precarga desde la narrativa.
- El reproductor está arriba, antes de la narrativa, y la imagen va plegada.
- Al elegir una prueba, la pantalla se desplaza hasta ella.

**Hash del catálogo.** `catalog.yaml` cambió solo en `pruebas_usuario` y en el estado de las métricas de usuario (`DEFINIDA_EVALUACION_USUARIOS`). Su nuevo sha256 está en `experimental_config.yaml` (anterior: `ff0404979a04…`). Las pruebas técnicas, los datasets y las métricas técnicas no cambiaron.

**Conjunto congelado (2026-10-01).** Seis audios (DS1-A2 práctica, C1, A8, C2, B2, A5), voz `azure:es-CO-SalomeNeural`, LLM `qwen/qwen3.8-27b`, umbral 0,35; todos aceptados en el primer intento. **Pendiente antes de la primera sesión real:** que una segunda persona escuche cada audio y confirme que dice literalmente la narrativa (`verificacion_fidelidad` en `congelados.yaml`).

## 1. Qué NO cambia

YOLO26s, pesos, umbrales, Dataset 1, `experimental_config.yaml`, lógica espacial, narrativa y TTS no cambian. El contrato de `POST /api/detect` es idéntico (`tests/regression/contracts.json`: solo se añaden rutas `/api/study/*`).

En el estudio, el cliente **no envía** `confidence_threshold`, así que el servidor aplica su valor por defecto congelado (0,35). Cada respuesta guarda el umbral informado por `/api/detect`.

## 2. Cadena de trazabilidad

```
PARTICIPANTE (código P0X, ficha)            sesion.json
  └ SESIÓN (consentimiento, grabación,      sesion.json
            contexto técnico, cierre)
     └ PRUEBA (catálogo del backend)        responses.jsonl → prueba{id, tipo, pista, estado_estimulos}
        └ ESTÍMULO (stimulus_id + sha256     responses.jsonl → estimulo  (sha256 lo pone el servidor desde el manifest)
                    del manifest)
           └ EJECUCIÓN (/api/detect)         ejecucion{request_id, narrativa_final, escenario, degradaciones, umbral}
              └ AUDIO escuchado              respuestas/R00N/audio_narrativa_api.mp3 (+ sha256 en el registro)
                 └ REPRODUCCIONES            reproducciones[{instante, inicial|repeticion}]
                    └ RESPUESTA              respuesta_transcrita, comprension{objetos, inventados, relaciones}, decision{seleccionada, esperada, correcto}
                       └ MÉTRICAS            metricas (calculadas por el servidor)
                          └ ERRORES          errores (registrados) + errores_derivados
                             └ ESCALAS / OBSERVACIONES / COMENTARIOS
```

## 3. Contrato (v3) — todas las rutas exigen `X-API-Key` del investigador

| Ruta | Cambio | Notas |
|---|---|---|
| `POST /api/study/sessions` | **Rompe el contrato v2** (2026-09-28, v3) | Recibe `codigo`, `tipo_participante`, `ficha`, `contexto`, `consentimiento` (`version` + 4 afirmaciones) y `grabacion_consentimiento` (`content_type`, `data_base64`, `duracion_s`). Ya **no** acepta `grabacion`, `investigador`, `notas`, `modalidad` ni `formato_referencia` (400). No acepta `nombre`, `edad` ni `genero`. Respuestas: 409 si el código está repetido o si un código real se guardaría dentro del repositorio; 400 si falta una afirmación, la grabación o la versión no corresponde al tipo de participante; 415 si el formato de audio no se admite. |
| `GET /api/study/sessions/{id}/consentimiento/audio` | Nueva (v3) | Grabación de la lectura del consentimiento. 404 si no se almacenó (PTEST sin `DATA_ROOT`). |
| `GET /api/study/sessions` | Cambia | Devuelve `sesiones[]` (código, tipo, estado, `num_formales`) y `sesiones_heredadas_omitidas`. |
| `GET /api/study/sessions/{id}` | Cambia | Devuelve `{sesion, respuestas, resumen}` (antes `{participant, respuestas}`). |
| `POST /api/study/sessions/{id}/responses` | **Rompe el contrato anterior** | Ver §5. |
| `POST /api/study/sessions/{id}/responses/{rid}/grabacion` | Nueva | Multipart. Responde 403 sin autorización y 409 si el almacenamiento está dentro del repositorio o la grabación ya existe. |
| `GET /api/study/sessions/{id}/responses/{rid}/audio/{narrativa\|participante}` | Nueva | Recupera el audio que se escuchó o la grabación del participante. |
| `POST /api/study/sessions/{id}/cierre` | Nueva | Recibe motivo, cuestionario posterior, entrevista, incidencias técnicas y observaciones. Después la sesión no admite respuestas (409). |
| `DELETE /api/study/sessions/{id}` | Igual | Ahora borra también las subcarpetas. En el cliente pide confirmación. |
| `GET /api/study/consolidado` | Nueva | Matriz consolidada: solo respuestas **formales**, sin PTEST y con el piloto separado de objetivo. |
| `GET /api/catalog` | Amplía | Cada prueba de usuario incluye `estado_estimulos` (`por_definir` / `no_requiere` / `definido`), `requiere_estimulo`, `ejecutable_formal` y `decision` (pregunta y alternativas de las tareas de decisión, **nunca la alternativa esperada**). **`catalog.yaml` no se modificó** (sigue congelado). |

- **Límite de peticiones:** las rutas del investigador tienen su propio límite, `RESEARCHER_RATE_LIMIT_REQUESTS` (por defecto 120 por ventana). Antes compartían el límite de 10/60 s, y una sesión podía recibir 429 a mitad de camino.
- **Compatibilidad:** el cliente anterior (`c69ca54`) **no funciona** con este backend. Hay que desplegar backend y cliente juntos.

## 4. Ficha mínima y justificación

| Campo | Valores | Por qué se registra |
|---|---|---|
| `codigo` | `P01…P999`; `PTEST01…` para pruebas técnicas | Identificador anonimizado. El nombre no entra al sistema: la correspondencia código ↔ persona queda en el formato de consentimiento en papel. |
| `tipo_participante` | objetivo / piloto | Separa la evidencia (objetivo) de la validación del instrumento (piloto, doc. 27 §M). |
| `condicion_visual.tipo_ceguera` | congénita / adquirida (piloto: no_aplica) | La experiencia visual condiciona la comprensión de objetos y relaciones espaciales. |
| `etapa_adquisicion` | infancia / adolescencia / adultez / no_informa (solo adquirida) | Mismo motivo. Es aproximada y **no clínica**. |
| `experiencia_visual_previa` | sí / no / no_informa | Mismo motivo. |
| `tecnologias.utiliza` | lector_pantalla, smartphone, computador, tableta, linea_braille, otra, ninguna | Familiaridad con interfaces auditivas. Esta lista es también el **dispositivo habitual**. |
| `lectores_pantalla` | NVDA, JAWS, VoiceOver, TalkBack, otro, no_utiliza | Familiaridad con voz sintética. |
| `frecuencia_uso` | diaria / varias por semana / ocasional / no utiliza actualmente | Mismo motivo. |
| `experiencia_descripcion_audio` | sí / no / no_informa | La experiencia previa con descripción de escenas por audio puede afectar la comprensión. |
| `contexto.dispositivo`, `reproduccion_audio` | computador/teléfono/tableta/otro; audífonos/parlantes/otro | Dispositivo **de la sesión** (≠ habitual) y condiciones de escucha. |
| `contexto.entorno_tecnico` | navegador, SO y tipo de dispositivo (user agent) | Lo registra el navegador automáticamente; no se pregunta. |

**No se registran:** nombre, edad, género ni historia clínica. La edad y el género existían en v1; se retiraron porque no aparecen en la ficha del doc. 27. Si los directores quieren describir la muestra por edad, debe decidirse con ellos (por ejemplo, un rango).

## 5. Respuesta (una por estímulo o tarea)

- **Campos:** `prueba_id`, `modo` (formal/ensayo), `estimulo`, `ejecucion`, `audio_narrativa_base64`, `reproducciones[]`, `tiempo_respuesta_ms`, `respuesta_transcrita`, `comprension{objetos[{objeto, identificado, ubicacion_reportada, ubicacion_correcta}], objetos_inventados[], relaciones[{relacion, respuesta, comprendida}]}`, `decision{seleccionada, coincide_con_narrativa}`, `escalas`, `criterios`, `errores[]`, `aspectos_confusos`, `comentarios` y `observaciones`.
- **Referencia de la codificación:** los objetos y relaciones **mencionados en la narrativa** que escuchó el participante (se evalúa la comprensión de la narrativa, doc. 27 §F). No se usa ground truth.
- **Métricas que calcula el servidor (doc. 27 §I):**
  - % de objetos identificados, objetos omitidos e inventados;
  - ubicaciones correctas;
  - % de relaciones comprendidas;
  - repeticiones (= reproducciones de tipo `repeticion`);
  - tiempo de respuesta (métrica débil, desde el fin de la última reproducción hasta la marca del investigador).
  - **No se calcula una tasa de éxito global.**
- **Escalas 1–5 (doc. 27 §J):** claridad, utilidad, suficiencia, naturalidad de la voz, carga percibida (débil) y redundancia. Cada una puede quedar como "No preguntado".
  - Se usan en cada estímulo y en el cuestionario posterior.
  - Las anclas (1 = nada… 5 = muy…) son una propuesta **pendiente de validar**.
- **Reglas que aplica el servidor:**
  - una prueba `POR_DEFINIR` nunca es formal (409);
  - formal = pista de la sesión + estímulo asignado en el catálogo + audio disponible + exactamente una reproducción inicial;
  - `ensayo` solo en sesiones piloto o PTEST;
  - el sha256 del audio recibido debe coincidir con el declarado.

## 5 bis. Tareas de decisión (tipo C)

- **Qué son:** el participante escucha la narrativa y elige entre alternativas. Es una decisión **hipotética** basada en información espacial auditiva: nadie camina, se desplaza ni controla un personaje (doc. 27, §H).
- **Definición:** `app/catalog/decisiones.yaml`, fuera de `catalog.yaml` porque este está congelado.
  - La pregunta y las alternativas se toman **literalmente** del guion del catálogo: OBJ-03 dice "izquierda/derecha/frente"; OBJ-04 usa las mismas.
  - `esperada_por_estimulo` está **POR_DEFINIR**, igual que los estímulos de OBJ-03/OBJ-04.
- **Quién decide la esperada:** solo el servidor, a partir de la definición.
  - El cliente envía `seleccionada` (una alternativa o `no_responde`) y, aparte, la codificación del investigador `coincide_con_narrativa`.
  - `GET /api/catalog` nunca publica la esperada.
- **Qué registra el servidor:** `decision{pregunta, alternativas, seleccionada, esperada, correcto, fuente_esperada, fixture_tecnico, definicion_sha256}`, la métrica `decision_correcta` y el error derivado `decision_incorrecta`.
  - `correcto` es `null` si no hay esperada definida o si el participante no respondió.
- **Reglas:**
  - Una decisión **formal** exige la esperada definida para ese estímulo (409 si está POR_DEFINIR).
  - Una tarea de decisión formal exige registrar la alternativa (400).
  - `decision` en una prueba que no es de decisión da 400.
- **Resumen:** conteos por resultado (`decisiones_registradas`, `decisiones_correctas`, `decisiones_incorrectas`, `decisiones_sin_respuesta`), solo sobre respuestas formales. **No hay tasa de éxito global.**
- **Fixture técnico `FIX-DEC-01`** (OBJ-03 × DS1-B3, esperada `izquierda`): sirve **solo** para verificar el mecanismo.
  - Se aplica únicamente en sesiones PTEST y en modo ensayo; nunca a participantes P0X.
  - Fundamento: el diseño geométrico de DS1-B3 en el manifest pone una persona al centro y una botella a la derecha, sin objetos a la izquierda.
  - **No es** una respuesta metodológica de OBJ-03.

## 5 ter. Consentimiento informado (v3; texto v0.4 desde el 2026-09-29)

- **Dos documentos, uno por tipo de participante:** `Consentimiento_informado_VisionNav_Personas_Piloto_v0.3` y `…_Personas_Objetivo_v0.3`.
  - El cliente descarga los archivos reales (PDF y Word) desde `visionnav-client/public/consentimientos/`.
  - El texto que se lee en voz alta está en `visionnav-client/lib/consent.ts`, con las erratas gramaticales corregidas.
  - El asistente muestra el documento que corresponde al tipo de participante elegido.
- **Cuatro afirmaciones obligatorias (§12):** `acepta_participar`, `puede_detenerse`, `autoriza_grabacion` y `autoriza_uso_academico`.
  - La 3 es la autorización de grabar las respuestas: según el §5 del documento, la grabación es condición para participar.
  - `sesion.grabacion.autoriza_grabacion_audio` se deriva de ella, así que la subida de grabaciones de respuestas no cambia.
- **Grabación de la lectura del consentimiento:** es obligatoria y se envía junto con la creación de la sesión, que queda atómica: sin grabación no se guarda nada.
  - Con `DATA_ROOT` se guarda como `grabacion_consentimiento.<ext>`.
  - Sin `DATA_ROOT` (solo PTEST) se guarda únicamente la huella: sha256, tamaño y duración.
- **Trazabilidad:** `consentimiento.version` (validada contra `CONSENT_VERSIONS` en `app/routes/study.py`), `documento`, `modalidad = "verbal"` y `grabacion`.
- **Compatibilidad:** las sesiones v2 existentes (por ejemplo, PTEST10 en Azure) se siguen leyendo; sus campos v2 se conservan.
- **v0.4 (2026-09-29):** el §9 dice que la información se guarda en el servidor del proyecto (Microsoft Azure), con acceso restringido por clave, y que las grabaciones se conservan un año.
  - `CONSENT_VERSIONS` solo acepta v0.4 para sesiones nuevas.
  - Las sesiones creadas con v0.3 conservan su versión.
  - Los .docx v0.3 del investigador siguen disponibles como "original".
- **Acta (§13):** el asistente pide nombre, lugar y fecha, e imprime en el navegador el documento completo con la tabla diligenciada.
  - Los Sí/No salen de las afirmaciones 1, 3 y 4.
  - El nombre **no** se envía al servidor.
- **Conservación de un año:** el sistema no borra las grabaciones automáticamente. Hay que eliminarlas a mano (`DELETE /api/study/sessions/{id}`) cuando se cumpla el plazo.

## 5 quater. Asignación de estímulos (`app/catalog/asignaciones.yaml`, 2026-09-29)

- `catalog.yaml` sigue congelado (su sha256 está en `experimental_config.yaml`).
- `asignaciones.yaml` solo resuelve las pruebas que el catálogo deja en `POR_DEFINIR`.
- Es una decisión del investigador, pendiente de revisión de los directores. Cada prueba tiene una escena, la misma para objetivo y piloto:

| Prueba | Escena | Motivo |
|---|---|---|
| OBJ-01 Identificación | DS1-C1 | Silla delante de mesa: dos objetos y una relación. Es la única escena donde YOLO detecta la mesa. |
| OBJ-02 Relación simple | DS1-A8 | Una silla a la derecha, a distancia media. |
| OBJ-03 Decisión | DS1-C2 | Sofá al centro y planta a la derecha. Esperada: **izquierda**. |
| OBJ-04 Actualización | DS1-B2 | Misma escena que C2 con el sofá movido a la izquierda. Esperada: **frente**. |
| OBJ-05 Voz | DS1-A5 | Narrativa corta y neutra (una silla al centro). |
| OBJ-06 / OBJ-07 | — | Preguntas sobre lo escuchado en la sesión. |

- Las esperadas de `decisiones.yaml` salen del diseño geométrico del manifest (no hay objetos en esa dirección), nunca de lo que dijo el sistema.
- Cada respuesta guarda `asignaciones_sha256`.

**Criterio de selección.** Se corrió YOLO26s en local con los umbrales congelados sobre las 18 escenas:
- La mesa ("dining table") **no se detecta** en B1, C3, D1, D2 ni D3; en D3 su confianza es de 0,064, frente a un umbral efectivo de 0,15. Solo se detecta en C1.
- Es un falso negativo sistemático del modelo con ese objeto 3D, no un error del pipeline.
- Esas escenas se excluyen de las pruebas con participantes, para no medir la comprensión sobre una narrativa incompleta. La omisión se reporta como resultado técnico.
- B3 se excluye porque la botella se detecta dos veces.

## 6. Audio: dos archivos distintos

| Archivo | Qué es | Cuándo se guarda |
|---|---|---|
| `audio_narrativa_api.mp3` | Audio TTS que devolvió `/api/detect` y que escuchó el participante (es el estímulo) | Siempre que la prueba usa estímulo; con su sha256 |
| `respuesta_participante.<webm/ogg/m4a>` | Voz del participante | **Solo** si `grabacion.autoriza_grabacion_audio = true` **y** el almacenamiento está fuera del repositorio. En otro caso, 403/409 y no se escribe nada. |
| `grabacion_consentimiento.<webm/ogg/m4a>` | Lectura del consentimiento y respuestas del participante (v3) | Al crear la sesión, solo fuera del repositorio; sin `DATA_ROOT` se guarda únicamente su huella. |

## 7. Correspondencia con la estructura prevista (doc. 27 §Q)

| Prevista | Implementada |
|---|---|
| `evaluation/users/P0X/` | `DATA_ROOT/study/sessions/<fecha>_<p0x>/` (fuera del repositorio) |
| `consentimiento` | `sesion.json → consentimiento` (el formato firmado en papel con el nombre queda **fuera** del sistema) |
| `sesion.json` | `sesion.json` |
| `ESC-0X/` | `respuestas/R00N/` (una carpeta por respuesta; el estímulo va en el registro) |
| `audio_narrativa_api.mp3` / `respuesta_participante.mp3` | Mismos nombres (la grabación conserva el formato del navegador) |
| `transcripcion.txt` / `registro_errores.json` | Campos `respuesta_transcrita` y `errores` + `errores_derivados` en `responses.jsonl` |
| `cuestionario_posterior.json` | `sesion.json → cierre.cuestionario_posterior` |
| `consolidado_resultados.json` | `GET /api/study/consolidado` |

## 8. Pendiente de confirmar (no inventado; depende de los directores)

1. **Consentimiento:** ya existen los documentos propios v0.3 (piloto y objetivo; ver §5 ter).
   - Falta el periodo de conservación.
   - Falta alinear el §9 con el almacenamiento en el servidor.
2. **Aprobación ética institucional:** no está verificado si se requiere (doc. 27 §L, §W-1).
3. **Grabación del participante:** no se sabe si forma parte del protocolo aprobado, quién accede a ella ni por cuánto tiempo se conserva (§W-2, §W-7). El mecanismo existe, pero solo funciona con la autorización registrada.
4. **Estímulos de OBJ-01…OBJ-07:** asignados en `asignaciones.yaml` (§5 quater). Falta la revisión de los directores. Antes de esa asignación estaban `POR_DEFINIR` en el catálogo:
   - El doc. 27 §G propone ESC-01…ESC-07 (oficina, biblioteca…), que **no** son las escenas del Dataset 1.
   - Mientras no se definan, ninguna prueba objetivo puede registrarse como formal.
   - Además, OBJ-03/OBJ-04 ("elección de ruta") se implementan como decisión hipotética sin desplazamiento (§5 bis).
   - El guion conserva el verbo "avanzaría"; conviene que los directores confirmen esa redacción y definan la alternativa esperada por estímulo.
5. **Anclas de las escalas 1–5**, y si se registran por estímulo, solo en el cuestionario posterior o en ambos.
6. **Consentimiento:** publicar una v0.4 de ambos documentos con el §9 corregido y el periodo de conservación, y actualizar `CONSENT_VERSIONS` y `lib/consent.ts`.
7. **Despliegue:** en Azure, `DATA_ROOT` debe ser persistente (`/home`). Sin él, el servidor solo acepta códigos PTEST.

## 9. Pruebas

- **Backend:** `tests/test_study_sessions.py`. Todas usan proveedores simulados y directorios temporales. Incluye las tareas de decisión y el fixture técnico. Incluye el consentimiento v3: 4 afirmaciones, grabación obligatoria, versión según el tipo de participante y huella sin `DATA_ROOT`.
- **Cliente:** `node scripts/check-study.mjs` (21 verificaciones, incluidos la decisión y los dos consentimientos), `tsc --noEmit` y `next build`.
- **E2E en Chromium con axe-core:** contra el backend local con proveedores simulados, fuera del repositorio. Resultado: fase 1 10/10, fase 2 3/3 (incluye un reinicio del backend).
- **E2E del 2026-09-28** (cliente compilado y backend en perfil production local, con la identidad verificada, YOLO26s real y LLM/TTS simulados; fuera del repositorio): PTEST01 y PTEST02 recorren la interfaz de principio a fin. Resultado: 47/47 comprobaciones, más 8/8 tras reiniciar el backend; axe WCAG 2.1 A/AA sin violaciones en 19 pantallas, más 1 tras el reinicio; matriz 401/401/2xx en todas las rutas del investigador. Los scripts y registros de esta E2E no están versionados en el repositorio. Es una prueba técnica simulada: **no es evaluación con usuarios**.
