# Instrumento de evaluación con usuarios (Objetivo Específico 3)

- **Fecha:** 2026-09-28.
- **Fuente metodológica:** "27 - Preparación evaluación final Objetivo 3" (diseño exploratorio, una sesión por participante, sin navegación 3D; el participante escucha el audio de la API y responde verbalmente).
- **Estado:** el instrumento está implementado y probado solo con datos técnicos (PTEST01, PTEST02). **La evaluación con usuarios objetivo NO se ha ejecutado.**
- **Código:** backend `app/routes/study.py`, `app/catalog/loader.py`; cliente `components/study/*`, `lib/study-protocol.ts`, `hooks/use-study.ts`.

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

## 3. Contrato (v2) — todas las rutas exigen `X-API-Key` del investigador

| Ruta | Cambio | Notas |
|---|---|---|
| `POST /api/study/sessions` | **Rompe el contrato anterior** | Recibe `codigo`, `tipo_participante`, `ficha`, `consentimiento`, `grabacion`, `contexto`, `investigador?` y `notas?`. Ya **no** acepta `nombre`, `edad` ni `genero` (400 si se envían). Respuestas: 409 si el código está repetido o si un código real se guardaría dentro del repositorio; 400 si el consentimiento está incompleto. |
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

## 5 ter. Consentimiento informado (borrador)

- **Fuente única:** `visionnav-client/lib/consent.ts`.
  - El asistente de sesión muestra ese texto para leerlo en voz alta y toma de ahí las tres afirmaciones obligatorias y la afirmación 4 (grabación, opcional).
  - El botón **"Descargar consentimiento informado"** genera un `.docx` (`lib/docx.ts`, sin dependencias) para la revisión de los directores.
- **Adaptación:** se adapta el formato de otro estudio.
  - Se conserva: participación voluntaria, confidencialidad y lectura en voz alta con consentimiento verbal.
  - Se reescribe: propósito (§F), procedimiento sin tareas físicas (§H), riesgos, y grabación solo de audio y opcional (sin video).
- **Marcado como `[PENDIENTE]`** (resaltado en el `.docx`): aprobación ética, almacenamiento, acceso y conservación de la información y de las grabaciones, duración, título, institución y contacto.
  - El documento declara que **no está aprobado**.
- **Trazabilidad:** la versión del texto (`TG2-OE3-CI v0.1…`) se guarda en `consentimiento.formato_referencia` de cada sesión.

## 6. Audio: dos archivos distintos

| Archivo | Qué es | Cuándo se guarda |
|---|---|---|
| `audio_narrativa_api.mp3` | Audio TTS que devolvió `/api/detect` y que escuchó el participante (es el estímulo) | Siempre que la prueba usa estímulo; con su sha256 |
| `respuesta_participante.<webm/ogg/m4a>` | Voz del participante | **Solo** si `grabacion.autoriza_grabacion_audio = true` **y** el almacenamiento está fuera del repositorio. En otro caso, 403/409 y no se escribe nada. |

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

1. **Consentimiento:** el formato disponible (`Formatos/Consentimiento informado.docx`) pertenece a **otro estudio** (doctorado; tareas físicas; grabación de audio **y video**).
   - No describe este protocolo.
   - El sistema registra las tres afirmaciones de ese formato, pero **no afirma** que exista un consentimiento aprobado para el Objetivo 3.
2. **Aprobación ética institucional:** no está verificado si se requiere (doc. 27 §L, §W-1).
3. **Grabación del participante:** no se sabe si forma parte del protocolo aprobado, quién accede a ella ni por cuánto tiempo se conserva (§W-2, §W-7). El mecanismo existe, pero solo funciona con la autorización registrada.
4. **Estímulos de OBJ-01…OBJ-07:** están `POR_DEFINIR` en el catálogo.
   - El doc. 27 §G propone ESC-01…ESC-07 (oficina, biblioteca…), que **no** son las escenas del Dataset 1.
   - Mientras no se definan, ninguna prueba objetivo puede registrarse como formal.
   - Además, OBJ-03/OBJ-04 ("elección de ruta") se implementan como decisión hipotética sin desplazamiento (§5 bis).
   - El guion conserva el verbo "avanzaría"; conviene que los directores confirmen esa redacción y definan la alternativa esperada por estímulo.
5. **Anclas de las escalas 1–5**, y si se registran por estímulo, solo en el cuestionario posterior o en ambos.
6. **Consentimiento:** revisar el borrador descargable (§5 ter) y completar los `[PENDIENTE]`.
7. **Despliegue:** en Azure, `DATA_ROOT` debe ser persistente (`/home`). Sin él, el servidor solo acepta códigos PTEST.

## 9. Pruebas

- **Backend:** `tests/test_study_sessions.py`. Todas usan proveedores simulados y directorios temporales. Incluye las tareas de decisión y el fixture técnico. El total de la suite es 397 correctas y 4 omitidas.
- **Cliente:** `node scripts/check-study.mjs` (21 verificaciones, incluidas la decisión, el consentimiento y el `.docx`), `tsc --noEmit` y `next build`.
- **E2E en Chromium con axe-core:** contra el backend local con proveedores simulados, fuera del repositorio. Resultado: fase 1 10/10, fase 2 3/3 (incluye un reinicio del backend).
- **E2E del 2026-09-28** (cliente compilado y backend en perfil production local, con la identidad verificada, YOLO26s real y LLM/TTS simulados; fuera del repositorio): PTEST01 y PTEST02 recorren la interfaz de principio a fin. Resultado: 47/47 comprobaciones, más 8/8 tras reiniciar el backend; axe WCAG 2.1 A/AA sin violaciones en 19 pantallas, más 1 tras el reinicio; matriz 401/401/2xx en todas las rutas del investigador. Los scripts y registros de esta E2E no están versionados en el repositorio. Es una prueba técnica simulada: **no es evaluación con usuarios**.
