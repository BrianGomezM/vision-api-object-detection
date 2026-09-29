# Cambios de la línea base de regresión

Registro de cada regeneración **intencional** de `baseline_phase2a.json`. La línea base
solo se regenera tras una decisión metodológica aprobada, y el diff queda documentado aquí.

## 2026-09-29 — `metricas.tts_modelo` en la respuesta de `/api/detect`

**Motivo:** registrar qué voz generó el audio (el estudio con usuarios pasa a usar Azure Speech, Salomé). Las 41 imágenes solo añaden `endpoint.json.metricas.tts_modelo` (en la regresión, la voz por defecto); todo lo demás es **idéntico** (verificado quitando la clave y comparando).

## 2026-09-29 — clave informativa `umbral` en la respuesta de `/api/detect`

**Motivo:** mostrar en Detectar el umbral efectivo de cada objeto (decisión del investigador: mantener la regla `min()` y explicarla en la interfaz).

**Resultado:** las 41 imágenes solo añaden `endpoint.json.umbral` (umbral de Ajustes, piso, regla y, por objeto, confianza, mínimo de la clase y umbral efectivo). Todo lo demás es **idéntico**: detecciones, confianzas, cajas, análisis espacial, espacio libre, decisión, prompts y narrativa. Verificado quitando la clave nueva y comparando con la línea base anterior. La regla del umbral no cambia.

## 2026-09-27 — regla del umbral por clase `max()` → `min()` (checkpoint final pre-F4)

**Motivo:** `docs/AUDITORIA_REGLA_UMBRAL_FASE10.md` §5 (decisión) y `experimental_config.yaml`.

**Resultado:**
- 22 de 41 imágenes cambian. **50 detecciones añadidas**, con confianzas entre 0,157 y 0,349; ninguna eliminada.
- Coincide con lo que la fase 2A midió al comparar las dos reglas (`phase2a/detection/analysis_summary.txt`: 22/41 imágenes, 50 detecciones, 0,157–0,349).
- El preprocesado no cambia: `resize` es idéntico en las 41 imágenes.
- La nueva línea base reproduce **41/41** lo que la fase 2A registró con `regla=min` en detecciones, confianzas y cajas, instrucción de movimiento, espacio libre y líneas del prompt.

**Columnas de la tabla:** cada cambio se propaga por el pipeline. El análisis espacial (`analyzed`) cambia en las 22 imágenes; `free_space`, `decision` y `narrativa_final` solo en las que se indican. El prompt del LLM y la respuesta de `/api/detect` cambian en las 22.

| Imagen | Detecciones añadidas (clase, confianza) | Partes del pipeline que cambian |
|---|---|---|
| W3D-G-04-Biblioteca.png | dining table 0.232, potted plant 0.263, potted plant 0.267 | analyzed, free_space, decision, narrativa_final |
| W3D-G-05-Ciudad.png | potted plant 0.282 | analyzed |
| W3D-G-06-Universidad.png | backpack 0.311, bench 0.338, person 0.324 | analyzed, free_space |
| W3D-I-JC3D-03-salad.png | bottle 0.309, couch 0.328 | analyzed |
| W3D-I-JC3D-04-school.png | chair 0.304, dining table 0.157, dining table 0.183, dining table 0.186, dining table 0.189, dining table 0.274, dining table 0.318 | analyzed, free_space, decision, narrativa_final |
| W3D-I-SKF-01-british.png | bottle 0.278, bottle 0.333, bottle 0.344, dining table 0.209, dining table 0.323 | analyzed, free_space |
| W3D-I-SKF-02-cartoon-low-poly.png | dining table 0.198 | analyzed, free_space |
| W3D-I-SKF-04-coffee.png | chair 0.335, dining table 0.199, potted plant 0.338 | analyzed |
| W3D-I-SKF-05-interior.png | couch 0.31, couch 0.34, dining table 0.178, vase 0.267 | analyzed |
| W3D-I-SKF-06-kitchen.png | dining table 0.325 | analyzed, free_space, decision, narrativa_final |
| W3D-I-SKF-07-living-room.png | dining table 0.175, potted plant 0.304 | analyzed |
| W3D-I-SKF-09-office.png | chair 0.317 | analyzed |
| W3D-I-SKF-10-room.png | couch 0.328, dining table 0.166, potted plant 0.257, vase 0.302 | analyzed, decision, narrativa_final |
| W3D-I-SKF-13-the-grand-budapest-hotel.png | dining table 0.179, potted plant 0.349 | analyzed, free_space |
| W3D-I-SKF-14-waiting-room.png | dining table 0.17 | analyzed, free_space |
| 01_persona_bolso.jpg | suitcase 0.305 | analyzed |
| 04_cocina_utensilios.jpg | dining table 0.158 | analyzed, free_space, decision, narrativa_final |
| 06_bicicleta_persona.jpg | backpack 0.324 | analyzed |
| 08_mercado_productos.jpg | dining table 0.159 | analyzed |
| 09_objetos_pequenos.jpg | dining table 0.179, dining table 0.219, dining table 0.238, dining table 0.276 | analyzed, free_space, decision, narrativa_final |
| 11_escritorio.jpeg | dining table 0.272 | analyzed, free_space |
| 12_sala.jpg | wine glass 0.334 | analyzed |
