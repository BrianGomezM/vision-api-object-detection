# Fase 7 — Diagnóstico y corrección controlada de H40 (Groq/LLM)

**Estado: diagnóstica. Ningún archivo de producción fue modificado (`.env`, `app/`, `requirements.txt` intactos). El modelo candidato se validó únicamente mediante una variable de entorno establecida dentro del proceso de un script aislado (`scripts/evaluation/test_llm_fix.py`), nunca persistida.**

## Origen exacto de H40

`app/utils/groq_client.py:35` — `GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")`. `.env` no define `GROQ_MODEL`, así que el valor efectivo siempre fue el default hardcodeado, `llama-3.3-70b-versatile`.

## Verificación del modelo Groq (evidencia, no suposición)

`client.models.list()` contra la API real de Groq, con la API key del proyecto, devolvió 13 modelos activos. **`llama-3.3-70b-versatile` NO aparece en esa lista** — confirma, con la fuente más autoritativa posible (la propia cuenta del proyecto), que el modelo ya no está disponible para esta API key. (Nota: la documentación pública de Groq consultada en paralelo todavía lo listaba como "producción" — discrepancia documentada, no resuelta; se prioriza la evidencia directa de la API sobre la documentación, que puede estar desactualizada respecto a esta cuenta específica.)

Modelos candidatos probados con una llamada real mínima:

| Modelo | Resultado | Nota |
|---|---|---|
| `openai/gpt-oss-20b` | Responde, pero es un modelo de razonamiento: con `max_tokens=150` (el valor que usa el proyecto) consume el presupuesto completo en tokens de razonamiento internos y devuelve `content=''` vacío en los prompts reales de `scene_classifier.py`/`llm_enhancer.py`. **No es un reemplazo mínimo compatible sin cambiar también la configuración de tokens.** |
| `openai/gpt-oss-120b` | Mismo comportamiento que el 20b (modelo de razonamiento). |
| `groq/compound-mini` | Responde correctamente en la prueba trivial, pero es un sistema "compuesto" de Groq con capacidades agénticas (posible uso de herramientas externas) no documentadas en detalle — riesgo de comportamiento no determinista/no reproducible para un caso de uso que debe ser trazable. No se investigó más a fondo por estar fuera del criterio de "cambio mínimo". |
| **`qwen/qwen3.8-27b`** | **Responde de forma directa, sin razonamiento intermedio, con el formato JSON exacto que ya usa `scene_classifier.py` y con el mismo presupuesto de tokens (150-160) que ya usa el proyecto.** Es el reemplazo más directamente comparable al modelo retirado. Estado "preview" en Groq (a tener en cuenta como riesgo de estabilidad futura, no como bloqueo). |

**Candidato recomendado: `qwen/qwen3.8-27b`** — único que fue evidenciado como compatible con la configuración de tokens y el formato de prompt existentes sin ningún otro cambio.

## Prueba antes/después (Paso 5) — las 9 escenas evaluables de la Fase 6

| Imagen | Escenario real | Antes (heurístico, Fase 6) | Después (LLM real, `qwen/qwen3.8-27b`) | ¿LLM usado? | Confianza | Observaciones |
|---|---|---|---|---|---|---|
| office | oficina | oficina (alta) ✓ | oficina (alta) ✓ | Sí | alta | Sin cambio, ya era correcto |
| biblioteca | biblioteca | tienda (alta) ✗ | **comedor (alta) ✗** | Sí | alta | **Sigue mal.** La categoría "biblioteca" tampoco existe en el prompt del LLM — H40 no es la causa de este error específico |
| living-room | sala de estar | sala de estar (alta) ✓ | sala de estar (alta) ✓ | Sí | alta | Sin cambio, ya era correcto |
| studio2 | ambiguo | sala de estar (media) ✗ | **espacio interior (baja)** | Sí | baja | Mejora: el LLM reconoce la ambigüedad y no fuerza una categoría específica (confidence baja suprime el texto en la narrativa real) |
| universidad | exterior/campus | tienda (alta) ✗ | **espacio interior (baja)** | Sí | baja | Mejora: de "confiadamente incorrecto" a "honestamente no determinado" |
| baño | baño | sala de estar (media) ✗ | **baño (alta) ✓** | Sí | alta | **Corregido por el LLM** |
| sala de espera | sala de espera | sala de estar (baja) | comedor (media) ✗ | Sí | media | Sigue mal, y la confianza subió de "baja" a "media" (antes se suprimía en narrativa, ahora no) |
| calle low-poly | exterior/calle | exterior (media) ✓ | **calle urbana (alta) ✓** | Sí | alta | Mejora: el LLM generalizó a una etiqueta más específica y correcta, no limitada a las categorías literales del prompt |
| sofá (sala real) | sala de estar | oficina (media) ✗ | **sala de estar (alta) ✓** | Sí | alta | **Corregido por el LLM** |

**Resumen:** 3/9 correctas antes → **5/9 correctas + 2/9 "honestamente no determinadas" (antes eran errores confiados)** después. Persisten 2/9 errores reales (biblioteca, sala de espera) — ambos por ausencia de esa categoría en el prompt del LLM, no por H40.

## Validación de generación narrativa (Paso 6)

5 escenas probadas (oficina, biblioteca, doméstica/sala de estar, exterior/calle, ambigua/estudio) — ver `narrative_after_fix.json` para el texto completo y los objetos de entrada exactos.

- Todas usaron el LLM real (sin fallback).
- En las 5, el texto generado **solo menciona objetos que efectivamente estaban en la entrada** — no se detectó invención de objetos ausentes.
- Perspectiva egocéntrica y español mantenidos en las 5.
- Se detectó un problema menor de gramática (pluralización): *"2 mesa de comedores"* en la narrativa de biblioteca, en vez de "2 mesas de comedor" — defecto de calidad narrativa, no de fidelidad a las detecciones.
- Tiempos de respuesta: 330-600 ms por llamada, similar al orden de magnitud esperado para una API de inferencia rápida.

## Comparación LLM vs. heurístico (Paso 7)

| Dimensión | LLM (`qwen/qwen3.8-27b`) | Heurístico (fallback actual) |
|---|---|---|
| Precisión en las 9 escenas | 5/9 correctas + 2/9 no forzadas | 3/9 correctas, 6/9 forzadas incorrectamente |
| Biblioteca | Sigue mal (comedor) | Mal (tienda/oficina) |
| Manejo de ambigüedad | Reconoce baja confianza y no fuerza (studio2, universidad) | Siempre fuerza, sin distinguir incertidumbre real |
| Vocabulario de salida | Puede generar etiquetas nuevas no listadas en los ejemplos del prompt ("calle urbana") | Estrictamente limitado a los 10 valores fijos de `_CONTEXT_GROUPS` |
| Estabilidad/reproducibilidad | Determinismo parcial (`temperature=0.1`, pero no garantizado bit a bit) | Totalmente determinista |
| Latencia | 330-600 ms por llamada (dependencia de red) | <1 ms (cálculo local) |
| Fallos observados | Ninguno en 9+5=14 llamadas reales de esta prueba | No aplica (no fue probado como "falla", es el camino normal cuando el LLM no está disponible) |

## Regresión (Paso 9)

No se modificó ningún archivo de producción, por lo que no existe superficie de regresión real en esta fase — la prueba se ejecutó con el modelo sobrescrito solo dentro del proceso del script aislado. Se confirma explícitamente que detección, análisis espacial, TTS y endpoints no fueron tocados (`git status` limpio en `app/`, `requirements.txt`, `.env`).
