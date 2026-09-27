# Fase 2A — Experimentos técnicos controlados (sin participantes)

**Naturaleza:** infraestructura auxiliar de evaluación. Estos scripts y resultados **no** forman parte del producto del Trabajo de Grado (imagen → detección → narrativa egocéntrica → audio). Sirven para obtener evidencia técnica antes de fijar la configuración experimental. Ninguna configuración queda declarada como oficial.

**Reglas aplicadas:**
- No se modificó código del producto.
- Los scripts importan las funciones reales (`resize_image`, `_CLASS_MIN_CONF`, `_NAV_CLASSES`, `analyze_spatial`, `estimate_steps`, `calculate_free_space`, `decide_movement`, `_seleccionar_y_ordenar`, `generate_description`, `classify_scene`, `_synthesize_gemini_tts`) y el `tts_service.py` exacto del commit desplegado `3a1ddb6`.
- Los audios de `tts/audio/` son **salidas de prueba, no estímulos**.

## Cómo reproducir

Desde la raíz del backend, con el `.venv` del proyecto:

```
python -m scripts.evaluation.phase2a_detection            # A y B (~15 min: CPU + GPU)
python -m scripts.evaluation.phase2a_detection_analysis
python -m scripts.evaluation.phase2a_llm                  # C (requiere GROQ_API_KEY; 144 llamadas)
python -m scripts.evaluation.phase2a_llm_analysis
PHASE2A_SCRATCH=<dir temporal> python -m scripts.evaluation.phase2a_tts   # D
python -m scripts.evaluation.phase2a_tts_analysis
```

Cada carpeta contiene `run_meta.jsonl` con:
- commit y estado del árbol;
- versiones de paquetes;
- hardware;
- valores no secretos del entorno;
- SHA-256 de pesos y módulos.

## Archivos

| Carpeta | Archivo | Contenido |
|---|---|---|
| `detection/` | `raw_predictions.jsonl` | Cajas crudas de YOLO (conf ≥ 0,15) por imagen × imgsz × dispositivo; tiempos (3 repeticiones), CPU, RSS, hashes de imagen original y procesada |
| | `configurations.jsonl` | Por imagen × imgsz × regla: detecciones filtradas, objetos analizados, **líneas exactas que recibe el generador narrativo**, espacio libre, instrucción de movimiento |
| | `validation_vs_run_yolo.jsonl` | Verificación de que el filtro replicado coincide con `run_yolo()` del producto (82/82) |
| | `summary_config.csv`, `compare_rule_min_vs_max.csv`, `compare_imgsz_640_vs_1280.csv`, `timing_resources.csv`, `gt_presence_5_images.csv`, `analysis_summary.txt` | Tablas de análisis |
| `llm/` | `llm_runs.jsonl` | 72 ejecuciones (3 modelos × 8 imágenes × 3 repeticiones): narrativa, prompt, escenario, latencias, errores, chequeos automáticos |
| | `llm_stability.jsonl`, `summary_by_model.csv`, `analysis_summary.txt` | Estabilidad y resumen por modelo |
| `tts/` | `tts_runs.jsonl`, `summary_tts.csv` | Intentos Gemini (1 por modelo) y edge-tts (4 voces × 2 textos × 3 repeticiones) |
| | `audio/*.mp3` | Primera repetición de cada combinación edge-tts (salida de prueba) |

## Limitaciones conocidas

- La ground truth de presencia existe solo para 5 imágenes (Fase 6). En el resto solo se reportan diferencias, no aciertos.
- Los tiempos se midieron en un portátil (Intel, 8 hilos lógicos; GPU NVIDIA disponible). **No son tiempos de Azure (B1).** CPU es la condición más cercana.
- El RSS es del proceso completo y acumulativo (incluye el contexto CUDA). No es un consumo aislado por configuración; la medida comparable es el tiempo de CPU por inferencia.
- Los chequeos de narrativa son automáticos (coincidencia de términos). No sustituyen la revisión humana de coherencia.
