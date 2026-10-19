# Evaluación

Este documento indica dónde están los scripts y los resultados de cada evaluación del sistema.

## Selección del modelo

Se compararon YOLO26s, Faster R-CNN, Mask R-CNN y SSD-MobileNet V2 sobre las 12 imágenes de `test_images/`, con umbrales de 0,3, 0,5 y 0,7 (144 ejecuciones). Los servicios de los modelos alternativos están en `app/experimental/` y no forman parte del servicio desplegado.

## Pruebas técnicas

| Evaluación | Script | Resultados |
|---|---|---|
| Comparación de YOLO26s entrenado en COCO frente a Objects365 (29 imágenes Web 3D) | `scripts/compare_checkpoints.py` | `evaluation/results/checkpoint_comparison/` |
| Clases de contexto para identificar el escenario | `scripts/evaluation/phase10_scene_context_experiment.py` | `evaluation/results/phase10/` |
| Tiempos de respuesta, CPU y memoria (NF-01 a NF-08) | `scripts/evaluation/plan_pruebas_nf.py` | `evaluation/results/plan_pruebas/` |
| Solicitudes simultáneas | `scripts/hardening/concurrency_bench.py` | `evaluation/results/hardening/` |
| Plan de pruebas (33 casos) | `pytest` | [PLAN_PRUEBAS.md](PLAN_PRUEBAS.md) |

Las imágenes Web 3D están en `evaluation/images/web3d/`, y su origen y licencia, en `evaluation/metadata/fuentes_y_licencias.csv`.

## Evaluación con usuarios

Estudio exploratorio por videollamada. La persona escucha la descripción de una escena y responde de forma verbal; el investigador registra las respuestas.

- **Escenas:** seis escenas tridimensionales de interiores en `stimuli/estudio2/`: práctica, oficina, cocina-comedor, dos versiones de una sala de estar y dormitorio.
- **Audios congelados:** la narrativa y el audio de cada escena se generaron una sola vez con la voz `es-CO-SalomeNeural` y se guardaron con su SHA-256, para que todas las sesiones escuchen exactamente lo mismo. Se generan con `scripts/study/freeze_study_audio.py` y se verifican con `scripts/study/verify_frozen_audio.py`.
- **Pruebas:** OBJ-01 a OBJ-07 (objetos, ubicación, dirección libre, cambios, voz, suficiencia y utilidad), definidas en `app/catalog/catalog.yaml`.
- **Escala:** de 1 a 5, con las opciones leídas en voz alta (1 nada, 2 poco, 3 más o menos, 4 bastante y 5 mucho).
- **Registro:** `app/routes/study.py` guarda cada sesión con un código, sin nombres, y la exporta como un archivo `.zip` con la hoja de cálculo y los registros originales.

Los datos de participantes no se guardan en el repositorio.
