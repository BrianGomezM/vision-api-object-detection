# VisionNav API — rama `herramientas/ajuste-fino`

Esta rama conserva las herramientas para preparar un conjunto de imágenes propio y reentrenar el detector. Son una base para trabajo futuro: no se usaron en la evaluación ni forman parte del servicio desplegado, que está en la rama `main`.

La rama parte del mismo código que `main` antes de separar las herramientas, así que también incluye la API completa. Para instalarla y ejecutarla, siga las instrucciones del README de `main`.

## Para qué sirve

El modelo preentrenado solo reconoce las 80 categorías de COCO, de modo que no detecta puertas, escaleras ni otros elementos importantes para orientarse. Estas herramientas permiten reunir imágenes etiquetadas de esos elementos y ajustar el modelo con ellas.

## Qué contiene

| Componente | Ubicación | Función |
|---|---|---|
| Subir imágenes | `POST /api/dataset/upload` | Guarda una imagen y, si se pide, genera sus etiquetas con el detector actual |
| Estadísticas | `GET /api/dataset/stats` | Cuenta las imágenes, las etiquetas y las clases reunidas |
| Preparar el conjunto | `POST /api/finetune/prepare` | Separa las imágenes en entrenamiento y validación y genera `data.yaml` |
| Estado | `GET /api/finetune/status` | Indica si el conjunto está listo para entrenar |
| Entrenamiento | `scripts/train_custom.py` | Ajusta YOLO26s con el conjunto preparado |
| Clases propuestas | `data/custom_dataset.yaml` | Clases arquitectónicas sugeridas (escaleras, marcos de puerta, rampas, entre otras) |

Los endpoints se publican solo en el perfil `development`.

## Cómo se usa

1. Inicie la API con `APP_PROFILE=development`.
2. Suba imágenes con `POST /api/dataset/upload`.
3. Cuando haya al menos 50 imágenes etiquetadas, llame a `POST /api/finetune/prepare`.
4. Entrene con `python scripts/train_custom.py`.

Las imágenes se guardan en la carpeta definida por `DATA_ROOT` (o en `dataset/`, que Git ignora).

## Autor

Brayan Julio Gómez Muñoz — Universidad del Valle.
