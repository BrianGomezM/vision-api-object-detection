# VisionNav API — rama `herramientas/estudio-usuarios`

Esta rama conserva la herramienta con la que se registraron las sesiones de la evaluación con usuarios. No forma parte del servicio: la API desplegada está en la rama `main`.

La rama parte del mismo código que `main` antes de separar las herramientas, así que también incluye la API completa. Para instalarla y ejecutarla, siga las instrucciones del README de `main`.

## Qué contiene

| Componente | Ubicación | Función |
|---|---|---|
| Sesiones | `app/routes/study.py` | Crear sesiones identificadas con un código, registrar el consentimiento, las respuestas y el cierre |
| Exportación | `app/study_export.py` | Descargar la evidencia de una sesión en un `.zip` con una hoja de cálculo y los registros originales |
| Catálogo de pruebas | `app/catalog/`, `app/routes/catalog.py` | Pruebas OBJ-01 a OBJ-07, escenas asignadas y respuestas de referencia |
| Escenas y audios | `stimuli/` | Escenas del estudio (`estudio2`) y del Dataset 1, con los audios congelados y su SHA-256 |
| Scripts | `scripts/study/` | Generar los audios congelados, verificarlos con transcripción automática y revisar las escenas |
| Importación | `scripts/import_dataset1.py`, `scripts/import_estudio2.py` | Copiar las escenas desde el generador y registrar su origen |

## Cómo se usa

1. Configure `APP_PROFILE=study` y una clave en `API_KEYS` dentro de `.env`.
2. Inicie la API con `python run.py`.
3. Use las rutas `/api/catalog` y `/api/study/*` con la cabecera `X-API-Key`.

El perfil `study` no admite respuestas degradadas: si el modelo de lenguaje o la voz no responden, la solicitud falla en lugar de entregar una respuesta incompleta.

## Datos de participantes

Las sesiones se guardan en la carpeta definida por `DATA_ROOT` (o en `study_data/`, que Git ignora). Cada sesión se identifica con un código y no guarda nombres; la correspondencia entre código y persona solo existe fuera del sistema.

## Autor

Brayan Julio Gómez Muñoz — Universidad del Valle.
