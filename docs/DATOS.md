# Datos que guarda el servicio

El código y los estímulos están en el repositorio. Los datos que el servicio genera se escriben en la carpeta indicada por `DATA_ROOT`, separados por tipo. Si `DATA_ROOT` no está definida, se usan carpetas dentro del proyecto que Git ignora.

| Tipo | Dato | Carpeta (con `DATA_ROOT`) | Conservación |
|---|---|---|---|
| Servicio | Imagen anotada con las cajas de los objetos | `product/annotated/` | En `production` se borra al responder; en los demás perfiles se conservan las últimas 10 |
| Servicio | Audio de la narrativa | `product/audio/` | En `production` se borra al responder; en los demás perfiles se conservan los últimos 5 |
| Servicio | Telemetría: identificador de la solicitud, número de objetos, confianza media, tiempos y escenario, sin imagen ni texto | `product/telemetry/` | Sin límite |
| Pruebas | Historial de `/api/test/*` | `evaluation/api_tests/` | Permanente |
| Caché | Traducciones de etiquetas que no están en el diccionario fijo | `cache/` | Sin límite |

## Privacidad

- En el perfil `production`, el servicio no conserva las imágenes de los usuarios ni los audios generados después de responder.
- La telemetría no guarda imágenes ni textos.
- Los datos de participantes no se versionan: la carpeta `study_data/` está en `.gitignore`, y la correspondencia entre código y nombre solo existe fuera del sistema. La herramienta de sesiones está en la rama `herramientas/estudio-usuarios`.
