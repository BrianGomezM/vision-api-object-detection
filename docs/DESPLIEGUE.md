# Despliegue

El servicio se ejecuta como un contenedor Docker en Azure App Service. La imagen incluye las dependencias y los pesos del modelo, así que al arrancar no se descarga nada.

## Imagen Docker

- Base: `python:3.13.15-slim`, fijada por versión y digest.
- PyTorch 2.13.0 para CPU y el resto de dependencias según `requirements-docker.lock.txt`.
- Pesos `yolo26s.pt` descargados de la versión v8.4.0 de Ultralytics y verificados con su SHA-256 durante la construcción y al arrancar.
- Perfil `production` y usuario sin privilegios; solo `/tmp` es escribible.
- Servidor: Gunicorn con un worker de Uvicorn en el puerto 8000.

Construir y ejecutar localmente:

```bash
docker build --build-arg APP_COMMIT=$(git rev-parse HEAD) -t visionnav-api .
docker run -p 8000:8000 -e GROQ_API_KEY=... -e AZURE_SPEECH_KEY=... -e AZURE_SPEECH_REGION=... visionnav-api
curl http://127.0.0.1:8000/api/health
```

## Integración continua

El flujo `.github/workflows/main_visionnav-api.yml` tiene tres etapas:

1. **Pruebas:** instala las mismas versiones que la imagen y ejecuta `pytest`.
2. **Construcción:** si las pruebas pasan, construye la imagen y la publica en GitHub Container Registry con la etiqueta del commit.
3. **Despliegue:** actualiza el App Service `visionnav-api` con esa imagen.

Una actualización de la rama `main` ejecuta las tres etapas. Una solicitud de cambio hacia `main` solo ejecuta las pruebas.

## Configuración en Azure

| Elemento | Valor |
|---|---|
| Servicio | Azure App Service, plan Linux B1 (1 núcleo, 1,75 GB) |
| URL | https://visionnav-api-c5g0gxamh2bug7be.canadacentral-01.azurewebsites.net |
| Puerto | `WEBSITES_PORT=8000` |
| Comprobación de estado | `/api/health` |

Variables de entorno que se configuran en el servicio:

| Variable | Uso |
|---|---|
| `GROQ_API_KEY` | Modelo de lenguaje (obligatoria) |
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | Voz, Azure AI Speech (es-CO-SalomeNeural). Sin ellas, la respuesta es solo texto |
| `CORS_ORIGINS` | Sitios web autorizados para llamar a la API desde el navegador |
| `RESEARCHER_API_KEYS` | Clave para las métricas y las rutas del investigador (opcional) |

## Encender y apagar el servicio

El plan se cobra por hora mientras está encendido. `scripts/azure-service.ps1` lo controla con Azure CLI:

```powershell
.\scripts\azure-service.ps1 start
.\scripts\azure-service.ps1 stop
.\scripts\azure-service.ps1 status
```

Después de encenderlo, el contenedor tarda entre 30 y 60 segundos en responder.
