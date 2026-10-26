# VisionNav API

API REST que describe una escena tridimensional para personas con ceguera total. Recibe una imagen de la vista, detecta los objetos, calcula dónde está cada uno respecto al usuario y devuelve una descripción en segunda persona, en texto y en audio.

> Ejemplo de respuesta: *"Parece que estás en una sala de estar. Sofá a tu derecha a aproximadamente 3 pasos. Mesa de comedor frente a ti a aproximadamente 4 pasos. El paso al frente está bloqueado. Puedes avanzar hacia la izquierda."*

Trabajo de grado de Ingeniería de Sistemas, Universidad del Valle.

## Cómo funciona

1. Valida la imagen (JPEG o PNG, hasta 10 MB).
2. Detecta los objetos con **YOLO26s**.
3. Ubica cada objeto a la izquierda, al centro o a la derecha, y estima a cuántos pasos está.
4. Calcula por dónde hay espacio libre para avanzar.
5. Redacta la descripción con un modelo de lenguaje (Groq).
6. La convierte en voz con Azure AI Speech (voz es-CO-SalomeNeural).

## Requisitos

- Python 3.13 (también funciona con 3.12)
- Una clave de [Groq](https://console.groq.com) para la descripción
- Una clave de [Azure AI Speech](https://portal.azure.com) para la voz (el nivel gratuito F0 es suficiente)
- Docker, solo si se quiere ejecutar en contenedor

## Instalación

**Windows, de forma guiada:** abra `docs/instalacion/index.html` en el navegador y siga los pasos. El archivo `instalar.bat` hace la instalación con doble clic.

**Manual:**

```bash
git clone https://github.com/BrianGomezM/vision-api-object-detection.git
cd vision-api-object-detection
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
pip install -r requirements.txt
copy .env.example .env            # Linux/macOS: cp .env.example .env
```

Después:

- Escriba sus claves en `.env` (`GROQ_API_KEY`, `AZURE_SPEECH_KEY` y `AZURE_SPEECH_REGION`).
- Descargue los pesos `yolo26s.pt` desde la [versión v8.4.0 de Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt) y déjelos en la carpeta del proyecto.

## Uso

```bash
python run.py
```

La API queda en `http://127.0.0.1:8000`, y la documentación interactiva, en `http://127.0.0.1:8000/docs`.

```bash
curl -F "file=@test_images/05_sala_muebles.jpg" http://127.0.0.1:8000/api/detect
curl -F "file=@test_images/05_sala_muebles.jpg" -F "audio=true" http://127.0.0.1:8000/api/detect --output descripcion.mp3
```

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/detect` | POST | Descripción de la escena en JSON, o el audio MP3 con `audio=true` |
| `/api/health` | GET | Estado del detector, del modelo de lenguaje y de la voz |
| `/api/tts/models` | GET | Voces disponibles |

Los parámetros, la respuesta y los códigos de error están en [docs/API.md](docs/API.md).

## Docker

```bash
docker build -t visionnav-api .
docker run -p 8000:8000 -e GROQ_API_KEY=... -e AZURE_SPEECH_KEY=... -e AZURE_SPEECH_REGION=... visionnav-api
```

La imagen descarga y verifica los pesos del modelo durante la construcción.

## Pruebas

```bash
pip install -r requirements-test.txt
python -m pytest -m "not entorno_referencia and not live_smoke"
```

Las pruebas usan versiones simuladas del detector, del modelo de lenguaje y de la voz, así que no consumen servicios externos. Se ejecutan automáticamente antes de cada despliegue.

## Estructura

```
app/
  core/          procesamiento completo de una imagen
  services/      detección, análisis espacial, pasos, espacio libre, escenario, narrativa y voz
  routes/        endpoints de la API
  utils/         clientes de servicios externos y utilidades
  experimental/  modelos comparados durante la selección (no se despliegan)
scripts/         scripts de evaluación y medición
evaluation/      imágenes y resultados de las evaluaciones
stimuli/         escenas usadas en la evaluación con usuarios
test_images/     imágenes de prueba
tests/           pruebas automatizadas
docs/            documentación
```

## Documentación

| Documento | Contenido |
|---|---|
| [API.md](docs/API.md) | Endpoints, parámetros, respuesta, errores y límites |
| [ARQUITECTURA.md](docs/ARQUITECTURA.md) | Capas, flujo de una solicitud y perfiles de ejecución |
| [DESPLIEGUE.md](docs/DESPLIEGUE.md) | Imagen Docker, integración continua y Azure |
| [DATOS.md](docs/DATOS.md) | Qué datos guarda el servicio y por cuánto tiempo |
| [EVALUACION.md](docs/EVALUACION.md) | Scripts y resultados de cada evaluación |
| [PLAN_PRUEBAS.md](docs/PLAN_PRUEBAS.md) | Los 33 casos del plan de pruebas y su resultado |
| [REPRODUCIBILIDAD.md](docs/REPRODUCIBILIDAD.md) | Pesos, versiones y configuración registrada |

## Ramas

| Rama | Contenido |
|---|---|
| `main` | La API y la evidencia de sus evaluaciones. Se despliega en Azure. |
| `herramientas/estudio-usuarios` | Herramienta usada para registrar las sesiones de la evaluación con usuarios. |
| `herramientas/ajuste-fino` | Endpoints y scripts para preparar un conjunto de datos y reentrenar el detector (trabajo futuro). |

Las ramas `herramientas/*` conservan herramientas de apoyo que no forman parte del servicio.

## Autor

Brayan Julio Gómez Muñoz — Universidad del Valle, Escuela de Ingeniería de Sistemas y Computación.
