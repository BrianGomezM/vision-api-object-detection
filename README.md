VisionNav API

API REST desarrollada para generar descripciones narrativas egocéntricas accesibles a partir de imágenes de escenas Web 3D.

El sistema recibe una imagen, detecta objetos relevantes, analiza su posición respecto al observador, genera una descripción en español y, cuando se solicita, produce el audio de la narrativa.

El proyecto corresponde al backend del Trabajo de Grado II. El cliente web se encuentra en un repositorio independiente.

1. Requisitos

Para ejecutar el proyecto localmente se requiere como mínimo:

Python 3.13.15.

Git.

Los pesos yolo26s.pt.

Una clave de Groq para generar la narrativa.

Una clave de Google AI Studio para generar el audio.

Docker, únicamente si se desea ejecutar la versión contenerizada.

Pesos del modelo

El archivo yolo26s.pt debe estar disponible en la raíz del proyecto cuando se ejecuta la API directamente con Python.

SHA-256 esperado:

646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b

2. Obtener el proyecto

Clonar el repositorio y entrar en su directorio:

git clone https://github.com/BrianGomezM/vision-api-object-detection.git
cd vision-api-object-detection

El repositorio no contiene las claves de los servicios externos ni el archivo .env.

3. Configuración

Crear el archivo .env a partir del ejemplo incluido:

cp .env.example .env

En Windows PowerShell:

Copy-Item .env.example .env

Como mínimo, configurar:

GROQ_API_KEY=...
GOOGLE_API_KEY=...
APP_PROFILE=development

Opcional — voces de Azure AI Speech (solo para el módulo Detectar):

AZURE_SPEECH_KEY=...           (Clave 1 del recurso Speech "visionnavSpeech", plan F0)
AZURE_SPEECH_REGION=canadacentral
AZURE_TTS_RATE=-5%             (opcional; ritmo de la voz)

- Con ambas variables, el selector "Voz de la narrativa (TTS)" de Ajustes ofrece Salomé
  (es-CO, propuesta) y Gonzalo (es-CO). Sin ellas, esas voces no aparecen.
- Son TTS neuronal: ~1–2 s por narrativa frente a ~12 s de Gemini.
- El estudio con usuarios y la evaluación usan siempre la voz por defecto (Gemini 3.1 Flash TTS, Sulafat).
- En Azure van en App Service visionnav-api → Variables de entorno → Configuración de la aplicación.
- La clave se obtiene en el recurso Speech → Keys and Endpoint (o con
  `az cognitiveservices account keys list -g rg-visionnav -n visionnavSpeech`).
- Nivel gratuito F0: 500.000 caracteres al mes (~2.000 narrativas).

No se debe publicar ni versionar el archivo .env.

4. Ejecución local

Crear un entorno virtual e instalar las dependencias:

Windows PowerShell

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
pip install -r requirements.txt

Linux/macOS

python3.13 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
pip install -r requirements.txt

Iniciar la API:

python run.py

Por defecto estará disponible en:

http://127.0.0.1:8000

Comprobar que el servicio está funcionando:

curl http://127.0.0.1:8000/api/health

5. Primera prueba

Enviar una imagen al endpoint de detección:

curl -F "file=@test_images/05_sala_muebles.jpg" \
     http://127.0.0.1:8000/api/detect

El endpoint devuelve la narrativa y la información generada por el pipeline.

Para solicitar audio:

curl -X POST \
     -F "file=@test_images/05_sala_muebles.jpg" \
     -F "audio=true" \
     http://127.0.0.1:8000/api/detect \
     --output respuesta.mp3

6. Flujo de procesamiento

La API procesa cada imagen mediante las siguientes etapas:

Imagen
  ↓
Validación
  ↓
Preprocesamiento
  ↓
Detección de objetos con YOLO26s
  ↓
Filtrado de objetos relevantes
  ↓
Análisis espacial egocéntrico
  ↓
Estimación de distancia en pasos
  ↓
Análisis de espacio libre y movimiento
  ↓
Generación de narrativa con LLM
  ↓
Conversión de texto a audio con TTS
  ↓
Respuesta de la API

Los parámetros experimentales se encuentran registrados en experimental_config.yaml.

7. API principal

POST /api/detect

Recibe una imagen mediante multipart/form-data.

Campo

Tipo

Descripción

file

archivo

Imagen JPEG o PNG. Es obligatorio.

confidence_threshold

número

Umbral de confianza de la solicitud.

audio

booleano

Solicita la generación de audio.

debug

booleano

Incluye información adicional del procesamiento.

tts_model

texto

Permite seleccionar un modelo TTS disponible en los perfiles que lo soportan.

Ejemplo:

curl -X POST \
     -F "file=@test_images/05_sala_muebles.jpg" \
     -F "confidence_threshold=0.35" \
     http://127.0.0.1:8000/api/detect

GET /api/health

Comprueba el estado de la API y de los componentes principales.

curl http://127.0.0.1:8000/api/health

La documentación interactiva de OpenAPI está disponible durante el desarrollo en:

http://127.0.0.1:8000/docs

8. Perfiles de ejecución

La variable APP_PROFILE determina las funciones disponibles.

Perfil

Uso

development

Desarrollo y pruebas locales. Incluye documentación y herramientas de evaluación.

study

Sesiones de evaluación con participantes. Incluye los recursos necesarios para el estudio.

production

Ejecución desplegada. Expone únicamente los endpoints necesarios para el servicio.

Para ejecutar otro perfil de forma local:

APP_PROFILE=study python run.py

En PowerShell:

$env:APP_PROFILE="study"
python run.py

9. Ejecución con Docker

Docker permite ejecutar la configuración destinada a producción.

Construir la imagen:

docker build --build-arg APP_COMMIT=$(git rev-parse HEAD) -t visionnav-api .

En PowerShell:

docker build --build-arg APP_COMMIT=$(git rev-parse HEAD) -t visionnav-api .

Ejecutar el contenedor:

docker run -p 8000:8000 \
  -e GROQ_API_KEY="TU_CLAVE_GROQ" \
  -e GOOGLE_API_KEY="TU_CLAVE_GOOGLE" \
  visionnav-api

Después comprobar:

curl http://127.0.0.1:8000/api/health

La imagen contiene los pesos del modelo y verifica su integridad antes de iniciar.

No utilizar el .env de desarrollo directamente con --env-file, porque puede contener rutas que no existen dentro del contenedor.

10. Pruebas

La suite de pruebas se ejecuta con:

python -m pytest

Las pruebas cubren, entre otros aspectos:

contrato de errores;

validación y límites de entrada;

limpieza de archivos temporales;

control de solicitudes;

request_id;

estado de salud;

integración del pipeline con proveedores simulados;

comportamiento de la imagen Docker.

Las pruebas automatizadas no sustituyen la evaluación experimental del sistema ni las pruebas con usuarios.

La documentación detallada se encuentra en:

docs/PRUEBAS.md

11. Evaluación experimental

La configuración experimental se encuentra en:

experimental_config.yaml

El proyecto incluye mecanismos para comprobar la correspondencia entre el código, los pesos, las versiones, los parámetros y los estímulos utilizados en la evaluación.

La evaluación formal F4 todavía se encuentra pendiente. Por esta razón, los resultados de las pruebas de software no deben interpretarse como resultados finales de la evaluación con usuarios.

La documentación del protocolo se encuentra en:

docs/CP3B_PROTOCOLO_EVALUACION.md
docs/CP3B_MATRIZ_DECISIONES.md

12. Reproducibilidad

Los principales elementos del entorno se encuentran versionados o registrados mediante archivos de configuración y dependencias fijadas.

Archivos principales:

experimental_config.yaml
requirements.txt
requirements-test.txt
requirements-dev.txt
requirements.lock.txt
requirements-docker.lock.txt

La guía completa se encuentra en:

docs/REPRODUCIBILIDAD.md

13. Despliegue

El backend está preparado para ejecutarse mediante Docker y desplegarse en Azure App Service.

El despliegue de Azure debe validarse con la configuración correspondiente a la versión experimental actual. La configuración anterior de Azure no debe utilizarse como evidencia de los resultados actuales.

El cliente web se mantiene en un repositorio independiente y puede desplegarse en Vercel. La URL del backend se configura mediante NEXT_PUBLIC_API_URL en el cliente.

La información operativa del despliegue se encuentra en:

docs/DEPLOYMENT.md

14. Estructura principal

app/
  core/           Pipeline principal
  services/       Detección, análisis espacial, pasos, narrativa y TTS
  routes/         Endpoints de la API
  catalog/        Catálogo de estímulos
  utils/          Utilidades y clientes de servicios externos
  main.py         Configuración de la aplicación

stimuli/           Estímulos del estudio
test_images/       Imágenes para pruebas
tests/             Pruebas automatizadas
scripts/           Scripts de experimentación y pruebas
evaluation/        Resultados y evidencias de evaluación
docs/              Documentación técnica

Dockerfile
run.py
experimental_config.yaml
requirements*.txt

15. Documentación

Archivo

Contenido

docs/CONTRATO_ERRORES.md

Contrato de errores y degradaciones

docs/POLITICA_API_DETECT.md

Políticas del endpoint de detección

docs/DATOS_PERSISTENCIA.md

Datos almacenados y persistencia

docs/REPRODUCIBILIDAD.md

Configuración y reproducibilidad

docs/PRUEBAS.md

Pruebas del sistema

docs/DEPLOYMENT.md

Despliegue

docs/CP3B_PROTOCOLO_EVALUACION.md

Protocolo de evaluación

docs/CP3B_MATRIZ_DECISIONES.md

Decisiones de evaluación
