@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Instalador de VisionNav API

echo.
echo  ==============================================
echo    VisionNav API - Instalacion
echo  ==============================================
echo.

rem -- 1. Python 3.13 o 3.12 ----------------------------------------------
set "PY="
py -3.13 --version >nul 2>&1 && set "PY=py -3.13"
if not defined PY py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
    python --version 2>nul | findstr /b /c:"Python 3.13" /c:"Python 3.12" >nul && set "PY=python"
)
if not defined PY (
    echo  [X] No se encontro Python 3.13 ni 3.12.
    echo      Descarguelo de https://www.python.org/downloads/ y, al instalarlo,
    echo      marque la casilla "Add python.exe to PATH". Luego ejecute este archivo de nuevo.
    echo.
    pause
    exit /b 1
)
echo  [1/5] Python encontrado: %PY%

rem -- 2. Entorno virtual --------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo  [2/5] Creando el entorno virtual...
    %PY% -m venv .venv || goto :error
) else (
    echo  [2/5] El entorno virtual ya existe.
)

rem -- 3. Dependencias -----------------------------------------------------
echo  [3/5] Instalando dependencias. Puede tardar varios minutos...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul || goto :error
".venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0 || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

rem -- 4. Pesos del modelo -------------------------------------------------
set "PESOS_SHA=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
call :verificar_pesos
if errorlevel 1 (
    if exist "yolo26s.pt" del "yolo26s.pt"
    echo  [4/5] Descargando el modelo de deteccion yolo26s.pt...
    curl -L --fail -o yolo26s.pt.part https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt || goto :error
    move /y "yolo26s.pt.part" "yolo26s.pt" >nul
    call :verificar_pesos
    if errorlevel 1 (
        echo  [X] El modelo descargado no coincide con el esperado.
        goto :error
    )
    echo        Modelo descargado y verificado.
) else (
    echo  [4/5] El modelo yolo26s.pt ya existe y es correcto.
)

rem -- 5. Configuracion ----------------------------------------------------
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo  [5/5] Configuracion de las claves.
    echo        Puede dejarlas en blanco y escribirlas despues en el archivo .env
    echo.
    set /p "GROQ=        Clave de Groq (descripcion): "
    set /p "AZURE=        Clave de Azure AI Speech (voz): "
    set "REGION=canadacentral"
    set /p "REGION=        Region de Azure AI Speech [canadacentral]: "
    call :guardar_claves
) else (
    echo  [5/5] El archivo .env ya existe; se conserva.
)

echo.
echo  ==============================================
echo    Instalacion terminada.
echo    Para iniciar la API, abra iniciar.bat
echo  ==============================================
echo.
pause
exit /b 0

:verificar_pesos
if not exist "yolo26s.pt" exit /b 1
set "SHA="
for /f %%h in ('powershell -NoProfile -Command "(Get-FileHash yolo26s.pt -Algorithm SHA256).Hash.ToLower()"') do set "SHA=%%h"
if /i "%SHA%"=="%PESOS_SHA%" exit /b 0
exit /b 1

:guardar_claves
powershell -NoProfile -Command ^
  "$f='.env'; $t=Get-Content $f -Raw;" ^
  "$t=$t -replace '(?m)^GROQ_API_KEY=.*$', ('GROQ_API_KEY=' + $env:GROQ);" ^
  "$t=$t -replace '(?m)^AZURE_SPEECH_KEY=.*$', ('AZURE_SPEECH_KEY=' + $env:AZURE);" ^
  "$t=$t -replace '(?m)^AZURE_SPEECH_REGION=.*$', ('AZURE_SPEECH_REGION=' + $env:REGION);" ^
  "Set-Content -Path $f -Value $t -NoNewline -Encoding utf8"
exit /b 0

:error
echo.
echo  [X] La instalacion se detuvo por un error. Revise el mensaje de arriba.
echo.
pause
exit /b 1
