@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title VisionNav API

if not exist ".venv\Scripts\python.exe" (
    echo  Primero ejecute instalar.bat
    pause
    exit /b 1
)

echo.
echo  Iniciando VisionNav API en http://127.0.0.1:8000
echo  La documentacion se abrira en el navegador en unos segundos.
echo  Para detener la API, cierre esta ventana o presione Ctrl+C.
echo.

start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 20; Start-Process 'http://127.0.0.1:8000/docs'"
".venv\Scripts\python.exe" run.py
