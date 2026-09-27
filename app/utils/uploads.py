"""
app/utils/uploads.py

Lectura acotada de archivos subidos (UploadFile).

Evita que un archivo excesivamente grande se cargue completo en memoria:
se leen como máximo MAX_UPLOAD_BYTES + 1 bytes y, si se supera el límite,
se responde HTTP 413 sin procesar el contenido.

CONFIGURACIÓN (.env):
  MAX_UPLOAD_MB → tamaño máximo por archivo en MB (default: 10).
                  Las imágenes del conjunto de evaluación Web3D pesan < 5 MB.
"""

import os

from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES: int = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)


async def read_upload_limited(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Lee el archivo completo o lanza HTTP 413 si supera max_bytes."""
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo supera el tamaño máximo permitido ({max_bytes // (1024 * 1024)} MB).",
        )
    return data
