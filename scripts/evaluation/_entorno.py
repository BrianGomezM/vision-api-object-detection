"""
scripts/evaluation/_entorno.py

Registro del entorno de medición para las campañas de evaluación (campaña definitiva
del Capítulo 3 y pruebas no funcionales R2). Solo lee: equipo, sistema, versiones,
hilos, plan de energía, commit y huellas SHA-256. No cambia nada del sistema.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PAQUETES = ("torch", "torchvision", "ultralytics", "tensorflow", "tensorflow-intel", "tensorflow-hub",
            "tf-keras", "numpy", "pillow", "psutil", "fastapi", "uvicorn", "httpx", "groq")
VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
             "CUDA_VISIBLE_DEVICES", "YOLO_WEIGHTS", "YOLO_IMGSZ", "YOLO_IOU", "MAX_IMAGE_DIM",
             "RATE_LIMIT_REQUESTS", "APP_PROFILE", "TTS_DEFAULT_MODEL", "GROQ_MODEL")


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def sha256_directorio(ruta: Path) -> str:
    """Huella de un directorio: SHA-256 de la lista ordenada (ruta relativa, huella de cada archivo)."""
    h = hashlib.sha256()
    for p in sorted(x for x in Path(ruta).rglob("*") if x.is_file()):
        h.update(f"{p.relative_to(ruta).as_posix()}:{sha256_archivo(p)}\n".encode())
    return h.hexdigest()


def _cpu_nombre() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def _salida(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git() -> dict:
    return {"commit": _salida(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
            "rama": _salida(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
            "cambios_sin_commit": _salida(["git", "-C", str(REPO), "status", "--porcelain"]) or ""}


def ahora_utc() -> str:
    """Fecha y hora UTC con milisegundos, para cada registro."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def filtrar_porcelain(porcelain: str, excluir: list[str]) -> list[str]:
    """
    Líneas de `git status --porcelain` que NO corresponden a rutas excluidas (prefijos relativos al
    repositorio, con barras /). Se usa para exigir que el código y los datos de entrada estén en un
    commit, ignorando solo la carpeta de resultados que la propia ejecución está escribiendo.
    """
    pref = [e.replace("\\", "/").rstrip("/") + "/" for e in excluir]
    quedan = []
    for linea in (porcelain or "").splitlines():
        ruta = linea[3:].strip().strip('"').replace("\\", "/")
        if " -> " in ruta:
            ruta = ruta.split(" -> ", 1)[1]
        if not any(ruta == p.rstrip("/") or ruta.startswith(p) for p in pref):
            quedan.append(linea)
    return quedan


def cambios_no_registrados(excluir: list[str], repo: Path = REPO) -> list[str]:
    salida = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    return filtrar_porcelain(salida, excluir)


def estado_git_archivo(ruta: Path, repo: Path = REPO) -> dict:
    """
    ¿Está el archivo versionado en un commit y sin cambios? Compara el identificador de objeto de Git
    del archivo actual (`git hash-object`, que aplica la misma conversión de fines de línea que Git)
    con el del archivo en el último commit que lo modificó (`git rev-parse <commit>:<ruta>`).
    """
    rel = Path(ruta).resolve().relative_to(Path(repo).resolve()).as_posix()
    rastreado = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                               capture_output=True).returncode == 0
    commit = _salida(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", rel]) if rastreado else None
    objeto_commit = _salida(["git", "-C", str(repo), "rev-parse", f"{commit}:{rel}"]) if commit else None
    objeto_actual = _salida(["git", "-C", str(repo), "hash-object", rel])
    return {"ruta": rel, "rastreado": rastreado, "commit": commit, "objeto_git_commit": objeto_commit,
            "objeto_git_actual": objeto_actual,
            "limpio": bool(commit) and objeto_commit is not None and objeto_commit == objeto_actual,
            "sha256_archivo": sha256_archivo(Path(repo) / rel)}


def entorno(archivos_codigo: list[Path] = ()) -> dict:
    import psutil
    bateria = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    versiones = {}
    for p in PAQUETES:
        try:
            versiones[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": _cpu_nombre(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "sistema": platform.platform(),
        "python": sys.version.split()[0],
        "ejecutable": sys.executable,
        "versiones": versiones,
        "paquetes_instalados": sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()
                                      if d.metadata.get("Name")),
        "variables": {v: os.environ.get(v) for v in VARIABLES},
        "corriente_conectada": None if bateria is None else bateria.power_plugged,
        "plan_energia": _salida(["powercfg", "/getactivescheme"]) if sys.platform == "win32" else None,
        "git": git(),
        "huellas_codigo": {Path(a).as_posix(): sha256_archivo(Path(a)) for a in archivos_codigo if Path(a).exists()},
    }
