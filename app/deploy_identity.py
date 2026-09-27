"""
app/deploy_identity.py — identidad del despliegue frente a la configuración congelada.

En el perfil production la aplicación se NIEGA A ARRANCAR si lo desplegado difiere
de `experimental_config.yaml` (sección `verificado`): pesos (SHA-256), código del
núcleo, parámetros de detección / espaciales / narrativa / TTS y versiones de las
librerías. Así lo evaluado en la tesis es exactamente lo que se despliega: una
variable de entorno olvidada (p. ej. GROQ_MODEL, cuyo valor por defecto en el
código NO es el congelado) no degrada el sistema en silencio.

Diferencias TOLERADAS (explícitas; se publican en /api/health):
  - ruta de los pesos: solo se compara el nombre del archivo (la identidad es el
    SHA-256, que sí se exige);
  - sufijo de compilación de torch / torchvision (+cu126 en el entorno local de
    referencia ↔ +cpu en la imagen): misma versión; el experimento se ejecuta en
    CPU (dispositivo congelado). Medido en Windows: +cpu y +cu126 dan cajas
    idénticas bit a bit (41/41). Linux difiere en la confianza ≤ 6,4·10⁻⁵
    (efecto de la plataforma, no de la compilación de torch;
    evaluation/results/hardening/torch/).
Cualquier otra diferencia → RuntimeError (el worker no arranca).

IDENTIDAD_EXPERIMENTAL=omitir desactiva la verificación (despliegues que NO son la
versión de la tesis); queda declarado como "omitida" en /api/health.
"""

import os
from pathlib import Path

from app import experiment

_TOLERATED_BUILD_SUFFIX = ("versiones.torch", "versiones.torchvision")
_state: dict = {"estado": "no_verificada"}


def _base_version(v):
    return v.split("+", 1)[0] if isinstance(v, str) else v


def compare(config: dict, snapshot: dict) -> tuple[list[str], list[str]]:
    """(diferencias no permitidas, diferencias toleradas) entre `verificado` y el entorno."""
    expected = config["verificado"]
    blocking, tolerated = [], []
    for d in experiment._diff(expected, snapshot):
        key = d.split(":", 1)[0]
        if key in _TOLERATED_BUILD_SUFFIX:
            name = key.split(".", 1)[1]
            if _base_version(expected["versiones"][name]) == _base_version(snapshot["versiones"].get(name)):
                tolerated.append(d)
                continue
        if key == "deteccion.weights":
            if Path(str(expected["deteccion"]["weights"])).name == Path(str(snapshot["deteccion"]["weights"])).name:
                tolerated.append(d)
                continue
        blocking.append(d)
    return blocking, tolerated


def verify(config_path: Path = experiment.CONFIG_PATH) -> dict:
    """Verifica y memoriza el resultado. Lanza RuntimeError ante cualquier diferencia."""
    global _state
    if os.getenv("IDENTIDAD_EXPERIMENTAL", "").strip().lower() == "omitir":
        _state = {"estado": "omitida"}
        print("[Identidad] ADVERTENCIA: verificación omitida (IDENTIDAD_EXPERIMENTAL=omitir).")
        return _state
    if not Path(config_path).exists():
        raise RuntimeError(f"[Identidad] No existe {Path(config_path).name}: no se puede verificar el despliegue.")
    config = experiment.load_config(config_path)
    blocking, tolerated = compare(config, experiment.runtime_snapshot())
    if blocking:
        raise RuntimeError("[Identidad] El despliegue NO coincide con experimental_config.yaml:\n  - "
                           + "\n  - ".join(blocking))
    _state = {"estado": "verificada", "config_sha256": experiment.sha256_text_file(Path(config_path)),
              "pesos_sha256": config["verificado"]["pesos"]["sha256"], "diferencias_toleradas": tolerated}
    print(f"[Identidad] Despliegue verificado contra experimental_config.yaml "
          f"({_state['config_sha256'][:12]}…; toleradas: {len(tolerated)}).")
    return _state


def status() -> dict:
    return {**_state, "commit": os.getenv("APP_COMMIT") or None}
