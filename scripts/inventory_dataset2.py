"""
scripts/inventory_dataset2.py

Inventario COMPLETO de evaluation/images (29 archivos) para el Dataset 2
(conjunto de robustez / variabilidad visual). Escribe
evaluation/metadata/inventario_dataset2_verificacion.csv con hechos medidos
(sha256, dimensiones, bytes) y el estado de verificación de procedencia y
licencia registrado el 2026-09-26.

No copia imágenes, no ejecuta YOLO y no usa resultados previos de detección.
"""

import csv
import hashlib
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "evaluation" / "images"
OUT = REPO / "evaluation" / "metadata" / "inventario_dataset2_verificacion.csv"

SKF = ("Sketchfab; URL del modelo no registrada", "NO VERIFICABLE: la licencia depende de cada modelo y no hay URL",
       "EXCLUIDA hasta registrar la URL y la licencia del modelo")
G = ("Generación sintética; herramienta, términos y prompt no documentados", "NO VERIFICADA: faltan la herramienta y sus términos de uso",
     "PENDIENTE: el autor debe documentar la herramienta, los términos y el prompt")

# (procedencia registrada, resultado de la verificación 2026-09-26, estado)
STATUS = {
    "W3D-G-01-granja": G, "W3D-G-02-parque": G, "W3D-G-03-calle": G,
    "W3D-G-04-Biblioteca": (G[0] + "; no estaba en inventario_imagenes.csv", G[1],
                            "CANDIDATA PROPUESTA; PENDIENTE (igual que G-*)"),
    "W3D-G-05-Ciudad": (G[0] + "; no estaba en inventario_imagenes.csv", G[1], G[2]),
    "W3D-G-06-Universidad": (G[0] + "; no estaba en inventario_imagenes.csv", G[1], G[2]),
    "W3D-I-JC3D-01-cars": (
        "JustCreate3D; estilo compatible con Low Poly Megapolis Pack",
        "PARCIAL: la licencia del pack (itch.io) permite 'creating media content' y prohíbe redistribuir "
        "los assets originales. No consta si la imagen es un render propio o una captura promocional de la tienda",
        "CANDIDATA PROPUESTA; PENDIENTE de que el autor confirme cómo obtuvo la imagen"),
    "W3D-I-JC3D-02-megapolis": (
        "JustCreate3D; Low Poly Megapolis Pack",
        "PARCIAL: igual que JC3D-01", "NO PROPUESTA: aporta menos que JC3D-01"),
    "W3D-I-JC3D-03-salad": (
        "Registrada como JustCreate3D",
        "NO VERIFICABLE: es un interior realista (recepción) y el catálogo de JustCreate3D en itch.io solo "
        "tiene packs low-poly o estilizados. Origen real desconocido",
        "EXCLUIDA (era candidata propuesta)"),
    "W3D-I-JC3D-04-school": (
        "Registrada como JustCreate3D",
        "NO VERIFICABLE: interior realista, igual que JC3D-03",
        "EXCLUIDA: además tiene artefactos de interfaz (FPS y barra de tareas)"),
    "W3D-I-KYK-01-restaurant": (
        "KayKit Restaurant Bits", "CC0 según la página del pack",
        "EXCLUIDA: imagen promocional isométrica, no una escena navegable"),
    "W3D-I-PHV-01-bathroom": (
        "Registrada como Poly Haven",
        "NO VERIFICABLE: no coincide con ningún asset de baño de la API de Poly Haven (bathroom, "
        "modern_bathroom, en_suite, creepy_bathroom son HDRI 360° distintos). Parece un render "
        "arquitectónico de otro origen",
        "EXCLUIDA (era candidata propuesta)"),
    "W3D-I-PHV-02-studio": (
        "Registrada como Poly Haven; página no registrada",
        "NO VERIFICADA: sin página exacta", "EXCLUIDA: foto HDRI con muy pocos objetos"),
    "W3D-I-PHV-03-studio2": (
        "Registrada como Poly Haven; página no registrada",
        "NO VERIFICADA: sin página exacta", "EXCLUIDA: formato vertical y muy pocos objetos"),
    "W3D-I-PHV-04-sofa": (
        "Registrada como Poly Haven; página no registrada",
        "NO VERIFICADA: sin página exacta", "EXCLUIDA: franja negra de captura y duplica salas de estar"),
}

FIELDS = ["archivo", "ruta", "bytes", "ancho", "alto", "modo", "sha256",
          "procedencia_registrada", "verificacion_2026_09_26", "estado_dataset2"]


def main() -> None:
    files = sorted(p for p in IMAGES.rglob("*") if p.is_file())
    rows = []
    for p in files:
        with Image.open(p) as im:
            w, h, mode = im.width, im.height, im.mode
        stem = p.stem
        if stem.startswith("W3D-I-SKF-"):
            status = SKF
        else:
            status = STATUS[stem]
        rows.append({
            "archivo": p.name, "ruta": p.relative_to(REPO).as_posix(), "bytes": p.stat().st_size,
            "ancho": w, "alto": h, "modo": mode,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "procedencia_registrada": status[0], "verificacion_2026_09_26": status[1],
            "estado_dataset2": status[2],
        })
    hashes = [r["sha256"] for r in rows]
    assert len(set(hashes)) == len(hashes), "hay imágenes duplicadas"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} imágenes inventariadas → {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
