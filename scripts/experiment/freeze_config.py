"""
scripts/experiment/freeze_config.py

Genera experimental_config.yaml: la configuración OFICIAL de F4.

  - La sección `verificado` sale de app.experiment.runtime_snapshot(), es decir,
    de los valores EFECTIVOS en ejecución (no de la documentación). preflight()
    exige que la ejecución coincida exactamente con ella.
  - Las secciones descriptivas documentan reglas y decisiones; lo que sigue
    abierto se marca como PENDIENTE (no se inventan valores).

Uso (desde la raíz, con el .env del proyecto):
    python scripts/experiment/freeze_config.py            # escribe experimental_config.yaml
    python scripts/experiment/freeze_config.py --check    # solo compara, no escribe
No ejecuta YOLO, LLM ni TTS (lee el checkpoint y aplica el letterbox sin inferencia).
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from app import experiment  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG  # noqa: E402

OUT = REPO / "experimental_config.yaml"


def _checkpoint_facts(weights: Path) -> dict:
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    m = ck["model"]
    return {"stride": [int(s) for s in m.stride.tolist()], "clases": len(m.names),
            "entrenado_imgsz": ck.get("train_args", {}).get("imgsz"),
            "ultralytics_de_guardado": ck.get("version"), "fecha_checkpoint": ck.get("date")}


def _letterbox_shape(img_path: Path, imgsz: int, stride: int) -> list:
    from ultralytics.data.augment import LetterBox
    im = np.asarray(Image.open(img_path).convert("RGB"))[..., ::-1]
    return list(LetterBox(imgsz, auto=True, stride=stride)(image=im).shape[:2])


def build() -> dict:
    snap = experiment.runtime_snapshot()
    det = snap["deteccion"]
    weights_path = REPO / det["weights"]
    ckf = _checkpoint_facts(weights_path)
    lb = _letterbox_shape(REPO / "stimuli/dataset1/A1.png", det["imgsz"], max(ckf["stride"]))
    manifest = REPO / "stimuli/dataset1/manifest.yaml"
    catalog = REPO / "app/catalog/catalog.yaml"
    commit = experiment.app_commit()

    return {
        "schema_version": 1,
        "descripcion": "Configuración OFICIAL de F4 (congelada). preflight() exige coincidencia exacta de `verificado`.",
        "generado_desde_commit": commit["commit"],
        "estado_decisiones": {
            "regla_umbral_por_clase": "APROBADA (min) — docs/AUDITORIA_REGLA_UMBRAL_FASE10.md §5",
            "preprocesamiento": "APROBADO (ruta de producción: core.pipeline.resize_image)",
            "modelo_y_pesos": "APROBADO (yolo26s.pt, hash verificado)",
            "dispositivo": "APROBADO (cpu) — CP3B D3, evidencia E11/E12",
            "umbral_confianza_endpoint": "APROBADO (0.35, valor por defecto desde el 19-05-2026) — CP3B D4",
            "numero_de_corridas": "PENDIENTE — CP3B D5",
            "caja_referencia_iou": "PENDIENTE — CP3B D1",
            "metricas": "PENDIENTE — CP3B D2, D7-D9, D12",
            "narrativa_generaciones": "PENDIENTE — CP3B D10",
            "estimulo_c2_espejo": "PENDIENTE — CP3B D6 (no generado)",
            "dataset2": "PENDIENTE — CP3B D13",
        },
        "verificado": snap,
        "ejecucion": {
            "punto_de_entrada": "app.core.pipeline.run (la misma función que /api/detect)",
            "device": "cpu",
            "como_se_fuerza_cpu": "CUDA_VISIBLE_DEVICES=-1 antes de importar torch (run_yolo no pasa device)",
            "variables_entorno": {
                "CUDA_VISIBLE_DEVICES": "-1",
                "YOLO_WEIGHTS_SHA256": snap["pesos"]["sha256"],
                "YOLO_ALLOW_DOWNLOAD": "false",
            },
            "data_root": "obligatorio; artefactos en DATA_ROOT/evaluation/{runs,stimuli_frozen}",
            "estado_por_estimulo": "app.experiment.reset_request_state() antes de cada estímulo (caché de escenario de 10 s)",
            "corridas": "PENDIENTE (CP3B D5)",
            "trazabilidad": "app.experiment.build_manifest(): request_id, stimulus_id, commit, config_sha256, pesos, "
                            "versiones, device, hilos, hashes de entrada/salida",
        },
        "imagen": {
            "entrada": "bytes del estímulo tal cual (PNG del Dataset 1)",
            "dataset1": {"resolucion": [800, 450], "formato": "PNG", "modo": "RGB", "exif": "ninguno"},
            "paso_1_resize_image": {
                "decodificacion": "PIL Image.open(...).convert('RGB') — descarta alfa; orientación EXIF NO se aplica",
                "regla": f"si max(ancho, alto) > {snap['imagen']['max_dim']}: escalar a {snap['imagen']['max_dim']} px "
                         "en el lado mayor (int(dim*r)), filtro LANCZOS, re-codificar JPEG calidad 90 "
                         "(submuestreo 4:2:0 por defecto de Pillow); si no, se pasan los bytes ORIGINALES",
                "dataset1": "800 px → sin redimensionar ni re-codificar (bytes idénticos, verificado)",
            },
            "paso_2_run_yolo": "PIL Image.open(bytes).convert('RGB') → model.predict(source=PIL)",
            "paso_3_ultralytics": {
                "conversion": "PIL RGB → numpy BGR (LoadPilAndNumpy) → RGB, /255, float32, BCHW",
                "letterbox": {"imgsz": det["imgsz"], "rect": True, "auto": True, "stride": max(ckf["stride"]),
                              "interpolacion": "cv2.INTER_LINEAR", "relleno": 114, "centrado": True, "scaleup": True},
                "tensor_final_dataset1_hw": lb,
            },
            "fase_10": "usó la imagen ORIGINAL sin resize_image (diferencia documentada; no es la configuración de F4)",
        },
        "detector": {
            "modelo": "YOLO26s (Ultralytics)", "pesos": snap["pesos"], "checkpoint": ckf,
            "regla_umbral": "effective = min(class_min_conf.get(clase, umbral), umbral); solo clases en nav_classes; "
                            "piso práctico = conf_interna",
            "umbral_endpoint": det["conf_endpoint_default"],
            "umbrales_efectivos_dataset1": {
                c: round(max(det["conf_interna"], min(det["class_min_conf"].get(c, det["conf_endpoint_default"]),
                                                      det["conf_endpoint_default"])), 2)
                for c in ("chair", "dining table", "couch", "potted plant", "person", "bottle", "laptop")},
            "nms": {"iou": det["iou_nms"], "max_det": DEFAULT_CFG.max_det, "agnostic": DEFAULT_CFG.agnostic_nms,
                    "fuente": "valores por defecto de ultralytics (run_yolo no los fija)"},
            "augment": det["augment"],
            "descarga_automatica": "prohibida en F4 (YOLO_ALLOW_DOWNLOAD=false + YOLO_WEIGHTS_SHA256)",
        },
        "espacial": {
            "horizontal": "centro x de la caja / ancho: < 1/3 izquierda, > 2/3 derecha, resto centro",
            "profundidad": "4 categorías (muy_cerca, cerca, medio, lejos) por área relativa, y2 o yc de la caja; "
                           "umbrales en verificado.espacial.profundidad",
            "relaciones": "solo 'sobre superficie' (_merge_surfaces): centro del objeto dentro del ancho de la superficie "
                          "y entre y1-60 px e y2+40 px (píxeles absolutos de la imagen procesada)",
            "deduplicacion": "fusiona detecciones con igual (clase, columna, profundidad); conserva la de mayor confianza",
            "pasos_y_espacio_libre": "parámetros en verificado.espacial",
        },
        "narrativa": {
            "prompt_version": "definida por el código (verificado.codigo_nucleo_sha256: llm_enhancer.py, scene_classifier.py)",
            "reglas_deterministas": "build_narrative: intro de escenario solo si confianza media/alta + descripción + instrucción",
            "llm": "ver verificado.narrativa (modelo, temperaturas, tokens); no determinista",
        },
        "tts": {
            "idioma": "es (instrucción de estilo en español)",
            "salida": "PCM 24 kHz, 16 bit, mono → MP3 lameenc 64 kbps, calidad 2",
            "alternativas": "ninguna en F4: solo el modelo por defecto (el parámetro tts_model no se usa)",
        },
        "experimento": {
            "protocolo": "CP3B v2 — docs/CP3B_MATRIZ_DECISIONES.md (decisiones abiertas: ver estado_decisiones)",
            "dataset1": {"manifest": "stimuli/dataset1/manifest.yaml",
                         "manifest_sha256": experiment.sha256_text_file(manifest),
                         "generador_commit": "ae85f90", "escenas": 18},
            "c2_espejo": "PENDIENTE (CP3B D6)",
            "catalogo": {"archivo": "app/catalog/catalog.yaml", "sha256": experiment.sha256_text_file(catalog)},
            "metricas_version": "PENDIENTE (CP3B)",
            "latencia": "NO forma parte de las métricas del CP3B; si se reporta, por separado (CPU local ≠ despliegue)",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg = build()
    header = ("# CONFIGURACIÓN EXPERIMENTAL OFICIAL DE F4 — GENERADA por scripts/experiment/freeze_config.py.\n"
              "# No editar a mano: regenerar y revisar el diff. `verificado` = valores efectivos en ejecución.\n")
    text = header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=110)
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        same = yaml.safe_load(current or "{}").get("verificado") == cfg["verificado"]
        print("verificado coincide" if same else "verificado DIFIERE")
        sys.exit(0 if same else 1)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"escrito {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
