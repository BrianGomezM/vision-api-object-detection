"""Mide una variante de YOLO26 sobre las 12 imágenes preparadas (800 px) de la campaña.
Uso: python medir.py <pesos.pt> <dir_imagenes_preparadas> <salida.jsonl>
Condiciones: CPU, 4 hilos, imgsz 1280, conf 0,15 (umbral interno de producción), 3 calentamientos, 3 repeticiones."""
import io, json, os, sys, time, hashlib, glob
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
from PIL import Image
from ultralytics import YOLO

pesos, carpeta, salida = sys.argv[1:4]
sha = hashlib.sha256(open(pesos, "rb").read()).hexdigest()
model = YOLO(pesos)
torch.set_num_threads(4)
imgs = sorted(glob.glob(os.path.join(carpeta, "*.bin")))

def det(path):
    im = Image.open(io.BytesIO(open(path, "rb").read())).convert("RGB")
    t0 = time.perf_counter()
    r = model.predict(source=im, conf=0.15, imgsz=1280, device="cpu", augment=False, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    torch.set_num_threads(4)
    d = [{"label": r.names[int(b.cls[0])], "confidence": round(float(b.conf[0]), 3),
          "bbox": [round(v, 1) for v in b.xyxy[0].tolist()]} for b in r.boxes]
    return ms, d

for _ in range(3):
    det(imgs[0])
with open(salida, "w", encoding="utf-8") as f:
    for rep in (1, 2, 3):
        for p in imgs:
            ms, d = det(p)
            f.write(json.dumps({"pesos": os.path.basename(pesos), "sha256": sha, "hilos": torch.get_num_threads(),
                                "imagen": os.path.basename(p), "repeticion": rep, "det_ms": round(ms, 1),
                                "detecciones": d}, ensure_ascii=False) + "\n")
print("listo", pesos, len(imgs), "imágenes")
