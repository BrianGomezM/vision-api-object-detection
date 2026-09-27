"""
scripts/evaluation/phase2a_llm_v1_audit.py

FASE 2A — Auditoría del experimento C, ejecución v1 (formato antiguo, terminada
el 2026-09-24 00:19:43). INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN.

Solo LEE los archivos originales (llm/llm_runs.jsonl, llm_stability.jsonl,
run_meta.jsonl); verifica sus hashes antes y después y escribe los resultados
en llm/auditoria_v1/. No reconstruye información que la v1 no guardó.

Chequeos automáticos (aproximaciones por coincidencia de términos; se listan
los casos para revisión humana):
  - Dirección conservada: para cada línea del prompt "- objeto: posición",
    se busca la oración de la narrativa que menciona el objeto y se comparan
    los términos de dirección {izquierda, derecha, frente, fondo}.
  - Pasos conservados: el número de pasos de la línea aparece en esa oración.
  - Referencias fuera de la regla 5 del prompt.
  - Origen de problemas en las 5 imágenes con inventario (Fase 6): objeto de
    la entrada no confirmado por la referencia → posible error del detector.
"""

import csv
import hashlib
import itertools
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.utils.translator import _STATIC_DICT  # noqa: E402

SRC = ROOT / "evaluation" / "results" / "phase2a" / "llm"
OUT = SRC / "auditoria_v1"
ORIGINALS = ["llm_runs.jsonl", "llm_stability.jsonl", "run_meta.jsonl"]
DIRS = ["izquierda", "derecha", "frente", "fondo"]
ALLOWED = ["frente a ti", "a tu derecha", "a tu izquierda", "al fondo a tu derecha", "al fondo a tu izquierda", "al fondo"]
NON_RULE5 = ["justo frente", "justo a tu", "un poco más adelante", "cerca a tu", "cerca de ti", "detrás", "delante de ti"]
# Traducción inversa ES→EN que prefiere las etiquetas que el detector realmente emite
# (p. ej. "sofá" → "couch" de COCO, no "sofa", que COCO no tiene).
_DETECTED = {d["label"] for l in (ROOT / "evaluation/results/phase2a/detection/configurations.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip() for d in json.loads(l)["detecciones"]}
ES2EN = {}
for _k, _v in sorted(_STATIC_DICT.items(), key=lambda kv: kv[0] not in _DETECTED):
    ES2EN.setdefault(_v.lower(), _k)
GT = json.loads((ROOT / "evaluation/results/phase6/ground_truth.json").read_text(encoding="utf-8"))
GT_PRESENT = {"evaluation/images/web3d/" + e["image"]: {o["class_coco"] for o in e["relevant_objects_present"]
                                                        if o.get("present") and o.get("class_coco")}
              for e in GT["objetos_relevantes"]}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dirs_of(text):
    t = text.lower()
    return frozenset(d for d in DIRS if d in t)


def term_regex(term):
    first, *rest = term.lower().split(" ")
    return r"\b" + re.escape(first) + r"(e?s)?\b" + "".join(r"\s+(de\s+)?" + re.escape(t) for t in rest if t != "de")


def parse_prompt(prompt):
    items = []
    for line in prompt.splitlines():
        m = re.match(r"^- (.+?): (.+)$", line.strip())
        if not m:
            continue
        name, pos = m.group(1), m.group(2)
        base = re.sub(r"^\d+\s+", "", name).strip()
        base = re.sub(r"(es|s)$", "", base) if base not in ES2EN else base
        steps = re.search(r"(\d+)\s+pasos?", pos)
        items.append({"linea": line.strip(), "objeto": name, "objeto_base": base, "posicion": pos,
                      "dirs": dirs_of(pos), "pasos": int(steps.group(1)) if steps else None})
    return items


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def audit_narrative(text, prompt_items):
    sents = sentences(text)
    res = []
    for it in prompt_items:
        rx = term_regex(it["objeto_base"].split(" ")[0] if it["objeto_base"] not in ES2EN else it["objeto_base"])
        hits = [s for s in sents if re.search(rx, s.lower())]
        if not hits:
            res.append({**it, "estado": "omitido"})
            continue
        match_dir = [s for s in hits if dirs_of(s) == it["dirs"] or (it["dirs"] and it["dirs"] <= dirs_of(s) and len(dirs_of(s)) <= len(it["dirs"]) + 1)]
        if it["dirs"] and not match_dir:
            res.append({**it, "estado": "direccion_alterada", "oraciones": hits})
            continue
        cand = match_dir or hits
        steps_ok = it["pasos"] is None or any(re.search(rf"\b{it['pasos']}\s+pasos?", s.lower()) for s in cand)
        res.append({**it, "estado": "conservado" if steps_ok else "pasos_alterados", "oraciones": cand})
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    before = {f: sha(SRC / f) for f in ORIGINALS}
    runs = [json.loads(l) for l in (SRC / "llm_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((SRC / "run_meta.jsonl").read_text(encoding="utf-8").splitlines()[0])
    L = []

    # ── 1. Integridad ─────────────────────────────────────────
    models, imgs, reps = meta["modelos"], meta["imagenes"], list(range(1, meta["repeticiones"] + 1))
    keys = Counter((r["modelo"], r["imagen"], r["repeticion"]) for r in runs)
    expected = set(itertools.product(models, imgs, reps))
    L.append("## 1. Integridad")
    L.append(f"registros={len(runs)} | combinaciones esperadas={len(expected)} | presentes={len(set(keys))} | "
             f"faltantes={len(expected - set(keys))} | inesperadas={len(set(keys) - expected)} | duplicadas={sum(v > 1 for v in keys.values())}")
    L.append(f"llamadas (2 tareas por combinación)={2 * len(runs)} | modelos={len({r['modelo'] for r in runs})} | "
             f"imágenes={len({r['imagen'] for r in runs})} | repeticiones={sorted({r['repeticion'] for r in runs})}")
    L.append(f"campos por registro presentes en todos: {all(set(r) == set(runs[0]) for r in runs)} | "
             f"la entrada (prompt) de cada imagen es idéntica entre modelos y repeticiones: "
             f"{all(len({r['prompt'] for r in runs if r['imagen'] == i}) == 1 for i in imgs)}")

    # ── 2 y 3. Resultados y errores por modelo ────────────────
    rows_exec, rows_err = [], []
    L.append("\n## 2. Resultados por modelo")
    for m in models:
        rs = [r for r in runs if r["modelo"] == m]
        empty = [r for r in rs if not r["narrativa"].strip()]
        nerr = [r for r in rs if r["narrativa_error_llm"]]
        serr = [r for r in rs if r["escenario_error_llm"]]
        nok = [r for r in rs if r["narrativa"].strip() and not r["narrativa_error_llm"]]
        sok = [r for r in rs if not r["escenario_error_llm"]]
        latn, lats = [r["latencia_narrativa_ms"] for r in rs], [r["latencia_escenario_ms"] for r in rs]
        L.append(f"{m}: narrativa con texto={len(nok)}/24 · vacía sin error={len(empty)}/24 · error declarado={len(nerr)}/24 · "
                 f"escenario del LLM={len(sok)}/24 · escenario con error (se usó heurística)={len(serr)}/24 "
                 f"({round(100 * len(serr) / 24, 1)} %) · latencia narrativa mediana={statistics.median(latn):.0f} ms "
                 f"[{min(latn):.0f}–{max(latn):.0f}] · latencia escenario mediana={statistics.median(lats):.0f} ms [{min(lats):.0f}–{max(lats):.0f}]")
        for r in serr:
            e = r["escenario_error_llm"]
            kind = ("respuesta_vacia (JSON vacío)" if "char 0" in e and "line 1 column 1" in e else
                    "json_truncado (cadena sin cerrar)" if "Unterminated string" in e else
                    "json_malformado")
            rows_err.append({"modelo": m, "imagen": r["imagen"], "repeticion": r["repeticion"], "tarea": "escenario",
                             "tipo": kind, "mensaje": e, "latencia_ms": r["latencia_escenario_ms"],
                             "escenario_heuristico_usado": r["escenario"]})
        for r in empty:
            rows_err.append({"modelo": m, "imagen": r["imagen"], "repeticion": r["repeticion"], "tarea": "narrativa",
                             "tipo": "respuesta_vacia_sin_error (el producto la acepta como éxito)", "mensaje": "",
                             "latencia_ms": r["latencia_narrativa_ms"], "escenario_heuristico_usado": None})

    L.append("\n## 3. Errores")
    by = Counter((e["modelo"], e["tarea"], e["tipo"]) for e in rows_err)
    for (m, t, k), n in sorted(by.items()):
        L.append(f"{m} | {t} | {k}: {n}")
    for m in models:
        for t in ("escenario",):
            per_img = Counter(e["imagen"].split("/")[-1] for e in rows_err if e["modelo"] == m and e["tarea"] == t)
            per_rep = Counter(e["repeticion"] for e in rows_err if e["modelo"] == m and e["tarea"] == t)
            L.append(f"{m} escenario: por imagen {dict(per_img)} | por repetición {dict(per_rep)}")

    # ── 4. Variabilidad ───────────────────────────────────────
    L.append("\n## 4. Variabilidad entre repeticiones (modelo × imagen × tarea)")
    var_rows = []
    for m, i in itertools.product(models, imgs):
        rs = sorted((r for r in runs if r["modelo"] == m and r["imagen"] == i), key=lambda r: r["repeticion"])
        items = parse_prompt(rs[0]["prompt"])
        texts = [r["narrativa"] for r in rs]
        if any(not t.strip() for t in texts):
            cls = ["error (narrativa vacía)"] if all(not t.strip() for t in texts) else ["error (alguna vacía)"]
        elif len(set(texts)) == 1:
            cls = ["idénticas"]
        else:
            aud = [audit_narrative(t, items) for t in texts]
            obj_sets = [frozenset(x["objeto"] for x in a if x["estado"] != "omitido") for a in aud]
            dir_sets = [tuple(x["estado"] == "direccion_alterada" for x in a) for a in aud]
            steps = [tuple(sorted(int(n) for n in re.findall(r"(\d+)\s+pasos?", t.lower()))) for t in texts]
            cls = []
            if len(set(obj_sets)) > 1:
                cls.append("cambio de objetos")
            if len(set(dir_sets)) > 1:
                cls.append("cambio de orientación")
            if len(set(steps)) > 1:
                cls.append("cambio de información (pasos)")
            if not cls:
                cls.append("variación lingüística")
        var_rows.append({"modelo": m, "imagen": i, "tarea": "narrativa", "clasificacion": "; ".join(cls),
                         "rep1": texts[0], "rep2": texts[1], "rep3": texts[2]})
        scenes = [(r["escenario"], bool(r["escenario_error_llm"])) for r in rs]
        if all(e for _, e in scenes):
            sc = ["error en las 3 (heurística)"]
        elif any(e for _, e in scenes):
            sc = ["error en alguna repetición"] + (["cambio de escenario"] if len({s for s, e in scenes if not e}) > 1 else [])
        else:
            sc = ["idénticas"] if len({s for s, _ in scenes}) == 1 else ["cambio de escenario"]
        var_rows.append({"modelo": m, "imagen": i, "tarea": "escenario", "clasificacion": "; ".join(sc),
                         "rep1": f"{scenes[0][0]}{' [ERR]' if scenes[0][1] else ''}",
                         "rep2": f"{scenes[1][0]}{' [ERR]' if scenes[1][1] else ''}",
                         "rep3": f"{scenes[2][0]}{' [ERR]' if scenes[2][1] else ''}"})
    for m in models:
        for t in ("narrativa", "escenario"):
            c = Counter(v["clasificacion"] for v in var_rows if v["modelo"] == m and v["tarea"] == t)
            L.append(f"{m} | {t}: {dict(c)}")

    # ── 5. Calidad observable (solo narrativas con texto) ─────
    L.append("\n## 5. Calidad observable (narrativas con texto)")
    for r in runs:
        items = parse_prompt(r["prompt"])
        text = r["narrativa"]
        aud = audit_narrative(text, items) if text.strip() else []
        c = r["chequeos"]
        gt = GT_PRESENT.get(r["imagen"])
        inp_en = {ES2EN.get(x.lower(), x) for x in r["objetos_en_prompt"]}
        rows_exec.append({
            "modelo": r["modelo"], "imagen": r["imagen"], "repeticion": r["repeticion"],
            "narrativa_vacia": not text.strip(), "palabras": c["palabras"], "oraciones": c["oraciones"],
            "supera_60_palabras": c["supera_60_palabras"],
            "lineas_prompt": len(items),
            "lineas_conservadas": sum(a["estado"] == "conservado" for a in aud),
            "lineas_omitidas": sum(a["estado"] == "omitido" for a in aud),
            "lineas_direccion_alterada": sum(a["estado"] == "direccion_alterada" for a in aud),
            "lineas_pasos_alterados": sum(a["estado"] == "pasos_alterados" for a in aud),
            "objetos_fuera_del_prompt": "; ".join(c["vocab_mencionado_fuera_del_prompt"]),
            "pasos_inexistentes": "; ".join(map(str, c["pasos_no_presentes_en_prompt"])),
            "verbos_movimiento": "; ".join(c["verbos_movimiento_prohibidos"]),
            "referencias_fuera_regla5": "; ".join(p for p in NON_RULE5 if p in text.lower()),
            "entrada_no_confirmada_por_gt": "; ".join(sorted(inp_en - gt)) if gt is not None else "sin GT",
            "detalle_lineas": " || ".join(f"{a['linea']} → {a['estado']}" for a in aud),
            "narrativa": text,
        })
    for m in models:
        ex = [e for e in rows_exec if e["modelo"] == m and not e["narrativa_vacia"]]
        if not ex:
            L.append(f"{m}: sin narrativas con texto → no evaluable")
            continue
        tot = sum(e["lineas_prompt"] for e in ex)
        L.append(f"{m}: narrativas={len(ex)} | líneas de entrada={tot} → conservadas={sum(e['lineas_conservadas'] for e in ex)}, "
                 f"omitidas={sum(e['lineas_omitidas'] for e in ex)}, dirección alterada={sum(e['lineas_direccion_alterada'] for e in ex)}, "
                 f"pasos alterados={sum(e['lineas_pasos_alterados'] for e in ex)} | objetos fuera del prompt en {sum(bool(e['objetos_fuera_del_prompt']) for e in ex)} | "
                 f"pasos inexistentes en {sum(bool(e['pasos_inexistentes']) for e in ex)} | verbos de movimiento en {sum(bool(e['verbos_movimiento']) for e in ex)} | "
                 f">60 palabras en {sum(e['supera_60_palabras'] for e in ex)} | referencias fuera de la regla 5 en {sum(bool(e['referencias_fuera_regla5']) for e in ex)}")

    # Problemas de la ENTRADA (producto, antes del LLM) — idénticos para todos los modelos
    L.append("\n## 5b. Entrada (generada por el producto antes del LLM)")
    for i in imgs:
        items = parse_prompt(next(r for r in runs if r["imagen"] == i)["prompt"])
        issues = []
        issues += [f"pluralización '{it['objeto']}'" for it in items if re.match(r"^\d+ .+ de \w+es$", it["objeto"])]
        issues += [f"sin pasos: '{it['linea']}'" for it in items if it["pasos"] is None]
        issues += [f"referencia fuera de la regla 5: '{it['posicion']}'" for it in items if any(p in it["posicion"] for p in NON_RULE5)]
        gt = GT_PRESENT.get(i)
        inp_en = {ES2EN.get(x.lower(), x) for x in next(r for r in runs if r["imagen"] == i)["objetos_en_prompt"]}
        if gt is not None and inp_en - gt:
            issues.append(f"objetos de la entrada no confirmados por el inventario de referencia: {sorted(inp_en - gt)}")
        L.append(f"{i.split('/')[-1]}: " + (" | ".join(issues) if issues else "sin observaciones"))

    # ── Salidas ───────────────────────────────────────────────
    for name, rows in [("ejecuciones.csv", rows_exec), ("errores.csv", rows_err), ("variabilidad.csv", var_rows)]:
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    after = {f: sha(SRC / f) for f in ORIGINALS}
    L.append(f"\nArchivos originales sin cambios (SHA-256 antes = después): {before == after}")
    L.append("SHA-256 originales: " + ", ".join(f"{k}={v[:16]}" for k, v in before.items()))
    (OUT / "resumen_auditoria.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
