"""
app/study_export.py

Exportación de la evidencia de UNA sesión del estudio con usuarios en un .zip:

  <codigo>_<session_id>/
    LEEME.txt                         qué contiene cada archivo
    evidencia_<codigo>.xlsx           hojas: Sesion, Respuestas, Codificacion, Resumen_por_prueba
    sesion.json · responses.jsonl     registro original (sin modificar)
    grabacion_consentimiento.<ext>    lectura del consentimiento (si se almacenó)
    respuestas/formales|ensayos/R00N/ audio escuchado y grabación del participante

El .xlsx se escribe sin dependencias externas (SpreadsheetML mínimo con cadenas en
línea), para no cambiar el entorno congelado de la imagen.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

# ──────────────────────────────────────────────────────────────
# XLSX mínimo
# ──────────────────────────────────────────────────────────────

def _col(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell(ref: str, v, header: bool) -> str:
    style = ' s="1"' if header else ""
    if v is None or v == "":
        return f'<c r="{ref}"{style}/>'
    if isinstance(v, bool):
        v = "sí" if v else "no"
    if isinstance(v, (int, float)):
        return f'<c r="{ref}"{style}><v>{v}</v></c>'
    if isinstance(v, (list, dict)):
        v = json.dumps(v, ensure_ascii=False)
    text = escape(str(v))[:32000].replace("\n", "&#10;")
    return f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">{text}</t></is></c>'


def _sheet(rows: list[list]) -> str:
    out = []
    for r, row in enumerate(rows):
        cells = "".join(_cell(f"{_col(c)}{r + 1}", v, r == 0) for c, v in enumerate(row))
        out.append(f'<row r="{r + 1}">{cells}</row>')
    widths = '<cols><col min="1" max="60" width="22" customWidth="1"/></cols>'
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews>'
            f'{widths}<sheetData>{"".join(out)}</sheetData></worksheet>')


def build_xlsx(sheets: list[tuple[str, list[list]]]) -> bytes:
    """[(nombre, filas)] → bytes de un .xlsx válido (la primera fila es el encabezado)."""
    ns = 'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                   + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                             'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                             for i in range(len(sheets)))
                   + '</Types>')
        z.writestr("_rels/.rels",
                   f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships {ns}>'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   '</Relationships>')
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   + "".join(f'<sheet name="{escape(n[:31])}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, (n, _) in enumerate(sheets))
                   + '</sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships {ns}>'
                   + "".join(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>'
                             for i in range(len(sheets)))
                   + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                   '</Relationships>')
        z.writestr("xl/styles.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                   '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
                   '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
                   '<fill><patternFill patternType="solid"><fgColor rgb="FFD9E2F3"/><bgColor indexed="64"/></patternFill></fill></fills>'
                   '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                   '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                   '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
                   '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>'
                   '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
                   '</styleSheet>')
        for i, (_, rows) in enumerate(sheets):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", _sheet(rows))
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────
# Contenido de la evidencia
# ──────────────────────────────────────────────────────────────

def _flat(prefix: str, d, out: list):
    if isinstance(d, dict):
        for k, v in d.items():
            _flat(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out.append([prefix, d])


def session_rows(sesion: dict, resumen: dict) -> list[list]:
    rows = [["Campo", "Valor"]]
    for k in ("codigo", "tipo_participante", "es_prueba_tecnica", "session_id", "creado", "estado"):
        rows.append([k, sesion.get(k)])
    rows.append(["duracion_sesion_min", resumen.get("duracion_sesion_min")])
    _flat("ficha", sesion.get("ficha") or {}, rows)
    _flat("contexto", sesion.get("contexto") or {}, rows)
    c = dict(sesion.get("consentimiento") or {})
    c.pop("grabacion", None)
    _flat("consentimiento", c, rows)
    _flat("cierre", sesion.get("cierre") or {}, rows)
    for k in ("backend_commit", "catalogo_schema_version", "almacenamiento"):
        rows.append([k, sesion.get(k)])
    return rows


RESP_HEADER = [
    "response_id", "modo", "prueba", "nombre_prueba", "estimulo", "origen_audio", "audio_sha256", "tts_modelo",
    "narrativa", "reproducciones", "repeticiones", "aclaraciones", "tiempo_respuesta_ms", "transcripcion",
    "objetos_identificados", "objetos_referencia", "pct_objetos_identificados", "objetos_omitidos", "objetos_inventados",
    "ubicaciones_correctas", "ubicaciones_evaluadas", "distancias_correctas", "distancias_evaluadas",
    "relaciones_comprendidas", "relaciones_evaluadas",
    "decision_seleccionada", "decision_narrativa", "decision_sigue_narrativa", "decision_esperada_diseno",
    "decision_coincide_diseno", "percepcion_cambio", "criterios", "comentarios", "observaciones",
    "incidencias", "errores_derivados", "grabacion_participante", "registrado_en",
]


def response_rows(respuestas: list[dict]) -> list[list]:
    rows = [RESP_HEADER]
    for r in respuestas:
        m, d, e = r.get("metricas") or {}, r.get("decision") or {}, r.get("ejecucion") or {}
        g = r.get("grabacion_participante") or {}
        rows.append([
            r["response_id"], r["modo"], r["prueba"]["id"], r["prueba"].get("nombre"),
            (r.get("estimulo") or {}).get("stimulus_id") or (r.get("estimulo") or {}).get("nombre_archivo"),
            e.get("origen_audio"), (e.get("audio") or {}).get("sha256"), e.get("tts_modelo"), e.get("narrativa_final"),
            len(r.get("reproducciones") or []), m.get("repeticiones_audio"), m.get("aclaraciones"), m.get("tiempo_respuesta_ms"),
            r.get("respuesta_transcrita"),
            m.get("objetos_identificados"), m.get("objetos_referencia"), m.get("pct_objetos_identificados"),
            ", ".join(m.get("objetos_omitidos") or []), ", ".join(m.get("objetos_inventados") or []),
            m.get("ubicaciones_correctas"), m.get("ubicaciones_evaluadas"), m.get("distancias_correctas"), m.get("distancias_evaluadas"),
            m.get("relaciones_comprendidas"), m.get("relaciones_evaluadas"),
            d.get("seleccionada"), d.get("direccion_narrativa"), d.get("coincide_con_narrativa"), d.get("esperada"),
            d.get("correcto"), r.get("percepcion_cambio"), r.get("criterios"), r.get("comentarios"), r.get("observaciones"),
            "; ".join(f'{x["tipo"]}: {x["descripcion"]}' for x in r.get("errores") or []),
            "; ".join(f'{x["tipo"]}: {x["elemento"]}' for x in r.get("errores_derivados") or []),
            g.get("archivo"), r.get("registrado_en"),
        ])
    return rows


def coding_rows(respuestas: list[dict]) -> list[list]:
    rows = [["response_id", "prueba", "tipo", "elemento_narrativa", "identificado", "ubicacion_reportada",
             "ubicacion_coincide", "distancia_reportada", "distancia_coincide", "respuesta_relacion", "relacion_comprendida"]]
    for r in respuestas:
        comp = r.get("comprension") or {}
        for o in comp.get("objetos") or []:
            rows.append([r["response_id"], r["prueba"]["id"], "objeto", o["objeto"], o.get("identificado"),
                         o.get("ubicacion_reportada"), o.get("ubicacion_correcta"), o.get("distancia_reportada"),
                         o.get("distancia_correcta"), None, None])
        for x in comp.get("objetos_inventados") or []:
            rows.append([r["response_id"], r["prueba"]["id"], "objeto_inventado", x, None, None, None, None, None, None, None])
        for rel in comp.get("relaciones") or []:
            rows.append([r["response_id"], r["prueba"]["id"], "relacion", rel["relacion"], None, None, None, None, None,
                         rel.get("respuesta"), rel.get("comprendida")])
    return rows


def summary_rows(resumen: dict) -> list[list]:
    keys = ["respuestas", "objetos_identificados", "objetos_referencia", "pct_objetos_identificados", "objetos_omitidos",
            "objetos_inventados", "ubicaciones_correctas", "ubicaciones_evaluadas", "pct_ubicaciones_correctas",
            "distancias_correctas", "distancias_evaluadas", "relaciones_comprendidas", "relaciones_evaluadas",
            "pct_relaciones_comprendidas", "decisiones_registradas", "decisiones_siguen_narrativa",
            "decisiones_no_siguen_narrativa", "decisiones_coinciden_diseno", "decisiones_no_coinciden_diseno",
            "decisiones_sin_respuesta", "percepcion_cambio", "criterios", "repeticiones_audio", "aclaraciones",
            "tiempo_respuesta_mediana_ms"]
    rows = [["prueba", *keys]]
    for pid, g in (resumen.get("por_prueba") or {}).items():
        rows.append([pid, *[g.get(k) for k in keys]])
    return rows


LEEME = """Evidencia de la sesión {codigo} ({session_id})
Exportada el {fecha} (UTC) desde VisionNav — Evaluación con usuarios.

evidencia_{codigo}.xlsx
  Sesion              ficha del participante, condiciones, consentimiento y cierre.
  Respuestas          una fila por respuesta (formales y práctica/ensayos), con sus métricas.
  Codificacion        una fila por objeto, objeto inventado o relación codificada.
  Resumen_por_prueba  conteos descriptivos de las respuestas FORMALES (sin tasa de éxito global).
sesion.json, responses.jsonl
  Registro original del servidor, sin modificar.
grabacion_consentimiento.*
  Lectura del consentimiento y respuestas del participante (si se almacenó).
respuestas/formales/R00N/   respuestas formales: evidencia.
respuestas/ensayos/R00N/    práctica y ensayos: NO son evidencia.
  audio_narrativa_api.*       audio que escuchó el participante (congelado; mismo sha256 que en el registro).
  respuesta_participante.*    grabación de la respuesta verbal del participante.

La correspondencia entre el código y el nombre del participante solo consta en el acta física.
"""


def build_session_zip(d: Path, sesion: dict, respuestas: list[dict], resumen: dict) -> tuple[bytes, str]:
    """(.zip en memoria, nombre de archivo) con toda la evidencia de la sesión."""
    codigo, sid = sesion["codigo"], sesion["session_id"]
    root = f"{codigo}_{sid}"
    xlsx = build_xlsx([
        ("Sesion", session_rows(sesion, resumen)),
        ("Respuestas", response_rows(respuestas)),
        ("Codificacion", coding_rows(respuestas)),
        ("Resumen_por_prueba", summary_rows(resumen)),
    ])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{root}/LEEME.txt", LEEME.format(codigo=codigo, session_id=sid,
                                                    fecha=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")))
        z.writestr(f"{root}/evidencia_{codigo}.xlsx", xlsx)
        for f in sorted(d.rglob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                z.write(f, f"{root}/{f.relative_to(d).as_posix()}")
    return buf.getvalue(), f"evidencia_{codigo}_{sid}.zip"
