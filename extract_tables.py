"""
extract_tables.py — Extractor de tablas científicas desde PDFs
===============================================================

Primary:   Docling (layout + TableFormer structure recognition)
Fallback1: PyMuPDF find_tables() (lines_strict → default)
Fallback2: Camelot lattice (si PyMuPDF tampoco detecta nada)

Por cada tabla produce:
  - PNG renderizado
  - Contenido estructurado (filas/columnas) en tables.json

Uso:
    python extract_tables.py paper.pdf --out extracted/
    python extract_tables.py paper.pdf --out extracted/ --dpi 250
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF >= 1.23

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DPI       = 200
DEFAULT_PADDING   = 16
DEFAULT_MIN_ROWS  = 2
DEFAULT_MIN_COLS  = 2
DEFAULT_MIN_CELLS = 6
MIN_AREA_PT2      = 5000
CAPTION_SEARCH_PT = 80
ROTATION_RATIO    = 2.5


# ─── Caption cercano ───────────────────────────────────────────────────────────
CAPTION_RE = re.compile(
    r"^(Table|TABLE|Tabla|TABLA)\s+(\w+[\.\:]?)",
    re.IGNORECASE,
)

def find_nearby_caption(page, table_bbox, search_pt=CAPTION_SEARCH_PT) -> str:
    x0, y0, x1, y1 = table_bbox
    for zone in [
        (x0 - 20, y0 - search_pt, x1 + 20, y0),
        (x0 - 20, y1,             x1 + 20, y1 + search_pt),
    ]:
        text = page.get_text("text", clip=fitz.Rect(zone)).strip()
        for line in text.split("\n"):
            line = line.strip()
            if CAPTION_RE.match(line):
                return " ".join(l.strip() for l in text.split("\n") if l.strip())[:500]
    return ""


# ─── Rotación ──────────────────────────────────────────────────────────────────
def is_rotated(bbox) -> bool:
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    return w > 0 and (h / w) > ROTATION_RATIO


# ─── Render ────────────────────────────────────────────────────────────────────
def render_table(page, bbox, out_path: Path, dpi: int, rotated: bool):
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    clip = fitz.Rect(bbox)
    pix  = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    if rotated:
        try:
            import PIL.Image, io
            img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
            img = img.rotate(90, expand=True)
            img.save(str(out_path))
            return (img.height, img.width)
        except ImportError:
            pass
    pix.save(str(out_path))
    return (pix.width, pix.height)


# ─── Markdown ──────────────────────────────────────────────────────────────────
def rows_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    def clean(c):
        return str(c).replace("|", "\\|").replace("\n", " ").strip() if c else ""
    header = rows[0]
    lines  = ["| " + " | ".join(clean(c) for c in header) + " |",
              "|" + "|".join("---" for _ in header) + "|"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(clean(c) for c in row) + " |")
    return "\n".join(lines)


# ─── Helpers ───────────────────────────────────────────────────────────────────
def _is_continued(rows) -> bool:
    if not rows or not rows[0]:
        return False
    first = " ".join(str(c) for c in rows[0] if c).lower()
    return "continued" in first


def _passes_filters(bbox, rows, min_rows, min_cols, min_cells) -> bool:
    if not rows:
        return False
    nr    = len(rows)
    nc    = max(len(r) for r in rows) if rows else 0
    cells = sum(1 for row in rows for c in row if c and str(c).strip())
    area  = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return (nr >= min_rows and nc >= min_cols
            and cells >= min_cells and area >= MIN_AREA_PT2)


# ─── Docling primary extractor ─────────────────────────────────────────────────
def _extract_docling_tables(pdf_path: str, doc, out_dir: Path, dpi: int,
                             min_rows: int, min_cols: int, min_cells: int,
                             quiet: bool) -> list[dict]:
    """Primary table extractor using Docling layout + TableFormer."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        if not quiet:
            print("  [Docling] not installed, skipping")
        return []

    try:
        if not quiet:
            print(f"  [Docling] converting {Path(pdf_path).name} for tables...")
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        dloc_doc = result.document
    except Exception as e:
        if not quiet:
            print(f"  [Docling] conversion failed: {e}")
        return []

    results = []

    for tbl in dloc_doc.tables:
        if not getattr(tbl, 'prov', None):
            continue
        prov = tbl.prov[0]
        page_no = prov.page_no
        fitz_page = doc[page_no - 1]
        ph = fitz_page.rect.height
        pw = fitz_page.rect.width

        # Convert Docling bbox to PyMuPDF coords
        try:
            tl = prov.bbox.to_top_left_origin(ph)
            x0, y0, x1, y1 = tl.l, tl.t, tl.r, tl.b
        except Exception:
            x0, x1 = prov.bbox.l, prov.bbox.r
            y0 = ph - prov.bbox.t
            y1 = ph - prov.bbox.b

        bbox = (
            max(0, x0 - DEFAULT_PADDING), max(0, y0 - DEFAULT_PADDING),
            min(pw, x1 + DEFAULT_PADDING), min(ph, y1 + DEFAULT_PADDING),
        )

        # Extract structured rows from Docling TableData
        rows = []
        try:
            for row in tbl.data.grid:
                rows.append([getattr(cell, 'text', '') or '' for cell in row])
        except Exception:
            rows = []

        if not rows:
            continue

        if not _passes_filters(bbox, rows, min_rows, min_cols, min_cells):
            continue

        # Markdown
        try:
            markdown = tbl.export_to_markdown(doc=dloc_doc)
        except Exception:
            try:
                markdown = tbl.export_to_markdown()
            except Exception:
                markdown = rows_to_markdown(rows)

        # Caption from Docling or nearby text
        caption = ''
        try:
            cap_texts = []
            for ref in getattr(tbl, 'captions', []):
                resolved = ref.resolve(dloc_doc) if hasattr(ref, 'resolve') else ref
                t = getattr(resolved, 'text', '') or ''
                if t:
                    cap_texts.append(t)
            caption = ' '.join(cap_texts)[:500]
        except Exception:
            pass
        if not caption:
            caption = find_nearby_caption(fitz_page, bbox)

        label = f"Table_{len(results) + 1}"
        filename = f"p{page_no:03d}_{label}.png"
        out_path = out_dir / filename

        rotated = is_rotated(bbox)
        img_size = render_table(fitz_page, bbox, out_path, dpi, rotated)

        clean_rows = [[str(c).strip() for c in row] for row in rows]
        n_rows = len(clean_rows)
        n_cols = max(len(r) for r in clean_rows) if clean_rows else 0

        entry = {
            "label": label, "kind": "table",
            "page": page_no, "pages": [page_no],
            "bbox": list(bbox), "caption": caption,
            "image_path": str(out_path),
            "image_size": list(img_size),
            "rotated": rotated,
            "row_count": n_rows, "col_count": n_cols,
            "rows": clean_rows,
            "markdown": markdown,
        }
        results.append(entry)

        if not quiet:
            rot_tag = " [rotada]" if rotated else ""
            cap_tag = f" | {caption[:50]}" if caption else ""
            print(f"  [Docling] p{page_no} {label}{rot_tag}: {n_rows}x{n_cols}{cap_tag}")

    return results


# ─── PyMuPDF fallback ──────────────────────────────────────────────────────────
def _extract_pymupdf(doc, min_rows, min_cols, min_cells) -> list[dict]:
    raw = []
    for pi, page in enumerate(doc):
        try:
            found = page.find_tables(
                horizontal_strategy="lines_strict",
                vertical_strategy="lines_strict",
            ).tables
        except Exception:
            found = []
        if not found:
            try:
                found = page.find_tables().tables
            except Exception:
                found = []

        seen_bboxes = set()
        for tab in found:
            rows = tab.extract()
            bbox = tuple(round(v, 1) for v in tab.bbox)
            if bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            if not _passes_filters(bbox, rows, min_rows, min_cols, min_cells):
                continue
            raw.append((pi, bbox, rows))

    merged = []
    for pi, bbox, rows in raw:
        if _is_continued(rows) and merged:
            prev = merged[-1]
            prev["rows_data"].extend(rows[1:])
            prev["pages"].append(pi + 1)
        else:
            merged.append({
                "pi":        pi,
                "page_obj":  doc[pi],
                "bbox":      bbox,
                "rows_data": list(rows),
                "pages":     [pi + 1],
            })

    return merged


# ─── Camelot fallback ──────────────────────────────────────────────────────────
def _extract_camelot(pdf_path: str, doc, min_rows, min_cols, min_cells) -> list[dict]:
    try:
        import camelot
    except ImportError:
        return []

    results = []
    for pi, page in enumerate(doc):
        try:
            tbls = camelot.read_pdf(
                pdf_path, pages=str(pi + 1),
                flavor="lattice", suppress_stdout=True,
            )
        except Exception:
            continue
        for tbl in tbls:
            acc = tbl.parsing_report.get("accuracy", 0)
            if acc < 50:
                continue
            rows = [list(row) for row in tbl.df.values.tolist()]
            x1, y1, x2, y2 = tbl._bbox
            ph   = page.rect.height
            bbox = (round(x1, 1), round(ph - y2, 1),
                    round(x2, 1), round(ph - y1, 1))
            if not _passes_filters(bbox, rows, min_rows, min_cols, min_cells):
                continue
            results.append({
                "pi":        pi,
                "page_obj":  page,
                "bbox":      bbox,
                "rows_data": rows,
                "pages":     [pi + 1],
                "camelot_accuracy": acc,
            })
    return results


# ─── Pipeline principal ────────────────────────────────────────────────────────
def extract_tables_all(pdf_path: str, out_dir: str = "extracted",
                       dpi: int = DEFAULT_DPI,
                       min_rows: int = DEFAULT_MIN_ROWS,
                       min_cols: int = DEFAULT_MIN_COLS,
                       min_cells: int = DEFAULT_MIN_CELLS,
                       quiet: bool = False) -> list[dict]:

    pdf_path_obj = Path(pdf_path)
    out_dir_obj  = Path(out_dir)
    out_dir_obj.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path_obj))

    # 1) Primary: Docling
    candidates_structured = _extract_docling_tables(
        str(pdf_path_obj), doc, out_dir_obj, dpi,
        min_rows, min_cols, min_cells, quiet,
    )
    if candidates_structured:
        doc.close()
        out_json = out_dir_obj / "tables.json"
        out_json.write_text(
            json.dumps({
                "pdf": str(pdf_path_obj),
                "total": len(candidates_structured),
                "items": candidates_structured,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not quiet:
            print(f"\n{len(candidates_structured)} tabla(s) via Docling -> {out_json}")
        return candidates_structured

    if not quiet:
        print("  [Docling] 0 tables found, trying PyMuPDF...")

    # 2) Fallback: PyMuPDF
    candidates = _extract_pymupdf(doc, min_rows, min_cols, min_cells)

    # 3) Tertiary fallback: Camelot
    if not candidates:
        if not quiet:
            print("  (PyMuPDF: 0 tablas → probando Camelot lattice...)")
        candidates = _extract_camelot(str(pdf_path_obj), doc, min_rows, min_cols, min_cells)

    tables = []
    for m in candidates:
        pi        = m["pi"]
        page      = m["page_obj"]
        bbox      = m["bbox"]
        rows_data = m["rows_data"]
        pages     = m["pages"]

        rotated  = is_rotated(bbox)
        label    = f"Table_{len(tables) + 1}"
        filename = f"p{pi + 1:03d}_{label}.png"
        out_path = out_dir_obj / filename

        img_size = render_table(page, bbox, out_path, dpi, rotated)
        caption  = find_nearby_caption(page, bbox)

        clean_rows = [
            [str(c).strip() if c is not None else "" for c in row]
            for row in rows_data
        ]
        n_rows = len(clean_rows)
        n_cols = max(len(r) for r in clean_rows) if clean_rows else 0

        entry = {
            "label":      label,
            "kind":       "table",
            "page":       pi + 1,
            "pages":      pages,
            "bbox":       list(bbox),
            "caption":    caption,
            "image_path": str(out_path),
            "image_size": list(img_size),
            "rotated":    rotated,
            "row_count":  n_rows,
            "col_count":  n_cols,
            "rows":       clean_rows,
            "markdown":   rows_to_markdown(clean_rows),
        }
        if "camelot_accuracy" in m:
            entry["camelot_accuracy"] = m["camelot_accuracy"]

        tables.append(entry)

        if not quiet:
            method = "camelot" if "camelot_accuracy" in m else "pymupdf"
            rot_tag = " [rotada]" if rotated else ""
            pp_tag  = f" pp={'+'.join(str(p) for p in pages)}" if len(pages) > 1 else ""
            cap_tag = f" | {caption[:50]}" if caption else ""
            print(f"  p{pi+1} {label} [{method}]{rot_tag}{pp_tag}: "
                  f"{n_rows}x{n_cols}{cap_tag}")

    doc.close()

    out_json = out_dir_obj / "tables.json"
    out_json.write_text(
        json.dumps({
            "pdf":   str(pdf_path_obj),
            "total": len(tables),
            "items": tables,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not quiet:
        print(f"\n{len(tables)} tabla(s) -> {out_json}")

    return tables


# ─── CLI ───────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Extrae tablas de PDFs científicos (Docling → PyMuPDF → Camelot)")
    p.add_argument("pdf",        help="PDF de entrada")
    p.add_argument("--out",      default="extracted")
    p.add_argument("--dpi",      type=int, default=DEFAULT_DPI)
    p.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    p.add_argument("--min-cols", type=int, default=DEFAULT_MIN_COLS)
    p.add_argument("--min-cells",type=int, default=DEFAULT_MIN_CELLS)
    p.add_argument("--quiet",    action="store_true")
    args = p.parse_args(argv)

    if not Path(args.pdf).exists():
        sys.exit(f"Archivo no encontrado: {args.pdf}")

    print(f"Extrayendo tablas: {args.pdf}")
    tables = extract_tables_all(
        args.pdf,
        out_dir   = args.out,
        dpi       = args.dpi,
        min_rows  = args.min_rows,
        min_cols  = args.min_cols,
        min_cells = args.min_cells,
        quiet     = args.quiet,
    )
    print(f"Total: {len(tables)} tabla(s)")


if __name__ == "__main__":
    main()
