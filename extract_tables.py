"""
extract_tables.py — Extractor de tablas científicas desde PDFs
===============================================================

Usa page.find_tables() de PyMuPDF para detectar tablas estructuralmente
(por alineación de texto y líneas de separación), completamente independiente
del extractor de figuras.

Por cada tabla produce:
  - PNG renderizado (corregido si está rotado)
  - Contenido estructurado (filas/columnas) en tables.json

Uso:
    python extract_tables.py paper.pdf --out extracted/
    python extract_tables.py paper.pdf --out extracted/ --dpi 250 --min-rows 2

Integración con pipeline:
    from extract_tables import extract_tables_all
    extract_tables_all("paper.pdf", out_dir="extracted/")
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF >= 1.23

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DPI       = 200
DEFAULT_MIN_ROWS  = 2    # tablas con menos filas se descartan
DEFAULT_MIN_COLS  = 2    # tablas con menos columnas se descartan
DEFAULT_MIN_CELLS = 6    # mínimo de celdas no vacías para considerar válida
CAPTION_SEARCH_PT = 80   # distancia en pt para buscar caption cerca de la tabla
ROTATION_RATIO    = 2.5  # si height/width > este valor → tabla rotada 90°


# ─── Búsqueda de captions cercanos ───────────────────────────────────────────
CAPTION_RE = re.compile(
    r"^(Table|TABLE|Tabla|TABLA)\s+(\w+[\.\:]?)",
    re.IGNORECASE,
)

def find_nearby_caption(page, table_bbox, search_pt=CAPTION_SEARCH_PT) -> str:
    """Busca texto tipo 'Table N ...' cerca de la bbox de la tabla."""
    x0, y0, x1, y1 = table_bbox
    # Buscar arriba y abajo de la tabla
    search_zones = [
        (x0 - 20, y0 - search_pt, x1 + 20, y0),   # zona sobre la tabla
        (x0 - 20, y1, x1 + 20, y1 + search_pt),    # zona bajo la tabla
    ]
    for zone in search_zones:
        clip = fitz.Rect(zone)
        text = page.get_text("text", clip=clip).strip()
        for line in text.split("\n"):
            line = line.strip()
            if CAPTION_RE.match(line):
                # Recuperar el párrafo completo
                full = " ".join(
                    l.strip() for l in text.split("\n") if l.strip()
                )[:500]
                return full
    return ""


# ─── Detección de rotación ────────────────────────────────────────────────────
def is_rotated(bbox) -> bool:
    """True si la tabla parece estar rotada 90° (mucho más alta que ancha)."""
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    return w > 0 and (h / w) > ROTATION_RATIO


# ─── Renderizado ──────────────────────────────────────────────────────────────
def render_table(page, bbox, out_path: Path, dpi: int, rotated: bool):
    """Renderiza la región de la tabla a PNG, rotando si es necesario."""
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    clip = fitz.Rect(bbox)
    pix  = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

    if rotated:
        # Rotar 90° en sentido antihorario para que sea legible
        try:
            import PIL.Image, io
            img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
            img = img.rotate(90, expand=True)
            img.save(str(out_path))
            return (img.height, img.width)   # (ancho, alto) tras rotación
        except ImportError:
            pass  # sin PIL: guardar sin rotar

    pix.save(str(out_path))
    return (pix.width, pix.height)


# ─── Conversión a markdown ────────────────────────────────────────────────────
def rows_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    def clean(c):
        return str(c).replace("|", "\\|").replace("\n", " ").strip() if c else ""

    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(clean(c) for c in header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows[1:]:
        lines.append("| " + " | ".join(clean(c) for c in row) + " |")
    return "\n".join(lines)


# ─── Pipeline principal ───────────────────────────────────────────────────────
def extract_tables_all(pdf_path: str, out_dir: str = "extracted",
                       dpi: int = DEFAULT_DPI,
                       min_rows: int = DEFAULT_MIN_ROWS,
                       min_cols: int = DEFAULT_MIN_COLS,
                       min_cells: int = DEFAULT_MIN_CELLS,
                       quiet: bool = False) -> list[dict]:
    pdf_path = Path(pdf_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc    = fitz.open(str(pdf_path))
    tables = []
    seen   = set()  # evitar duplicados por bbox

    for pi, page in enumerate(doc):
        page_width  = page.rect.width
        page_height = page.rect.height

        try:
            tab_finder = page.find_tables(
                horizontal_strategy="lines_strict",
                vertical_strategy="lines_strict",
            )
            found = tab_finder.tables
        except Exception:
            # Fallback a estrategia más permisiva si no hay líneas
            try:
                tab_finder = page.find_tables()
                found = tab_finder.tables
            except Exception:
                found = []

        for ti, tab in enumerate(found):
            bbox = tuple(round(v, 1) for v in tab.bbox)

            # Deduplicar (misma bbox en páginas distintas no debería pasar,
            # pero por si acaso)
            key = (pi, bbox)
            if key in seen:
                continue
            seen.add(key)

            # Filtros de calidad
            rows = tab.extract()
            if not rows:
                continue

            n_rows = len(rows)
            n_cols = max(len(r) for r in rows) if rows else 0
            n_cells_filled = sum(
                1 for row in rows for cell in row
                if cell and str(cell).strip()
            )

            if n_rows < min_rows or n_cols < min_cols:
                continue
            if n_cells_filled < min_cells:
                continue

            # Detectar rotación
            rotated = is_rotated(bbox)

            # Nombre de archivo
            label    = f"Table_{len(tables) + 1}"
            filename = f"p{pi + 1:03d}_{label}.png"
            out_path = out_dir / filename

            # Renderizar
            img_size = render_table(page, bbox, out_path, dpi, rotated)

            # Buscar caption cercano
            caption = find_nearby_caption(page, bbox)

            # Normalizar filas (reemplazar None por "")
            clean_rows = [
                [str(c).strip() if c is not None else "" for c in row]
                for row in rows
            ]

            entry = {
                "label":      label,
                "kind":       "table",
                "page":       pi + 1,
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
            tables.append(entry)

            if not quiet:
                rot_tag = " [rotada]" if rotated else ""
                cap_tag = f" | {caption[:60]}" if caption else ""
                print(f"  p{pi + 1} {label}: {n_rows}×{n_cols}{rot_tag}{cap_tag}")

    doc.close()

    # Guardar tables.json
    out_json = out_dir / "tables.json"
    out_json.write_text(
        json.dumps({
            "pdf":    str(pdf_path),
            "total":  len(tables),
            "items":  tables,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not quiet:
        print(f"\n{len(tables)} tabla(s) → {out_json}")

    return tables


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Extrae tablas de PDFs científicos usando PyMuPDF find_tables()")
    p.add_argument("pdf",       help="PDF de entrada")
    p.add_argument("--out",     default="extracted",
                   help="Directorio de salida (def: extracted/)")
    p.add_argument("--dpi",     type=int, default=DEFAULT_DPI)
    p.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS,
                   help=f"Mínimo de filas para considerar una tabla (def: {DEFAULT_MIN_ROWS})")
    p.add_argument("--min-cols", type=int, default=DEFAULT_MIN_COLS,
                   help=f"Mínimo de columnas (def: {DEFAULT_MIN_COLS})")
    p.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS,
                   help=f"Mínimo de celdas no vacías (def: {DEFAULT_MIN_CELLS})")
    p.add_argument("--quiet",   action="store_true")
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
