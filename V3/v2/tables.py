from __future__ import annotations

import json
import sys
from pathlib import Path


LEGACY_DIR = Path(__file__).resolve().parent.parent
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import extract_tables as legacy_tables


def extract_tables_for_pdf(
    pdf_path: str | Path,
    paper_dir: str | Path,
    dpi: int = 200,
    min_rows: int = 2,
    min_cols: int = 2,
    min_cells: int = 6,
    quiet: bool = False,
) -> list[dict]:
    paper_dir = Path(paper_dir)
    results = legacy_tables.extract_tables_all(
        pdf_path=str(pdf_path),
        out_dir=str(paper_dir),
        dpi=dpi,
        min_rows=min_rows,
        min_cols=min_cols,
        min_cells=min_cells,
        quiet=quiet,
    )
    return normalize_table_items(results, paper_dir)


def normalize_table_items(items: list[dict], paper_dir: str | Path) -> list[dict]:
    paper_dir = Path(paper_dir)
    normalized: list[dict] = []
    for item in items:
        image_path = item.get("image_path") or ""
        try:
            image_name = Path(image_path).name if image_path else ""
        except Exception:
            image_name = str(image_path)
        normalized.append(
            {
                "label": item.get("label", ""),
                "kind": item.get("kind", "table"),
                "page": item.get("page"),
                "pages": item.get("pages", [item.get("page")]),
                "bbox": item.get("bbox", []),
                "render_bbox": item.get("render_bbox", []),
                "caption": item.get("caption", ""),
                "image_path": image_name,
                "image_size": item.get("image_size", []),
                "row_count": item.get("row_count"),
                "col_count": item.get("col_count"),
                "rows": item.get("rows", []),
                "markdown": item.get("markdown", ""),
            }
        )
    out_path = paper_dir / "tables.normalized.json"
    out_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalized
