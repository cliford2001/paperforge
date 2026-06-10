from __future__ import annotations

import json
import sys
from pathlib import Path


LEGACY_DIR = Path(__file__).resolve().parent.parent
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import extract_figures as legacy_figures


def extract_figures_for_pdf(
    pdf_path: str | Path,
    paper_dir: str | Path,
    dpi: int = 200,
    max_height: int = 9999,
    quiet: bool = False,
) -> list[dict]:
    paper_dir = Path(paper_dir)
    results = legacy_figures.extract_all(
        pdf_path=str(pdf_path),
        out_dir=str(paper_dir),
        dpi=dpi,
        max_height=max_height,
        quiet=quiet,
    )
    return normalize_figure_items(results, paper_dir)


def normalize_figure_items(items: list[dict], paper_dir: str | Path) -> list[dict]:
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
                "kind": item.get("kind", "figure"),
                "page": item.get("page"),
                "bbox": item.get("bbox", []),
                "caption": item.get("caption", ""),
                "image_path": image_name,
                "image_size": item.get("image_size", []),
            }
        )
    out_path = paper_dir / "figures.normalized.json"
    out_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalized
