from __future__ import annotations

import json
import sys
from pathlib import Path


LEGACY_DIR = Path(__file__).resolve().parent.parent
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import pipeline_db_to_analysis as legacy_pipeline


def download_pdf_for_paper(pmcid: str, doi: str, paper_dir: str | Path, reuse_existing: bool = True) -> Path:
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / "paper.pdf"

    if reuse_existing and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    ok = legacy_pipeline.download_pdf(pmcid, doi, pdf_path)
    if not ok or not pdf_path.exists():
        raise RuntimeError(f"pdf_download_failed:{pmcid}")
    return pdf_path


def write_paper_context(paper_dir: str | Path, paper: dict) -> Path:
    paper_dir = Path(paper_dir)
    context_path = paper_dir / "context.json"
    context_path.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
    text_clean = str(paper.get("text_clean", "") or "").strip()
    if text_clean:
        (paper_dir / "paper_context.txt").write_text(text_clean, encoding="utf-8")
    return context_path
