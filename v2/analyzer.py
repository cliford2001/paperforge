from __future__ import annotations

import sys
from pathlib import Path


LEGACY_DIR = Path(__file__).resolve().parent.parent
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import analyze_figures_v2_rag as legacy_analyzer


CONTEXT_MODES = {"none", "bm25", "full"}


def analyze_paper_outputs(
    paper_dir: str | Path,
    server: str,
    context_mode: str = "bm25",
    max_context_words: int = 0,
    top_k: int = 6,
    chunk_words: int = 180,
    chunk_overlap: int = 30,
    max_tokens: int = 800,
    temperature: float = 0.0,
    timeout: int = 600,
) -> Path:
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"context_mode invǭlido: {context_mode}")

    paper_dir = Path(paper_dir)
    figures_json = paper_dir / "figures.json"
    tables_json = paper_dir / "tables.json"
    pdf_path = paper_dir / "paper.pdf"
    context_file = paper_dir / "paper_context.txt"
    out_path = paper_dir / "analyses_rag.json"

    if not figures_json.exists():
        raise FileNotFoundError(f"no existe {figures_json}")

    kwargs = {
        "figures_json": str(figures_json),
        "tables_json": str(tables_json) if tables_json.exists() else None,
        "server": server,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
        "out_path": str(out_path),
    }

    if context_mode == "none":
        kwargs["pdf_path"] = None
        kwargs["context_file"] = None
    else:
        kwargs["context_strategy"] = context_mode
        kwargs["pdf_path"] = str(pdf_path) if pdf_path.exists() else None
        kwargs["context_file"] = str(context_file) if context_file.exists() else None
        kwargs["max_context_words"] = max_context_words
        kwargs["top_k"] = top_k
        kwargs["chunk_words"] = chunk_words
        kwargs["overlap"] = chunk_overlap

    legacy_analyzer.analyze_all(**kwargs)
    return out_path
