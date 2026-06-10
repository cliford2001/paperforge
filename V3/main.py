from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from analyze import analyze_paper_outputs
from db import load_papers
from download_pdf import download_pdf_for_paper, write_paper_context
from figures import extract_figures_for_pdf
from tables import extract_tables_for_pdf


def process_one_paper(
    paper,
    out_dir: str | Path,
    keep_pdf: bool = True,
    quiet: bool = False,
    reuse_existing_pdf: bool = True,
    skip_if_done: bool = False,
    run_analysis: bool = False,
    server: str | None = None,
    context_mode: str = "bm25",
    max_context_words: int = 0,
    top_k: int = 6,
    chunk_words: int = 180,
    chunk_overlap: int = 30,
    max_tokens: int = 800,
    temperature: float = 0.0,
    timeout: int = 600,
) -> dict:
    out_dir = Path(out_dir)
    paper_dir = out_dir / paper.pmcid
    paper_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "pmcid": paper.pmcid,
        "doi": paper.doi,
        "title": paper.title,
        "status": "ok",
        "error": None,
        "figures": 0,
        "tables": 0,
        "analysis": None,
        "elapsed_sec": 0.0,
    }
    t0 = time.time()

    try:
        if skip_if_done and (paper_dir / "figures.json").exists() and (paper_dir / "tables.json").exists():
            figures = json.loads((paper_dir / "figures.json").read_text(encoding="utf-8")).get("items", [])
            tables = json.loads((paper_dir / "tables.json").read_text(encoding="utf-8")).get("items", [])
            summary["figures"] = len(figures)
            summary["tables"] = len(tables)
            return summary

        write_paper_context(paper_dir, asdict(paper))
        pdf_path = download_pdf_for_paper(
            pmcid=paper.pmcid,
            doi=paper.doi,
            paper_dir=paper_dir,
            reuse_existing=reuse_existing_pdf,
        )
        figures = extract_figures_for_pdf(pdf_path=pdf_path, paper_dir=paper_dir, quiet=quiet)
        tables = extract_tables_for_pdf(pdf_path=pdf_path, paper_dir=paper_dir, quiet=quiet)

        summary["figures"] = len(figures)
        summary["tables"] = len(tables)

        if run_analysis:
            if not server:
                raise ValueError("run_analysis requiere --server")
            analysis_path = analyze_paper_outputs(
                paper_dir=paper_dir,
                server=server,
                context_mode=context_mode,
                max_context_words=max_context_words,
                top_k=top_k,
                chunk_words=chunk_words,
                chunk_overlap=chunk_overlap,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            summary["analysis"] = analysis_path.name

        if not keep_pdf and pdf_path.exists():
            pdf_path.unlink()
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = repr(exc)

    summary["elapsed_sec"] = round(time.time() - t0, 2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaperForge V3: parquet -> PDF -> figures/tables -> optional layered VLM analysis")
    parser.add_argument("--input-parquet", help="parquet con pmcid, doi, text_clean", default=None)
    parser.add_argument("--metadata-parquet", help="parquet metadata", default=None)
    parser.add_argument("--texts-parquet", help="parquet con text_clean", default=None)
    parser.add_argument("--out-dir", required=True, help="directorio raiz de resultados")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--keep-pdf", action="store_true", help="conservar paper.pdf")
    parser.add_argument("--reuse-existing-pdf", action="store_true", help="reutilizar paper.pdf si ya existe")
    parser.add_argument("--skip-if-done", action="store_true", help="saltar papers con figures.json y tables.json")
    parser.add_argument("--clean-out-dir", action="store_true", help="borrar out-dir antes de empezar")
    parser.add_argument("--run-analysis", action="store_true", help="correr analisis VLM despues de extraer")
    parser.add_argument("--server", default=None, help="endpoint OpenAI-compatible del VLM")
    parser.add_argument("--context-mode", choices=["none", "bm25", "full", "layered"], default="bm25",
                        help="none = imagen+caption | bm25 = contexto recuperado | full = texto completo | layered = mapa global + contexto local")
    parser.add_argument("--max-context-words", type=int, default=0, help="solo para context-mode=full")
    parser.add_argument("--top-k", type=int, default=6, help="solo para context-mode=bm25/layered")
    parser.add_argument("--chunk-words", type=int, default=180, help="solo para context-mode=bm25/layered")
    parser.add_argument("--chunk-overlap", type=int, default=30, help="solo para context-mode=bm25/layered")
    parser.add_argument("--max-tokens", type=int, default=800, help="maximo de tokens de salida por item")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = load_papers(
        input_parquet=args.input_parquet,
        metadata_parquet=args.metadata_parquet,
        texts_parquet=args.texts_parquet,
        limit=args.limit,
        offset=args.offset,
    )

    summaries = []
    for idx, paper in enumerate(papers, start=1):
        if not args.quiet:
            print(f"[{idx}/{len(papers)}] {paper.pmcid}")
        summary = process_one_paper(
            paper=paper,
            out_dir=out_dir,
            keep_pdf=args.keep_pdf,
            quiet=args.quiet,
            reuse_existing_pdf=args.reuse_existing_pdf,
            skip_if_done=args.skip_if_done,
            run_analysis=args.run_analysis,
            server=args.server,
            context_mode=args.context_mode,
            max_context_words=args.max_context_words,
            top_k=args.top_k,
            chunk_words=args.chunk_words,
            chunk_overlap=args.chunk_overlap,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        summaries.append(summary)
        if not args.quiet:
            print(
                f"  -> {summary['status']} figures={summary['figures']} "
                f"tables={summary['tables']} analysis={summary['analysis']} "
                f"elapsed={summary['elapsed_sec']}s"
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.quiet:
        print(f"\nsummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
