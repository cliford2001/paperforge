"""
LLM Analyzer for Extracted Figures
====================================

Toma el output de extract_figures.py (figures.json + PNGs) y envía cada
imagen a un servidor de inferencia OpenAI-compatible (llama.cpp, Ollama,
vLLM, etc.) junto con el contexto textual del paper (texto de las páginas
adyacentes al caption).

Para cada figura/tabla genera DOS análisis:
  1. Sin contexto:  solo la imagen
  2. Con contexto:  imagen + texto de páginas vecinas del PDF

Uso:
    python analyze_figures.py extracted/figures.json --pdf paper.pdf \\
        --server http://127.0.0.1:8080/v1/chat/completions

    python analyze_figures.py extracted/figures.json --pdf paper.pdf \\
        --no-context              # solo análisis sin contexto

Resume automático: si analyses.json ya existe, salta items ya procesados.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_SERVER        = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MAX_TOKENS    = 600
DEFAULT_TEMPERATURE   = 0.1
DEFAULT_TIMEOUT       = 300
DEFAULT_PAGE_WINDOW   = 3
DEFAULT_MAX_CTX_CHARS = 30000
MAX_RETRIES           = 5
RETRY_BACKOFF         = 3  # seg × 2^intento


PROMPT_FIG_NO_CTX = (
    "You are an expert scientist analyzing a scientific figure from a research paper.\n"
    "INTERPRET the science, do not just describe pixels.\n\n"
    "- What is the figure showing? (data, comparison, mechanism, workflow)\n"
    "- Key findings, trends or patterns visible\n"
    "- What this figure proves, suggests, or rules out\n"
    "- If multi-panel (A, B, C...), address each panel briefly\n\n"
    "Be precise and concise."
)

PROMPT_FIG_WITH_CTX = (
    "You are an expert scientist. Below is text from the paper containing this figure, "
    "followed by the figure itself.\n\n"
    "PAPER TEXT:\n{context}\n\n---\n\n"
    "INTERPRET the figure using the paper as context:\n"
    "- What specific hypothesis or claim does this figure test or support?\n"
    "- What does the data reveal about the biology/mechanism/system being studied?\n"
    "- Key quantitative findings (values, fold-changes, p-values, comparisons)\n"
    "- How does this figure advance the paper's overall argument?\n\n"
    "Connect specific visual evidence to claims in the text. Be precise."
)

PROMPT_TBL_NO_CTX = (
    "You are an expert analyzing a table from a scientific paper.\n"
    "- What does this table compare or summarize?\n"
    "- Which entries stand out (best/worst/surprising)?\n"
    "- What conclusion does it support?\n"
    "Be precise and concise."
)

PROMPT_TBL_WITH_CTX = (
    "You are an expert scientist. Below is text from the paper containing this table.\n\n"
    "PAPER TEXT:\n{context}\n\n---\n\n"
    "INTERPRET the table using the paper as context:\n"
    "- What is being compared and why?\n"
    "- Most relevant entries given the paper's claims\n"
    "- How does this table support the paper's argument?\n"
    "Be precise and quantitative."
)


# ─── Extracción de contexto textual ─────────────────────────────────────────
def extract_page_context(pdf_path, center_page, window, max_chars):
    """Texto de páginas adyacentes a center_page (1-indexed), truncado."""
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    start = max(0, center_page - 1 - window)
    end   = min(total, center_page + window)
    parts = []
    for i in range(start, end):
        text = doc[i].get_text("text").strip()
        if text:
            parts.append(f"[Page {i + 1}]\n{text}")
    doc.close()
    full = "\n\n".join(parts)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n[... truncated ...]"
    return full


# ─── Cliente HTTP con retries ────────────────────────────────────────────────
def ask_api(server, img_bytes, prompt, max_tokens, temperature, timeout):
    b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(server, json=payload, timeout=timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            if not content:
                raise ValueError("respuesta vacía")
            return content
        except Exception as e:
            last_err = e
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"    intento {attempt + 1}/{MAX_RETRIES}: {e} - retry en {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"falló tras {MAX_RETRIES} intentos: {last_err}")


def server_health(server):
    try:
        url = server.rsplit("/v1/", 1)[0] + "/health"
        r = requests.get(url, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ─── Pipeline ────────────────────────────────────────────────────────────────
def analyze_all(figures_json, pdf_path, server, with_context=True,
                max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE,
                timeout=DEFAULT_TIMEOUT, window=DEFAULT_PAGE_WINDOW,
                max_ctx_chars=DEFAULT_MAX_CTX_CHARS, out_path=None):
    fig_json = Path(figures_json)
    pdf_path = Path(pdf_path) if pdf_path else None
    meta = json.loads(fig_json.read_text(encoding="utf-8"))
    items = meta["items"]

    if out_path is None:
        out_path = fig_json.parent / "analyses.json"
    else:
        out_path = Path(out_path)

    # Resume
    results = []
    done = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for it in prev.get("items", []):
                if "analysis" in it or "analysis_no_ctx" in it:
                    results.append(it)
                    done.add(it["label"])
        except Exception:
            pass

    print(f"Items: {len(items)} | hechos: {len(done)} | server: {server}")
    print(f"Contexto: {'sí' if with_context else 'no'}\n")

    for i, item in enumerate(items):
        if item["label"] in done:
            print(f"[{i + 1}/{len(items)}] {item['label']} SKIP")
            continue

        img_path = Path(item["image_path"])
        if not img_path.is_absolute():
            img_path = fig_json.parent / img_path.name
        img_bytes = img_path.read_bytes()

        is_table = item["kind"] == "table"
        result = dict(item)
        t_start = time.time()

        # Análisis sin contexto
        prompt_nc = PROMPT_TBL_NO_CTX if is_table else PROMPT_FIG_NO_CTX
        try:
            ans = ask_api(server, img_bytes, prompt_nc, max_tokens, temperature, timeout)
            result["analysis_no_ctx"] = ans
        except Exception as e:
            result["error_no_ctx"] = str(e)

        # Análisis con contexto (si pdf disponible)
        if with_context and pdf_path and pdf_path.exists():
            ctx_text = extract_page_context(pdf_path, item["page"], window, max_ctx_chars)
            tmpl = PROMPT_TBL_WITH_CTX if is_table else PROMPT_FIG_WITH_CTX
            try:
                ans = ask_api(server, img_bytes, tmpl.format(context=ctx_text),
                              max_tokens, temperature, timeout)
                result["analysis_with_ctx"] = ans
            except Exception as e:
                result["error_with_ctx"] = str(e)

        result["elapsed_sec"] = round(time.time() - t_start, 1)
        results.append(result)

        preview = (result.get("analysis_no_ctx") or "")[:100].replace("\n", " ")
        print(f"[{i + 1}/{len(items)}] {item['label']} p{item['page']} "
              f"({result['elapsed_sec']}s) {preview}...")

        # Guardado incremental
        out_path.write_text(
            json.dumps({"total": len(results), "items": results},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nResultados: {out_path}")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Analiza figuras extraídas con un LLM multimodal vía API OpenAI-compatible.",
    )
    p.add_argument("figures_json",  help="ruta al figures.json producido por extract_figures.py")
    p.add_argument("--pdf",         help="PDF original (para contexto de páginas)")
    p.add_argument("--server",      default=DEFAULT_SERVER,
                   help=f"URL del endpoint chat/completions (def: {DEFAULT_SERVER})")
    p.add_argument("--no-context",  action="store_true",
                   help="solo análisis sin contexto del paper")
    p.add_argument("--out",         help="ruta de salida (def: analyses.json al lado de figures.json)")
    p.add_argument("--max-tokens",  type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--timeout",     type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--window",      type=int, default=DEFAULT_PAGE_WINDOW,
                   help="páginas ±N para contexto (def: 3)")
    args = p.parse_args(argv)

    if not server_health(args.server):
        sys.exit(f"Servidor no responde: {args.server}")

    if not args.no_context and not args.pdf:
        sys.exit("--pdf requerido para análisis con contexto (o usar --no-context)")

    analyze_all(
        args.figures_json,
        pdf_path=args.pdf,
        server=args.server,
        with_context=not args.no_context,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        window=args.window,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
