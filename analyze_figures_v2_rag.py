"""
LLM Analyzer for Extracted Figures — v2 RAG
=============================================

Versión RAG de analyze_figures.py.

En lugar de mandar el paper completo (o su resumen) como contexto para todas
las figuras, construye un índice BM25 sobre el texto del paper dividido en
chunks y recupera solo los fragmentos más relevantes para cada figura
(usando el caption como query).

Ventajas sobre v1:
  - Contexto específico por figura, no genérico
  - Preserva detalles exactos (valores, métodos) que el resumen perdería
  - Más señal, menos ruido por figura

Uso:
    python analyze_figures_v2_rag.py extracted/figures.json --pdf paper.pdf
    python analyze_figures_v2_rag.py extracted/figures.json --context-file paper.txt
    python analyze_figures_v2_rag.py extracted/figures.json --pdf paper.pdf --top-k 8

Requiere (opcional, mejora la retrieval):
    pip install rank-bm25
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_SERVER         = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MAX_TOKENS     = 1500
DEFAULT_TEMPERATURE    = 0.0
DEFAULT_TIMEOUT        = 300
DEFAULT_CHUNK_WORDS    = 250    # palabras por chunk
DEFAULT_CHUNK_OVERLAP  = 40     # palabras de overlap entre chunks
DEFAULT_TOP_K          = 5      # chunks recuperados por figura
MAX_RETRIES            = 5
RETRY_BACKOFF          = 3


# ─── Helpers de prompt ───────────────────────────────────────────────────────

def _fmt_abstract(text: str, max_words: int = 350) -> str:
    """Extrae las primeras max_words del texto del paper como bloque de contexto."""
    if not text or not text.strip():
        return ""
    snippet = " ".join(text.split()[:max_words])
    return f"PAPER CONTEXT (opening section):\n{snippet}\n\n"


def _parse_json(text: str) -> dict | None:
    """Intenta extraer un JSON válido de la respuesta del LLM."""
    clean = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except Exception:
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ─── Prompts ─────────────────────────────────────────────────────────────────

PROMPT_INFERENCE_FIG_TEMPLATE = """{abstract_block}Figure caption: {caption}

Analyze this scientific figure. Respond ONLY with valid JSON — no markdown fences, no text outside the JSON object:

{{
  "figure_type": "bar chart | scatter plot | line graph | Western blot | microscopy | heatmap | survival curve | flow cytometry | schematic | other",
  "visual_description": "panels, axes, labels, colors, units, legends, organisms/structures shown. 2-3 sentences.",
  "groups_compared": "conditions, treatments, timepoints, genotypes or cell lines contrasted in the figure.",
  "statistical_markers": "every visible statistical element: n=, error bars type (SD/SEM/95%CI), p-values, R², fold-changes. Write 'None visible' if absent.",
  "key_finding": "the main result shown, citing specific numbers from the image when visible.",
  "caption_accurate": true,
  "caption_discrepancy": "elements in image not described by caption, or caption claims not supported visually. Write 'None' if accurate.",
  "scientific_interpretation": "what biological or scientific question this figure addresses and what the data demonstrates. Be specific about mechanism, pathway, or phenomenon.",
  "confidence": "high | medium | low"
}}

confidence: high=evidence clearly readable in image; medium=partially visible; low=inferring beyond what is shown.
If multi-panel (A, B, C...), address all panels in visual_description and key_finding.
Never refuse — all figures must be analyzed."""

PROMPT_INFERENCE_TBL_TEMPLATE = """{abstract_block}Table caption: {caption}

Analyze this scientific table. Respond ONLY with valid JSON — no markdown fences, no text outside the JSON object:

{{
  "table_type": "results comparison | parameter table | patient demographics | statistical summary | ablation study | other",
  "structure": "columns and rows described: what is compared, against what baselines, using what metric and units.",
  "statistical_markers": "significance markers (*, **, ***), p-values, CIs, n=, standard deviations visible. Write 'None visible' if absent.",
  "best_result": "the row or cell showing the strongest or most notable result, with its exact value.",
  "key_pattern": "the main trend, comparison or contrast that stands out across the table.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between caption and actual table content. Write 'None' if accurate.",
  "scientific_interpretation": "what question this table addresses and what the numbers prove. Cite specific values.",
  "confidence": "high | medium | low"
}}

confidence: high=all values clearly readable; medium=some cells hard to read; low=inferring.
Never refuse — all tables must be analyzed."""


PROMPT_ANCHORED_FIG_TEMPLATE = """{abstract_block}RELEVANT PAPER SECTIONS (BM25-retrieved for this figure):
{context}

---

Figure caption: {caption}

Analyze this figure using the retrieved paper sections as authoritative reference. Respond ONLY with valid JSON — no markdown fences, no text outside the JSON object:

{{
  "figure_type": "bar chart | scatter | Western blot | microscopy | heatmap | survival curve | other",
  "visual_description": "panels, axes, labels, colors, units, organisms visible. 2-3 sentences.",
  "hypothesis_tested": "the specific claim or hypothesis from the paper this figure tests. Quote the relevant sentence from the retrieved sections.",
  "paper_quote": "exact sentence from the retrieved text that this figure is meant to support.",
  "key_findings": "specific quantitative results visible in the figure, connected to numbers mentioned in the retrieved text.",
  "statistical_markers": "every visible statistical element: n=, error bars (SD/SEM/CI), p-values, R², significance markers. Write 'None visible' if absent.",
  "controls_assessment": "experimental controls present (positive, negative, baseline). Note absent controls given the experimental design in the retrieved sections.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between caption and visual. Write 'None' if accurate.",
  "scientific_conclusion": "what this figure definitively demonstrates, what alternative explanations it rules out, and its role in the paper's argument. Let the conclusion be earned by the evidence.",
  "confidence": "high | medium | low"
}}

confidence: high=clear visual evidence + strong paper alignment; medium=partial; low=inferring.
Never refuse."""

PROMPT_ANCHORED_TBL_TEMPLATE = """{abstract_block}RELEVANT PAPER SECTIONS (BM25-retrieved for this table):
{context}

---

Table caption: {caption}

Analyze this table using the retrieved paper sections as authoritative reference. Respond ONLY with valid JSON — no markdown fences, no text outside the JSON object:

{{
  "table_type": "results comparison | parameter table | patient demographics | statistical summary | ablation | other",
  "structure": "what is compared, by what metric, against what baselines.",
  "paper_quote": "exact sentence from the retrieved text that this table is meant to support.",
  "key_entries": "most relevant rows and values given the paper's claims. Cite specific numbers.",
  "statistical_markers": "significance markers (*, **, ***), p-values, CIs, n= visible. Write 'None visible' if absent.",
  "controls_assessment": "baseline or reference conditions used. Note missing controls given the retrieved context.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between caption and actual table content. Write 'None' if accurate.",
  "scientific_conclusion": "what this table definitively demonstrates, what it rules out, its role in the paper's argument.",
  "confidence": "high | medium | low"
}}

confidence: high=values clearly readable + strong paper alignment; medium=partial; low=inferring.
Never refuse."""


# ─── HTTP client ─────────────────────────────────────────────────────────────
def ask_api(server, prompt, image_bytes=None, max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT):
    content = []
    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "messages":    [{"role": "user", "content": content}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(server, json=payload, timeout=timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if not text:
                raise ValueError("empty response")
            return text
        except Exception as e:
            last_err = e
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"    retry {attempt + 1}/{MAX_RETRIES}: {e} ({wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed after {MAX_RETRIES}: {last_err}")


def server_health(server):
    try:
        url = server.rsplit("/v1/", 1)[0] + "/health"
        return requests.get(url, timeout=5).status_code == 200
    except Exception:
        return False


# ─── Text extraction ──────────────────────────────────────────────────────────
def extract_paper_text(pdf_path=None, context_file=None):
    """Extrae texto del PDF o lo lee desde un archivo .txt pre-extraído."""
    if context_file:
        return Path(context_file).read_text(encoding="utf-8")
    doc = fitz.open(str(pdf_path))
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"[Page {i + 1}]\n{text}")
    doc.close()
    return "\n\n".join(parts)


# ─── RAG: chunking ───────────────────────────────────────────────────────────
def _tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def chunk_text(text, chunk_words=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Divide el texto en chunks por párrafos, fusionando los pequeños y
    dividiendo los grandes. Aplica overlap entre chunks consecutivos.
    """
    # Dividir por párrafo (doble salto de línea)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    chunks = []
    current_words = []

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        # Si el párrafo solo es demasiado grande, lo dividimos
        if len(words) > chunk_words * 1.5:
            for start in range(0, len(words), chunk_words - overlap):
                segment = words[start:start + chunk_words]
                if len(segment) < 20:
                    continue
                chunks.append(" ".join(segment))
            continue

        # Acumular hasta llegar al tamaño objetivo
        if len(current_words) + len(words) > chunk_words:
            if current_words:
                chunks.append(" ".join(current_words))
            # Mantener overlap del chunk anterior
            current_words = current_words[-overlap:] + words
        else:
            current_words.extend(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ─── RAG: índice BM25 (con fallback TF-IDF simple) ──────────────────────────
def _build_bm25(tokenized_chunks):
    """BM25 usando rank_bm25 si está disponible, o TF-IDF simple como fallback."""
    try:
        from rank_bm25 import BM25Okapi
        index = BM25Okapi(tokenized_chunks)

        def score_fn(query_tokens):
            return index.get_scores(query_tokens)

        return score_fn

    except ImportError:
        # Fallback: TF simple + IDF aproximado
        n_docs = len(tokenized_chunks)
        df = {}
        for tokens in tokenized_chunks:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        def score_fn(query_tokens):
            scores = []
            for tokens in tokenized_chunks:
                tf_map = {}
                for t in tokens:
                    tf_map[t] = tf_map.get(t, 0) + 1
                score = 0.0
                for qt in query_tokens:
                    if qt in tf_map:
                        tf  = tf_map[qt] / max(len(tokens), 1)
                        idf = math.log((n_docs + 1) / (df.get(qt, 0) + 1)) + 1
                        score += tf * idf
                scores.append(score)
            return scores

        return score_fn


def build_index(chunks):
    """Construye el índice de retrieval sobre la lista de chunks."""
    tokenized = [_tokenize(c) for c in chunks]
    score_fn  = _build_bm25(tokenized)
    return score_fn, tokenized


def retrieve(query, score_fn, chunks, top_k=DEFAULT_TOP_K):
    """Retorna los top_k chunks más relevantes para la query."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_k]

    scores = score_fn(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    # Mantener orden original de aparición (más legible para el LLM)
    top_indices = sorted([i for i, _ in ranked[:top_k]])
    return [chunks[i] for i in top_indices]


# ─── RAG: pipeline de contexto ───────────────────────────────────────────────
def build_rag_index(pdf_path=None, context_file=None,
                    chunk_words=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_CHUNK_OVERLAP):
    """Extrae texto, divide en chunks y construye el índice BM25."""
    src = context_file if context_file else str(pdf_path)
    print(f"Construyendo índice RAG: {src}")

    text   = extract_paper_text(pdf_path=pdf_path, context_file=context_file)
    chunks = chunk_text(text, chunk_words=chunk_words, overlap=overlap)

    print(f"  Texto: {len(text):,} chars | Chunks: {len(chunks)} ({chunk_words} palabras c/u, {overlap} overlap)")

    score_fn, _ = build_index(chunks)
    return chunks, score_fn


# ─── Pipeline ────────────────────────────────────────────────────────────────
def analyze_all(figures_json, pdf_path=None, context_file=None, server=DEFAULT_SERVER,
                mode_inference=True, mode_anchored=True,
                chunk_words=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_CHUNK_OVERLAP,
                top_k=DEFAULT_TOP_K, max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT, out_path=None,
                tables_json=None):

    fig_json = Path(figures_json)
    if pdf_path:
        pdf_path = Path(pdf_path)

    meta  = json.loads(fig_json.read_text(encoding="utf-8"))
    items = list(meta["items"])

    # Merge tables from extract_tables.py if provided
    if tables_json:
        tbl_path = Path(tables_json)
        if tbl_path.exists():
            tbl_meta = json.loads(tbl_path.read_text(encoding="utf-8"))
            items.extend(tbl_meta["items"])

    if out_path is None:
        out_path = fig_json.parent / "analyses_rag.json"
    else:
        out_path = Path(out_path)

    # Resume
    results = []
    done = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for it in prev.get("items", []):
                has_inf  = "inference" in it
                has_anch = "anchored"  in it
                ok = (not mode_inference or has_inf) and (not mode_anchored or has_anch)
                if ok:
                    results.append(it)
                    done.add(it["label"])
        except Exception:
            pass

    n_figs = sum(1 for it in items if it.get("kind") != "table")
    n_tbls = sum(1 for it in items if it.get("kind") == "table")
    print(f"\nServer: {server}")
    print(f"Items: {len(items)} ({n_figs} figuras + {n_tbls} tablas) | hechos: {len(done)}")
    print(f"Modos: inference={mode_inference} anchored={mode_anchored} | top_k={top_k}\n")

    # Extraer abstract (primeras ~350 palabras) para inyectar en todos los prompts
    abstract_block = ""
    try:
        if context_file and Path(context_file).exists():
            raw = Path(context_file).read_text(encoding="utf-8", errors="ignore")
        elif pdf_path:
            raw = extract_paper_text(pdf_path=pdf_path)
        else:
            raw = ""
        abstract_block = _fmt_abstract(raw)
        if abstract_block:
            print(f"  Abstract block: {len(abstract_block.split())} palabras inyectadas en prompts")
    except Exception as e:
        print(f"  WARN abstract: {e}")

    # Construir índice RAG una sola vez
    chunks   = None
    score_fn = None
    if mode_anchored and (pdf_path or context_file):
        chunks, score_fn = build_rag_index(
            pdf_path=pdf_path, context_file=context_file,
            chunk_words=chunk_words, overlap=overlap,
        )
        print()

    for i, item in enumerate(items):
        if item["label"] in done:
            print(f"[{i + 1}/{len(items)}] {item['label']} SKIP")
            continue

        img_path = Path(item["image_path"])
        if not img_path.is_absolute():
            img_path = fig_json.parent / img_path.name
        img_bytes = img_path.read_bytes()

        is_table = item["kind"] == "table"
        caption  = item.get("caption", "Not provided.")
        result   = dict(item)
        t_start  = time.time()

        # 1) Inferencia pura
        if mode_inference:
            tmpl_inf   = PROMPT_INFERENCE_TBL_TEMPLATE if is_table else PROMPT_INFERENCE_FIG_TEMPLATE
            prompt_inf = tmpl_inf.format(caption=caption, abstract_block=abstract_block)
            try:
                ans = ask_api(server, prompt_inf, image_bytes=img_bytes,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout)
                result["inference"] = ans
                parsed = _parse_json(ans)
                if parsed:
                    result["inference_parsed"] = parsed
                    result["confidence"]       = parsed.get("confidence", "unknown")
            except Exception as e:
                result["inference_error"] = str(e)

        # 2) Anchored con RAG
        if mode_anchored and chunks:
            retrieved  = retrieve(caption, score_fn, chunks, top_k=top_k)
            context    = "\n\n---\n\n".join(retrieved)
            tmpl       = PROMPT_ANCHORED_TBL_TEMPLATE if is_table else PROMPT_ANCHORED_FIG_TEMPLATE
            prompt_anc = tmpl.format(context=context, caption=caption, abstract_block=abstract_block)

            try:
                ans = ask_api(server, prompt_anc, image_bytes=img_bytes,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout)
                result["anchored"]         = ans
                result["retrieved_chunks"] = len(retrieved)
                result["retrieved_words"]  = sum(len(c.split()) for c in retrieved)
                parsed = _parse_json(ans)
                if parsed:
                    result["anchored_parsed"] = parsed
                    # confidence from anchored overwrites (more context = more reliable)
                    result["confidence"]      = parsed.get("confidence", result.get("confidence", "unknown"))
            except Exception as e:
                result["anchored_error"] = str(e)

        result["elapsed_sec"]  = round(time.time() - t_start, 1)
        result["context_mode"] = "rag"
        results.append(result)

        preview = (result.get("inference") or result.get("anchored") or "")[:90].replace("\n", " ")
        chunks_info = f" [{result.get('retrieved_chunks', '?')} chunks]" if mode_anchored else ""
        print(f"[{i + 1}/{len(items)}] {item['label']} p{item['page']} ({result['elapsed_sec']}s){chunks_info} {preview}...")

        out_path.write_text(
            json.dumps({
                "total":        len(results),
                "context_mode": "rag",
                "top_k":        top_k,
                "chunk_words":  chunk_words,
                "items":        results,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nResultados: {out_path}")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Analiza figuras con RAG: retrieval de chunks relevantes por figura.")
    p.add_argument("figures_json",   help="figures.json de extract_figures.py")
    p.add_argument("--pdf",          help="PDF original")
    p.add_argument("--context-file", help="Texto plano pre-extraído del paper (.txt)")
    p.add_argument("--server",       default=DEFAULT_SERVER)
    p.add_argument("--inference-only",  action="store_true")
    p.add_argument("--anchored-only",   action="store_true")
    p.add_argument("--out",          help="ruta de salida JSON")
    p.add_argument("--top-k",        type=int, default=DEFAULT_TOP_K,
                   help=f"chunks recuperados por figura (def: {DEFAULT_TOP_K})")
    p.add_argument("--chunk-words",  type=int, default=DEFAULT_CHUNK_WORDS,
                   help=f"palabras por chunk (def: {DEFAULT_CHUNK_WORDS})")
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                   help=f"palabras de overlap entre chunks (def: {DEFAULT_CHUNK_OVERLAP})")
    p.add_argument("--max-tokens",   type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature",  type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--timeout",      type=int, default=DEFAULT_TIMEOUT)
    args = p.parse_args(argv)

    if not server_health(args.server):
        sys.exit(f"Servidor no responde: {args.server}")

    mode_inf = not args.anchored_only
    mode_anc = not args.inference_only

    if mode_anc and not args.pdf and not args.context_file:
        sys.exit("Requiere --pdf o --context-file para anchored. Usa --inference-only si no tenés el paper.")

    analyze_all(
        args.figures_json,
        pdf_path=args.pdf,
        context_file=args.context_file,
        server=args.server,
        mode_inference=mode_inf,
        mode_anchored=mode_anc,
        chunk_words=args.chunk_words,
        overlap=args.chunk_overlap,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
