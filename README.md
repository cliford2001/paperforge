# PaperForge

PaperForge extracts scientific figures and tables from PDFs with **Docling**, builds paper-aware context, and analyzes each visual item with a vision-language model using a structured dynamic prompt.

The goal is not only to describe an image. The goal is to explain what each figure/table shows, how it supports the paper's argument, what evidence is direct, what comes from context, and what is only an extra model inference.

---

## Dynamic Prompt Structure

```text
                                Dynamic Prompt Structure - PaperForge
                 Per-figure/table multimodal analysis: Docling + context + JSON output


┌──────────────────────┐
│ SQLite / DB / parquet │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐        ┌──────────────────────┐
│ DOI + full text       ├───────►│ paper_context.txt    │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│ PDF download          │        │ paper global context │
│ legal fallback chain  │        │ abstract + full map  │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│ DOCLING extraction    │        │ local retrieval      │
│ figures + tables      │        │ exact refs + BM25    │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           ▼                               │
┌──────────────────────┐                   │
│ Figure/Table PNG      │                   │
│ caption + bbox/page   │                   │
└──────────┬───────────┘                   │
           │                               │
           ▼                               │
┌──────────────────────┐                   │
│ Filtered OCR summary  │                   │
│ never raw OCR         │                   │
└──────────┬───────────┘                   │
           │                               │
           └───────────────┬───────────────┘
                           │  text + image packet
                           ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │                           DYNAMIC PROMPT                             │
        ├──────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  ① IMAGE PNG                                           [always]      │
        │     Primary evidence. Sent as image/base64.                          │
        │                                                                      │
        │  ② PAPER GLOBAL CONTEXT                                [if text]     │
        │     Abstract, research question, system, methods,                    │
        │     main findings, global conclusion, figure map.                    │
        │                                                                      │
        │  ③ FIGURE / TABLE LOCAL CONTEXT                        [if avail.]   │
        │     Official caption, exact Fig/Table mentions, nearby paragraphs,   │
        │     same-section text, BM25 windows, cross-figure references.        │
        │                                                                      │
        │  ④ FILTERED OCR SUMMARY                                [optional]    │
        │     Only labels, axes, units, conditions, genes/proteins, p-values,  │
        │     sample sizes. Broken OCR and isolated numbers are ignored.       │
        │                                                                      │
        │  ───────────────── STRUCTURED ANALYSIS SECTIONS ─────────────────   │
        │                                                                      │
        │  Visual Description                                    [always]      │
        │  Figure / Table Type                                  [always]      │
        │  Experimental Design                                  [always]      │
        │  Statistical Markers                                  [always]      │
        │  Data and Patterns                                    [always]      │
        │  Caption Alignment                                    [always]      │
        │  Evidence Separation                                  [always]      │
        │  Scientific Interpretation                            [always]      │
        │                                                                      │
        │  Hypothesis Tested                                   [only RAG]     │
        │  Controls Assessment                                 [only RAG]     │
        │  Paper Quote                                         [only RAG]     │
        │                                                                      │
        │  Scientific Conclusion                              [synthesis]    │
        │  Model Extra Inference                               [separated]    │
        │                                                                      │
        │  Respond ONLY with JSON.                                             │
        │                                                                      │
        └──────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       │ + image PNG
                                       ▼
                              ┌──────────────────┐
                              │     VLM / LLM    │
                              │ InternVL / Qwen  │
                              └────────┬─────────┘
                                       │
                                       ▼
                         ┌────────────────────────────┐
                         │ analysis_parsed { ... }    │
                         │ structured JSON per item   │
                         └────────┬───────────────────┘
                                  │
                                  ▼
                         ┌────────────────────────────┐
                         │ paper_summary { ... }      │
                         │ final text-only synthesis  │
                         └────────────────────────────┘

Legend:
  [always]     included for every item
  [if text]    included when paper text/context exists
  [if avail.]  included when local retrieval finds useful passages
  [only RAG]   included only when retrieved/full paper context exists
  [optional]   included only after OCR filtering/classification
```

---

## Structured JSON Output

Each visual item in `analyses_rag.json` keeps the original metadata plus the raw model response and parsed JSON.

```json
{
  "label": "Figure 2",
  "kind": "figure",
  "page": 4,
  "caption": "...",
  "image_path": "p004_Figure_2.png",
  "analysis": "raw model response",
  "analysis_parsed": {
    "item_identity": {},
    "visual_form": {},
    "experimental_design": {},
    "markers_and_statistics": {},
    "evidence_separation": {},
    "scientific_findings": [],
    "paper_level_interpretation": {},
    "scientific_conclusion": "...",
    "model_extra_inference": {
      "extra_inference": "...",
      "support_status": "image_supported | caption_supported | context_supported | speculative | none",
      "supporting_evidence": "...",
      "risk": "low | medium | high",
      "open_questions": "...",
      "alternative_interpretation": "..."
    },
    "quality_and_limits": {},
    "confidence": "high | medium | low"
  }
}
```

`scientific_conclusion` is the supported conclusion from image, caption, and paper context.

`model_extra_inference` is deliberately separated. It may go beyond the paper, but must include support status, evidence, and risk.

---

## Main Pipeline

```bash
python v2/main.py \
  --input-parquet sample5.parquet \
  --out-dir results_extract \
  --keep-pdf \
  --reuse-existing-pdf
```

With VLM analysis:

```bash
python v2/main.py \
  --input-parquet sample5.parquet \
  --out-dir results_model_run \
  --keep-pdf \
  --reuse-existing-pdf \
  --run-analysis \
  --server http://127.0.0.1:8097/v1/chat/completions \
  --context-mode bm25 \
  --top-k 6 \
  --chunk-words 250 \
  --chunk-overlap 40 \
  --max-tokens 1200 \
  --temperature 0.0
```

---

## Context Modes

| Mode | Input to prompt | Use |
|---|---|---|
| `none` | image + caption + abstract when available | pure visual/caption baseline |
| `bm25` | image + caption + top retrieved chunks | efficient grounded baseline |
| `full` | image + caption + full/trimmed paper text | broad context baseline |

The planned `layered` mode will add a paper-level context map plus per-figure context packets.

---

## Important Files

| File | Purpose |
|---|---|
| `v2/main.py` | End-to-end pipeline entrypoint |
| `v2/pdfs.py` | PDF download wrapper |
| `extract_figures.py` | Docling/PyMuPDF figure and table image extraction |
| `extract_tables.py` | Table extraction and normalization |
| `analyze_figures_v2_rag.py` | Dynamic prompt and VLM analysis |
| `v2/analyzer.py` | v2 wrapper around the VLM analyzer |

---

## Current Extraction Behavior

Docling is the primary extractor. Small uncaptained Docling picture artifacts, such as publisher logos/icons, are filtered before writing figure outputs. Skipped artifacts are audited in `skipped_figures.json`.

OCR from crops is treated as auxiliary structured evidence only. Raw OCR text should not be inserted directly into the prompt.
