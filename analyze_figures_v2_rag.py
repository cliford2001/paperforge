"""
LLM Analyzer for Extracted Figures — v2 RAG
=============================================

Analiza figuras Y tablas científicas con dos modos de contexto:

  context_strategy="full"  (por defecto)
      Inyecta el texto COMPLETO del paper como contexto, truncado solo si
      supera MAX_CONTEXT_WORDS (0 = sin límite). Ideal para GPUs con ctx
      grande (≥ 32 768 tokens).

  context_strategy="bm25"
      Recupera los top_k chunks más relevantes por figura/tabla usando BM25
      (rank_bm25 o fallback TF-IDF). Mejor para GPUs pequeñas o papers muy
      largos.

En ambos modos:
  - El abstract completo (o primeras ABSTRACT_MAX_WORDS palabras) se inyecta
    en los prompts de INFERENCIA pura también.
  - Las tablas y figuras comparten el mismo pipeline (kind="table" activa
    los templates especializados).

Uso:
    python analyze_figures_v2_rag.py extracted/figures.json --pdf paper.pdf
    python analyze_figures_v2_rag.py figures.json --context-file paper.txt --context-strategy full
    python analyze_figures_v2_rag.py figures.json --pdf paper.pdf --context-strategy bm25 --top-k 8
    python analyze_figures_v2_rag.py figures.json --pdf paper.pdf --max-context-words 6000

Requiere (para estrategia BM25):
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


# ─── Defaults ─────────────────────────────────────────────────────────────────
# Estos parámetros se pueden sobreescribir por CLI o por run_analysis_batch.py.
# Ajustar según la GPU disponible:
#   GTX 1080 Ti (11 GB, ctx 12 288) → MAX_CONTEXT_WORDS ≈ 3000–4000
#   RTX 3090 / A100  (ctx 32 768)   → MAX_CONTEXT_WORDS = 0 (sin límite)

DEFAULT_SERVER            = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MAX_TOKENS        = 1500       # tokens de respuesta del LLM
DEFAULT_TEMPERATURE       = 0.0        # determinista; evita alucinaciones
DEFAULT_TIMEOUT           = 300        # segundos por llamada
DEFAULT_CONTEXT_STRATEGY  = "bm25"     # "bm25" (default, testeo) | "full" (GPU grande)
DEFAULT_MAX_CONTEXT_WORDS = 0          # 0 = sin límite; >0 = truncar texto completo
DEFAULT_ABSTRACT_WORDS    = 0          # 0 = abstract completo detectado por sección
DEFAULT_CHUNK_WORDS       = 250        # tamaño de chunk en modo bm25
DEFAULT_CHUNK_OVERLAP     = 40         # overlap entre chunks en modo bm25
DEFAULT_TOP_K             = 10         # chunks BM25 recuperados por ítem (Lewis et al. 2020 usa 5; subimos a 10 por ventanas mayores)
MAX_RETRIES               = 5
RETRY_BACKOFF             = 3

# ─── Configuración recomendada por GPU ────────────────────────────────────────
#
#  GTX 1080 Ti  (11 GB, ctx 12 288):
#    --context-strategy full --max-context-words 3000
#
#  RTX 3090 / RTX 4090  (ctx 32 768):
#    --context-strategy full --max-context-words 0
#
#  A100 / H100  (ctx 128 000+):
#    --context-strategy full --max-context-words 0
#
#  Papers muy largos en GPU chica:
#    --context-strategy bm25 --top-k 10
#
# Referencia BM25/RAG: Lewis et al. 2020 (DOI: 10.48550/arXiv.2005.11401)
# ─────────────────────────────────────────────────────────────────────────────


# ─── Helpers de prompt ────────────────────────────────────────────────────────

def _extract_abstract(text: str, fallback_words: int = 400) -> str:
    """
    Extrae la sección Abstract real del paper buscando el encabezado 'Abstract'
    y cortando en el siguiente encabezado de sección (Introduction, Methods, etc.).
    Fallback: primeras fallback_words palabras si no se encuentra.
    """
    # Encabezados que marcan el fin del abstract
    END_SECTIONS = re.compile(
        r'^\s*(introduction|background|methods?|materials?\s+and\s+methods?|'
        r'results?|discussion|keywords?|1[\.\s]|2[\.\s])',
        re.IGNORECASE | re.MULTILINE,
    )
    # Buscar inicio del abstract
    start_m = re.search(r'(?:^|\n)\s*abstract\s*\n', text, re.IGNORECASE)
    if start_m:
        body = text[start_m.end():]
        end_m = END_SECTIONS.search(body)
        snippet = body[:end_m.start()].strip() if end_m else body[:3000].strip()
        if snippet:
            return snippet
    # Fallback: primeras N palabras
    return " ".join(text.split()[:fallback_words])


def _fmt_abstract(text: str, max_words: int = 0) -> str:
    """
    Formatea el abstract real del paper para inyectarlo en prompts.
    max_words ignorado — se usa detección de sección.
    """
    if not text or not text.strip():
        return ""
    snippet = _extract_abstract(text)
    return f"PAPER ABSTRACT:\n{snippet}\n\n"


def _fmt_full_context(text: str, max_words: int = 0) -> str:
    """
    Formatea el texto completo del paper para el modo anchored=full.
    max_words=0 → sin límite.
    """
    if not text or not text.strip():
        return ""
    words = text.split()
    total = len(words)
    if max_words > 0 and total > max_words:
        words = words[:max_words]
        truncated = True
    else:
        truncated = False
    snippet = " ".join(words)
    note = f" [truncated at {max_words} words, original: {total}]" if truncated else f" [{len(words)} words]"
    return snippet, note


def _parse_json(text: str) -> dict | None:
    """Extrae JSON válido de la respuesta del LLM (maneja markdown fences)."""
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


# ─── Prompts ──────────────────────────────────────────────────────────────────
# Un solo prompt por tipo (figura / tabla).
# Orden anti "lost in the middle" (Liu et al. 2023, arXiv:2307.03172):
#
#   A. caption          → inicio  (el modelo atiende el comienzo del contexto)
#   B. chunk más relevante (BM25 rank-1) → justo después del caption
#   C. chunks de apoyo  → medio  (orden documental, para coherencia)
#   D. chunk rank-2     → último del bloque RAG  (el modelo atiende el final)
#   E. abstract         → cierre del contexto, antes de las instrucciones
#
# La salida siempre es JSON estructurado.

PROMPT_FIG_TEMPLATE = """\
Figure caption: {caption}
{context_block}\
PAPER ABSTRACT:
{abstract}

You are generating high-quality training data for a vision-language model. \
Using the abstract as context for what this paper investigates{context_hint}, \
analyze this figure systematically following each aspect below:

## Visual Description
What is visible: panels, axes, colors, labels, legends, units, symbols, \
organisms or structures shown. 2-4 sentences. \
If multi-panel (A, B, C...), describe each panel's content.

## Figure Type
Type of visualization (bar chart, scatter plot, line graph, Western blot, \
heatmap, microscopy image, survival curve, flow cytometry, schematic, etc.) \
and what experimental data it represents.

## Experimental Design
Identify the experimental system, groups compared, treatments or perturbations, \
controls, measured variables, assay or method, and the hypothesis or question \
tested by this figure. If any element is not visible or not stated in the \
caption/context, write "Not determinable".

## Statistical Markers
Every statistical element visible in the image: sample sizes (n=), \
error bars (SD, SEM, 95% CI), p-values, R², fold-changes, significance \
markers (*, **, ***). \
Write "None visible" if absent. Never infer or assume values not shown.

## Data and Patterns
Specific values, trends, comparisons and relationships visible in the image. \
Cite numbers directly from the figure. \
Identify the groups, conditions, timepoints or genotypes being compared.

## Caption Alignment
Does the caption accurately describe what is shown? \
Note discrepancies: visual elements absent from the caption, \
or caption claims not supported by what is visible.

## Evidence Separation
Separate direct visual evidence, caption-supported evidence, paper-context \
supported interpretation, and unsupported or not-determinable claims. \
Do not mix paper-grounded conclusions with model speculation.

## Scientific Interpretation
What biological or scientific question does this figure address, \
given what the abstract says this paper investigates? \
What does the data demonstrate? \
Be specific about mechanism, pathway, or phenomenon — \
let the abstract inform your interpretation without inventing unseen data.
{anchored_sections}
## Scientific Conclusion
Write a single cohesive paragraph of 4–6 sentences that unifies your findings \
from the sections above. \
The paragraph must explicitly integrate: the visual patterns and layout \
(Visual Description), the specific values and group comparisons \
(Data and Patterns), the statistical strength or absence thereof \
(Statistical Markers), and the scientific meaning (Scientific Interpretation). \
If RAG context was available, also incorporate the hypothesis this figure tests \
and the adequacy of the experimental controls. \
End by stating clearly what this figure definitively demonstrates, \
what it rules out, and its role in the paper's overall argument. \
Write as a unified inference paragraph — not a bullet list, not a summary of \
the sections. The reader should be able to understand the figure's contribution \
to the paper from this paragraph alone.

## Model Extra Inference
This is optional analysis outside the paper's explicit claims. \
It must be clearly separated from the supported scientific conclusion.

- Extra inference: what might this data suggest beyond the paper's explicit \
claims? Ground it in visible/caption/context evidence when possible.
- Open questions: what scientific questions does this figure raise that \
the paper does not address or resolve?
- Alternative interpretation: propose one alternative valid reading of \
this data — a different mechanism, confound, or explanation consistent \
with the visual evidence. Write "None" if the data is unambiguous.
- Support status: mark whether this inference is image_supported, \
caption_supported, context_supported, or speculative.

Respond ONLY with valid JSON — no markdown fences, no text outside the JSON object:

{{
  "figure_type": "bar chart | scatter plot | line graph | Western blot | microscopy | heatmap | survival curve | flow cytometry | schematic | other",
  "visual_form": {{
    "graph_or_visual_type": "bar_plot | line_plot | scatter_plot | heatmap | microscopy | pathway_diagram | schematic | table | gel_blot | map | multi_panel_composite | other | unclear",
    "panel_count": "number or 'unclear'",
    "panel_labels": ["A", "B"],
    "axes_or_dimensions": ["axis labels, dimensions or columns visible"],
    "legend_elements": ["legend items visible"],
    "visible_entities": ["species, genes, proteins, metabolites, treatments, tissues, timepoints, doses, conditions visible"]
  }},
  "visual_description": "panels, axes, colors, units and structures visible. 2-4 sentences, one per panel if multi-panel.",
  "experimental_design": {{
    "hypothesis_or_question_tested": "what this figure appears to test. 'Not determinable' if unavailable.",
    "experimental_groups": ["groups, genotypes, treatments, cohorts or conditions compared"],
    "controls": ["positive, negative, vehicle, untreated, baseline or reference controls visible/stated"],
    "perturbations_or_treatments": ["drug, knockout, dose, timepoint, environmental condition, etc."],
    "measured_variables": ["dependent variables or readouts"],
    "assay_or_method": "assay, imaging method, sequencing, qPCR, western blot, etc. 'Not determinable' if unavailable.",
    "biological_or_experimental_system": "cell type, organism, tissue, patient cohort, model system, etc."
  }},
  "statistical_markers": "exhaustive extraction of ALL quantitative statistical data visible in the image: sample sizes (n=X per group), error bar type and magnitude (SD/SEM/95%CI with values if legible), p-values and significance markers (exact values or */**, report per comparison), effect sizes (fold-change, Cohen d, OR/HR/RR), regression metrics (R², slope, r), test statistics (F, t, chi2, Z). Quote exact numbers from the image when readable. Format: 'n=12/group; error bars=SEM; p<0.001 (A vs B), p=0.03 (A vs C); 2.4-fold increase'. Write 'None visible' ONLY if the image contains zero statistical annotation.",
  "markers_and_statistics": {{
    "statistical_markers": ["p-values, confidence intervals, error bars, significance letters, asterisks, regression/correlation, fold-change, sample size"],
    "sample_sizes": ["n values if visible/stated"],
    "units": ["units visible/stated"],
    "effect_directions": ["increase, decrease, no change, association direction"],
    "quantitative_values": ["only exact values explicitly visible or stated in caption/context"]
  }},
  "data_and_patterns": "specific values and trends visible. cite numbers from the image. identify groups compared.",
  "groups_compared": "conditions, treatments, timepoints, genotypes or cell lines contrasted.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between image and caption. 'None' if accurate.",
  "evidence_separation": {{
    "direct_visual_evidence": ["claims supported by visible image only"],
    "caption_supported_evidence": ["claims supported by caption"],
    "local_context_supported_evidence": ["claims supported by retrieved/full paper context"],
    "global_paper_supported_interpretation": ["claims supported by abstract/paper-level context"],
    "unsupported_or_not_determinable": ["claims that cannot be determined"]
  }},
  "scientific_interpretation": "what question this figure answers given the paper's topic. mechanism or phenomenon demonstrated.",{anchored_json}
  "scientific_conclusion": "unified synthesis paragraph (4-6 sentences) integrating visual evidence, statistics, interpretation{anchored_hint}. What this figure definitively demonstrates, what it rules out, and its role in the paper's argument. Written as flowing prose, not a list.",
  "model_extra_inference": {{
    "extra_inference": "optional interpretation beyond the paper's explicit claims. Keep separate from scientific_conclusion.",
    "support_status": "image_supported | caption_supported | context_supported | speculative | none",
    "supporting_evidence": "what visible/caption/context evidence supports the inference. 'None' if speculative.",
    "risk": "low | medium | high",
    "open_questions": "scientific questions this figure raises that the paper does not address or resolve.",
    "alternative_interpretation": "one alternative valid reading of this data — different mechanism, confound, or explanation. 'None' if unambiguous."
  }},
  "context_used": "{context_used}",
  "confidence": "high | medium | low"
}}

confidence: high = evidence clearly readable in image{conf_hint}; medium = partially visible or ambiguous; low = inferring beyond what is shown.
Never refuse — all figures must be analyzed."""


PROMPT_TBL_TEMPLATE = """\
Table caption: {caption}
{context_block}\
PAPER ABSTRACT:
{abstract}

You are generating high-quality training data for a vision-language model. \
Using the abstract as context for what this paper investigates{context_hint}, \
analyze this table systematically following each aspect below:

## Table Description
What is compared, by what metric, against what baselines. \
Describe columns, rows, units, and scale.

## Table Type
Type of table (results comparison, ablation study, patient demographics, \
parameter table, statistical summary, etc.) \
and what experimental data it represents.

## Experimental Design
Identify the experimental system, groups compared, treatments or perturbations, \
controls, measured variables, assay or method, and the hypothesis or question \
tested by this table. If any element is not visible or not stated in the \
caption/context, write "Not determinable".

## Statistical Markers
Every statistical annotation visible: significance markers (*, **, ***), \
p-values, confidence intervals, sample sizes (n=), standard deviations. \
Write "None visible" if absent. Never infer values not shown.

## Key Entries
The most important rows, columns or cells given the paper's research question. \
Cite specific values. Identify the best result and any surprising entries.

## Patterns and Trends
The main trend, comparison or contrast that stands out across the table. \
What does the distribution of values reveal?

## Caption Alignment
Does the caption accurately describe what the table contains? \
Note discrepancies between actual content and what the caption states or implies.

## Evidence Separation
Separate direct table evidence, caption-supported evidence, paper-context \
supported interpretation, and unsupported or not-determinable claims. \
Do not mix paper-grounded conclusions with model speculation.

## Scientific Interpretation
What question does this table address, given what the abstract says this paper investigates? \
What do the numbers prove or argue? \
Be specific — cite values and connect them to the paper's claims.
{anchored_sections}
## Scientific Conclusion
Write a single cohesive paragraph of 4–6 sentences that unifies your findings \
from the sections above. \
The paragraph must explicitly integrate: the table structure and what is being \
compared (Table Description), the specific values and key entries \
(Key Entries + Patterns and Trends), the statistical annotations or their absence \
(Statistical Markers), and the scientific meaning (Scientific Interpretation). \
If RAG context was available, also incorporate the claim this table supports \
and the experimental design context. \
End by stating clearly what this table definitively demonstrates, \
what it rules out, and its role in the paper's overall argument. \
Write as a unified inference paragraph — not a bullet list, not a section recap. \
The reader should be able to understand the table's contribution to the paper \
from this paragraph alone.

## Model Extra Inference
This is optional analysis outside the paper's explicit claims. \
It must be clearly separated from the supported scientific conclusion.

- Extra inference: what might these values suggest beyond the paper's explicit \
claims? Ground it in visible table/caption/context evidence when possible.
- Open questions: what scientific questions does this table raise that \
the paper does not address or resolve?
- Alternative interpretation: propose one alternative valid reading of \
these results — a different explanation, confound, or mechanism consistent \
with the tabulated data. Write "None" if the data is unambiguous.
- Support status: mark whether this inference is table_supported, \
caption_supported, context_supported, or speculative.

Respond ONLY with valid JSON — no markdown fences, no text outside the JSON object:

{{
  "table_type": "results comparison | ablation study | patient demographics | parameter table | statistical summary | other",
  "visual_form": {{
    "graph_or_visual_type": "table",
    "panel_count": "number or 'unclear'",
    "panel_labels": ["A", "B"],
    "axes_or_dimensions": ["columns, rows, group dimensions or table sections visible"],
    "legend_elements": ["footnotes, legends or table annotations visible"],
    "visible_entities": ["species, genes, proteins, metabolites, treatments, tissues, timepoints, doses, conditions visible"]
  }},
  "structure": "what is compared, by what metric, against what baselines. units and scale.",
  "experimental_design": {{
    "hypothesis_or_question_tested": "what this table appears to test. 'Not determinable' if unavailable.",
    "experimental_groups": ["groups, genotypes, treatments, cohorts or conditions compared"],
    "controls": ["positive, negative, vehicle, untreated, baseline or reference controls visible/stated"],
    "perturbations_or_treatments": ["drug, knockout, dose, timepoint, environmental condition, etc."],
    "measured_variables": ["dependent variables or readouts"],
    "assay_or_method": "assay, statistical method, cohort comparison, model evaluation, etc. 'Not determinable' if unavailable.",
    "biological_or_experimental_system": "cell type, organism, tissue, patient cohort, model system, etc."
  }},
  "statistical_markers": "exhaustive extraction of ALL statistical data present in the table cells: sample sizes (n=), p-values (exact or bounded, per row/comparison), confidence intervals with bounds, means, medians, SDs, SEs, percentages, test statistics (F, t, chi2). Report specific cell-level values when legible. Format: 'n=45 control / 52 treatment; mean±SD: 12.3±2.1 vs 18.7±3.4; p=0.002; 95%CI [1.2-2.8]'. Write 'None visible' ONLY if the table contains zero statistical annotation.",
  "markers_and_statistics": {{
    "statistical_markers": ["p-values, confidence intervals, SD/SE, percentages, means/medians, test statistics, sample size"],
    "sample_sizes": ["n values if visible/stated"],
    "units": ["units visible/stated"],
    "effect_directions": ["increase, decrease, no change, association direction"],
    "quantitative_values": ["only exact values explicitly visible or stated in caption/context"]
  }},
  "key_entries": "most relevant rows/cells given the paper's claims. cite specific values.",
  "best_result": "the row or cell with the strongest or most notable result, with its exact value.",
  "patterns_and_trends": "main trend or contrast that stands out across the table.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between table content and caption. 'None' if accurate.",
  "evidence_separation": {{
    "direct_visual_evidence": ["claims supported by table values only"],
    "caption_supported_evidence": ["claims supported by caption"],
    "local_context_supported_evidence": ["claims supported by retrieved/full paper context"],
    "global_paper_supported_interpretation": ["claims supported by abstract/paper-level context"],
    "unsupported_or_not_determinable": ["claims that cannot be determined"]
  }},
  "scientific_interpretation": "what question this table answers. what the numbers prove. cite specific values.",{anchored_json}
  "scientific_conclusion": "unified synthesis paragraph (4-6 sentences) integrating table data, statistics, interpretation{anchored_hint}. What this table definitively demonstrates, what it rules out, and its role in the paper's argument. Written as flowing prose, not a list.",
  "model_extra_inference": {{
    "extra_inference": "optional interpretation beyond the paper's explicit claims. Keep separate from scientific_conclusion.",
    "support_status": "table_supported | caption_supported | context_supported | speculative | none",
    "supporting_evidence": "what table/caption/context evidence supports the inference. 'None' if speculative.",
    "risk": "low | medium | high",
    "open_questions": "scientific questions this table raises that the paper does not address or resolve.",
    "alternative_interpretation": "one alternative valid reading of these results — different explanation, confound, or mechanism. 'None' if unambiguous."
  }},
  "context_used": "{context_used}",
  "confidence": "high | medium | low"
}}

confidence: high = values clearly readable{conf_hint}; medium = some cells hard to read or ambiguous; low = inferring beyond what is shown.
Never refuse — all tables must be analyzed."""


def _build_prompt(template: str, abstract: str, caption: str,
                  context_text: str = "", context_label: str = "") -> str:
    """
    Construye el prompt final inyectando abstract, caption y contexto RAG.
    Si no hay contexto, el bloque de secciones ancladas se omite.
    """
    has_ctx = bool(context_text and context_text.strip())

    if has_ctx:
        context_block = (
            f"\n{context_label}:\n"
            f"{context_text}\n"
            f"\n---\n\n"
        )
        context_hint    = ", and using the paper text above as authoritative reference"
        conf_hint       = " + paper text alignment"
        context_used    = context_label.split("(")[0].strip().lower()
        anchored_hint   = ", hypothesis tested, and controls assessment"
        anchored_sections = """
## Hypothesis Tested
The specific claim or hypothesis from the paper that this figure/table tests or supports. \
Quote the exact sentence from the paper text.

## Controls Assessment
Experimental controls present (positive, negative, baseline comparisons). \
Note controls conspicuously absent given the experimental design described in the paper.

"""
        anchored_json = """
  "hypothesis_tested": "specific claim from the paper this item tests. exact quote.",
  "paper_quote": "exact sentence from the paper text this item is meant to support.",
  "controls_assessment": "controls present. note absent controls given the experimental design.","""
    else:
        context_block     = "\n"
        context_hint      = ""
        conf_hint         = ""
        context_used      = "abstract only"
        anchored_hint     = ""
        anchored_sections = "\n"
        anchored_json     = ""

    return template.format(
        abstract=abstract,
        caption=caption,
        context_block=context_block,
        context_hint=context_hint,
        conf_hint=conf_hint,
        context_used=context_used,
        anchored_hint=anchored_hint,
        anchored_sections=anchored_sections,
        anchored_json=anchored_json,
    )


# ─── Prompt de síntesis final (texto puro, sin imagen) ───────────────────────

PROMPT_PAPER_SUMMARY_TEMPLATE = """You have analyzed all figures and tables of a scientific paper.

PAPER ABSTRACT:
{abstract}

INDIVIDUAL ANALYSES ({n_items} items):
{items_block}

Based on the above analyses, write a paper-level synthesis. Respond ONLY with valid JSON:

{{
  "main_contribution": "the central claim or finding the paper makes, in 1-2 sentences.",
  "narrative": "how the figures and tables build the paper's argument step by step. Cite specific items (Fig1, Table2, etc.) and connect their findings.",
  "key_evidence": ["label of the 3 most critical items that support the main contribution"],
  "contradictions_or_gaps": "any figures or tables that contradict each other, or gaps in evidence. Write 'None detected' if coherent.",
  "limitations_noted": "methodological or statistical limitations visible across the analyses.",
  "overall_confidence": "high | medium | low"
}}

Never refuse. Base all claims strictly on the individual analyses provided above."""


def _build_items_block(results: list) -> str:
    """Formatea los análisis individuales para el prompt de síntesis."""
    lines = []
    for r in results:
        label   = r.get("label", "?")
        kind    = "Table" if r.get("kind") == "table" else "Figure"
        caption = r.get("caption", "")[:120]
        parsed  = r.get("analysis_parsed") or {}
        finding = (parsed.get("data_and_patterns") or parsed.get("key_entries")
                   or parsed.get("best_result") or "")
        conclusion = (parsed.get("scientific_conclusion") or
                      parsed.get("scientific_interpretation") or "")
        conf    = r.get("confidence", "?")
        lines.append(
            f"[{label}] ({kind}) caption: {caption}\n"
            f"  hallazgo: {finding}\n"
            f"  conclusión: {conclusion}\n"
            f"  confianza: {conf}"
        )
    return "\n\n".join(lines)


# ─── HTTP client ──────────────────────────────────────────────────────────────
def ask_api(server, prompt, image_bytes=None, max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT):
    content = []
    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "messages":     [{"role": "user", "content": content}],
        "max_tokens":   max_tokens,
        "temperature":  temperature,
        "repeat_penalty": 1.15,
        "stream":       False,
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


# ─── RAG: chunking (solo para context_strategy="bm25") ───────────────────────
def _tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def chunk_text(text, chunk_words=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_CHUNK_OVERLAP):
    """Divide el texto en chunks con overlap. Usado solo en modo bm25."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current_words = []
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        if len(words) > chunk_words * 1.5:
            for start in range(0, len(words), chunk_words - overlap):
                segment = words[start:start + chunk_words]
                if len(segment) >= 20:
                    chunks.append(" ".join(segment))
            continue
        if len(current_words) + len(words) > chunk_words:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] + words
        else:
            current_words.extend(words)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def _build_bm25(tokenized_chunks):
    try:
        from rank_bm25 import BM25Okapi
        index = BM25Okapi(tokenized_chunks)
        return lambda q: index.get_scores(q)
    except ImportError:
        n = len(tokenized_chunks)
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
                s = 0.0
                for qt in query_tokens:
                    if qt in tf_map:
                        tf  = tf_map[qt] / max(len(tokens), 1)
                        idf = math.log((n + 1) / (df.get(qt, 0) + 1)) + 1
                        s  += tf * idf
                scores.append(s)
            return scores

        return score_fn


def build_index(chunks):
    tokenized = [_tokenize(c) for c in chunks]
    return _build_bm25(tokenized), tokenized


def retrieve(query, score_fn, chunks, top_k=DEFAULT_TOP_K):
    """
    Recupera top_k chunks y los reordena aplicando la estrategia anti
    'lost in the middle' (Liu et al. 2023, arXiv:2307.03172):

      posición 0          → chunk con mayor score BM25  (modelo atiende inicio)
      posiciones 1..k-2   → chunks restantes en orden documental (coherencia)
      posición k-1        → chunk con segundo mayor score  (modelo atiende final)

    Así los dos fragmentos más relevantes quedan en los extremos del bloque
    de contexto, donde el modelo presta más atención.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_k]

    scores = score_fn(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top    = ranked[:top_k]

    if len(top) <= 2:
        return [chunks[i] for i, _ in top]

    best_idx   = top[0][0]
    second_idx = top[1][0]
    middle_idx = sorted(i for i, _ in top[2:])   # orden documental → coherencia

    return [chunks[i] for i in [best_idx] + middle_idx + [second_idx]]


def build_rag_index(text, chunk_words=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_CHUNK_OVERLAP):
    chunks = chunk_text(text, chunk_words=chunk_words, overlap=overlap)
    score_fn, _ = build_index(chunks)
    return chunks, score_fn


# ─── Pipeline ─────────────────────────────────────────────────────────────────
def analyze_all(
    figures_json,
    pdf_path=None,
    context_file=None,
    server=DEFAULT_SERVER,
    context_strategy=DEFAULT_CONTEXT_STRATEGY,    # "full" | "bm25"
    max_context_words=DEFAULT_MAX_CONTEXT_WORDS,  # 0 = sin límite (solo en full)
    abstract_words=DEFAULT_ABSTRACT_WORDS,        # 0 = abstract completo
    chunk_words=DEFAULT_CHUNK_WORDS,              # solo en bm25
    overlap=DEFAULT_CHUNK_OVERLAP,                # solo en bm25
    top_k=DEFAULT_TOP_K,                          # solo en bm25
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    timeout=DEFAULT_TIMEOUT,
    out_path=None,
    tables_json=None,
):
    fig_json = Path(figures_json)
    if pdf_path:
        pdf_path = Path(pdf_path)

    meta  = json.loads(fig_json.read_text(encoding="utf-8"))
    items = list(meta["items"])

    # Merge tablas de extract_tables.py
    if tables_json:
        tbl_path = Path(tables_json)
        if tbl_path.exists():
            tbl_meta = json.loads(tbl_path.read_text(encoding="utf-8"))
            items.extend(tbl_meta["items"])

    if out_path is None:
        out_path = fig_json.parent / "analyses_rag.json"
    else:
        out_path = Path(out_path)

    # Resume: saltar ítems ya analizados (tienen campo "analysis")
    results = []
    done    = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for it in prev.get("items", []):
                if "analysis" in it:
                    results.append(it)
                    done.add(it["label"])
        except Exception:
            pass

    n_figs = sum(1 for it in items if it.get("kind") != "table")
    n_tbls = sum(1 for it in items if it.get("kind") == "table")
    print(f"\nServer: {server}")
    print(f"Items: {len(items)} ({n_figs} figuras + {n_tbls} tablas) | ya hechos: {len(done)}")
    print(f"Estrategia: context_strategy={context_strategy} | abstract_words={abstract_words or 'completo'}")
    if context_strategy == "bm25":
        print(f"BM25: top_k={top_k} | chunk_words={chunk_words} | overlap={overlap}")
    if context_strategy == "full" and max_context_words:
        print(f"Full-text: max_context_words={max_context_words}")
    print()

    # ── Extraer texto del paper (una sola vez) ──────────────────────────────
    raw_text = ""
    try:
        if context_file and Path(context_file).exists():
            raw_text = Path(context_file).read_text(encoding="utf-8", errors="ignore")
        elif pdf_path:
            raw_text = extract_paper_text(pdf_path=pdf_path)
    except Exception as e:
        print(f"  WARN texto: {e}")

    # ── Abstract real detectado por sección ────────────────────────────────
    abstract = _extract_abstract(raw_text) if raw_text else ""
    if abstract:
        print(f"  Abstract: {len(abstract.split())} palabras detectadas")

    # ── Preparar contexto RAG (una sola vez) ───────────────────────────────
    full_context_text = ""
    full_context_note = ""
    chunks            = None
    score_fn          = None

    if raw_text:
        if context_strategy == "full":
            full_context_text, full_context_note = _fmt_full_context(
                raw_text, max_words=max_context_words
            )
            print(f"  Full-text: {len(full_context_text.split())} palabras{full_context_note}")
        else:  # bm25
            print(f"  Construyendo índice BM25 sobre {len(raw_text):,} chars...")
            chunks = chunk_text(raw_text, chunk_words=chunk_words, overlap=overlap)
            score_fn, _ = build_index(chunks)
            print(f"  Chunks: {len(chunks)} ({chunk_words}w c/u, {overlap}w overlap)")
    print()

    # ── Loop principal — una llamada por ítem ──────────────────────────────
    for i, item in enumerate(items):
        if item["label"] in done:
            print(f"[{i + 1}/{len(items)}] {item['label']} SKIP")
            continue

        img_path = Path(item["image_path"])
        if not img_path.is_absolute():
            img_path = fig_json.parent / img_path.name
        img_bytes = img_path.read_bytes()

        is_table = item.get("kind") == "table"
        caption  = item.get("caption", "Not provided.")
        result   = dict(item)
        t_start  = time.time()

        # ── Construir bloque de contexto RAG para este ítem ────────────────
        context_text  = ""
        context_label = ""

        if raw_text:
            if context_strategy == "full" and full_context_text:
                context_text  = full_context_text
                context_label = f"FULL PAPER TEXT{full_context_note}"
                result["context_words"]    = len(full_context_text.split())
                result["context_strategy"] = "full"
            elif context_strategy == "bm25" and chunks:
                retrieved     = retrieve(caption, score_fn, chunks, top_k=top_k)
                context_text  = "\n\n---\n\n".join(retrieved)
                context_label = f"RELEVANT PAPER SECTIONS (BM25 top-{top_k} chunks)"
                result["retrieved_chunks"] = len(retrieved)
                result["retrieved_words"]  = sum(len(c.split()) for c in retrieved)
                result["context_strategy"] = "bm25"

        # ── Una sola llamada: abstract (dinámico) + secciones + contexto RAG
        tmpl   = PROMPT_TBL_TEMPLATE if is_table else PROMPT_FIG_TEMPLATE
        prompt = _build_prompt(
            template      = tmpl,
            abstract      = abstract or "Not available.",
            caption       = caption,
            context_text  = context_text,
            context_label = context_label,
        )

        try:
            ans = ask_api(server, prompt, image_bytes=img_bytes,
                          max_tokens=max_tokens, temperature=temperature, timeout=timeout)
            result["analysis"] = ans
            parsed = _parse_json(ans)
            if parsed:
                result["analysis_parsed"] = parsed
                result["confidence"]      = parsed.get("confidence", "unknown")
        except Exception as e:
            result["analysis_error"] = str(e)

        result["elapsed_sec"] = round(time.time() - t_start, 1)
        results.append(result)

        ctx_info = ""
        if context_strategy == "full":
            ctx_info = f" [{result.get('context_words', '?')}w full]"
        elif context_strategy == "bm25":
            ctx_info = f" [{result.get('retrieved_chunks', '?')} chunks]"
        preview  = (result.get("analysis") or "")[:80].replace("\n", " ")
        kind_tag = "TBL" if is_table else "FIG"
        print(f"[{i + 1}/{len(items)}] [{kind_tag}] {item['label']} p{item['page']} "
              f"({result['elapsed_sec']}s){ctx_info} {preview}...")

        out_path.write_text(
            json.dumps({
                "total":             len(results),
                "context_strategy":  context_strategy,
                "abstract_words":    abstract_words or "full",
                "max_context_words": max_context_words or "unlimited",
                "top_k":             top_k if context_strategy == "bm25" else None,
                "chunk_words":       chunk_words if context_strategy == "bm25" else None,
                "items":             results,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Síntesis final del paper (texto puro, sin imagen) ──────────────────
    paper_summary = None
    if results:
        print("\nGenerando síntesis final del paper...")
        try:
            abstract_text = _extract_abstract(raw_text) if raw_text else ""
            items_block   = _build_items_block(results)
            prompt_sum    = PROMPT_PAPER_SUMMARY_TEMPLATE.format(
                abstract=abstract_text or "Not available.",
                n_items=len(results),
                items_block=items_block,
            )
            ans_sum      = ask_api(server, prompt_sum, image_bytes=None,
                                   max_tokens=max_tokens, temperature=temperature,
                                   timeout=timeout)
            paper_summary = _parse_json(ans_sum) or {"raw": ans_sum}
            print(f"  Síntesis: {str(paper_summary)[:100]}...")
        except Exception as e:
            print(f"  WARN síntesis: {e}")

    out_path.write_text(
        json.dumps({
            "total":             len(results),
            "context_strategy":  context_strategy,
            "abstract_words":    abstract_words or "full",
            "max_context_words": max_context_words or "unlimited",
            "top_k":             top_k if context_strategy == "bm25" else None,
            "chunk_words":       chunk_words if context_strategy == "bm25" else None,
            "paper_summary":     paper_summary,
            "items":             results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResultados guardados: {out_path}")
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Analiza figuras Y tablas científicas — un prompt completo por ítem.")
    p.add_argument("figures_json",       help="figures.json de extract_figures.py")
    p.add_argument("--pdf",              help="PDF original")
    p.add_argument("--context-file",     help="Texto plano pre-extraído del paper (.txt)")
    p.add_argument("--server",           default=DEFAULT_SERVER)
    p.add_argument("--out",              help="ruta de salida JSON")

    # Estrategia de contexto
    p.add_argument("--context-strategy", choices=["bm25", "full"],
                   default=DEFAULT_CONTEXT_STRATEGY,
                   help="'bm25' (default) = top-k chunks | 'full' = texto completo")
    p.add_argument("--max-context-words", type=int, default=DEFAULT_MAX_CONTEXT_WORDS,
                   help="Palabras máx en modo full (0=sin límite)")
    p.add_argument("--abstract-words",   type=int, default=DEFAULT_ABSTRACT_WORDS,
                   help="Ignorado — el abstract se detecta por sección automáticamente")

    # Parámetros BM25
    p.add_argument("--top-k",            type=int, default=DEFAULT_TOP_K)
    p.add_argument("--chunk-words",      type=int, default=DEFAULT_CHUNK_WORDS)
    p.add_argument("--chunk-overlap",    type=int, default=DEFAULT_CHUNK_OVERLAP)

    # LLM
    p.add_argument("--max-tokens",       type=int,   default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature",      type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--timeout",          type=int,   default=DEFAULT_TIMEOUT)
    args = p.parse_args(argv)

    if not server_health(args.server):
        sys.exit(f"Servidor no responde: {args.server}")

    analyze_all(
        args.figures_json,
        pdf_path=args.pdf,
        context_file=args.context_file,
        server=args.server,
        context_strategy=args.context_strategy,
        max_context_words=args.max_context_words,
        abstract_words=args.abstract_words,
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
