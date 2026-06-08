#!/usr/bin/env python3
"""
03_sintetizador_contexto_v3_1.py
==============================
Convierte context.json en prompts factuales para un VLM.

Uso:
    python 03_sintetizador_contexto_v3_1.py --json_dir salida/paper_stem
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# ------------------------------ parámetros ---------------------------------
MAX_AUTHOR_SNIPPETS = 5
MAX_AUTHOR_TOTAL_CHARS = 2200
MAX_BACKGROUND_CHARS = 420
MAX_RAW_VISIBLE_CHARS = 900
DEDUPE_JACCARD = 0.80

AUTHOR_SOURCES = {
    "exact_figure_reference",
    "nearby_after",
    "nearby_before",
    "section_context",
    "bm25_sentence_window",
}
REFERENCE_SOURCES = {
    "cross_reference_caption",
    "cross_reference_mention",
}

NOISE_RE = re.compile(
    r"(?i)(we\s+thank|funded\s+by|grant\s+agreement|author\s+contributions?|"
    r"competing\s+interests?|data\s+availability|code\s+availability|github|"
    r"accession\s+number|reporting\s+summary|peer\s+review|doi\.org|e-?mail)"
)
CAPTION_HEADER_RE = re.compile(
    r"(?i)\b(?:extended\s+data\s+|supplementary\s+)?"
    r"(?:fig(?:ure)?\.?|table)\s*s?\d+[a-z]?\s*\|"
)
FIG_LABEL_RE = re.compile(
    r"(?i)\b(extended\s+data\s+fig(?:ure)?\.?\s*\d+[a-z]?|"
    r"supplementary\s+fig(?:ure)?\.?\s*s?\d+[a-z]?|"
    r"fig(?:ure)?\.?\s*\d+[a-z]?|table\s*s?\d+[a-z]?)\b"
)

# ------------------------------ utilidades ----------------------------------
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


GENERIC_CONTEXT_TERMS = {
    "fig", "figure", "extended", "data", "supplementary", "table", "panel", "panels",
    "chromosome", "chromosomes", "shown", "see", "details", "analysis", "same",
    "gene", "genes", "genome", "genomes", "across", "based", "using", "grouped",
}


def semantic_keywords(text: str) -> set[str]:
    text = FIG_LABEL_RE.sub(" ", text or "")
    return {t for t in tokens(text) if len(t) >= 3 and t not in GENERIC_CONTEXT_TERMS and not t.isdigit()}


def figure_identity(label: str | None) -> tuple[str, str] | None:
    if not label:
        return None
    s = clean(label).lower()
    m = re.search(r"extended\s+data\s+fig(?:ure)?\.?\s*(\d+)", s)
    if m:
        return ("extended_fig", m.group(1))
    m = re.search(r"supplementary\s+fig(?:ure)?\.?\s*s?(\d+)", s)
    if m:
        return ("supp_fig", m.group(1))
    m = re.search(r"fig(?:ure)?\.?\s*(\d+)", s)
    if m:
        return ("fig", m.group(1))
    m = re.search(r"table\s*s?(\d+)", s)
    if m:
        return ("table", m.group(1))
    return None


def label_key(label: str | None) -> str | None:
    fid = figure_identity(label)
    return f"{fid[0]}:{fid[1]}" if fid else None


def referenced_labels(text: str) -> set[str]:
    keys = set()
    for lab in FIG_LABEL_RE.findall(text or ""):
        k = label_key(lab)
        if k:
            keys.add(k)
    return keys


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def dedupe_texts(texts: list[str], threshold: float = DEDUPE_JACCARD) -> list[str]:
    out: list[str] = []
    for s in texts:
        s = clean(s)
        if not s:
            continue
        duplicate = False
        for k in out:
            if s[:200].lower() == k[:200].lower() or jaccard(s, k) >= threshold:
                duplicate = True
                break
            if len(s) < len(k) and s[:120].lower() in k.lower():
                duplicate = True
                break
        if not duplicate:
            out.append(s)
    return out


# Nomenclatura biológica/química que NUNCA debe perder sus dígitos al limpiar citas.
.
_BIO_PROTECT_RE = re.compile(
    r"\b("
    r"\d+[A-Za-z][A-Za-z\d]*"            # empieza con dígito: 15N, 13C, 2H
    r"|[A-Za-z]*\d+[A-Za-z][A-Za-z\d]*"  # dígito seguido de letra: H3K27, N2O
    r"|[A-Z][A-Za-z]*\d+\b"              # símbolo en mayúscula + dígito final: CO2, N2, O2
    r"|[a-z]{1,3}\d{1,3}\b"              # gen/proteína minúsculo corto + dígito: p53, il6, cox2
    r")"
)


def _protect_bio_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Reemplaza nomenclatura alfanumérica por placeholders para protegerla de strip_refs."""
    mapping: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        key = f"\x00{len(mapping)}\x00"
        mapping[key] = m.group(0)
        return key

    protected = _BIO_PROTECT_RE.sub(_sub, text)
    return protected, mapping


def strip_refs(text: str) -> str:
    # 1) Proteger nomenclatura (CO2, p53, 15N, H3K27, N2O, etc.).
    protected, mapping = _protect_bio_tokens(text)
    # 2) Borrar superíndices de citación: dígitos pegados a una letra minúscula común
    #    y seguidos de puntuación de cierre o fin de palabra. Como la nomenclatura ya
    #    está protegida, aquí solo caen citas reales tipo "...growth12." o "...rates3,".
    protected = re.sub(r"(?<=[a-záéíóúñü])\d{1,3}(?=[.,;:)\s]|$)", "", protected)
    # Limpia restos de listas de citas tipo "increased,6" -> "increased" (coma + dígitos colgantes).
    protected = re.sub(r"(?<=[a-záéíóúñü])\s*,\s*\d{1,3}(?:\s*,\s*\d{1,3})*(?=[.,;:)\s]|$)", "", protected)
    protected = re.sub(r"\(refs?\.\s*[\d,\s]+\)", "", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\s+([.,;:])", r"\1", protected)
    # 3) Restaurar la nomenclatura protegida.
    for key, tok in mapping.items():
        protected = protected.replace(key, tok)
    return clean(protected)


def extract_panel_labels(caption: str, visible_structured: dict[str, Any] | None, raw_visible: str = "") -> list[str]:
    labels: set[str] = set()
    src = f"{caption} {raw_visible}"
    for m in re.finditer(r"(?i)(?:^|[.;|]\s+|\b)([a-l])\s*[,\u2013-]\s*[A-Z(]", src):
        labels.add(m.group(1).lower())
    for m in re.finditer(r"(?i)\b([a-l])\s*[\u2013-]\s*([a-l])\b", src):
        start, end = m.group(1).lower(), m.group(2).lower()
        if 0 <= ord(end) - ord(start) <= 11:
            for c in range(ord(start), ord(end) + 1):
                labels.add(chr(c))
    if visible_structured:
        for lab in visible_structured.get("panel_labels", []) or []:
            if re.fullmatch(r"[a-l]", str(lab).lower()):
                labels.add(str(lab).lower())
    # Devuelve secuencia contigua si existe desde a; si no, orden simple.
    if "a" in labels:
        seq = []
        c = ord("a")
        while chr(c) in labels:
            seq.append(chr(c))
            c += 1
        return seq
    return sorted(labels)


def caption_block_label(status: str) -> str:
    if status == "clean":
        return "[CAPTION — high confidence]"
    if status == "truncated":
        return "[CAPTION — partially recovered, may be truncated]"
    if status == "contaminated":
        return "[CAPTION — possibly corrupted; use cautiously]"
    if status == "missing":
        return "[CAPTION — unavailable]"
    return "[CAPTION — uncertain quality]"


def select_author_framing(prioritized: list[dict[str, Any]], caption: str, figure_label: str | None = None, cross_refs: list[str] | None = None) -> list[str]:
    snippets: list[str] = []
    cap_prefix = clean(caption)[:100].lower()
    target_keywords = semantic_keywords(caption)
    target_key = label_key(figure_label)
    allowed_ref_keys = {label_key(r) for r in (cross_refs or []) if label_key(r)}

    for e in prioritized:
        if e.get("source") not in AUTHOR_SOURCES:
            continue
        text = clean(e.get("text", ""))
        if not text or NOISE_RE.search(text):
            continue
        # Quita si es básicamente caption repetida.
        if cap_prefix and cap_prefix in text.lower() and len(text) <= len(caption) + 120:
            continue

        # Drop snippets that mainly discuss other figures not allowed by a cross-reference.
        refs = referenced_labels(text)
        foreign_refs = refs - ({target_key} if target_key else set()) - allowed_ref_keys
        overlap = semantic_keywords(text) & target_keywords
        if foreign_refs and len(overlap) < 3:
            continue
        if target_keywords and not refs and len(overlap) < 2:
            # Keep conservative nearby text only if short and non-specific.
            if e.get("source") not in {"nearby_after", "nearby_before"} or len(text.split()) > 160:
                continue

        snippets.append(strip_refs(text))
    snippets = dedupe_texts(snippets)[:MAX_AUTHOR_SNIPPETS]
    out: list[str] = []
    total = 0
    for s in snippets:
        if total + len(s) > MAX_AUTHOR_TOTAL_CHARS:
            remain = MAX_AUTHOR_TOTAL_CHARS - total
            if remain > 120:
                out.append(s[:remain].rsplit(" ", 1)[0] + " [...]")
            break
        out.append(s)
        total += len(s)
    return out


def select_reference_context(prioritized: list[dict[str, Any]]) -> list[str]:
    """Prefer referenced figure captions over arbitrary mentions."""
    by_ref: dict[str, dict[str, list[str]]] = {}
    for e in prioritized:
        if e.get("source") not in REFERENCE_SOURCES:
            continue
        text = clean(e.get("text", ""))
        if not text or NOISE_RE.search(text):
            continue
        ref = e.get("referenced_label") or "referenced figure"
        by_ref.setdefault(ref, {"caption": [], "mention": []})
        kind = "caption" if e.get("source") == "cross_reference_caption" else "mention"
        by_ref[ref][kind].append(strip_refs(text))

    snippets: list[str] = []
    for ref, groups in by_ref.items():
        chosen = groups["caption"] if groups["caption"] else groups["mention"][:1]
        for text in chosen[:2]:
            snippets.append(f"{ref}: {text}")
    return dedupe_texts(snippets)[:3]


def summarize_visible_text(visible_structured: dict[str, Any], raw_visible: str) -> str:
    if not visible_structured:
        return clean(raw_visible)[:MAX_RAW_VISIBLE_CHARS]
    parts = []
    panels = visible_structured.get("panel_labels") or []
    labels = visible_structured.get("labels_candidates") or []
    nums = visible_structured.get("numeric_values_sample") or []
    if panels:
        parts.append("panel_labels=" + ", ".join(map(str, panels)))
    if labels:
        parts.append("labels_or_axes_candidates=" + "; ".join(map(str, labels[:45])))
    if nums:
        parts.append("numeric_values_sample=" + ", ".join(map(str, nums[:60])))
    return "\n".join(parts) if parts else clean(raw_visible)[:MAX_RAW_VISIBLE_CHARS]


def context_quality_block(q: dict[str, Any]) -> str:
    if not q:
        return "quality=unknown"
    lines = [
        f"quality={q.get('quality', 'unknown')}",
        f"caption_status={q.get('caption_status', 'unknown')}",
        f"author_framing_status={q.get('author_framing_status', 'unknown')}",
        f"visible_text_status={q.get('visible_text_status', 'unknown')}",
    ]
    issues = q.get("issues") or []
    if issues:
        lines.append("issues=" + ", ".join(map(str, issues)))
    refs = q.get("cross_references_detected") or []
    if refs:
        lines.append("cross_references_detected=" + ", ".join(map(str, refs)))
    if q.get("quality") == "low":
        lines.append("instruction=Context is weak; the VLM should avoid unsupported scientific interpretation and produce only visually/caption-supported claims.")
    elif q.get("quality") == "medium":
        lines.append("instruction=Context has limitations; the VLM must explicitly mark uncertainty and avoid extrapolation.")
    return "\n".join(lines)

# --------------------------- síntesis del contexto ---------------------------
def synthesize_context(item: dict[str, Any]) -> dict[str, Any]:
    ctx = item.get("context_by_source", {}) or {}
    quality = ctx.get("context_quality", {}) or item.get("context_retrieval_metadata", {}).get("context_quality", {}) or {}
    figure_label = item.get("figure_label") or ctx.get("figure_label") or "Figure"
    caption = clean(item.get("caption_oficial", "") or ctx.get("caption", ""))
    visible_structured = item.get("visible_text_structured") or ctx.get("visible_text_structured") or {}
    raw_visible = clean(item.get("visible_text_from_crop", "") or ctx.get("visible_text_from_crop", ""))
    prioritized = ctx.get("prioritized_context", []) or []

    cross_refs = (quality.get("cross_references_detected") or []) if isinstance(quality, dict) else []
    author_framing = select_author_framing(prioritized, caption, figure_label=figure_label, cross_refs=cross_refs)
    reference_context = select_reference_context(prioritized)
    panels = extract_panel_labels(caption, visible_structured, raw_visible)

    abstract = clean(item.get("abstract_paper", ""))
    background = ""
    if abstract:
        sents = re.split(r"(?<=[.])\s+", abstract)
        background = clean(" ".join(sents[:2]))[:MAX_BACKGROUND_CHARS]

    cap_status = quality.get("caption_status") or item.get("caption_audit", {}).get("status") or "unknown"
    return {
        "figure_label": figure_label,
        "panels_detected": panels,
        "block0_context_quality": quality,
        "block1_caption_label": caption_block_label(cap_status),
        "block1_caption": caption,
        "block2_author_framing": author_framing,
        "block2b_reference_context": reference_context,
        "block3_visible_text_structured": summarize_visible_text(visible_structured, raw_visible),
        "block3_raw_visible_text_sample": raw_visible[:MAX_RAW_VISIBLE_CHARS],
        "block4_background": background,
    }

# --------------------------- prompt y esquema VLM ----------------------------
SYSTEM_PREAMBLE = (
    "You are a scientific figure interpreter for a biology corpus. Your job is to "
    "produce a FACTUAL description of the figure shown, to be used as training data. "
    "Accuracy is more important than completeness. If the evidence is insufficient, "
    "say so and do not infer beyond the image and reliable context."
)

FACTUALITY_RULES = (
    "RULES (read carefully):\n"
    "1. Base your interpretation on what you OBSERVE in the image. Use the CAPTION and AUTHOR FRAMING only as supporting evidence.\n"
    "2. Treat CONTEXT QUALITY as binding. If caption_status is truncated/contaminated/missing, do not treat the caption as authoritative.\n"
    "3. If TEXT-IN-FIGURE contradicts what you see, trust what you SEE. The detected text may be incomplete or contain extraction errors.\n"
    "4. Do NOT invent numeric values, axis ranges, p-values, sample counts, biological mechanisms, or conclusions. Write \"not legible\" or \"not supported by provided context\" instead of guessing.\n"
    "5. Do NOT add biological claims that are not supported by the image, the reliable caption, or author framing.\n"
    "6. Describe each panel separately when the figure has labeled panels.\n"
    "7. First count the panels you can SEE in the image. The panel hint may be incomplete or wrong.\n"
    "8. In the JSON, fill evidence_used with short evidence phrases from visual/caption/author_context/OCR.\n"
    "9. If context quality is low or author framing is missing, provide a conservative visual description and explicitly mark limitations."
)


def build_expected_schema() -> dict[str, Any]:
    return {
        "figure_label": "<label>",
        "figure_type": "<e.g. heatmap, bar chart, scatter plot, micrograph, schematic, multipanel mixed figure>",
        "one_line_summary": "<single factual sentence about the whole figure>",
        "n_panels_seen": "<integer>",
        "panels": [
            {
                "panel": "<panel letter seen in the image, or 'whole figure'>",
                "what_it_shows": "<factual description of visual content>",
                "axes_or_variables": "<x/y axes, color scale, units, groups, or 'not legible'>",
                "key_observation": "<main visible pattern, only if supported>",
                "in_provided_text": "<true|false>",
                "uncertain": "<uncertainty or empty>",
            }
        ],
        "factual_caption_paraphrase": "<2-4 sentence factual restatement, no new claims>",
        "evidence_used": {
            "visual": ["<evidence from image>"],
            "caption": ["<evidence from reliable caption, or empty>"],
            "author_context": ["<evidence from author framing, or empty>"],
            "ocr": ["<evidence from detected text, or empty>"],
        },
        "context_limitations": "<state missing/truncated/contaminated context or empty>",
        "confidence": "<high|medium|low>",
    }


def build_vlm_prompt(synth: dict[str, Any]) -> str:
    fl = synth["figure_label"]
    parts: list[str] = [SYSTEM_PREAMBLE, ""]
    parts.append(f"[FIGURE] {fl}")
    if synth.get("panels_detected"):
        parts.append(
            "[PANEL HINT — non-binding, may be incomplete] The text/context mentions panels: "
            + ", ".join(synth["panels_detected"])
            + ". Count panels yourself from the image."
        )
    parts.append("")
    parts.append("[CONTEXT QUALITY — use this to decide how cautious to be]")
    parts.append(context_quality_block(synth.get("block0_context_quality", {})))
    parts.append("")
    parts.append(synth.get("block1_caption_label") or "[CAPTION — uncertain quality]")
    parts.append(synth.get("block1_caption") or "(caption unavailable)")
    parts.append("")
    if synth.get("block2_author_framing"):
        parts.append("[AUTHOR FRAMING — from article body; use only when directly relevant]")
        for s in synth["block2_author_framing"]:
            parts.append(f"- {s}")
        parts.append("")
    if synth.get("block2b_reference_context"):
        parts.append("[REFERENCE FIGURE CONTEXT — use as visual convention guidance, not as automatic claim transfer]")
        for s in synth["block2b_reference_context"]:
            parts.append(f"- {s}")
        parts.append("")
    if synth.get("block3_visible_text_structured"):
        parts.append("[TEXT DETECTED IN FIGURE — structured, may be incomplete; trust what you see over this]")
        parts.append(synth["block3_visible_text_structured"])
        parts.append("")
    if synth.get("block3_raw_visible_text_sample"):
        parts.append("[RAW TEXT SAMPLE — low confidence, for disambiguation only]")
        parts.append(synth["block3_raw_visible_text_sample"])
        parts.append("")
    if synth.get("block4_background"):
        parts.append("[BACKGROUND — paper abstract, context only]")
        parts.append(synth["block4_background"])
        parts.append("")
    parts.append(FACTUALITY_RULES)
    parts.append("")
    parts.append("TASK: Produce BOTH of the following, in this order:")
    parts.append("")
    parts.append("=== PART 1: DESCRIPTION (free text) ===")
    parts.append(
        "First, look at the image and count how many panels it has. Then write a factual, self-contained description for a reader who cannot see it. "
        "Describe every panel you see. Ground every claim in the visible image, reliable caption, author framing, or detected text. Mark uncertainty explicitly."
    )
    parts.append("")
    parts.append("=== PART 2: STRUCTURED (JSON) ===")
    parts.append("Output a JSON object with EXACTLY this schema (no extra keys, no markdown fences):")
    parts.append(json.dumps(build_expected_schema(), ensure_ascii=False, indent=2))
    return "\n".join(parts)

# ------------------------------- proceso ------------------------------------
def process_prompts(json_dir: Path, write_txt: bool = True) -> None:
    context_path = json_dir / "context.json"
    if not context_path.exists():
        raise FileNotFoundError(f"No encontré context.json en {json_dir}")
    with open(context_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    out_items: list[dict[str, Any]] = []
    prompts_dir = json_dir / "vlm_prompts"
    if write_txt:
        prompts_dir.mkdir(exist_ok=True)

    audit_rows: list[dict[str, Any]] = []
    for item in items:
        synth = synthesize_context(item)
        prompt = build_vlm_prompt(synth)
        out = {
            "item_id": item.get("item_id"),
            "crop_path": item.get("crop_path"),
            "page": item.get("page"),
            "figure_label": synth.get("figure_label"),
            "synthesized_context": synth,
            "vlm_prompt_text": prompt,
            "expected_output_schema": build_expected_schema(),
        }
        out_items.append(out)
        q = synth.get("block0_context_quality", {}) or {}
        audit_rows.append({
            "item_id": item.get("item_id"),
            "figure_label": synth.get("figure_label"),
            "page": item.get("page"),
            "context_quality": q.get("quality", "unknown"),
            "caption_status": q.get("caption_status", "unknown"),
            "author_framing_status": q.get("author_framing_status", "unknown"),
            "visible_text_status": q.get("visible_text_status", "unknown"),
            "issues": q.get("issues", []),
        })
        if write_txt:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.get("item_id", "prompt")))
            (prompts_dir / f"{safe}.txt").write_text(prompt, encoding="utf-8")

    out_path = json_dir / "vlm_prompts.json"
    audit_path = json_dir / "prompt_context_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, indent=2, ensure_ascii=False)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_rows, f, indent=2, ensure_ascii=False)
    print(f"✅ Prompts generados: {out_path}")
    print(f"✅ Auditoría resumida: {audit_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--no-txt", action="store_true", help="No escribir prompts individuales .txt")
    args = parser.parse_args()
    process_prompts(Path(args.json_dir), write_txt=not args.no_txt)


if __name__ == "__main__":
    main()