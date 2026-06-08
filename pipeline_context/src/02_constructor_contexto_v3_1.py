#!/usr/bin/env python3
"""
02_constructor_contexto_v3_1_1.py
=============================
Construye contexto textual para cada figura detectada por 01_parser_hibrido_v3.py.


Uso:
    python 02_constructor_contexto_v3_1_1.py --json_dir salida/paper_stem
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except Exception:  # fallback si no está instalado
    BM25Okapi = None

from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.document import DoclingDocument, SectionHeaderItem, TextItem, TitleItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("constructor_contexto_v3_1")

# ------------------------------ parámetros ---------------------------------
MAX_CONTEXT_PASSAGES_FOR_PROMPT = 12
MAX_ENTRY_CHARS = 1200
PARAGRAPH_MIN_CHARS = 80
BM25_TOP_K = 12
SENTENCE_WINDOW = 1  # 1 => oración match ±1 vecina
MAX_SENTENCE_WINDOW_CHARS = 950
BM25_PAGE_RADIUS = 2  # acota el corpus BM25 a ±N páginas de la figura antes de rankear

# ------------------------------ regex genéricos -----------------------------
FIG_LABEL_RE = re.compile(
    r"(?i)\b("
    r"(?:extended\s+data\s+fig(?:ure)?\.?\s*\d+[a-z]?)"
    r"|(?:supplementary\s+fig(?:ure)?\.?\s*s?\d+[a-z]?)"
    r"|(?:fig(?:ure)?\.?\s*\d+[a-z]?)"
    r"|(?:table\s*s?\d+[a-z]?)"
    r"|(?:supplementary\s+table\s*s?\d+[a-z]?)"
    r")\b"
)
CAPTION_HEADER_RE = re.compile(
    r"(?i)\b(?:extended\s+data\s+|supplementary\s+)?"
    r"(?:fig(?:ure)?\.?|table)\s*s?\d+[a-z]?\s*\|"
)
BAD_SECTION_RE = re.compile(
    r"(?i)\b(references|data availability|code availability|author contributions?|"
    r"competing interests?|acknowledg|reporting summary|additional information|"
    r"peer review|supplementary information)\b"
)
NOISE_RE = re.compile(
    r"(?i)(doi\.org|e-?mail|correspondence|received:|accepted:|published online:|"
    r"check for updates|copyright|creative commons|we thank|funded by|grant agreement|"
    r"github|accession number|reporting summary|data availability|code availability)"
)
BODY_LIKE_RE = re.compile(
    r"(?i)\b(we\s+(?:show|find|found|identified|investigated|used|selected|classified|"
    r"observed|tested|performed|analyzed|analysed|conclude|note|emphasize)|"
    r"in\s+(?:this|the)\s+(?:study|paper|work)|our\s+(?:results|analysis|data|study)|"
    r"therefore|however|nevertheless|in\s+contrast|as\s+expected)\b"
)
CROSS_REFERENCE_RE = re.compile(
    r"(?i)\bsee\s+(?:also\s+)?("
    r"(?:extended\s+data\s+)?fig(?:ure)?\.?\s*\d+[a-z]?"
    r"(?:\s*[–-]\s*[a-z])?"
    r"|supplementary\s+fig(?:ure)?\.?\s*s?\d+[a-z]?"
    r"|table\s*\d+[a-z]?"
    r")"
)

# ------------------------------ utilidades ----------------------------------
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def truncate(text: str, limit: int = MAX_ENTRY_CHARS) -> str:
    text = clean(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " [...]"


def tokenize(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "was", "were",
        "with", "that", "this", "from", "by", "on", "as", "at", "be", "it", "we", "our", "their",
        "la", "el", "los", "las", "de", "del", "y", "o", "en", "para", "con", "por", "un", "una",
    }
    toks = re.findall(r"[a-záéíóúñü0-9]+", (text or "").lower())
    return [t for t in toks if len(t) > 1 and t not in stop]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


GENERIC_CONTEXT_TERMS = {
    "fig", "figure", "extended", "data", "supplementary", "table", "panel", "panels",
    "chromosome", "chromosomes", "shown", "see", "details", "analysis", "same",
    "gene", "genes", "genome", "genomes", "across", "based", "using", "grouped",
}


def semantic_keywords(text: str) -> set[str]:
    """Content keywords used for relevance filtering, excluding generic figure words."""
    txt = re.sub(FIG_LABEL_RE, " ", text or "")
    toks = set(tokenize(txt))
    return {t for t in toks if len(t) >= 3 and t not in GENERIC_CONTEXT_TERMS and not t.isdigit()}


def reference_identities(text: str) -> list[tuple[str, str]]:
    ids = []
    for lab in FIG_LABEL_RE.findall(text or ""):
        fid = figure_identity(lab)
        if fid and fid not in ids:
            ids.append(fid)
    return ids


def label_key(label: str | None) -> str | None:
    fid = figure_identity(label)
    return f"{fid[0]}:{fid[1]}" if fid else None


def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def dedupe_entries(entries: list[dict[str, Any]], threshold: float = 0.82) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        t = clean(e.get("text", ""))
        if not t:
            continue
        duplicate = False
        for k in out:
            kt = clean(k.get("text", ""))
            if t[:220].lower() == kt[:220].lower() or jaccard(t, kt) >= threshold:
                duplicate = True
                break
        if not duplicate:
            out.append(e)
    return out


def split_sentences(text: str) -> list[str]:
    text = clean(text)
    text = re.sub(r"\b(Fig|fig|Dr|Prof|Ref|refs)\.\s", lambda m: m.group(0).replace(". ", "<DOT> "), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def figure_identity(label: str | None) -> tuple[str, str] | None:
    if not label:
        return None
    s = clean(label).lower()
    m = re.search(r"extended\s+data\s+fig(?:ure)?\.?\s*(\d+)", s)
    if m: return ("extended_fig", m.group(1))
    m = re.search(r"supplementary\s+fig(?:ure)?\.?\s*s?(\d+)", s)
    if m: return ("supp_fig", m.group(1))
    m = re.search(r"fig(?:ure)?\.?\s*(\d+)", s)
    if m: return ("fig", m.group(1))
    m = re.search(r"table\s*s?(\d+)", s)
    if m: return ("table", m.group(1))
    return None


def caption_header_identities(text: str) -> list[tuple[str, str]]:
    ids: list[tuple[str, str]] = []
    for m in CAPTION_HEADER_RE.finditer(text or ""):
        lab = m.group(0).split("|")[0]
        fid = figure_identity(lab)
        if fid:
            ids.append(fid)
    return ids


def figure_reference_regex(label: str | None) -> re.Pattern | None:
    fid = figure_identity(label)
    if not fid:
        return None
    kind, num = fid
    if kind == "extended_fig":
        return re.compile(rf"(?i)\bextended\s+data\s+fig(?:ure)?\.?\s*{re.escape(num)}(?:\s*[a-z])?(?:\s*[–-]\s*[a-z])?\b")
    if kind == "supp_fig":
        return re.compile(rf"(?i)\bsupplementary\s+fig(?:ure)?\.?\s*s?{re.escape(num)}(?:\s*[a-z])?(?:\s*[–-]\s*[a-z])?\b")
    if kind == "table":
        return re.compile(rf"(?i)\b(?:supplementary\s+)?table\s*s?{re.escape(num)}(?:\s*[a-z])?\b")
    return re.compile(rf"(?i)(?<!extended\sdata\s)(?<!supplementary\s)\bfig(?:ure)?\.?\s*{re.escape(num)}(?:\s*[a-z])?(?:\s*[–-]\s*[a-z])?\b")


def extract_figure_label(caption: str, visible_text: str = "") -> str | None:
    m = FIG_LABEL_RE.search(f"{caption}\n{visible_text}")
    return clean(m.group(1)) if m else None


def is_metadata_or_noise(text: str, section: str | None = None) -> bool:
    t = clean(text)
    if not t:
        return True
    if NOISE_RE.search(t[:700]):
        return True
    if section and BAD_SECTION_RE.search(section):
        return True
    if len(t) < 50:
        return True
    return False

# ------------------------- linearización del documento ----------------------
def bbox_from_prov(prov, page_height: float) -> list[float] | None:
    if not prov or not getattr(prov, "bbox", None):
        return None
    bb = prov.bbox
    if bb.coord_origin == CoordOrigin.BOTTOMLEFT:
        bb = bb.to_top_left_origin(page_height=page_height)
    return [float(bb.l), float(bb.t), float(bb.r), float(bb.b)]


def linearize_with_layout(doc: DoclingDocument) -> list[dict[str, Any]]:
    seq: list[dict[str, Any]] = []
    current_section = None
    for idx, (item, _) in enumerate(doc.iterate_items()):
        text = getattr(item, "text", None)
        if not text:
            continue
        text = clean(text)
        page = None
        bbox = None
        if getattr(item, "prov", None):
            try:
                prov = item.prov[0]
                page = prov.page_no
                page_h = doc.pages[prov.page_no].size.height
                bbox = bbox_from_prov(prov, page_h)
            except Exception:
                pass
        typ = "text"
        if isinstance(item, SectionHeaderItem):
            typ = "section"; current_section = text
        elif isinstance(item, TitleItem):
            typ = "title"
        elif isinstance(item, TextItem):
            typ = "text"
        seq.append({"idx": idx, "type": typ, "text": text, "page": page, "bbox": bbox, "section": current_section})
    return seq


def extract_abstract(seq: list[dict[str, Any]]) -> str:
    # Docling no siempre marca "Abstract"; tomamos primeros párrafos largos no metadata.
    blocks: list[str] = []
    for n in seq:
        if n["type"] != "text":
            continue
        if is_metadata_or_noise(n["text"], n.get("section")):
            continue
        if len(n["text"]) >= 180:
            blocks.append(n["text"])
        if len(blocks) >= 2:
            break
    return clean(" ".join(blocks))[:1200] if blocks else ""


def paragraph_nodes(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [n for n in seq if n["type"] == "text" and len(clean(n["text"])) >= PARAGRAPH_MIN_CHARS and not is_metadata_or_noise(n["text"], n.get("section"))]


def make_sentence_windows(paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for p in paragraphs:
        sents = split_sentences(p["text"])
        for i, sent in enumerate(sents):
            lo = max(0, i - SENTENCE_WINDOW)
            hi = min(len(sents), i + SENTENCE_WINDOW + 1)
            text = clean(" ".join(sents[lo:hi]))
            if len(text) < 50:
                continue
            windows.append({
                "window_id": len(windows),
                "text": text[:MAX_SENTENCE_WINDOW_CHARS],
                "center_sentence": sent,
                "page": p.get("page"),
                "section": p.get("section"),
                "idx": p.get("idx"),
            })
    return windows

# ------------------------------ recuperación --------------------------------
def context_entry(source: str, node_or_text: Any, score: float = 0.0, **extra) -> dict[str, Any]:
    if isinstance(node_or_text, dict):
        e = {
            "source": source,
            "text": truncate(node_or_text.get("text", "")),
            "idx": node_or_text.get("idx"),
            "page": node_or_text.get("page"),
            "section": node_or_text.get("section"),
            "score": score,
        }
    else:
        e = {"source": source, "text": truncate(str(node_or_text)), "score": score}
    e.update(extra)
    return e


def find_caption_anchor_idx(seq: list[dict[str, Any]], caption: str, page: int | None) -> int | None:
    prefix = clean(caption)[:120]
    if prefix:
        for n in seq:
            if n["type"] in {"text", "title"} and prefix in n["text"]:
                return n["idx"]
    if page is not None:
        candidates = [n for n in seq if n["type"] == "text" and n.get("page") == page]
        if candidates:
            return candidates[0]["idx"]
    return None


def find_reference_mentions(paragraphs: list[dict[str, Any]], figure_label: str | None, target_page: int | None = None, limit: int = 12) -> list[dict[str, Any]]:
    pat = figure_reference_regex(figure_label)
    target_id = figure_identity(figure_label)
    if not pat:
        return []
    hits: list[dict[str, Any]] = []
    for p in paragraphs:
        text = p["text"]
        if not pat.search(text):
            continue
        # Evita captions de otra figura que contienen "Fig. N" en un "see Fig. N".
        ids = caption_header_identities(text)
        if ids and target_id not in ids:
            continue
        e = dict(p)
        if target_page is not None and p.get("page") is not None:
            e["page_distance"] = abs(int(p["page"]) - int(target_page))
        else:
            e["page_distance"] = None
        hits.append(e)
    hits.sort(key=lambda x: (999 if x.get("page_distance") is None else x["page_distance"], x.get("idx") or 0))
    return hits[:limit]


def neighboring_paragraphs(paragraphs: list[dict[str, Any]], anchor_idx: int | None, page: int | None, n_before: int = 2, n_after: int = 4) -> dict[str, list[dict[str, Any]]]:
    if anchor_idx is None:
        return {"before": [], "after": []}
    candidates = [p for p in paragraphs if page is None or p.get("page") == page]
    if not candidates:
        candidates = paragraphs
    pos = min(range(len(candidates)), key=lambda i: abs((candidates[i].get("idx") or 0) - anchor_idx))
    return {"before": candidates[max(0, pos - n_before):pos], "after": candidates[pos + 1:pos + 1 + n_after]}


def bm25_search_windows(windows: list[dict[str, Any]], query: str, top_k: int = BM25_TOP_K, item_page: int | None = None, page_radius: int = BM25_PAGE_RADIUS) -> list[dict[str, Any]]:
    q = tokenize(query)
    if not q or not windows:
        return []
    # Acota el corpus a ±page_radius páginas de la figura ANTES de rankear, para
    # que un match léxico lejano (otra sección/figura) no entre solo por coincidencia.
    scoped = windows
    if item_page is not None:
        scoped = [
            w for w in windows
            if w.get("page") is None or abs(int(w["page"]) - int(item_page)) <= page_radius
        ]
        # Fallback: si la ventana quedó vacía o muy pobre, usa el corpus completo.
        if len(scoped) < max(top_k, 5):
            scoped = windows
    corpus = [tokenize(w["text"]) for w in scoped]
    valid = [(i, c) for i, c in enumerate(corpus) if c]
    if not valid:
        return []
    valid_idx, valid_tokens = zip(*valid)
    if BM25Okapi is not None:
        bm25 = BM25Okapi(list(valid_tokens))
        scores = bm25.get_scores(q)
    else:
        qset = set(q)
        scores = [len(qset & set(toks)) / math.sqrt(len(set(toks)) + 1) for toks in valid_tokens]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[dict[str, Any]] = []
    for r in ranked[:top_k * 2]:
        score = float(scores[r])
        if score <= 0:
            continue
        w = dict(scoped[valid_idx[r]])
        w["score_bm25"] = score
        out.append(w)
        if len(out) >= top_k:
            break
    return out


def section_title_for_anchor(seq: list[dict[str, Any]], anchor_idx: int | None) -> str | None:
    if anchor_idx is None:
        return None
    before = [n for n in seq if (n.get("idx") or 0) <= anchor_idx and n["type"] == "section"]
    return before[-1]["text"] if before else None


def section_context(paragraphs: list[dict[str, Any]], section: str | None, query: str, limit: int = 4) -> list[dict[str, Any]]:
    if not section:
        return []
    same = [p for p in paragraphs if clean(p.get("section") or "") == clean(section)]
    if not same:
        return []
    qset = token_set(query)
    scored = []
    for p in same:
        score = len(token_set(p["text"]) & qset)
        # Da algo de peso a párrafos interpretativos, pero sin exigir frase específica.
        if BODY_LIKE_RE.search(p["text"]):
            score += 2
        scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [p for s, p in scored if s > 0][:limit]


def find_cross_references(text: str) -> list[str]:
    out: list[str] = []
    for m in CROSS_REFERENCE_RE.finditer(text or ""):
        lab = clean(m.group(1))
        if lab not in out:
            out.append(lab)
    return out


def build_caption_lookup(visual_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index captions from the visual manifest by normalized figure identity.

    This is safer for cross-references than retrieving arbitrary body mentions.
    """
    lookup: dict[str, dict[str, Any]] = {}
    for item in visual_items:
        caption = clean(item.get("caption_oficial", ""))
        label = item.get("figure_label_guess") or extract_figure_label(caption, item.get("visible_text_from_crop", ""))
        key = label_key(label)
        if not key or not caption:
            continue
        existing = lookup.get(key)
        # Prefer captions with clean audit and longer text.
        audit = item.get("caption_audit", {}) or {}
        rank = (1 if audit.get("status") == "clean" else 0, len(caption))
        if not existing or rank > existing.get("_rank", (0, 0)):
            lookup[key] = {
                "text": caption,
                "page": item.get("page"),
                "section": "visual_manifest_caption",
                "idx": None,
                "label": label,
                "_rank": rank,
            }
    return lookup


def find_caption_for_label(seq: list[dict[str, Any]], label: str, limit: int = 2) -> list[dict[str, Any]]:
    pat = figure_reference_regex(label)
    if not pat:
        return []
    hits: list[dict[str, Any]] = []
    for n in seq:
        if n["type"] not in {"text", "title"}:
            continue
        t = clean(n["text"])
        # Prefer true caption headers, not arbitrary references.
        if pat.search(t) and CAPTION_HEADER_RE.search(t):
            hits.append(n)
            if len(hits) >= limit:
                break
    return hits

def has_foreign_caption_leak(text: str, target_label: str | None) -> bool:
    target = figure_identity(target_label)
    ids = caption_header_identities(text)
    return bool(target and any(fid != target for fid in ids))


def has_foreign_caption_leak(text: str, target_label: str | None) -> bool:
    target = figure_identity(target_label)
    ids = caption_header_identities(text)
    return bool(target and any(fid != target for fid in ids))


def is_relevant_to_target(entry: dict[str, Any], target_keywords: set[str], target_label: str | None, allowed_ref_keys: set[str]) -> bool:
    """Generic relevance gate to avoid leaking context from other figures/topics."""
    source = entry.get("source")
    text = clean(entry.get("text", ""))
    if source == "official_caption":
        return True
    if source == "cross_reference_caption":
        return True

    target = figure_identity(target_label)
    refs = reference_identities(text)
    ref_keys = {f"{k}:{n}" for k, n in refs}

    # Direct references to the target are relevant.
    if target and target in refs:
        return True

    # If the snippet mostly talks about a different figure that is not an allowed cross-reference, reject.
    foreign_refs = ref_keys - ({f"{target[0]}:{target[1]}"} if target else set()) - allowed_ref_keys
    if foreign_refs and source not in {"cross_reference_mention", "cross_reference_caption"}:
        # Keep only if it also has strong semantic overlap with target caption.
        overlap = semantic_keywords(text) & target_keywords
        return len(overlap) >= 3

    # Semantic overlap with the target caption. Require at least two specific terms.
    # This prevents generic snippets about another figure with words like "genes" or "chromosome".
    if target_keywords:
        overlap = semantic_keywords(text) & target_keywords
        if len(overlap) >= 2:
            return True

    # Nearby context may still be useful if it is short and not figure-specific.
    if source in {"nearby_after", "nearby_before"} and not foreign_refs:
        return len(text.split()) <= 180

    return False


def bad_entry(entry: dict[str, Any], target_label: str | None, target_keywords: set[str] | None = None, allowed_ref_keys: set[str] | None = None) -> bool:
    t = clean(entry.get("text", ""))
    if not t or len(t) < 50:
        return True
    if NOISE_RE.search(t[:700]):
        return True
    if BAD_SECTION_RE.search(clean(entry.get("section") or "")):
        return True
    if has_foreign_caption_leak(t, target_label):
        return True
    if target_keywords is not None and allowed_ref_keys is not None:
        if not is_relevant_to_target(entry, target_keywords, target_label, allowed_ref_keys):
            return True
    return False

def score_entry(entry: dict[str, Any], query: str, item_page: int | None, target_keywords: set[str] | None = None) -> float:
    source_weight = {
        "official_caption": 100.0,
        "exact_figure_reference": 85.0,
        "nearby_after": 72.0,
        "nearby_before": 45.0,
        "section_context": 66.0,
        "bm25_sentence_window": 62.0,
        "cross_reference_caption": 58.0,
        "cross_reference_mention": 52.0,
    }
    score = source_weight.get(entry.get("source"), 1.0)
    score += min(5.0, len(token_set(entry.get("text", "")) & token_set(query)) * 0.25)
    if target_keywords:
        score += min(6.0, len(semantic_keywords(entry.get("text", "")) & target_keywords) * 1.0)
    if entry.get("score_bm25"):
        score += min(4.0, math.log1p(float(entry["score_bm25"])) * 1.2)
    if item_page is not None and entry.get("page") is not None:
        dist = abs(int(entry["page"]) - int(item_page))
        if dist > 3:
            score -= min(15.0, dist * 2.0)
    return score


def select_context(entries: list[dict[str, Any]], query: str, target_label: str | None, item_page: int | None, target_keywords: set[str], allowed_ref_keys: set[str]) -> list[dict[str, Any]]:
    filtered = [e for e in entries if not bad_entry(e, target_label, target_keywords, allowed_ref_keys)]
    filtered = dedupe_entries(filtered)
    for e in filtered:
        e["prompt_score"] = score_entry(e, query, item_page, target_keywords)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for e in filtered:
        by_source.setdefault(e["source"], []).append(e)
    for src in by_source:
        by_source[src].sort(key=lambda x: -x["prompt_score"])

    selected: list[dict[str, Any]] = []
    quotas = [
        ("official_caption", 1),
        ("exact_figure_reference", 3),
        ("nearby_after", 2),
        ("section_context", 3),
        ("bm25_sentence_window", 4),
        ("cross_reference_caption", 3),
        ("cross_reference_mention", 0),
        ("nearby_before", 1),
    ]
    for src, n in quotas:
        selected.extend(by_source.get(src, [])[:n])
    selected = dedupe_entries(selected)
    if len(selected) < MAX_CONTEXT_PASSAGES_FOR_PROMPT:
        for e in sorted(filtered, key=lambda x: -x["prompt_score"]):
            if e not in selected:
                selected.append(e)
            if len(selected) >= MAX_CONTEXT_PASSAGES_FOR_PROMPT:
                break
    return sorted(selected[:MAX_CONTEXT_PASSAGES_FOR_PROMPT], key=lambda x: -x.get("prompt_score", 0))


def build_hybrid_context(seq: list[dict[str, Any]], paragraphs: list[dict[str, Any]], windows: list[dict[str, Any]], item: dict[str, Any], caption_lookup: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    caption = clean(item.get("caption_oficial", ""))
    visible_raw = clean(item.get("visible_text_from_crop", ""))
    visible_struct = item.get("visible_text_structured", {}) or {}
    figure_label = item.get("figure_label_guess") or extract_figure_label(caption, visible_raw)
    item_page = item.get("page")
    anchor_idx = find_caption_anchor_idx(seq, caption, item_page)
    sec_title = section_title_for_anchor(seq, anchor_idx)

    query_parts = [figure_label or "", caption]
    # Agrega labels estructuradas del gráfico, no OCR crudo completo.
    if visible_struct:
        query_parts.extend(visible_struct.get("labels_candidates", [])[:25])
    else:
        query_parts.append(visible_raw[:600])
    query = clean(" ".join(query_parts))
    target_keywords = semantic_keywords(caption)

    entries: list[dict[str, Any]] = []
    if caption:
        entries.append(context_entry("official_caption", caption, score=3.0, page=item_page, section=sec_title))

    for p in find_reference_mentions(paragraphs, figure_label, target_page=item_page):
        entries.append(context_entry("exact_figure_reference", p, score=2.8))

    neigh = neighboring_paragraphs(paragraphs, anchor_idx, item_page)
    for p in neigh["before"]:
        entries.append(context_entry("nearby_before", p, score=1.6))
    for p in neigh["after"]:
        entries.append(context_entry("nearby_after", p, score=1.9))

    for p in section_context(paragraphs, sec_title, query, limit=6):
        entries.append(context_entry("section_context", p, score=2.2))

    for w in bm25_search_windows(windows, query, top_k=BM25_TOP_K, item_page=item_page):
        entries.append(context_entry("bm25_sentence_window", w, score=w.get("score_bm25", 0), score_bm25=w.get("score_bm25", 0), window_id=w.get("window_id")))

    # Referencias cruzadas desde la caption o texto visible: útiles para Extended Data.
    # Preferimos captions limpias del visual_manifest; solo si no existe, buscamos en el texto.
    cross_refs = find_cross_references(caption) + [r for r in find_cross_references(visible_raw) if r not in find_cross_references(caption)]
    allowed_ref_keys = {label_key(r) for r in cross_refs if label_key(r)}
    caption_lookup = caption_lookup or {}
    for ref in cross_refs[:4]:
        rkey = label_key(ref)
        added_caption = False
        if rkey and rkey in caption_lookup:
            citem = caption_lookup[rkey]
            entries.append(context_entry("cross_reference_caption", citem, score=2.6, referenced_label=ref))
            added_caption = True
        if not added_caption:
            for n in find_caption_for_label(seq, ref, limit=2):
                entries.append(context_entry("cross_reference_caption", n, score=2.0, referenced_label=ref))
                added_caption = True
        # Body mentions of referenced figures are risky; include only as fallback.
        if not added_caption:
            for p in find_reference_mentions(paragraphs, ref, target_page=item_page, limit=2):
                entries.append(context_entry("cross_reference_mention", p, score=1.0, referenced_label=ref))

    prioritized = select_context(entries, query, figure_label, item_page, target_keywords, allowed_ref_keys)

    # Auditoría de contexto.
    cap_audit = item.get("caption_audit", {}) or {}
    vis_audit = item.get("visible_text_audit", {}) or {}
    non_caption = [e for e in prioritized if e.get("source") != "official_caption"]
    author_entries = [e for e in non_caption if e.get("source") in {"exact_figure_reference", "nearby_after", "section_context", "bm25_sentence_window"}]
    foreign_leaks = [e for e in prioritized if has_foreign_caption_leak(e.get("text", ""), figure_label)]
    author_status = "sufficient" if len(author_entries) >= 2 else ("partial" if author_entries else "missing")
    cap_status = cap_audit.get("status") or ("clean" if caption else "missing")
    ocr_status = vis_audit.get("status") or "unknown"

    quality = "high"
    issues: list[str] = []
    if cap_status != "clean":
        issues.append(f"caption_{cap_status}")
    if author_status == "missing":
        issues.append("author_framing_missing")
    elif author_status == "partial":
        issues.append("author_framing_partial")
    if ocr_status in {"noisy", "contaminated"}:
        issues.append(f"visible_text_{ocr_status}")
    if foreign_leaks:
        issues.append("foreign_figure_leak_detected")
    if len(issues) >= 3 or cap_status in {"contaminated", "missing"}:
        quality = "low"
    elif issues:
        quality = "medium"

    return {
        "figure_label": figure_label,
        "anchor_idx": anchor_idx,
        "section_title": sec_title,
        "caption": caption,
        "visible_text_from_crop": visible_raw,
        "visible_text_structured": visible_struct,
        "nearby_before": [context_entry("nearby_before", n) for n in neigh["before"]],
        "nearby_after": [context_entry("nearby_after", n) for n in neigh["after"]],
        "prioritized_context": prioritized,
        "context_quality": {
            "quality": quality,
            "issues": issues,
            "caption_status": cap_status,
            "caption_audit": cap_audit,
            "visible_text_status": ocr_status,
            "visible_text_audit": vis_audit,
            "author_framing_status": author_status,
            "num_author_framing_entries": len(author_entries),
            "num_prioritized_context_passages": len(prioritized),
            "foreign_figure_leak_count": len(foreign_leaks),
            "cross_references_detected": cross_refs,
        },
    }

# -------------------------------- proceso -----------------------------------
def process_context(docling_json: Path, manifest_json: Path) -> None:
    stem = docling_json.stem.replace(".docling", "")
    out_dir = docling_json.parent
    with open(manifest_json, "r", encoding="utf-8") as f:
        visual_items = json.load(f)

    doc = DoclingDocument.load_from_json(docling_json)
    seq = linearize_with_layout(doc)
    paragraphs = paragraph_nodes(seq)
    windows = make_sentence_windows(paragraphs)
    abstract = extract_abstract(seq)

    caption_lookup = build_caption_lookup(visual_items)

    results: list[dict[str, Any]] = []
    for item in visual_items:
        hybrid = build_hybrid_context(seq, paragraphs, windows, item, caption_lookup=caption_lookup)
        hybrid_passages = [e["text"] for e in hybrid["prioritized_context"]]
        results.append({
            "item_id": item["item_id"],
            "crop_path": item["crop_path"],
            "page": item.get("page"),
            "caption_oficial": item.get("caption_oficial", ""),
            "caption_audit": item.get("caption_audit", {}),
            "figure_label": hybrid.get("figure_label"),
            "abstract_paper": abstract,
            "visible_text_from_crop": item.get("visible_text_from_crop", ""),
            "visible_text_structured": item.get("visible_text_structured", {}),
            "visible_text_audit": item.get("visible_text_audit", {}),
            "bbox_original": item.get("bbox_original"),
            "bbox_expanded": item.get("bbox_expanded"),
            "bbox_rendered": item.get("bbox_rendered"),
            "context_by_source": hybrid,
            "bm25_top_passages": hybrid_passages,
            "context_retrieval_metadata": {
                "version": "context_v3_sentence_windows_quality_audit",
                "strategies": [
                    "official_caption",
                    "exact_figure_reference",
                    "nearby_paragraphs",
                    "same_section_context",
                    "bm25_sentence_windows",
                    "cross_reference_caption_and_mentions",
                    "quality_audit",
                ],
                "num_paragraphs": len(paragraphs),
                "num_sentence_windows": len(windows),
                "num_prioritized_context_passages": len(hybrid_passages),
                "context_quality": hybrid.get("context_quality", {}),
            },
        })

    out_json = out_dir / "context.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("✅ [%s] Contexto construido para %d imágenes → %s", stem, len(results), out_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True, help="Directorio con .docling.json y visual_manifest.json")
    args = parser.parse_args()
    base = Path(args.json_dir)
    docling_files = list(base.glob("*.docling.json"))
    if not docling_files:
        raise FileNotFoundError(f"No encontré .docling.json en {base}")
    manifest = base / "visual_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No encontré visual_manifest.json en {base}")
    process_context(docling_files[0], manifest)


if __name__ == "__main__":
    main()