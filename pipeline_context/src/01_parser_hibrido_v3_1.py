#!/usr/bin/env python3
"""
01_parser_hibrido_v3_1_1.py
=======================
Parsea un PDF científico y genera:
  - <paper>.docling.json
  - visual_manifest.json
  - crops/*.png

Uso:
    python 01_parser_hibrido_v3_1_1.py --pdf paper.pdf --out salida/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import fitz
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.base import BoundingBox, CoordOrigin, ImageRefMode
from docling_core.types.doc.document import PictureItem, TableItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("parser_hibrido_v3_1")

# --------------------------- parámetros generales ---------------------------
RENDER_DPI = 300
CROP_PADDING_PT = 5.0
TEXT_LABEL_MARGIN = 50.0
CLUSTER_GAP_FRAC = 0.06
MIN_PERP_OVERLAP = 0.15
MIN_AREA_FRAC = 0.015
MAX_ASPECT_RATIO = 14.0
HEADER_FOOTER_BAND = 0.08
MAX_VISIBLE_TEXT_CHARS = 4000
MAX_CAPTION_CHARS = 5000

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
NEXT_CAPTION_HEADER_RE = CAPTION_HEADER_RE
PANEL_MARKER_RE = re.compile(r"(?i)(?:^|[.;]\s+|\|\s*)([a-z])\s*[,\u2013-]\s+")

# Heurística genérica de posible frase de cuerpo. No contiene contenido específico del paper.
BODY_LIKE_RE = re.compile(
    r"(?i)\b("
    r"we\s+(?:show|find|found|identified|investigated|used|selected|classified|"
    r"observed|tested|performed|analyzed|analysed|conclude|note|emphasize)|"
    r"in\s+(?:this|the)\s+(?:study|paper|work)|"
    r"our\s+(?:results|analysis|data|study)|"
    r"therefore|however|nevertheless|in\s+contrast|as\s+expected"
    r")\b"
)

LOW_VALUE_TEXT_RE = re.compile(
    r"(?i)\b("
    r"received:|accepted:|published online:|check for updates|doi\.org|"
    r"correspondence|e-?mail|copyright|creative commons|"
    r"(?:nature|science|cell)\s+[a-z]+\s*\|"
    r")\b"
)

# Footers/journal headers that can be accidentally appended to captions.
# Keep this generic: strip only when they occur at the END of a candidate caption.
CAPTION_TRAILING_FOOTER_RE = re.compile(
    r"(?i)\s*(?:Nature\s+[A-Za-z]+|Science|Cell|PNAS|eLife|BMC\s+[A-Za-z]+)"
    r"(?:\s*\|\s*[^.]{0,120})?\s*$"
)

# Panel markers in captions can be individual (a, b,) or ranges (a–d, e-h, i-l).
PANEL_RANGE_RE = re.compile(r"(?i)\b([a-z])\s*[\u2013-]\s*([a-z])\s*,")

# ------------------------------ utilidades bbox -----------------------------
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def bbox_topleft(prov, page_height: float) -> BoundingBox:
    bb = prov.bbox
    if bb.coord_origin == CoordOrigin.BOTTOMLEFT:
        bb = bb.to_top_left_origin(page_height=page_height)
    return bb


def bbox_to_list(bb: BoundingBox | fitz.Rect) -> list[float]:
    if isinstance(bb, fitz.Rect):
        return [float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1)]
    return [float(bb.l), float(bb.t), float(bb.r), float(bb.b)]


def block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        spans = [s.get("text", "") for s in line.get("spans", [])]
        txt = "".join(spans).strip()
        if txt:
            lines.append(txt)
    return "\n".join(lines).strip()


def vertical_gap(a: BoundingBox, b: BoundingBox) -> float:
    return (b.t - a.b) if a.b <= b.t else (a.t - b.b if b.b <= a.t else 0.0)


def horizontal_gap(a: BoundingBox, b: BoundingBox) -> float:
    return (b.l - a.r) if a.r <= b.l else (a.l - b.r if b.r <= a.l else 0.0)


def perp_overlap_frac(a: BoundingBox, b: BoundingBox, axis: str) -> float:
    if axis == "vertical":
        lo, hi = max(a.l, b.l), min(a.r, b.r)
        span = min(a.r - a.l, b.r - b.l)
    else:
        lo, hi = max(a.t, b.t), min(a.b, b.b)
        span = min(a.b - a.t, b.b - b.t)
    return max(0.0, hi - lo) / span if span > 0 else 0.0


def are_contiguous(a: BoundingBox, b: BoundingBox, page_w: float, page_h: float) -> bool:
    if vertical_gap(a, b) <= CLUSTER_GAP_FRAC * page_h and perp_overlap_frac(a, b, "vertical") >= MIN_PERP_OVERLAP:
        return True
    if horizontal_gap(a, b) <= CLUSTER_GAP_FRAC * page_w and perp_overlap_frac(a, b, "horizontal") >= MIN_PERP_OVERLAP:
        return True
    return False


def is_garbage_bbox(bbox: BoundingBox, page_w: float, page_h: float) -> bool:
    w, h = bbox.r - bbox.l, bbox.b - bbox.t
    if w <= 0 or h <= 0:
        return True
    area_frac = (w * h) / (page_w * page_h)
    if area_frac < MIN_AREA_FRAC:
        return True
    if max(w / h, h / w) > MAX_ASPECT_RATIO:
        return True
    if (bbox.t < HEADER_FOOTER_BAND * page_h or bbox.b > (1 - HEADER_FOOTER_BAND) * page_h) and area_frac < 0.05:
        return True
    return False

# --------------------------- caption y auditoría ----------------------------
def resolve_docling_caption(item, doc) -> str:
    parts: list[str] = []
    if hasattr(item, "captions") and item.captions:
        for cap_ref in item.captions:
            try:
                node = cap_ref.resolve(doc)
                if getattr(node, "text", None):
                    parts.append(node.text.strip())
            except Exception:
                continue
    return clean_text(" ".join(parts))


def guess_figure_label(text: str) -> str | None:
    m = FIG_LABEL_RE.search(text or "")
    return clean_text(m.group(1)) if m else None


def clean_caption_artifacts(text: str) -> str:
    """Remove generic page/journal artifacts that are not part of captions."""
    text = clean_text(text)
    # Remove page/footer tails only at the end; do not remove journal names inside real text.
    text = CAPTION_TRAILING_FOOTER_RE.sub("", text).strip()
    text = re.sub(r"(?i)\s+Volume\s+\d+\s*\|.*$", "", text).strip()
    return clean_text(text)


def figure_identity(label: str | None) -> tuple[str, str] | None:
    if not label:
        return None
    s = clean_text(label).lower()
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


def caption_header_identities(text: str) -> list[tuple[str, str]]:
    """Return identities of explicit caption headers only (e.g. 'Fig. 2 |').

    Important: a phrase such as 'see Fig. 2a-c' is a legitimate cross-reference,
    not contamination. Only 'Fig. X |' / 'Extended Data Fig. X |' counts here.
    """
    ids: list[tuple[str, str]] = []
    for m in CAPTION_HEADER_RE.finditer(text or ""):
        header = m.group(0).split("|")[0]
        fid = figure_identity(header)
        if fid:
            ids.append(fid)
    return ids


def count_panel_markers(caption: str) -> int:
    labels: set[str] = set()
    for m in PANEL_MARKER_RE.finditer(caption):
        lab = m.group(1).lower()
        if "a" <= lab <= "z":
            labels.add(lab)
    for m in PANEL_RANGE_RE.finditer(caption):
        start, end = m.group(1).lower(), m.group(2).lower()
        if 0 <= ord(end) - ord(start) <= 25:
            for c in range(ord(start), ord(end) + 1):
                labels.add(chr(c))
    return len(labels)


def normalize_label_to_header_regex(label: str) -> re.Pattern | None:
    label = clean_text(label)
    if not label:
        return None
    # Convierte "Extended Data Fig. 1" en patrón flexible que exige "|".
    tokens = re.split(r"\s+", label)
    pattern = r"\s*".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?i)\b{pattern}\s*\|")


def page_text_blocks_sorted(page: fitz.Page) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        text = block_text(b)
        if not text:
            continue
        rect = fitz.Rect(b["bbox"])
        blocks.append({"text": text, "rect": rect, "bbox": bbox_to_list(rect)})
    blocks.sort(key=lambda x: (round(x["rect"].y0, 1), round(x["rect"].x0, 1)))
    return blocks


def split_sentences(text: str) -> list[str]:
    # Split simple y conservador; evita partir justo después de "Fig." o "Dr.".
    text = clean_text(text)
    text = re.sub(r"\b(Fig|fig|Dr|Prof|Ref|refs)\.\s", lambda m: m.group(0).replace(". ", "<DOT> "), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def recover_caption_from_page(page: fitz.Page, current_caption: str) -> str:
    """Recupera caption sin reglas específicas del paper.

    Usa la cabecera de figura y acumula frases que parecen caption. Si aparece
    una oración claramente de cuerpo entre medio, no se corta necesariamente:
    se salta. Esto ayuda cuando el PDF tiene dos columnas y la lectura se mezcla.
    """
    current_caption = clean_caption_artifacts(current_caption)
    label = guess_figure_label(current_caption)
    if not label:
        return clean_caption_artifacts(current_caption)[:MAX_CAPTION_CHARS]

    header_re = normalize_label_to_header_regex(label)
    if not header_re:
        return clean_caption_artifacts(current_caption)[:MAX_CAPTION_CHARS]

    blocks = page_text_blocks_sorted(page)
    start_i = None
    for i, blk in enumerate(blocks):
        if header_re.search(clean_text(blk["text"])):
            start_i = i
            break
    if start_i is None:
        # Fallback: si Docling ya dio caption, usarla auditada.
        return clean_caption_artifacts(current_caption)[:MAX_CAPTION_CHARS]

    start_y = blocks[start_i]["rect"].y0
    page_h = page.rect.height
    kept: list[str] = []
    prev_kept = False

    for j in range(start_i, len(blocks)):
        blk = blocks[j]
        txt = clean_text(blk["text"])
        if not txt:
            continue

        # Fin si empieza otra caption.
        if j != start_i and NEXT_CAPTION_HEADER_RE.search(txt):
            break
        # Evita ir demasiado lejos verticalmente.
        if blk["rect"].y0 - start_y > page_h * 0.60:
            break
        if LOW_VALUE_TEXT_RE.search(txt):
            continue

        for sent in split_sentences(txt):
            if not sent:
                continue
            is_header = bool(header_re.search(sent))
            has_panel = bool(PANEL_MARKER_RE.search(sent))
            body_like = bool(BODY_LIKE_RE.search(sent))
            starts_lower = bool(sent[:1].islower())
            previous_unclosed = bool(kept and clean_text(" ".join(kept))[-1:] not in ".;:!?)")
            continuation = prev_kept and (starts_lower or previous_unclosed) and not body_like

            if is_header or has_panel or continuation or (prev_kept and not body_like and len(sent.split()) <= 45):
                kept.append(sent)
                prev_kept = True
            else:
                prev_kept = False

            if len(clean_text(" ".join(kept))) >= MAX_CAPTION_CHARS:
                return clean_caption_artifacts(" ".join(kept))[:MAX_CAPTION_CHARS]

    recovered = clean_caption_artifacts(" ".join(kept))
    # Usa la versión recuperada si aporta algo; si no, usa Docling.
    if len(recovered) >= max(80, len(current_caption) * 0.7):
        return recovered[:MAX_CAPTION_CHARS]
    return clean_caption_artifacts(current_caption)[:MAX_CAPTION_CHARS]


def caption_audit(caption: str, label: str | None = None) -> dict[str, Any]:
    caption = clean_caption_artifacts(caption)
    target_label = label or guess_figure_label(caption)
    target_id = figure_identity(target_label)

    header_ids = caption_header_identities(caption)
    # Foreign figure contamination means another caption header is embedded,
    # not a legitimate cross-reference like "see Fig. 2a-c".
    foreign_headers = [fid for fid in header_ids if target_id and fid != target_id]

    # Cross-references are informative metadata, not contamination.
    all_labels = [clean_text(x) for x in FIG_LABEL_RE.findall(caption)]
    cross_refs = []
    for lab in all_labels:
        if target_label and clean_text(lab).lower() == clean_text(target_label).lower():
            continue
        # If it is not an explicit foreign caption header, keep as cross-reference only.
        if lab not in cross_refs:
            cross_refs.append(lab)

    has_header = bool(header_ids and (not target_id or target_id in header_ids))
    has_body_leak = bool(BODY_LIKE_RE.search(caption))
    has_truncation_marker = "[TRUNCATED]" in caption
    ends_cleanly = bool(re.search(r"[.!?)]\s*$", caption))
    n_panel_markers = count_panel_markers(caption)
    word_count = len(caption.split())

    status = "clean"
    issues: list[str] = []
    if not caption:
        status = "missing"; issues.append("caption_missing")
    elif not has_header:
        status = "uncertain"; issues.append("no_caption_header")

    if caption and (has_truncation_marker or not ends_cleanly):
        status = "truncated"; issues.append("caption_may_be_truncated")
    if has_body_leak:
        status = "contaminated"; issues.append("caption_has_body_like_sentence")
    if foreign_headers:
        status = "contaminated"; issues.append("caption_contains_foreign_caption_header")
    if word_count > 650:
        status = "contaminated"; issues.append("caption_unusually_long")

    return {
        "status": status,
        "issues": issues,
        "has_header": has_header,
        "ends_cleanly": ends_cleanly,
        "n_panel_markers": n_panel_markers,
        "foreign_caption_headers": [f"{k}:{n}" for k, n in foreign_headers],
        "cross_references": cross_refs,
        "word_count": word_count,
    }

# ------------------------ crop y texto visible estructurado -----------------
def expand_bbox_with_nearby_labels(bbox: list[float], page: fitz.Page) -> tuple[list[float], list[dict[str, Any]]]:
    x0, y0, x1, y1 = bbox
    pw, ph = page.rect.width, page.rect.height
    search_rect = fitz.Rect(max(0, x0 - TEXT_LABEL_MARGIN), max(0, y0 - TEXT_LABEL_MARGIN),
                            min(pw, x1 + TEXT_LABEL_MARGIN), min(ph, y1 + TEXT_LABEL_MARGIN))
    nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    used: list[dict[str, Any]] = []

    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        rect = fitz.Rect(b["bbox"])
        if not rect.intersects(search_rect):
            continue
        text = clean_text(block_text(b))
        if not text or CAPTION_HEADER_RE.search(text) or LOW_VALUE_TEXT_RE.search(text):
            continue
        # Evita absorber párrafos largos del cuerpo. Mantén etiquetas/ejes/leyendas.
        if len(text.split()) > 28 and (rect.width > pw * 0.25 or rect.height > ph * 0.08):
            continue
        nx0, ny0, nx1, ny1 = min(nx0, rect.x0), min(ny0, rect.y0), max(nx1, rect.x1), max(ny1, rect.y1)
        used.append({"text": text, "bbox": bbox_to_list(rect)})
    return [nx0, ny0, nx1, ny1], used


def is_visible_text_noise(text: str, caption_label: str | None = None) -> bool:
    t = clean_text(text)
    if not t:
        return True
    if CAPTION_HEADER_RE.search(t):
        return True
    if LOW_VALUE_TEXT_RE.search(t):
        return True
    if len(t.split()) > 45 and not re.search(r"[%=×≤≥0-9]|\b[A-Z]{2,}\b", t):
        return True
    # Si menciona muchas veces Fig/Table, suele ser caption/cuerpo, no etiqueta visual.
    if len(FIG_LABEL_RE.findall(t)) >= 2:
        return True
    return False


def extract_text_blocks_in_rect(page: fitz.Page, rect: fitz.Rect, caption_label: str | None, max_chars: int = MAX_VISIBLE_TEXT_CHARS) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    total = 0
    sorted_blocks = sorted(
        [b for b in page.get_text("dict").get("blocks", []) if b.get("type") == 0],
        key=lambda bb: (round(fitz.Rect(bb["bbox"]).y0, 1), round(fitz.Rect(bb["bbox"]).x0, 1)),
    )
    for b in sorted_blocks:
        brect = fitz.Rect(b["bbox"])
        if not brect.intersects(rect):
            continue
        text = clean_text(block_text(b))
        if is_visible_text_noise(text, caption_label=caption_label):
            continue
        if total >= max_chars:
            break
        remaining = max_chars - total
        if len(text) > remaining:
            text = text[:remaining] + " [TRUNCATED]"
        total += len(text)
        blocks.append({"text": text, "bbox": bbox_to_list(brect)})
    return blocks


def structure_visible_text(raw_text: str) -> dict[str, Any]:
    raw = clean_text(raw_text)
    panel_labels = sorted(set(re.findall(r"(?:^|\s)([a-z])(?=\s)", f" {raw} ")))
    # Limita a secuencia contigua desde a para evitar falsos positivos.
    contiguous = []
    for c in range(ord("a"), ord("z") + 1):
        if chr(c) in panel_labels:
            contiguous.append(chr(c))
        elif contiguous:
            break
    numbers = re.findall(r"(?<![A-Za-z])(?:≤|≥)?\d+(?:[,.]\d+)?(?:\s*[×x]\s*10\s*\d+|%)?", raw)
    # Candidatos a labels: frases con letras y pocas palabras, no demasiado largas.
    candidates = []
    for part in re.split(r"\s{2,}|\n|;", raw_text or ""):
        p = clean_text(part)
        if 2 <= len(p) <= 80 and re.search(r"[A-Za-z]", p) and len(p.split()) <= 8:
            if not CAPTION_HEADER_RE.search(p):
                candidates.append(p)
    # Dedup simple.
    out_labels = []
    for c in candidates:
        key = c.lower()
        if key not in {x.lower() for x in out_labels}:
            out_labels.append(c)
        if len(out_labels) >= 40:
            break
    return {
        "panel_labels": contiguous,
        "labels_candidates": out_labels,
        "numeric_values_sample": numbers[:80],
        "raw_text_char_count": len(raw),
    }


def visible_text_audit(raw_text: str) -> dict[str, Any]:
    raw = clean_text(raw_text)
    issues: list[str] = []
    if CAPTION_HEADER_RE.search(raw):
        issues.append("visible_text_contains_caption_header")
    if len(raw.split()) > 500:
        issues.append("visible_text_unusually_long")
    if "[TRUNCATED]" in raw:
        issues.append("visible_text_truncated")
    status = "clean" if not issues else ("noisy" if len(issues) <= 2 else "contaminated")
    return {"status": status, "issues": issues, "word_count": len(raw.split())}

# -------------------------------- main logic --------------------------------
def process_pdf(pdf_path: Path, output_dir: Path, images_scale: float = 2.0) -> None:
    stem = pdf_path.stem
    paper_out_dir = output_dir / stem
    crops_dir = paper_out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    json_path = paper_out_dir / f"{stem}.docling.json"
    manifest_path = paper_out_dir / "visual_manifest.json"

    logger.info("[%s] Parseo estructural con Docling...", stem)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = images_scale
    pipeline_options.do_table_structure = True
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
    doc_result = converter.convert(pdf_path)
    doc = doc_result.document

    raw_items: list[dict[str, Any]] = []
    for docling_idx, (item, _) in enumerate(doc.iterate_items()):
        if isinstance(item, (PictureItem, TableItem)) and item.prov:
            prov = item.prov[0]
            page_h = doc.pages[prov.page_no].size.height
            raw_items.append({
                "docling_idx": docling_idx,
                "item_kind": type(item).__name__,
                "page": prov.page_no,
                "bbox": bbox_topleft(prov, page_h),
                "caption": resolve_docling_caption(item, doc),
            })

    pdf_doc = fitz.open(str(pdf_path))
    by_page: dict[int, list[dict[str, Any]]] = {}
    for r in raw_items:
        by_page.setdefault(r["page"], []).append(r)

    clusters: list[dict[str, Any]] = []
    for page_no, items in by_page.items():
        page_w = doc.pages[page_no].size.width
        page_h = doc.pages[page_no].size.height
        parent = list(range(len(items)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(y)] = find(x)

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if are_contiguous(items[i]["bbox"], items[j]["bbox"], page_w, page_h):
                    union(i, j)

        groups: dict[int, list[dict[str, Any]]] = {}
        for idx, p in enumerate(items):
            groups.setdefault(find(idx), []).append(p)

        for group in groups.values():
            bboxes = [p["bbox"] for p in group]
            captions = [p["caption"] for p in group if p.get("caption")]
            clusters.append({
                "page": page_no,
                "bbox": BoundingBox.enclosing_bbox(bboxes),
                "caption": max(captions, key=len) if captions else "",
                "cluster_size": len(group),
                "item_kinds": sorted(set(p["item_kind"] for p in group)),
                "docling_indices": [p["docling_idx"] for p in group],
                "raw_bboxes": [bbox_to_list(p["bbox"]) for p in group],
                "raw_captions": captions,
            })

    valid_items: list[dict[str, Any]] = []
    for item_idx, cluster in enumerate(clusters):
        page_no = cluster["page"]
        page = pdf_doc[page_no - 1]
        page_w = doc.pages[page_no].size.width
        page_h = doc.pages[page_no].size.height
        if is_garbage_bbox(cluster["bbox"], page_w, page_h):
            continue

        bbox_original = bbox_to_list(cluster["bbox"])
        bbox_expanded, label_blocks = expand_bbox_with_nearby_labels(bbox_original, page)
        clip = fitz.Rect(
            max(0.0, bbox_expanded[0] - CROP_PADDING_PT),
            max(0.0, bbox_expanded[1] - CROP_PADDING_PT),
            min(page.rect.width, bbox_expanded[2] + CROP_PADDING_PT),
            min(page.rect.height, bbox_expanded[3] + CROP_PADDING_PT),
        )

        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0), clip=clip, alpha=False)
        crop_filename = f"{stem}__item{item_idx}.png"
        pix.save(str(crops_dir / crop_filename))

        caption_docling = clean_text(cluster.get("caption", ""))
        caption = recover_caption_from_page(page, caption_docling)
        fig_label = guess_figure_label(caption)
        cap_audit = caption_audit(caption, fig_label)

        visible_blocks = extract_text_blocks_in_rect(page, clip, caption_label=fig_label)
        visible_raw = "\n".join(b["text"] for b in visible_blocks)[:MAX_VISIBLE_TEXT_CHARS]
        visible_structured = structure_visible_text(visible_raw)
        vis_audit = visible_text_audit(visible_raw)

        valid_items.append({
            "item_id": crop_filename,
            "crop_path": f"crops/{crop_filename}",
            "caption_oficial": caption,
            "caption_docling_original": caption_docling,
            "caption_audit": cap_audit,
            "figure_label_guess": fig_label,
            "page": page_no,
            "item_kinds": cluster["item_kinds"],
            "cluster_size": cluster["cluster_size"],
            "docling_indices": cluster["docling_indices"],
            "bbox_original": [float(x) for x in bbox_original],
            "bbox_expanded": [float(x) for x in bbox_expanded],
            "bbox_rendered": bbox_to_list(clip),
            "page_size": {"width": float(page.rect.width), "height": float(page.rect.height)},
            "render_dpi": RENDER_DPI,
            "raw_bboxes": cluster["raw_bboxes"],
            "raw_captions": cluster["raw_captions"],
            "label_text_blocks_used_for_expansion": label_blocks,
            "visible_text_blocks": visible_blocks,
            "visible_text_from_crop": visible_raw,
            "visible_text_structured": visible_structured,
            "visible_text_audit": vis_audit,
        })

    pdf_doc.close()
    doc.save_as_json(json_path, image_mode=ImageRefMode.PLACEHOLDER)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid_items, f, indent=2, ensure_ascii=False)
    logger.info("✅ [%s] Manifiesto guardado con %d elementos: %s", stem, len(valid_items), manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--images-scale", type=float, default=2.0)
    args = parser.parse_args()
    process_pdf(Path(args.pdf), Path(args.out), images_scale=args.images_scale)


if __name__ == "__main__":
    main()
