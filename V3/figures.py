"""
Scientific Figure Extractor
============================

Primary:  Docling (layout-aware, multi-column)
Fallback: PyMuPDF (caption-based, used when Docling returns 0 results)

Produces per-figure/table PNG images and figures.json.

Uso:
    python figures.py paper.pdf --out out/
    python figures.py paper.pdf --out out/ --dpi 300
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# ─── Configuración por defecto ───────────────────────────────────────────────
DEFAULT_DPI            = 200
DEFAULT_PADDING        = 16
DEFAULT_MAX_HEIGHT     = 9999
CROSS_COLUMN_TOL       = 50
FULL_WIDTH_RATIO       = 0.45
DOC_SMALL_AREA         = 20000
DOC_SMALL_WIDTH        = 180
DOC_SMALL_HEIGHT       = 120

CAPTION_RE = re.compile(
    r'^\s*(Figure|Fig\.?|Table|Extended\s+Data\s+Fig(?:ure|\.)?)\s*(\d+)',
    re.IGNORECASE,
)


@dataclass
class Caption:
    page:  int
    label: str
    bbox:  tuple
    text:  str
    kind:  str


# ─── Render ──────────────────────────────────────────────────────────────────
def render_region(page, bbox, out_path, dpi):
    clip = fitz.Rect(*bbox)
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    pix.save(str(out_path))
    return clip.width, clip.height


def _docling_skip_reason(kind, bbox, caption_text):
    """Return a skip reason for non-scientific Docling visual artifacts."""
    if kind != "figure":
        return ""
    if (caption_text or "").strip():
        return ""

    x0, y0, x1, y1 = bbox
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    area = width * height

    if area <= DOC_SMALL_AREA:
        return "small_uncaptioned_docling_picture"
    if width <= DOC_SMALL_WIDTH and height <= DOC_SMALL_HEIGHT:
        return "compact_uncaptioned_docling_picture"
    return ""


def _write_skipped_docling_items(out_dir, pdf_path, skipped):
    skipped_path = out_dir / "skipped_figures.json"
    skipped_path.write_text(
        json.dumps({"pdf": str(pdf_path), "total": len(skipped), "items": skipped},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ─── Docling primary extractor ────────────────────────────────────────────────
def _extract_docling(pdf_path, fitz_doc, out_dir, dpi, quiet):
    """Extract figures and tables using Docling layout analysis."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        if not quiet:
            print("  [Docling] not installed, skipping")
        return []

    try:
        if not quiet:
            print(f"  [Docling] converting {pdf_path.name}...")
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        dloc_doc = result.document
    except Exception as e:
        if not quiet:
            print(f"  [Docling] conversion failed: {e}")
        return []

    results = []
    skipped = []
    picture_seen = 0
    uncaptained_figure_count = 0

    def _prov_to_clip(prov, page_no):
        fitz_page = fitz_doc[page_no - 1]
        ph = fitz_page.rect.height
        pw = fitz_page.rect.width
        try:
            tl = prov.bbox.to_top_left_origin(ph)
            x0, y0, x1, y1 = tl.l, tl.t, tl.r, tl.b
        except Exception:
            # Manual conversion: Docling default is BOTTOMLEFT origin
            x0, x1 = prov.bbox.l, prov.bbox.r
            y0 = ph - prov.bbox.t
            y1 = ph - prov.bbox.b
        pad = DEFAULT_PADDING
        return (max(0, x0 - pad), max(0, y0 - pad),
                min(pw, x1 + pad), min(ph, y1 + pad)), fitz_page

    def _get_caption(item):
        try:
            texts = []
            for ref in getattr(item, 'captions', []):
                resolved = ref.resolve(dloc_doc) if hasattr(ref, 'resolve') else ref
                t = getattr(resolved, 'text', '') or ''
                if t:
                    texts.append(t)
            return ' '.join(texts)[:500]
        except Exception:
            return ''

    # Extract figures (pictures)
    for pic in dloc_doc.pictures:
        picture_seen += 1
        if not getattr(pic, 'prov', None):
            continue
        prov = pic.prov[0]
        page_no = prov.page_no
        try:
            clip_bbox, fitz_page = _prov_to_clip(prov, page_no)
        except Exception:
            continue
        caption_text = _get_caption(pic)
        m = re.search(r'(Figure|Fig\.?)\s+(\d+)', caption_text, re.IGNORECASE)
        label = f"Figure {m.group(2)}" if m else ""
        skip_reason = _docling_skip_reason("figure", clip_bbox, caption_text)
        if skip_reason:
            x0, y0, x1, y1 = clip_bbox
            skipped_label = label or f"Picture_{picture_seen}"
            skipped.append({
                "label": skipped_label,
                "kind": "figure",
                "page": page_no,
                "bbox": [round(v, 1) for v in clip_bbox],
                "caption": caption_text,
                "reason": skip_reason,
                "bbox_size": [round(max(0, x1 - x0)), round(max(0, y1 - y0))],
            })
            if not quiet:
                print(f"  [Docling] [{skipped_label:14}] p{page_no} -> skip {skip_reason}")
            continue
        if not label:
            uncaptained_figure_count += 1
            label = f"Figure_uncaptioned_{uncaptained_figure_count}"
        safe = label.replace(" ", "_").replace(".", "")
        fname = f"p{page_no:03d}_{safe}.png"
        out_path = out_dir / fname
        try:
            w, h = render_region(fitz_page, clip_bbox, out_path, dpi)
        except Exception as e:
            if not quiet:
                print(f"  [Docling] render failed {label}: {e}")
            continue
        size_kb = out_path.stat().st_size / 1024
        if not quiet:
            print(f"  [Docling] [{label:14}] p{page_no} -> {fname} ({w:.0f}x{h:.0f}px, {size_kb:.1f}KB)")
        results.append({
            "label": label, "kind": "figure", "page": page_no,
            "bbox": [round(v, 1) for v in clip_bbox],
            "caption": caption_text,
            "image_path": str(out_path),
            "image_size": [round(w), round(h)],
        })

    # Extract tables as images (for visual analysis by the analyzer)
    for tbl in dloc_doc.tables:
        if not getattr(tbl, 'prov', None):
            continue
        prov = tbl.prov[0]
        page_no = prov.page_no
        try:
            clip_bbox, fitz_page = _prov_to_clip(prov, page_no)
        except Exception:
            continue
        caption_text = _get_caption(tbl)
        m = re.search(r'Table\s+(\w+)', caption_text, re.IGNORECASE)
        if m:
            label = f"Table {m.group(1)}"
        else:
            label = f"Table_{sum(1 for r in results if r['kind'] == 'table') + 1}"
        safe = label.replace(" ", "_").replace(".", "")
        fname = f"p{page_no:03d}_{safe}.png"
        out_path = out_dir / fname
        try:
            w, h = render_region(fitz_page, clip_bbox, out_path, dpi)
        except Exception as e:
            if not quiet:
                print(f"  [Docling] render failed {label}: {e}")
            continue
        size_kb = out_path.stat().st_size / 1024
        if not quiet:
            print(f"  [Docling] [{label:14}] p{page_no} -> {fname} ({w:.0f}x{h:.0f}px, {size_kb:.1f}KB)")
        results.append({
            "label": label, "kind": "table", "page": page_no,
            "bbox": [round(v, 1) for v in clip_bbox],
            "caption": caption_text,
            "image_path": str(out_path),
            "image_size": [round(w), round(h)],
        })

    _write_skipped_docling_items(out_dir, pdf_path, skipped)
    return results


# ─── PyMuPDF fallback ─────────────────────────────────────────────────────────

def find_captions(doc) -> list[Caption]:
    captions: list[Caption] = []
    for pi, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0 or not b.get("lines"):
                continue
            first_line = "".join(s["text"] for s in b["lines"][0]["spans"]).strip()
            m = CAPTION_RE.match(first_line)
            if not m:
                continue
            bx0, _, bx1, _ = b["bbox"]
            if (bx1 - bx0) < 20:
                continue
            raw = m.group(1).strip().lower()
            if "table" in raw:
                kind, prefix = "table", "Table"
            elif "extended" in raw:
                kind, prefix = "figure", "ExtFig"
            else:
                kind, prefix = "figure", "Figure"
            label = f"{prefix} {m.group(2)}"
            full_text = " ".join(
                "".join(s["text"] for s in line["spans"])
                for line in b["lines"]
            )[:500]
            captions.append(Caption(
                page=pi + 1, label=label, bbox=tuple(b["bbox"]),
                text=full_text, kind=kind,
            ))
    return captions


def get_column(bbox, page_width):
    x0, _, x1, _ = bbox
    mid = page_width / 2
    if x1 < mid + CROSS_COLUMN_TOL and x0 < mid:
        return "left"
    if x0 > mid - CROSS_COLUMN_TOL and x1 > mid:
        return "right"
    return "full"


def in_column(bbox, column, page_width):
    x0, _, x1, _ = bbox
    mid = page_width / 2
    if column == "left":
        return x1 <= mid + CROSS_COLUMN_TOL
    if column == "right":
        return x0 >= mid - CROSS_COLUMN_TOL
    return True


TEXT_LABEL_MARGIN = 60


def find_figure_region(page, caption, prev_boundary_y, max_height):
    pw = page.rect.width
    _, cy0, _, _ = caption.bbox
    column = get_column(caption.bbox, pw)

    visuals = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") == 1:
            visuals.append(tuple(b["bbox"]))
    for d in page.get_drawings():
        r = d.get("rect")
        if r and r.width > 2 and r.height > 2:
            visuals.append((r.x0, r.y0, r.x1, r.y1))

    all_candidates = []
    for bb in visuals:
        bx0, by0, bx1, by1 = bb
        if by0 > cy0 + 5:
            continue
        if by1 < prev_boundary_y - 5:
            continue
        all_candidates.append(bb)

    if not all_candidates:
        return None

    combined_x0 = min(b[0] for b in all_candidates)
    combined_x1 = max(b[2] for b in all_candidates)
    if (combined_x1 - combined_x0) > pw * FULL_WIDTH_RATIO:
        effective_column = "full"
    else:
        effective_column = column

    candidates = [bb for bb in all_candidates if in_column(bb, effective_column, pw)]
    if not candidates:
        candidates = all_candidates

    vx0 = min(b[0] for b in candidates)
    vy0 = min(b[1] for b in candidates)
    vx1 = max(b[2] for b in candidates)
    vy1 = max(b[3] for b in candidates)

    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        bb = tuple(b["bbox"])
        if tuple(b["bbox"]) == tuple(caption.bbox):
            continue
        bx0, by0, bx1, by1 = bb
        if by0 > cy0 + 5:
            continue
        if by1 < prev_boundary_y - 5:
            continue
        if (bx1 - bx0) > pw * 0.35:
            continue
        if (bx1 > vx0 - TEXT_LABEL_MARGIN and bx0 < vx1 + TEXT_LABEL_MARGIN and
                by1 > vy0 - TEXT_LABEL_MARGIN and by0 < vy1 + TEXT_LABEL_MARGIN):
            candidates.append(bb)

    return _combine_bboxes(candidates, page)


def find_table_region(page, caption, prev_boundary_y, max_height):
    pw = page.rect.width
    _, cy0, _, _ = caption.bbox
    column = get_column(caption.bbox, pw)

    text_blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0 or not b.get("lines"):
            continue
        bb = tuple(b["bbox"])
        if bb == caption.bbox:
            continue
        if bb[3] > cy0 + 5:
            continue
        if bb[1] < prev_boundary_y - 5:
            continue
        if cy0 - bb[3] > max_height:
            continue
        if not in_column(bb, column, pw):
            continue
        text_blocks.append(bb)

    if not text_blocks:
        return None

    text_blocks.sort(key=lambda b: -b[3])
    gathered = [text_blocks[0]]
    for b in text_blocks[1:]:
        if gathered[-1][1] - b[3] < 30:
            gathered.append(b)
        else:
            break
    return _combine_bboxes(gathered, page)


def _fallback_region(page, caption, prev_boundary_y, white_threshold=0.97):
    pw, ph = page.rect.width, page.rect.height
    cy0 = caption.bbox[1]
    col = get_column(caption.bbox, pw)
    if col == "left":
        x0, x1 = 0.0, pw / 2
    elif col == "right":
        x0, x1 = pw / 2, pw
    else:
        x0, x1 = 0.0, pw

    clip = fitz.Rect(x0, max(0, prev_boundary_y), x1, cy0)
    if clip.height < 10:
        return None

    mat = fitz.Matrix(72 / 72, 72 / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    samples = pix.samples
    n = pix.width * pix.height
    if n == 0:
        return None
    white = sum(
        1 for i in range(0, len(samples), 3)
        if samples[i] > 240 and samples[i + 1] > 240 and samples[i + 2] > 240
    )
    if white / n > white_threshold:
        return None
    return (
        max(0, x0 - DEFAULT_PADDING),
        max(0, clip.y0 - DEFAULT_PADDING),
        min(pw, x1 + DEFAULT_PADDING),
        min(ph, cy0 + DEFAULT_PADDING),
    )


def _combine_bboxes(boxes, page):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pw, ph = page.rect.width, page.rect.height
    return (
        max(0, x0 - DEFAULT_PADDING),
        max(0, y0 - DEFAULT_PADDING),
        min(pw, x1 + DEFAULT_PADDING),
        min(ph, y1 + DEFAULT_PADDING),
    )


def _extract_pymupdf(doc, out_dir, dpi, max_height, quiet):
    """PyMuPDF caption-based extraction."""
    def log(msg):
        if not quiet:
            print(msg)

    captions = find_captions(doc)
    log(f"Captions detectados: {len(captions)}")

    by_page = {}
    for c in captions:
        by_page.setdefault(c.page, []).append(c)

    results = []
    for cap in captions:
        page = doc[cap.page - 1]

        prev_y = 0.0
        col = get_column(cap.bbox, page.rect.width)
        for other in by_page[cap.page]:
            if other is cap:
                continue
            if get_column(other.bbox, page.rect.width) != col:
                continue
            if other.bbox[3] < cap.bbox[1] and other.bbox[3] > prev_y:
                prev_y = other.bbox[3]

        render_page = page
        if cap.kind == "figure":
            region = find_figure_region(page, cap, prev_y, max_height)
        else:
            region = find_table_region(page, cap, prev_y, max_height)

        if region is None and cap.page < len(doc):
            next_page = doc[cap.page]
            next_caps = by_page.get(cap.page + 1, [])
            boundary_y = min(c.bbox[1] for c in next_caps) if next_caps else next_page.rect.height
            syn_cap = Caption(
                page=cap.page + 1, label=cap.label,
                bbox=(0, boundary_y, next_page.rect.width, boundary_y),
                text=cap.text, kind=cap.kind,
            )
            region = find_figure_region(next_page, syn_cap, 0, max_height)
            if region is not None:
                render_page = next_page
                log(f"  [{cap.label}] p{cap.page} -> figura en página siguiente")

        if region is None:
            region = _fallback_region(page, cap, prev_y)
            if region is not None:
                log(f"  [{cap.label}] p{cap.page} -> fallback render (Form XObject?)")

        if region is None:
            log(f"  [{cap.label}] p{cap.page} - sin región visual, skip")
            continue

        safe = cap.label.replace(" ", "_").replace(".", "")
        fname = f"p{cap.page:03d}_{safe}.png"
        out_path = out_dir / fname
        w, h = render_region(render_page, region, out_path, dpi)
        size_kb = out_path.stat().st_size / 1024
        log(f"  [{cap.label:14}] p{cap.page} -> {fname} ({w:.0f}x{h:.0f}px, {size_kb:.1f}KB)")

        results.append({
            "label":      cap.label,
            "kind":       cap.kind,
            "page":       cap.page,
            "bbox":       [round(v, 1) for v in region],
            "caption":    cap.text,
            "image_path": str(out_path),
            "image_size": [round(w), round(h)],
        })

    return results


# ─── Pipeline ────────────────────────────────────────────────────────────────
def extract_all(pdf_path, out_dir, dpi=DEFAULT_DPI, max_height=DEFAULT_MAX_HEIGHT, quiet=False):
    pdf_path = Path(pdf_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if not quiet:
            print(msg)

    doc = fitz.open(str(pdf_path))

    # Primary: Docling layout analysis
    log("[Docling] primary extraction...")
    results = _extract_docling(pdf_path, doc, out_dir, dpi, quiet)

    if not results:
        log("[PyMuPDF] Docling returned 0 results, falling back...")
        results = _extract_pymupdf(doc, out_dir, dpi, max_height, quiet)

    doc.close()

    meta_path = out_dir / "figures.json"
    meta_path.write_text(
        json.dumps({"pdf": str(pdf_path), "total": len(results), "items": results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"\nMetadata: {meta_path}")
    log(f"Total extraído: {len(results)} de {len(results)}")
    return results


def normalize_figure_items(items: list[dict], paper_dir: str | Path) -> list[dict]:
    paper_dir = Path(paper_dir)
    normalized: list[dict] = []
    for item in items:
        image_path = item.get("image_path") or ""
        try:
            image_name = Path(image_path).name if image_path else ""
        except Exception:
            image_name = str(image_path)
        normalized.append({
            "label": item.get("label", ""),
            "kind": item.get("kind", "figure"),
            "page": item.get("page"),
            "bbox": item.get("bbox", []),
            "caption": item.get("caption", ""),
            "image_path": image_name,
            "image_size": item.get("image_size", []),
        })
    out_path = paper_dir / "figures.normalized.json"
    out_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalized


def extract_figures_for_pdf(
    pdf_path: str | Path,
    paper_dir: str | Path,
    dpi: int = DEFAULT_DPI,
    max_height: int = DEFAULT_MAX_HEIGHT,
    quiet: bool = False,
) -> list[dict]:
    paper_dir = Path(paper_dir)
    results = extract_all(
        pdf_path=str(pdf_path),
        out_dir=str(paper_dir),
        dpi=dpi,
        max_height=max_height,
        quiet=quiet,
    )
    return normalize_figure_items(results, paper_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Extrae figuras y tablas individuales de PDFs científicos.",
    )
    p.add_argument("pdf",  help="ruta al PDF de entrada")
    p.add_argument("--out", default="extracted", help="directorio de salida")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="resolución del render (def: 200)")
    p.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not Path(args.pdf).exists():
        sys.exit(f"PDF no encontrado: {args.pdf}")

    extract_all(args.pdf, args.out, dpi=args.dpi, max_height=args.max_height, quiet=args.quiet)


if __name__ == "__main__":
    main()
