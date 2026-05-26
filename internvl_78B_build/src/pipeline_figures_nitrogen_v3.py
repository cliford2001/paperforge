#!/usr/bin/env python3
"""
src/pipeline_figures_nitrogen_v3.py
──────────────────────────────────────────────────────────────────────────────
Pipeline v3 para papers científicos:

PDF → extracción textual PyMuPDF (con Chandra OCR como fallback para páginas
escaneadas) → extracción/crop de figuras → interpretación multimodal con
InternVL3-14B → validación anti-alucinación → texto enriquecido trazable.

Cambios v3.2 respecto a v3.1 (basados en auditoría con Gemini):
  - FIGURE_SYSTEM_PROMPT: instrucciones explícitas para conteo de paneles
    sin etiqueta visible, transcripción exacta de texto (genome≠gene),
    inclusión de accession IDs en printed_terms, y símbolos matemáticos.
  - build_figure_user_prompt: nuevo campo evidence_source="caption_inferred"
    para observaciones que mezclan lo visible con el caption.
  - HeuristicValidator: acepta "caption_inferred" como evidence_source
    válido; flag medium_risk cuando num_panels>4 y crop heurístico.

Cambios v3.1 respecto a v3:
  - _run_ocr usa pymupdf4llm.to_markdown() por página en lugar de
    page.get_text("text"). Produce markdown estructurado con tablas,
    fórmulas, columnas múltiples y jerarquía de headings. Chandra sigue
    siendo el fallback para páginas escaneadas (char_count < umbral).
  - extract_title y _detect_caption_blocks con filtros de metadatos
    de journal y referencias inline dentro de paréntesis (v3.1).

Cambios v3 respecto a v2:
  - OCR inteligente: PyMuPDF directo si hay texto seleccionable (≥150 chars),
    Chandra solo en páginas escaneadas o sin texto. Elimina el cuello de
    botella de 54 min observado en arabidopsis.
  - Warmup de Chandra antes del loop para evitar spike JIT en página 1.
  - extract_title filtra líneas markdown de imagen (![](...)) y URLs.
  - FIGURE_RE ampliado: Supplementary Fig, Extended Data Fig, Fig. S1, etc.
  - _merge_adjacent_text_blocks: fusiona bloques de texto contiguos para
    capturar captions multi-bloque (frecuente en papers LaTeX).
  - _heuristic_region_from_caption: rechaza cuando y0 queda en 0 o el caption
    está en el primer 15% de la página. Corrige el bug de bbox y0=0.
  - _save_crop: DPI adaptativo — garantiza mínimo 800px en la dimensión corta.
  - Sistema prompt InternVL mejorado para diagramas de vías de señalización.
  - HeuristicValidator refactorizado: fallos duros vs advertencias suaves.
    image_block_crop con términos de nitrógeno visibles no se rechaza aunque
    el modelo declare uncertainty/hallucination high (fix falso negativo en
    diagramas complejos como el de NO₃⁻ crosstalk).
  - Resumen final con tabla alineada.

Estructura esperada:
  data/raw_pdfs/             PDFs de entrada
  data/extracted_figures/    crops PNG por paper
  data/figure_captions/      candidatos de figuras por paper
  data/final_text/           texto OCR, JSON final, texto enriquecido
  models/chandra-ocr-2/      modelo local Chandra OCR 2 (opcional)
  models/internvl3-14b/      modelo local InternVL3-14B

Uso recomendado (dos fases en el mismo job SLURM):

  # Fase 1: OCR + crops (sin GPU pesada, Chandra solo si se necesita)
  python src/pipeline_figures_nitrogen_v3.py \
    --pdf data/raw_pdfs/mi_paper.pdf \
    --no-internvl \
    --no-chandra \
    --dpi 220 \
    --force

  # Fase 2: InternVL sobre crops ya generados
  python src/pipeline_figures_nitrogen_v3.py \
    --pdf data/raw_pdfs/mi_paper.pdf \
    --internvl-only \
    --internvl-mode offline \
    --internvl-model-name internvl3-14b \
    --force
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "raw_pdfs":          PROJECT_ROOT / "data" / "raw_pdfs",
    "extracted_figures": PROJECT_ROOT / "data" / "extracted_figures",
    "figure_captions":   PROJECT_ROOT / "data" / "figure_captions",
    "final_text":        PROJECT_ROOT / "data" / "final_text",
    "chandra":           PROJECT_ROOT / "models" / "chandra-ocr-2",
    "internvl3":         PROJECT_ROOT / "models" / "internvl3-14b",
    "logs":              PROJECT_ROOT / "logs",
}

for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

log_file = PATHS["logs"] / f"pipeline_figures_v3_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)

log = logging.getLogger("pipeline_figures_v3")


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OCRPageResult:
    page: int
    markdown: str = ""
    layout_json: Optional[dict[str, Any]] = None
    raw: Optional[str] = None


@dataclass
class CaptionBlock:
    page: int
    text: str
    figure_label: Optional[str]
    figure_number: Optional[str]
    bbox_pdf: list[float]
    source: str = "pymupdf_text"


@dataclass
class FigureCandidate:
    figure_id: str
    page: int
    caption: str
    figure_label: Optional[str]
    figure_number: Optional[str]
    bbox_pdf: list[float]
    caption_bbox_pdf: Optional[list[float]]
    crop_path: str
    detection_method: str
    crop_quality: str
    crop_includes_caption: bool = False
    page_width_pdf: Optional[float] = None
    page_height_pdf: Optional[float] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class NitrogenRelevance:
    is_relevant: bool = False
    visible_terms: list[str] = field(default_factory=list)
    evidence_source: list[str] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class FigureInterpretation:
    target_figure_only: Optional[bool] = None
    figure_type: str = "unknown"
    visible_text: dict[str, Any] = field(default_factory=dict)
    visual_structure: dict[str, Any] = field(default_factory=dict)
    factual_observations: list[dict[str, str]] = field(default_factory=list)
    nitrogen_relevance: NitrogenRelevance = field(default_factory=NitrogenRelevance)
    not_determinable: list[str] = field(default_factory=list)
    hallucination_risk: str = "high"
    uncertainty_level: str = "high"
    uncertainty_reason: Optional[str] = None
    raw_response: Optional[str] = None
    model: str = "internvl3-78b"


@dataclass
class ValidationReport:
    accepted_for_training: bool = False
    validation_level: str = "high_risk"
    reasons: list[str] = field(default_factory=list)
    validator: str = "heuristic_v2"


@dataclass
class FigureRecord:
    figure_id: str
    page: int
    caption: str
    context_sentences: list[str]
    image_path: str
    candidate: FigureCandidate
    interpretation: Optional[FigureInterpretation] = None
    validation: Optional[ValidationReport] = None


@dataclass
class PaperResult:
    pdf_path: str
    paper_stem: str
    paper_title: str
    full_text: str
    figures: list[FigureRecord] = field(default_factory=list)
    processing_time_s: float = 0.0
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades generales
# ─────────────────────────────────────────────────────────────────────────────

# v3: regex ampliado para Supplementary Fig, Extended Data Fig, Fig. S1, etc.
FIGURE_RE = re.compile(
    r"\b(?:"
    r"Supplementary\s+Fig(?:ure)?s?\.?|"
    r"Extended\s+Data\s+Fig(?:ure)?s?\.?|"
    r"Fig(?:ure)?s?\.?|"
    r"FIGURE|FIG\.?"
    r")\s*"
    r"([0-9]+[A-Za-z]?(?:\s*[,\-–]\s*[0-9]+[A-Za-z]?)*)\s*"
    r"[:\.\-–—]?\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)

NITROGEN_TERMS = [
    "nitrogen", "nitrate", "nitrite", "ammonium", "ammonia", "urea",
    "n2", "n fixation", "nitrogen fixation", "n-fixation", "nitrate reductase",
    "glutamine", "glutamate", "n assimilation", "nitrogen assimilation",
    "denitrification", "nitrification", "anammox", "nodulation", "nodule",
    "n-starvation", "n starvation", "low-n", "high-n", "no3", "no3-",
    "nh4", "nh4+", "nar", "nir", "nif", "amoa", "nrt", "amt",
    "nrt1", "nrt2", "nrt1.1", "nrt2.1", "nrt1.5", "nrt2.5",
]

RISKY_INFERENCE_WORDS = [
    "causes", "caused", "due to", "therefore", "indicates that",
    "demonstrates that", "proves", "suggests that", "mechanism",
    "regulates", "responsible for", "leads to", "drives", "because",
]

# Umbral mínimo de chars de texto seleccionable para saltarse Chandra
CHANDRA_SKIP_THRESHOLD = 150

# Resolución mínima en píxeles para la dimensión corta de un crop
MIN_CROP_DIMENSION_PX = 800


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_page_range(page_range: Optional[str]) -> Optional[list[int]]:
    if not page_range:
        return None
    pages: list[int] = []
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Rango inválido: {part}")
            pages.extend(range(start, end + 1))
        else:
            page = int(part)
            if page <= 0:
                raise ValueError("Las páginas deben ser positivas")
            pages.append(page)
    out: list[int] = []
    seen: set[int] = set()
    for page in pages:
        if page not in seen:
            out.append(page)
            seen.add(page)
    return out


def select_pages(
    total_pages: int,
    max_pages: Optional[int],
    page_range: Optional[str],
) -> list[int]:
    selected = parse_page_range(page_range)
    if selected:
        valid = [p for p in selected if 1 <= p <= total_pages]
        invalid = [p for p in selected if p < 1 or p > total_pages]
        if invalid:
            log.warning(f"Páginas fuera de rango omitidas: {invalid} (total={total_pages})")
        if not valid:
            raise ValueError("No hay páginas válidas para procesar")
        return valid
    if max_pages is not None:
        return list(range(1, min(total_pages, max_pages) + 1))
    return list(range(1, total_pages + 1))


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


# v3.1: patrones de metadatos de journal a ignorar en extract_title
_TITLE_SKIP_RE = re.compile(
    r"^(?:"
    r"published\s*[:\|]|"          # "published: 10 December 2020"
    r"received\s*[:\|]|"           # "received: 12 Jan 2020"
    r"accepted\s*[:\|]|"           # "accepted: ..."
    r"doi\s*[:\|]|"                # "doi: 10.3389/..."
    r"copyright\b|"
    r"frontiers\s+in\b|"           # "Frontiers in Plant Science"
    r"mini\s+review\b|"            # "Mini Review"
    r"review\s+article\b|"         # "Review Article"
    r"original\s+research\b|"      # "Original Research"
    r"\d{1,2}\s+\w+\s+\d{4}\s*$|" # fechas sueltas "10 December 2020"
    r"vol(?:ume)?\.?\s+\d|"
    r"pages?\s+\d|"
    r"issn\s*[:\|]|"
    r"e-?issn\s*[:\|]|"
    r"correspondence\s*[:\|]|"
    r"edited\s+by\b|"
    r"keywords?\s*[:\|]"
    r")",
    re.IGNORECASE,
)


def extract_title(text: str) -> str:
    """
    v3.1: filtra imágenes markdown, URLs, tablas, headings vacíos,
    links solos, y líneas de metadatos de journal (published, doi, etc.).
    """
    for line in text.splitlines():
        line = clean_text(line)
        if not line:
            continue
        if line.startswith("!"):                        # imágenes markdown
            continue
        if line.startswith("|"):                        # tablas markdown
            continue
        if line.startswith("http"):                     # URLs sueltas
            continue
        if re.match(r"^\s*#+\s*$", line):              # headings vacíos
            continue
        if re.match(r"^\s*\[.*\]\(.*\)\s*$", line):   # links solos
            continue
        if _TITLE_SKIP_RE.match(line):                 # metadatos de journal
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()[:180]
        if len(line) > 25:
            return line[:180]
    return "Unknown title"


def rect_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def union_bbox(boxes: list[list[float]]) -> list[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def expand_bbox(
    b: list[float],
    margin: float,
    page_w: float,
    page_h: float,
) -> list[float]:
    return [
        max(0.0, b[0] - margin),
        max(0.0, b[1] - margin),
        min(page_w, b[2] + margin),
        min(page_h, b[3] + margin),
    ]


def x_overlap_ratio(a: list[float], b: list[float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return overlap / denom


def bbox_iou(a: list[float], b: list[float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = rect_area(a) + rect_area(b) - inter
    return inter / union if union > 0 else 0.0


def extract_first_json_object(text: str) -> tuple[Optional[dict[str, Any]], str]:
    if not text:
        return None, text
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?", "", clean).strip()
    clean = re.sub(r"```$", "", clean).strip()
    start = clean.find("{")
    if start < 0:
        return None, clean
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(clean[start:])
        if isinstance(obj, dict):
            return obj, clean
    except json.JSONDecodeError:
        pass
    candidate = clean[start:]
    diff = candidate.count("{") - candidate.count("}")
    if diff > 0:
        candidate += "}" * diff
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj, clean
    except json.JSONDecodeError:
        return None, clean
    return None, clean


def image_file_to_data_uri(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ─────────────────────────────────────────────────────────────────────────────
# Chandra OCR 2
# ─────────────────────────────────────────────────────────────────────────────

class ChandraOCR:
    """
    Wrapper para Chandra OCR 2.

    v3: añade warmup antes del loop principal para evitar el spike JIT
    observado en la primera página (12 min en arabidopsis).
    """

    def __init__(
        self,
        model_path: Path,
        backend: str = "hf_lowlevel",
        enabled: bool = True,
    ):
        self.model_path = str(model_path)
        self.backend = backend
        self.enabled = enabled
        self.model = None
        self.processor = None
        self.manager = None

        if self.enabled:
            self._load()

    def _load(self) -> None:
        if self.backend == "hf_lowlevel":
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            # v3: device explícito en lugar de "auto" para evitar split CPU/GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Cargando Chandra OCR 2 en device={device} (backend=hf_lowlevel)...")

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"
            self.model.processor = self.processor

            # v3: warmup para evitar spike JIT en la primera inferencia real
            self._warmup()
            log.info("Chandra OCR 2 listo.")
            return

        if self.backend in {"manager_hf", "manager_vllm"}:
            from chandra.model import InferenceManager
            method = "hf" if self.backend == "manager_hf" else "vllm"
            log.info(f"Cargando Chandra vía InferenceManager(method={method})...")
            self.manager = InferenceManager(method=method)
            self._warmup()
            log.info("Chandra manager listo.")
            return

        raise ValueError(f"Backend Chandra no soportado: {self.backend}")

    def _warmup(self) -> None:
        """Dispara compilación JIT con imagen dummy antes del loop principal."""
        try:
            import numpy as np
            from PIL import Image
            from chandra.model.schema import BatchInputItem

            dummy = Image.fromarray(
                np.ones((256, 256, 3), dtype=np.uint8) * 255
            )
            batch = [BatchInputItem(image=dummy, prompt_type="ocr_layout")]

            if self.backend == "hf_lowlevel":
                from chandra.model.hf import generate_hf
                generate_hf(batch, self.model)
            else:
                self.manager.generate(batch)

            log.info("Chandra warmup completado.")
        except Exception as exc:
            log.warning(f"Chandra warmup falló (no crítico): {exc}")

    def run_page(self, page_image, page_num: int) -> OCRPageResult:
        if not self.enabled:
            return OCRPageResult(page=page_num)

        from chandra.model.schema import BatchInputItem

        batch = [BatchInputItem(image=page_image, prompt_type="ocr_layout")]

        try:
            if self.backend == "hf_lowlevel":
                from chandra.model.hf import generate_hf
                result = generate_hf(batch, self.model)[0]
            else:
                result = self.manager.generate(batch)[0]
        except Exception as exc:
            log.warning(f"Chandra falló en página {page_num}: {exc}")
            return OCRPageResult(page=page_num, raw=str(exc))

        markdown = getattr(result, "markdown", None)
        layout_json = None
        raw = getattr(result, "raw", None)

        if not markdown and raw:
            try:
                from chandra.output import parse_markdown
                markdown = parse_markdown(raw) or ""
            except Exception:
                markdown = raw if isinstance(raw, str) else ""

        for attr in ("json", "json_output", "layout_json", "structured", "blocks"):
            value = getattr(result, attr, None)
            if value:
                if isinstance(value, dict):
                    layout_json = value
                else:
                    try:
                        layout_json = json.loads(value)
                    except Exception:
                        layout_json = {"raw_layout": str(value)}
                break

        return OCRPageResult(
            page=page_num,
            markdown=markdown or "",
            layout_json=layout_json,
            raw=raw if isinstance(raw, str) else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de figuras — v3.3
# Integra lógica de scientific-figure-extractor (cliford2001):
#   - get_drawings() detecta figuras vectoriales (LaTeX) además de imágenes
#   - clasificación de columna izq/der/full respeta layout multi-columna
#   - boundary previo en la misma columna evita mezcla entre figuras
#   - fallback a página siguiente cuando caption precede a la figura
#   - fallback render: detecta contenido no-blanco sin objetos explícitos
#   - expand con etiquetas de ejes/paneles (bloques de texto estrechos)
# ─────────────────────────────────────────────────────────────────────────────

# Constantes del extractor
_CROSS_COLUMN_TOL  = 50.0   # pt — tolerancia para clasificar columna
_FULL_WIDTH_RATIO  = 0.45   # si visuals > 45% ancho de página → full-width
_TEXT_LABEL_MARGIN = 60.0   # pt — margen para capturar etiquetas de ejes
_WHITE_THRESHOLD   = 0.97   # fracción de pixels blancos para considerar región vacía
_MIN_DRAWING_DIM   = 2.0    # pt — ignorar trazos/líneas muy finos


def _get_column(bbox: list[float], page_width: float) -> str:
    """Clasifica el bbox como columna izquierda, derecha o full-width."""
    x0, _, x1, _ = bbox
    mid = page_width / 2.0
    if x1 < mid + _CROSS_COLUMN_TOL and x0 < mid:
        return "left"
    if x0 > mid - _CROSS_COLUMN_TOL and x1 > mid:
        return "right"
    return "full"


def _in_column(bbox: list[float], column: str, page_width: float) -> bool:
    """Retorna True si bbox pertenece a la columna indicada."""
    x0, _, x1, _ = bbox
    mid = page_width / 2.0
    if column == "left":
        return x1 <= mid + _CROSS_COLUMN_TOL
    if column == "right":
        return x0 >= mid - _CROSS_COLUMN_TOL
    return True


def _combine_bboxes_padded(
    boxes: list[list[float]],
    page_width: float,
    page_height: float,
    padding: float = 8.0,
) -> list[float]:
    return [
        max(0.0,         min(b[0] for b in boxes) - padding),
        max(0.0,         min(b[1] for b in boxes) - padding),
        min(page_width,  max(b[2] for b in boxes) + padding),
        min(page_height, max(b[3] for b in boxes) + padding),
    ]


def _find_figure_region_v3(
    page,
    caption_bbox: list[float],
    prev_boundary_y: float,
    page_width: float,
    page_height: float,
) -> Optional[list[float]]:
    """
    Busca la región visual de una figura arriba del caption.
    Detecta imágenes embebidas Y figuras vectoriales (drawings).
    Respeta columna del caption y límite del caption anterior.
    """
    cy0 = caption_bbox[1]
    column = _get_column(caption_bbox, page_width)

    # Recopilar todos los elementos visuales: imágenes + drawings vectoriales
    visuals: list[list[float]] = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") == 1:  # imagen embebida
            visuals.append(list(b["bbox"]))

    for d in page.get_drawings():
        r = d.get("rect")
        if r and r.width > _MIN_DRAWING_DIM and r.height > _MIN_DRAWING_DIM:
            visuals.append([r.x0, r.y0, r.x1, r.y1])

    # Candidatos: por encima del caption y por debajo del boundary previo
    candidates_all = [
        bb for bb in visuals
        if bb[1] <= cy0 + 5          # empieza antes del caption
        and bb[3] >= prev_boundary_y - 5  # no está completamente sobre el boundary
    ]

    if not candidates_all:
        return None

    # Detectar si el contenido es full-width (multi-panel que abarca columnas)
    combined_x0 = min(b[0] for b in candidates_all)
    combined_x1 = max(b[2] for b in candidates_all)
    if (combined_x1 - combined_x0) > page_width * _FULL_WIDTH_RATIO:
        effective_column = "full"
    else:
        effective_column = column

    candidates = [bb for bb in candidates_all if _in_column(bb, effective_column, page_width)]
    if not candidates:
        candidates = candidates_all

    vx0 = min(b[0] for b in candidates)
    vy0 = min(b[1] for b in candidates)
    vx1 = max(b[2] for b in candidates)
    vy1 = max(b[3] for b in candidates)

    # Expandir con bloques de texto estrechos adyacentes:
    # etiquetas de ejes, panel labels (a, b, c...), tick labels
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        bb = list(b["bbox"])
        if bb == list(caption_bbox):
            continue
        bx0, by0, bx1, by1 = bb
        if by0 > cy0 + 5:
            continue
        if by1 < prev_boundary_y - 5:
            continue
        # Solo bloques estrechos (etiquetas, no párrafos de cuerpo)
        if (bx1 - bx0) > page_width * 0.35:
            continue
        # Adyacente al área visual
        if (bx1 > vx0 - _TEXT_LABEL_MARGIN and bx0 < vx1 + _TEXT_LABEL_MARGIN
                and by1 > vy0 - _TEXT_LABEL_MARGIN and by0 < vy1 + _TEXT_LABEL_MARGIN):
            candidates.append(bb)

    return _combine_bboxes_padded(candidates, page_width, page_height)


def _fallback_render_region(
    page,
    caption_bbox: list[float],
    prev_boundary_y: float,
    page_width: float,
    page_height: float,
) -> Optional[list[float]]:
    """
    Renderiza el área sobre el caption a baja resolución y la retorna
    si tiene contenido no-blanco (Form XObjects, contenido no listado por PyMuPDF).
    """
    import fitz

    cy0 = caption_bbox[1]
    col = _get_column(caption_bbox, page_width)

    if col == "left":
        x0, x1 = 0.0, page_width / 2.0
    elif col == "right":
        x0, x1 = page_width / 2.0, page_width
    else:
        x0, x1 = 0.0, page_width

    clip = fitz.Rect(x0, max(0.0, prev_boundary_y), x1, cy0)
    if clip.height < 10:
        return None

    # Renderizar a 72 DPI (baja res, solo para detectar contenido)
    mat = fitz.Matrix(1.0, 1.0)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    samples = pix.samples
    n = pix.width * pix.height
    if n == 0:
        return None

    white = sum(
        1 for i in range(0, len(samples), 3)
        if samples[i] > 240 and samples[i + 1] > 240 and samples[i + 2] > 240
    )
    if white / n > _WHITE_THRESHOLD:
        return None  # región esencialmente en blanco → no hay figura

    return [
        max(0.0, x0 - 8.0),
        max(0.0, clip.y0 - 8.0),
        min(page_width, x1 + 8.0),
        min(page_height, cy0 + 8.0),
    ]


class PDFFigureExtractor:
    """
    Extrae captions y crops de figuras con PyMuPDF.

    v3.3 — integra lógica de scientific-figure-extractor:

    Detección visual:
      1. Imágenes embebidas (type==1) — igual que antes
      2. Figuras vectoriales via get_drawings() — NUEVO: detecta gráficos
         LaTeX que antes eran invisibles (causaban heurístico ciego)
      3. Expansión con etiquetas de ejes/paneles (bloques de texto estrechos)

    Layout multi-columna:
      - Clasifica caption como izq/der/full-width según posición horizontal
      - Limita búsqueda de visuals a la misma columna del caption
      - Si los visuals abarcan >45% del ancho → trata como full-width

    Boundary previo:
      - Busca el caption anterior en la MISMA columna de la página
      - Usa su borde inferior como límite para no mezclar figuras consecutivas

    Fallbacks en orden:
      1. get_drawings() + imágenes: región exacta con columna correcta
      2. Página siguiente: si el caption precede a la figura (poco común)
      3. Render a baja resolución: detecta Form XObjects y contenido
         que PyMuPDF no enumera pero sí renderiza
      4. Descarte (sin --allow-full-page-fallback)
    """

    def __init__(
        self,
        dpi: int = 220,
        crop_margin_pt: float = 8.0,
        include_caption_in_crop: bool = False,
        allow_full_page_fallback: bool = False,
    ):
        self.dpi = dpi
        self.crop_margin_pt = crop_margin_pt
        self.include_caption_in_crop = include_caption_in_crop
        self.allow_full_page_fallback = allow_full_page_fallback

    @staticmethod
    def _pil_from_page(page, dpi: int):
        import fitz
        from PIL import Image
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _save_crop(self, page, bbox: list[float], out_path: Path) -> None:
        """DPI adaptativo: garantiza mínimo MIN_CROP_DIMENSION_PX px en el lado corto."""
        import fitz
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        needed_w = (MIN_CROP_DIMENSION_PX / bbox_w * 72) if bbox_w > 0 else self.dpi
        needed_h = (MIN_CROP_DIMENSION_PX / bbox_h * 72) if bbox_h > 0 else self.dpi
        effective_dpi = max(self.dpi, min(int(max(needed_w, needed_h)), 400))
        rect = fitz.Rect(*bbox)
        mat  = fitz.Matrix(effective_dpi / 72.0, effective_dpi / 72.0)
        pix  = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        pix.save(str(out_path))

    def _detect_caption_blocks(
        self,
        page_num: int,
        page,
    ) -> list[CaptionBlock]:
        """
        Detecta captions usando FIGURE_RE sobre bloques de texto,
        fusionando bloques adyacentes (captions multi-línea en LaTeX).
        Filtra referencias inline (texto largo antes del match, o entre paréntesis).
        """
        raw = page.get_text("dict")
        text_blocks: list[dict] = []
        for b in raw.get("blocks", []):
            if b.get("type") != 0:
                continue
            chunks = []
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    chunks.append(span.get("text", ""))
            text = clean_text(" ".join(chunks))
            if text:
                text_blocks.append({"bbox": list(b["bbox"]), "text": text})

        # Fusionar bloques adyacentes verticalmente
        if text_blocks:
            sorted_blocks = sorted(text_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
            merged = [sorted_blocks[0].copy()]
            for block in sorted_blocks[1:]:
                prev = merged[-1]
                gap = block["bbox"][1] - prev["bbox"][3]
                xov = x_overlap_ratio(prev["bbox"], block["bbox"])
                if 0 <= gap <= 14.0 and xov > 0.3:
                    merged[-1] = {
                        "bbox": union_bbox([prev["bbox"], block["bbox"]]),
                        "text": prev["text"] + " " + block["text"],
                    }
                else:
                    merged.append(block.copy())
            text_blocks = merged

        captions: list[CaptionBlock] = []
        for block in text_blocks:
            text = block["text"]
            m = FIGURE_RE.search(text)
            if not m:
                continue

            match_start = m.start()
            text_before = text[:match_start].strip()
            # Referencia inline: mucho texto antes del match
            if len(text_before) > 60:
                continue
            # Referencia entre paréntesis
            paren_m = re.search(
                r"\([^)]*(?:Fig\.?|Figure)\s*\w+[^)]*\)", text, re.IGNORECASE
            )
            if paren_m and paren_m.start() <= match_start <= paren_m.end():
                continue

            after_label = clean_text(m.group(2))
            looks_like_caption = len(text) >= 35 or len(after_label) >= 20
            is_inline_ref = (
                bool(re.search(r"(see|shown in|as in|in Fig)", text, re.I))
                and len(text) < 120
            )
            if is_inline_ref and not looks_like_caption:
                continue

            fig_number = m.group(1).strip()
            captions.append(CaptionBlock(
                page=page_num,
                text=text,
                figure_label=f"Figure {fig_number}",
                figure_number=fig_number,
                bbox_pdf=block["bbox"],
            ))

        captions.sort(key=lambda c: (c.bbox_pdf[1], c.bbox_pdf[0]))
        return captions

    def _candidate_from_caption(
        self,
        page,
        next_page,           # página siguiente (puede ser None)
        page_num: int,
        caption: CaptionBlock,
        all_captions_this_page: list[CaptionBlock],
        figures_dir: Path,
        paper_stem: str,
        idx: int,
    ) -> Optional[FigureCandidate]:
        import fitz

        page_w = float(page.rect.width)
        page_h = float(page.rect.height)

        col = _get_column(caption.bbox_pdf, page_w)

        # Boundary previo: caption anterior en la MISMA columna
        prev_boundary_y = 0.0
        for other in all_captions_this_page:
            if other is caption:
                continue
            if _get_column(other.bbox_pdf, page_w) != col:
                continue
            if other.bbox_pdf[3] < caption.bbox_pdf[1]:
                prev_boundary_y = max(prev_boundary_y, other.bbox_pdf[3])

        notes: list[str] = []
        detection_method = ""
        crop_quality = ""
        render_page = page
        region: Optional[list[float]] = None

        # ── Intento 1: drawings + imágenes embebidas ─────────────────────────
        region = _find_figure_region_v3(
            page=page,
            caption_bbox=caption.bbox_pdf,
            prev_boundary_y=prev_boundary_y,
            page_width=page_w,
            page_height=page_h,
        )
        if region is not None:
            detection_method = "visual_region_drawings_images"
            crop_quality     = "image_block_crop"

        # ── Intento 2: figura en página siguiente ────────────────────────────
        if region is None and next_page is not None:
            np_w = float(next_page.rect.width)
            np_h = float(next_page.rect.height)
            region = _find_figure_region_v3(
                page=next_page,
                caption_bbox=[0.0, np_h, np_w, np_h],  # caption ficticio al final
                prev_boundary_y=0.0,
                page_width=np_w,
                page_height=np_h,
            )
            if region is not None:
                detection_method = "next_page_visual_region"
                crop_quality     = "image_block_crop"
                render_page      = next_page
                notes.append("Figure found on next page (caption precedes figure).")
                log.info(f"    [{caption.figure_label}] p{page_num} → figura en página siguiente")

        # ── Intento 3: fallback render (Form XObjects) ───────────────────────
        if region is None:
            region = _fallback_render_region(
                page=page,
                caption_bbox=caption.bbox_pdf,
                prev_boundary_y=prev_boundary_y,
                page_width=page_w,
                page_height=page_h,
            )
            if region is not None:
                detection_method = "fallback_render_nonwhite"
                crop_quality     = "heuristic_caption_region"
                notes.append("Region detected via low-res render (Form XObject or unlisted content).")

        # ── Fallback final: página completa ──────────────────────────────────
        if region is None:
            if not self.allow_full_page_fallback:
                log.warning(
                    f"    Sin crop confiable para {caption.figure_label} p{page_num}; se omite"
                )
                return None
            region = [0.0, 0.0, page_w, page_h]
            detection_method = "full_page_fallback"
            crop_quality     = "full_page_fallback_high_risk"
            notes.append("Full page used as fallback; high risk of unrelated content.")

        # Incluir caption en el crop si se pide
        if self.include_caption_in_crop:
            region = union_bbox([region, caption.bbox_pdf])

        # Ajustar a límites de página
        rp_w = float(render_page.rect.width)
        rp_h = float(render_page.rect.height)
        region = [
            max(0.0, region[0]),
            max(0.0, region[1]),
            min(rp_w, region[2]),
            min(rp_h, region[3]),
        ]

        # Guardar crop
        fig_num  = caption.figure_number or f"u{idx + 1}"
        safe_num = re.sub(r"[^0-9A-Za-z_\-]", "_", str(fig_num))
        out_path = figures_dir / f"{paper_stem}_p{page_num:03d}_fig{safe_num}_{idx:03d}.png"
        self._save_crop(render_page, region, out_path)

        figure_id = f"fig_{safe_num}_p{page_num:03d}_{idx:03d}"
        return FigureCandidate(
            figure_id=figure_id,
            page=page_num,
            caption=caption.text,
            figure_label=caption.figure_label,
            figure_number=caption.figure_number,
            bbox_pdf=[round(v, 2) for v in region],
            caption_bbox_pdf=[round(v, 2) for v in caption.bbox_pdf],
            crop_path=str(out_path),
            detection_method=detection_method,
            crop_quality=crop_quality,
            crop_includes_caption=self.include_caption_in_crop,
            page_width_pdf=round(page_w, 2),
            page_height_pdf=round(page_h, 2),
            notes=notes,
        )

    def extract_from_pdf(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        max_figures: Optional[int] = None,
    ) -> list[FigureCandidate]:
        import fitz

        stem = pdf_path.stem
        figures_dir = PATHS["extracted_figures"] / stem
        figures_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        total_pages    = len(doc)
        all_candidates: list[FigureCandidate] = []

        try:
            for page_num in pages_to_process:
                if max_figures is not None and len(all_candidates) >= max_figures:
                    break

                page      = doc[page_num - 1]
                next_page = doc[page_num] if page_num < total_pages else None

                captions = self._detect_caption_blocks(page_num, page)
                if captions:
                    log.info(f"  Figuras/captions p{page_num}: {len(captions)} detectados")

                for cap in captions:
                    if max_figures is not None and len(all_candidates) >= max_figures:
                        break
                    cand = self._candidate_from_caption(
                        page=page,
                        next_page=next_page,
                        page_num=page_num,
                        caption=cap,
                        all_captions_this_page=captions,
                        figures_dir=figures_dir,
                        paper_stem=stem,
                        idx=len(all_candidates),
                    )
                    if cand:
                        all_candidates.append(cand)
        finally:
            doc.close()

        return self._deduplicate(all_candidates)

    @staticmethod
    def _deduplicate(candidates: list[FigureCandidate]) -> list[FigureCandidate]:
        if not candidates:
            return []
        quality_rank = {
            "image_block_crop":          0,
            "heuristic_caption_region":  1,
            "full_page_fallback_high_risk": 2,
        }
        kept: list[FigureCandidate] = []
        for cand in candidates:
            dup_idx = None
            for i, other in enumerate(kept):
                if cand.page != other.page:
                    continue
                same_num = cand.figure_number and cand.figure_number == other.figure_number
                high_iou = bbox_iou(cand.bbox_pdf, other.bbox_pdf) > 0.75
                if same_num or high_iou:
                    dup_idx = i
                    break
            if dup_idx is None:
                kept.append(cand)
            else:
                old = kept[dup_idx]
                if quality_rank.get(cand.crop_quality, 9) < quality_rank.get(old.crop_quality, 9):
                    kept[dup_idx] = cand
        return kept


# ─────────────────────────────────────────────────────────────────────────────
# InternVL3 — sistema prompt v3
# ─────────────────────────────────────────────────────────────────────────────

# v3.2: sistema prompt ampliado con fixes derivados de auditoría Gemini:
# - conteo de paneles por estructura visual, no solo por etiquetas
# - transcripción exacta carácter a carácter (genome≠gene, nitrate≠nitrogen)
# - printed_terms exhaustivo: IDs numéricos, operadores matemáticos, unidades
# - evidence_source = caption_inferred para observaciones que mezclan imagen+caption
FIGURE_SYSTEM_PROMPT = """You are a scientific figure analyst specialized in biology and nitrogen-related research.
You must describe only what is directly visible in the image crop and in the provided caption/context.
Do not use external scientific knowledge.
Do not infer mechanisms, causality, treatments, species, genes, pathways, or statistical significance unless explicitly visible.
Use exact numerical values only if printed and clearly readable.
If values, labels, units, or meanings are unclear, mark them as not determinable.

IMPORTANT — pathway and signaling diagrams:
These figures often contain many nodes, arrows, and labels. This complexity is EXPECTED and NORMAL.
For pathway/network diagrams: list all visible gene/protein/molecule names in printed_terms.
Describe arrow types as factual observations (pointed arrow = activation direction visible,
blunt/flat arrow = inhibition direction visible) using ONLY what is printed in the image.
Do NOT infer biological meaning from arrow topology.
A visually complex diagram is NOT a reason for high uncertainty if the visual elements are clearly readable.
Set hallucination_risk and uncertainty_level based on readability, NOT on diagram complexity.

IMPORTANT — multi-panel figures and panel counting:
Count ALL visually distinct plot areas, not just those with an explicit letter or number label.
If you see 4 heatmaps in a 2x2 grid, set num_panels=4 even if only 2 have visible labels.
For unlabeled panels adjacent to labeled ones, infer their label from sequence:
  if you see labels "f" and "h", the unlabeled panels are likely "e" and "g".
List ALL panel labels (observed and inferred) in visual_structure.panel_labels.
Set target_figure_only: true and describe ALL panels collectively.
Do NOT refuse to interpret because there are multiple panels.
If the crop clearly contains content unrelated to this figure caption, set target_figure_only: false.

IMPORTANT — printed_terms must be exhaustive:
Include ALL text visible in the image, even if it seems trivial:
  - Mathematical operators and symbols: =, +, -, *, >, <, ~, plus, minus, times
  - Sample and accession identifiers on axes: numeric IDs like 6024, 9543, 22001m
  - Scale bar values, p-values, numeric axis tick labels visible in the figure
  - Units even if small: Mb, kb, %, um, RPM, RPKM, TPM, RPKM
  - Color scale labels such as numbers on a color bar legend
Do NOT skip any symbol or number because it seems minor or decorative.
CRITICAL: You MUST include the 'printed_terms' array in your JSON output under 'visible_text', even if you think it is empty. Do not delete this key.

IMPORTANT — text transcription accuracy:
Transcribe visible text character by character. Do NOT substitute visually similar words.
Forbidden substitutions (observed errors):
  genome is NOT the same as gene
  nitrate is NOT the same as nitrogen
  expression is NOT the same as enrichment
  density is NOT the same as diversity
  synteny is NOT the same as sequence
If a word is partially visible or ambiguous, write exactly what you can see and add it to not_determinable.

IMPORTANT — evidence_source rules:
Use exactly one of these four values per observation:
  visual           = the fact is verifiable purely from the image without reading any text
  caption          = the fact is stated explicitly in the caption text provided
  caption_inferred = the fact requires BOTH seeing the image AND reading the caption together
  context          = the fact comes from the surrounding paper text references only
Rule: when in doubt between visual and caption_inferred, always use caption_inferred.
Do NOT mark as visual any observation that requires knowing what the figure represents conceptually.
Example of wrong usage: marking "the graph shows nestedness of TE annotations" as visual
  when nestedness is a concept from the caption, not something directly visible.

Return valid JSON only. No markdown. No preamble."""


def build_figure_user_prompt(
    title: str,
    caption: str,
    context_sentences: list[str],
    candidate: FigureCandidate,
) -> str:
    context = (
        "\n".join(f"- {s}" for s in context_sentences[:5])
        or "- No nearby textual reference found."
    )

    return f"""
Paper title:
{title or "Unknown title"}

Figure candidate metadata:
- figure_id: {candidate.figure_id}
- page: {candidate.page}
- detection_method: {candidate.detection_method}
- crop_quality: {candidate.crop_quality}
- crop_includes_caption: {candidate.crop_includes_caption}

Caption:
{caption or "No caption detected"}

Nearby text references:
{context}

Task:
Analyze the image crop as a scientific figure. Be conservative and factual.

Return exactly this JSON schema:
{{
  "target_figure_only": true,
  "figure_type": "line_plot|bar_chart|scatter|heatmap|diagram|table|microscopy|gel|map|mixed_multi_panel|other|unknown",
  "visible_text": {{
    "title": null,
    "x_axis_label": [],
    "y_axis_label": [],
    "units": [],
    "legend_items": [],
    "panel_labels": [],
    "printed_terms": []
  }},
  "visual_structure": {{
    "num_panels": null,
    "plot_elements": [],
    "groups_or_conditions": [],
    "statistical_markers_visible": []
  }},
  "factual_observations": [
    {{
      "observation": "Directly visible fact only",
      "evidence_source": "visual|caption|caption_inferred|context",
      "certainty": "low|medium|high"
    }}
  ],
  "nitrogen_relevance": {{
    "is_relevant": false,
    "visible_terms": [],
    "evidence_source": [],
    "reason": null
  }},
  "not_determinable": [],
  "hallucination_risk": "low|medium|high",
  "uncertainty_level": "low|medium|high",
  "uncertainty_reason": null
}}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Clientes InternVL3
# ─────────────────────────────────────────────────────────────────────────────

class InternVLServerClient:
    """Cliente vLLM server OpenAI-compatible."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: Optional[str] = None,
        timeout_s: int = 180,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.model_name = model_name or self._infer_model_name()
        log.info(f"InternVL server: model={self.model_name}, url={base_url}")

    def _infer_model_name(self) -> str:
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
        except Exception as exc:
            log.warning(f"No pude consultar /models del servidor vLLM: {exc}")
        return "internvl3-14b"

    def interpret(
        self,
        image_path: str,
        title: str,
        caption: str,
        context_sentences: list[str],
        candidate: FigureCandidate,
        max_tokens: int = 1800,
    ) -> FigureInterpretation:
        prompt = build_figure_user_prompt(title, caption, context_sentences, candidate)
        data_uri = image_file_to_data_uri(image_path)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                top_p=0.9,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": FIGURE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            log.warning(f"InternVL server falló en {candidate.figure_id}: {exc}")
            return FigureInterpretation(
                hallucination_risk="high",
                uncertainty_level="high",
                uncertainty_reason=f"Server error: {exc}",
            )

        return parse_interpretation(raw)


class InternVLOfflineClient:
    """
    Cliente offline con vLLM en el mismo proceso.

    Para InternVL3-78B en 2×H100 80GB se necesitan ajustes específicos:
    - enforce_eager=True: desactiva CUDA graph profiling que causa OOM durante
      el warmup del vision encoder (el modelo ocupa ~95% de cada GPU).
    - gpu_memory_utilization conservador (0.60): con 78B los pesos ya ocupan
      ~73GB/79GB por GPU, el margen para KV cache es mínimo.
    - limit_mm_per_prompt={"image": 1}: ya establecido, limita a 1 imagen.

    Nota: mm_processor_kwargs con max_dynamic_patch no es compatible con
    vLLM 0.21.0 (InternVLVideoProcessor no lo acepta). Se usa enforce_eager
    como única mitigación de OOM para modelos grandes.
    """

    def __init__(
        self,
        model_path: Path,
        gpu_util: float = 0.90,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.model_path = str(model_path)

        # Detectar modelo grande (>60GB) para aplicar enforce_eager
        model_size_gb = self._estimate_model_size_gb(model_path)
        is_large_model = model_size_gb > 60

        if is_large_model:
            log.info(
                f"Modelo grande detectado (~{model_size_gb:.0f}GB). "
                "Aplicando enforce_eager=True para 2×H100."
            )
            enforce_eager = True
        else:
            enforce_eager = False

        log.info(
            f"Cargando InternVL offline desde {self.model_path} | "
            f"tp={tensor_parallel_size} | enforce_eager={enforce_eager} | "
            f"gpu_util={gpu_util} | max_model_len={max_model_len}"
        )


        self.llm = LLM(
            model=self.model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_util,
            max_model_len=max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 1, "video": 0},
            enforce_eager=enforce_eager,
            disable_custom_all_reduce=True,
        )



        try:
            self.tokenizer = self.llm.get_tokenizer()
        except Exception:
            self.tokenizer = None

        log.info("InternVL offline cargado.")

    @staticmethod
    def _estimate_model_size_gb(model_path: Path) -> float:
        """Estima el tamaño del modelo en GB sumando los safetensors."""
        try:
            total = sum(
                f.stat().st_size
                for f in Path(model_path).glob("*.safetensors")
            )
            return total / (1024 ** 3)
        except Exception:
            return 0.0

    def _internvl_stop_token_ids(self) -> Optional[list[int]]:
        if self.tokenizer is None:
            return None
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        ids: list[int] = []
        for tok in stop_tokens:
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tok_id = None
            if tok_id is not None and isinstance(tok_id, int) and tok_id >= 0:
                ids.append(tok_id)
        return ids or None

    def _build_offline_prompt(self, prompt_body: str) -> str:
        question = (
            f"{FIGURE_SYSTEM_PROMPT}\n\n"
            f"{prompt_body}\n\n"
            "Important: return only a valid JSON object. Do not return markdown."
        )
        messages = [{"role": "user", "content": f"<image>\n{question}"}]

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                log.warning(f"apply_chat_template falló; usando fallback manual: {exc}")

        return (
            "<|im_start|>user\n"
            f"<image>\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def interpret(
        self,
        image_path: str,
        title: str,
        caption: str,
        context_sentences: list[str],
        candidate: FigureCandidate,
        max_tokens: int = 1800,
    ) -> FigureInterpretation:
        from PIL import Image

        prompt_body = build_figure_user_prompt(
            title, caption, context_sentences, candidate
        )
        prompt = self._build_offline_prompt(prompt_body)

        try:
            img = Image.open(image_path).convert("RGB")
            stop_token_ids = self._internvl_stop_token_ids()

            sampling = self.SamplingParams(
                temperature=0.0,
                top_p=0.9,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
            )

            outputs = self.llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": img}}],
                sampling,
            )
            raw = outputs[0].outputs[0].text or ""
            log.info(f"    InternVL raw chars: {len(raw.strip())}")

            if not raw.strip():
                log.warning("    InternVL devolvió vacío; reintento sin stop_token_ids")
                sampling_retry = self.SamplingParams(
                    temperature=0.0, top_p=0.9, max_tokens=max_tokens
                )
                outputs = self.llm.generate(
                    [{"prompt": prompt, "multi_modal_data": {"image": img}}],
                    sampling_retry,
                )
                raw = outputs[0].outputs[0].text or ""
                log.info(f"    InternVL retry raw chars: {len(raw.strip())}")

        except Exception as exc:
            log.warning(f"InternVL offline falló en {candidate.figure_id}: {exc}")
            return FigureInterpretation(
                hallucination_risk="high",
                uncertainty_level="high",
                uncertainty_reason=f"Offline error: {exc}",
            )

        return parse_interpretation(raw)


def parse_interpretation(raw: str) -> FigureInterpretation:
    data, clean = extract_first_json_object(raw)
    if data is None:
        return FigureInterpretation(
            hallucination_risk="high",
            uncertainty_level="high",
            uncertainty_reason="JSON parse failed",
            raw_response=clean,
        )

    nr = data.get("nitrogen_relevance") or {}
    if not isinstance(nr, dict):
        nr = {"is_relevant": bool(nr), "reason": str(nr)}

    return FigureInterpretation(
        target_figure_only=data.get("target_figure_only"),
        figure_type=data.get("figure_type", "unknown") or "unknown",
        visible_text=(
            data.get("visible_text")
            if isinstance(data.get("visible_text"), dict)
            else {}
        ),
        visual_structure=(
            data.get("visual_structure")
            if isinstance(data.get("visual_structure"), dict)
            else {}
        ),
        factual_observations=(
            data.get("factual_observations")
            if isinstance(data.get("factual_observations"), list)
            else []
        ),
        nitrogen_relevance=NitrogenRelevance(
            is_relevant=bool(nr.get("is_relevant", False)),
            visible_terms=(
                nr.get("visible_terms")
                if isinstance(nr.get("visible_terms"), list)
                else []
            ),
            evidence_source=(
                nr.get("evidence_source")
                if isinstance(nr.get("evidence_source"), list)
                else []
            ),
            reason=nr.get("reason"),
        ),
        not_determinable=(
            data.get("not_determinable")
            if isinstance(data.get("not_determinable"), list)
            else []
        ),
        hallucination_risk=data.get("hallucination_risk", "high") or "high",
        uncertainty_level=data.get("uncertainty_level", "high") or "high",
        uncertainty_reason=data.get("uncertainty_reason"),
        raw_response=clean,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Contexto textual
# ─────────────────────────────────────────────────────────────────────────────

def find_context_sentences(
    full_text: str,
    fig_number: Optional[str],
    max_sentences: int = 5,
) -> list[str]:
    if not fig_number:
        return []
    text = re.sub(r"\s+", " ", full_text or " ").strip()
    if not text:
        return []

    patterns = [
        rf"(?:Fig\.?|Figure)\s*{re.escape(fig_number)}\b[^.。;]*[.。;]",
        rf"\((?:Fig\.?|Figure)\s*{re.escape(fig_number)}\b[^)]*\)",
    ]

    hits: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            sent = clean_text(m.group(0))
            if sent and sent not in hits:
                hits.append(sent)
            if len(hits) >= max_sentences:
                return hits

    return hits[:max_sentences]


def collect_visible_terms_blob(
    caption: str,
    context_sentences: list[str],
    interpretation: FigureInterpretation,
) -> str:
    vt = interpretation.visible_text or {}
    pieces = [caption or "", " ".join(context_sentences)]
    for value in vt.values():
        if isinstance(value, list):
            pieces.extend(str(x) for x in value)
        elif value is not None:
            pieces.append(str(value))
    # Incluye también printed_terms del visible_text
    printed = vt.get("printed_terms", [])
    if isinstance(printed, list):
        pieces.extend(str(t) for t in printed)
    return clean_text(" ".join(pieces)).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Validador v3
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicValidator:
    """
    Validador v3 — separa fallos duros de advertencias suaves.

    Fallos duros → rechazo categórico:
      - target_figure_only is False
      - full_page_fallback
      - inferencia causal/mecanística en observaciones
      - nitrógeno declarado sin términos visibles

    Advertencias suaves → acepta con medium_risk:
      - uncertainty/hallucination high (pero NO si es image_block_crop
        con términos de nitrógeno visibles — fix falso negativo diagramas)
      - crop heurístico

    Esta separación permite rescatar diagramas de señalización complejos
    (como el de NO₃⁻ crosstalk) que InternVL calificaba como "high" por
    complejidad visual, no por alucinación real.
    """

    def validate(
        self,
        candidate: FigureCandidate,
        interpretation: Optional[FigureInterpretation],
        context_sentences: list[str],
    ) -> ValidationReport:
        if interpretation is None:
            return ValidationReport(False, "high_risk", ["No interpretation available"])

        hard_failures: list[str] = []
        soft_reasons: list[str] = []

        # ── Fallos duros ────────────────────────────────────────────────────

        if interpretation.target_figure_only is False:
            hard_failures.append(
                "Model says crop does not contain only target figure"
            )

        if candidate.crop_quality == "full_page_fallback_high_risk":
            hard_failures.append("Crop is full-page fallback")

        # Inferencia causal en observaciones
        observations_text = " ".join(
            str(obs.get("observation", ""))
            for obs in interpretation.factual_observations
            if isinstance(obs, dict)
        ).lower()
        for risky in RISKY_INFERENCE_WORDS:
            if risky in observations_text:
                hard_failures.append(
                    f"Potential causal/mechanistic inference: '{risky}'"
                )
                break

        # Nitrógeno declarado sin evidencia visible
        if interpretation.nitrogen_relevance.is_relevant:
            blob = collect_visible_terms_blob(
                candidate.caption, context_sentences, interpretation
            )
            if not any(term in blob for term in NITROGEN_TERMS):
                hard_failures.append(
                    "Nitrogen relevance asserted but no nitrogen term found in "
                    "caption/context/visible_text"
                )

        if hard_failures:
            return ValidationReport(False, "high_risk", hard_failures)

        # ── Advertencias suaves ─────────────────────────────────────────────

        # v3: image_block_crop con términos de nitrógeno visibles no se penaliza
        # por uncertainty/hallucination high — diagramas complejos de señalización
        # son conservadoramente calificados como "high" por el modelo aunque sean
        # perfectamente legibles (falso negativo observado en paper_nitrate fig_1_p003).
        is_reliable_crop = candidate.crop_quality == "image_block_crop"
        nitrogen_blob = collect_visible_terms_blob(
            candidate.caption, context_sentences, interpretation
        )
        has_nitrogen_terms = any(term in nitrogen_blob for term in NITROGEN_TERMS)
        also_nitrogen_in_visible = any(
            term in " ".join(
                str(t) for t in interpretation.nitrogen_relevance.visible_terms
            ).lower()
            for term in NITROGEN_TERMS
        )
        nitrogen_evidence = has_nitrogen_terms or also_nitrogen_in_visible

        if interpretation.uncertainty_level == "high":
            if is_reliable_crop and nitrogen_evidence:
                soft_reasons.append(
                    "uncertainty_level=high but image_block_crop with nitrogen "
                    "evidence — accepted with review flag (complex signaling diagram)"
                )
            else:
                soft_reasons.append("Model uncertainty is high")

        if interpretation.hallucination_risk == "high":
            if is_reliable_crop and nitrogen_evidence:
                soft_reasons.append(
                    "hallucination_risk=high but image_block_crop with nitrogen "
                    "evidence — accepted with review flag (complex signaling diagram)"
                )
            else:
                soft_reasons.append("Model hallucination_risk is high — review before use")

        if candidate.crop_quality == "heuristic_caption_region":
            soft_reasons.append("Crop is heuristic, not image-block crop")

        # v3.2: num_panels > 4 con crop heurístico → flag de revisión
        num_panels = (interpretation.visual_structure or {}).get("num_panels") or 0
        if (
            isinstance(num_panels, int)
            and num_panels > 4
            and candidate.crop_quality == "heuristic_caption_region"
        ):
            soft_reasons.append(
                f"num_panels={num_panels} with heuristic crop — "
                "complex multi-panel figure, manual review recommended"
            )

        # Calidad de observaciones — v3.2 acepta caption_inferred
        VALID_EVIDENCE_SOURCES = {"visual", "caption", "caption_inferred", "context"}
        for obs in interpretation.factual_observations:
            if not isinstance(obs, dict):
                soft_reasons.append("Malformed observation object")
                break
            if obs.get("evidence_source") not in VALID_EVIDENCE_SOURCES:
                soft_reasons.append(
                    f"Observation with invalid evidence_source: "
                    f"'{obs.get('evidence_source')}'"
                )
                break

        if soft_reasons:
            return ValidationReport(True, "medium_risk", soft_reasons)

        return ValidationReport(True, "low_risk", ["Passed heuristic validation"])


# ─────────────────────────────────────────────────────────────────────────────
# Guardado de outputs
# ─────────────────────────────────────────────────────────────────────────────

class OutputWriter:
    def save(self, result: PaperResult) -> None:
        stem = result.paper_stem

        txt_path      = PATHS["final_text"]     / f"{stem}.ocr.txt"
        json_path     = PATHS["final_text"]     / f"{stem}.figures.json"
        enriched_path = PATHS["final_text"]     / f"{stem}.enriched.txt"
        captions_path = PATHS["figure_captions"] / f"{stem}.candidates.json"

        atomic_write_text(txt_path, result.full_text)
        atomic_write_json(json_path, self._to_payload(result))
        atomic_write_json(captions_path, [asdict(f.candidate) for f in result.figures])
        atomic_write_text(enriched_path, self._build_enriched_text(result))

        low  = sum(1 for f in result.figures if f.validation and f.validation.validation_level == "low_risk")
        med  = sum(1 for f in result.figures if f.validation and f.validation.validation_level == "medium_risk")
        high = sum(1 for f in result.figures if f.validation and f.validation.validation_level == "high_risk")
        accepted = sum(1 for f in result.figures if f.validation and f.validation.accepted_for_training)

        print("\n" + "─" * 72)
        print(f"Paper       : {result.paper_title[:68]}")
        print(f"Texto OCR   : {len(result.full_text):,} chars → {txt_path.name}")
        print(f"Figuras     : {len(result.figures)} total | accepted={accepted}")
        print(f"Riesgo      : low={low} medium={med} high={high}")
        print(f"JSON        : {json_path.name}")
        print(f"Enriquecido : {enriched_path.name}")
        print(f"Tiempo      : {result.processing_time_s:.1f}s")
        print("─" * 72 + "\n")

    @staticmethod
    def _to_payload(result: PaperResult) -> dict[str, Any]:
        return {
            "pdf_path":          result.pdf_path,
            "paper_stem":        result.paper_stem,
            "paper_title":       result.paper_title,
            "text_length_chars": len(result.full_text),
            "figure_count":      len(result.figures),
            "processing_time_s": round(result.processing_time_s, 2),
            "timestamp":         result.timestamp,
            "metadata":          result.metadata,
            "figures":           [asdict(fig) for fig in result.figures],
        }

    @staticmethod
    def _build_enriched_text(result: PaperResult) -> str:
        parts = [result.full_text.strip(), "\n\n"]
        usable = [
            f for f in result.figures
            if f.validation
            and f.validation.accepted_for_training
            and f.interpretation
        ]

        if not usable:
            parts.append(
                "\n[FIGURE_INTERPRETATIONS]\n"
                "No accepted figure interpretations.\n"
                "[/FIGURE_INTERPRETATIONS]\n"
            )
            return "".join(parts)

        parts.append("\n[FIGURE_INTERPRETATIONS]\n")
        for fig in usable:
            interp = fig.interpretation
            assert interp is not None
            parts.append("[FIGURE_INTERPRETATION_START]\n")
            parts.append(f"paper_id: {result.paper_stem}\n")
            parts.append(f"figure_id: {fig.figure_id}\n")
            parts.append(f"page: {fig.page}\n")
            parts.append(f"source: multimodal_interpretation\n")
            parts.append(f"model: {interp.model}\n")
            parts.append(f"crop_quality: {fig.candidate.crop_quality}\n")
            parts.append(
                f"validation_level: "
                f"{fig.validation.validation_level if fig.validation else 'unknown'}\n"
            )
            parts.append(f"caption: {clean_text(fig.caption)}\n")
            if fig.context_sentences:
                parts.append("context:\n")
                for s in fig.context_sentences:
                    parts.append(f"- {clean_text(s)}\n")
            parts.append(f"figure_type: {interp.figure_type}\n")
            parts.append(
                f"nitrogen_relevance: "
                f"{json.dumps(asdict(interp.nitrogen_relevance), ensure_ascii=False)}\n"
            )
            parts.append("visible_text:\n")
            parts.append(
                json.dumps(interp.visible_text, ensure_ascii=False, indent=2) + "\n"
            )
            parts.append("factual_observations:\n")
            for obs in interp.factual_observations:
                if isinstance(obs, dict):
                    parts.append(
                        f"- {clean_text(obs.get('observation', ''))} "
                        f"[source={obs.get('evidence_source')}, "
                        f"certainty={obs.get('certainty')}]\n"
                    )
            if interp.not_determinable:
                parts.append("not_determinable:\n")
                for item in interp.not_determinable:
                    parts.append(f"- {clean_text(str(item))}\n")
            parts.append("[FIGURE_INTERPRETATION_END]\n\n")
        parts.append("[/FIGURE_INTERPRETATIONS]\n")
        return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

class FigurePipeline:
    def __init__(
        self,
        chandra: Optional[ChandraOCR],
        figure_extractor: PDFFigureExtractor,
        internvl: Optional[Any],
        validator: HeuristicValidator,
    ):
        self.chandra = chandra
        self.figure_extractor = figure_extractor
        self.internvl = internvl
        self.validator = validator
        self.writer = OutputWriter()

    def process_pdf(
        self,
        pdf_path: Path,
        force: bool = False,
        max_pages: Optional[int] = None,
        page_range: Optional[str] = None,
        max_figures: Optional[int] = None,
    ) -> Optional[PaperResult]:
        import fitz

        stem = pdf_path.stem
        final_json = PATHS["final_text"] / f"{stem}.figures.json"
        if final_json.exists() and not force:
            log.info(
                f"Saltando {stem}: ya existe {final_json.name}. "
                "Usa --force para reprocesar."
            )
            return None

        t0 = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        log.info("\n" + "═" * 72)
        log.info(f"Procesando: {pdf_path.name}")
        log.info("═" * 72)

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        pages_to_process = select_pages(
            total_pages, max_pages=max_pages, page_range=page_range
        )
        doc.close()

        log.info(f"Páginas seleccionadas: {len(pages_to_process)} / {total_pages}")

        full_text = self._run_ocr(pdf_path, pages_to_process)
        title = extract_title(full_text)
        log.info(f"Título detectado: {title[:90]}")

        candidates = self.figure_extractor.extract_from_pdf(
            pdf_path=pdf_path,
            pages_to_process=pages_to_process,
            max_figures=max_figures,
        )
        log.info(f"Candidatos de figura extraídos: {len(candidates)}")

        figures: list[FigureRecord] = []
        for idx, cand in enumerate(candidates, start=1):
            context = find_context_sentences(full_text, cand.figure_number)
            interpretation = None
            validation = None

            if self.internvl is not None:
                log.info(
                    f"InternVL → {idx}/{len(candidates)} "
                    f"{cand.figure_id} ({cand.crop_quality})"
                )
                interpretation = self.internvl.interpret(
                    image_path=cand.crop_path,
                    title=title,
                    caption=cand.caption,
                    context_sentences=context,
                    candidate=cand,
                )
                validation = self.validator.validate(cand, interpretation, context)
                log.info(
                    f"  tipo={interpretation.figure_type} | "
                    f"uncertainty={interpretation.uncertainty_level} | "
                    f"validation={validation.validation_level} | "
                    f"accepted={validation.accepted_for_training}"
                )
            else:
                log.info(f"InternVL omitido para {cand.figure_id}")

            figures.append(
                FigureRecord(
                    figure_id=cand.figure_id,
                    page=cand.page,
                    caption=cand.caption,
                    context_sentences=context,
                    image_path=cand.crop_path,
                    candidate=cand,
                    interpretation=interpretation,
                    validation=validation,
                )
            )

        result = PaperResult(
            pdf_path=str(pdf_path),
            paper_stem=stem,
            paper_title=title,
            full_text=full_text,
            figures=figures,
            processing_time_s=time.time() - t0,
            timestamp=timestamp,
            metadata={
                "pipeline":        "figures_nitrogen_v3",
                "dpi":             self.figure_extractor.dpi,
                "chandra_enabled": self.chandra is not None and self.chandra.enabled,
                "internvl_enabled": self.internvl is not None,
                "max_pages":       max_pages,
                "page_range":      page_range,
                "max_figures":     max_figures,
            },
        )

        self.writer.save(result)
        return result

    def _run_ocr(self, pdf_path: Path, pages_to_process: list[int]) -> str:
        """
        v3.1: OCR inteligente por página con pymupdf4llm.

        Jerarquía de calidad de extracción (mejor a peor):
          1. pymupdf4llm.to_markdown() — markdown estructurado con tablas,
             fórmulas, multi-columna y headings. Es el método principal.
          2. page.get_text("text") — fallback si pymupdf4llm no está instalado
             o falla en una página concreta.
          3. Chandra OCR — solo para páginas escaneadas donde ambos métodos
             anteriores producen < CHANDRA_SKIP_THRESHOLD chars.

        pymupdf4llm trabaja sobre páginas individuales usando el parámetro
        `pages=[0-indexed]`, por lo que se puede integrar en el loop página
        a página manteniendo la detección de páginas escaneadas.
        """
        import fitz

        # Intentar importar pymupdf4llm una sola vez
        try:
            import pymupdf4llm
            _has_pymupdf4llm = True
        except ImportError:
            log.warning(
                "pymupdf4llm no instalado — usando page.get_text('text'). "
                "Instala con: pip install pymupdf4llm"
            )
            _has_pymupdf4llm = False

        doc = fitz.open(str(pdf_path))
        parts: list[str] = []

        try:
            for i, page_num in enumerate(pages_to_process, start=1):
                page = doc[page_num - 1]

                # ── Método principal: pymupdf4llm ────────────────────────────
                page_text = ""
                if _has_pymupdf4llm:
                    try:
                        # pages usa índice 0-based
                        md = pymupdf4llm.to_markdown(
                            doc,
                            pages=[page_num - 1],
                            show_progress=False,
                        )
                        page_text = md.strip()
                    except Exception as exc:
                        log.warning(
                            f"  p{page_num}: pymupdf4llm falló ({exc}), "
                            "usando fallback get_text"
                        )

                # ── Fallback: get_text plano ─────────────────────────────────
                if not page_text:
                    page_text = page.get_text("text").strip()

                char_count = len(page_text)

                # ── Chandra para páginas escaneadas ──────────────────────────
                use_chandra = (
                    self.chandra is not None
                    and self.chandra.enabled
                    and char_count < CHANDRA_SKIP_THRESHOLD
                )

                if use_chandra:
                    img = PDFFigureExtractor._pil_from_page(
                        page, self.figure_extractor.dpi
                    )
                    log.info(
                        f"Chandra OCR → página {page_num} "
                        f"({i}/{len(pages_to_process)}) "
                        f"[texto insuficiente: {char_count} chars]"
                    )
                    ocr = self.chandra.run_page(img, page_num)
                    page_text = ocr.markdown.strip() or page_text
                    char_count = len(page_text)
                else:
                    method = "pymupdf4llm" if _has_pymupdf4llm else "get_text"
                    if char_count >= CHANDRA_SKIP_THRESHOLD:
                        log.info(
                            f"  p{page_num}: {method} "
                            f"({char_count} chars)"
                        )
                    else:
                        log.warning(
                            f"  p{page_num}: texto escaso ({char_count} chars), "
                            f"Chandra desactivado — usando {method}"
                        )

                parts.append(f"\n\n<!-- PAGE {page_num} -->\n\n{page_text}")
        finally:
            doc.close()

        return "\n".join(parts).strip()

    def run_internvl_only(self, pdf_path: Path) -> Optional[PaperResult]:
        stem = pdf_path.stem
        ocr_path      = PATHS["final_text"]     / f"{stem}.ocr.txt"
        captions_path = PATHS["figure_captions"] / f"{stem}.candidates.json"

        if self.internvl is None:
            log.error("--internvl-only requiere InternVL activo")
            return None
        if not ocr_path.exists():
            log.error(f"No existe texto OCR previo: {ocr_path}")
            return None
        if not captions_path.exists():
            log.error(f"No existe JSON de candidatos previo: {captions_path}")
            return None

        t0 = time.time()
        full_text  = ocr_path.read_text(encoding="utf-8")
        title      = extract_title(full_text)
        raw_cands  = json.loads(captions_path.read_text(encoding="utf-8"))
        candidates = [FigureCandidate(**c) for c in raw_cands]

        figures: list[FigureRecord] = []
        for idx, cand in enumerate(candidates, start=1):
            context = find_context_sentences(full_text, cand.figure_number)
            log.info(f"InternVL-only → {idx}/{len(candidates)} {cand.figure_id}")
            interpretation = self.internvl.interpret(
                image_path=cand.crop_path,
                title=title,
                caption=cand.caption,
                context_sentences=context,
                candidate=cand,
            )
            validation = self.validator.validate(cand, interpretation, context)
            log.info(
                f"  tipo={interpretation.figure_type} | "
                f"uncertainty={interpretation.uncertainty_level} | "
                f"validation={validation.validation_level} | "
                f"accepted={validation.accepted_for_training}"
            )
            figures.append(
                FigureRecord(
                    figure_id=cand.figure_id,
                    page=cand.page,
                    caption=cand.caption,
                    context_sentences=context,
                    image_path=cand.crop_path,
                    candidate=cand,
                    interpretation=interpretation,
                    validation=validation,
                )
            )

        result = PaperResult(
            pdf_path=str(pdf_path),
            paper_stem=stem,
            paper_title=title,
            full_text=full_text,
            figures=figures,
            processing_time_s=time.time() - t0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata={"pipeline": "figures_nitrogen_v3", "mode": "internvl_only"},
        )
        self.writer.save(result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_internvl_client(args) -> Optional[Any]:
    if args.no_internvl:
        return None

    if args.internvl_mode == "server":
        return InternVLServerClient(
            base_url=args.internvl_base_url,
            api_key=args.internvl_api_key,
            model_name=args.internvl_model_name,
            timeout_s=args.internvl_timeout,
        )

    if args.internvl_mode == "offline":
        return InternVLOfflineClient(
            model_path=args.internvl_model_path,
            gpu_util=args.gpu_util,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tp_size,
        )

    raise ValueError(f"internvl_mode inválido: {args.internvl_mode}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline v3: OCR inteligente + crops + InternVL3 + validación.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pdf", type=Path, default=None,
        help="PDF específico. Si se omite, procesa data/raw_pdfs/*.pdf",
    )
    parser.add_argument("--force", action="store_true",
                        help="Reprocesar aunque exista output final")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Procesar solo primeras N páginas")
    parser.add_argument("--page-range", type=str, default=None,
                        help='Páginas específicas. Ej: "8-10", "1,3,5"')
    parser.add_argument("--max-figures", type=int, default=None,
                        help="Máximo de figuras a procesar")
    parser.add_argument("--dpi", type=int, default=220,
                        help="DPI base para OCR y crops (min efectivo: 800px corto)")

    parser.add_argument("--no-chandra", action="store_true",
                        help="No usar Chandra en ninguna página")
    parser.add_argument("--chandra-model-path", type=Path, default=PATHS["chandra"],
                        help="Ruta local Chandra OCR 2")
    parser.add_argument(
        "--chandra-backend",
        choices=["hf_lowlevel", "manager_hf", "manager_vllm"],
        default="hf_lowlevel",
    )

    parser.add_argument("--no-internvl", action="store_true",
                        help="Solo OCR + crops, sin interpretar figuras")
    parser.add_argument("--internvl-only", action="store_true",
                        help="Reusar OCR/candidatos previos y correr solo InternVL")
    parser.add_argument("--internvl-mode", choices=["server", "offline"],
                        default="offline")
    parser.add_argument("--internvl-model-path", type=Path,
                        default=PATHS["internvl3"])
    parser.add_argument("--internvl-base-url", type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--internvl-api-key", type=str, default="EMPTY")
    parser.add_argument("--internvl-model-name", type=str, default=None)
    parser.add_argument("--internvl-timeout", type=int, default=180)

    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=1)

    parser.add_argument("--include-caption-in-crop", action="store_true")
    parser.add_argument(
        "--allow-full-page-fallback", action="store_true",
        help="Usar página completa si no se aísla crop (marcado high-risk).",
    )

    return parser


def validate_args(args) -> None:
    if args.max_pages is not None and args.max_pages <= 0:
        raise ValueError("--max-pages debe ser positivo")
    if args.max_figures is not None and args.max_figures <= 0:
        raise ValueError("--max-figures debe ser positivo")
    if args.dpi <= 0:
        raise ValueError("--dpi debe ser positivo")
    if args.page_range:
        parse_page_range(args.page_range)
    if args.page_range and args.max_pages:
        log.warning("--page-range y --max-pages: --page-range tiene prioridad")
    if args.internvl_only and args.no_internvl:
        raise ValueError("--internvl-only no puede combinarse con --no-internvl")


def discover_pdfs(pdf: Optional[Path]) -> list[Path]:
    if pdf is not None:
        if not pdf.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf}")
        return [pdf]
    pdfs = sorted(PATHS["raw_pdfs"].glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No hay PDFs en {PATHS['raw_pdfs']}")
    return pdfs


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except Exception as exc:
        log.error(str(exc))
        sys.exit(1)

    try:
        pdfs = discover_pdfs(args.pdf)
    except Exception as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info(f"PDFs a procesar: {len(pdfs)}")
    log.info(f"Log: {log_file}")

    chandra = None
    if not args.internvl_only:
        chandra = ChandraOCR(
            model_path=args.chandra_model_path,
            backend=args.chandra_backend,
            enabled=not args.no_chandra,
        )

    internvl = build_internvl_client(args)

    figure_extractor = PDFFigureExtractor(
        dpi=args.dpi,
        include_caption_in_crop=args.include_caption_in_crop,
        allow_full_page_fallback=args.allow_full_page_fallback,
    )

    pipeline = FigurePipeline(
        chandra=chandra,
        figure_extractor=figure_extractor,
        internvl=internvl,
        validator=HeuristicValidator(),
    )

    t0 = time.time()
    processed = 0

    for pdf_path in pdfs:
        if args.internvl_only:
            result = pipeline.run_internvl_only(pdf_path)
        else:
            result = pipeline.process_pdf(
                pdf_path=pdf_path,
                force=args.force,
                max_pages=args.max_pages,
                page_range=args.page_range,
                max_figures=args.max_figures,
            )
        if result is not None:
            processed += 1

    elapsed = time.time() - t0
    log.info(f"✓ Completado: {processed}/{len(pdfs)} papers en {elapsed:.1f}s")
    log.info(f"Outputs: {PATHS['final_text']}")


if __name__ == "__main__":
    main()