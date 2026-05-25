# scientific-figure-extractor

Two-stage pipeline for scientific PDF analysis:

1. **`extract_figures.py`** — extracts individual figures and tables from PDFs as PNG images (no ML, pure geometry)
2. **`analyze_figures.py`** — analyzes each extracted figure with a vision-language model via any OpenAI-compatible API, generating structured training data

---

## Why

Feeding a full PDF page to a vision-language model is poor practice: the model sees body text, multiple figures, headers, and captions all mixed together. Interpretation quality drops significantly.

This pipeline isolates each figure into its own image with precise bounding boxes, then prompts the model with a rigid structure designed to produce high-quality, consistent training labels.

---

## Pipeline 1: extract_figures.py

### How it works

1. Detects captions via regex over text blocks: `Figure N`, `Fig. N`, `Table N`, `Extended Data Fig. N`
2. Classifies the caption's column (left / right / full-width) from its horizontal position
3. Finds the visual content above the caption in the same column, within a configurable max height
4. Expands the bounding box to include narrow text elements (axis labels, panel letters, colorbars) within 60pt
5. Filters body text wider than 35% of page width to avoid capturing paragraphs
6. Handles cross-page figures: if the caption is on page N but the visual is on page N+1, searches the next page
7. Falls back to pixel rendering for Form XObjects (figures invisible to PyMuPDF's drawing API)
8. Renders the final bbox to PNG at configurable DPI

No ML. Runs on CPU in milliseconds per page.

### Usage

```bash
python extract_figures.py paper.pdf --out extracted/
```

Output structure:
```
extracted/
├── p002_Figure_1.png
├── p003_Figure_2.png
├── p005_Table_1.png
├── ...
└── figures.json
```

Options:
```
--out DIR        output directory (default: extracted/)
--dpi N          render resolution in DPI (default: 200)
--max-height N   max figure height in points (default: 9999)
--quiet          suppress stdout
```

### figures.json format

```json
{
  "pdf": "paper.pdf",
  "total": 15,
  "items": [
    {
      "label":      "Figure 3",
      "kind":       "figure",
      "page":       3,
      "bbox":       [300.4, 64.0, 553.1, 219.0],
      "caption":    "Fig. 3 | Overall architecture of ...",
      "image_path": "extracted/p003_Figure_3.png",
      "image_size": [503, 310]
    }
  ]
}
```

`kind` is either `"figure"` or `"table"`.

---

## Pipeline 2: analyze_figures.py

Takes `figures.json` from the extractor and sends each figure to a vision-language model. Generates **two analyses per figure**:

- `inference` — pure visual analysis: image + caption only, no paper context
- `anchored` — grounded analysis: image + caption + full paper context (or LLM-generated summary if the paper is too large)

Designed to produce **structured training data** with consistent, parseable sections.

### Usage

```bash
python analyze_figures.py extracted/figures.json \
    --pdf paper.pdf \
    --server http://127.0.0.1:8080/v1/chat/completions \
    --out results/model_name.json
```

Options:
```
--pdf FILE           original PDF (required for anchored analysis)
--server URL         OpenAI-compatible endpoint (default: http://127.0.0.1:8080/v1/chat/completions)
--out FILE           output path (default: analyses.json next to figures.json)
--inference-only     skip anchored analysis
--anchored-only      skip inference analysis (requires --pdf)
--budget-tokens N    max tokens reserved for paper context (default: 6000)
--max-tokens N       max tokens to generate per response (default: 1500)
--temperature F      sampling temperature (default: 0.0 — deterministic)
--timeout N          seconds per request before retry (default: 300)
```

Resumes automatically if interrupted: already-completed figures are skipped.

### Smart paper context

The script reads the full paper text and estimates its token count:

- If it fits within `--budget-tokens` → sends the full paper as context
- If not → calls the LLM once to generate a structured summary, then reuses it for all figures

The summary prompt asks for: research question, methods, main findings, per-figure descriptions, and conclusions. This happens once, not once per figure.

### Inference prompt structure (figures)

Generated for every figure using the image and its caption:

```
## Visual Description
## Figure Type
## Statistical Markers
## Data and Patterns
## Caption Alignment
## Scientific Interpretation
## Significance
```

**Statistical Markers** forces the model to report exactly what is visible (n=, error bars, p-values, R²) or state "None visible" — it cannot infer or assume values not shown in the image.

**Caption Alignment** checks whether the caption accurately describes what is shown, surfacing omissions or overstatements in the original paper.

### Anchored prompt structure (figures)

Generated for every figure using the image, caption, and paper context:

```
## Visual Description
## Caption Alignment
## Hypothesis Tested
## Key Data and Findings
## Statistical Markers
## Controls Assessment
## Mechanistic Insight
## Narrative Role
## Limitations / Caveats
## Scientific Conclusion
```

**Controls Assessment** identifies experimental controls present in the figure and flags any conspicuously absent ones given the experimental design described in the paper.

**Scientific Conclusion** synthesizes visual evidence with the paper's framework into a precise, self-contained conclusion: what this figure definitively demonstrates, what alternative explanations it rules out, and its conceptual significance in the paper's broader argument.

The inference prompt is intentionally left without this section — drawing scientific conclusions requires paper context.

### Output format

```json
{
  "total": 15,
  "context_mode": "summary",
  "items": [
    {
      "label":        "Figure 1",
      "kind":         "figure",
      "page":         2,
      "caption":      "Fig. 1 | Genome assemblies and size variation...",
      "image_path":   "extracted/p002_Figure_1.png",
      "image_size":   [571, 291],
      "inference":    "## Visual Description\n...",
      "anchored":     "## Visual Description\n...",
      "elapsed_sec":  297.1,
      "context_mode": "summary"
    }
  ]
}
```

`context_mode` is `"full"` if the paper fit within the budget, `"summary"` if it was summarized.

### Running a local vision-language model

Any OpenAI-compatible server with vision support works. Tested setup using [llama.cpp](https://github.com/ggml-org/llama.cpp) with InternVL3-14B on a single GPU:

```bash
llama-server \
    -m InternVL3-14B-Q4_K_M.gguf \
    --mmproj mmproj-InternVL3-14B-Q8_0.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 \
    -c 8192 \
    --jinja
```

For limited VRAM, reduce parallel slots (`-np 1`) to free KV cache memory for larger context:

```bash
llama-server \
    -m InternVL3-14B-Q4_K_M.gguf \
    --mmproj mmproj-InternVL3-14B-Q8_0.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 -np 1 \
    -c 16384 \
    --jinja
```

Recommended models (GGUF):
- [ggml-org/InternVL3-14B-Instruct-GGUF](https://huggingface.co/ggml-org/InternVL3-14B-Instruct-GGUF)
- [Qwen/Qwen2.5-VL-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF)

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.9+
- PyMuPDF >= 1.24
- requests >= 2.31

---

## Full example

```bash
# Step 1: extract figures from a paper
python extract_figures.py paper.pdf --out extracted/ --dpi 200

# Step 2: analyze with a local VLM (runs overnight for large papers)
python analyze_figures.py extracted/figures.json \
    --pdf paper.pdf \
    --server http://localhost:8080/v1/chat/completions \
    --out results/internvl3_14b.json

# Resume if interrupted — already-done figures are skipped automatically
python analyze_figures.py extracted/figures.json \
    --pdf paper.pdf \
    --out results/internvl3_14b.json
```

---

## Limitations

- Assumes captions appear below the figure (standard in most journals). Captions above figures are not detected.
- Figures embedded as Form XObjects (rare) fall back to pixel rendering, which is slower.
- Inline figure references ("see Fig. 5") occasionally produce false caption detections — the extractor skips them if no visual region is found above.
- Table extraction renders the text block as an image rather than parsing cell structure.

---

## License

MIT — see [LICENSE](LICENSE).
