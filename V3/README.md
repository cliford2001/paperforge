# PaperForge V3

V3 is the runnable snapshot of the current layered PaperForge pipeline.

It contains only the files needed to run the latest workflow:

1. read papers from parquet;
2. download PDFs;
3. extract figures and tables with Docling/PyMuPDF;
4. filter Docling logo/icon artifacts;
5. build prompt context with `none`, `bm25`, `full`, or `layered`;
6. summarize OCR dynamically instead of inserting raw OCR;
7. call an OpenAI-compatible VLM server;
8. write `analyses_rag.json`.

## Files

| Path | Purpose |
|---|---|
| `main.py` | Single entrypoint for the full pipeline |
| `db.py` | Reads parquet input |
| `download_pdf.py` | Downloads PDFs and writes `paper_context.txt` |
| `figures.py` | Docling/PyMuPDF figure extraction and logo/icon filtering |
| `tables.py` | Table extraction |
| `analyze.py` | Dynamic prompt, context modes, filtered OCR, JSON schema, VLM analysis |
| `slurm/pforge_4models_all_modes.slurm` | Slurm used for the current 4-model comparison |
| `slurm/paperforge_layered_template.slurm` | Generic Slurm template for another cluster |

## Install

```bash
cd V3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install vllm
```

`vllm` should be installed according to the CUDA/PyTorch stack of the target cluster.

## Run Extraction Only

```bash
python main.py \
  --input-parquet /path/to/input.parquet \
  --out-dir /path/to/results_extract \
  --keep-pdf \
  --reuse-existing-pdf
```

## Run Layered Analysis

Start a VLM server first:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --port 8097 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code
```

Then run:

```bash
python main.py \
  --input-parquet /path/to/input.parquet \
  --out-dir /path/to/results_layered \
  --keep-pdf \
  --reuse-existing-pdf \
  --run-analysis \
  --server http://127.0.0.1:8097/v1/chat/completions \
  --context-mode layered \
  --top-k 6 \
  --chunk-words 250 \
  --chunk-overlap 40 \
  --max-tokens 2200 \
  --temperature 0.0 \
  --timeout 900
```

## Context Modes

| Mode | Prompt input |
|---|---|
| `none` | image + caption + abstract if available |
| `bm25` | image + caption + retrieved local chunks |
| `full` | image + caption + full or trimmed paper text |
| `layered` | image + caption + compact paper map + local BM25 chunks + filtered OCR summary |

`layered` is the recommended mode for the current system.

## Output

Each paper directory contains:

| File | Meaning |
|---|---|
| `paper.pdf` | Downloaded source PDF |
| `paper_context.txt` | Metadata/full text context from input parquet |
| `figures.json` | Extracted figure metadata |
| `tables.json` | Extracted table metadata |
| `skipped_figures.json` | Docling picture artifacts skipped as likely logos/icons |
| `analyses_rag.json` | Final VLM analysis |

`analyses_rag.json` contains item metadata, filtered OCR summary, raw model response, parsed JSON, context metadata, and final paper summary.
