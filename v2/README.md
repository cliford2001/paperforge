# paperforge/v2

`v2` deja el pipeline separado y reusable:

- `db.py`: carga de papers desde parquet o metadata+texts
- `pdfs.py`: descarga y persistencia de `paper.pdf`
- `figures.py`: extracción de figuras
- `tables.py`: extracción de tablas
- `analyzer.py`: wrapper del análisis multimodal
- `main.py`: orquestación de punta a punta

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SQLite / Parquet                               │
│                    DOI  ·  text_clean  ·  metadata                      │
└────────────────────┬────────────────────────────┬───────────────────────┘
                     │                            │
                     ▼                            ▼
          ┌──────────────────┐        ┌───────────────────────┐
          │  Descarga PDF    │        │  context.json /       │
          │  (DOI / PMCID /  │        │  text_clean del       │
          │   legal sources) │        │  parquet              │
          └────────┬─────────┘        └──────────┬────────────┘
                   │                             │
                   ▼                             ▼
          ┌──────────────────┐        ┌──────────────────────────────┐
          │   paper.pdf      │        │   Preparación de contexto    │
          └────────┬─────────┘        │                              │
                   │                  │  1. limpieza ligera          │
                   ▼                  │     espacios / ruido menor   │
          ┌──────────────────┐        │                              │
          │   Extracción     │        │  2. abstract detectado       │
          │                  │        │     por heading / heurística │
          │  figures.json    │        │                              │
          │  + PNGs          │        │  3. chunking moderado        │
          │                  │        │     unidades de sentido      │
          │  tables.json     │        │                              │
          │  + PNGs          │        │  4. índice BM25              │
          └────────┬─────────┘        │                              │
                   │                  │  5. query por item:          │
                   │                  │     label + caption + terms  │
                   │                  └──────────┬───────────────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                 Por cada figura / tabla                              │
    │                                                                      │
    │   item visual = PNG recortado                                        │
    │   item textual = label + caption                                     │
    │                                                                      │
    │   recuperación de contexto                                           │
    │                                                                      │
    │   ┌──────────────────────────────────────────────────────────────┐   │
    │   │   BM25 -> chunks candidatos                                  │   │
    │   │   presupuesto -> limitar largo total                         │   │
    │   │                                                              │   │
    │   │   bloque final anti "lost in the middle"                     │   │
    │   │   (Liu et al. 2023, arXiv:2307.03172)                        │   │
    │   │   A. caption                  ← inicio (máx. atención)       │   │
    │   │   B. chunk más relevante      ← justo después del caption    │   │
    │   │   C. chunks de apoyo          ← medio (orden documental)     │   │
    │   │   D. chunk segundo relevante  ← final del bloque RAG         │   │
    │   │   E. abstract                 ← cierre (máx. atención)       │   │
    │   └──────────────────────────────┬───────────────────────────────┘   │
    │                                  │                                   │
    │                                  ▼                                   │
    │   PROMPT DINÁMICO                                                     │
    │                                                                      │
    │   ┌──────────────────────────────────────────────────────────────┐   │
    │   │ Figure/Table caption: {caption}                              │   │
    │   │                                                              │   │
    │   │ RELEVANT PAPER SECTIONS:                                     │   │
    │   │ {chunk más relevante}                                        │   │
    │   │ ---                                                          │   │
    │   │ {chunks de apoyo en orden documental}                        │   │
    │   │ ---                                                          │   │
    │   │ {segundo chunk más relevante}                                │   │
    │   │                                                              │   │
    │   │ PAPER ABSTRACT:                                              │   │
    │   │ {abstract}                                                   │   │
    │   │                                                              │   │
    │   │ instrucciones de análisis                                    │   │
    │   │ - visual description                                         │   │
    │   │ - type                                                       │   │
    │   │ - statistical markers                                        │   │
    │   │ - data/patterns o key entries                                │   │
    │   │ - caption alignment                                          │   │
    │   │ - scientific interpretation                                  │   │
    │   │ - hypothesis_tested / paper_quote / controls                 │   │
    │   │   solo si hubo contexto recuperado                           │   │
    │   │                                                              │   │
    │   │ output fijo: JSON parseable                                  │   │
    │   └──────────────────────────────┬───────────────────────────────┘   │
    │                                  │ + imagen PNG                       │
    │                                  ▼                                   │
    │                         ┌──────────────────────┐                     │
    │                         │   VLM / llama.cpp    │                     │
    │                         │   Qwen / InternVL /  │                     │
    │                         │   MiniCPM            │                     │
    │                         └──────────┬───────────┘                     │
    │                                    ▼                                 │
    │              analysis_parsed / table_analysis_parsed                 │
    │                                                                      │
    │   {                                                                  │
    │     figure_type or table_type,                                       │
    │     visual_description or structure,                                 │
    │     statistical_markers,                                             │
    │     data_and_patterns or key_entries,                                │
    │     caption_accurate,                                                │
    │     scientific_interpretation,                                       │
    │     scientific_conclusion,                                           │
    │     hypothesis_tested*,                                              │
    │     paper_quote*,                                                    │
    │     controls_assessment*,                                            │
    │     context_used,                                                    │
    │     confidence                                                       │
    │   }                                                                  │
    │                                                                      │
    │   * solo si hubo contexto RAG / BM25                                 │
    └──────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   │ repite para todos los items
                                   ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    Síntesis final del paper                          │
    │                   (llamada de solo texto)                            │
    │                                                                      │
    │   input: todos los analysis_parsed del paper                         │
    │                                                                      │
    │   output:                                                            │
    │   {                                                                  │
    │     main_contribution,                                               │
    │     narrative,                                                       │
    │     key_evidence,                                                    │
    │     contradictions_or_gaps,                                          │
    │     limitations_noted,                                               │
    │     overall_confidence                                               │
    │   }                                                                  │
    └──────────────────────────────┬───────────────────────────────────────┘
                                   ▼
                         analyses_rag.json
                       (un archivo por paper)


IDEA CENTRAL
============

No mandar el paper completo al modelo.
Mandar:

    imagen + caption + contexto corto, relevante y ordenado

para evitar "lost in the middle" y mantener fijo el JSON de salida
que ya sirve aguas abajo.
```

## Flujo simplificado

```text
input parquet
  -> load_papers()       db.py
  -> download_pdf()      pdfs.py
  -> extract_figures()   figures.py
  -> extract_tables()    tables.py
  -> analyze()           analyzer.py   (opcional, --run-analysis)
  -> summary.json
```

## Salida por paper

```text
results_x/
  PMCxxxx/
    context.json
    paper_context.txt
    paper.pdf
    figures.json
    tables.json
    figures.normalized.json
    tables.normalized.json
    pXXX_Figure_N.png
    pXXX_Table_N.png
    analyses_rag.json        # solo si --run-analysis
```

## Modos de anǭlisis de contexto

`v2` soporta tres modos cuando corres `--run-analysis`:

- `--context-mode none`
  - el modelo recibe solo `imagen + caption`
  - no se le pasa contexto textual del paquete
  - ǧtil para test rǭpido o para aislar el comportamiento visual

- `--context-mode bm25`
  - usa `paper_context.txt` / `text_clean`
  - hace chunking y recuperaci��n BM25 por item
  - pasa solo contexto relevante
  - es el modo recomendado por defecto

- `--context-mode full`
  - pasa el texto completo del paper como contexto
  - ǧtil solo con servidor/modelo que aguante contexto amplio
  - puede requerir `--max-context-words`

## Uso

### 1. Solo extracci��n

```bash
python E:\TEST_PAPERFORGE\paperforge\v2\main.py ^
  --input-parquet E:\TEST_PAPERFORGE\sample5.parquet ^
  --out-dir E:\TEST_PAPERFORGE\results_sample5_v2 ^
  --keep-pdf ^
  --reuse-existing-pdf
```

### 2. Extracci��n + anǭlisis sin contexto

```bash
python E:\TEST_PAPERFORGE\paperforge\v2\main.py ^
  --input-parquet E:\TEST_PAPERFORGE\sample5.parquet ^
  --out-dir E:\TEST_PAPERFORGE\results_sample5_v2_nocontext ^
  --keep-pdf ^
  --reuse-existing-pdf ^
  --run-analysis ^
  --server http://127.0.0.1:8080/v1/chat/completions ^
  --context-mode none
```

### 3. Extracci��n + anǭlisis con contexto BM25

```bash
python E:\TEST_PAPERFORGE\paperforge\v2\main.py ^
  --input-parquet E:\TEST_PAPERFORGE\sample5.parquet ^
  --out-dir E:\TEST_PAPERFORGE\results_sample5_v2_bm25 ^
  --keep-pdf ^
  --reuse-existing-pdf ^
  --run-analysis ^
  --server http://127.0.0.1:8080/v1/chat/completions ^
  --context-mode bm25 ^
  --top-k 6 ^
  --chunk-words 180 ^
  --chunk-overlap 30 ^
  --max-tokens 800
```

### 4. Extracci��n + anǭlisis con full context

```bash
python E:\TEST_PAPERFORGE\paperforge\v2\main.py ^
  --input-parquet E:\TEST_PAPERFORGE\sample5.parquet ^
  --out-dir E:\TEST_PAPERFORGE\results_sample5_v2_full ^
  --keep-pdf ^
  --reuse-existing-pdf ^
  --run-analysis ^
  --server http://127.0.0.1:8080/v1/chat/completions ^
  --context-mode full ^
  --max-context-words 3000 ^
  --max-tokens 800
```

## Recomendaci��n prǭctica

Si estǭs probando modelos chicos o con contexto limitado:

- partir con `--context-mode none`
- luego probar `--context-mode bm25`
- dejar `full` para servidores/modelos con mǭs ventana de contexto

## Notas

- `paper_context.txt` se genera automǭticamente desde `text_clean`
- si ya existe `paper.pdf`, `--reuse-existing-pdf` evita redescarga
- `summary.json` resume estado por paper
- `analyses_rag.json` contiene anǭlisis por item y s��ntesis final del paper
