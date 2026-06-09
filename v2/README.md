# paperforge/v2

`v2` deja el pipeline separado y reusable:

- `db.py`: carga de papers desde parquet o metadata+texts
- `pdfs.py`: descarga y persistencia de `paper.pdf`
- `figures.py`: extracci«¸n de figuras
- `tables.py`: extracci«¸n de tablas
- `analyzer.py`: wrapper del an«≠lisis multimodal
- `main.py`: orquestaci«¸n de punta a punta

## Flujo

```text
input parquet / paquete
  -> load papers
  -> download paper.pdf
  -> extract figures
  -> extract tables
  -> optional multimodal analysis
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

## Modos de an«≠lisis de contexto

`v2` soporta tres modos cuando corres `--run-analysis`:

- `--context-mode none`
  - el modelo recibe solo `imagen + caption`
  - no se le pasa contexto textual del paquete
  - «ßtil para test r«≠pido o para aislar el comportamiento visual

- `--context-mode bm25`
  - usa `paper_context.txt` / `text_clean`
  - hace chunking y recuperaci«¸n BM25 por item
  - pasa solo contexto relevante
  - es el modo recomendado por defecto

- `--context-mode full`
  - pasa el texto completo del paper como contexto
  - «ßtil solo con servidor/modelo que aguante contexto amplio
  - puede requerir `--max-context-words`

## Uso

### 1. Solo extracci«¸n

```bash
python E:\TEST_PAPERFORGE\paperforge\v2\main.py ^
  --input-parquet E:\TEST_PAPERFORGE\sample5.parquet ^
  --out-dir E:\TEST_PAPERFORGE\results_sample5_v2 ^
  --keep-pdf ^
  --reuse-existing-pdf
```

### 2. Extracci«¸n + an«≠lisis sin contexto

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

### 3. Extracci«¸n + an«≠lisis con contexto BM25

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

### 4. Extracci«¸n + an«≠lisis con full context

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

## Recomendaci«¸n pr«≠ctica

Si est«≠s probando modelos chicos o con contexto limitado:

- partir con `--context-mode none`
- luego probar `--context-mode bm25`
- dejar `full` para servidores/modelos con m«≠s ventana de contexto

## Notas

- `paper_context.txt` se genera autom«≠ticamente desde `text_clean`
- si ya existe `paper.pdf`, `--reuse-existing-pdf` evita redescarga
- `summary.json` resume estado por paper
- `analyses_rag.json` contiene an«≠lisis por item y s«ntesis final del paper
