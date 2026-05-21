# scientific-figure-extractor

Extracción inteligente de figuras y tablas individuales desde PDFs científicos, con análisis opcional vía LLM multimodal (InternVL, Qwen-VL, LLaVA, etc.).

## Por qué

Renderizar páginas completas de un PDF como imagen es un **pésimo input** para modelos visión-lenguaje: el modelo ve texto del paper, varias figuras, headers, todo mezclado. La interpretación se diluye.

Este extractor produce **una imagen por figura/tabla**, con bbox preciso, respetando:
- Layout multi-columna (izquierda / derecha / full-width)
- Gráficos vectoriales (drawings) **además** de imágenes embebidas
- Límites entre figuras consecutivas (Figure 9 no contamina Figure 10)
- Tablas de texto puro (no son imágenes, son bloques formateados)

## Cómo funciona

1. **Detecta captions** vía regex sobre cada bloque de texto: `Figure N`, `Fig. N`, `Table N`, `Extended Data Fig. N`.
2. **Clasifica columna** del caption (izq / der / full) según posición horizontal.
3. **Encuentra contenido visual arriba del caption**, en la misma columna, dentro de una altura máxima razonable.
4. **Limita con el caption anterior** en la misma columna para que figuras consecutivas no se mezclen.
5. **Tablas**: recolecta bloques de texto contiguos en lugar de imágenes/drawings.
6. **Renderiza** el bbox resultante a PNG en DPI configurable.

Sin ML, solo geometría. Funciona en CPU en milisegundos.

## Instalación

```bash
pip install -r requirements.txt
```

Requisitos:
- Python 3.9+
- PyMuPDF (`fitz`)
- requests (solo para `analyze_figures.py`)

## Uso

### Solo extracción

```bash
python extract_figures.py paper.pdf --out extracted/
```

Output:
```
extracted/
├── p001_Figure_1.png
├── p002_Figure_2.png
├── p003_Table_1.png
├── ...
└── figures.json     ← metadata con bbox, caption, página
```

Opciones:
```
--out DIR          directorio de salida (def: extracted/)
--dpi N            resolución del render (def: 200)
--max-height N     altura máxima del bbox de una figura en pt (def: 600)
--quiet            sin output a stdout
```

### Análisis con LLM (opcional)

Requiere un servidor OpenAI-compatible con un modelo visión-lenguaje (probado con [InternVL3-14B](https://huggingface.co/OpenGVLab/InternVL3-14B) en llama.cpp).

Análisis con contexto del paper (recomendado):
```bash
python analyze_figures.py extracted/figures.json --pdf paper.pdf \
    --server http://127.0.0.1:8080/v1/chat/completions
```

Solo análisis sin contexto:
```bash
python analyze_figures.py extracted/figures.json --no-context
```

Output: `extracted/analyses.json` con dos análisis por figura:
- `analysis_no_ctx` — interpretación basada solo en la imagen
- `analysis_with_ctx` — interpretación con texto de las páginas adyacentes (±3 por defecto)

Resume automático: si se interrumpe, al reiniciar salta items ya procesados.

Opciones:
```
--server URL       endpoint OpenAI-compatible (def: localhost:8080)
--pdf FILE         PDF original (necesario para --with-context, default)
--no-context       solo análisis sin contexto del paper
--window N         páginas ±N alrededor de la figura para contexto (def: 3)
--max-tokens N     límite de tokens generados (def: 600)
--temperature F    (def: 0.1, determinístico)
--timeout N        seg por request (def: 300)
--out FILE         ruta de salida JSON
```

## Servir el modelo local

Cualquier servidor OpenAI-compatible con soporte vision funciona. Ejemplo con [llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
llama-server \
    -m InternVL3-14B-Instruct-Q4_K_M.gguf \
    --mmproj mmproj-InternVL3-14B-Instruct-Q8_0.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 -c 16384 \
    -ctk q4_0 -ctv q4_0 \
    --jinja
```

Modelo recomendado: GGUFs oficiales en [ggml-org/InternVL3-14B-Instruct-GGUF](https://huggingface.co/ggml-org/InternVL3-14B-Instruct-GGUF).

## Formato de figures.json

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
      "caption":    "Figure 3. Overall Architecture. ...",
      "image_path": "extracted/p003_Figure_3.png",
      "image_size": [253, 155]
    }
  ]
}
```

## Limitaciones

- Asume captions debajo de la figura (estándar en papers de ciencias). Captions sobre la figura no detectadas.
- Tablas reportadas como ASCII/Unicode (no imágenes) — la "imagen" extraída es el bloque de texto rasterizado. Para tablas como bitmap, usar `find_figure_region`.
- Pocas falsas detecciones en PDFs con menciones inline tipo "see Fig. 5".

## Licencia

MIT — ver [LICENSE](LICENSE).
