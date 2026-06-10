from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PaperRecord:
    pmcid: str
    doi: str = ""
    title: str = ""
    text_clean: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_columns(df, required: Iterable[str]) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"faltan columnas requeridas: {', '.join(missing)}")


def _normalize_record(row) -> PaperRecord:
    return PaperRecord(
        pmcid=str(getattr(row, "pmcid", "") or "").strip(),
        doi=str(getattr(row, "doi", "") or "").strip(),
        title=str(getattr(row, "title", "") or "").strip(),
        text_clean=str(getattr(row, "text_clean", "") or "").strip(),
    )


def load_papers(
    input_parquet: str | None = None,
    metadata_parquet: str | None = None,
    texts_parquet: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[PaperRecord]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("requiere pandas y pyarrow para leer parquet") from exc

    if input_parquet:
        df = pd.read_parquet(Path(input_parquet))
        _ensure_columns(df, ["pmcid"])
        if "doi" not in df.columns:
            df["doi"] = ""
        if "title" not in df.columns:
            df["title"] = ""
        if "text_clean" not in df.columns:
            df["text_clean"] = ""
    else:
        if not metadata_parquet or not texts_parquet:
            raise ValueError("debes pasar --input-parquet o bien --metadata-parquet y --texts-parquet")
        meta = pd.read_parquet(Path(metadata_parquet))
        texts = pd.read_parquet(Path(texts_parquet))
        _ensure_columns(meta, ["pmcid"])
        _ensure_columns(texts, ["pmcid", "text_clean"])
        if "doi" not in meta.columns:
            meta["doi"] = ""
        if "title" not in meta.columns:
            meta["title"] = ""
        df = meta.merge(texts[["pmcid", "text_clean"]], on="pmcid", how="left")
        df["text_clean"] = df["text_clean"].fillna("")

    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.iloc[:limit]

    papers: list[PaperRecord] = []
    for row in df.itertuples(index=False):
        record = _normalize_record(row)
        if not record.pmcid:
            continue
        papers.append(record)
    return papers
