"""
io_corpus.py
──────────────────────────────────────────────────────────────
Pure helpers to build a unified corpus from a DMO file (imagen proyectada)
and one or more UGC files (imagen percibida).

No prints, no side effects: the CLI entrypoint is responsible for logging.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

# Column and label conventions used across the pipeline.
SOURCE_COLUMN = "source"
CORPUS_COLUMN = "corpus"
DMO_LABEL = "sectur"

# Aliases accepted when the requested text column is not present.
TEXT_COLUMN_ALIASES: Sequence[str] = (
    "text",
    "texto",
    "comment",
    "comentario",
    "body",
    "contenido",
    "description",
    "descripcion",
)


def resolve_text_column(
    df: pd.DataFrame,
    requested: str,
    path: Optional[Path] = None,
) -> str:
    """Return the text column to use, falling back to common aliases."""
    if requested in df.columns:
        return requested

    lowered = {str(col).lower(): col for col in df.columns}
    for alias in TEXT_COLUMN_ALIASES:
        if alias in lowered:
            return lowered[alias]

    where = f" de {path}" if path is not None else ""
    raise KeyError(
        f"No se encontró la columna de texto '{requested}'{where}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def load_corpus(
    path: str,
    text_column: str,
    source_column: str = SOURCE_COLUMN,
    corpus_label: str = "UGC",
    source_label: Optional[str] = None,
    dmo_label: str = DMO_LABEL,
) -> pd.DataFrame:
    """
    Lee un CSV de corpus y normaliza sus columnas.

    - Renombra la columna de texto resuelta a `text_column`.
    - Garantiza la columna de fuente (`source_column`).
    - Añade la columna de corpus (`corpus`) con DMO / UGC.
    """
    csv_path = Path(path)
    df = pd.read_csv(csv_path)

    resolved_text = resolve_text_column(df, text_column, csv_path)
    if resolved_text != text_column:
        df = df.rename(columns={resolved_text: text_column})

    if source_column not in df.columns:
        df[source_column] = source_label or corpus_label.lower()

    if corpus_label.upper() == "DMO":
        # El corpus proyectado siempre queda identificado con la etiqueta DMO.
        df[source_column] = dmo_label

    df[CORPUS_COLUMN] = corpus_label.upper()
    return df


def build_unified_frame(
    dmo_csv: str,
    ugc_csvs: Sequence[str],
    text_column: str,
    source_column: str = SOURCE_COLUMN,
    dmo_label: str = DMO_LABEL,
    drop_dmo_from_ugc: bool = True,
) -> pd.DataFrame:
    """
    Construye el DataFrame unificado DMO + UGC.

    Las filas del CSV DMO se etiquetan con `dmo_label`; las del CSV UGC
    conservan su fuente original. Si el mismo archivo se pasa en ambos
    parámetros, las filas DMO se eliminan del lado UGC para no duplicarlas.
    """
    frames: List[pd.DataFrame] = []

    dmo_df = load_corpus(
        dmo_csv,
        text_column=text_column,
        source_column=source_column,
        corpus_label="DMO",
        dmo_label=dmo_label,
    )
    frames.append(dmo_df)

    for ugc_csv in ugc_csvs:
        ugc_df = load_corpus(
            ugc_csv,
            text_column=text_column,
            source_column=source_column,
            corpus_label="UGC",
            dmo_label=dmo_label,
        )
        if drop_dmo_from_ugc and source_column in ugc_df.columns:
            ugc_df = ugc_df[ugc_df[source_column].astype(str) != str(dmo_label)]
        frames.append(ugc_df)

    unified = pd.concat(frames, ignore_index=True)
    return unified.reset_index(drop=True)


def corpus_summary(df: pd.DataFrame, source_column: str = SOURCE_COLUMN) -> dict:
    """Resumen por corpus y por fuente (para el log del CLI)."""
    summary = {
        "corpus": df[CORPUS_COLUMN].value_counts().to_dict()
        if CORPUS_COLUMN in df.columns
        else {},
        "source": df[source_column].value_counts().to_dict()
        if source_column in df.columns
        else {},
    }
    return summary
