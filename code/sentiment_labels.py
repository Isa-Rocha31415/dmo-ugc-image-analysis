"""Etiquetas ternarias de sentimiento derivadas del ``sentiment_score`` cacheado.

Funciones puras: no leen caché, no imprimen y no mutan el DataFrame de entrada.
La banda neutral evita reclasificar modelos ya ejecutados.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from io_corpus import SOURCE_COLUMN

NEUTRAL_TAU: float = 0.15
SENTIMENT_LABELS: tuple[str, str, str] = ("positive", "neutral", "negative")
LABEL_TO_ID: dict[str, int] = {"positive": 1, "neutral": 0, "negative": -1}
LABEL_ES: dict[str, str] = {
    "positive": "POSITIVO", "neutral": "NEUTRO", "negative": "NEGATIVO",
}


def label_from_score(score: float, neutral_tau: float = NEUTRAL_TAU) -> str:
    """Devuelve positive/neutral/negative aplicando la banda ``|score| < tau``.

    Un score faltante se considera neutral: no hay evidencia para polaridad.
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "neutral"
    if not np.isfinite(value):
        return "neutral"
    if value > neutral_tau:
        return "positive"
    if value < -neutral_tau:
        return "negative"
    return "neutral"


def apply_sentiment_labels(
    df: pd.DataFrame,
    neutral_tau: float = NEUTRAL_TAU,
    score_column: str = "sentiment_score",
    label_column: str = "sentiment_label",
) -> pd.DataFrame:
    """Reemplaza etiquetas binarias cacheadas por etiquetas ternarias.

    Añade ``sentiment_label_id`` (1/0/-1) para contratos que usan códigos.
    """
    out = df.copy()
    if score_column not in out.columns:
        raise KeyError(f"No existe la columna de score: {score_column}")
    labels = [label_from_score(v, neutral_tau) for v in out[score_column]]
    out[label_column] = labels
    out["sentiment_label_id"] = [LABEL_TO_ID[x] for x in labels]
    return out


def sentiment_distribution(
    df: pd.DataFrame,
    group_columns: Sequence[str] = (SOURCE_COLUMN,),
    label_column: str = "sentiment_label",
) -> pd.DataFrame:
    """Tabla descriptiva POS/NEU/NEG por fuente, corpus o ambas."""
    cols = list(group_columns)
    missing = [c for c in cols + [label_column] if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas ausentes: {missing}")

    rows = []
    for key, sub in df.groupby(cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        total = int(len(sub))
        counts = {
            label: int((sub[label_column] == label).sum())
            for label in SENTIMENT_LABELS
        }
        row = dict(zip(cols, key_tuple))
        row.update({
            "total": total,
            "positivos": counts["positive"],
            "neutros": counts["neutral"],
            "negativos": counts["negative"],
        })
        row["pct_positivos"] = round(100 * counts["positive"] / total, 2) if total else 0.0
        row["pct_neutros"] = round(100 * counts["neutral"] / total, 2) if total else 0.0
        row["pct_negativos"] = round(100 * counts["negative"] / total, 2) if total else 0.0
        rows.append(row)

    return pd.DataFrame(rows, columns=cols + [
        "total", "positivos", "neutros", "negativos",
        "pct_positivos", "pct_neutros", "pct_negativos",
    ])
