"""Visualización interactiva de n-gramas con muestras de comentarios en el hover.

Funciones puras: no imprimen, no escriben archivos y no mutan los DataFrames.
La tokenización es la misma de ``metrics_lexical`` para mantener consistencia
con log-odds y cobertura.
"""
from __future__ import annotations

import html
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from metrics_lexical import tokenize
from utils import palette_to_plot_colors

NGRAM_LABEL_ES: dict[int, str] = {1: "unigramas", 2: "bigramas", 3: "trigramas"}
DEFAULT_TOP_N: int = 15
DEFAULT_MAX_SAMPLES: int = 3
SAMPLE_CHARS: int = 120


def _ensure_text_norm(
    df: pd.DataFrame,
    text_column: str = "text",
    lang: str = "es",
) -> pd.DataFrame:
    """Devuelve copia con ``text_norm``; si falta, la genera con el pipeline local.

    Intenta la lematización del PASO 1; si los recursos no están disponibles,
    cae a ``clean_natural`` y finalmente a minúsculas. Nunca descarga nada.
    """
    out = df.copy()
    if "text_norm" in out.columns:
        return out
    if text_column not in out.columns:
        raise KeyError(f"No existe la columna de texto: {text_column}")

    normalizer = None
    try:
        from text_preprocessing import clean_natural, get_language_resources, lemmatize_text

        stop_words, nlp, _ = get_language_resources(lang)
        normalizer = lambda t: lemmatize_text(clean_natural(t), nlp, stop_words)
    except Exception:
        try:
            from text_preprocessing import clean_natural

            normalizer = clean_natural
        except Exception:
            normalizer = lambda t: str(t).lower().strip()

    out["text_norm"] = [normalizer(t) for t in out[text_column]]
    return out


def _sample_text(text: str) -> str:
    """Trunca a 120 caracteres y escapa HTML antes de inyectarlo en customdata."""
    return html.escape(str(text)[:SAMPLE_CHARS].strip())


def map_ngrams_to_comments(
    df: pd.DataFrame,
    n: int,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    text_column: str = "text_norm",
) -> dict[str, list[str]]:
    """Mapea cada n-grama a hasta ``max_samples`` comentarios normalizados.

    Un comentario aporta una sola muestra por n-grama aunque este aparezca
    varias veces en el mismo texto. El resultado es determinista (orden de filas).
    """
    if n not in NGRAM_LABEL_ES:
        raise ValueError(f"n debe ser 1, 2 o 3; se recibió {n}")
    if text_column not in df.columns:
        raise KeyError(f"No existe la columna de texto normalizado: {text_column}")

    mapping: dict[str, list[str]] = {}
    for value in df[text_column]:
        tokens = tokenize(value)
        seen_in_comment: set[str] = set()
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            if ngram in seen_in_comment:
                continue
            seen_in_comment.add(ngram)
            samples = mapping.setdefault(ngram, [])
            if len(samples) < max_samples:
                samples.append(_sample_text(value))
    return mapping


def _resolve_colors(palette: Any) -> tuple[str, str]:
    """Azul (normales) y naranja (outliers) coherentes con la paleta del dashboard."""
    resolved: Mapping[str, Any]
    if isinstance(palette, str):
        resolved = palette_to_plot_colors(palette)
    elif isinstance(palette, Mapping):
        resolved = palette
    else:
        resolved = palette_to_plot_colors("ejecutiva")
    primary = str(resolved.get("primary") or resolved.get("secondary") or "#2E6FB7")
    outlier = str(
        resolved.get("neutral")
        or resolved.get("accent")
        or resolved.get("tertiary")
        or "#E07B39"
    )
    return primary, outlier


def _trace(
    mapping: dict[str, list[str]],
    top_n: int,
    color: str,
    label: str,
    visible: bool,
    max_samples: int,
) -> go.Bar:
    """Construye la traza de barras con customdata (n_barras, max_samples)."""
    ranked = sorted(mapping.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top_n]
    ngrams = [gram for gram, _ in ranked]
    freqs = [len(samples) for _, samples in ranked]
    customdata = [
        [samples[i] if i < len(samples) else "" for i in range(max_samples)]
        for _, samples in ranked
    ]
    sample_lines = "<br>".join(
        f"{i + 1}. %{{customdata[{i}]}}" for i in range(max_samples)
    )
    hovertemplate = (
        "<b>%{x}</b><br>Frecuencia: %{y}<br>Muestras:<br>"
        + sample_lines + "<extra></extra>"
    )
    return go.Bar(
        x=ngrams,
        y=freqs,
        customdata=customdata,
        hovertemplate=hovertemplate,
        name=label,
        marker_color=color,
        visible=visible,
    )


def build_ngram_interactive_plot(
    df_normales: pd.DataFrame,
    df_outliers: pd.DataFrame,
    n: int = 1,
    top_n: int = DEFAULT_TOP_N,
    palette: Any = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> go.Figure:
    """Figura única con dos trazas y botones para alternar normales/outliers.

    Cada barra expone hasta tres comentarios de ``text_norm`` en ``customdata``
    y en el ``hovertemplate``, truncados a 120 caracteres y escapados para HTML.
    """
    if n not in NGRAM_LABEL_ES:
        raise ValueError(f"n debe ser 1, 2 o 3; se recibió {n}")

    normales = _ensure_text_norm(df_normales)
    outliers = _ensure_text_norm(df_outliers)
    map_norm = map_ngrams_to_comments(normales, n, max_samples)
    map_out = map_ngrams_to_comments(outliers, n, max_samples)
    color_norm, color_out = _resolve_colors(palette)
    label = NGRAM_LABEL_ES[n]

    title_norm = f"Top {top_n} {label} — Comentarios Normales"
    title_out = f"Top {top_n} {label} — Comentarios Outliers"

    fig = go.Figure(data=[
        _trace(map_norm, top_n, color_norm, "Normales", True, max_samples),
        _trace(map_out, top_n, color_out, "Outliers", False, max_samples),
    ])
    fig.update_layout(
        title=title_norm,
        barmode="group",
        template="plotly_white",
        xaxis_title=f"{label.capitalize()}",
        yaxis_title="Frecuencia (comentarios)",
        xaxis_tickangle=-45,
        height=520,
        margin=dict(b=160),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.0, xanchor="left", y=1.18, yanchor="top",
            showactive=True,
            buttons=[
                dict(
                    label="Comentarios Normales",
                    method="update",
                    args=[{"visible": [True, False]},
                          {"title": {"text": title_norm}}],
                ),
                dict(
                    label="Comentarios Outliers",
                    method="update",
                    args=[{"visible": [False, True]},
                          {"title": {"text": title_out}}],
                ),
            ],
        )],
    )
    return fig


def build_all_ngram_figures(
    df_normales: pd.DataFrame,
    df_outliers: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    palette: Any = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> dict[int, go.Figure]:
    """Devuelve un Figure por orden de n-grama (1, 2 y 3)."""
    return {
        n: build_ngram_interactive_plot(
            df_normales, df_outliers, n=n, top_n=top_n,
            palette=palette, max_samples=max_samples,
        )
        for n in (1, 2, 3)
    }
