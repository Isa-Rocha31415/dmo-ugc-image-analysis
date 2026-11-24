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
    """Figura única con 6 trazas y dos menús: orden del n-grama y grupo.

    Cada barra expone hasta tres comentarios de ``text_norm`` en ``customdata``
    y en el ``hovertemplate``, truncados a 120 caracteres y escapados para HTML.
    ``n`` mantiene la firma legacy y determina el estado inicial del menú.
    """
    if n not in NGRAM_LABEL_ES:
        raise ValueError(f"n debe ser 1, 2 o 3; se recibió {n}")

    normales = _ensure_text_norm(df_normales)
    outliers = _ensure_text_norm(df_outliers)
    color_norm, color_out = _resolve_colors(palette)
    traces: list[go.Bar] = []
    titles: dict[tuple[int, int], str] = {}
    for order in (1, 2, 3):
        label = NGRAM_LABEL_ES[order]
        maps = (
            map_ngrams_to_comments(normales, order, max_samples),
            map_ngrams_to_comments(outliers, order, max_samples),
        )
        for group, (mapping, color, name) in enumerate(zip(
            maps, (color_norm, color_out), ("Normales", "Outliers")
        )):
            who = "normales" if group == 0 else "outliers"
            titles[order, group] = f"Top {top_n} {label} — Comentarios {who}"
            traces.append(_trace(
                mapping, top_n, color, name,
                visible=(order == n and group == 0), max_samples=max_samples,
            ))

    def _mask(order: int, group: int) -> list[bool]:
        return [i == 2 * (order - 1) + group for i in range(6)]

    def _button(label: str, order: int, group: int) -> dict[str, Any]:
        return dict(
            label=label, method="update",
            args=[{"visible": _mask(order, group)},
                  {"title": {"text": titles[order, group]}}],
        )

    top_buttons = [
        _button(name, order, 0)
        for order, name in ((1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas"))
    ]
    group_buttons = [
        _button("Comentarios normales", n, 0),
        _button("Comentarios outliers", n, 1),
    ]
    fig = go.Figure(data=traces)
    menu_style = dict(bgcolor=color_norm, bordercolor=color_norm,
                      font=dict(size=12, color="#ffffff"))
    fig.update_layout(
        title=titles[n, 0],
        barmode="group",
        template="plotly_white",
        xaxis_title="Frecuencia",
        yaxis_title="Frecuencia (comentarios)",
        xaxis_tickangle=-45,
        height=520,
        margin=dict(b=160),
        updatemenus=[
            dict(
                type="buttons", direction="right", showactive=True,
                x=0.0, xanchor="left", y=1.18, yanchor="top",
                **menu_style, buttons=top_buttons,
            ),
            dict(
                type="buttons", direction="right", showactive=True,
                x=0.0, xanchor="left", y=1.09, yanchor="top",
                **menu_style, buttons=group_buttons,
            ),
        ],
    )
    return fig


def build_all_ngram_figures(
    df_normales: pd.DataFrame,
    df_outliers: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    palette: Any = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> dict[int, go.Figure]:
    """Devuelve un único Figure con las dos pestañas (clave legacy 1)."""
    return {
        1: build_ngram_interactive_plot(
            df_normales, df_outliers, n=1, top_n=top_n,
            palette=palette, max_samples=max_samples,
        )
    }
