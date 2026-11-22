"""Secciones auxiliares del informe HTML (helpers puros, sin efectos secundarios).

Jaccard queda como validación de consola: estas funciones solo formatean datos.
"""
from __future__ import annotations

import html
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.io as pio

JACCARD_METRICS: tuple[str, ...] = ("jaccard_freq", "jaccard_tfidf")
HEADER_ES: dict[str, str] = {
    "positivos": "POSITIVO", "neutros": "NEUTRO", "negativos": "NEGATIVO",
    "pct_positivos": "% POSITIVO", "pct_neutros": "% NEUTRO",
    "pct_negativos": "% NEGATIVO", "total": "TOTAL",
}


def jaccard_console_lines(metrics: pd.DataFrame, scope: str = "ugc_total") -> list[str]:
    """Devuelve las líneas de consola de Jaccard@100 (nunca van al HTML)."""
    if metrics is None or metrics.empty:
        return []
    needed = {"metric", "scope", "k", "value"}
    if not needed.issubset(metrics.columns):
        return []
    sel = metrics[
        (metrics["scope"] == scope)
        & (metrics["metric"].isin(JACCARD_METRICS))
        & (metrics["k"] == 100)
    ]
    lines: list[str] = []
    for _, row in sel.iterrows():
        value = float(row["value"])
        if "ci_low" in metrics.columns and "ci_high" in metrics.columns:
            lo, hi = row["ci_low"], row["ci_high"]
            ci = ("" if pd.isna(lo) or pd.isna(hi)
                  else f"  IC95% [{float(lo):.3f}, {float(hi):.3f}]")
        else:
            ci = ""
        lines.append(f"  [AUX] {row['metric']}@100 = {value:.3f}{ci}")
    return lines


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    return html.escape(str(value))


def sentiment_table_html(table: pd.DataFrame) -> str:
    """Tabla HTML de la distribución descriptiva de polaridad."""
    if table is None or table.empty:
        return '<div class="placeholder">Sin datos de sentimiento</div>'
    head = "".join(
        f"<th>{html.escape(HEADER_ES.get(str(c), str(c)))}</th>" for c in table.columns
    )
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    return (
        '<div class="table-wrap"><table class="table table--zebra">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


_NGRAM_TITLES: dict[int, str] = {1: "Unigramas", 2: "Bigramas", 3: "Trigramas"}


def generic_table_html(
    table: pd.DataFrame | None,
    header_map: Mapping[str, str] | None = None,
    max_rows: int | None = None,
) -> str:
    """Tabla HTML genérica, con cabeceras opcionales en español."""
    if table is None or getattr(table, "empty", True):
        return '<div class="placeholder">Sin datos disponibles</div>'
    view = table if max_rows is None else table.head(max_rows)
    mapping = header_map or {}
    head = "".join(
        f"<th>{html.escape(str(mapping.get(str(c), str(c))))}</th>" for c in view.columns
    )
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
        for row in view.itertuples(index=False, name=None)
    )
    return (
        '<table class="table table--zebra"><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def stat_tests_html(
    gaps: pd.DataFrame | None,
    price_valence: pd.DataFrame | None,
) -> str:
    """Tablas de pruebas de brecha temática y de precio/valencia."""
    parts: list[str] = []
    gap_map = {
        "topic_name": "Tópico", "p_dmo": "% DMO", "p_ugc": "% UGC",
        "gap_pp": "GAP (pp)", "verdict": "Veredicto", "p_value": "p-valor",
        "p_adj": "p ajustado", "scope": "Alcance",
    }
    price_map = {
        "scope": "Alcance", "sentiment_label": "Sentimiento", "n": "n",
        "rate": "Tasa", "odds_ratio": "OR", "p_value": "p-valor",
        "ci_low": "IC bajo", "ci_high": "IC alto",
    }
    if gaps is not None and not gaps.empty:
        parts.append("<h3>Pruebas temáticas (gaps DMO vs UGC)</h3>")
        parts.append(generic_table_html(gaps, gap_map, max_rows=15))
    if price_valence is not None and not price_valence.empty:
        parts.append("<h3>Precio × valencia (condicionado)</h3>")
        parts.append(generic_table_html(price_valence, price_map, max_rows=15))
    return "".join(parts) or '<div class="placeholder">Sin pruebas estadísticas</div>'


def run_metadata_html(
    metadata: pd.DataFrame | None,
    flags: dict[str, object],
    sample_sizes: dict[str, object],
) -> str:
    """Tabla de metadata del run: versión, flags, timestamp y tamaños."""
    rows: list[dict[str, object]] = []
    if metadata is not None and not metadata.empty:
        for _, row in metadata.iterrows():
            rows.append({
                "timestamp": row.get("timestamp", ""),
                "task": row.get("task", ""),
                "versiones": ", ".join(
                    f"{k}={v}" for k, v in row.items()
                    if str(k).startswith("version_") and pd.notna(v)
                ),
                "detalle": str(row.get("params", ""))[:140],
            })
    for name, value in flags.items():
        rows.append({"timestamp": "", "task": f"flag:{name}", "versiones": "", "detalle": str(value)})
    for name, value in sample_sizes.items():
        rows.append({"timestamp": "", "task": f"n:{name}", "versiones": "", "detalle": str(value)})
    table = pd.DataFrame(rows, columns=["timestamp", "task", "versiones", "detalle"])
    return generic_table_html(table, {
        "timestamp": "Timestamp", "task": "Tarea / parámetro",
        "versiones": "Versiones", "detalle": "Detalle",
    })


def ngram_section_html(figures: Mapping[int, Any] | None) -> str:
    """Bloque HTML con los tres Figures interactivos de n-gramas embebidos."""
    if not figures:
        return '<div class="placeholder">Visualización de n-gramas no disponible</div>'

    parts = [
        '<section class="row" id="ngramas">',
        '<div class="panel">',
        '<h2 class="panel__title">Análisis de N-gramas</h2>',
        '<p class="section-note">Pasa el cursor sobre cada barra para ver hasta tres '
        'comentarios normalizados donde aparece el n-grama. Usa los botones para '
        'alternar entre comentarios normales y outliers.</p>',
    ]
    for n in (1, 2, 3):
        fig = figures.get(n)
        if fig is None:
            continue
        parts.append(
            f'<h3 class="subtitle">{_NGRAM_TITLES[n]}</h3>'
            '<div class="figure-box">'
            + pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            + "</div>"
        )
    parts.extend(["</div>", "</section>"])
    return "\n".join(parts)
