"""Secciones auxiliares del informe HTML (helpers puros, sin efectos secundarios).

"""
from __future__ import annotations

import html
import re
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

JACCARD_METRICS: tuple[str, ...] = ("jaccard_freq", "jaccard_tfidf")
HEADER_ES: dict[str, str] = {
    "positivos": "POSITIVO", "neutros": "NEUTRO", "negativos": "NEGATIVO",
    "pct_positivos": "% POSITIVO", "pct_neutros": "% NEUTRO",
    "pct_negativos": "% NEGATIVO", "total": "TOTAL",
}


def jaccard_console_lines(metrics: pd.DataFrame, scope: str = "ugc_total") -> list[str]:
    """Devuelve las líneas de consola de Jaccard@100."""
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
    *,
    titles: Mapping[str, str] | None = None,
    formats: Mapping[str, Callable[[Any, Mapping[str, Any]], str]] | None = None,
    safe_cols: tuple[str, ...] = (),
) -> str:
    """Tabla HTML genérica; `formats` emite HTML seguro y `safe_cols` ya escapado."""
    if table is None or getattr(table, "empty", True):
        return '<div class="placeholder">Sin datos disponibles</div>'
    view = table if max_rows is None else table.head(max_rows)
    mapping, titles, formats = header_map or {}, titles or {}, formats or {}
    safe = set(safe_cols)
    head = "".join(
        f'<th title="{html.escape(str(titles.get(str(c), mapping.get(str(c), str(c)))))}">'
        f'{html.escape(str(mapping.get(str(c), str(c))))}</th>'
        for c in view.columns
    )
    rows: list[str] = []
    for values in view.itertuples(index=False, name=None):
        row = dict(zip(view.columns, values))
        cells: list[str] = []
        for col in view.columns:
            formatter = formats.get(str(col))
            if formatter is not None:
                cells.append(f"<td>{formatter(row[col], row)}</td>")
            else:
                text = "" if pd.isna(row[col]) else str(row[col])
                cells.append(f"<td>{text if str(col) in safe else html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="table table--zebra"><thead><tr>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def stat_tests_html(
    gaps: pd.DataFrame | None,
    price_valence: pd.DataFrame | None,
) -> str:
    """Las pruebas temáticas y de precio/valencia ya no se renderizan en HTML."""
    return ""


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

    fig = next((f for f in figures.values() if f is not None), None)
    if fig is None:
        return '<div class="placeholder">Visualización de n-gramas no disponible</div>'
    return "\n".join([
        '<section class="row" id="ngramas">',
        '<div class="panel">',
        '<h2 class="panel__title">Análisis de N-gramas</h2>',
        '<div class="figure-box">',
        pio.to_html(fig, full_html=False, include_plotlyjs="cdn"),
        "</div></div></section>",
    ])


# ── Componentes de la capa de presentación del informe final ──

def _pct(value: Any) -> str:
    """Tasa en porcentaje con dos decimales."""
    return "—" if pd.isna(value) else f"{float(value) * 100:.2f}%"


def _num(value: Any, decimals: int) -> str:
    """Número con decimales fijos o guion si falta."""
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def source_badge(value: Any) -> str:
    """Insignia con texto escapado; el valor del CSV puede traer markup literal."""
    raw = html.unescape(str(value))
    label = re.sub(r"<[^>]*>", "", raw).strip() or "otro"
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "otro"
    return f'<span class="src-badge src-{slug}">{html.escape(label)}</span>'


def sentiment_chip(value: Any) -> str:
    """Chip POS/NEU/NEG derivado de la etiqueta, con color de la paleta."""
    key = str(value).lower()
    tag = ("pos" if key.startswith("pos") else
           "neg" if key.startswith("neg") else "neu")
    return f'<span class="sent-chip sent-chip--{tag}">{tag.upper()}</span>'


def gaps_sidebar_html(gaps: pd.DataFrame | None) -> str:
    """Tabla compacta de brechas: solo tópico, % DMO, % UGC y GAP."""
    if gaps is None or getattr(gaps, "empty", True):
        return '<div class="placeholder">Sin datos de brechas temáticas</div>'

    def pair(row: Mapping[str, Any]) -> tuple[float, float]:
        dmo, ugc = float(row["p_dmo"]), float(row["p_ugc"])
        if max(dmo, ugc) <= 1.5:
            dmo, ugc = dmo * 100, ugc * 100
        return dmo, ugc

    def pct_cell(key: str) -> Callable[[Any, Mapping[str, Any]], str]:
        def render(_: Any, row: Mapping[str, Any]) -> str:
            dmo, ugc = pair(row)
            value = dmo if key == "p_dmo" else ugc
            wins = dmo >= ugc if key == "p_dmo" else ugc > dmo
            cls = ' class="bold-cell"' if wins else ""
            return f"<span{cls}>{value:.2f}%</span>"
        return render

    def gap_cell(_: Any, row: Mapping[str, Any]) -> str:
        diff = pair(row)[1] - pair(row)[0]
        cls = "gap-pos" if diff > 0 else "gap-neg" if diff < 0 else ""
        return f'<span class="{cls}">{diff:+.2f}</span>'

    view = gaps.copy()
    view["_order"] = [abs(pair(r)[1] - pair(r)[0]) for _, r in view.iterrows()]
    view = view.nlargest(10, "_order")[["topic_name", "p_dmo", "p_ugc", "gap_pp"]]
    return '<div class="table-wrap">' + generic_table_html(
        view,
        {"topic_name": "Tópico", "p_dmo": "% DMO", "p_ugc": "% UGC", "gap_pp": "GAP"},
        formats={
            "topic_name": lambda v, _: html.escape(str(v)),
            "p_dmo": pct_cell("p_dmo"),
            "p_ugc": pct_cell("p_ugc"),
            "gap_pp": gap_cell,
        },
    ) + "</div>"


_FREQ_COLS: tuple[str, ...] = (
    "scope", "n_mentions_dmo", "n_docs_dmo", "rate_dmo", "ci_dmo",
    "n_mentions_ugc", "n_docs_ugc", "rate_ugc", "ci_ugc",
    "gap_pp", "cliffs_delta", "p_adj_bh",
)
_FREQ_HEAD: dict[str, str] = {
    "scope": "Alcance", "n_mentions_dmo": "Menciones DMO", "n_docs_dmo": "n DMO",
    "rate_dmo": "Tasa DMO", "ci_dmo": "IC 95% DMO",
    "n_mentions_ugc": "Menciones UGC", "n_docs_ugc": "n UGC",
    "rate_ugc": "Tasa UGC", "ci_ugc": "IC 95% UGC",
    "gap_pp": "GAP (pp)", "cliffs_delta": "Cliff's δ", "p_adj_bh": "p ajustada BH",
}
_FREQ_TITLES: dict[str, str] = {
    "scope": "Comparación analizada",
    "n_mentions_dmo": "Documentos DMO que mencionan el concepto",
    "n_docs_dmo": "Documentos DMO en el alcance",
    "rate_dmo": "Proporción de menciones en DMO",
    "ci_dmo": "Intervalo de confianza del 95% para la tasa DMO",
    "n_mentions_ugc": "Documentos UGC que mencionan el concepto",
    "n_docs_ugc": "Documentos UGC en el alcance",
    "rate_ugc": "Proporción de menciones en UGC",
    "ci_ugc": "Intervalo de confianza del 95% para la tasa UGC",
    "gap_pp": "Diferencia de tasas en puntos porcentuales (UGC − DMO)",
    "cliffs_delta": "Tamaño del efecto de Cliff, de −1 a 1",
    "p_adj_bh": "Valor p ajustado por Benjamini–Hochberg",
}


def price_section_html(
    comparison: pd.DataFrame | None,
    dist_tests: pd.DataFrame | None,
    crosstab: pd.DataFrame | None,
    examples: pd.DataFrame | None,
    alignment: pd.DataFrame | None,
    *,
    density_html: str = "",
    scatter_html: str = "",
    umap_html: str = "",
) -> str:
    """Sección de precio renderizada solo a partir de CSV existentes."""
    parts: list[str] = []

    def wrap(identifier: str, title: str, body: str) -> None:
        parts.append(
            f'<section id="{identifier}"><div class="panel"><h3 class="panel__title">{title}</h3>'
            f"{body}</div></section>"
        )

    def figure(fragment: str, label: str) -> str:
        return (f'<div class="figure-box">{fragment}</div>' if fragment
                else f'<div class="placeholder">{label} no disponible</div>')

    def kpi(label: str, value: str, delta: str) -> str:
        return (
            f'<article class="kpi kpi--price"><div class="kpi__label">{label}</div>'
            f'<div class="kpi__value">{value}<span class="delta-chip">{delta}</span></div>'
            '</article>'
        )

    total = comparison[comparison["scope"] == "DMO_vs_UGC_total"] if comparison is not None else None
    if total is None or total.empty:
        total = comparison.head(1) if comparison is not None else None

    def _total_value(column: str, default: float = 0.0) -> float:
        if total is None or total.empty or column not in total:
            return default
        return float(total.iloc[0][column])

    r_dmo = _total_value("rate_dmo") * 100
    r_ugc = _total_value("rate_ugc") * 100
    gap_pp = _total_value("gap_pp", r_ugc - r_dmo)
    n_mentions = int(_total_value("n_dmo_mentions")
                     + _total_value("n_ugc_mentions"))
    delta_cos = np.nan
    if alignment is not None and not alignment.empty and "gap_cosine" in alignment:
        values = alignment["gap_cosine"].dropna()
        delta_cos = float(values.iloc[0]) if not values.empty else np.nan
    if pd.isna(delta_cos) and alignment is not None and "scope" in alignment:
        means = alignment.set_index("scope")["mean_cosine"]
        if "DMO" in means.index and "UGC" in means.index:
            delta_cos = float(means["DMO"]) - float(means["UGC"])
    card1 = kpi(
        "Presencia", f"{r_dmo:.2f}% vs {r_ugc:.2f}%", f"{gap_pp:+.2f} pp",
    )
    card2 = kpi(
        "Comentarios con este concepto", f"{n_mentions:,}", f"{r_ugc:.2f}% UGC",
    )
    arrow = "↓" if delta_cos < 0 else "↑"
    card3 = kpi(
        "Proximidad semántica", f"{delta_cos:+.4f}" if pd.notna(delta_cos) else "—",
        f"{arrow} DMO − UGC",
    )
    wrap("precio-lectura", "Lectura rápida", f'<div class="kpi-grid">{card1}{card2}{card3}</div>')

    freq = comparison.copy() if comparison is not None else None
    if freq is not None and not freq.empty:
        for col in ("ci_low_dmo", "ci_high_dmo", "ci_low_ugc", "ci_high_ugc"):
            if col not in freq.columns:
                freq[col] = np.nan
        freq["ci_dmo"] = [
            f"{lo:.3f}–{hi:.3f}" if pd.notna(lo) and pd.notna(hi) else "—"
            for lo, hi in zip(freq["ci_low_dmo"], freq["ci_high_dmo"])
        ]
        freq["ci_ugc"] = [
            f"{lo:.3f}–{hi:.3f}" if pd.notna(lo) and pd.notna(hi) else "—"
            for lo, hi in zip(freq["ci_low_ugc"], freq["ci_high_ugc"])
        ]

        def signed(_: Any, row: Mapping[str, Any]) -> str:
            value = float(row["gap_pp"])
            cls = "gap-pos" if value > 0 else "gap-neg" if value < 0 else ""
            return f'<span class="{cls}">{value:+.2f}</span>'

        fmts: dict[str, Callable[[Any, Mapping[str, Any]], str]] = {
            "scope": lambda v, _: html.escape(str(v)),
            "rate_dmo": lambda v, _: _pct(v),
            "rate_ugc": lambda v, _: _pct(v),
            "gap_pp": signed,
            "cliffs_delta": lambda v, _: _num(v, 3),
            "p_adj_bh": lambda v, _: "—" if pd.isna(v) else f"{float(v):.2g}",
        }
        freq_html = '<div class="table-wrap">' + generic_table_html(
            freq[list(_FREQ_COLS)], _FREQ_HEAD, titles=_FREQ_TITLES, formats=fmts,
        ) + "</div>"
    else:
        freq_html = '<div class="placeholder">Sin datos de frecuencia</div>'
    wrap("precio-frecuencia", "a) Frecuencia y tamaños del efecto", freq_html)

    dist_cols = ("scope", "emd_wasserstein", "jsd_base2", "n_a", "n_b")
    dist = dist_tests[list(dist_cols)] if dist_tests is not None and not dist_tests.empty else None
    dist_html = '<div class="table-wrap">' + generic_table_html(
        dist,
        {"scope": "Alcance", "emd_wasserstein": "EMD (Wasserstein)",
         "jsd_base2": "JSD (base 2)", "n_a": "n A", "n_b": "n B"},
        titles={"emd_wasserstein": "Distancia de Wasserstein sobre el coseno",
                "jsd_base2": "Divergencia de Jensen–Shannon en base 2",
                "n_a": "Tamaño del grupo A", "n_b": "Tamaño del grupo B"},
        formats={
            "scope": lambda v, _: html.escape(str(v)),
            "emd_wasserstein": lambda v, _: _num(v, 4),
            "jsd_base2": lambda v, _: _num(v, 4),
        },
    ) + "</div>" if dist is not None else '<div class="placeholder">Sin pruebas de distribución</div>'
    wrap("precio-distribucion", "Distribuciones", dist_html)

    wrap("precio-scatter", "c) Similitud semántica", figure(scatter_html, "Scatter de similitud"))
    wrap("precio-umap", "Proximidad semántica (UMAP 3D)", figure(umap_html, "UMAP 3D"))

    ct_cols = ("topic_id", "topic_name", "source", "n", "denominator", "pct_within_source")
    ct = crosstab.copy() if crosstab is not None and not crosstab.empty else None
    if ct is not None:
        ct = ct[list(ct_cols)].sort_values(["source", "pct_within_source"], ascending=[True, False])
    ct_html = '<div class="table-wrap">' + generic_table_html(
        ct,
        {"topic_id": "Tópico", "topic_name": "Nombre", "source": "Fuente", "n": "n",
         "denominator": "Denominador", "pct_within_source": "% dentro de fuente"},
        formats={
            "source": lambda v, _: source_badge(v),
            "topic_name": lambda v, _: html.escape(str(v)),
            "pct_within_source": lambda v, _: _pct(v),
        },
    ) + "</div>" if ct is not None else '<div class="placeholder">Sin crosstab</div>'
    wrap("precio-contexto", "e) Contexto temático por fuente", ct_html)

    ex_cols = ("rank", "scope", "doc_id", "source", "corpus", "cos_semantic",
               "sentiment_label", "topic_name", "registro_dominante", "text_norm")
    ex = None
    if examples is not None and not examples.empty:
        ex = examples[examples["scope"].isin(
            {"dmo", "tripadvisor", "instagram", "reddit"}
        ) & (pd.to_numeric(examples["rank"], errors="coerce") <= 3)]
        ex = ex[list(ex_cols)]

    def comment(value: Any, _: Mapping[str, Any]) -> str:
        txt = html.escape(str(value))
        return f'<details class="cell-details"><summary title="{txt}">{txt}</summary></details>'

    ex_html = '<div class="table-wrap">' + generic_table_html(
        ex,
        {"rank": "Rank", "scope": "Alcance", "doc_id": "ID", "source": "Fuente",
         "corpus": "Corpus", "cos_semantic": "Coseno", "sentiment_label": "Polaridad",
         "topic_name": "Tópico", "registro_dominante": "Registro", "text_norm": "Comentario"},
        formats={
            "source": lambda v, _: source_badge(v),
            "sentiment_label": lambda v, _: sentiment_chip(v),
            "cos_semantic": lambda v, _: _num(v, 3),
            "topic_name": lambda v, _: html.escape(str(v)),
            "text_norm": comment,
        },
    ) + "</div>" if ex is not None and not ex.empty else '<div class="placeholder">Sin ejemplos</div>'
    wrap("precio-ejemplos", "Ejemplos de comentarios con Precio, valor y costo", ex_html)
    return "\n".join(parts)


def density_figure_html(density: pd.DataFrame | None) -> str:
    """Figura de densidad normalizada del coseno a partir de `price_density.csv`."""
    if density is None or getattr(density, "empty", True):
        return ""
    fig = go.Figure()
    for source, group in density.groupby("source", sort=True):
        fig.add_trace(go.Scatter(
            x=group["bin_center"], y=group["density"], mode="lines",
            name=str(source), fill="tozeroy",
            hovertemplate="<b>%{fullData.name}</b><br>coseno %{x:.3f}<br>densidad %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        title="Densidad normalizada del coseno por fuente",
        template="plotly_white", height=420, hovermode="x unified",
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def inline_plotly_fragment(full_html: str) -> str:
    """Extrae el div y el script de un HTML de Plotly para incrustarlo en el informe."""
    match = re.search(
        r'(<div\b[^>]*plotly-graph-div[^>]*>[\s\S]*?</div>)'
        r'([\s\S]*?<script>[\s\S]*?Plotly\.newPlot[\s\S]*?</script>)',
        full_html,
    )
    return match.group(0) if match else ""


def strip_plotly_updatemenus(fragment: str) -> str:
    """Quita `updatemenus` del JSON sin dejar comas colgantes en JavaScript."""
    pattern = re.compile(r'(?:"updatemenus"|updatemenus)\s*:\s*\[')
    out: list[str] = []
    cursor = 0
    for match in pattern.finditer(fragment):
        i, depth, in_str, escaped = match.end() - 1, 0, False, False
        while i < len(fragment):
            char = fragment[i]
            if in_str:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_str = False
            elif char == '"':
                in_str = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1

        start = match.start()
        while start > cursor and fragment[start - 1].isspace():
            start -= 1
        if start > cursor and fragment[start - 1] == ",":
            start -= 1
        else:
            while i < len(fragment) and fragment[i].isspace():
                i += 1
            if i < len(fragment) and fragment[i] == ",":
                i += 1
        out.append(fragment[cursor:start])
        cursor = i
    out.append(fragment[cursor:])
    return "".join(out)
