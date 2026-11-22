"""
sentiment_report.py
─────────────────────────────────────────────────────────────
• Polaridad POS/NEU/NEG POR FUENTE → consola + gráficos + CSV
• Polaridad de OUTLIERS            → consola + gráficos + CSV

Métrica descriptiva: no se usa como prueba de la brecha DMO vs UGC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px

from comparison.common import save_csv
from sentiment_labels import LABEL_ES, SENTIMENT_LABELS


def _resolve_palette(palette: Any):
    """Devuelve (positivo, neutro, negativo, categóricos) desde dict o lista."""
    if isinstance(palette, dict):
        positive = palette.get("positive", "#009E73")
        neutral = palette.get("neutral", palette.get("tertiary", "#7F8C8D"))
        negative = palette.get("negative", "#D55E00")
        categorical = [
            color for color in palette.get("categorical", []) if isinstance(color, str)
        ] or [positive, neutral, negative]
        return positive, neutral, negative, categorical
    if isinstance(palette, (list, tuple)) and palette:
        colors = [str(color) for color in palette]
        return (colors[0],
                colors[1] if len(colors) > 1 else colors[0],
                colors[2] if len(colors) > 2 else colors[0],
                colors)
    return "#009E73", "#7F8C8D", "#D55E00", ["#009E73", "#7F8C8D", "#D55E00"]


def _counts(df: pd.DataFrame, source_column: str) -> pd.DataFrame:
    """Conteo POS/NEU/NEG por fuente (columnas garantizadas)."""
    tmp = df.copy()
    tmp["sentiment_label"] = tmp["sentiment_label"].astype(str)
    counts = (
        tmp.groupby([source_column, "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(SENTIMENT_LABELS), fill_value=0)
        .reset_index()
    )
    return counts


def _summary(counts: pd.DataFrame, source_column: str) -> pd.DataFrame:
    """Totales y porcentajes por fuente."""
    out = counts.rename(columns={
        "positive": "positivos", "neutral": "neutros", "negative": "negativos",
    }).copy()
    out["total"] = out[["positivos", "neutros", "negativos"]].sum(axis=1)
    for col, pct in (("positivos", "pct_pos"), ("neutros", "pct_neu"),
                     ("negativos", "pct_neg")):
        out[pct] = (100 * out[col] / out["total"].replace(0, 1)).round(2)
    return out


def _print_summary(summary: pd.DataFrame, source_column: str, total_label: str) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"   💬  {total_label}")
    print(sep)
    header = (f"  {'Fuente':<22s} {'Total':>7s} {'Pos':>7s} {'Neu':>7s} "
              f"{'Neg':>7s} {'%Pos':>7s} {'%Neu':>7s} {'%Neg':>7s}")
    print(header)
    print("  " + " ".join(["─" * 22] + ["─" * 7] * 7))
    for _, row in summary.iterrows():
        print(
            f"  {str(row[source_column]):<22s} "
            f"{int(row['total']):>7,d} {int(row['positivos']):>7,d} "
            f"{int(row['neutros']):>7,d} {int(row['negativos']):>7,d} "
            f"{row['pct_pos']:>6.1f}% {row['pct_neu']:>6.1f}% {row['pct_neg']:>6.1f}%"
        )
    for label, col in (("POS", "positivos"), ("NEU", "neutros"), ("NEG", "negativos")):
        value = int(summary[col].sum())
        print(f"  TOTAL {label}: {value:,}")
    print("  ℹ️  Descriptivo: no evidencia de la brecha DMO vs UGC.")
    print(f"{sep}\n")


def _color_map(palette: Any) -> dict[str, str]:
    pos, neu, neg, _ = _resolve_palette(palette)
    return {"positive": pos, "neutral": neu, "negative": neg}


def _spanish_labels(series: pd.Series) -> pd.Series:
    return series.map(lambda x: LABEL_ES.get(str(x), str(x)))


# ──────────────────────────────────────────────
# 1) Sentimiento por fuente
# ──────────────────────────────────────────────
def reporte_sentimiento_por_fuente(
    df: pd.DataFrame,
    source_column: str,
    output_dir: str = "results/sentiment",
    palette: Any = None,
    title: str = "Análisis de Sentimiento por Fuente",
) -> Dict[str, Any]:
    """Distribución ternaria por fuente; consola + HTML + tablas CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "sentiment_label" not in df.columns:
        raise KeyError("El DataFrame no tiene 'sentiment_label'. Ejecuta annotate_sentiment() primero.")

    counts = _counts(df, source_column)
    summary = _summary(counts, source_column)
    _print_summary(summary, source_column, title)

    save_csv(summary, out / "sentiment_by_source_summary.csv")
    save_csv(counts, out / "sentiment_by_source_counts.csv")

    _, _, _, categorical = _resolve_palette(palette)
    cmap = _color_map(palette)

    # Barras apiladas (%)
    pct = counts.copy()
    for label in SENTIMENT_LABELS:
        pct[label] = 100 * pct[label] / pct[list(SENTIMENT_LABELS)].sum(axis=1).replace(0, 1)
    long = pct.melt(id_vars=source_column, value_vars=list(SENTIMENT_LABELS),
                    var_name="sentiment_label", value_name="pct")
    long["sentiment_label"] = _spanish_labels(long["sentiment_label"])
    cmap_es = {LABEL_ES[k]: v for k, v in cmap.items()}

    fig1 = px.bar(
        long, x=source_column, y="pct", color="sentiment_label", barmode="stack",
        title=f"{title} – Distribución (%) POS/NEU/NEG",
        labels={"pct": "Porcentaje (%)", source_column: "Fuente",
                "sentiment_label": "Sentimiento"},
        color_discrete_map=cmap_es,
    )
    fig1.update_yaxes(range=[0, 100], ticksuffix="%")
    fig1.update_layout(template="plotly_white", height=450)
    fig1.write_html(str(out / "sentiment_by_source_stacked.html"), include_plotlyjs="cdn")
    save_csv(long, out / "sentiment_by_source_stacked_data.csv")

    # Histograma de scores
    if "sentiment_score" in df.columns:
        hist = df[[source_column, "sentiment_score"]].copy()
        fig2 = px.histogram(
            hist, x="sentiment_score", color=source_column, barmode="overlay",
            marginal="box", title=f"{title} – Distribución de scores",
            labels={"sentiment_score": "Score de sentimiento", source_column: "Fuente"},
            opacity=0.7, color_discrete_sequence=categorical,
        )
        fig2.update_layout(template="plotly_white", height=450)
        fig2.write_html(str(out / "sentiment_score_hist.html"), include_plotlyjs="cdn")
        save_csv(hist, out / "sentiment_score_hist_data.csv")

    print(f"  ✅ Gráficos y tablas guardados en: {out}/")
    return {"summary": summary, "counts": counts}


# ──────────────────────────────────────────────
# 2) Sentimiento de outliers
# ──────────────────────────────────────────────
def reporte_sentimiento_outliers(
    df_outliers: pd.DataFrame,
    source_column: str,
    output_dir: str = "results/sentiment",
    palette: Any = None,
    title: str = "Sentimiento de Comentarios Outlier",
) -> Dict[str, Any]:
    """Distribución ternaria de outliers; consola + HTML + tablas CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df_outliers.empty:
        print("⚠️  No hay outliers para analizar sentimiento.")
        return {}

    counts = _counts(df_outliers, source_column)
    summary = _summary(counts, source_column)
    _print_summary(summary, source_column, title)
    save_csv(summary, out / "outlier_sentiment_summary.csv")
    save_csv(counts, out / "outlier_sentiment_counts.csv")

    cmap = _color_map(palette)
    cmap_es = {LABEL_ES[k]: v for k, v in cmap.items()}

    pie = counts.melt(id_vars=source_column, value_vars=list(SENTIMENT_LABELS),
                      var_name="sentiment_label", value_name="count")
    pie["sentiment_label"] = _spanish_labels(pie["sentiment_label"])
    fig_pie = px.pie(
        pie, names="sentiment_label", values="count",
        title=f"{title} – Distribución global POS/NEU/NEG",
        color="sentiment_label", color_discrete_map=cmap_es,
    )
    fig_pie.update_layout(template="plotly_white", height=400)
    fig_pie.write_html(str(out / "outlier_sentiment_pie.html"), include_plotlyjs="cdn")
    save_csv(pie, out / "outlier_sentiment_pie_data.csv")

    bar = summary[[source_column, "positivos", "neutros", "negativos"]].copy()
    fig_bar = px.bar(
        bar, x=source_column, y=["positivos", "neutros", "negativos"],
        barmode="group", title=f"{title} – Por fuente",
        labels={"value": "Cantidad", source_column: "Fuente", "variable": "Sentimiento"},
        color_discrete_map={"positivos": cmap["positive"], "neutros": cmap["neutral"],
                            "negativos": cmap["negative"]},
    )
    fig_bar.update_layout(template="plotly_white", height=400)
    fig_bar.write_html(str(out / "outlier_sentiment_by_source.html"), include_plotlyjs="cdn")
    save_csv(bar, out / "outlier_sentiment_by_source_data.csv")

    print(f"  ✅ Gráficos y tablas guardados en: {out}/")
    return {"summary": summary, "counts": counts}
