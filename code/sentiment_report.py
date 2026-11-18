"""
sentiment_report.py
─────────────────────────────────────────────────────────────
• Sentimiento positivo/negativo POR FUENTE  → consola + gráfico
• Sentimiento de OUTLIERS (todos)           → consola + gráfico
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _resolve_palette(palette: Any):
    """Devuelve (positivo, negativo, categóricos) desde dict o lista."""
    if isinstance(palette, dict):
        positive = palette.get("positive", "#009E73")
        negative = palette.get("negative", "#D55E00")
        categorical = [
            color for color in palette.get("categorical", []) if isinstance(color, str)
        ] or [positive, negative]
        return positive, negative, categorical
    if isinstance(palette, (list, tuple)) and palette:
        colors = [str(color) for color in palette]
        return colors[0], colors[1] if len(colors) > 1 else colors[0], colors
    return "#009E73", "#D55E00", ["#009E73", "#D55E00"]

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
    """
    Imprime en consola la distribución de sentimiento por fuente
    y genera dos gráficos HTML:
      • Barras apiladas (%)
      • Histograma de scores
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "sentiment_label" not in df.columns:
        raise KeyError("El DataFrame no tiene 'sentiment_label'. Ejecuta annotate_sentiment() primero.")

    tmp = df.copy()
    tmp["sentiment_label"] = tmp["sentiment_label"].astype(str)
    tmp["is_positive"] = (tmp["sentiment_label"] == "positive").astype(int)

    # ── Tabla resumen ──
    summary = (
        tmp.groupby(source_column)
        .agg(
            total=("sentiment_label", "size"),
            positivos=("is_positive", "sum"),
        )
        .reset_index()
    )
    summary["negativos"] = summary["total"] - summary["positivos"]
    summary["pct_pos"] = (summary["positivos"] / summary["total"] * 100).round(2)
    summary["pct_neg"] = (summary["negativos"] / summary["total"] * 100).round(2)
    if "sentiment_score" in tmp.columns:
        score_medio = tmp.groupby(source_column)["sentiment_score"].mean().round(4)
        summary["score_medio"] = summary[source_column].map(score_medio)

    # ── CONSOLA ──
    SEP = "═" * 70
    print(f"\n{SEP}")
    print(f"   💬  {title.upper()}")
    print(f"{SEP}")
    print(f"  {'Fuente':<22s} {'Total':>7s} {'Pos':>7s} {'Neg':>7s} {'%Pos':>7s} {'%Neg':>7s} {'Score':>8s}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    for _, row in summary.iterrows():
        score = row.get("score_medio", 0.0)
        print(
            f"  {row[source_column]:<22s} "
            f"{int(row['total']):>7,d} "
            f"{int(row['positivos']):>7,d} "
            f"{int(row['negativos']):>7,d} "
            f"{row['pct_pos']:>6.1f}% "
            f"{row['pct_neg']:>6.1f}% "
            f"{score:>8.4f}"
        )

    total_pos = int(summary["positivos"].sum())
    total_neg = int(summary["negativos"].sum())
    total_all = int(summary["total"].sum())
    print(f"\n  TOTAL GENERAL: {total_all:,} | Pos: {total_pos:,} ({total_pos/total_all:.1%}) | Neg: {total_neg:,} ({total_neg/total_all:.1%})")
    print(f"{SEP}\n")

    # ── GRÁFICO 1: barras apiladas ──
    palette_pos, palette_neg, palette_cat = _resolve_palette(palette)

    counts = (
        tmp.groupby([source_column, "sentiment_label"])
        .size()
        .reset_index(name="count")
    )
    counts["pct"] = counts.groupby(source_column)["count"].transform(lambda x: x / x.sum() * 100)

    fig1 = px.bar(
        counts,
        x=source_column,
        y="pct",
        color="sentiment_label",
        barmode="stack",
        title=f"{title} – Distribución (%)",
        labels={"pct": "Porcentaje (%)", source_column: "Fuente", "sentiment_label": "Sentimiento"},
        color_discrete_map={"positive": palette_pos, "negative": palette_neg},
    )
    fig1.update_yaxes(range=[0, 100], ticksuffix="%")
    fig1.update_layout(template="plotly_white", height=450)
    fig1.write_html(str(out / "sentiment_by_source_stacked.html"), include_plotlyjs="cdn")

    # ── GRÁFICO 2: histograma scores ──
    if "sentiment_score" in tmp.columns:
        fig2 = px.histogram(
            tmp,
            x="sentiment_score",
            color=source_column,
            barmode="overlay",
            marginal="box",
            title=f"{title} – Distribución de scores",
            labels={"sentiment_score": "Score de sentimiento", source_column: "Fuente"},
            opacity=0.7,
            color_discrete_sequence=palette_cat,
        )
        fig2.update_layout(template="plotly_white", height=450)
        fig2.write_html(str(out / "sentiment_score_hist.html"), include_plotlyjs="cdn")

    print(f"  ✅ Gráficos guardados en: {out}/")
    print(f"     • sentiment_by_source_stacked.html")
    if "sentiment_score" in tmp.columns:
        print(f"     • sentiment_score_hist.html")

    return {"summary": summary, "counts": counts}


# ──────────────────────────────────────────────
# 2) Sentimiento de OUTLIERS
# ──────────────────────────────────────────────
def reporte_sentimiento_outliers(
    df_outliers: pd.DataFrame,
    source_column: str,
    output_dir: str = "results/sentiment",
    palette: Any = None,
    title: str = "Sentimiento de Comentarios Outlier",
) -> Dict[str, Any]:
    """
    Analiza el sentimiento SOLO de los outliers.
    Imprime consola + genera gráficos.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df_outliers.empty:
        print("⚠️  No hay outliers para analizar sentimiento.")
        return {}

    tmp = df_outliers.copy()
    tmp["sentiment_label"] = tmp["sentiment_label"].astype(str)
    tmp["is_positive"] = (tmp["sentiment_label"] == "positive").astype(int)

    # ── Resumen por fuente ──
    summary = (
        tmp.groupby(source_column)
        .agg(total=("sentiment_label", "size"), positivos=("is_positive", "sum"))
        .reset_index()
    )
    summary["negativos"] = summary["total"] - summary["positivos"]
    summary["pct_pos"] = (summary["positivos"] / summary["total"] * 100).round(2)
    summary["pct_neg"] = (summary["negativos"] / summary["total"] * 100).round(2)

    # ── CONSOLA ──
    SEP = "═" * 70
    print(f"\n{SEP}")
    print(f"   🔍  {title.upper()}")
    print(f"{SEP}")
    print(f"  Total outliers con sentimiento: {len(tmp):,}")
    print(f"\n  {'Fuente':<22s} {'Total':>7s} {'Pos':>7s} {'Neg':>7s} {'%Pos':>7s} {'%Neg':>7s}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for _, row in summary.iterrows():
        print(
            f"  {row[source_column]:<22s} "
            f"{int(row['total']):>7,d} "
            f"{int(row['positivos']):>7,d} "
            f"{int(row['negativos']):>7,d} "
            f"{row['pct_pos']:>6.1f}% "
            f"{row['pct_neg']:>6.1f}%"
        )

    tot = len(tmp)
    pos = int(tmp["is_positive"].sum())
    neg = tot - pos
    print(f"\n  TOTAL OUTLIERS: {tot:,} | Pos: {pos:,} ({pos/tot:.1%}) | Neg: {neg:,} ({neg/tot:.1%})")

    # Comparación con no-outliers si se pasa
    print(f"{SEP}\n")

    # ── GRÁFICO: pie chart ──
    palette_pos, palette_neg, palette_cat = _resolve_palette(palette)

    fig_pie = px.pie(
        tmp,
        names="sentiment_label",
        title=f"{title} – Distribución global",
        color_discrete_map={"positive": palette_pos, "negative": palette_neg},
    )
    fig_pie.update_layout(template="plotly_white", height=400)
    fig_pie.write_html(str(out / "outlier_sentiment_pie.html"), include_plotlyjs="cdn")

    # ── GRÁFICO: barras por fuente ──
    fig_bar = px.bar(
        summary,
        x=source_column,
        y=["positivos", "negativos"],
        barmode="group",
        title=f"{title} – Por fuente",
        labels={"value": "Cantidad", source_column: "Fuente", "variable": "Sentimiento"},
        color_discrete_map={"positivos": palette_pos, "negativos": palette_neg},
    )
    fig_bar.update_layout(template="plotly_white", height=400)
    fig_bar.write_html(str(out / "outlier_sentiment_by_source.html"), include_plotlyjs="cdn")

    print(f"  ✅ Gráficos guardados en: {out}/")
    print(f"     • outlier_sentiment_pie.html")
    print(f"     • outlier_sentiment_by_source.html\n")

    return {"summary": summary}