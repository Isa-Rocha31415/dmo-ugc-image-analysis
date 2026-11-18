"""
outlier_visualizations.py
─────────────────────────────────────────────────────────────
Visualizaciones complementarias para el análisis de outliers:
  • Wordclouds de unigramas, bigramas y trigramas
  • Gráficas de barras horizontales (top N) de n-gramas
  • Comparativa de outliers POR FUENTE (barras + pastel + stacked)
  • Heatmap de densidad de outliers por fuente
Todo se guarda en results/outliers/visualizations/
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ← modo no interactivo (servidor/CI)
import matplotlib.pyplot as plt
from matplotlib import colormaps
from wordcloud import WordCloud

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ══════════════════════════════════════════════
#  PALETA DE COLORES (accesible, apta para daltonismo)
# ══════════════════════════════════════════════
PALETTE = {
    "primary":    "#0072B2",   # azul
    "secondary":  "#D55E00",   # naranja
    "tertiary":   "#009E73",   # verde
    "quaternary": "#CC79A7",   # rosa
    "quinary":    "#56B4E9",   # celeste
    "neutral":    "#999999",
    "background": "#FAFAFA",
}

PALETTE_FUENTES = px.colors.qualitative.Safe   # ← paleta apta para daltonismo


def _merge_palette(palette: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Combina la paleta por defecto con la elegida por el usuario."""
    merged = dict(PALETTE)
    if palette:
        for key in ("primary", "secondary", "tertiary", "quaternary", "background"):
            if palette.get(key):
                merged[key] = palette[key]
    return merged


# ══════════════════════════════════════════════
#  1. WORDCLOUDS
# ══════════════════════════════════════════════
def generar_wordcloud_ngramas(
    ngramas: List[Tuple[str, int]],
    titulo: str,
    output_path: Path,
    colormap: str = "viridis",
    max_words: int = 150,
    width: int = 1200,
    height: int = 600,
) -> Path:
    """
    Genera una wordcloud a partir de una lista [(ngrama, conteo), ...].
    Devuelve la ruta del archivo PNG generado.
    """
    if not ngramas:
        print(f"  ⚠️  No hay n-gramas para wordcloud: {titulo}")
        return output_path

    freq = {w: c for w, c in ngramas}

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap=colormap,
        max_words=max_words,
        max_font_size=150,
        random_state=42,
        prefer_horizontal=0.7,
        contour_width=1.5,
        contour_color="steelblue",
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=150)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo, fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()

    out_file = output_path / f"{titulo.lower().replace(' ', '_').replace('—', '_')}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_file


# ══════════════════════════════════════════════
#  2. GRÁFICAS DE BARRAS HORIZONTALES (top N)
# ══════════════════════════════════════════════
def plot_barras_ngramas(
    ngramas: List[Tuple[str, int]],
    titulo: str,
    output_path: Path,
    top_n: int = 50,
    color: str = PALETTE["primary"],
) -> Path:
    """
    Gráfica de barras horizontal para los top N n-gramas.
    Usa plotly para interactividad (hover muestra el conteo exacto).
    """
    top = ngramas[:top_n]
    if not top:
        print(f"  ⚠️  No hay n-gramas para graficar: {titulo}")
        return output_path

    words, counts = zip(*top)
    # Invertimos para que el #1 quede arriba
    words = words[::-1]
    counts = counts[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(words),
        x=list(counts),
        orientation="h",
        marker_color=color,
        text=list(counts),
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Frecuencia: %{x}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"<b>{titulo}</b>", x=0.5, font=dict(size=16)),
        xaxis_title="Frecuencia",
        yaxis_title="N-grama",
        template="plotly_white",
        height=max(500, len(words) * 22),
        margin=dict(l=220, r=60, t=80, b=60),
        yaxis=dict(tickfont=dict(size=11)),
    )

    out_file = output_path / f"{titulo.lower().replace(' ', '_').replace('—', '_')}.html"
    fig.write_html(str(out_file), include_plotlyjs="cdn")
    return out_file


# ══════════════════════════════════════════════
#  3. COMPARATIVA DE OUTLIERS POR FUENTE
# ══════════════════════════════════════════════
def plot_comparativa_outliers_por_fuente(
    stats_por_fuente: List[Dict[str, Any]],
    output_path: Path,
    palette: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:
    """
    Genera 3 gráficas comparativas:
      1) Barras agrupadas (Total / Outliers / Normales)
      2) Pastel de outliers por fuente
      3) Barras apiladas 100% con % de outliers
    Devuelve dict con las rutas de los archivos generados.
    """
    pal = _merge_palette(palette)

    df_stats = pd.DataFrame(stats_por_fuente)
    if df_stats.empty:
        print("  ⚠️  No hay estadísticas por fuente para graficar.")
        return {}

    # Ordenamos por % de outliers descendente
    df_stats = df_stats.sort_values("pct_outliers", ascending=False).reset_index(drop=True)
    archivos = {}

    # ──────────────────────────────────────
    # 3.1 Barras agrupadas
    # ──────────────────────────────────────
    fig1 = make_subplots(rows=1, cols=1)
    fig1.add_trace(go.Bar(
        name="Outliers",
        x=df_stats["fuente"],
        y=df_stats["outliers"],
        marker_color=pal["secondary"],
        text=df_stats["outliers"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Outliers: %{y}<extra></extra>",
    ))
    fig1.add_trace(go.Bar(
        name="Normales",
        x=df_stats["fuente"],
        y=df_stats["normales"],
        marker_color=pal["tertiary"],
        text=df_stats["normales"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Normales: %{y}<extra></extra>",
    ))
    fig1.update_layout(
        title=dict(text="<b>Outliers vs Normales por Fuente</b>", x=0.5, font=dict(size=16)),
        barmode="group",
        yaxis_title="Cantidad de comentarios",
        xaxis_title="Fuente",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    f1 = output_path / "01_outliers_vs_normales_por_fuente.html"
    fig1.write_html(str(f1), include_plotlyjs="cdn")
    archivos["barras_agrupadas"] = f1

    # ──────────────────────────────────────
    # 3.2 Pastel de outliers
    # ──────────────────────────────────────
    fig2 = px.pie(
        df_stats,
        names="fuente",
        values="outliers",
        title="Distribución de Outliers por Fuente",
        color_discrete_sequence=(palette.get("categorical") if palette and palette.get("categorical") else PALETTE_FUENTES),
        hole=0.35,
    )
    fig2.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Outliers: %{value}<br>%{percent}<extra></extra>",
    )
    fig2.update_layout(
        template="plotly_white",
        height=500,
        title=dict(x=0.5, font=dict(size=16)),
        annotations=[dict(
            text=f"Total: {df_stats['outliers'].sum():,}",
            x=0.5, y=0.5, font_size=14, showarrow=False,
        )],
    )
    f2 = output_path / "02_pie_outliers_por_fuente.html"
    fig2.write_html(str(f2), include_plotlyjs="cdn")
    archivos["pie"] = f2

    # ──────────────────────────────────────
    # 3.3 Barras apiladas 100% (% outliers)
    # ──────────────────────────────────────
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="% Outliers",
        x=df_stats["fuente"],
        y=df_stats["pct_outliers"],
        marker_color=pal["secondary"],
        text=df_stats["pct_outliers"].apply(lambda x: f"{x:.1f}%"),
        textposition="inside",
        textfont=dict(color="white", size=12),
        hovertemplate="<b>%{x}</b><br>% Outliers: %{y:.2f}%<extra></extra>",
    ))
    fig3.update_layout(
        title=dict(text="<b>Porcentaje de Outliers por Fuente</b>", x=0.5, font=dict(size=16)),
        yaxis_title="% de Outliers",
        xaxis_title="Fuente",
        template="plotly_white",
        height=450,
        yaxis=dict(range=[0, max(df_stats["pct_outliers"].max() * 1.2, 5)]),
    )
    f3 = output_path / "03_pct_outliers_por_fuente.html"
    fig3.write_html(str(f3), include_plotlyjs="cdn")
    archivos["pct_barras"] = f3

    # ──────────────────────────────────────
    # 3.4 BONUS: Tabla resumen interactiva
    # ──────────────────────────────────────
    fig4 = go.Figure(data=[go.Table(
        header=dict(
            values=[
                "<b>Fuente</b>", "<b>Total</b>", "<b>Outliers</b>",
                "<b>Normales</b>", "<b>% Outliers</b>"
            ],
            fill_color=pal["primary"],
            align="center",
            font=dict(color="white", size=12),
            line_color="white",
        ),
        cells=dict(
            values=[
                df_stats["fuente"],
                df_stats["total"].apply(lambda x: f"{x:,}"),
                df_stats["outliers"].apply(lambda x: f"{x:,}"),
                df_stats["normales"].apply(lambda x: f"{x:,}"),
                df_stats["pct_outliers"].apply(lambda x: f"{x:.2f}%"),
            ],
            fill_color=[[pal["background"], "white"] * len(df_stats)],
            align="center",
            font=dict(size=11),
            line_color="lightgray",
        ),
    )])
    fig4.update_layout(
        title=dict(text="<b>Tabla Resumen: Estadísticas de Outliers por Fuente</b>", x=0.5, font=dict(size=16)),
        template="plotly_white",
        height=50 + 35 * len(df_stats),
    )
    f4 = output_path / "04_tabla_resumen_outliers.html"
    fig4.write_html(str(f4), include_plotlyjs="cdn")
    archivos["tabla"] = f4

    return archivos


# ══════════════════════════════════════════════
#  4. HEATMAP: densidad de outliers por fuente
# ══════════════════════════════════════════════
def plot_heatmap_outliers_por_fuente(
    df_outliers: pd.DataFrame,
    source_column: str,
    output_path: Path,
    palette: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Heatmap de densidad de outliers por fuente,
    útil para detectar fuentes con comportamiento atípico concentrado.
    """
    if df_outliers.empty:
        print("  ⚠️  No hay outliers para heatmap.")
        return output_path

    pal = _merge_palette(palette)

    conteo = df_outliers[source_column].value_counts().reset_index()
    conteo.columns = ["fuente", "outliers"]
    conteo = conteo.sort_values("outliers", ascending=True)

    fig = go.Figure(data=go.Heatmap(
        x=conteo["fuente"],
        y=["Outliers"],
        z=[conteo["outliers"].tolist()],
        colorscale=[[0, pal["background"]], [1, pal["primary"]]],
        text=[[f"{v:,}" for v in conteo["outliers"].tolist()]],
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        hovertemplate="<b>%{x}</b><br>Outliers: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>Heatmap: Densidad de Outliers por Fuente</b>", x=0.5, font=dict(size=16)),
        template="plotly_white",
        height=200,
        yaxis=dict(showticklabels=False),
    )
    out_file = output_path / "05_heatmap_outliers_por_fuente.html"
    fig.write_html(str(out_file), include_plotlyjs="cdn")
    return out_file


# ══════════════════════════════════════════════
#  5. FUNCIÓN ORQUESTADORA
# ══════════════════════════════════════════════
def generar_visualizaciones_outliers(
    df_outliers: pd.DataFrame,
    text_column: str,
    source_column: str,
    unigramas: List[Tuple[str, int]],
    bigramas: List[Tuple[str, int]],
    trigramas: List[Tuple[str, int]],
    stats_por_fuente: List[Dict[str, Any]],
    output_dir: str = "results/outliers/visualizations",
    top_n: int = 50,
    palette: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Genera TODAS las visualizaciones del análisis de outliers:
      • 3 Wordclouds (uni/bi/tri-gramas)
      • 3 Gráficas de barras horizontales (uni/bi/tri-gramas)
      • 4 Comparativas por fuente (barras, pastel, %, tabla)
      • 1 Heatmap de densidad
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pal = _merge_palette(palette)

    SEP = "═" * 70
    print(f"\n{SEP}")
    print("   🎨  GENERANDO VISUALIZACIONES DE OUTLIERS")
    print(f"{SEP}")

    resultados: Dict[str, Any] = {}

    # ── 5.1 WORDCLOUDS ──
    print("\n  📌  Generando Wordclouds...")
    wc_uni = generar_wordcloud_ngramas(
        unigramas, "Wordcloud Unigramas Outliers", out,
        colormap="viridis", max_words=150,
    )
    wc_bi = generar_wordcloud_ngramas(
        bigramas, "Wordcloud Bigramas Outliers", out,
        colormap="plasma", max_words=120,
    )
    wc_tri = generar_wordcloud_ngramas(
        trigramas, "Wordcloud Trigramas Outliers", out,
        colormap="magma", max_words=100,
    )
    resultados["wordclouds"] = {"unigramas": wc_uni, "bigramas": wc_bi, "trigramas": wc_tri}
    print(f"     ✅ Wordclouds guardadas en: {out}/")

    # ── 5.2 BARRAS DE N-GRAMAS ──
    print("\n  📌  Generando gráficas de barras de n-gramas...")
    bar_uni = plot_barras_ngramas(
        unigramas, f"Top {top_n} Unigramas — Outliers", out,
        top_n=top_n, color=pal["primary"],
    )
    bar_bi = plot_barras_ngramas(
        bigramas, f"Top {top_n} Bigramas — Outliers", out,
        top_n=top_n, color=pal["tertiary"],
    )
    bar_tri = plot_barras_ngramas(
        trigramas, f"Top {top_n} Trigramas — Outliers", out,
        top_n=top_n, color=pal["quaternary"],
    )
    resultados["barras_ngramas"] = {
        "unigramas": bar_uni, "bigramas": bar_bi, "trigramas": bar_tri,
    }
    print(f"     ✅ Barras guardadas en: {out}/")

    # ── 5.3 COMPARATIVA POR FUENTE ──
    print("\n  📌  Generando comparativas por fuente...")
    comp = plot_comparativa_outliers_por_fuente(stats_por_fuente, out, palette=palette)
    resultados["comparativa_fuente"] = comp
    print(f"     ✅ Comparativas guardadas en: {out}/")

    # ── 5.4 HEATMAP ──
    print("\n  📌  Generando heatmap de densidad...")
    heatmap = plot_heatmap_outliers_por_fuente(df_outliers, source_column, out, palette=palette)
    resultados["heatmap"] = heatmap
    print(f"     ✅ Heatmap guardado en: {out}/")

    # ── RESUMEN FINAL ──
    print(f"\n{SEP}")
    print("   ✅  TODAS LAS VISUALIZACIONES GENERADAS")
    print(f"{SEP}")
    print(f"  📂  Directorio: {out}/")
    print(f"     • 3 Wordclouds (PNG)")
    print(f"     • 3 Gráficas de barras de n-gramas (HTML interactivo)")
    print(f"     • 3 Comparativas por fuente (HTML interactivo)")
    print(f"     • 1 Tabla resumen (HTML interactivo)")
    print(f"     • 1 Heatmap de densidad (HTML interactivo)")
    print(f"{SEP}\n")

    return resultados