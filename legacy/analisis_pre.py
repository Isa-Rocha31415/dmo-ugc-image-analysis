"""
analisis_pre.py
================================================================
Análisis exploratorio previo (EDA) del DataFrame concatenado
de comentarios turísticos, desglosado por fuente (source) y
consolidado en "all_sources".

Autor: Asistente de Análisis de Datos
Fecha: 2026-11-06
================================================================
"""

from typing import Dict, List, Optional
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud


# =====================================================================
# 1. PALETA "EJECUTIVA CONFIABLE" — Constantes de diseño
# =====================================================================
PALETTE = {
    "text": "#1A202C",
    "title": "#2D3748",
    "subtitle": "#4A5568",
    "paragraph": "#718096",
    "primary": "#2B6CB0",
    "secondary": "#319795",
    "tertiary": "#E2E8F0",
    "orange": "#DD6B20",
    "green": "#38A169",
    "red": "#E53E3E",
}

COMPLEMENTARY_COLORS = [
    "#2B6CB0", "#319795", "#DD6B20", "#38A169", "#E53E3E",
    "#805AD5", "#D69E2E", "#3182CE", "#38B2AC", "#E53E3E",
]

OUTPUT_DIR = Path("results/analisis_pre")


# =====================================================================
# 2. CONFIGURACIÓN GLOBAL DE MATPLOTLIB / SEABORN
# =====================================================================
def configure_plot_style() -> None:
    """Aplica el estilo ejecutivo confiable a todas las gráficas."""
    sns.set_theme(style="white", font="DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": PALETTE["tertiary"],
        "axes.labelcolor": PALETTE["title"],
        "xtick.color": PALETTE["paragraph"],
        "ytick.color": PALETTE["paragraph"],
        "axes.titlecolor": PALETTE["title"],
        "text.color": PALETTE["text"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def sanitize_filename(title: str) -> str:
    """Convierte un título en un nombre de archivo válido."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        title = title.replace(char, '_')
    title = title.replace(' ', '_').replace('—', '-').replace('(', '').replace(')', '')
    title = title.replace('__', '_').replace('___', '_')
    return title[:100]


def save_plot(fig: plt.Figure, filename: str) -> None:
    """Guarda la figura en el directorio de salida con el nombre indicado."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(filename) + ".png"
    filepath = OUTPUT_DIR / safe_name
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Plot guardado: {filepath}")


# =====================================================================
# 3. FUNCIONES AUXILIARES DE VISUALIZACIÓN
# =====================================================================
def plot_donut_chart(
    df: pd.DataFrame,
    col: str,
    palette_map: Optional[Dict[str, str]] = None,
    title: str = "Distribución",
    subtitle: Optional[str] = None,
    filename: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> None:
    """Grafica una dona minimalista con etiquetas externas para categorías pequeñas."""
    target_counts = df[col].value_counts()
    counts = target_counts.values
    labels = target_counts.index

    if palette_map is None:
        palette_map = {}
    default_palette = sns.color_palette("muted", n_colors=len(labels))
    colors = [
        palette_map.get(label, default_palette[i])
        for i, label in enumerate(labels)
    ]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    wedges, _ = ax.pie(
        counts, labels=None, startangle=140, colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )

    total = counts.sum()
    ax.text(
        0, 0, f"Total\n{total:,}",
        ha="center", va="center",
        fontsize=14, weight="bold", color=PALETTE["text"],
    )

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["tertiary"], lw=1)

    for i, wedge in enumerate(wedges):
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = "left" if x >= 0 else "right"

        ax.annotate(
            "",
            xy=(x * 0.85, y * 0.85),
            xytext=(x * 1.3, y * 1.3),
            arrowprops=dict(arrowstyle="-", color=PALETTE["paragraph"], lw=1),
        )

        label_text = f"{labels[i]}\n{counts[i]:,} ({counts[i]/total*100:.1f}%)"
        ax.text(
            x * 1.35, y * 1.35, label_text,
            horizontalalignment=horizontalalignment,
            fontsize=9, color=PALETTE["text"], bbox=bbox_props,
        )

    fig.text(0.5, 0.98, title, ha="center", fontsize=16, weight="bold", color=PALETTE["title"])
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=11, color=PALETTE["paragraph"])

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])
    save_plot(fig, filename or title)


def generate_wordcloud(
    text_series: pd.Series,
    title: str,
    subtitle: Optional[str] = None,
    colormap: str = "Blues",
    max_words: int = 150,
    filename: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Genera una nube de palabras estilizada con la paleta ejecutiva."""
    full_text = " ".join(text_series.dropna().astype(str).tolist())

    if not full_text.strip():
        print(f"[WARN] No hay texto disponible para '{title}'.")
        return

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap=colormap,
        max_words=max_words,
        contour_width=0,
        relative_scaling=0.5,
        collocations=False,
    ).generate(full_text)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    fig.text(0.5, 0.97, title, ha="center",
             fontsize=16, weight="bold", color=PALETTE["title"])
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center",
                 fontsize=11, color=PALETTE["paragraph"])

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.88])
    save_plot(fig, filename or title)


def plot_token_distribution(
    df: pd.DataFrame,
    token_col: str,
    source_label: str,
    bins: int = 40,
    filename: Optional[str] = None,
) -> None:
    """Histograma de la distribución de tokens por documento."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

    ax.hist(
        df[token_col], bins=bins,
        color=PALETTE["primary"], alpha=0.85,
        edgecolor="white", linewidth=1.2,
    )

    mean_val = df[token_col].mean()
    median_val = df[token_col].median()
    ax.axvline(mean_val, color=PALETTE["orange"],
               linestyle="--", linewidth=2, label=f"Media: {mean_val:.1f}")
    ax.axvline(median_val, color=PALETTE["green"],
               linestyle="--", linewidth=2, label=f"Mediana: {median_val:.1f}")

    title = f"Distribución de tokens por documento — {source_label}"
    ax.set_title(title, fontsize=14, weight="bold", color=PALETTE["title"], pad=12)
    ax.set_xlabel("Número de tokens", color=PALETTE["title"])
    ax.set_ylabel("Frecuencia", color=PALETTE["title"])
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color=PALETTE["tertiary"])
    sns.despine(ax=ax)
    plt.tight_layout()
    save_plot(fig, filename or title)


def plot_top_words_bar(
    text_series: pd.Series,
    source_label: str,
    top_n: int = 20,
    color: str = PALETTE["primary"],
    filename: Optional[str] = None,
) -> None:
    """Gráfica de barras horizontales con las top_n palabras más frecuentes."""
    all_tokens = " ".join(text_series.dropna().astype(str)).split()
    counter = Counter(all_tokens)
    top_words = counter.most_common(top_n)

    if not top_words:
        print(f"[WARN] No hay palabras para graficar en '{source_label}'.")
        return

    words = [w for w, _ in top_words][::-1]
    counts = [c for _, c in top_words][::-1]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    bars = ax.barh(words, counts, color=color, edgecolor="white", height=0.7)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", fontsize=9, color=PALETTE["paragraph"],
        )

    title = f"Top {top_n} palabras más frecuentes — {source_label}"
    ax.set_title(title, fontsize=14, weight="bold", color=PALETTE["title"], pad=12)
    ax.set_xlabel("Frecuencia", color=PALETTE["title"])
    ax.grid(axis="x", linestyle=":", alpha=0.5, color=PALETTE["tertiary"])
    sns.despine(ax=ax)
    plt.tight_layout()
    save_plot(fig, filename or title)


def plot_document_length_boxplot(
    df: pd.DataFrame,
    token_col: str,
    source_col: str,
    title: str = "Comparativo de longitud de documentos entre fuentes",
    filename: Optional[str] = None,
) -> None:
    """Boxplot comparativo de la longitud (tokens) entre fuentes."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    sources = df[source_col].unique()
    palette_dict = {
        src: COMPLEMENTARY_COLORS[i % len(COMPLEMENTARY_COLORS)]
        for i, src in enumerate(sources)
    }

    # Corrección del FutureWarning: usar hue=source_col y legend=False
    sns.boxplot(
        data=df, x=source_col, y=token_col,
        hue=source_col, palette=palette_dict, ax=ax,
        width=0.5, fliersize=3, legend=False,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(color=PALETTE["text"], linewidth=2),
    )

    ax.set_title(title, fontsize=14, weight="bold",
                 color=PALETTE["title"], pad=12)
    ax.set_xlabel("Fuente", color=PALETTE["title"])
    ax.set_ylabel("Número de tokens", color=PALETTE["title"])
    ax.grid(axis="y", linestyle=":", alpha=0.5, color=PALETTE["tertiary"])
    sns.despine(ax=ax)
    plt.tight_layout()
    save_plot(fig, filename or title)


def plot_lexical_richness(
    df: pd.DataFrame,
    token_col: str,
    source_col: str,
    text_col: str = "processed_text",
    title: str = "Riqueza léxica por fuente (palabras únicas / total)",
    filename: Optional[str] = None,
) -> None:
    """Compara la riqueza léxica (palabras únicas / total) por fuente."""
    richness = []
    for src, group in df.groupby(source_col):
        all_tokens = " ".join(group[text_col].dropna().astype(str)).split()
        total = len(all_tokens)
        unique = len(set(all_tokens))
        ratio = unique / total if total > 0 else 0
        richness.append({
            "source": src, "total_tokens": total,
            "unique_tokens": unique, "richness": ratio,
        })

    richness_df = pd.DataFrame(richness).sort_values("richness", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    bars = ax.bar(
        richness_df["source"], richness_df["richness"],
        color=PALETTE["secondary"], edgecolor="white", linewidth=1.2,
    )

    for bar, val in zip(bars, richness_df["richness"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.2%}", ha="center", fontsize=10,
            weight="bold", color=PALETTE["title"],
        )

    ax.set_title(title, fontsize=14, weight="bold", color=PALETTE["title"], pad=12)
    ax.set_ylabel("Ratio de vocabulario único", color=PALETTE["title"])
    ax.set_ylim(0, max(richness_df["richness"]) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.5, color=PALETTE["tertiary"])
    sns.despine(ax=ax)
    plt.tight_layout()
    save_plot(fig, filename or title)


# =====================================================================
# 4. MÉTRICAS Y TABLA RESUMEN
# =====================================================================
def compute_source_metrics(
    df: pd.DataFrame,
    text_col: str,
    source_col: str,
) -> pd.DataFrame:
    """Calcula métricas descriptivas por fuente."""
    df = df.copy()
    df["_token_count"] = df[text_col].dropna().astype(str).apply(
        lambda x: len(str(x).split())
    )

    metrics = df.groupby(source_col)["_token_count"].agg(
        n_docs="count",
        mean_tokens="mean",
        median_tokens="median",
        min_tokens="min",
        max_tokens="max",
        std_tokens="std",
        total_tokens="sum",
    ).reset_index()

    for col in ["mean_tokens", "median_tokens", "std_tokens"]:
        metrics[col] = metrics[col].round(1)
    return metrics


def print_summary_table(metrics_df: pd.DataFrame, label: str) -> None:
    """Imprime una tabla resumen estilizada en consola."""
    print("\n" + "=" * 70)
    print(f"  📊 RESUMEN MÉTRICAS — {label}")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70 + "\n")


# =====================================================================
# 5. PIPELINE DE ANÁLISIS POR FUENTE
# =====================================================================
def analyze_single_source(
    df: pd.DataFrame,
    source_label: str,
    source_col: str = "source",
    processed_col: str = "processed_text",
    raw_col: str = "text",
    top_n_words: int = 20,
    plot_counter: int = 2,
) -> int:
    """Ejecuta el análisis descriptivo completo para una fuente concreta.

    Devuelve el contador actualizado para mantener la numeración consecutiva.
    """
    print(f"\n🔎 Iniciando análisis para: {source_label.upper()}")
    print("-" * 50)

    # --- 5.1 Nubes de palabras: ANTES vs DESPUÉS ---
    generate_wordcloud(
        df[raw_col],
        title=f"Wordcloud ANTES — {source_label} (texto crudo)",
        subtitle=f"Fuente: {source_label}  |  Columna: '{raw_col}'",
        colormap="Greys",
        max_words=120,
        filename=f"{plot_counter:02d}_wordcloud_antes_{source_label}",
    )
    plot_counter += 1

    generate_wordcloud(
        df[processed_col],
        title=f"Wordcloud DESPUÉS — {source_label} (texto limpio)",
        subtitle=f"Fuente: {source_label}  |  Columna: '{processed_col}'",
        colormap="Blues",
        max_words=150,
        filename=f"{plot_counter:02d}_wordcloud_despues_{source_label}",
    )
    plot_counter += 1

    # --- 5.2 Distribución de longitud de documentos ---
    df = df.copy()
    df["_token_count"] = df[processed_col].dropna().astype(str).apply(
        lambda x: len(str(x).split())
    )
    plot_token_distribution(
        df, "_token_count", source_label,
        filename=f"{plot_counter:02d}_distribucion_tokens_{source_label}",
    )
    plot_counter += 1

    # --- 5.3 Top palabras más frecuentes ---
    plot_top_words_bar(
        df[processed_col], source_label,
        top_n=top_n_words, color=PALETTE["primary"],
        filename=f"{plot_counter:02d}_top_palabras_{source_label}",
    )
    plot_counter += 1

    return plot_counter


def run_full_analysis(
    df: pd.DataFrame,
    source_col: str = "source",
    processed_col: str = "processed_text",
    raw_col: str = "text",
    top_n_words: int = 20,
) -> pd.DataFrame:
    """Ejecuta el análisis completo: por cada fuente + consolidado."""
    configure_plot_style()

    for col in [source_col, processed_col, raw_col]:
        if col not in df.columns:
            raise KeyError(f"❌ La columna '{col}' no existe en el DataFrame.")

    sources = sorted(df[source_col].unique().tolist())
    print(f"\n🌐 Fuentes detectadas: {sources}")

    # Contador para numerar los plots en orden de generación
    plot_counter = 1

    # --- 6.1 Donut: proporción de comentarios por fuente ---
    plot_donut_chart(
        df, source_col,
        title="Proporción de comentarios por fuente",
        subtitle="Distribución del volumen de documentos en el dataset concatenado",
        filename=f"{plot_counter:02d}_proporcion_comentarios_por_fuente",
    )
    plot_counter += 1

    # --- 6.2 Análisis individual por fuente ---
    for src in sources:
        df_src = df[df[source_col] == src].copy()
        plot_counter = analyze_single_source(
            df_src, source_label=src,
            source_col=source_col,
            processed_col=processed_col,
            raw_col=raw_col,
            top_n_words=top_n_words,
            plot_counter=plot_counter,
        )

    # --- 6.3 Análisis consolidado (all_sources) ---
    plot_counter = analyze_single_source(
        df, source_label="all_sources",
        source_col=source_col,
        processed_col=processed_col,
        raw_col=raw_col,
        top_n_words=top_n_words,
        plot_counter=plot_counter,
    )

    # --- 6.4 Comparativos transversales ---
    df["_token_count"] = df[processed_col].dropna().astype(str).apply(
        lambda x: len(str(x).split())
    )

    plot_document_length_boxplot(
        df, "_token_count", source_col,
        title="Comparativo de longitud de documentos entre fuentes",
        filename=f"{plot_counter:02d}_boxplot_longitud_entre_fuentes",
    )
    plot_counter += 1

    plot_lexical_richness(
        df, "_token_count", source_col, processed_col,
        title="Riqueza léxica por fuente (palabras únicas / total)",
        filename=f"{plot_counter:02d}_riqueza_lexica_entre_fuentes",
    )
    plot_counter += 1

    # --- 6.5 Tabla resumen de métricas ---
    metrics_df = compute_source_metrics(df, processed_col, source_col)
    print_summary_table(metrics_df, label="POR FUENTE")

    global_metrics = compute_source_metrics(
        df.assign(source="__all__"), processed_col, "source"
    )
    print_summary_table(global_metrics, label="CONSOLIDADO (all_sources)")

    # Guardar la tabla de métricas como CSV para referencia
    metrics_path = OUTPUT_DIR / "99_resumen_metricas_fuentes.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"✅ Tabla de métricas guardada: {metrics_path}")

    return metrics_df


# =====================================================================
# 6. PUNTO DE ENTRADA
# =====================================================================
if __name__ == "__main__":
    df = pd.read_csv("results/df_preprocessed.csv")
    run_full_analysis(df, source_col="source",
                      processed_col="processed_text", raw_col="text")