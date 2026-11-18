"""Stage A: plumb the --palette hyperparameter through the plotting modules."""
from pathlib import Path

CODE = Path(__file__).resolve().parent


def patch(fname, pairs):
    p = CODE / fname
    s = p.read_text()
    for old, new in pairs:
        assert old in s, f"{fname}: NOT FOUND -> {old[:70]!r}"
        s = s.replace(old, new, 1)
    p.write_text(s)
    print(f"patched {fname}")


# ---------------------------------------------------------------- utils.py
patch("utils.py", [(
    '''# ============================================================
# CONSTANTES DE ESTILO EJECUTIVO (para compatibilidad con código existente)''',
    '''def palette_to_plot_colors(
    palette_name: str,
    vision_type: str = "normal",
) -> Dict[str, Any]:
    """
    Convierte una paleta de `utils` a los nombres de color que usan los
    gráficos del proyecto (primary/secondary/... + positive/negative).

    Permite que todos los HTML compartan la misma gama de colores.
    """
    palette = get_palette(palette_name, vision_type)
    complementarios = list(palette.get("complementarios", []))

    def _pick(index: int, fallback: str) -> str:
        if len(complementarios) > index:
            return complementarios[index]
        return fallback

    return {
        "primary": palette["primario"],
        "secondary": palette["secundario"],
        "tertiary": _pick(0, palette["secundario"]),
        "quaternary": _pick(1, palette["primario"]),
        "background": palette.get("fondo", "#FFFFFF"),
        "text": palette.get("texto", "#1A202C"),
        "title": palette.get("titulo", "#2D3748"),
        "positive": _pick(1, "#009E73"),
        "negative": _pick(2, "#D55E00"),
        "categorical": get_categorical_colors(palette_name, vision_type, n_colors=8),
    }


# ============================================================
# CONSTANTES DE ESTILO EJECUTIVO (para compatibilidad con código existente)''',
)])


# ------------------------------------------------- outlier_visualizations.py
patch("outlier_visualizations.py", [
    (
        "from typing import Any, Dict, List, Tuple",
        "from typing import Any, Dict, List, Optional, Tuple",
    ),
    (
        '''PALETTE_FUENTES = px.colors.qualitative.Safe   # ← paleta apta para daltonismo''',
        '''PALETTE_FUENTES = px.colors.qualitative.Safe   # ← paleta apta para daltonismo


def _merge_palette(palette: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Combina la paleta por defecto con la elegida por el usuario."""
    merged = dict(PALETTE)
    if palette:
        for key in ("primary", "secondary", "tertiary", "quaternary", "background"):
            if palette.get(key):
                merged[key] = palette[key]
    return merged''',
    ),
    (
        '''def plot_comparativa_outliers_por_fuente(
    stats_por_fuente: List[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Path]:''',
        '''def plot_comparativa_outliers_por_fuente(
    stats_por_fuente: List[Dict[str, Any]],
    output_path: Path,
    palette: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:''',
    ),
    (
        '''    df_stats = pd.DataFrame(stats_por_fuente)
    if df_stats.empty:
        print("  ⚠️  No hay estadísticas por fuente para graficar.")
        return {}''',
        '''    pal = _merge_palette(palette)

    df_stats = pd.DataFrame(stats_por_fuente)
    if df_stats.empty:
        print("  ⚠️  No hay estadísticas por fuente para graficar.")
        return {}''',
    ),
    (
        '''        marker_color=PALETTE["secondary"],
        text=df_stats["outliers"],''',
        '''        marker_color=pal["secondary"],
        text=df_stats["outliers"],''',
    ),
    (
        '''        marker_color=PALETTE["tertiary"],
        text=df_stats["normales"],''',
        '''        marker_color=pal["tertiary"],
        text=df_stats["normales"],''',
    ),
    (
        '''        color_discrete_sequence=PALETTE_FUENTES,
        hole=0.35,''',
        '''        color_discrete_sequence=pal["categorical"] if palette and palette.get("categorical") else PALETTE_FUENTES,
        hole=0.35,''',
    ),
    (
        '''        marker_color=PALETTE["secondary"],
        text=df_stats["pct_outliers"].apply(lambda x: f"{x:.1f}%"),''',
        '''        marker_color=pal["secondary"],
        text=df_stats["pct_outliers"].apply(lambda x: f"{x:.1f}%"),''',
    ),
    (
        '''            fill_color=PALETTE["primary"],
            align="center",''',
        '''            fill_color=pal["primary"],
            align="center",''',
    ),
    (
        '''            fill_color=[[PALETTE["background"], "white"] * len(df_stats)],''',
        '''            fill_color=[[pal["background"], "white"] * len(df_stats)],''',
    ),
    (
        '''def plot_heatmap_outliers_por_fuente(
    df_outliers: pd.DataFrame,
    source_column: str,
    output_path: Path,
) -> Path:''',
        '''def plot_heatmap_outliers_por_fuente(
    df_outliers: pd.DataFrame,
    source_column: str,
    output_path: Path,
    palette: Optional[Dict[str, str]] = None,
) -> Path:''',
    ),
    (
        '''    conteo = df_outliers[source_column].value_counts().reset_index()''',
        '''    pal = _merge_palette(palette)

    conteo = df_outliers[source_column].value_counts().reset_index()''',
    ),
    (
        '''        colorscale="Viridis",''',
        '''        colorscale=[[0, pal["background"]], [1, pal["primary"]]],''',
    ),
    (
        '''    output_dir: str = "results/outliers/visualizations",
    top_n: int = 50,
) -> Dict[str, Any]:''',
        '''    output_dir: str = "results/outliers/visualizations",
    top_n: int = 50,
    palette: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:''',
    ),
    (
        '''    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    SEP = "═" * 70
    print(f"\\n{SEP}")
    print("   🎨  GENERANDO VISUALIZACIONES DE OUTLIERS")''',
        '''    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pal = _merge_palette(palette)

    SEP = "═" * 70
    print(f"\\n{SEP}")
    print("   🎨  GENERANDO VISUALIZACIONES DE OUTLIERS")''',
    ),
    (
        '''        top_n=top_n, color=PALETTE["primary"],''',
        '''        top_n=top_n, color=pal["primary"],''',
    ),
    (
        '''        top_n=top_n, color=PALETTE["tertiary"],''',
        '''        top_n=top_n, color=pal["tertiary"],''',
    ),
    (
        '''        top_n=top_n, color=PALETTE["quaternary"],''',
        '''        top_n=top_n, color=pal["quaternary"],''',
    ),
    (
        '''    comp = plot_comparativa_outliers_por_fuente(stats_por_fuente, out)''',
        '''    comp = plot_comparativa_outliers_por_fuente(stats_por_fuente, out, palette=palette)''',
    ),
    (
        '''    heatmap = plot_heatmap_outliers_por_fuente(df_outliers, source_column, out)''',
        '''    heatmap = plot_heatmap_outliers_por_fuente(df_outliers, source_column, out, palette=palette)''',
    ),
])


# ------------------------------------------------------------ visualization.py
patch("visualization.py", [
    (
        '''def plot_topics_3d_umap(
    topics_df: pd.DataFrame,
    embeddings_dict: Dict[str, np.ndarray],
    text_column: str,
    source_column: str,
    output_dir: str = "results/topics",
    ) -> None:''',
        '''def plot_topics_3d_umap(
    topics_df: pd.DataFrame,
    embeddings_dict: Dict[str, np.ndarray],
    text_column: str,
    source_column: str,
    output_dir: str = "results/topics",
    palette: Optional[Dict[str, str]] = None,
    ) -> None:''',
    ),
    (
        '''        # Agregar puntos positivos (verde)''',
        '''        color_pos = (palette or {}).get("positive", "#00A859")
        color_neg = (palette or {}).get("negative", "#E74C3C")

        # Agregar puntos positivos (verde)''',
    ),
    (
        '''                color="#00A859",  # Verde''',
        '''                color=color_pos,  # Verde''',
    ),
    (
        '''                color="#E74C3C",  # Rojo''',
        '''                color=color_neg,  # Rojo''',
    ),
    (
        '''def plot_topic_distribution_by_source_3d(
    topics_df: pd.DataFrame,
    source_column: str,
    output_dir: str = "results/topics",
) -> None:''',
        '''def plot_topic_distribution_by_source_3d(
    topics_df: pd.DataFrame,
    source_column: str,
    output_dir: str = "results/topics",
    palette: Optional[Dict[str, str]] = None,
) -> None:''',
    ),
    (
        '''        color = "#00A859" if sentiment == "positive" else "#E74C3C"''',
        '''        if sentiment == "positive":
            color = (palette or {}).get("positive", "#00A859")
        else:
            color = (palette or {}).get("negative", "#E74C3C")''',
    ),
])


# ------------------------------------------------------ topic_console_report.py
patch("topic_console_report.py", [
    (
        """    top_n_terms: int = 20,
    embeddings: Optional[np.ndarray] = None,
) -> Dict[str, Any]:""",
        """    top_n_terms: int = 20,
    embeddings: Optional[np.ndarray] = None,
    palette: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:""",
    ),
    (
        """        text_column=text_column,
        source_column=source_column,
        output_dir=output_dir,
    )

    plot_topic_distribution_by_source_3d(
        topics_df=topics_df,
        source_column=source_column,
        output_dir=output_dir,
    )""",
        """        text_column=text_column,
        source_column=source_column,
        output_dir=output_dir,
        palette=palette,
    )

    plot_topic_distribution_by_source_3d(
        topics_df=topics_df,
        source_column=source_column,
        output_dir=output_dir,
        palette=palette,
    )""",
    ),
])


# ------------------------------------------------------ sentiment_wordclouds.py
patch("sentiment_wordclouds.py", [
    (
        """    colormap_pos: str = "Greens",
    colormap_neg: str = "Reds",
    top_n: int = 80,
) -> Dict[str, Any]:""",
        """    colormap_pos: str = "Greens",
    colormap_neg: str = "Reds",
    top_n: int = 80,
    palette: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:""",
    ),
    (
        """    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resultados: Dict[str, Any] = {}
    grupos = [
        ("Normales — Positivos",   df_clean, "positive", colormap_pos),
        ("Normales — Negativos",   df_clean, "negative", colormap_neg),
        ("Outliers — Positivos",   df_outliers, "positive", colormap_pos),
        ("Outliers — Negativos",   df_outliers, "negative", colormap_neg),
    ]""",
        """    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if palette:
        colormap_pos = _colormap_from_colors(
            palette.get("positive", "#009E73"), palette.get("tertiary", "#E2E8F0")
        )
        colormap_neg = _colormap_from_colors(
            palette.get("negative", "#D55E00"), palette.get("tertiary", "#E2E8F0")
        )

    resultados: Dict[str, Any] = {}
    grupos = [
        ("Normales — Positivos",   df_clean, "positive", colormap_pos),
        ("Normales — Negativos",   df_clean, "negative", colormap_neg),
        ("Outliers — Positivos",   df_outliers, "positive", colormap_pos),
        ("Outliers — Negativos",   df_outliers, "negative", colormap_neg),
    ]""",
    ),
    (
        """# ══════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════""",
        '''def _colormap_from_colors(dark: str, light: str):
    """Crea un colormap de matplotlib a partir de dos colores de la paleta."""
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return "viridis"

    try:
        return LinearSegmentedColormap.from_list("paleta_pipeline", [light, dark])
    except Exception:
        return "viridis"


# ══════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════''',
    ),
])

print("stage A done")
