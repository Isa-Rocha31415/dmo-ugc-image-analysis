import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from typing import List
import spacy
from nltk.corpus import stopwords
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np



from utils import (
    PALETAS_ACCESIBLES,
    aplicar_estilo_ejecutivo,
    get_categorical_colors,
    get_palette,
    palette_to_plot_colors,
)


# ============================================================
# EXTRACCIÓN DE N-GRAMAS
# ============================================================
def extraer_ngramas(textos: List[str], n: int) -> Counter:
    """Extrae n-gramas de una lista de textos filtrando stopwords en español."""
    ngramas = []
    nlp_es = spacy.load("es_core_news_sm")
    stop_words_es = set(stopwords.words('spanish'))
    
    for texto in textos:
        words = re.findall(r'\b\w+\b', str(texto).lower())
        words = [w for w in words if w not in stop_words_es and len(w) > 2]
        
        for i in range(len(words) - n + 1):
            ngramas.append(" ".join(words[i:i+n]))
    
    return Counter(ngramas)


# ============================================================
# VISUALIZACIÓN DE N-GRAMAS
# ============================================================
def graficar_top_ngramas(
    ngramas_counter: Counter,
    n: int,
    titulo: str,
    ruta_guardado: str,
    palette_name: str = "ejecutiva",
    vision_type: str = "normal",
    top_n: int = 15
) -> go.Figure:
    """Genera y guarda un gráfico de barras horizontales de n-gramas."""
    df_ngramas = pd.DataFrame(
        ngramas_counter.most_common(top_n),
        columns=['N-grama', 'Frecuencia']
    )
    
    palette = get_palette(palette_name, vision_type)
    color_scale = [palette["terciario"], palette["secundario"], palette["primario"]]
    
    fig = px.bar(
        df_ngramas,
        x='Frecuencia',
        y='N-grama',
        orientation='h',
        color='Frecuencia',
        color_continuous_scale=color_scale,
        title=titulo
    )
    
    fig = aplicar_estilo_ejecutivo(fig, titulo, palette_name, vision_type)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_yaxes(autorange='reversed')
    
    fig.write_html(ruta_guardado)
    return fig


def graficar_comparacion_metodos_outliers(
    df: pd.DataFrame,
    titulo: str,
    ruta_guardado: str,
    palette_name: str = "ejecutiva",
    vision_type: str = "normal"
) -> go.Figure:
    """Compara visualmente la distribución de outliers entre Isolation Forest y LOF."""
    df_viz = df.copy()
    df_viz['IF_Label'] = df_viz['outlier_IF'].map({1: 'Normal', -1: 'Outlier'})
    df_viz['LOF_Label'] = df_viz['outlier_LOF'].map({1: 'Normal', -1: 'Outlier'})
    
    conteo = pd.DataFrame({
        'Método': ['Isolation Forest', 'Isolation Forest', 'LOF', 'LOF'],
        'Categoría': ['Normal', 'Outlier', 'Normal', 'Outlier'],
        'Cantidad': [
            (df_viz['IF_Label'] == 'Normal').sum(),
            (df_viz['IF_Label'] == 'Outlier').sum(),
            (df_viz['LOF_Label'] == 'Normal').sum(),
            (df_viz['LOF_Label'] == 'Outlier').sum()
        ]
    })
    
    colors = get_categorical_colors(palette_name, vision_type, n_colors=5)
    
    fig = px.bar(
        conteo,
        x='Método',
        y='Cantidad',
        color='Categoría',
        barmode='group',
        color_discrete_map={
            'Normal': colors[1],
            'Outlier': colors[3] if len(colors) > 3 else colors[0]
        },
        title=titulo
    )
    
    fig = aplicar_estilo_ejecutivo(fig, titulo, palette_name, vision_type)
    fig.write_html(ruta_guardado)
    return fig


def graficar_distribucion_fuentes(
    df: pd.DataFrame,
    fuente_col: str,
    titulo: str,
    ruta_guardado: str,
    palette_name: str = "ejecutiva",
    vision_type: str = "normal"
) -> go.Figure:
    """Genera gráfico de distribución de comentarios por fuente."""
    conteo = df[fuente_col].value_counts().reset_index()
    conteo.columns = ['Fuente', 'Cantidad']
    
    colors = get_categorical_colors(palette_name, vision_type, n_colors=len(conteo))
    
    fig = px.bar(
        conteo,
        x='Fuente',
        y='Cantidad',
        color='Fuente',
        color_discrete_sequence=colors,
        title=titulo
    )
    
    fig = aplicar_estilo_ejecutivo(fig, titulo, palette_name, vision_type)
    fig.write_html(ruta_guardado)
    return fig


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _td_ensure_dir(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _td_save_html(fig: go.Figure, path: Path) -> None:
    """
    Guarda un gráfico Plotly como HTML.

    Si necesitas que los HTML funcionen sin internet, cambia:
      include_plotlyjs="cdn"
    por:
      include_plotlyjs="inline"
    """
    _td_ensure_dir(path)
    fig.write_html(str(path), include_plotlyjs="cdn")


def _td_palette(palette: Any) -> Dict[str, Any]:
    """Resuelve nombre, lista o dict de paleta al contrato del proyecto."""
    default = palette_to_plot_colors("ejecutiva")
    if palette is None:
        resolved = default
    elif isinstance(palette, str):
        name = palette if palette in PALETAS_ACCESIBLES else "ejecutiva"
        resolved = palette_to_plot_colors(name)
    elif isinstance(palette, dict):
        resolved = dict(palette)
    elif isinstance(palette, (list, tuple)):
        colors = [str(color) for color in palette if isinstance(color, str)]
        resolved = dict(default)
        if colors:
            resolved["primary"] = colors[0]
            resolved["secondary"] = colors[1] if len(colors) > 1 else colors[0]
            resolved["positive"] = colors[0]
            resolved["negative"] = colors[1] if len(colors) > 1 else colors[0]
            resolved["categorical"] = colors
    else:
        resolved = default

    for key, value in default.items():
        resolved.setdefault(key, value)

    categorical = [
        str(color)
        for color in resolved.get("categorical", [])
        if isinstance(color, str)
    ]
    for key in ("primary", "secondary", "tertiary", "quaternary", "positive", "negative"):
        color = resolved.get(key)
        if isinstance(color, str) and color not in categorical:
            categorical.append(color)
    resolved["categorical"] = categorical
    return resolved


def _td_colors(palette: Any, n: Optional[int] = None) -> List[str]:
    """Convierte una paleta del proyecto en lista de colores categóricos."""
    colors = list(_td_palette(palette)["categorical"])
    if n is not None and len(colors) < n:
        repeats = (n // len(colors)) + 1
        colors = (colors * repeats)[:n]
    return list(colors)


def _td_truncate(value: Any, max_chars: int = 140) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _td_normalize_terms(values: Any) -> List[str]:
    """
    Normaliza keywords/términos que pueden venir como string,
    lista, tupla o diccionario.
    """
    if values is None:
        return []

    if isinstance(values, str):
        parts = re.split(r",|;|\|", values)
        return [part.strip() for part in parts if part.strip()]

    if isinstance(values, dict):
        values = list(values.keys())

    normalized = []

    for value in values:
        if isinstance(value, (list, tuple)) and len(value) > 0:
            normalized.append(str(value[0]))
        else:
            normalized.append(str(value))

    return normalized


# ---------------------------------------------------------------------------
# Reducción 2D para embeddings
# ---------------------------------------------------------------------------

def td_reduce_embeddings_2d(
    embeddings: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Tuple[np.ndarray, str]:
    """
    Reduce embeddings a 2D.

    Usa UMAP si está disponible.
    Si falla, usa PCA como respaldo.
    """
    X = np.asarray(embeddings, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError("embeddings debe ser una matriz 2D: (documentos, dimensiones).")

    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )

            coords = reducer.fit_transform(X)
            return coords, "umap"

        except Exception:
            # Si UMAP no está instalado o falla, usamos PCA.
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(X)
        return coords, "pca"

    raise ValueError(f"Método de reducción no soportado: {method}")


# ---------------------------------------------------------------------------
# Espacio semántico UMAP / PCA
# ---------------------------------------------------------------------------

def plot_embedding_space(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    color_col: Optional[str],
    title: str,
    output_html: Path,
    palette: Any = "Plotly",
    text_column: Optional[str] = None,
    max_points: int = 12000,
    method: str = "umap",
    random_state: int = 42,
) -> Optional[go.Figure]:
    """
    Grafica el espacio semántico de comentarios.

    Puedes colorear por:
      - source_column
      - sentiment_label
      - topic_name
    """
    if df.empty:
        return None

    if len(df) != len(embeddings):
        raise ValueError(
            f"df tiene {len(df)} filas pero embeddings tiene {len(embeddings)}."
        )

    pal = _td_palette(palette)
    tmp = df.copy().reset_index(drop=True)
    emb = np.asarray(embeddings)

    if max_points is not None and len(tmp) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(tmp), size=max_points, replace=False)
        tmp = tmp.iloc[idx].reset_index(drop=True)
        emb = emb[idx]

    coords, used_method = td_reduce_embeddings_2d(
        embeddings=emb,
        method=method,
        random_state=random_state,
    )

    tmp["x"] = coords[:, 0]
    tmp["y"] = coords[:, 1]

    hover_data = {}

    if text_column is not None and text_column in tmp.columns:
        tmp["texto"] = tmp[text_column].map(lambda value: _td_truncate(value, 160))
        hover_data["texto"] = True

    if color_col is not None and color_col in tmp.columns:
        tmp[color_col] = tmp[color_col].astype(str)

        fig = px.scatter(
            tmp,
            x="x",
            y="y",
            color=color_col,
            title=f"{title} ({used_method.upper()})",
            color_discrete_sequence=_td_colors(palette, tmp[color_col].nunique()),
            hover_data=hover_data,
        )

    else:
        fig = px.scatter(
            tmp,
            x="x",
            y="y",
            title=f"{title} ({used_method.upper()})",
            hover_data=hover_data,
        )

    fig.update_traces(marker=dict(size=7, opacity=0.75))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"]),
    )

    _td_save_html(fig, output_html)

    return fig


# ---------------------------------------------------------------------------
# Sentimiento
# ---------------------------------------------------------------------------

def plot_sentiment_by_source(
    df: pd.DataFrame,
    source_column: str,
    output_html: Path,
    palette: Any = None,
    title: str = "Sentimiento por fuente",
) -> Optional[go.Figure]:
    """
    Barras apiladas con porcentaje de positivo/negativo por fuente.
    """
    if df.empty or "sentiment_label" not in df.columns:
        return None

    pal = _td_palette(palette)
    tmp = df.copy()
    tmp["sentiment_label"] = tmp["sentiment_label"].astype(str)

    counts = (
        tmp.groupby([source_column, "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    if counts.empty:
        return None

    totals = counts.groupby(source_column)["count"].transform("sum")
    counts["pct"] = 100.0 * counts["count"] / totals

    fig = px.bar(
        counts,
        x=source_column,
        y="pct",
        color="sentiment_label",
        barmode="stack",
        title=title,
        labels={
            "pct": "Porcentaje (%)",
            source_column: "Fuente",
            "sentiment_label": "Sentimiento",
        },
        color_discrete_map={
            "positive": pal["positive"],
            "negative": pal["negative"],
        },
    )

    fig.update_yaxes(range=[0, 100])
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"]),
    )

    _td_save_html(fig, output_html)

    return fig


def plot_sentiment_score_distribution(
    df: pd.DataFrame,
    source_column: str,
    output_html: Path,
    palette: Any = "Plotly",
    title: str = "Distribución del score de sentimiento",
) -> Optional[go.Figure]:
    """
    Histograma del score continuo de sentimiento por fuente.
    """
    if df.empty or "sentiment_score" not in df.columns:
        return None

    pal = _td_palette(palette)
    tmp = df.copy()
    tmp[source_column] = tmp[source_column].astype(str)

    fig = px.histogram(
        tmp,
        x="sentiment_score",
        color=source_column,
        barmode="overlay",
        marginal="box",
        title=title,
        labels={
            "sentiment_score": "Score de sentimiento",
            source_column: "Fuente",
        },
        color_discrete_sequence=_td_colors(palette, tmp[source_column].nunique()),
        opacity=0.75,
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"]),
    )

    _td_save_html(fig, output_html)

    return fig


# ---------------------------------------------------------------------------
# Tópicos
# ---------------------------------------------------------------------------

def plot_topic_distribution_by_source(
    topics_df: pd.DataFrame,
    source_column: str,
    output_html: Path,
    palette: Any = "Plotly",
    top_topics: int = 15,
    title: str = "Distribución de tópicos por fuente",
) -> Optional[go.Figure]:
    """
    Barras apiladas con porcentaje de cada tópico por fuente.
    """
    if topics_df.empty or "topic_name" not in topics_df.columns:
        return None

    pal = _td_palette(palette)
    tmp = topics_df.copy()
    tmp[source_column] = tmp[source_column].astype(str)
    tmp["topic_name"] = tmp["topic_name"].astype(str)

    top_topic_names = tmp["topic_name"].value_counts().head(top_topics).index
    tmp["topic_name"] = tmp["topic_name"].where(
        tmp["topic_name"].isin(top_topic_names),
        "Otros",
    )

    counts = (
        tmp.groupby([source_column, "topic_name"])
        .size()
        .reset_index(name="count")
    )

    if counts.empty:
        return None

    totals = counts.groupby(source_column)["count"].transform("sum")
    counts["pct"] = 100.0 * counts["count"] / totals

    fig = px.bar(
        counts,
        x=source_column,
        y="pct",
        color="topic_name",
        barmode="stack",
        title=title,
        labels={
            "pct": "Porcentaje (%)",
            source_column: "Fuente",
            "topic_name": "Tópico",
        },
        color_discrete_sequence=_td_colors(palette, tmp["topic_name"].nunique()),
    )

    fig.update_yaxes(range=[0, 100])
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"]),
    )

    _td_save_html(fig, output_html)

    return fig


def plot_topic_keywords_table(
    keywords: Dict[Any, Any],
    output_html: Path,
    top_n: int = 10,
    title: str = "Palabras clave por tópico",
    palette: Any = None,
) -> Optional[go.Figure]:
    """
    Tabla HTML con keywords por tópico.
    """
    pal = _td_palette(palette)
    rows = []

    for topic_id, terms in keywords.items():
        normalized_terms = _td_normalize_terms(terms)

        if not normalized_terms:
            continue

        rows.append(
            {
                "Tópico": str(topic_id),
                "Keywords": ", ".join(normalized_terms[:top_n]),
            }
        )

    if not rows:
        return None

    table_df = pd.DataFrame(rows)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table_df.columns),
                    align="left",
                    fill_color=pal["primary"],
                    font=dict(color="white"),
                ),
                cells=dict(
                    values=[table_df[col] for col in table_df.columns],
                    align="left",
                    fill_color=pal["background"],
                    font=dict(color=pal["text"]),
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=max(250, 45 * len(table_df)),
    )

    _td_save_html(fig, output_html)

    return fig


def plot_top_terms_bar(
    terms: Sequence[Any],
    title: str,
    output_html: Path,
    palette: Any = "Plotly",
    top_n: int = 20,
) -> Optional[go.Figure]:
    """
    Barras horizontales para términos más frecuentes.

    `terms` puede ser:
      - [("palabra", conteo), ...]
      - ["palabra", ...]
    """
    rows = []

    for idx, item in enumerate(terms[:top_n]):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            term = str(item[0])
            count = float(item[1])
        else:
            term = str(item)
            count = float(top_n - idx)

        rows.append({"term": term, "count": count})

    if not rows:
        return None

    pal = _td_palette(palette)
    terms_df = pd.DataFrame(rows)
    terms_df = terms_df.sort_values("count", ascending=True).tail(top_n)

    fig = px.bar(
        terms_df,
        x="count",
        y="term",
        orientation="h",
        title=title,
        labels={
            "count": "Frecuencia",
            "term": "Término",
        },
        color_discrete_sequence=_td_colors(palette, 1),
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"]),
    )

    _td_save_html(fig, output_html)

    return fig


def plot_topic_sentiment_sunburst(
    topics_df: pd.DataFrame,
    source_column: str,
    output_html: Path,
    palette: Any = "Plotly",
    top_topics: int = 12,
    title: str = "Fuente -> Sentimiento -> Tópico",
) -> Optional[go.Figure]:
    """
    Sunburst para explorar:
      fuente -> sentimiento -> tópico
    """
    if topics_df.empty:
        return None

    required = [source_column, "sentiment_label", "topic_name"]

    for col in required:
        if col not in topics_df.columns:
            return None

    pal = _td_palette(palette)
    tmp = topics_df.copy()

    for col in required:
        tmp[col] = tmp[col].astype(str)

    top_topic_names = tmp["topic_name"].value_counts().head(top_topics).index
    tmp["topic_name"] = tmp["topic_name"].where(
        tmp["topic_name"].isin(top_topic_names),
        "Otros",
    )

    fig = px.sunburst(
        tmp,
        path=[source_column, "sentiment_label", "topic_name"],
        title=title,
        color_discrete_sequence=_td_colors(palette, tmp[source_column].nunique()),
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"]),
    )

    _td_save_html(fig, output_html)

    return fig


def plot_dmo_vs_perceived_topic_gap(
    topics_df: pd.DataFrame,
    source_column: str,
    dmo_label: str,
    output_html: Path,
    palette: Any = "Plotly",
    min_docs: int = 5,
    title: str = "Brecha de tópicos: DMO vs percibido",
) -> Optional[go.Figure]:
    """
    Compara la proporción de tópicos en DMO contra el resto de fuentes.

    gap_pct = DMO_pct - perceived_pct

    Si gap_pct > 0:
      La DMO menciona más ese tópico que el agregado percibido.

    Si gap_pct < 0:
      El agregado percibido menciona más ese tópico que la DMO.
    """
    if topics_df.empty or "topic_name" not in topics_df.columns:
        return None

    pal = _td_palette(palette)
    tmp = topics_df.copy()
    tmp[source_column] = tmp[source_column].astype(str)
    tmp["topic_name"] = tmp["topic_name"].astype(str)

    dmo_df = tmp[tmp[source_column] == str(dmo_label)]
    perceived_df = tmp[tmp[source_column] != str(dmo_label)]

    if len(dmo_df) < min_docs or len(perceived_df) < min_docs:
        return None

    dmo_share = dmo_df["topic_name"].value_counts(normalize=True) * 100
    perceived_share = perceived_df["topic_name"].value_counts(normalize=True) * 100

    gap = pd.DataFrame(
        {
            "dmo_pct": dmo_share,
            "perceived_pct": perceived_share,
        }
    ).fillna(0.0)

    gap["gap_pct"] = gap["dmo_pct"] - gap["perceived_pct"]

    gap = (
        gap.reset_index()
        .rename(columns={"index": "topic_name"})
        .sort_values("gap_pct", ascending=True)
    )

    fig = px.bar(
        gap,
        x="gap_pct",
        y="topic_name",
        orientation="h",
        color="gap_pct",
        color_continuous_scale=[
            pal["negative"],
            pal["background"],
            pal["positive"],
        ],
        title=title,
        labels={
            "gap_pct": "DMO - Percibido (%)",
            "topic_name": "Tópico",
        },
    )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"]),
        height=max(350, 30 * len(gap)),
    )

    _td_save_html(fig, output_html)

    return fig


def plot_topic_selection_curve(
    selection: Dict[str, Any],
    output_html: Path,
    palette: Any = None,
    sentiment_label: str = "",
) -> Optional[go.Figure]:
    """
    Curva de selección de k: puntaje compuesto y métricas de calidad.

    La evidencia se guarda como HTML con la paleta elegida por el usuario.
    """
    candidates = selection.get("candidates", [])
    if not candidates:
        return None

    from plotly.subplots import make_subplots

    pal = _td_palette(palette)
    rows = sorted(candidates, key=lambda item: item.get("k", 0))
    df = pd.DataFrame(rows)
    selected_k = int(selection.get("selected_k", df["k"].max()))
    accent = (
        pal["positive"]
        if str(sentiment_label).lower().startswith("pos")
        else pal["negative"]
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
        subplot_titles=(
            "Puntaje compuesto (mayor es mejor)",
            "Métricas de calidad (mayor es mejor, salvo redundancia)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df["k"],
            y=df["score"],
            mode="lines+markers",
            name="Puntaje",
            line=dict(color=pal["primary"], width=2),
            marker=dict(color=pal["primary"], size=7),
            hovertemplate="k=%{x}<br>Puntaje=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    selected_row = df[df["k"] == selected_k]
    if not selected_row.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_row["k"],
                y=selected_row["score"],
                mode="markers",
                name="Seleccionado",
                marker=dict(color=accent, size=15, symbol="star"),
                hovertemplate="k=%{x}<br>Puntaje=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    metrics = [
        ("coherence", "Coherencia NPMI"),
        ("diversity", "Diversidad"),
        ("stability", "Estabilidad ARI"),
        ("redundancy", "Redundancia"),
        ("balance", "Balance"),
    ]
    colors = _td_colors(palette, n=len(metrics))
    for (column, label), color in zip(metrics, colors):
        if column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["k"],
                y=df[column],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=1.8),
                marker=dict(color=color, size=5),
                connectgaps=False,
            ),
            row=2,
            col=1,
        )

    for row in (1, 2):
        fig.add_vline(
            x=selected_k,
            line_dash="dash",
            line_color=accent,
            line_width=1.5,
            row=row,
            col=1,
        )

    reason = str(selection.get("reason", ""))
    if len(reason) > 520:
        reason = reason[:517] + "..."
    fig.add_annotation(
        text=reason,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.24,
        showarrow=False,
        align="left",
        font=dict(color=pal["text"], size=11),
    )
    fig.update_layout(
        title=(
            f"Selección del número de tópicos – {sentiment_label or 'global'} "
            f"(k={selected_k})"
        ),
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"], family="Arial, sans-serif"),
        legend=dict(orientation="h", y=1.08, x=0),
        height=720,
        margin=dict(b=170),
    )
    fig.update_xaxes(title_text="Número de tópicos (k)", row=2, col=1)

    _td_save_html(fig, output_html)

    return fig


    # ═══════════════════════════════════════════════════════════
# NUEVAS FUNCIONES PARA EL PIPELINE
# ═══════════════════════════════════════════════════════════

def plot_ngramas_outliers(
    unigramas: list,
    bigramas: list,
    trigramas: list,
    output_dir: str = "results/plots",
    palette_name: str = "viridis",
) -> None:
    """Genera 3 gráficos 3D (tallos + marcadores) para n-gramas de outliers."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("unigramas", unigramas),
        ("bigramas", bigramas),
        ("trigramas", trigramas),
    ]

    for name, terms in datasets:
        if not terms:
            continue

        df_g = pd.DataFrame(terms, columns=["ngrama", "freq"])
        df_g["rank"] = np.arange(len(df_g))

        fig = go.Figure()

        # Tallos verticales (efecto barra 3D): de z=0 hasta z=freq
        for _, row in df_g.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row["rank"], row["rank"]],
                y=[0, 0],
                z=[0, row["freq"]],
                mode="lines",
                line=dict(color="rgba(120,120,120,0.35)", width=6),
                showlegend=False,
                hoverinfo="skip",
            ))

        # Marcadores en la punta, coloreados por frecuencia
        fig.add_trace(go.Scatter3d(
            x=df_g["rank"],
            y=np.zeros(len(df_g)),
            z=df_g["freq"],
            mode="markers",
            marker=dict(
                size=10,
                color=df_g["freq"],
                colorscale=palette_name,
                colorbar=dict(title="Frecuencia"),
                line=dict(width=0.5, color="white"),
            ),
            text=df_g["ngrama"],
            hovertemplate="<b>%{text}</b><br>Frecuencia: %{z}<extra></extra>",
            name=name,
        ))

        # Etiquetas de los n-gramas sobre el eje X
        fig.update_layout(
            title=f"Top {len(df_g)} {name.capitalize()} de Outliers (3D)",
            template="plotly_white",
            height=600,
            scene=dict(
                xaxis=dict(
                    title="Ranking",
                    tickvals=df_g["rank"].tolist(),
                    ticktext=df_g["ngrama"].tolist(),
                    tickangle=-45,
                ),
                yaxis=dict(title="", showticklabels=False, range=[-1, 1]),
                zaxis=dict(title="Frecuencia"),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            ),
        )
        fig.write_html(str(out / f"outlier_{name}_3d.html"), include_plotlyjs="cdn")

def plot_sentiment_by_source_grouped(
    df: pd.DataFrame,
    source_column: str,
    output_dir: str = "results/sentiment",
    title: str = "Sentimiento por Fuente",
) -> None:
    """Gráfico 3D: eje X = fuente, eje Y = sentimiento, eje Z = cantidad."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tmp = df.copy()
    tmp["sentiment_label"] = tmp["sentiment_label"].astype(str)
    counts = tmp.groupby([source_column, "sentiment_label"]).size().reset_index(name="n")

    # Mapeo de categóricos a numéricos para ejes 3D
    fuentes = sorted(counts[source_column].unique().tolist())
    sentimientos = sorted(counts["sentiment_label"].unique().tolist())
    fuente_idx = {f: i for i, f in enumerate(fuentes)}
    sent_idx = {s: i for i, s in enumerate(sentimientos)}

    counts["x"] = counts[source_column].map(fuente_idx)
    counts["y"] = counts["sentiment_label"].map(sent_idx)

    color_map = {"positive": "#009E73", "negative": "#D55E00"}

    fig = go.Figure()

    # Tallos desde z=0
    for _, row in counts.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row["x"], row["x"]],
            y=[row["y"], row["y"]],
            z=[0, row["n"]],
            mode="lines",
            line=dict(color="rgba(120,120,120,0.35)", width=6),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Marcadores por sentimiento
    for sent in sentimientos:
        sub = counts[counts["sentiment_label"] == sent]
        fig.add_trace(go.Scatter3d(
            x=sub["x"],
            y=sub["y"],
            z=sub["n"],
            mode="markers",
            marker=dict(
                size=12,
                color=color_map.get(sent, "#56B4E9"),
                opacity=0.9,
                line=dict(width=0.5, color="white"),
            ),
            name=sent,
            text=sub[source_column],
            hovertemplate=(
                f"<b>{sent}</b><br>Fuente: %{{text}}<br>"
                f"Cantidad: %{{z}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"{title} (3D)",
        template="plotly_white",
        height=600,
        scene=dict(
            xaxis=dict(
                title="Fuente",
                tickvals=list(fuente_idx.values()),
                ticktext=list(fuente_idx.keys()),
                tickangle=-30,
            ),
            yaxis=dict(
                title="Sentimiento",
                tickvals=list(sent_idx.values()),
                ticktext=list(sent_idx.keys()),
            ),
            zaxis=dict(title="Cantidad"),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=1.0)),
        ),
        legend=dict(title="Sentimiento"),
    )
    fig.write_html(str(out / "sentiment_grouped_by_source_3d.html"), include_plotlyjs="cdn")


def plot_embeddings_comparison(
    df: pd.DataFrame,
    emb_bert: np.ndarray,
    emb_xidoo: np.ndarray,
    source_column: str,
    dmo_label: str,
    output_dir: str = "results/embeddings_comparison",
) -> None:
    """Comparativa BERT vs XidooTopic en espacio semántico 3D (UMAP)."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tmp = df.copy()
    tmp["is_dmo"] = tmp[source_column].astype(str).apply(
        lambda x: "DMO" if x == str(dmo_label) else "Percibido"
    )

    # ── UMAP a 3 componentes ──
    umap_b = _reducir_umap_3d(emb_bert, n_components=3)
    umap_x = _reducir_umap_3d(emb_xidoo, n_components=3)

    fuentes = tmp[source_column].unique().tolist()
    n_f = len(fuentes)

    fig = go.Figure()

    # ── Traza BERT (visible por defecto) ──
    for src in fuentes:
        mask = (tmp[source_column] == src).values
        is_dmo = tmp.loc[mask, "is_dmo"].iloc[0] if mask.any() else "Percibido"
        fig.add_trace(go.Scatter3d(
            x=umap_b[mask, 0],
            y=umap_b[mask, 1],
            z=umap_b[mask, 2],
            mode="markers",
            name=f"BERT – {src}",
            marker=dict(
                size=4,
                opacity=0.65,
                symbol="diamond" if is_dmo == "DMO" else "circle",
            ),
            legendgroup="bert",
            legendgrouptitle_text="BERT",
            visible=True,
        ))

    # ── Traza XidooTopic (oculta por defecto) ──
    for src in fuentes:
        mask = (tmp[source_column] == src).values
        is_dmo = tmp.loc[mask, "is_dmo"].iloc[0] if mask.any() else "Percibido"
        fig.add_trace(go.Scatter3d(
            x=umap_x[mask, 0],
            y=umap_x[mask, 1],
            z=umap_x[mask, 2],
            mode="markers",
            name=f"Xidoo – {src}",
            marker=dict(
                size=4,
                opacity=0.65,
                symbol="diamond" if is_dmo == "DMO" else "x",
            ),
            legendgroup="xidoo",
            legendgrouptitle_text="XidooTopic",
            visible=False,
        ))

    # ── Botones para alternar BERT / XidooTopic ──
    fig.update_layout(
        title="BERT vs XidooTopic – Espacio semántico 3D (UMAP)",
        template="plotly_white",
        height=700,
        scene=dict(
            xaxis=dict(title="UMAP 1"),
            yaxis=dict(title="UMAP 2"),
            zaxis=dict(title="UMAP 3"),
        ),
        updatemenus=[dict(
            buttons=[
                dict(
                    label="BERT",
                    method="update",
                    args=[{"visible": [True] * n_f + [False] * n_f}],
                ),
                dict(
                    label="XidooTopic",
                    method="update",
                    args=[{"visible": [False] * n_f + [True] * n_f}],
                ),
            ],
            direction="down",
            x=0, y=1.15,
        )],
    )
    fig.write_html(str(out / "bert_vs_xidoo_interactive_3d.html"), include_plotlyjs="cdn")


def plot_topics_3d_umap(
    topics_df: pd.DataFrame,
    embeddings_dict: Dict[str, np.ndarray],
    text_column: str,
    source_column: str,
    output_dir: str = "results/topics",
    palette: Optional[Dict[str, str]] = None,
    ) -> None:

    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if topics_df.empty:
        print("⚠️ topics_df está vacío, no se puede graficar")
        return

    pal = _td_palette(palette)

    # Verificar que tenemos embeddings
    if not embeddings_dict or "positive" not in embeddings_dict:
        print("⚠️ No hay embeddings disponibles para visualización 3D")
        return

    print(f"\n  🎨 Generando visualización 3D de tópicos...")

    # Reducir dimensionalidad con UMAP
    try:
        import umap
        
        # Combinar todos los embeddings
        all_embeddings = []
        all_labels = []
        all_topics = []
        all_sources = []
        all_sentiments = []
        
        for sentiment in ["positive", "negative"]:
            if sentiment not in embeddings_dict:
                continue
            
            emb = embeddings_dict[sentiment]
            mask = topics_df["sentiment_label"] == sentiment
            
            if mask.sum() != len(emb):
                print(f"⚠️ Desajuste en embeddings de {sentiment}: "
                      f"{mask.sum()} vs {len(emb)}")
                continue
            
            all_embeddings.append(emb)
            all_labels.extend(topics_df.loc[mask, "topic_id"].astype(str).tolist())
            all_topics.extend(topics_df.loc[mask, "topic_name"].tolist())
            all_sources.extend(topics_df.loc[mask, source_column].tolist())
            all_sentiments.extend([sentiment] * len(emb))
        
        if not all_embeddings:
            print("⚠️ No hay embeddings válidos para graficar")
            return
        
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"     Aplicando UMAP a {len(all_embeddings)} puntos...")
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15)
        coords = reducer.fit_transform(all_embeddings)
        
        # Crear DataFrame para plotear
        df_plot = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "topic_id": all_labels,
            "topic_name": all_topics,
            "source": all_sources,
            "sentiment": all_sentiments,
        })
        
        # Crear figura 3D
        fig = go.Figure()
        
        color_pos = pal["positive"]
        color_neg = pal["negative"]

        # Agregar puntos positivos (verde)
        pos_mask = df_plot["sentiment"] == "positive"
        fig.add_trace(go.Scatter3d(
            x=df_plot.loc[pos_mask, "x"],
            y=df_plot.loc[pos_mask, "y"],
            z=df_plot.loc[pos_mask, "z"],
            mode="markers",
            marker=dict(
                size=4,
                color=color_pos,  # Verde
                opacity=0.6,
            ),
            name="Positivo",
            text=df_plot.loc[pos_mask, ["topic_name", "source"]].apply(
                lambda x: f"{x['topic_name']}<br>Fuente: {x['source']}", axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
        ))
        
        # Agregar puntos negativos (rojo)
        neg_mask = df_plot["sentiment"] == "negative"
        fig.add_trace(go.Scatter3d(
            x=df_plot.loc[neg_mask, "x"],
            y=df_plot.loc[neg_mask, "y"],
            z=df_plot.loc[neg_mask, "z"],
            mode="markers",
            marker=dict(
                size=4,
                color=color_neg,  # Rojo
                opacity=0.6,
            ),
            name="Negativo",
            text=df_plot.loc[neg_mask, ["topic_name", "source"]].apply(
                lambda x: f"{x['topic_name']}<br>Fuente: {x['source']}", axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
        ))
        
        fig.update_layout(
            title="Espacio semántico de tópicos (UMAP 3D)",
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
            width=1000,
            height=800,
            legend_title="Sentimiento",
            template="plotly_white",
            plot_bgcolor=pal["background"],
            paper_bgcolor=pal["background"],
            font=dict(color=pal["text"]),
        )
        
        html_path = out / "topics_3d_umap.html"
        fig.write_html(str(html_path))
        print(f"     ✅ Visualización 3D guardada: {html_path}")
        
    except ImportError:
        print("⚠️ UMAP no está instalado. Instala con: pip install umap-learn")
    except Exception as e:
        print(f"❌ Error en visualización 3D: {e}")


def plot_topic_distribution_by_source_3d(
    topics_df: pd.DataFrame,
    source_column: str,
    output_dir: str = "results/topics",
    palette: Optional[Dict[str, str]] = None,
) -> None:
    """
    Gráfico 3D de distribución de tópicos por fuente.
    Ejes: Fuente × Tópico × Sentimiento
    """
    from pathlib import Path
    import plotly.graph_objects as go
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if topics_df.empty or "topic_name" not in topics_df:
        return

    pal = _td_palette(palette)
    print(f"\n  📊 Generando distribución 3D de tópicos por fuente...")

    # Agrupar por fuente, tópico y sentimiento
    grouped = (
        topics_df.groupby([source_column, "topic_name", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    # Mapear a coordenadas numéricas
    sources = sorted(grouped[source_column].unique())
    topics = sorted(grouped["topic_name"].unique())
    sentiments = ["positive", "negative"]
    
    source_idx = {s: i for i, s in enumerate(sources)}
    topic_idx = {t: i for i, t in enumerate(topics)}
    sentiment_idx = {"positive": 0, "negative": 1}
    
    grouped["x"] = grouped[source_column].map(source_idx)
    grouped["y"] = grouped["topic_name"].map(topic_idx)
    grouped["z"] = grouped["count"]
    
    # Crear figura 3D
    fig = go.Figure()
    
    for sentiment in sentiments:
        subset = grouped[grouped["sentiment_label"] == sentiment]
        if sentiment == "positive":
            color = pal["positive"]
        else:
            color = pal["negative"]
        
        fig.add_trace(go.Scatter3d(
            x=subset["x"],
            y=subset["y"],
            z=subset["z"],
            mode="markers",
            marker=dict(
                size=subset["count"] / subset["count"].max() * 20 + 5,
                color=color,
                opacity=0.7,
            ),
            name=sentiment.capitalize(),
            text=subset.apply(
                lambda r: f"{r[source_column]}<br>{r['topic_name']}<br>Count: {r['count']}",
                axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
        ))
    
    fig.update_layout(
        title="Distribución de tópicos por fuente (3D)",
        scene=dict(
            xaxis=dict(
                title="Fuente",
                tickvals=list(source_idx.values()),
                ticktext=list(source_idx.keys()),
            ),
            yaxis=dict(
                title="Tópico",
                tickvals=list(topic_idx.values()),
                ticktext=list(topic_idx.keys()),
            ),
            zaxis_title="Cantidad",
        ),
        width=1000,
        height=800,
        template="plotly_white",
        plot_bgcolor=pal["background"],
        paper_bgcolor=pal["background"],
        font=dict(color=pal["text"]),
    )
    
    html_path = out / "topic_distribution_3d.html"
    fig.write_html(str(html_path))
    print(f"     ✅ Distribución 3D guardada: {html_path}")