"""
sentiment_wordclouds.py
─────────────────────────────────────────────────────────────
Genera 4 wordclouds de sentimiento:
  • Positivos / Negativos  ×  Comentarios normales / Outliers
Además imprime en consola los top-80 unigramas por grupo
con su frecuencia y fuente predominante.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ══════════════════════════════════════════════
#  STOPWORDS BÁSICAS (ES / EN)
# ══════════════════════════════════════════════
STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "en", "con", "por", "para", "que", "qué", "se", "es", "son", "fue",
    "ser", "muy", "no", "si", "sí", "y", "o", "u", "pero", "más", "mas",
    "me", "mi", "mis", "te", "tu", "tus", "le", "les", "lo", "su", "sus",
    "este", "esta", "esto", "estos", "estas", "ese", "esa", "eso", "esos",
    "esas", "aquel", "aquella", "hay", "he", "ha", "han", "todo", "toda",
    "todos", "todas", "como", "cuando", "donde", "quien", "cual", "ya",
    "también", "asi", "así", "muy", "tan", "solo", "sólo", "sobre", "entre",
    "sin", "desde", "hasta", "durante", "según", "mismo", "misma", "cada",
    "otro", "otra", "otros", "otras", "poco", "mucho", "bastante", "algo",
    "nada", "bien", "mal", "mejor", "peor", "bueno", "buena", "buen",
    "grande", "pequeño", "puede", "pueden", "hacer", "hace", "hizo",
    "tiene", "tienen", "estar", "ir", "va", "van", "fui", "fuimos",
    "hotel", "lugar", "viaje", "vez", "día", "días", "noche", "parte",
    "cosa", "cosas", "persona", "personas", "tiempo", "forma", "modo",
}

STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "this", "that",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "they", "them", "their", "what", "which", "who",
    "when", "where", "how", "not", "no", "so", "very", "just", "about",
    "also", "than", "then", "too", "here", "there", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "good", "great", "nice", "bad", "well", "really", "much",
}

# ══════════════════════════════════════════════
#  UTILIDADES DE PREPROCESAMIENTO
# ══════════════════════════════════════════════
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_PUNCT_RE = re.compile(r"[^\w\sñáéíóúü]", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\d+")


def _limpiar_texto(texto: str) -> str:
    """Limpieza mínima para extracción de unigramas."""
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _DIGIT_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    return t


def _tokenizar(texto: str, stopwords: set, min_len: int = 3) -> List[str]:
    tokens = texto.split()
    return [t for t in tokens if len(t) >= min_len and t not in stopwords]


# ══════════════════════════════════════════════
#  WORDCLOUD (tu función original, reutilizada)
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
    """Genera una wordcloud a partir de [(palabra, frecuencia), ...]."""
    if not ngramas:
        print(f"  ⚠️  No hay datos para wordcloud: {titulo}")
        return output_path / f"{titulo}.png"

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

    safe = titulo.lower().replace(" ", "_").replace("—", "_").replace("(", "").replace(")", "")
    out_file = output_path / f"{safe}.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_file


# ══════════════════════════════════════════════
#  EXTRACCIÓN DE UNIGRAMAS POR GRUPO
# ══════════════════════════════════════════════
def _extraer_unigramas(
    df: pd.DataFrame,
    text_column: str,
    source_column: str,
    lang: str = "es",
) -> Tuple[List[Tuple[str, int]], Dict[str, Counter]]:
    """
    Devuelve:
      - lista [(palabra, freq_global)] ordenada desc
      - dict {palabra: Counter({fuente: conteo})}
    """
    stopwords = STOPWORDS_ES if lang.startswith("es") else STOPWORDS_EN
    freq_global: Counter = Counter()
    freq_por_fuente: Dict[str, Counter] = {}

    for _, row in df.iterrows():
        texto_limpio = _limpiar_texto(row[text_column])
        tokens = _tokenizar(texto_limpio, stopwords)
        fuente = str(row[source_column])

        for tok in tokens:
            freq_global[tok] += 1
            if tok not in freq_por_fuente:
                freq_por_fuente[tok] = Counter()
            freq_por_fuente[tok][fuente] += 1

    top = freq_global.most_common()
    return top, freq_por_fuente


# ══════════════════════════════════════════════
#  REPORTE EN CONSOLA
# ══════════════════════════════════════════════
def _imprimir_top_unigramas(
    top: List[Tuple[str, int]],
    freq_por_fuente: Dict[str, Counter],
    titulo: str,
    top_n: int = 80,
):
    SEP = "─" * 78
    print(f"\n  📝  TOP-{top_n} UNIGRAMAS — {titulo}")
    print(f"  {SEP}")
    print(f"  {'#':>3s}  {'Palabra':<22s} {'Frec.':>8s}  {'Fuente predominante':<22s} {'%':>6s}")
    print(f"  {SEP}")

    for i, (palabra, freq) in enumerate(top[:top_n], start=1):
        contador_fuentes = freq_por_fuente.get(palabra, Counter())
        if contador_fuentes:
            fuente_top, cnt_top = contador_fuentes.most_common(1)[0]
            pct = cnt_top / freq * 100
        else:
            fuente_top, pct = "—", 0.0

        print(
            f"  {i:>3d}  {palabra:<22s} {freq:>8,d}  "
            f"{fuente_top:<22s} {pct:>5.1f}%"
        )
    print(f"  {SEP}\n")


# ══════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════
def _colormap_from_colors(dark: str, light: str):
    """Crea un colormap de matplotlib a partir de dos colores de la paleta."""
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return "viridis"

    try:
        return LinearSegmentedColormap.from_list("paleta_pipeline", [light, dark])
    except Exception:
        return "viridis"


def generar_wordclouds_sentimiento(
    df_clean: pd.DataFrame,
    df_outliers: pd.DataFrame,
    text_column: str,
    source_column: str,
    output_dir: str = "results/sentiment/wordclouds",
    lang: str = "es",
    colormap_pos: str = "Greens",
    colormap_neg: str = "Reds",
    top_n: int = 80,
    palette: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Genera 4 wordclouds + imprime top-80 unigramas en consola.
    Devuelve un diccionario con las rutas de los PNG generados.
    """
    out = Path(output_dir)
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
    ]

    for titulo, df, label, cmap in grupos:
        print(f"\n{'═'*78}")
        print(f"  🌥️   WORDCLOUD: {titulo.upper()}")
        print(f"{'═'*78}")

        sub = df[df["sentiment_label"] == label].copy()
        print(f"  Comentarios en este grupo: {len(sub):,}")

        if sub.empty:
            print(f"  ⚠️  No hay comentarios '{label}' para generar wordcloud.")
            resultados[titulo] = None
            continue

        top, freq_fuente = _extraer_unigramas(sub, text_column, source_column, lang)
        _imprimir_top_unigramas(top, freq_fuente, titulo, top_n=top_n)

        png = generar_wordcloud_ngramas(
            ngramas=top,
            titulo=titulo,
            output_path=out,
            colormap=cmap,
            max_words=150,
        )
        resultados[titulo] = png
        print(f"  ✅ Wordcloud guardada: {png}")

    # ── Resumen final ──
    print(f"\n{'═'*78}")
    print(f"  📊  RESUMEN WORDCLOUDS DE SENTIMIENTO")
    print(f"{'═'*78}")
    for k, v in resultados.items():
        estado = str(v) if v else "⚠️  sin datos"
        print(f"  • {k:<32s} → {estado}")
    print(f"{'═'*78}\n")

    return resultados