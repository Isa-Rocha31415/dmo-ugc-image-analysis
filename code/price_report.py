"""price_report.py — artefactos, figuras y sección HTML del análisis Precio/Valor/Costo.

Funciones puras (sin efectos secundarios) para construir las tablas canónicas y
las dos figuras interactivas. Las funciones ``write_*`` hacen la única E/S.
Todo el material se alimenta de scores ya calculados y de embeddings cacheados:
no se reentrena ningún modelo ni se recalcula sentimiento/tópicos.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale, sequential

from comparison.common import append_metadata, palette_colors, print_header, save_csv
from io_corpus import CORPUS_COLUMN, SOURCE_COLUMN
from metrics_lexical import logodds_z, tokenize
from utils import palette_to_plot_colors

PRICE_SUBDIR = "price"
TOP_K = 5
DENSITY_BINS = 40
JACKKNIFED_K = 10
SMALL_SAMPLE_MIN = 10


# ─────────────────────────── Helpers puros ───────────────────────────
def _escape(text: Any, limit: int = 200) -> str:
    """Trunca y escapa HTML para customdata de Plotly (función pura)."""
    value = str(text) if text is not None else ""
    if limit and len(value) > limit:
        value = value[:limit] + "…"
    return html.escape(value, quote=True)


def _is_dmo(frame: pd.DataFrame) -> np.ndarray:
    """Máscara DMO sobre una columna de corpus (función pura)."""
    if CORPUS_COLUMN not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    return frame[CORPUS_COLUMN].astype(str).str.upper().to_numpy() == "DMO"


def _source_label(frame: pd.DataFrame) -> np.ndarray:
    """Etiqueta de fuente legible: DMO o el nombre de plataforma del UGC."""
    dmo = _is_dmo(frame)
    src = (frame[SOURCE_COLUMN].astype(str).to_numpy()
           if SOURCE_COLUMN in frame.columns
           else np.array(["UGC"] * len(frame), dtype=object))
    return np.where(dmo, "DMO", np.where(src == "", "percibido", src))


def _aligned_embeddings(scores: pd.DataFrame,
                        embeddings: np.ndarray | None) -> np.ndarray | None:
    """Alinea la matriz cacheada a los doc_id de scores usando embedding_row."""
    if embeddings is None:
        return None
    emb = np.asarray(embeddings, dtype=float)
    if "embedding_row" in scores.columns and emb.ndim == 2:
        rows = scores["embedding_row"].astype(int).to_numpy()
        if len(rows) and rows.max() < len(emb):
            return emb[rows]
    return emb if emb.shape[0] == len(scores) else None


def _color_scale(n: int) -> list[str]:
    """n colores de una escala continua accesible (Cividis)."""
    if n <= 1:
        return [sequential.Cividis[3]]
    return sample_colorscale(sequential.Cividis, [i / (n - 1) for i in range(n)])


def enrich_scores(
    scores: pd.DataFrame,
    doc_topics: pd.DataFrame | None,
    tau: float,
) -> pd.DataFrame:
    """Añade topic_id/topic_name y los alias exigidos por el enunciado."""
    out = scores.copy().reset_index(drop=True)
    if doc_topics is not None and not doc_topics.empty and "doc_id" in doc_topics:
        cols = [c for c in ("doc_id", "topic_id", "topic_name")
                if c in doc_topics.columns]
        # Idempotente: la corrida anterior ya pudo enriquecer price_scores.csv.
        out = out.drop(columns=[c for c in ("topic_id", "topic_name")
                                if c in out.columns])
        out = out.merge(doc_topics[cols].drop_duplicates("doc_id"),
                        on="doc_id", how="left")
    for col in ("topic_id", "topic_name"):
        if col not in out.columns:
            out[col] = "SIN_TÓPICO" if col == "topic_name" else -1
    out["topic_name"] = out["topic_name"].fillna("SIN_TÓPICO").astype(str)

    if "score_semantic" in out.columns:
        sem = out["score_semantic"].astype(float)
    elif "price_score" in out.columns:
        sem = out["price_score"].astype(float)
    else:
        raise KeyError("price_scores.csv sin score_semantic/price_score")
    if "score_centered" in out.columns:
        centered = out["score_centered"].astype(float)
    else:
        centered = sem
    resena = (out["sim_resena_centered"].astype(float)
              if "sim_resena_centered" in out.columns else sem)
    promo = (out["sim_promo_centered"].astype(float)
             if "sim_promo_centered" in out.columns else sem)
    out["cos_semantic"] = sem.round(6)
    out["cos_centered"] = centered.round(6)
    out["sim_resena"] = resena.round(6)
    out["sim_promocional"] = promo.round(6)
    out["registro_dominante"] = np.where(resena.to_numpy() >= promo.to_numpy(),
                                         "reseña", "promocional")
    out["es_mencion"] = (sem.to_numpy() >= float(tau)) if "above_tau" not in out \
        else out["above_tau"].astype(bool).to_numpy()
    out["tau"] = round(float(tau), 6)
    out["source_label"] = _source_label(out)
    return out


def density_table(scores: pd.DataFrame, bins: int = DENSITY_BINS) -> pd.DataFrame:
    """Histograma normalizado del coseno por fuente (área = 1 por fuente)."""
    values = scores["cos_semantic"].astype(float).to_numpy()
    finite = values[np.isfinite(values)]
    if not finite.size:
        return pd.DataFrame(columns=["source", "bin_low", "bin_high",
                                     "bin_center", "density", "count", "n_total"])
    edges = np.linspace(float(finite.min()), float(finite.max()), bins + 1)
    rows: list[dict] = []
    for label in sorted(scores["source_label"].astype(str).unique()):
        vals = scores.loc[scores["source_label"] == label, "cos_semantic"].dropna()
        counts, _ = np.histogram(vals, bins=edges)
        n = int(counts.sum())
        density = counts / n if n else np.zeros_like(counts, dtype=float)
        for i in range(bins):
            rows.append({
                "source": label,
                "bin_low": round(float(edges[i]), 6),
                "bin_high": round(float(edges[i + 1]), 6),
                "bin_center": round(float((edges[i] + edges[i + 1]) / 2), 6),
                "density": round(float(density[i]), 8),
                "count": int(counts[i]),
                "n_total": n,
            })
    return pd.DataFrame(rows)


def distribution_tests(density: pd.DataFrame) -> pd.DataFrame:
    """EMD/Wasserstein + JSD entre histogramas normalizados DMO vs UGC."""
    from scipy.stats import wasserstein_distance

    if density.empty:
        return pd.DataFrame(columns=[
            "scope", "group_a", "group_b", "emd_wasserstein", "jsd_base2",
            "n_a", "n_b", "method"])
    piv = density.pivot_table(index="bin_center", columns="source",
                              values="density", fill_value=0.0).sort_index()
    groups = list(piv.columns)
    dmo = [g for g in groups if str(g).upper() == "DMO"]
    others = [g for g in groups if str(g).upper() != "DMO"]
    pairs = [(d, "UGC_total") for d in dmo for _ in [0]]
    pairs += [(d, o) for d in dmo for o in others]
    rows: list[dict] = []
    centers = piv.index.to_numpy()
    step = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
    for a, b in pairs:
        if b == "UGC_total":
            ugc_cols = [c for c in others]
            if not ugc_cols:
                continue
            q = piv[ugc_cols].mean(axis=1).to_numpy()
            scope = "DMO_vs_UGC_total"
            n_b = int(density.loc[density["source"].isin(ugc_cols), "n_total"].max())
        elif b in piv.columns:
            q = piv[b].to_numpy()
            scope = f"DMO_vs_{b}"
            n_b = int(density.loc[density["source"] == b, "n_total"].max())
        else:
            continue
        p = piv[a].to_numpy()
        p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
        m = 0.5 * (p + q)
        def _kl(x: np.ndarray) -> float:
            mask = x > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = x[mask] * np.log2(x[mask] / m[mask])
            return float(np.nansum(terms))
        jsd = 0.5 * _kl(p) + 0.5 * _kl(q)
        a_vals = density.loc[density["source"] == a, "bin_center"].to_numpy()
        rows.append({
            "scope": scope,
            "group_a": a,
            "group_b": b,
            "emd_wasserstein": round(wasserstein_distance(
                a_vals, a_vals, u_weights=p, v_weights=q), 6),
            "jsd_base2": round(jsd, 6),
            "n_a": int(density.loc[density["source"] == a, "n_total"].max()),
            "n_b": n_b,
            "method": "histogramas normalizados + EMD/JSD (Díaz-Pacheco)",
        })
    return pd.DataFrame(rows)


def alignment_table(
    scores: pd.DataFrame,
    embeddings: np.ndarray | None = None,
    anchor_centroid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Media/mediana de coseno por fuente, coseno de centroides y gaps."""
    rows: list[dict] = []
    dmo = _is_dmo(scores)
    cos = scores["cos_semantic"].astype(float).to_numpy()
    mention = scores["es_mencion"].astype(bool).to_numpy()
    for label in sorted(scores["source_label"].astype(str).unique()):
        mask = scores["source_label"].astype(str).to_numpy() == label
        rows.append({
            "scope": label,
            "n_docs": int(mask.sum()),
            "n_mentions": int((mask & mention).sum()),
            "mean_cosine": round(float(np.nanmean(cos[mask])), 6),
            "median_cosine": round(float(np.nanmedian(cos[mask])), 6),
            "rate": round(float(mention[mask].mean()), 6) if mask.any() else np.nan,
        })
    table = pd.DataFrame(rows)
    r_dmo = table[table["scope"] == "DMO"]
    r_ugc = table[table["scope"] != "DMO"]
    rate_dmo = float(r_dmo["rate"].iloc[0]) if len(r_dmo) else np.nan
    # Agregación ponderada por n: la tasa global de UGC no es la media de tasas
    # de plataforma (tripadvisor domina el corpus).
    n_ugc = float(r_ugc["n_docs"].sum())
    rate_ugc = (float((r_ugc["n_docs"] * r_ugc["rate"]).sum() / n_ugc)
                if n_ugc else np.nan)
    mu_dmo = float(r_dmo["mean_cosine"].iloc[0]) if len(r_dmo) else np.nan
    mu_ugc = (float((r_ugc["n_docs"] * r_ugc["mean_cosine"]).sum() / n_ugc)
              if n_ugc else np.nan)
    centroid_cos = np.nan
    if embeddings is not None and anchor_centroid is not None and mention.any():
        emb = np.asarray(embeddings, dtype=float)
        if "embedding_row" in scores.columns:
            idx = scores["embedding_row"].astype(int).to_numpy()
            if idx.max() < len(emb):
                emb = emb[idx]
        if emb.shape[0] == len(scores):
            c_dmo = emb[mention & dmo].mean(axis=0)
            c_ugc = emb[mention & ~dmo].mean(axis=0)
            if np.linalg.norm(c_dmo) and np.linalg.norm(c_ugc):
                centroid_cos = float(c_dmo @ c_ugc /
                                     (np.linalg.norm(c_dmo) * np.linalg.norm(c_ugc)))
    table.loc[len(table)] = {
        "scope": "DMO_vs_UGC",
        "n_docs": int(len(scores)),
        "n_mentions": int(mention.sum()),
        "mean_cosine": round(mu_dmo - mu_ugc, 6),
        "median_cosine": np.nan,
        "rate": round(rate_dmo - rate_ugc, 6),
    }
    table["centroid_cosine_mentions"] = np.nan
    table.loc[table["scope"] == "DMO_vs_UGC", "centroid_cosine_mentions"] = \
        round(centroid_cos, 6)
    table["centroid_cosine_note"] = ""
    table.loc[table["scope"] == "DMO_vs_UGC", "centroid_cosine_note"] = (
        "no computable: 0 menciones en alguno de los dos grupos"
        if pd.isna(centroid_cos) else "centroides L2 sobre docs que mencionan el concepto")
    table["gap_rate_pp"] = round(100 * (rate_dmo - rate_ugc), 4)
    table["gap_cosine"] = round(mu_dmo - mu_ugc, 6)
    table["anchor_centroid_used"] = anchor_centroid is not None
    return table


def valence_conditioned_table(scores: pd.DataFrame) -> pd.DataFrame:
    """POS/NEU/NEG entre menciones, por fuente (valencia condicionada)."""
    m = scores[scores["es_mencion"].astype(bool)].copy()
    if m.empty:
        return pd.DataFrame(columns=["source", "sentiment_label", "n",
                                     "share_within_source", "denominator"])
    rows: list[dict] = []
    for source, g in m.groupby("source_label", sort=True):
        den = len(g)
        for label, count in g["sentiment_label"].astype(str).value_counts().items():
            rows.append({
                "source": source,
                "sentiment_label": label,
                "n": int(count),
                "denominator": den,
                "share_within_source": round(count / den, 6),
            })
    return pd.DataFrame(rows).sort_values(["source", "sentiment_label"])


def context_logodds_table(scores: pd.DataFrame, top_k: int = 15,
                          alpha0: float = 1000.0) -> pd.DataFrame:
    """Top-k términos por fuente entre menciones (log-odds Monroe)."""
    m = scores[scores["es_mencion"].astype(bool)].copy()
    if m.empty:
        return pd.DataFrame(columns=["source", "term", "n_docs", "z",
                                     "denominator", "rank"])
    text_col = "text_norm" if "text_norm" in m.columns else "text"
    sources = sorted(m["source_label"].astype(str).unique())
    doc_terms = {
        s: [set(tokenize(str(t))) for t in
            m.loc[m["source_label"] == s, text_col].astype(str)]
        for s in sources
    }
    rows: list[dict] = []
    for s in sources:
        other = [d for o in sources if o != s for d in doc_terms[o]]
        vocab = sorted({term for d in doc_terms[s] for term in d} |
                       {term for d in other for term in d})
        fd = np.asarray([sum(term in d for d in doc_terms[s]) for term in vocab])
        fu = np.asarray([sum(term in d for d in other) for term in vocab])
        z = logodds_z(fd, fu, alpha0=alpha0)
        order = np.argsort(-np.abs(z))[:top_k]
        for rank, i in enumerate(order, 1):
            rows.append({
                "source": s,
                "rank": rank,
                "term": vocab[i],
                "n_docs": int(fd[i]),
                "z": round(float(z[i]), 4),
                "denominator": len(doc_terms[s]),
            })
    return pd.DataFrame(rows)


def topic_crosstab_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Tópico × fuente entre menciones, % normalizado dentro de cada fuente."""
    m = scores[scores["es_mencion"].astype(bool)].copy()
    if m.empty:
        return pd.DataFrame(columns=["topic_id", "topic_name", "source", "n",
                                     "denominator", "pct_within_source"])
    g = m.groupby(["topic_id", "topic_name", "source_label"], dropna=False).size()
    out = g.reset_index(name="n")
    out = out.rename(columns={"source_label": "source"})
    den = out.groupby("source")["n"].transform("sum")
    out["denominator"] = den.astype(int)
    out["pct_within_source"] = (out["n"] / den).round(6)
    return out.sort_values(["source", "pct_within_source"], ascending=[True, False])


def top_examples_table(scores: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Top-k global, DMO, UGC y por plataforma, con fuente visible."""
    cols = ["rank", "scope", "doc_id", SOURCE_COLUMN, CORPUS_COLUMN,
            "cos_semantic", "sentiment_label", "topic_name",
            "registro_dominante", "text_norm"]
    rows: list[dict] = []
    dmo = _is_dmo(scores)
    scopes: dict[str, np.ndarray] = {"global": np.ones(len(scores), dtype=bool),
                                     "dmo": dmo,
                                     "ugc": ~dmo}
    for src in sorted(scores.loc[~dmo, SOURCE_COLUMN].astype(str).unique()):
        scopes[src] = (~dmo) & (scores[SOURCE_COLUMN].astype(str) == src).to_numpy()
    for scope, mask in scopes.items():
        sub = scores.loc[mask].nlargest(top_k, "cos_semantic")
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            rows.append({
                "rank": rank,
                "scope": scope,
                "doc_id": r.get("doc_id"),
                SOURCE_COLUMN: r.get(SOURCE_COLUMN),
                CORPUS_COLUMN: r.get(CORPUS_COLUMN),
                "cos_semantic": round(float(r["cos_semantic"]), 6),
                "sentiment_label": r.get("sentiment_label", ""),
                "topic_name": r.get("topic_name", ""),
                "registro_dominante": r.get("registro_dominante", ""),
                "text_norm": r.get("text_norm", ""),
            })
    return pd.DataFrame(rows, columns=cols)


def enhance_comparison(rates: pd.DataFrame, tau: float,
                       fpr_controls: float, fpr_random: float,
                       seed: int) -> pd.DataFrame:
    """Añade n/denominador explícitos, test_used y razón a price_comparison."""
    out = rates.copy()
    if out.empty:
        return out
    reasons: list[str] = []
    tests: list[str] = []
    for _, r in out.iterrows():
        n1, k1 = int(r.get("n_docs_dmo", 0)), int(r.get("n_mentions_dmo", 0))
        n2, k2 = int(r.get("n_docs_ugc", 0)), int(r.get("n_mentions_ugc", 0))
        table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
        expected = (np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
                    if table.sum() else np.zeros((2, 2)))
        if table.sum() and (expected >= 5).all():
            tests.append("chi2")
            reasons.append(f"todos los esperados ≥5 (mín={expected.min():.2f})")
        else:
            tests.append("fisher")
            reasons.append(f"conteo esperado <5 (mín={expected.min():.2f}) → Fisher exacto")
        reasons[-1] += f"; p de '{r.get('test', '')}' recalculado por two_by_two_test"
        reasons[-1] += "; BH aplicado a la familia de scopes de price"
    out["test_used"] = tests
    out["test_reason"] = reasons
    out["tau"] = round(float(tau), 6)
    out["placebo_fpr_controls"] = round(float(fpr_controls), 6)
    out["placebo_fpr_random"] = round(float(fpr_random), 6)
    out["seed"] = int(seed)
    out["n_dmo_mentions"] = out["n_mentions_dmo"].astype(int)
    out["n_dmo_total"] = out["n_docs_dmo"].astype(int)
    out["n_ugc_mentions"] = out["n_mentions_ugc"].astype(int)
    out["n_ugc_total"] = out["n_docs_ugc"].astype(int)
    if "p_adj" in out.columns:
        out["p_adj_bh"] = out["p_adj"]
    return out


# ─────────────────────────── Figuras ───────────────────────────
def build_similarity_figure(scores: pd.DataFrame, tau: float,
                            palette: Any = None) -> go.Figure:
    """Scatter interactivo: un punto por comentario normal, 3 modos de color."""
    if palette is None:
        palette = palette_colors(palette_to_plot_colors("ejecutiva"))
    sources = sorted(scores["source_label"].astype(str).unique())
    topics = sorted(scores["topic_name"].astype(str).unique())
    sents = sorted(scores["sentiment_label"].astype(str).unique())
    src_colors = dict(zip(sources, _color_scale(max(len(sources), 2))))
    topic_colors = dict(zip(topics, _color_scale(max(len(topics), 2))))
    sent_colors = dict(zip(sents, _color_scale(max(len(sents), 2))))
    symbols = ["circle", "diamond", "square", "triangle-up", "cross", "star"]
    src_symbols = dict(zip(sources, symbols))

    fig = go.Figure()
    for src in sources:
        sub = scores[scores["source_label"].astype(str) == src]
        custom = [
            [str(r.doc_id), str(r.source_label), str(r.topic_name),
             str(r.sentiment_label), round(float(r.cos_semantic), 6),
             str(r.registro_dominante), _escape(r.text_norm, 200)]
            for r in sub.itertuples(index=False)
        ]
        fig.add_trace(go.Scattergl(
            x=sub["cos_semantic"].astype(float),
            y=sub["sentiment_score"].astype(float),
            mode="markers",
            name=src,
            customdata=custom,
            marker=dict(
                size=6, color=src_colors[src], symbol=src_symbols[src],
                opacity=0.7, line=dict(width=0),
            ),
            hovertemplate=(
                "<b>doc %{customdata[0]}</b><br>"
                "Fuente: %{customdata[1]}<br>"
                "Tópico: %{customdata[2]}<br>"
                "Sentimiento: %{customdata[3]}<br>"
                "Coseno: %{customdata[4]}<br>"
                "Registro: %{customdata[5]}<br>"
                "Texto: %{customdata[6]}<extra></extra>"
            ),
        ))
    def _per_trace(column: str, palette_map: dict[str, str],
                   use_symbols: bool) -> dict[str, list]:
        """Un array de colores por trace; los símbolos siguen codificando fuente."""
        colors: list[Any] = []
        symbols: list[Any] = []
        for src in sources:
            sub = scores.loc[scores["source_label"].astype(str) == src, column]
            colors.append([palette_map[str(v)] for v in sub.astype(str)])
            symbols.append([src_symbols[src]] * len(sub) if use_symbols
                           else [src_symbols[src]] * len(sub))
        return {"color": colors, "symbol": symbols}

    src_style = _per_trace("source_label", src_colors, True)
    topic_style = _per_trace("topic_name", topic_colors, True)
    sent_style = _per_trace("sentiment_label", sent_colors, True)
    fig.update_layout(
        title="Similitud semántica al concepto y tono (todos los comentarios normales)",
        xaxis_title="Coseno al concepto sintético",
        yaxis_title="sentiment_score",
        template="plotly_white",
        height=720,
        legend=dict(title="Fuente", orientation="h", y=1.08),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.0, xanchor="left",
            y=1.16, yanchor="top", showactive=True,
            buttons=[
                dict(label="Por fuente", method="update",
                     args=[src_style,
                           {"title.text": "Similitud — color y símbolo por fuente"}]),
                dict(label="Por tópico", method="update",
                     args=[topic_style,
                           {"title.text": "Similitud — color por tópico, símbolo por fuente"}]),
                dict(label="Por sentimiento", method="update",
                     args=[sent_style,
                           {"title.text": "Similitud — color por sentimiento, símbolo por fuente"}]),
            ],
        )],
    )
    fig.add_vline(
        x=float(tau), line_dash="dash", line_color="rgb(180,40,40)",
        annotation_text=f"umbral calibrado por placebo (τ={float(tau):.4f})",
        annotation_position="top left",
    )
    return fig


def build_umap_concept_figure(
    scores: pd.DataFrame,
    coords: np.ndarray,
    anchor_centroid: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    k: int = JACKKNIFED_K,
) -> go.Figure:
    """UMAP 3D reutilizando ``build_umap_3d_plot`` con color continuo de coseno.

    El modelo UMAP no se persiste: el concepto se sitúa en el baricentro
    ponderado de los k documentos más cercanos y se etiqueta como aproximación.
    """
    from comparison.common import l2_normalize
    from visualization import build_umap_3d_plot

    emb = _aligned_embeddings(scores, embeddings)
    if emb is None:
        emb = np.zeros((len(scores), 1))
    fig = build_umap_3d_plot(
        scores, emb,
        text_column="text_norm" if "text_norm" in scores.columns else "text",
        source_column="source_label",
        topic_column="topic_name",
        sentiment_column="sentiment_label",
        coords_3d=coords,
    )
    pts = np.asarray(coords, dtype=float)
    cos = scores["cos_semantic"].astype(float).to_numpy()
    topics = sorted(scores["topic_name"].astype(str).unique().tolist())
    for i, topic in enumerate(topics):
        if i >= len(fig.data):
            break
        mask = scores["topic_name"].astype(str).to_numpy() == topic
        trace = fig.data[i]
        trace.customdata = [
            [str(topic), str(src), str(sent), _escape(txt, 200), round(float(c), 6)]
            for src, sent, txt, c in zip(
                scores.loc[mask, "source_label"].astype(str),
                scores.loc[mask, "sentiment_label"].astype(str),
                scores.loc[mask, "text_norm"].astype(str), cos[mask])
        ]
        trace.marker.color = cos[mask].tolist()
        trace.marker.colorscale = sequential.Cividis
        trace.marker.size = 4
        trace.marker.opacity = 0.8
        if i == 0:
            trace.marker.colorbar = dict(title="coseno al concepto")
        trace.hovertemplate = (
            "<b>Tópico: %{customdata[0]}</b><br>"
            "Fuente: %{customdata[1]}<br>"
            "Sentimiento: %{customdata[2]}<br>"
            "Coseno: %{customdata[4]}<br>"
            "Texto: %{customdata[3]}<extra></extra>"
        )
    if anchor_centroid is not None and emb.shape[1] > 1:
        sims = l2_normalize(emb) @ l2_normalize(
            np.asarray(anchor_centroid, dtype=float).reshape(1, -1)).reshape(-1)
    else:
        sims = cos
    idx = np.argsort(-sims)[:min(k, len(sims))]
    weights = np.clip(np.asarray(sims[idx], dtype=float), 0, None) + 1e-6
    weights /= weights.sum()
    anchor_xyz = (pts[idx] * weights[:, None]).sum(axis=0)
    fig.add_trace(go.Scatter3d(
        x=[float(anchor_xyz[0])], y=[float(anchor_xyz[1])],
        z=[float(anchor_xyz[2])],
        mode="markers+text",
        name="posición aproximada del concepto (baricentro k-NN)",
        text=["concepto (k-NN)"],
        marker=dict(size=12, symbol="diamond", color="rgb(255,205,50)",
                    line=dict(color="black", width=2)),
        hovertemplate="Concepto sintético (baricentro k-NN)<extra></extra>",
    ))
    fig.update_layout(
        title="UMAP 3D del concepto: color continuo por coseno semántico",
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        legend=dict(orientation="h", y=1.02),
        template="plotly_white", height=820,
        margin=dict(l=0, r=0, b=0, t=80),
    )
    return fig


def build_density_figure(density: pd.DataFrame) -> go.Figure:
    """Histogramas de densidad normalizados, uno por fuente."""
    fig = go.Figure()
    colors = _color_scale(max(density["source"].nunique(), 2)) \
        if not density.empty else []
    for (src, g), color in zip(density.groupby("source"), colors):
        fig.add_trace(go.Scatter(
            x=g["bin_center"], y=g["density"], mode="lines",
            name=str(src), line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=color,
            hovertemplate=("coseno %{x:.3f}<br>densidad %{y:.4f}<br>"
                           f"fuente {html.escape(str(src))}<extra></extra>"),
        ))
    fig.update_layout(
        title="Densidad normalizada del coseno al concepto por fuente",
        xaxis_title="coseno al concepto", yaxis_title="densidad (área=1)",
        template="plotly_white", height=420,
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _df_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Tabla HTML escapada, con tope de filas."""
    if df is None or df.empty:
        return '<p class="muted">Sin datos disponibles.</p>'
    view = df.head(max_rows).copy()
    for col in view.columns:
        if view[col].dtype == object:
            view[col] = view[col].astype(str).map(lambda x: html.escape(x[:220]))
    return view.to_html(index=False, classes="table", border=0, escape=True)


def _badge(value: Any) -> str:
    """Badge de fuente con color estable."""
    text = html.escape(str(value))
    palette = {"DMO": "#2c7bb6", "percibido": "#d7191c"}
    color = palette.get(str(value), "#566573")
    return (f'<span style="background:{color};color:#fff;padding:2px 7px;'
            f'border-radius:10px;font-size:0.85em">{text}</span>')


def interpretive_bullets(alignment: pd.DataFrame, rates: pd.DataFrame,
                         valence: pd.DataFrame, small: bool) -> list[str]:
    """3-5 bullets de lectura interpretativa derivados de los números."""
    rows = alignment[alignment["scope"] == "DMO_vs_UGC"]
    gap_rate = float(rows["gap_rate_pp"].iloc[0]) if len(rows) else np.nan
    gap_cos = float(rows["gap_cosine"].iloc[0]) if len(rows) else np.nan
    centroid = float(rows["centroid_cosine_mentions"].iloc[0]) \
        if len(rows) and pd.notna(rows["centroid_cosine_mentions"].iloc[0]) else np.nan
    dmo_rate = float(alignment.loc[alignment["scope"] == "DMO", "rate"].iloc[0]) \
        if (alignment["scope"] == "DMO").any() else np.nan
    ugc_rate = float(alignment.loc[alignment["scope"] != "DMO", "rate"].mean()) \
        if (alignment["scope"] != "DMO").any() else np.nan
    bullets = [
        (f"**Presencia:** la tasa de mención es {dmo_rate:.1%} en DMO frente a "
         f"{ugc_rate:.1%} en UGC (brecha {gap_rate:+.1f} pp). "
         + ("Muestra DMO pequeña: leer tamaños de efecto, no solo p."
            if small else "Diferencia de frecuencia interpretable con IC bootstrap.")),
        (f"**Proximidad semántica:** el coseno medio DMO−UGC difiere en "
         f"{gap_cos:+.4f}; el coseno entre centroides de documentos que mencionan "
         f"el concepto es {centroid:.3f}." if pd.notna(centroid)
         else f"**Proximidad semántica:** el coseno medio DMO−UGC difiere en {gap_cos:+.4f}."),
    ]
    if not valence.empty:
        piv = valence.pivot_table(index="sentiment_label", columns="source",
                                  values="share_within_source", fill_value=0.0)
        if "DMO" in piv.columns and len(piv.columns) > 1:
            others = piv.drop(columns="DMO").mean(axis=1)
            top = (piv["DMO"] - others).abs().idxmax()
            bullets.append(
                f"**Tono condicionado:** la valencia con mayor desvío relativo en DMO "
                f"es {top} ({piv['DMO'].get(top, 0):.1%} vs {others.get(top, 0):.1%} en UGC); "
                "es valencia dentro de quienes mencionan el concepto, no el sentimiento global.")
    bullets.append(
        "**Contexto:** los términos de log-odds y el cruce tópico×fuente describen "
        "el encuadre; Jaccard se dejó fuera de esta sección por no ser comparable entre fuentes.")
    return bullets[:5]


def build_price_section_html(
    scores: pd.DataFrame,
    rates: pd.DataFrame,
    dist: pd.DataFrame,
    alignment: pd.DataFrame,
    valence: pd.DataFrame,
    context: pd.DataFrame,
    topics: pd.DataFrame,
    examples: pd.DataFrame,
    validation: Mapping[str, Any],
    anchors: Mapping[str, Any],
    figures_html: Mapping[str, str],
    small_sample: bool,
) -> str:
    """Sección HTML `precio` con el orden a)–g) exigido."""
    n_by_source = scores["source_label"].astype(str).value_counts().to_dict()
    banner = (
        '<div class="notice" style="border-left:5px solid #b03a2e;padding:10px;'
        'background:#fdf2f2;margin:12px 0"><b>Sin anotación humana; validación '
        'convergente automática.</b> La referencia léxica es débil, no gold standard.'
        '</div>')
    small = (
        '<div class="notice" style="border-left:5px solid #b7950b;padding:10px;'
        'background:#fef9e7;margin:12px 0"><b>LIMITACIÓN:</b> tasas inestables '
        'por muestra pequeña (n DMO con mención &lt; 10); interpretar tamaños de '
        'efecto, no solo p.</div>' if small_sample else "")
    method = (
        '<h3>a) Ficha metodológica</h3><ul class="bullets">'
        f'<li><b>Ancla:</b> v{anchors.get("anchors_version", "?")} · '
        f'sha256={html.escape(str(anchors.get("anchors_sha256", "?"))[:16])}…</li>'
        f'<li><b>Detector primario:</b> semantic · <b>τ=</b>'
        f'{float(anchors.get("tau", 0)):.4f} · <b>seed=</b>{anchors.get("seed", 42)}</li>'
        f'<li><b>n por fuente:</b> {html.escape(str(n_by_source))} · '
        f'normales, outliers excluidos</li>'
        f'<li><b>Validación:</b> {html.escape(str(validation.get("status", "?")))} · '
        f'κ semantic↔lexical={float(validation.get("kappa", np.nan)):.3f} · '
        f'FPR placebo={float(validation.get("fpr_random", np.nan)):.3f} · '
        f'tau estable={validation.get("gap_sign_stable", False)} · '
        f'jackknife={validation.get("n_jackknife", 0)} réplicas</li>'
        '</ul>' + banner + small)
    freq = (
        '<h3>b) FRECUENCIA</h3>'
        '<p>Tasa de mención por fuente con n, denominador e IC bootstrap 95%.</p>'
        + _df_html(rates, 10)
        + '<div class="figure">' + figures_html.get("density", "") + '</div>'
        + _df_html(dist, 10))
    scatter = (
        '<h3>c) SCATTER interactivo y UMAP del concepto</h3>'
        '<p>Un punto por comentario normal; color y símbolo por fuente. '
        'Usa los botones para cambiar el color por tópico o sentimiento.</p>'
        '<div class="figure">' + figures_html.get("similarity", "") + '</div>'
        '<p>UMAP 3D: color continuo por coseno; diamante amarillo = posición '
        'aproximada del concepto (baricentro k-NN).</p>'
        '<div class="figure">' + figures_html.get("umap", "") + '</div>')
    valence_html = valence.copy()
    if not valence_html.empty:
        valence_html["share_within_source"] = (
            100 * valence_html["share_within_source"]).round(2)
        valence_html = valence_html.rename(
            columns={"share_within_source": "pct_within_source"})
    registro = (scores[scores["es_mencion"].astype(bool)]
                .groupby(["source_label", "registro_dominante"]).size()
                .reset_index(name="n"))
    denom = registro.groupby("source_label")["n"].transform("sum")
    registro["pct_within_source"] = (100 * registro["n"] / denom).round(2)
    tono = (
        '<h3>d) TONO: valencia condicionada y registro</h3>'
        + _df_html(valence_html, 15) + _df_html(registro, 15)
        + '<p class="muted">Valencia <b>condicionada</b> a la mención del concepto; '
        'no es el sentimiento global descartado como evidencia de brecha.</p>')
    contexto = (
        '<h3>e) CONTEXTO</h3><p>Top-15 de log-odds por fuente y cruce tópico×fuente '
        '(% normalizado por fuente). Jaccard NO aparece aquí.</p>'
        + _df_html(context, 30) + _df_html(topics, 30))
    ex = examples.copy()
    if not ex.empty:
        ex[SOURCE_COLUMN] = ex[SOURCE_COLUMN].map(_badge)
        ex["text_norm"] = ex["text_norm"].astype(str).map(
            lambda t: html.escape(t[:200]))
    ejemplos = (
        '<h3>f) EJEMPLOS: 5 más cercanos (fuente visible)</h3>'
        + _df_html(ex, 40))
    def _md_bold(text: str) -> str:
        return html.escape(text).replace("**", "<b>", 1).replace("**", "</b>", 1)

    lectura = ('<h3>g) Lectura interpretativa</h3><ul class="bullets">'
               + "".join(f"<li>{_md_bold(b)}</li>" for b in interpretive_bullets(
                   alignment, rates, valence, small_sample)) + '</ul>')
    return method + freq + scatter + tono + contexto + ejemplos + lectura


def build_findings_md(
    alignment: pd.DataFrame, rates: pd.DataFrame, dist: pd.DataFrame,
    valuation: pd.DataFrame, validation: Mapping[str, Any],
    anchors: Mapping[str, Any], small_sample: bool,
) -> str:
    """Síntesis markdown copiable al informe escrito."""
    row = alignment[alignment["scope"] == "DMO_vs_UGC"]
    gap_rate = float(row["gap_rate_pp"].iloc[0]) if len(row) else np.nan
    gap_cos = float(row["gap_cosine"].iloc[0]) if len(row) else np.nan
    cent = float(row["centroid_cosine_mentions"].iloc[0]) \
        if len(row) and pd.notna(row["centroid_cosine_mentions"].iloc[0]) else np.nan
    cent_note = (str(row["centroid_cosine_note"].iloc[0])
                 if len(row) and "centroid_cosine_note" in row.columns else "")
    r0 = rates.iloc[0] if not rates.empty else pd.Series(dtype=float)
    dmo_mentions = int(alignment.loc[alignment["scope"] == "DMO",
                                     "n_mentions"].iloc[0]) \
        if (alignment["scope"] == "DMO").any() else 0
    d = dist[dist["scope"] == "DMO_vs_UGC_total"]
    emd = float(d["emd_wasserstein"].iloc[0]) if len(d) else np.nan
    jsd = float(d["jsd_base2"].iloc[0]) if len(d) else np.nan
    if "n" in valuation.columns:
        counts = valuation.groupby("registro_dominante")["n"].sum()
    else:
        counts = valuation["registro_dominante"].value_counts()
    promo = int(counts.get("promocional", 0))
    resena = int(counts.get("reseña", 0))
    lines = [
        "# Precio, valor y costo — síntesis para el informe",
        "",
        "## Pregunta de investigación",
        "¿Proyecta la DMO el precio/valor/costo de forma distinta a como lo perciben "
        "los usuarios (UGC), en frecuencia, encuadre y valencia condicionada?",
        "",
        "## Números clave",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| τ (placebo) | {float(anchors.get('tau', 0)):.4f} |",
        f"| Detector primario | {anchors.get('detector_primario', 'semantic')} |",
        f"| Ancla | v{anchors.get('anchors_version', '?')} · "
        f"{str(anchors.get('anchors_sha256', '?'))[:12]}… |",
        f"| n por fuente | {anchors.get('n_por_fuente', '?')} |",
        f"| Tasa DMO | {r0.get('rate_dmo', np.nan):.4f} "
        f"[{r0.get('ci_low_dmo', np.nan):.4f}, {r0.get('ci_high_dmo', np.nan):.4f}] |",
        f"| Tasa UGC | {r0.get('rate_ugc', np.nan):.4f} "
        f"[{r0.get('ci_low_ugc', np.nan):.4f}, {r0.get('ci_high_ugc', np.nan):.4f}] |",
        f"| Brecha de tasa (pp) | {gap_rate:+.2f} |",
        f"| Brecha de coseno medio | {gap_cos:+.4f} |",
        f"| Coseno de centroides (menciones) | "
        f"{cent:.4f} |" if pd.notna(cent)
        else f"| Coseno de centroides (menciones) | {html.escape(cent_note)} |",
        f"| EMD/Wasserstein histogramas | {emd:.4f} |",
        f"| JSD (base 2) | {jsd:.4f} |",
        f"| FPR placebo random | {validation.get('fpr_random', np.nan):.4f} |",
        f"| κ semantic↔lexical (referencia débil) | "
        f"{validation.get('kappa', np.nan):.3f} |",
        f"| gap_sign_stable | {validation.get('gap_sign_stable', False)} |",
        "",
        "## Conclusión en tres niveles",
        f"(i) **Presencia:** gap de {gap_rate:+.2f} pp en la tasa de mención; "
        "ver IC bootstrap y p ajustada BH en `price_comparison.csv`.",
        f"(ii) **Encuadre/registro:** entre menciones, {promo} registros "
        f"promocionales frente a {resena} experienciales (ver `price_scores.csv`, "
        "`registro_dominante`).",
        "(iii) **Valencia condicionada:** POS/NEU/NEG se calculan solo dentro de "
        "quienes mencionan el concepto; no sustituyen al sentimiento global, "
        "que se descartó como prueba de brecha.",
        "",
        "## Limitaciones",
        f"- n de la DMO: {anchors.get('n_por_fuente', '?')} "
        f"({dmo_mentions} menciones sobre τ)"
        + (" — **muestra pequeña: interpretar tamaños de efecto, no solo p**."
           if small_sample else "."),
        "- Sesgo de plataforma del UGC: la frecuencia observada no es prevalencia poblacional.",
        "- Ancla v1.0 sin validación humana: validación convergente automática; "
        "κ usa una referencia léxica débil, no gold standard.",
        f"- Dependencia de τ: `gap_sign_stable="
        f"{validation.get('gap_sign_stable', False)}`; FPR placebo reportado arriba.",
        "- Outliers excluidos de tasas y pruebas; no se muestran en la figura principal.",
        "",
        "## Referencia metodológica",
        "- Tópicos sintéticos / ancla semántica: Guerrero-Rodríguez et al. (2025), §3.3.",
        "- Comparación por histogramas normalizados + EMD: Díaz-Pacheco et al. (2023/2024).",
        "",
    ]
    return "\n".join(lines)


def _json(path: Path) -> dict:
    """Lee JSON o devuelve dict vacío (E/S aislada en esta capa)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_umap_coords(results_dir: Path, scores: pd.DataFrame,
                      embeddings: np.ndarray | None, seed: int) -> np.ndarray | None:
    """Coordenadas UMAP cacheadas por doc_id; si faltan, calcula 3D una vez."""
    path = results_dir / "comparison" / "report" / "umap_3d_points.csv"
    if path.exists():
        try:
            pts = pd.read_csv(path, encoding="utf-8-sig")
            cols = ["umap_1", "umap_2", "umap_3"]
            if {"doc_id"} <= set(pts.columns) and set(cols) <= set(pts.columns):
                ordered = scores[["doc_id"]].merge(
                    pts[["doc_id"] + cols], on="doc_id", how="left")
                if ordered[cols].notna().all().all():
                    return ordered[cols].to_numpy(float)
            # El CSV del PASO 9 no guarda doc_id: su orden es el de analysis_base.
            if set(cols) <= set(pts.columns) and len(pts) == len(scores):
                return pts[cols].to_numpy(float)
        except Exception:
            pass
    emb = _aligned_embeddings(scores, embeddings)
    if emb is None:
        return None
    import umap

    return umap.UMAP(n_components=3, random_state=seed,
                     n_neighbors=15).fit_transform(emb)


def write_price_report(
    args: Any,
    results_dir: Path,
    anchor_centroid: np.ndarray | None = None,
) -> list[Path]:
    """Construye y escribe todos los artefactos del bloque Precio (PASO 12.8)."""
    results_dir = Path(results_dir)
    price_dir = results_dir / "comparison" / PRICE_SUBDIR
    scores_path = price_dir / "price_scores.csv"
    if not scores_path.exists():
        print("  ⚠️ price_report omitido: falta price_scores.csv")
        return []
    params = _json(price_dir / "price_params.json")
    status = _json(price_dir / "price_validation_status.json")
    auto_path = price_dir / "price_validation_auto.csv"
    auto = (pd.read_csv(auto_path, encoding="utf-8-sig")
            if auto_path.exists() else pd.DataFrame())
    tau = float(params.get("tau", status.get("tau", 0.0)))
    seed = int(params.get("seed", getattr(args, "seed", 42)))

    doc_path = results_dir / "comparison" / "topics_joint" / "doc_topics.csv"
    doc_topics = (pd.read_csv(doc_path, encoding="utf-8-sig")
                  if doc_path.exists() else None)
    scores = enrich_scores(pd.read_csv(scores_path, encoding="utf-8-sig"),
                           doc_topics, tau)
    n_by_source = ";".join(
        f"{k}={int(v)}" for k, v in
        scores["source_label"].astype(str).value_counts().sort_index().items())
    trace = {
        "anchors_version": str(params.get("anchors_version", "")),
        "anchors_sha256": str(params.get("anchors_sha256", "")),
        "seed": seed,
        "tau": round(tau, 6),
        "detector_primario": "semantic",
        "n_normales": int(len(scores)),
        "n_por_fuente": n_by_source,
    }
    written: list[Path] = [save_csv(scores, scores_path)]

    density = density_table(scores)
    dist = distribution_tests(density)
    rates_path = price_dir / "price_comparison.csv"
    rates_raw = (pd.read_csv(rates_path, encoding="utf-8-sig")
                 if rates_path.exists() else pd.DataFrame())
    fpr_ctrl = float(status.get("placebo_fpr_controls", np.nan))
    fpr_rand = float(status.get("placebo_fpr_random", np.nan))
    rates = enhance_comparison(
        rates_raw, tau, fpr_ctrl, fpr_rand, seed)
    for col, value in trace.items():
        rates[col] = value
    written.append(save_csv(rates, rates_path))

    emb_path = results_dir / "comparison" / "base" / "embeddings_clean.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else None
    anchor_path = price_dir / "anchor_centroid.npy"
    if anchor_centroid is None and anchor_path.exists():
        anchor_centroid = np.load(anchor_path)
    alignment = alignment_table(scores, embeddings, anchor_centroid)
    valence = valence_conditioned_table(scores)
    context = context_logodds_table(scores, top_k=15)
    topics = topic_crosstab_table(scores)
    examples = top_examples_table(scores, top_k=TOP_K)
    tables = {
        "price_density.csv": density,
        "price_distribution_tests.csv": dist,
        "price_alignment.csv": alignment,
        "price_valence_conditioned.csv": valence,
        "price_context_logodds.csv": context,
        "price_topic_crosstab.csv": topics,
        "price_examples.csv": examples,
    }
    for name, df in tables.items():
        for col, value in trace.items():
            df[col] = value
        written.append(save_csv(df, price_dir / name))

    palette = palette_colors(palette_to_plot_colors(
        getattr(args, "palette", "ejecutiva")))
    fig_sim = build_similarity_figure(scores, tau, palette)
    fig_den = build_density_figure(density)
    coords = _load_umap_coords(results_dir, scores, embeddings, seed)
    figures_html: dict[str, str] = {}
    if coords is not None:
        fig_umap = build_umap_concept_figure(
            scores, coords, anchor_centroid, embeddings,
            k=int(getattr(args, "price_umap_k", JACKKNIFED_K)))
        written.append(save_csv(
            pd.DataFrame({
                "doc_id": scores["doc_id"].to_numpy(),
                "source": scores["source_label"].to_numpy(),
                "cos_semantic": scores["cos_semantic"].to_numpy(),
                "umap_1": coords[:, 0], "umap_2": coords[:, 1],
                "umap_3": coords[:, 2],
                "text_norm": scores["text_norm"].astype(str)
                .map(lambda t: t[:200]).to_numpy(),
            }), price_dir / "price_umap_concept_data.csv"))
        figures_html["umap"] = pio.to_html(
            fig_umap, full_html=False, include_plotlyjs="cdn")
        written.append(price_dir / "price_umap_concept.html")
        (price_dir / "price_umap_concept.html").write_text(
            figures_html["umap"], encoding="utf-8")
    else:
        figures_html["umap"] = ('<div class="placeholder">UMAP no disponible: '
                                'faltan coordenadas o embeddings</div>')
    written.append(save_csv(
        pd.DataFrame({
            "doc_id": scores["doc_id"].to_numpy(),
            "source": scores["source_label"].to_numpy(),
            "cos_semantic": scores["cos_semantic"].to_numpy(),
            "sentiment_score": scores["sentiment_score"].to_numpy(),
            "topic_name": scores["topic_name"].to_numpy(),
            "sentiment_label": scores["sentiment_label"].to_numpy(),
            "registro_dominante": scores["registro_dominante"].to_numpy(),
            "text_norm": scores["text_norm"].astype(str).map(
                lambda t: t[:200]).to_numpy(),
        }), price_dir / "price_scatter_similarity_data.csv"))
    figures_html["similarity"] = pio.to_html(
        fig_sim, full_html=False, include_plotlyjs="cdn")
    (price_dir / "price_scatter_similarity.html").write_text(
        figures_html["similarity"], encoding="utf-8")
    written.append(price_dir / "price_scatter_similarity.html")
    figures_html["density"] = pio.to_html(
        fig_den, full_html=False, include_plotlyjs="cdn")

    n_dmo_mentions = int(((scores["source_label"] == "DMO") &
                          scores["es_mencion"].astype(bool)).sum())
    small = n_dmo_mentions < SMALL_SAMPLE_MIN
    validation_view = {
        "status": status.get("status", "?"),
        "kappa": float(auto["kappa_semantic_vs_lexical"].iloc[0])
        if "kappa_semantic_vs_lexical" in auto.columns and len(auto)
        else np.nan,
        "fpr_random": fpr_rand,
        "gap_sign_stable": bool(status.get("gap_sign_stable", False)),
        "n_jackknife": int(status.get("n_jackknife", 0)),
    }
    section = build_price_section_html(
        scores, rates, dist, alignment, valence, context, topics, examples,
        validation_view, trace, figures_html, small)
    section_path = results_dir / "price_section.html"
    section_path.write_text(section, encoding="utf-8")
    written.append(section_path)
    findings = build_findings_md(
        alignment, rates, dist, scores[scores["es_mencion"].astype(bool)],
        validation_view, trace, small)
    findings_path = price_dir / "price_findings.md"
    findings_path.write_text(findings, encoding="utf-8")
    written.append(findings_path)
    written.append(append_metadata(
        results_dir / "comparison" / "base" / "run_metadata.csv",
        task="price_report",
        params={**trace, "n_jackknife": int(status.get("n_jackknife", 0)),
                "small_sample_warning": small},
        repo_dir=Path(__file__).resolve().parents[1]))

    print_header("PASO 12.8: INFORME PRECIO/VALOR/COSTO (corpus real)")
    print(f"  Comentarios normales: {len(scores):,} | τ={tau:.4f} | "
          f"menciones={int(scores['es_mencion'].sum()):,}")
    print(rates[["scope", "n_mentions_dmo", "n_docs_dmo", "rate_dmo",
                 "n_mentions_ugc", "n_docs_ugc", "rate_ugc", "gap_pp",
                 "test_used", "p_adj"]].to_string(index=False)
          if not rates.empty else "  Sin tabla de comparación.")
    print(f"  FPR placebo: controles={fpr_ctrl:.4f} random={fpr_rand:.4f} | "
          f"EMD={dist['emd_wasserstein'].iloc[0]:.4f} | "
          f"JSD={dist['jsd_base2'].iloc[0]:.4f}" if len(dist)
          else "  Sin pruebas de distribución.")
    if small:
        print(f"  ⚠️ n_DMO con menciones={n_dmo_mentions} < 10: tasas inestables; "
              "interpretar tamaños de efecto, no solo p.")
    print(f"  Top-5 por scope: {len(examples)} filas | "
          f"figuras: price_scatter_similarity.html, price_umap_concept.html")
    print(f"  Sección HTML -> {section_path} | síntesis -> {findings_path.name}")
    return written
