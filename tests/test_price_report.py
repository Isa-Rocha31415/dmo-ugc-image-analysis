"""Tests del bloque Precio (tablas canónicas, figuras y sección HTML)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from price_report import (  # noqa: E402
    alignment_table, build_density_figure, build_findings_md,
    build_price_section_html, build_similarity_figure, build_umap_concept_figure,
    context_logodds_table, density_table, distribution_tests, enhance_comparison,
    enrich_scores, top_examples_table, topic_crosstab_table,
    valence_conditioned_table,
)

TAU = 0.5


def _scores(n: int = 40, seed: int = 42, with_topic: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "doc_id": np.arange(n),
        "text": [f"texto {i} precio valor" for i in range(n)],
        "text_norm": [f"texto {i} precio valor" for i in range(n)],
        "corpus": np.where(np.arange(n) % 4 == 0, "DMO", "UGC"),
        "source": np.where(np.arange(n) % 4 == 0, "sectur",
                           np.where(np.arange(n) % 2 == 0, "reddit", "tripadvisor")),
        "sentiment_label": rng.choice(["POS", "NEU", "NEG"], size=n),
        "sentiment_score": rng.uniform(-1, 1, size=n),
        "price_score": rng.uniform(0.2, 0.9, size=n),
        "score_centered": rng.uniform(-0.3, 0.5, size=n),
        "sim_resena_centered": rng.uniform(0.1, 0.8, size=n),
        "sim_promo_centered": rng.uniform(0.1, 0.8, size=n),
        "above_tau": rng.uniform(0, 1, size=n) >= TAU,
    })
    if with_topic:
        df["topic_id"] = np.arange(n) % 3
        df["topic_name"] = [f"t{i % 3}" for i in range(n)]
    return df


def _enriched(n: int = 40) -> pd.DataFrame:
    return enrich_scores(_scores(n), None, TAU)


def test_enrich_scores_alias_registro_y_tema():
    raw = _scores(12)
    out = enrich_scores(raw, raw[["doc_id", "topic_id", "topic_name"]], TAU)
    assert np.allclose(out["cos_semantic"], raw["price_score"])
    assert np.allclose(out["sim_resena"], raw["sim_resena_centered"])
    assert set(out["registro_dominante"]) <= {"reseña", "promocional"}
    assert out["es_mencion"].equals(raw["above_tau"])
    assert (out["tau"] == TAU).all()


def test_density_normalizado_por_fuente():
    den = density_table(_enriched(), bins=10)
    total = den.groupby("source")["density"].sum()
    assert np.allclose(total.to_numpy(), 1.0)
    assert set(den["source"]) == {"DMO", "reddit", "tripadvisor"}


def test_distribution_tests_identicas_y_distintas():
    den = density_table(_enriched(60), bins=8)
    two = den[den["source"].isin(["DMO", "reddit"])].copy().reset_index(drop=True)
    dm = two.loc[two["source"] == "DMO"].sort_values("bin_center")[
        "density"].to_numpy()
    two.loc[two["source"] == "reddit", "density"] = dm
    same = distribution_tests(two)
    row = same[same["scope"] == "DMO_vs_UGC_total"].iloc[0]
    assert row["emd_wasserstein"] == pytest.approx(0.0, abs=1e-6)
    assert row["jsd_base2"] == pytest.approx(0.0, abs=1e-6)
    diff = distribution_tests(den)
    assert (diff["emd_wasserstein"] > 0).any()
    assert (diff["jsd_base2"] > 0).any()


def test_valencia_y_topics_solo_entre_menciones():
    sc = _enriched()
    val = valence_conditioned_table(sc)
    assert val["n"].sum() == int(sc["es_mencion"].sum())
    assert val.groupby("source")["n"].sum().sum() == int(sc["es_mencion"].sum())
    ct = topic_crosstab_table(sc)
    assert np.allclose(ct.groupby("source")["pct_within_source"].sum(), 1.0)


def test_context_logodds_top_15_por_fuente():
    ctx = context_logodds_table(_enriched(), top_k=15)
    assert set(ctx["source"]) == {"DMO", "reddit", "tripadvisor"}
    assert ctx.groupby("source").size().max() <= 15


def test_top_examples_incluye_todos_los_scopes_y_fuente():
    ex = top_examples_table(_enriched())
    assert {"global", "dmo", "ugc", "reddit", "tripadvisor"} <= set(ex["scope"])
    assert (ex.groupby("scope")["rank"].max() <= 5).all()
    assert {"doc_id", "source", "corpus", "cos_semantic", "text_norm"} <= set(ex)
    assert ex["cos_semantic"].notna().all()


def test_alignment_reporta_gap_y_centroides():
    al = alignment_table(_enriched())
    assert "DMO_vs_UGC" in set(al["scope"])
    assert {"gap_rate_pp", "gap_cosine", "centroid_cosine_mentions"} <= set(al)


def test_enhance_comparison_fisher_y_denominadores():
    rates = pd.DataFrame([{
        "scope": "DMO_vs_UGC_total", "n_docs_dmo": 8, "n_mentions_dmo": 1,
        "n_docs_ugc": 1000, "n_mentions_ugc": 20,
        "rate_dmo": 0.125, "rate_ugc": 0.02, "gap_pp": 10.5,
        "test": "fisher", "p_value": 0.1, "p_adj": 0.2,
    }])
    out = enhance_comparison(rates, TAU, 0.01, 0.02, 42)
    assert out["test_used"].iloc[0] == "fisher"
    assert "<5" in out["test_reason"].iloc[0]
    assert out["n_dmo_total"].iloc[0] == 8
    assert out["n_ugc_total"].iloc[0] == 1000
    assert out["p_adj_bh"].iloc[0] == 0.2


def test_similarity_figura_todos_puntos_tres_modos_y_redundancia():
    sc = _enriched(30)
    fig = build_similarity_figure(sc, TAU)
    n_points = sum(len(t.x) for t in fig.data)
    assert n_points == len(sc)
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == [
        "Por fuente", "Por tópico", "Por sentimiento"]
    symbols = {t.marker.symbol for t in fig.data}
    assert len(symbols) == len({s for s in sc["source_label"]})
    assert "umbral calibrado por placebo" in fig.layout.annotations[0].text
    hover = fig.data[0].hovertemplate
    for token in ("doc ", "Fuente", "Tópico", "Coseno", "Texto"):
        assert token in hover


def test_hover_escapa_html():
    sc = _enriched(6)
    sc.loc[0, "text_norm"] = "<script>alert(1)</script>"
    fig = build_similarity_figure(sc, TAU)
    assert "&lt;script&gt;" in str(fig.data[0].customdata)


def test_umap_reutiliza_maquinaria_y_etiqueta_concepto():
    sc = _enriched(20)
    coords = np.linspace(0, 1, 60).reshape(20, 3)
    emb = np.random.default_rng(0).normal(size=(20, 4))
    fig = build_umap_concept_figure(sc, coords, embeddings=emb)
    assert "posición aproximada del concepto" in fig.data[-1].name
    assert np.issubdtype(np.asarray(fig.data[0].marker.color).dtype, np.number)
    assert "Coseno" in fig.data[0].hovertemplate


def test_density_figura_y_html_cdn():
    import plotly.io as pio

    fig = build_density_figure(density_table(_enriched()))
    html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    assert "plotly" in html and "cdn" in html


def test_seccion_html_orden_y_banner_sin_anotacion():
    sc = _enriched()
    rates = enhance_comparison(pd.DataFrame([{
        "scope": "DMO_vs_UGC_total", "n_docs_dmo": 10, "n_mentions_dmo": 2,
        "n_docs_ugc": 30, "n_mentions_ugc": 9, "rate_dmo": 0.2,
        "ci_low_dmo": 0.1, "ci_high_dmo": 0.4, "rate_ugc": 0.3,
        "ci_low_ugc": 0.2, "ci_high_ugc": 0.5, "gap_pp": -10.0,
        "test": "chi2", "p_value": 0.2, "p_adj": 0.3,
    }]), TAU, 0.01, 0.02, 42)
    alignment = alignment_table(sc)
    section = build_price_section_html(
        sc, rates, distribution_tests(density_table(sc)), alignment,
        valence_conditioned_table(sc), context_logodds_table(sc, 5),
        topic_crosstab_table(sc), top_examples_table(sc),
        {"status": "validada_automaticamente", "kappa": 0.7,
         "fpr_random": 0.02, "gap_sign_stable": True, "n_jackknife": 200},
        {"anchors_version": "1.0", "anchors_sha256": "abc123", "tau": TAU,
         "seed": 42},
        {"similarity": "<div>scatter</div>", "umap": "<div>umap</div>",
         "density": "<div>densidad</div>"},
        True,
    )
    heads = ["a) Ficha metodológica", "b) FRECUENCIA", "c) SCATTER",
             "d) TONO", "e) CONTEXTO", "f) EJEMPLOS", "g) Lectura"]
    pos = [section.index(h) for h in heads]
    assert pos == sorted(pos)
    assert "Sin anotación humana" in section
    assert "tasas inestables por muestra pequeña" in section
    assert "Jaccard NO aparece aquí" in section
    assert "validada_automaticamente" in section


def test_findings_md_tres_niveles_y_limitaciones():
    sc = _enriched()
    md = build_findings_md(
        alignment_table(sc), pd.DataFrame(), distribution_tests(density_table(sc)),
        sc[sc["es_mencion"]], {"fpr_random": 0.01, "kappa": 0.6,
                               "gap_sign_stable": True},
        {"tau": TAU, "detector_primario": "semantic", "n_por_fuente": "DMO=10"},
        False,
    )
    for token in ("(i) **Presencia:**", "(ii) **Encuadre", "(iii) **Valencia",
                  "Limitaciones", "Guerrero-Rodríguez", "Díaz-Pacheco"):
        assert token in md
