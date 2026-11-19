"""Tests de F4.4b: comparación DMO vs percibido + contexto + KPIs + caché.

No descarga modelos ni lee el corpus real: fixtures sintéticos en memoria y
escritura de artefactos solo en tmp_path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from metrics_thematic import bh_adjust  # noqa: E402
from price_concept import (  # noqa: E402
    PRICE_OUTPUTS,
    bootstrap_rate_ci,
    mann_whitney_cliffs,
    price_comparison_table,
    price_context_cooccur,
    price_kpi_table,
    run_price_comparison_step,
)

PRICE_WORDS = ("precio", "caro", "tarifa", "valor", "costo", "barato", "euros")


def make_scores(n_dmo: int = 30, n_ugc: int = 30, seed: int = 42) -> pd.DataFrame:
    """Corpus sintético DMO/UGC con tasas de mención distintas por grupo."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def add(prefix: str, corpus: str, source: str, n: int, mention_rate: float) -> None:
        n_ment = int(round(n * mention_rate))
        for i in range(n):
            mention = i < n_ment
            words = " ".join(rng.choice(PRICE_WORDS, size=3))
            text = (f"El precio del tour {words} vale la pena documento {i}"
                    if mention else
                    f"La playa y el clima fueron increíbles documento {i} {words}")
            score = float(rng.uniform(0.35, 0.95) if mention else rng.uniform(0.0, 0.30))
            rows.append({
                "doc_id": f"{prefix}{i}", "text": text,
                "corpus": corpus, "source": source,
                "sentiment_score": float(rng.uniform(-0.8, 0.9)),
                "lex_count": 3 if mention else 0,
                "sim_resena_centered": score, "sim_promo_centered": score - 0.05,
                "price_score": score, "above_tau": mention,
            })

    add("d", "DMO", "sectur", n_dmo, 0.60)
    half = n_ugc // 2
    add("t", "UGC", "tripadvisor", half, 0.30)
    add("w", "UGC", "twitter", n_ugc - half, 0.20)
    df = pd.DataFrame(rows)
    s = df["sentiment_score"]
    df["sentiment_label"] = np.where(
        s > 0.15, "positive", np.where(s < -0.15, "negative", "neutral"))
    return df


def make_args(**overrides) -> argparse.Namespace:
    """Namespace mínimo que consume run_price_comparison_step."""
    base = dict(cache_dir=None, seed=42, bootstrap=200, permutations=200,
                price_top_k=3, prior_strength=1000.0, palette="ejecutiva",
                no_cache=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def run_step(tmp_path: Path, scores: pd.DataFrame, args=None) -> list[Path]:
    return run_price_comparison_step(
        args or make_args(), tmp_path, scores,
        tau=0.30, tau_quantile=0.95, detector="semantic",
        score_mode="centered", valid={})


# ── funciones puras ──
def test_bootstrap_rate_ci_determinista_y_acotado():
    mask = np.array([True, False] * 20)
    lo1, hi1 = bootstrap_rate_ci(mask, 200, 42)
    lo2, hi2 = bootstrap_rate_ci(mask, 200, 42)
    assert (lo1, hi1) == (lo2, hi2)          # misma seed → mismo CI
    assert 0.0 <= lo1 <= 0.5 <= hi1 <= 1.0
    lo0, hi0 = bootstrap_rate_ci(np.array([], dtype=bool), 200, 42)
    assert np.isnan(lo0) and np.isnan(hi0)
    lo_b, hi_b = bootstrap_rate_ci(mask, 0, 42)
    assert np.isnan(lo_b) and np.isnan(hi_b)


def test_mann_whitney_cliffs_separacion_y_empate():
    a = np.arange(1, 11, dtype=float)
    sep = mann_whitney_cliffs(a, np.arange(21, 31, dtype=float))
    assert sep["cliffs_delta"] == pytest.approx(-1.0)   # a < b en todos los pares
    assert sep["cliffs_interp"] == "GRANDE"
    assert mann_whitney_cliffs(np.arange(21, 31), a)["cliffs_delta"] == pytest.approx(1.0)
    emp = mann_whitney_cliffs(a, a)
    assert emp["cliffs_delta"] == pytest.approx(0.0)
    assert emp["mw_p_value"] == pytest.approx(1.0)
    vacio = mann_whitney_cliffs(a, [])
    assert np.isnan(vacio["cliffs_delta"]) and vacio["cliffs_interp"] == "SIN_DATOS"


def test_bh_adjust_monotono_y_acotado():
    adj = bh_adjust(np.array([0.001, 0.01, 0.04, 0.30, 0.80]))
    assert np.all(adj >= np.array([0.001, 0.01, 0.04, 0.30, 0.80]) - 1e-12)
    assert np.all(np.diff(adj) >= -1e-12)   # BH monótono en p ordenadas
    assert np.all((adj >= 0) & (adj <= 1))


# ── price_comparison.csv ──
def test_price_comparison_table_tasas_ci_y_veredicto():
    scores = make_scores()
    tbl = price_comparison_table(scores, bootstrap=200, seed=42)
    assert set(tbl["scope"]) >= {
        "DMO_vs_UGC_total", "DMO_vs_tripadvisor", "DMO_vs_twitter"}
    for col in ("rate_dmo", "rate_ugc", "ci_low_dmo", "ci_high_dmo",
                "ci_low_ugc", "ci_high_ugc", "p_value", "p_adj"):
        assert tbl[col].between(0, 1).all(), col
    assert (tbl["ci_low_dmo"] <= tbl["ci_high_dmo"]).all()
    assert (tbl["ci_low_ugc"] <= tbl["ci_high_ugc"]).all()
    assert (tbl["p_adj"] >= tbl["p_value"] - 1e-12).all()
    assert (tbl["n_docs_dmo"] > 0).all() and (tbl["n_docs_ugc"] > 0).all()
    assert (tbl["n_mentions_dmo"] <= tbl["n_docs_dmo"]).all()
    assert set(tbl["cliffs_interp"]) <= {"NEGLIGIBLE", "PEQUENO", "MEDIANO", "GRANDE"}
    # La fixture da más mención a DMO → gap positivo en todos los scopes.
    assert (tbl["gap_pp"] > 0).all()
    assert set(tbl["verdict"]) <= {"SOBRE_DMO", "SOBRE_UGC", "SIN_DIF"}


def test_price_comparison_table_determinista_con_seed():
    scores = make_scores()
    a = price_comparison_table(scores, bootstrap=300, seed=42)
    b = price_comparison_table(scores, bootstrap=300, seed=42)
    pd.testing.assert_frame_equal(a, b)
    c = price_comparison_table(scores, bootstrap=300, seed=7)
    assert not a["ci_low_dmo"].equals(c["ci_low_dmo"])   # la seed cambia el CI


# ── contexto + KPIs ──
def test_price_context_cooccur_top20_y_direccion():
    terms = price_context_cooccur(make_scores(), prior_strength=1000.0, top_k=20)
    assert len(terms) <= 20
    assert {"term", "docs_dmo", "docs_ugc", "freq_dmo", "freq_ugc",
            "z", "direction"} <= set(terms.columns)
    assert set(terms["direction"]) <= {"DMO", "UGC"}
    assert terms["z"].abs().is_monotonic_decreasing
    assert (terms["docs_dmo"] + terms["docs_ugc"]).min() >= 3


def test_price_kpi_table_columnas_planas():
    scores = make_scores()
    rates = price_comparison_table(scores, bootstrap=100, seed=42)
    kpis = price_kpi_table(scores, rates)
    assert list(kpis.columns) == ["kpi", "value", "unit", "scope"]
    assert {"n_docs", "mention_rate_dmo", "mention_rate_ugc",
            "cliffs_delta", "verdict"} <= set(kpis["kpi"])


# ── integración: artefactos en tmp_path ──
def test_price_kpis_csv_sin_anotaciones(tmp_path):
    run_step(tmp_path, make_scores())
    kpis_path = tmp_path / "comparison" / "price" / "price_kpis.csv"
    assert kpis_path.exists()
    header = kpis_path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == "kpi,value,unit,scope"
    for forbidden in ("label", "annotator", "note", "annot"):
        assert forbidden not in header.lower()


def test_price_examples_topk_limites(tmp_path):
    run_step(tmp_path, make_scores())
    ex = pd.read_csv(tmp_path / "comparison" / "price" / "price_examples_topk.csv",
                     encoding="utf-8-sig")
    assert len(ex) > 0
    assert ex["text_excerpt"].str.len().max() <= 280
    assert ex.groupby("source").size().max() <= 3          # price_top_k=3
    assert {"corpus", "source", "price_score"} <= set(ex.columns)


def test_price_scatter_valence_html_se_genera(tmp_path):
    run_step(tmp_path, make_scores())
    html_path = tmp_path / "comparison" / "price" / "price_scatter_valence.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "proximidad" in html          # eje x = proximidad al concepto
    assert "sentiment_score" in html     # eje y = valencia
    data = pd.read_csv(tmp_path / "comparison" / "price" /
                       "price_scatter_valence_data.csv", encoding="utf-8-sig")
    assert {"doc_id", "corpus", "source", "price_score",
            "sentiment_score"} <= set(data.columns)


def test_todos_los_outputs_declarados_se_escriben(tmp_path):
    written = run_step(tmp_path, make_scores())
    names = {p.name for p in written}
    assert set(PRICE_OUTPUTS) <= names
    out_dir = tmp_path / "comparison" / "price"
    for name in PRICE_OUTPUTS:
        assert (out_dir / name).exists(), name


# ── caché idempotente ──
def test_cache_segunda_corrida_no_recalcula(tmp_path):
    scores = make_scores()
    run_step(tmp_path, scores)
    kpis_path = tmp_path / "comparison" / "price" / "price_kpis.csv"
    sentinel = "kpi,value,unit,scope\nSENTINEL,1,unidad,total\n"
    kpis_path.write_text(sentinel, encoding="utf-8-sig")

    returned = run_step(tmp_path, scores)
    assert [p.name for p in returned] == list(PRICE_OUTPUTS)
    assert kpis_path.read_text(encoding="utf-8-sig") == sentinel  # no recalculó

    # Cambio de datos → fingerprint distinto → recalcula.
    changed = scores.copy()
    changed.loc[changed.index[0], "price_score"] += 0.25
    run_step(tmp_path, changed)
    assert "SENTINEL" not in kpis_path.read_text(encoding="utf-8-sig")


def test_cache_no_cache_fuerza_recalculo(tmp_path):
    scores = make_scores()
    run_step(tmp_path, scores)
    kpis_path = tmp_path / "comparison" / "price" / "price_kpis.csv"
    sentinel = "kpi,value,unit,scope\nSENTINEL,1,unidad,total\n"
    kpis_path.write_text(sentinel, encoding="utf-8-sig")

    run_step(tmp_path, scores, args=make_args(no_cache=True))
    assert "SENTINEL" not in kpis_path.read_text(encoding="utf-8-sig")
    assert (tmp_path / "comparison" / "price" / "price_comparison.csv").exists()
