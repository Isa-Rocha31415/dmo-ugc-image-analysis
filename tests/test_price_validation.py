"""Tests de price_validation.py (F4.3) con fixtures sintéticos."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from price_validation import (  # noqa: E402
    ablation_predictions, bootstrap_metrics_ci, classification_metrics,
    cohen_kappa, consensus_labels, load_annotations, make_stratified_sample,
)


def _synth_scores(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "doc_id": np.arange(n),
        "corpus": rng.choice(["DMO", "UGC"], size=n, p=[0.2, 0.8]),
        "source": rng.choice(["a", "b", "c"], size=n),
        "text": ["texto sintético %d" % i for i in range(n)],
        "price_score": rng.beta(2, 5, size=n),
        "sim_resena_centered": rng.beta(2, 5, size=n),
        "sim_promo_centered": rng.beta(2, 5, size=n),
        "score_raw": rng.beta(2, 5, size=n),
        "score_centered": rng.beta(2, 5, size=n),
        "lex_count": rng.integers(0, 4, size=n),
        "sentiment_label": rng.choice(["positive", "negative"], size=n),
        "sentiment_score": rng.uniform(-1, 1, size=n),
    })
    return df


# ── make_stratified_sample ──
def test_stratified_sample_size_and_weights():
    df = _synth_scores()
    sample = make_stratified_sample(df, n=100, seed=42)
    assert len(sample) == 100
    assert {"stratum", "weight"} <= set(sample.columns)
    assert sample["stratum"].nunique() >= 2
    N = df.groupby(["corpus"]).size()  # sanity: pesos > 0
    assert (sample["weight"] > 0).all()
    # pesos = N_h/n_h: máximos plausibles
    assert sample["weight"].max() <= len(df)
    # determinista
    sample2 = make_stratified_sample(df, n=100, seed=42)
    pd.testing.assert_frame_equal(sample, sample2)


def test_stratified_sample_covers_all_bands_when_possible():
    df = _synth_scores()
    sample = make_stratified_sample(df, n=100, seed=42)
    bands = sample["stratum"].str.split("_", regex=False).str[0].unique()
    assert set(bands) == {"b1", "b2", "b3", "b4", "b5"}


# ── load_annotations ──
def test_load_annotations_missing_returns_empty(tmp_path):
    out = load_annotations(tmp_path / "no_existe.csv")
    assert out.empty


def test_load_annotations_invalid_columns_raises(tmp_path):
    p = tmp_path / "ann.csv"
    p.write_text("doc_id,etiqueta\n1,precio\n", encoding="utf-8-sig")
    with pytest.raises(ValueError):
        load_annotations(p)


def test_load_annotations_ok(tmp_path):
    p = tmp_path / "ann.csv"
    p.write_text("doc_id,annotator,label,note\n1,a,precio,x\n", encoding="utf-8-sig")
    out = load_annotations(p)
    assert len(out) == 1 and out.iloc[0]["label"] == "precio"


# ── consensus_labels ──
def test_consensus_majority_and_tie_discard():
    long = pd.DataFrame({
        "doc_id": [1, 1, 2, 2, 3],
        "annotator": ["a", "b", "a", "b", "a"],
        "label": ["precio", "precio", "precio", "no_precio", "ambiguo"],
    })
    cons = consensus_labels(long)
    row1 = cons[cons["doc_id"] == 1].iloc[0]
    assert row1["label"] == "precio" and row1["n_annotators"] == 2
    assert cons[cons["doc_id"] == 2]["label"].isna().all()  # empate → descartado
    assert cons[cons["doc_id"] == 3]["label"].isna().all()  # etiqueta inválida


# ── cohen_kappa ──
def test_cohen_kappa_perfect_and_random():
    assert cohen_kappa(["precio"] * 10, ["precio"] * 10) == pytest.approx(1.0)
    # Un rater constante frente a otro constante con categoría opuesta: κ=0
    # (definición estándar: pe=0 cuando un rater no varía).
    assert cohen_kappa(["precio"] * 10, ["no_precio"] * 10) == pytest.approx(0.0)
    a = ["precio", "no_precio"] * 5
    b = ["no_precio", "precio"] * 5
    assert cohen_kappa(a, b) == pytest.approx(-1.0)  # desacuerdo total (50/50)


# ── classification_metrics ──
def test_classification_metrics_perfect_and_weighted():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, False]
    m = classification_metrics(y_true, y_pred)
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 0 and m["tn"] == 2
    assert m["recall"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(1.0)
    # ponderado: el FP pesa el triple
    mw = classification_metrics(y_true, y_pred, weights=[1, 1, 3, 1])
    assert mw["fp"] == 0 and mw["tn"] == pytest.approx(4)
    assert mw["fn"] == pytest.approx(1)


def test_bootstrap_ci_sane():
    rng = np.random.default_rng(0)
    y_true = rng.random(200) > 0.5
    y_pred = y_true.copy()
    y_pred[:20] = ~y_pred[:20]
    ci = bootstrap_metrics_ci(y_true, y_pred, B=300, seed=42)
    for key in ("precision", "recall", "f1", "mcc"):
        lo, hi = ci[key]
        assert lo <= hi


# ── ablation_predictions ──
def test_ablation_rules():
    df = pd.DataFrame({
        "lex_count": [0, 3, 0, 1],
        "sim_resena_centered": [0.9, 0.1, 0.1, 0.4],
        "sim_promo_centered": [0.1, 0.1, 0.9, 0.4],
        "score_raw": [0.1, 0.1, 0.1, 0.6],
        "score_centered": [0.9, 0.1, 0.9, 0.45],
    })
    tau = 0.5
    preds = ablation_predictions(df, tau)
    assert list(preds["lexical_only"]) == [False, True, False, True]
    assert list(preds["resena_only"]) == [True, False, False, False]
    assert list(preds["promo_only"]) == [False, False, True, False]
    assert list(preds["centered_max"]) == [True, False, True, False]
    assert list(preds["hybrid"]) == [True, True, True, False]  # lex>=2 activa b2
    assert list(preds["tau_fijo_0.5"]) == [True, False, True, False]
