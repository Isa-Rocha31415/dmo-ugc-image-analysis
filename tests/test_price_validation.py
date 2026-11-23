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

from price_concept import (  # noqa: E402
    DEFAULT_ANCHORS, filtrar_normales, load_price_anchors,
    price_traceability_context, propagate_price_traceability,
    synthetic_anchor_centroid,
)
from price_validation import (  # noqa: E402
    ablation_predictions, bootstrap_metrics_ci, classification_metrics,
    cohen_kappa, consensus_labels, load_annotations, make_stratified_sample,
    run_price_validation,
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


def test_validacion_automatica_sin_escrituras(tmp_path):
    """La rama automática sustituye a la antigua salida 'no_realizada'."""
    args = type("Args", (), {"bootstrap": 10, "seed": 42,
                             "price_sample_size": 40})()
    out = run_price_validation(args, tmp_path, _synth_scores(20), 0.42, 0.95)
    assert out["status"] == "validada_automaticamente"
    assert out["validated"] is True
    assert out["tau"] == 0.42
    assert out["detector_primario"] == "semantic"
    auto = tmp_path / "comparison" / "price" / "price_validation_auto.csv"
    assert auto.exists()


# ── F4.3 automática (sin intervención humana) ──
def _auto_args(**extra):
    base = {"bootstrap": 10, "seed": 42, "price_sample_size": 60,
            "price_jackknife_replicas": 5, "price_tau_quantile": 0.95}
    base.update(extra)
    return type("Args", (), base)()


def test_no_hay_rama_humana():
    src = (CODE_DIR / "price_validation.py").read_text(encoding="utf-8")
    assert "human_label" not in src
    assert "annotation_sample.csv" not in src
    assert 'status="no_realizada"' not in src
    cli = (CODE_DIR / "run_analysis.py").read_text(encoding="utf-8")
    assert "--price-annotation-file" in cli  # se conserva, ignorado


def test_run_price_validation_no_retorna_no_realizada(tmp_path):
    out = run_price_validation(_auto_args(), tmp_path, _synth_scores(60), 0.42, 0.95)
    assert out["status"] == "validada_automaticamente"
    price_dir = tmp_path / "comparison" / "price"
    for name in ("price_validation_auto.csv", "price_threshold_sensitivity.csv",
                 "price_anchor_jackknife.csv", "price_validation_status.json"):
        assert (price_dir / name).exists(), name


def test_kappa_semantic_vs_lexical_se_calcula_y_esta_etiquetado_como_referencia_debil(tmp_path):
    df = _synth_scores(80)
    df["score_semantic"] = np.where(df["lex_count"] > 1, 0.8, 0.1)
    df["score_lexical"] = df["lex_count"].astype(float)
    out = run_price_validation(_auto_args(), tmp_path, df, 0.5, 0.95)
    auto = pd.read_csv(tmp_path / "comparison" / "price" / "price_validation_auto.csv",
                       encoding="utf-8-sig")
    assert "kappa_semantic_vs_lexical" in auto.columns
    assert auto["kappa_semantic_vs_lexical"].notna().any()
    assert (auto["reference_type"] == "weak_lexical_reference_not_gold_standard").any()
    assert "accuracy" not in [str(c).lower() for c in auto.columns]
    assert out["reference_type"] == "weak_lexical_reference_not_gold_standard"


def test_sensibilidad_tau_reporta_gap_sign_stable(tmp_path):
    placebo = np.linspace(0, 1, 300)
    out = run_price_validation(_auto_args(), tmp_path, _synth_scores(80), 0.42, 0.95,
                               placebo_scores=placebo)
    sens = pd.read_csv(
        tmp_path / "comparison" / "price" / "price_threshold_sensitivity.csv",
        encoding="utf-8-sig")
    assert len(sens) == 3
    assert {"tau_quantile", "tau", "rate_dmo", "rate_ugc", "gap_sign",
            "gap_sign_stable"} <= set(sens.columns)
    assert set(sens["gap_sign_stable"].unique()) <= {True, False}
    assert out["checks_ejecutados"] and "threshold_sensitivity" in out["checks_ejecutados"]


def test_jackknife_es_reproducible_con_seed_42(tmp_path):
    rng = np.random.default_rng(7)
    df = _synth_scores(60)
    doc_emb = rng.normal(size=(60, 8)).astype(np.float32)
    anchor_emb = rng.normal(size=(20, 8)).astype(np.float32)
    kw = dict(doc_emb=doc_emb, anchor_emb=anchor_emb,
              placebo_scores=np.linspace(0, 1, 100))
    run_price_validation(_auto_args(), tmp_path / "a", df, 0.42, 0.95, **kw)
    run_price_validation(_auto_args(), tmp_path / "b", df, 0.42, 0.95, **kw)
    a = pd.read_csv(tmp_path / "a" / "comparison" / "price" / "price_anchor_jackknife.csv",
                    encoding="utf-8-sig")
    b = pd.read_csv(tmp_path / "b" / "comparison" / "price" / "price_anchor_jackknife.csv",
                    encoding="utf-8-sig")
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 5
    assert {"replicate", "spearman_top5", "rate_delta"} <= set(a.columns)


def test_placebo_fpr_se_reporta(tmp_path):
    placebo = np.linspace(0, 1, 200)
    components = {"neg_control_seguridad": np.linspace(0, 0.4, 50),
                  "random_docs": placebo}
    out = run_price_validation(_auto_args(), tmp_path, _synth_scores(80), 0.42, 0.95,
                               placebo_scores=placebo, placebo_components=components)
    auto = pd.read_csv(tmp_path / "comparison" / "price" / "price_validation_auto.csv",
                       encoding="utf-8-sig")
    assert {"placebo_fpr_controls", "placebo_fpr_random"} <= set(auto.columns)
    assert 0.0 <= float(out["placebo_fpr_random"]) <= 1.0
    assert "placebo_fpr" in out["checks_ejecutados"]


def test_anchors_sha256_se_propaga_a_todos_los_csv(tmp_path):
    anchors = load_price_anchors(DEFAULT_ANCHORS)
    scores = _synth_scores(20)
    ctx = price_traceability_context(anchors, DEFAULT_ANCHORS, scores, 0.42, 42)
    price_dir = tmp_path / "comparison" / "price"
    price_dir.mkdir(parents=True)
    for name in ("price_scores.csv", "price_comparison.csv"):
        pd.DataFrame({"doc_id": [1]}).to_csv(
            price_dir / name, index=False, encoding="utf-8-sig")
    written = propagate_price_traceability(tmp_path, ctx)
    assert len(written) == 2
    for path in written:
        df = pd.read_csv(path, encoding="utf-8-sig")
        for col in ("anchors_version", "anchors_sha256", "seed", "tau",
                    "detector_primario", "n_normales", "n_por_fuente"):
            assert col in df.columns, (path.name, col)
        assert df["anchors_sha256"].iloc[0] == ctx["anchors_sha256"]


def test_precio_excluye_outliers():
    base = pd.DataFrame({"doc_id": [0, 1, 2], "is_outlier": [False, True, False]})
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    clean, clean_emb = filtrar_normales(base, emb)
    assert clean["doc_id"].tolist() == [0, 2]
    assert clean_emb.shape == (2, 4)


def test_synthetic_centroid_es_unitario_y_puro():
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(9, 6))
    centroid = synthetic_anchor_centroid(emb)
    assert centroid.shape == (6,)
    assert np.linalg.norm(centroid) == pytest.approx(1.0)

