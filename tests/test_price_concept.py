"""Tests de price_concept.py (F4.2) con fixtures sintéticos.

No descarga modelos ni lee el corpus real: solo funciones puras.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from price_concept import (  # noqa: E402
    calibrate_tau,
    centered_cosine_scores,
    compute_price_scores,
    lexical_hits,
    load_price_anchors,
    prototype_registry,
    raw_cosine_scores,
)

ANCHORS_PATH = CODE_DIR / "resources" / "anchors" / "precio_valor.json"


# ── load_price_anchors ──
def test_load_price_anchors_valid_real_file():
    anchors = load_price_anchors(ANCHORS_PATH)
    assert anchors["concept_id"] == "precio_valor"
    assert set(anchors["prototypes"]) >= {"resena_es", "promo_es", "resena_en", "promo_en"}
    assert set(anchors["negative_controls"]) >= {
        "seguridad", "clima", "transporte", "gastronomia"}


def test_load_price_anchors_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_price_anchors(ANCHORS_PATH.with_name("no_existe.json"))


@pytest.mark.parametrize("mutacion", ["concept_id", "provenance", "prototypes", "lexical"])
def test_load_price_anchors_invalid_schema_raises(mutacion, tmp_path):
    data = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    if mutacion == "concept_id":
        data["concept_id"] = "otro"
    elif mutacion == "provenance":
        data["provenance"]["prompt_sha256"] = ""
    elif mutacion == "prototypes":
        data["prototypes"]["promo_en"] = []
    else:
        data["lexical_anchor"]["patterns"] = []
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_price_anchors(p)


def test_prototype_registry_registers():
    anchors = load_price_anchors(ANCHORS_PATH)
    reg = prototype_registry(anchors)
    registers = {r["register"] for r in reg}
    assert registers == {"resena", "promo"}
    assert len(reg) == sum(len(v) for v in anchors["prototypes"].values())


# ── lexical_hits ──
def test_lexical_hits_detecta_es_en():
    lex = {
        "terms_es": ["precio", "caro", "tarifa"],
        "terms_en": ["price", "cheap", "overpriced"],
        "patterns": [r"(?i)vale\s+la\s+pena", r"(?i)value\s+for\s+money"],
    }
    hits = lexical_hits(
        [
            "El precio era caro para la tarifa de temporada",     # 3 términos es
            "the price was cheap but the tour felt overpriced",   # 3 términos en
            "Vale la pena aunque es carisimo",                     # patrón es + ? (no term)
            "Value for money, the best rate we found",             # patrón en + rate? no
            "La playa y el clima fueron increíbles",               # 0
        ],
        lex,
    )
    assert hits[0]["lex_count"] == 3
    assert {"precio", "caro", "tarifa"} <= set(hits[0]["terms"].split(";"))
    assert hits[1]["lex_count"] == 3
    assert {"price", "cheap", "overpriced"} <= set(hits[1]["terms"].split(";"))
    assert hits[2]["patterns_matched"] and hits[2]["lex_count"] >= 1
    assert hits[3]["patterns_matched"]
    assert hits[4]["lex_count"] == 0 and hits[4]["terms"] == ""


def test_lexical_hits_folds_accents():
    lex = {"terms_es": ["valio", "costo"], "terms_en": [], "patterns": []}
    (hit,) = lexical_hits(["Valió la pena, costó barato"], lex)
    assert hit["lex_count"] == 2


# ── scoring ──
def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_centered_changes_scores():
    anchor = _unit([1.0, 0.0])
    docs = np.vstack([_unit([1.0, 0.0]), _unit([0.9, 0.435889])])
    raw = raw_cosine_scores(docs, anchor)
    centered = centered_cosine_scores(docs, anchor)
    assert raw.shape == (2,) and centered.shape == (2,)
    assert not np.allclose(raw, centered)
    # El doc alineado con el ancla gana al centrar; el 'solo diferente' baja más.
    assert centered[0] > centered[1]


def test_raw_cosine_scores_max_over_anchors():
    docs = _unit(np.array([[1.0, 0.0]]))
    anchors = np.vstack([_unit([0.0, 1.0]), _unit([1.0, 0.0])])
    scores = raw_cosine_scores(docs, anchors)
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(1.0)


def test_compute_price_scores_selects_registers():
    rng = np.random.default_rng(42)
    d = rng.normal(size=(5, 4))
    docs = np.asarray(d / np.linalg.norm(d, axis=1, keepdims=True), dtype=float)
    resena = np.vstack([_unit([1.0, 0.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0, 0.0])])
    promo = np.vstack([_unit([0.0, 0.0, 1.0, 0.0])])
    by_register = {"resena": resena, "promo": promo}
    for mode in ("raw", "centered"):
        out = compute_price_scores(docs, by_register, mode=mode)
        assert set(out) == {"sim_resena", "sim_promo", "score"}
        assert np.allclose(out["score"], np.fmax(out["sim_resena"], out["sim_promo"]))
    with pytest.raises(ValueError):
        compute_price_scores(docs, by_register, mode="bogus")


# ── calibrate_tau ──
def test_calibrate_tau_matches_numpy_quantile():
    scores = np.arange(0, 101, dtype=float)  # 0..100
    for q in (0.50, 0.90, 0.95, 0.99):
        assert calibrate_tau(scores, q) == pytest.approx(np.quantile(scores, q))
    assert calibrate_tau(scores, 0.95) == pytest.approx(95.0)


def test_calibrate_tau_ignores_nan_and_validates():
    scores = np.array([1.0, np.nan, 3.0, 5.0, 7.0, 9.0])
    assert calibrate_tau(scores, 0.5) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        calibrate_tau(np.array([np.nan, np.nan]), 0.95)
    with pytest.raises(ValueError):
        calibrate_tau(np.array([1.0, 2.0]), 1.5)
