"""Contrato de aislamiento de backends en el PASO 9 (BERTopic vs XidooTopic)."""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from comparison.common import joint_backend, topics_joint_dir  # noqa: E402
from joint_topic_model import base_cache_payload, grid_cache_payload  # noqa: E402
from xidoo_topic_modeler import expand_xidoo_labels, normalize_xidoo_keywords  # noqa: E402


def _args(model: str) -> Namespace:
    return Namespace(model=model, seed=42, base_min_topic_size=10,
                     min_df=5, max_df=0.4, ngram_max=2, stability_runs=2,
                     subsample=0.75, reassign_threshold=0.3, dmo_weight=1.0,
                     k_min=4, k_max=30)


def test_backend_normalization_single_source_of_truth():
    assert joint_backend(_args("bertopic")) == "bertopic"
    assert joint_backend(_args("xidoo")) == "xidoo"
    assert joint_backend(_args("xidootopic")) == "xidoo"


def test_paths_are_backend_scoped():
    root = Path("/tmp/resultados")
    b = topics_joint_dir(_args("bertopic"), root)
    x = topics_joint_dir(_args("xidoo"), root)
    assert b == root / "comparison" / "topics_joint" / "bertopic"
    assert x == root / "comparison" / "topics_joint" / "xidoo"
    assert b != x


def test_cache_payloads_include_backend():
    emb = np.zeros((5, 3), dtype=np.float32)
    args = _args("bertopic")
    b_base = base_cache_payload(emb, args, "bertopic")
    x_base = base_cache_payload(emb, args, "xidoo")
    assert b_base["backend"] == "bertopic"
    assert x_base["backend"] == "xidoo"
    assert b_base != x_base

    b_grid = grid_cache_payload([4, 5], args, "bertopic", "clave_bert")
    x_grid = grid_cache_payload([4, 5], args, "xidoo", "clave_xidoo")
    assert b_grid["backend"] == "bertopic"
    assert x_grid["backend"] == "xidoo"
    assert b_grid["base_key"] != x_grid["base_key"]
    assert b_grid != x_grid


def test_expand_xidoo_labels_alignment():
    # 5 docs; el detector de outliers quita el índice 1; el filtro quita el 3.
    labels = np.array([0, 7, 7])
    full = expand_xidoo_labels(
        labels,
        outlier_mask=np.array([True, False, True, True, True]),
        cluster_mask=np.array([True, True, True, False]),
        n_docs=5,
    )
    assert full.tolist() == [0, -1, 7, 7, -1]


def test_normalize_xidoo_keyword_tuples():
    raw = {0: (["playa", "arena"], "[KEYWORDS] playa,arena"), 1: ([], "[EMPTY]")}
    out = normalize_xidoo_keywords(raw)
    assert out == {"0": ["playa", "arena"], "1": []}


def test_xidoo_backend_failure_is_fatal(monkeypatch):
    from topic_modeling import run_topic_modeling
    from xidoo_topic_modeler import XidooTopicModeler

    def _boom(self, *args, **kwargs):
        raise RuntimeError("xidootopic roto")

    monkeypatch.setattr(XidooTopicModeler, "fit_transform", _boom)
    df = pd.DataFrame({
        "text": ["hotel playa servicio comida"] * 40,
        "source": ["tripadvisor"] * 40,
        "sentiment_label": ["positive"] * 20 + ["negative"] * 20,
    })
    with pytest.raises(RuntimeError, match="PASO 7"):
        run_topic_modeling(
            df=df, text_column="text", source_column="source",
            topic_backend="xidoo", min_partition_size=10,
            labels=["positive"], parallel=False,
        )


def test_paso9_xidoo_failure_propagates(monkeypatch, tmp_path):
    from joint_topic_model import run_topics_task
    from xidoo_topic_modeler import XidooTopicModeler

    (tmp_path / "comparison" / "base").mkdir(parents=True)
    pd.DataFrame({
        "doc_id": [0, 1], "text": ["playa bonita", "precio justo"],
        "corpus": ["DMO", "UGC"], "embedding_row": [0, 1],
    }).to_csv(tmp_path / "comparison" / "base" / "analysis_base.csv",
              index=False, encoding="utf-8-sig")
    np.save(tmp_path / "comparison" / "base" / "embeddings_clean.npy",
            np.zeros((2, 4), dtype=np.float32))

    def _boom(self, *args, **kwargs):
        raise RuntimeError("xidootopic roto")

    monkeypatch.setattr(XidooTopicModeler, "fit_transform", _boom)
    with pytest.raises(RuntimeError, match="xidootopic roto"):
        run_topics_task(_args("xidoo"), tmp_path)


def test_real_xidoo_wrapper_alignment():
    pytest.importorskip("xidootopic")
    from xidoo_topic_modeler import XidooTopicModeler

    rng = np.random.default_rng(42)
    texts = []
    for centro, palabra in ((0.0, "playa"), (4.0, "precio")):
        texts.extend(
            f"{palabra} hotel servicio habitaciónopinion numero {i}"
            for i in range(40)
        )
    emb = np.vstack([
        rng.normal(0.0, 0.05, size=(40, 8)),
        rng.normal(4.0, 0.05, size=(40, 8)),
    ]).astype(np.float32)
    modeler = XidooTopicModeler(random_state=42, min_cluster_size=5)
    result = modeler.fit_transform(texts, embeddings=emb)
    assert len(result["topic_id"]) == len(texts)
    assert len(modeler.base_labels_) == len(texts)
    assert modeler.reduction_.shape[0] == len(texts)
    assert len(set(result["topic_id"]) - {-1}) >= 1
