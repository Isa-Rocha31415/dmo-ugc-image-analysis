"""Tests de funciones puras de P1 (sin entrenar BERTopic)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from joint_topic_model import (  # noqa: E402
    composite_score, merge_to_k, npmi_coherence, reassign_and_probs,
    select_k, topic_diversity,
)


def test_merge_identidad_si_pocos_topicos():
    lab = np.array([0, 0, 1, -1])
    emb = np.eye(4)[:, :2]
    out, mapping = merge_to_k(lab, emb, k=5, w=np.ones(4))
    assert list(out) == [0, 0, 1, -1] and mapping == {0: 0, 1: 1}


def test_merge_fusiona_a_k():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(c, 0.05, size=(20, 8)) for c in (0, 3, 6)])
    lab = np.repeat(np.arange(6), 10)  # 6 tópicos base, 3 grupos reales
    out, _ = merge_to_k(lab, emb, k=3, w=np.ones(60))
    assert len(set(out.tolist())) == 3


def test_npmi_coocurrencia_perfecta():
    # dos palabras que siempre aparecen juntas → NPMI ≈ 1
    X = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
    vocab = np.array(["a", "b", "c"])
    val = npmi_coherence({0: ["a", "b"]}, X, vocab)
    assert val > 0.99


def test_diversity():
    assert topic_diversity({0: ["a", "b"], 1: ["b", "c"]}) == 3 / 4


def test_parsimonia_elige_menor_k():
    grid = pd.DataFrame({
        "k": [4, 8, 12], "coherence": [0.1, 0.2, 0.21],
        "diversity": [0.8, 0.7, 0.69], "stability": [0.9, 0.9, 0.9],
        "stability_ami": [0.9, 0.9, 0.9], "min_topic_size": [50, 40, 39],
        "unassigned_share": [0.1, 0.1, 0.1], "n_assigned": [900, 900, 900],
    })
    grid["composite_score"] = composite_score(grid)
    k, just = select_k(grid)
    assert k <= 12 and "parsimonia" in just


def test_reasignacion_por_umbral():
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    lab = np.array([0, 0, -1])
    out, prob, uniq, _ = reassign_and_probs(lab, emb, np.ones(3), threshold=0.5)
    assert out[2] == -1 and out[0] == 0 and prob[0] > 0.9