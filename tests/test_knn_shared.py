"""Tests mínimos del grafo NN compartido."""
import numpy as np

from knn_shared import knn_k, subset_knn


def test_knn_k_cubre_fraccion():
    assert knn_k(15, 0.5, 1.0) == 30


def test_subset_knn_cobertura_y_fallback():
    rng = np.random.default_rng(42)
    emb = rng.random((12, 4)).astype(np.float32)
    from knn_shared import build_global_knn

    graph = build_global_knn(emb, 8, 42)
    pos = np.arange(12)
    knn, cov = subset_knn(graph, pos, 3)
    assert cov.ok and knn[0].shape == (12, 3)
    _, bad = subset_knn(graph, np.arange(6), 7)
    assert not bad.ok
