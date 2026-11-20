"""Grafo NN compartido: una sola construcción, subconjuntos con cobertura verificada."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KnnCoverage:
    """Estadísticas de cobertura del grafo global sobre un subconjunto."""

    rows: int
    needed: int
    min_survivors: int
    ok: bool


def knn_k(n_neighbors: int, min_fraction: float, mult: float) -> int:
    """K global que cubre el peor subconjunto conocido (fracción mínima)."""
    frac = max(1e-6, min(1.0, min_fraction))
    return max(n_neighbors + 1, math.ceil(n_neighbors / frac * mult))


def build_global_knn(
    embeddings: np.ndarray, k: int, seed: int, metric: str = "cosine"
) -> tuple[np.ndarray, np.ndarray]:
    """Construye el grafo global una sola vez con pynndescent (offline)."""
    from pynndescent import NNDescent

    nn = NNDescent(
        np.asarray(embeddings, dtype=np.float32), n_neighbors=int(k),
        metric=metric, random_state=seed, low_memory=True,
    )
    idx, dist = nn.neighbor_graph
    return np.asarray(idx, dtype=np.int64), np.asarray(dist, dtype=np.float32)


def subset_knn(
    graph: tuple[np.ndarray, np.ndarray] | None,
    positions: np.ndarray,
    n_neighbors: int,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, KnnCoverage]:
    """Filtra filas/columnas del grafo; solo lo usa si TODAS las filas cubren k."""
    pos = np.asarray(positions, dtype=np.int64)
    needed = int(n_neighbors)
    if graph is None or not len(pos) or needed <= 0:
        return None, KnnCoverage(len(pos), needed, 0, False)
    gidx, gdist = graph
    keep = np.isin(gidx[pos], pos)
    counts = keep.sum(axis=1)
    minimum = int(counts.min()) if len(counts) else 0
    cov = KnnCoverage(len(pos), needed, minimum, minimum >= needed)
    if not cov.ok:
        return None, cov
    local = {int(p): i for i, p in enumerate(pos)}
    out_idx = np.empty((len(pos), needed), dtype=np.int64)
    out_dist = np.empty((len(pos), needed), dtype=np.float32)
    for r, global_row in enumerate(pos):
        cols = np.flatnonzero(keep[r])[:needed]
        out_idx[r] = [local[int(g)] for g in gidx[global_row, cols]]
        out_dist[r] = gdist[global_row, cols]
    return (out_idx, out_dist), cov
