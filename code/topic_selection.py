"""
topic_selection.py
──────────────────
Selección del número de tópicos para BERTopic y XidooTopic.

Método (agnóstico al backend):
  1. El backend se ajusta una sola vez y produce una solución fina.
  2. Se construye una jerarquía de fusiones de tópicos por similitud de
     centroides (o Jaccard de keywords si no hay embeddings).
  3. Para cada k candidato se calculan coherencia NPMI, diversidad,
     estabilidad bootstrap (ARI), redundancia y balance.
  4. Se elige por puntaje compuesto con regla de parsimonia: entre las
     soluciones dentro de `PARSIMONY_TOLERANCE` del mejor puntaje se toma
     la de menor k.

No se usa el codo de error de reconstrucción: no es comparable entre
BERTopic y XidooTopic y favorece soluciones sobresegmentadas cuando el
modelo base produce muchos tópicos pequeños.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score

try:
    from nltk.corpus import stopwords as _nltk_stopwords
except Exception:  # pragma: no cover - depende de recursos locales
    _nltk_stopwords = None


AUTO_ALIASES = {"auto", "automatic", "automatico", "automático"}
OUTLIER_IDS = {-1, "-1"}
DEFAULT_TOP_N_WORDS = 10
DEFAULT_BOOTSTRAPS = 8
PARSIMONY_TOLERANCE = 0.02
MAX_CANDIDATES = 30
DEFAULT_WEIGHTS = {
    "coherence": 0.35,
    "stability": 0.25,
    "diversity": 0.15,
    "balance": 0.15,
    "redundancy": 0.10,
}


def parse_num_topics(value: Any) -> Any:
    """Normaliza `num_topicos`: devuelve ``"auto"`` o un entero >= 1."""
    if value is None:
        return "auto"
    if isinstance(value, bool):
        raise ValueError("num_topicos debe ser un entero >= 1 o 'automatico'.")
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if number < 1:
            raise ValueError("num_topicos debe ser un entero >= 1 o 'automatico'.")
        return number

    text = str(value).strip().lower()
    if text in AUTO_ALIASES:
        return "auto"
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(
            "num_topicos debe ser un entero >= 1 o 'automatico'."
        ) from exc
    if number < 1:
        raise ValueError("num_topicos debe ser un entero >= 1 o 'automatico'.")
    return number


@dataclass
class TopicCandidate:
    """Métricas de una solución con k tópicos."""

    k: int
    coherence: Optional[float]
    diversity: float
    stability: Optional[float]
    redundancy: float
    balance: float
    min_topic_size: int
    eligible: bool = True
    selected: bool = False
    score: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TopicSelectionSummary:
    """Contrato serializable de la selección."""

    method: str
    requested: str
    selected_k: int
    base_topics: int
    n_docs: int
    n_outliers: int
    reason: str
    candidates: List[TopicCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "requested": self.requested,
            "selected_k": int(self.selected_k),
            "base_topics": int(self.base_topics),
            "n_docs": int(self.n_docs),
            "n_outliers": int(self.n_outliers),
            "reason": self.reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def select_num_topics(
    texts: Sequence[str],
    topics: Sequence[Any],
    embeddings: Optional[np.ndarray] = None,
    keywords: Optional[Dict[Any, Any]] = None,
    requested: Any = "auto",
    top_n_words: int = DEFAULT_TOP_N_WORDS,
    random_state: int = 42,
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
) -> Dict[str, Any]:
    """
    Selecciona (o fija) el número de tópicos y devuelve el mapeo necesario.

    Returns:
        {
          "summary": dict serializable,
          "mapping": {topic_id_base: topic_id_nuevo},
          "keywords": {topic_id_nuevo: [términos]},
          "topic_names": {topic_id_nuevo: nombre},
        }
    """
    requested = parse_num_topics(requested)
    texts = [str(text) for text in texts]
    topic_ids = [str(topic) for topic in topics]
    n_docs = len(topic_ids)
    if n_docs != len(texts):
        raise ValueError("texts y topics deben tener la misma longitud.")

    base_ids = _ordered_unique(topic_ids)
    outlier_ids = [tid for tid in base_ids if tid in OUTLIER_IDS]
    base_ids = [tid for tid in base_ids if tid not in OUTLIER_IDS]
    m = len(base_ids)
    n_outliers = sum(1 for tid in topic_ids if tid in OUTLIER_IDS)

    if n_docs == 0 or m == 0:
        summary = TopicSelectionSummary(
            method="merge_hierarchy_npmi_stability",
            requested=str(requested),
            selected_k=0,
            base_topics=0,
            n_docs=n_docs,
            n_outliers=n_outliers,
            reason="No hay tópicos base para seleccionar.",
        )
        return {
            "summary": summary.to_dict(),
            "mapping": {tid: -1 for tid in outlier_ids},
            "keywords": {},
            "topic_names": {},
        }

    X_bin, terms, term_index = _build_doc_term_matrix(texts)
    base_of_doc = np.array(
        [base_ids.index(tid) if tid in base_ids else -1 for tid in topic_ids],
        dtype=int,
    )
    base_sizes = np.bincount(
        base_of_doc[base_of_doc >= 0], minlength=m
    ).astype(int)
    base_term_counts = _base_term_counts(X_bin, base_of_doc, m, len(terms))
    base_keywords = _normalize_keywords(keywords)
    keyword_sets = _keyword_sets(
        base_ids, base_keywords, base_term_counts, terms, top_n_words
    )

    centroids = _centroids(
        np.asarray(embeddings) if embeddings is not None else None,
        base_of_doc,
        m,
    )
    full_partitions = _merge_sequence(centroids, keyword_sets, min_k=1)
    bootstrap_partitions = _bootstrap_partitions(
        np.asarray(embeddings) if embeddings is not None else None,
        base_of_doc,
        m,
        random_state=random_state,
        n_bootstraps=n_bootstraps,
    )

    min_k = 1 if isinstance(requested, int) else 2
    min_k = min(min_k, m)
    candidate_ks = _candidate_grid(
        m, min_k, requested if isinstance(requested, int) else None
    )
    min_topic_size = int(max(3, min(20, round(0.01 * n_docs))))

    candidates: List[TopicCandidate] = []
    for k in candidate_ks:
        partition = full_partitions.get(k)
        if partition is None:
            continue
        members_by_cluster = _members_by_cluster(partition)
        sizes = {
            cluster: int(sum(base_sizes[idx] for idx in members))
            for cluster, members in members_by_cluster.items()
        }
        cluster_terms = _cluster_terms(
            members_by_cluster,
            base_term_counts,
            terms,
            keyword_sets,
            top_n_words,
        )
        coherence = _npmi_coherence(
            cluster_terms, X_bin, term_index, n_docs
        )
        diversity = _diversity(cluster_terms)
        redundancy = _redundancy(cluster_terms)
        balance = _balance(list(sizes.values()))
        stability = _stability(
            partition, bootstrap_partitions, k
        )
        smallest = min(sizes.values()) if sizes else 0
        candidates.append(
            TopicCandidate(
                k=int(k),
                coherence=coherence,
                diversity=float(diversity),
                stability=stability,
                redundancy=float(redundancy),
                balance=float(balance),
                min_topic_size=int(smallest),
                eligible=smallest >= min_topic_size,
                note="" if smallest >= min_topic_size else "tópico menor al mínimo",
            )
        )

    _score_candidates(candidates)
    if not any(candidate.eligible for candidate in candidates):
        for candidate in candidates:
            candidate.eligible = True
            candidate.note = "sin restricción de tamaño"

    selected = _choose_candidate(candidates, requested)
    reason = _build_reason(selected, candidates, requested, m, n_docs)
    selected.selected = True

    partition = full_partitions[selected.k]
    members_by_cluster = _members_by_cluster(partition)
    ordered_clusters = sorted(
        members_by_cluster,
        key=lambda cluster: (
            -sum(base_sizes[idx] for idx in members_by_cluster[cluster]),
            str(cluster),
        ),
    )
    new_id = {cluster: idx for idx, cluster in enumerate(ordered_clusters)}
    mapping = {
        base_ids[idx]: int(new_id[cluster])
        for idx, cluster in partition.items()
    }
    for tid in outlier_ids:
        mapping[tid] = -1

    cluster_terms = _cluster_terms(
        members_by_cluster,
        base_term_counts,
        terms,
        keyword_sets,
        top_n_words,
    )
    new_keywords: Dict[int, List[str]] = {}
    new_names: Dict[int, str] = {}
    for cluster in ordered_clusters:
        tid = int(new_id[cluster])
        cluster_words = cluster_terms.get(cluster, [])
        new_keywords[tid] = cluster_words
        label = ", ".join(cluster_words[:3]) if cluster_words else ""
        new_names[tid] = f"Topic {tid}: {label}" if label else f"Topic {tid}"

    if outlier_ids:
        outlier_words = _outlier_terms(
            outlier_ids, base_keywords, X_bin, term_index, base_of_doc, top_n_words
        )
        new_keywords[-1] = outlier_words
        new_names[-1] = "Topic -1: outliers"

    summary = TopicSelectionSummary(
        method="merge_hierarchy_npmi_stability",
        requested=str(requested),
        selected_k=int(selected.k),
        base_topics=int(m),
        n_docs=int(n_docs),
        n_outliers=int(n_outliers),
        reason=reason,
        candidates=sorted(candidates, key=lambda item: item.k),
    )
    return {
        "summary": summary.to_dict(),
        "mapping": mapping,
        "keywords": new_keywords,
        "topic_names": new_names,
    }


# ---------------------------------------------------------------------------
# Matrices de términos y métricas
# ---------------------------------------------------------------------------


def _build_doc_term_matrix(
    texts: Sequence[str],
) -> Tuple[Optional[Any], List[str], Dict[str, int]]:
    """Matriz binaria documento-término (frecuencia de documento)."""
    stop_words = _stopword_list()
    configs = [
        {"stop_words": stop_words, "token_pattern": r"(?u)\b\w{3,}\b"},
        {"stop_words": None, "token_pattern": r"(?u)\b\w{3,}\b"},
        {"stop_words": None, "token_pattern": r"(?u)\b\w{2,}\b"},
    ]
    for config in configs:
        try:
            vectorizer = CountVectorizer(
                lowercase=True,
                min_df=1,
                max_features=5000,
                **config,
            )
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            continue
        if matrix.shape[1] == 0:
            continue
        binary = (matrix > 0).astype(np.float32).tocsr()
        terms = [str(term) for term in vectorizer.get_feature_names_out()]
        return binary, terms, {term: idx for idx, term in enumerate(terms)}
    return None, [], {}


def _stopword_list() -> Optional[List[str]]:
    if _nltk_stopwords is None:
        return None
    words = set()
    for language in ("spanish", "english"):
        try:
            words.update(_nltk_stopwords.words(language))
        except Exception:  # pragma: no cover - recursos no descargados
            pass
    return sorted(words) if words else None


def _base_term_counts(
    X_bin: Optional[Any],
    base_of_doc: np.ndarray,
    m: int,
    n_terms: int,
) -> Optional[np.ndarray]:
    if X_bin is None or n_terms == 0:
        return None
    counts = np.zeros((m, n_terms), dtype=np.float32)
    for idx in range(m):
        rows = np.flatnonzero(base_of_doc == idx)
        if len(rows):
            counts[idx] = np.asarray(X_bin[rows].sum(axis=0)).ravel()
    return counts


def _cluster_terms(
    members_by_cluster: Dict[int, List[int]],
    base_term_counts: Optional[np.ndarray],
    terms: List[str],
    keyword_sets: List[set],
    top_n: int,
) -> Dict[int, List[str]]:
    result: Dict[int, List[str]] = {}
    for cluster, members in members_by_cluster.items():
        if base_term_counts is not None:
            counts = base_term_counts[members].sum(axis=0)
            order = np.argsort(counts)[::-1][:top_n]
            result[cluster] = [
                terms[idx] for idx in order if counts[idx] > 0
            ]
        else:
            merged = set()
            for member in members:
                merged.update(keyword_sets[member])
            result[cluster] = sorted(merged)[:top_n]
    return result


def _npmi_coherence(
    cluster_terms: Dict[int, List[str]],
    X_bin: Optional[Any],
    term_index: Dict[str, int],
    n_docs: int,
) -> Optional[float]:
    if X_bin is None or n_docs < 2:
        return None
    unique = sorted(
        {
            term
            for cluster_words in cluster_terms.values()
            for term in cluster_words
            if term in term_index
        }
    )
    if len(unique) < 2:
        return None
    indices = [term_index[term] for term in unique]
    sub = X_bin[:, indices]
    doc_freq = np.asarray(sub.sum(axis=0)).ravel().astype(float)
    cooc = np.asarray((sub.T @ sub).todense()).astype(float)
    p_i = np.clip(doc_freq / n_docs, 1e-12, 1.0)
    p_ij = np.clip(cooc / n_docs, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi = np.log(p_ij / np.outer(p_i, p_i)) / (-np.log(p_ij))
    npmi = np.where((p_ij > 0) & np.isfinite(npmi), npmi, -1.0)
    np.fill_diagonal(npmi, 0.0)
    position = {term: idx for idx, term in enumerate(unique)}

    scores = []
    for cluster_words in cluster_terms.values():
        ids = [position[term] for term in cluster_words if term in position]
        if len(ids) < 2:
            continue
        values = [
            npmi[ids[i], ids[j]]
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
        ]
        if values:
            scores.append(float(np.mean(values)))
    return float(np.mean(scores)) if scores else None


def _diversity(cluster_terms: Dict[int, List[str]]) -> float:
    all_terms = [term for words in cluster_terms.values() for term in words]
    if not all_terms:
        return 0.0
    return len(set(all_terms)) / len(all_terms)


def _redundancy(cluster_terms: Dict[int, List[str]]) -> float:
    sets = [set(words) for words in cluster_terms.values() if words]
    if len(sets) < 2:
        return 0.0
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            values.append(len(sets[i] & sets[j]) / len(union) if union else 0.0)
    return float(np.mean(values)) if values else 0.0


def _balance(sizes: Sequence[int]) -> float:
    values = np.asarray([size for size in sizes if size > 0], dtype=float)
    if len(values) <= 1:
        return 1.0
    probabilities = values / values.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return entropy / float(np.log(len(values)))


# ---------------------------------------------------------------------------
# Jerarquía de fusiones y estabilidad
# ---------------------------------------------------------------------------


def _centroids(
    embeddings: Optional[np.ndarray],
    base_of_doc: np.ndarray,
    m: int,
    doc_indices: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if embeddings is None or len(embeddings) != len(base_of_doc):
        return None
    if doc_indices is None:
        doc_indices = np.arange(len(base_of_doc))
    matrix = np.asarray(embeddings, dtype=float)
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    sampled_base = base_of_doc[doc_indices]
    valid = sampled_base >= 0
    if not np.any(valid):
        return None
    sums = np.zeros((m, matrix.shape[1]), dtype=float)
    np.add.at(sums, sampled_base[valid], matrix[doc_indices[valid]])
    counts = np.bincount(sampled_base[valid], minlength=m).astype(float)
    centroids = np.zeros_like(sums)
    present = counts > 0
    centroids[present] = sums[present] / counts[present, None]
    centroids[present] = centroids[present] / np.clip(
        np.linalg.norm(centroids[present], axis=1, keepdims=True), 1e-12, None
    )
    return centroids


def _merge_sequence(
    centroids: Optional[np.ndarray],
    keyword_sets: Sequence[set],
    min_k: int = 1,
) -> Dict[int, Dict[int, int]]:
    """Particiones base_index -> cluster para cada número de clusters."""
    m = len(keyword_sets)
    if m == 0:
        return {}
    clusters = {idx: [idx] for idx in range(m)}
    sizes = {idx: 1 for idx in range(m)}
    vectors = (
        {idx: centroids[idx].copy() for idx in range(m)}
        if centroids is not None
        else None
    )
    words = {idx: set(keyword_sets[idx]) for idx in range(m)}
    partitions = {m: _labels(clusters)}

    while len(clusters) > 1:
        ids = list(clusters)
        best: Optional[Tuple[float, int, int]] = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if vectors is not None:
                    similarity = float(np.dot(vectors[a], vectors[b]))
                else:
                    union = words[a] | words[b]
                    similarity = len(words[a] & words[b]) / len(union) if union else 0.0
                if best is None or similarity > best[0]:
                    best = (similarity, a, b)
        if best is None:
            break
        _, a, b = best
        clusters[a].extend(clusters[b])
        if vectors is not None:
            total = sizes[a] + sizes[b]
            vectors[a] = (
                vectors[a] * sizes[a] + vectors[b] * sizes[b]
            ) / max(total, 1)
            vectors[a] = vectors[a] / max(np.linalg.norm(vectors[a]), 1e-12)
        sizes[a] = sizes[a] + sizes[b]
        words[a].update(words[b])
        del clusters[b]
        del sizes[b]
        del words[b]
        if vectors is not None:
            del vectors[b]
        k = len(clusters)
        if k >= min_k:
            partitions[k] = _labels(clusters)
    return partitions


def _labels(clusters: Dict[int, List[int]]) -> Dict[int, int]:
    return {
        member: cluster
        for cluster, members in clusters.items()
        for member in members
    }


def _members_by_cluster(partition: Dict[int, int]) -> Dict[int, List[int]]:
    members: Dict[int, List[int]] = {}
    for idx, cluster in partition.items():
        members.setdefault(cluster, []).append(idx)
    for cluster in members:
        members[cluster].sort()
    return members


def _bootstrap_partitions(
    embeddings: Optional[np.ndarray],
    base_of_doc: np.ndarray,
    m: int,
    random_state: int,
    n_bootstraps: int,
) -> List[Dict[int, Dict[int, int]]]:
    if embeddings is None or n_bootstraps <= 0:
        return []
    rng = np.random.default_rng(random_state)
    result = []
    n_docs = len(base_of_doc)
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_docs, size=n_docs)
        counts = np.bincount(
            base_of_doc[indices][base_of_doc[indices] >= 0], minlength=m
        )
        present = np.flatnonzero(counts > 0)
        if len(present) < 2:
            continue
        centroids = _centroids(embeddings, base_of_doc, m, indices)
        if centroids is None:
            continue
        local_centroids = centroids[present]
        local_keywords = [set() for _ in present]
        local_partitions = _merge_sequence(local_centroids, local_keywords, min_k=2)
        result.append(
            {
                k: {
                    int(present[idx]): cluster
                    for idx, cluster in partition.items()
                }
                for k, partition in local_partitions.items()
            }
        )
    return result


def _stability(
    full_partition: Dict[int, int],
    bootstrap_partitions: Sequence[Dict[int, Dict[int, int]]],
    k: int,
) -> Optional[float]:
    scores = []
    for partitions in bootstrap_partitions:
        boot = partitions.get(k)
        if boot is None:
            continue
        common = sorted(set(full_partition) & set(boot))
        if len(common) < 2:
            continue
        scores.append(
            adjusted_rand_score(
                [full_partition[idx] for idx in common],
                [boot[idx] for idx in common],
            )
        )
    return float(np.mean(scores)) if scores else None


# ---------------------------------------------------------------------------
# Puntaje, parsimonia y utilidades
# ---------------------------------------------------------------------------


def _score_candidates(candidates: List[TopicCandidate]) -> None:
    if not candidates:
        return
    metric_values = {
        "coherence": [c.coherence for c in candidates],
        "stability": [c.stability for c in candidates],
        "diversity": [c.diversity for c in candidates],
        "balance": [c.balance for c in candidates],
        "redundancy": [1.0 - c.redundancy for c in candidates],
    }
    normalized = {
        name: _minmax(values) for name, values in metric_values.items()
    }
    for idx, candidate in enumerate(candidates):
        available = {
            name: DEFAULT_WEIGHTS[name]
            for name in DEFAULT_WEIGHTS
            if metric_values[name][idx] is not None
        }
        total_weight = sum(available.values())
        if total_weight <= 0:
            candidate.score = 0.0
            continue
        score = sum(
            weight * normalized[name][idx]
            for name, weight in available.items()
        )
        candidate.score = float(score / total_weight)


def _minmax(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return [None for _ in values]
    low, high = min(numeric), max(numeric)
    if high - low <= 1e-12:
        return [1.0 if value is not None else None for value in values]
    return [
        (float(value) - low) / (high - low) if value is not None else None
        for value in values
    ]


def _candidate_grid(
    m: int,
    min_k: int,
    requested: Optional[int] = None,
) -> List[int]:
    k_min = max(1, min(min_k, m))
    if m <= MAX_CANDIDATES:
        ks = set(range(k_min, m + 1))
    else:
        ks = {
            int(round(value))
            for value in np.linspace(k_min, m, MAX_CANDIDATES)
        }
        ks.add(m)
    if requested is not None:
        ks.add(max(1, min(int(requested), m)))
    return sorted(k for k in ks if 1 <= k <= m)


def _choose_candidate(
    candidates: List[TopicCandidate],
    requested: Any,
) -> TopicCandidate:
    by_k = {candidate.k: candidate for candidate in candidates}
    if isinstance(requested, int):
        return by_k[max(1, min(requested, max(by_k)))]
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        eligible = candidates
    best = max(eligible, key=lambda candidate: candidate.score)
    tied = [
        candidate
        for candidate in eligible
        if candidate.score >= best.score - PARSIMONY_TOLERANCE
    ]
    return min(tied, key=lambda candidate: candidate.k)


def _build_reason(
    selected: TopicCandidate,
    candidates: List[TopicCandidate],
    requested: Any,
    base_topics: int,
    n_docs: int,
) -> str:
    if isinstance(requested, int):
        if requested > base_topics:
            return (
                f"Se solicitaron {requested} tópicos, pero el modelo base solo "
                f"produjo {base_topics}; se conservan {base_topics}."
            )
        return (
            f"k={selected.k} fijado por el usuario (num_topicos={requested}). "
            "No se aplicó la selección automática."
        )

    best = max(
        [candidate for candidate in candidates if candidate.eligible] or candidates,
        key=lambda candidate: candidate.score,
    )
    tied = [
        candidate
        for candidate in candidates
        if candidate.eligible
        and candidate.score >= best.score - PARSIMONY_TOLERANCE
    ]
    tied_ks = ", ".join(str(candidate.k) for candidate in sorted(tied, key=lambda c: c.k))
    parts = [
        f"Se eligió k={selected.k} de {base_topics} tópicos base "
        f"({n_docs:,} comentarios).",
        f"Puntaje compuesto={selected.score:.3f}; mejor={best.score:.3f}.",
        f"Regla de parsimonia: de las soluciones dentro de {PARSIMONY_TOLERANCE:.2f} "
        f"del mejor puntaje (k={tied_ks}), se tomó la menor.",
        (
            "Evidencia: "
            f"coherencia NPMI={_fmt(selected.coherence)}, "
            f"estabilidad ARI={_fmt(selected.stability)}, "
            f"diversidad={selected.diversity:.3f}, "
            f"redundancia={selected.redundancy:.3f}, "
            f"balance={selected.balance:.3f}."
        ),
    ]
    discarded = [
        candidate.k
        for candidate in candidates
        if not candidate.eligible and candidate.min_topic_size > 0
    ]
    if discarded:
        parts.append(
            "Descartados por tamaño mínimo de tópico: "
            + ", ".join(str(k) for k in sorted(discarded))
            + "."
        )
    return " ".join(parts)


def _fmt(value: Optional[float]) -> str:
    return "n/d" if value is None else f"{value:.3f}"


def _ordered_unique(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _normalize_keywords(keywords: Optional[Dict[Any, Any]]) -> Dict[str, List[str]]:
    if not isinstance(keywords, dict):
        return {}
    normalized: Dict[str, List[str]] = {}
    for key, value in keywords.items():
        if isinstance(value, str):
            terms = [item.strip() for item in value.replace(";", ",").split(",")]
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            terms = []
            for item in value:
                if isinstance(item, (list, tuple)) and item:
                    terms.append(str(item[0]))
                else:
                    terms.append(str(item))
        else:
            terms = [str(value)]
        normalized[str(key)] = [term for term in terms if term]
    return normalized


def _keyword_sets(
    base_ids: Sequence[str],
    base_keywords: Dict[str, List[str]],
    base_term_counts: Optional[np.ndarray],
    terms: Sequence[str],
    top_n: int,
) -> List[set]:
    sets = []
    for idx, base_id in enumerate(base_ids):
        words = set(base_keywords.get(base_id, []))
        if not words and base_term_counts is not None:
            counts = base_term_counts[idx]
            order = np.argsort(counts)[::-1][:top_n]
            words = {terms[j] for j in order if counts[j] > 0}
        sets.append(words)
    return sets


def _outlier_terms(
    outlier_ids: Sequence[str],
    base_keywords: Dict[str, List[str]],
    X_bin: Optional[Any],
    term_index: Dict[str, int],
    base_of_doc: np.ndarray,
    top_n: int,
) -> List[str]:
    words = []
    for tid in outlier_ids:
        words.extend(base_keywords.get(tid, []))
    if words:
        return list(dict.fromkeys(words))[:top_n]
    if X_bin is None:
        return []
    rows = np.flatnonzero(base_of_doc < 0)
    if len(rows) == 0:
        return []
    counts = np.asarray(X_bin[rows].sum(axis=0)).ravel()
    order = np.argsort(counts)[::-1][:top_n]
    reverse = {idx: term for term, idx in term_index.items()}
    return [reverse[idx] for idx in order if counts[idx] > 0]
