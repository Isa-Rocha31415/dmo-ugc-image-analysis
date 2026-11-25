"""
joint_topic_model.py — TAREA P1: modelado de tópicos CONJUNTO (DMO+UGC).

Metodología
-----------
1. Un único modelo base sobre TODO el corpus limpio, elegido por `--model`:
   BERTopic (UMAP+HDBSCAN) o XidooTopic (UMAP+Leiden+Filtros+Merge). Ambos
   alimentan el mismo c-TF-IDF con filtros min_df/max_df (sin stopwords).
2. Grid de k (default 4..30): los tópicos base se fusionan a exactamente k con
   clustering aglomerativo WARD sobre centroides L2-normalizados. Ward minimiza el
   incremento de varianza intra-grupo, penaliza fusionar clusters grandes y evita
   el mega-tópico + satélites que producía average-linkage con coseno.
3. Métricas por k: coherencia NPMI local (co-ocurrencia documental booleana),
   diversidad (palabras únicas en top-10 / total), estabilidad (5 submuestras 80%,
   HDBSCAN re-ajustado en el espacio UMAP fijo, ARI/AMI por pares), tamaño mínimo,
   proporción del tópico mayor (max_share), balance entrópico (H/log2 K) y
   proporción sin asignar (tras reasignación por coseno).
4. Score compuesto (pesos explícitos en WEIGHTS) + GUARDAS ANTI-DEGENERACIÓN
   (max_share y tamaño mínimo) + regla de parsimonia (tol 0.02).
5. Outliers del modelo: reasignación por similitud coseno al centroide (equivalente
   a reduce_outliers con embeddings). Los restantes son la categoría explícita
   SIN_ASIGNAR (topic_id=-1); nunca se eliminan. Se reportan crudo y post-reasignación.
6. Desbalance DMO<<UGC: NO se remuestrea. --dmo-weight pondera centroides y c-TF-IDF;
   la comparación entre corpus se hace por proporciones con bootstrap (tarea P3).

Outputs en ../results/comparison/topics_joint/<backend>/:
  topic_selection_grid.csv, curvas_k.html, topics_summary.csv, doc_topics.csv
  (+ topic_centroids.npy y caché propia en _cache/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize

from bertopic_modeler import BertopicModeler
from comparison.common import (
    append_metadata, joint_backend, palette_colors, print_header, save_csv,
    topics_joint_dir,
)
from io_corpus import CORPUS_COLUMN
from utils import palette_to_plot_colors

UNASSIGNED = "SIN_ASIGNAR"
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"
PARSIMONY_TOL = 0.02
GRID_VERSION = 5         # 5: analysis_base incluye text_norm; representación sin cambio
MAX_TOPIC_SHARE = 0.20   # ningún tópico puede acaparar >20% de los docs asignados
MIN_TOPIC_FLOOR = 50     # tamaño mínimo absoluto de tópico

WEIGHTS: dict[str, float] = {
    "coherence": 0.30,
    "stability": 0.25,
    "diversity": 0.10,   # ↓ sesgo mecánico hacia k pequeño
    "coverage": 0.05,
    "balance": 0.30,     # ↑ ahora sí penaliza mega-tópicos
}

REQUIRED_GRID_COLS = {
    "k", "coherence", "diversity", "stability", "stability_ami",
    "min_topic_size", "max_share", "balance_entropy", "n_effective",
    "unassigned_share", "n_assigned",
}


def backend_version(backend: str) -> str:
    """Versión del backend, parte de la huella de caché del PASO 9."""
    if backend == "xidoo":
        from xidoo_topic_modeler import xidoo_backend_version
        return xidoo_backend_version()
    try:
        from importlib import metadata
        return metadata.version("bertopic")
    except Exception:
        return "desconocida"


def base_cache_payload(emb: np.ndarray, args: Any, backend: str) -> dict[str, Any]:
    """Huella del modelo base. El backend cambia la clave siempre."""
    return {
        "backend": backend,
        "backend_version": backend_version(backend),
        "rows": emb.shape[0], "dim": emb.shape[1], "seed": args.seed,
        "base_min_topic_size": args.base_min_topic_size,
        "min_df": args.min_df, "max_df": args.max_df, "ngram_max": args.ngram_max,
        "emb_sha": hashlib.sha256(emb.tobytes()).hexdigest(),
    }


def grid_cache_payload(ks: list[int], args: Any, backend: str,
                       base_key: str) -> dict[str, Any]:
    """Huella del grid. Incluye backend y la clave base, nunca compartida."""
    return {
        "grid_version": GRID_VERSION,
        "backend": backend,
        "backend_version": backend_version(backend),
        "base_key": base_key, "ks": ks, "runs": args.stability_runs,
        "subsample": args.subsample, "seed": args.seed,
        "threshold": args.reassign_threshold, "dmo_weight": args.dmo_weight,
        "weights": WEIGHTS,
    }


def build_count_vectorizer(min_df: int = 5, max_df: float = 0.40,
                           ngram_max: int = 2) -> CountVectorizer:
    """Vectorizador SIN listas de stopwords (ni archivos ni descargas).

    Solo token alfabético unicode de longitud>=3 en minúsculas, más dos filtros
    estadísticos estimados del PROPIO corpus combinado (DMO+UGC):
      · min_df: elimina tipos raros (ruido tipográfico, hapax).
      · max_df: elimina tipos ubicuos (proxy de palabras funcionales).
    Se ajusta sobre el corpus combinado, no por grupo, para no introducir asimetría
    entre DMO y UGC. max_df=1.0 desactiva el filtro.
    """
    return CountVectorizer(
        token_pattern=TOKEN_PATTERN, lowercase=True,
        min_df=min_df, max_df=max_df,
        ngram_range=(1, ngram_max), max_features=60000,
    )


# ─────────────────────────── Funciones puras ───────────────────────────
def _centroids(
    labels: np.ndarray, emb: np.ndarray, w: np.ndarray
) -> tuple[list[int], np.ndarray]:
    """Centroides ponderados por tópico (excluye -1)."""
    uniq = sorted(int(t) for t in set(labels.tolist()) if t >= 0)
    if not uniq:
        return [], np.empty((0, emb.shape[1]))
    cents = np.vstack(
        [np.average(emb[labels == t], axis=0, weights=w[labels == t]) for t in uniq]
    )
    return uniq, cents


def merge_to_k(
    base_labels: np.ndarray, emb: np.ndarray, k: int, w: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """Fusiona tópicos base a exactamente k con Ward sobre centroides L2-normalizados.

    Ward minimiza el incremento de varianza intra-grupo, lo que penaliza fusionar
    clusters grandes y produce particiones mucho más balanceadas que average-linkage
    con coseno (que encadena y genera un mega-tópico + satélites). Sobre vectores
    normalizados la distancia euclídea es monótona en la coseno, así que se conserva
    la geometría semántica.
    """
    uniq, cents = _centroids(base_labels, emb, w)
    if len(uniq) <= k:
        return base_labels.copy(), {t: t for t in uniq}
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    agg.fit(normalize(cents))
    mapping = {t: int(c) for t, c in zip(uniq, agg.labels_)}
    merged = np.array([mapping.get(int(t), -1) for t in base_labels], dtype=int)
    return merged, mapping


def size_stats(labels: np.ndarray, k: int) -> tuple[int, float, float, float]:
    """Estadísticos de balance de la partición (excluye SIN_ASIGNAR).

    Returns:
        (min_topic_size, max_share, entropía_normalizada, topicos_efectivos)
        · entropía normalizada = H/log2(K) ∈ [0,1]; 1 = tamaños uniformes, →0 = mega-tópico.
        · tópicos efectivos = K**H_norm (perplejidad de la distribución de tamaños):
          k=4 con un tópico al 98% da ≈1.2, delatando la degeneración.
    """
    sizes = np.bincount(labels[labels >= 0], minlength=k)
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return 0, 1.0, 0.0, 0.0
    shares = sizes / sizes.sum()
    K = int(sizes.size)
    ent = float(-(shares * np.log2(shares)).sum() / np.log2(K)) if K > 1 else 0.0
    return int(sizes.min()), float(shares.max()), ent, float(K ** ent)


def reassign_and_probs(
    labels: np.ndarray, emb: np.ndarray, w: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    """Reasigna -1 por coseno al centroide si sim >= threshold; da prob-proxy a todos."""
    uniq, cents = _centroids(labels, emb, w)
    if not uniq:
        return labels.copy(), np.zeros(len(labels)), [], cents
    sims = normalize(emb) @ normalize(cents).T
    best = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    out = labels.copy()
    mask = (labels < 0) & (best_sim >= threshold)
    out[mask] = np.asarray(uniq)[best[mask]]
    colmap = {t: i for i, t in enumerate(uniq)}
    cols = np.array([colmap.get(int(t), -1) for t in out])
    prob = np.where(cols >= 0, sims[np.arange(len(out)), np.maximum(cols, 0)], best_sim)
    return out, np.clip(prob, 0.0, 1.0), uniq, cents


def ctfidf_terms(
    labels: np.ndarray, X: sparse.spmatrix, vocab: np.ndarray,
    w: np.ndarray, top_n: int = 10,
) -> dict[int, list[str]]:
    """c-TF-IDF local: (tf_clase / palabras_clase) * log(1 + K / (1 + df_clases))."""
    uniq = sorted(int(t) for t in set(labels.tolist()) if t >= 0)
    if not uniq:
        return {}
    pos = {t: i for i, t in enumerate(uniq)}
    cols = np.flatnonzero(labels >= 0)
    rows = np.array([pos[int(labels[i])] for i in cols])
    S = sparse.csr_matrix((w[cols], (rows, cols)), shape=(len(uniq), X.shape[0]))
    TF = np.asarray((S @ X).todense(), dtype=float)
    words = TF.sum(axis=1, keepdims=True)
    words[words == 0] = 1.0
    df_t = (TF > 0).sum(axis=0)
    idf = np.log(1.0 + len(uniq) / (1.0 + df_t))
    scores = (TF / words) * idf
    order = np.argsort(-scores, axis=1)
    return {t: [str(vocab[j]) for j in order[i, :top_n]] for i, t in enumerate(uniq)}


def npmi_coherence(
    terms_by_topic: dict[int, list[str]], B: sparse.spmatrix, vocab: np.ndarray
) -> float:
    """NPMI medio (pares del top-10 por tópico), co-ocurrencia documental booleana."""
    if not terms_by_topic:
        return 0.0
    vindex = {str(t): i for i, t in enumerate(vocab)}
    all_terms = sorted({t for ts in terms_by_topic.values() for t in ts[:10]})
    idx = [vindex[t] for t in all_terms if t in vindex]
    if len(idx) < 2:
        return 0.0
    Bt = B[:, idx]
    N = B.shape[0]
    df = np.asarray(Bt.sum(axis=0)).ravel().astype(float)
    cooc = np.asarray((Bt.T @ Bt).todense(), dtype=float)
    p = df / N
    pij = cooc / N
    denom = -np.log(pij + 1e-12)
    num = np.log((pij + 1e-12) / (np.outer(p, p) + 1e-12))
    npm = np.divide(num, denom, out=np.full_like(num, -1.0), where=denom > 1e-9)
    npm[cooc == 0] = -1.0
    np.fill_diagonal(npm, 0.0)
    tpos = {t: i for i, t in enumerate(all_terms)}
    vals: list[float] = []
    for ts in terms_by_topic.values():
        ids = [tpos[t] for t in ts[:10] if t in tpos]
        if len(ids) < 2:
            continue
        sub = npm[np.ix_(ids, ids)]
        iu = np.triu_indices(len(ids), 1)
        vals.append(float(sub[iu].mean()))
    return float(np.mean(vals)) if vals else 0.0


def topic_diversity(terms_by_topic: dict[int, list[str]]) -> float:
    """Proporción de palabras únicas en el top-10 de todos los tópicos."""
    words = [t for ts in terms_by_topic.values() for t in ts[:10]]
    return len(set(words)) / max(1, len(words))


def composite_score(grid: pd.DataFrame, weights: dict[str, float] = WEIGHTS) -> np.ndarray:
    """Score compuesto: coherencia (min-max), estabilidad, diversidad, cobertura y
    balance = entropía normalizada de la distribución de tamaños (H/log2 K)."""
    coh = grid["coherence"].to_numpy(dtype=float)
    rng = coh.max() - coh.min()
    coh_n = (coh - coh.min()) / rng if rng > 1e-12 else np.ones_like(coh)
    coverage = 1.0 - grid["unassigned_share"].to_numpy(dtype=float)
    balance = grid["balance_entropy"].to_numpy(dtype=float)
    return (
        weights["coherence"] * coh_n
        + weights["stability"] * grid["stability"].to_numpy(dtype=float)
        + weights["diversity"] * grid["diversity"].to_numpy(dtype=float)
        + weights["coverage"] * coverage
        + weights["balance"] * balance
    )


def select_k(grid: pd.DataFrame, tol: float = PARSIMONY_TOL,
             n_docs: int | None = None) -> tuple[int, str]:
    """Elige k con guardas anti-degeneración + parsimonia.

    Descarta soluciones con un mega-tópico (max_share > MAX_TOPIC_SHARE) o con
    tópicos minúsculos (min_topic_size < floor). Si ninguna cumple, relaja de forma
    GRADUADA (primero el tamaño mínimo, luego max_share a 0.35 y 0.60) y lo deja
    explícito en la justificación.
    """
    floor = max(MIN_TOPIC_FLOOR, int(0.002 * (n_docs or 0)))
    notas = [f"Guardas: max_share≤{MAX_TOPIC_SHARE} y min_topic_size≥{floor}."]
    ok = (grid["max_share"] <= MAX_TOPIC_SHARE) & (grid["min_topic_size"] >= floor)
    if not ok.any():
        ok = grid["max_share"] <= MAX_TOPIC_SHARE
        notas.append(f"Ningún k cumplió el tamaño mínimo; se relajó min_topic_size "
                     f"(queda solo max_share≤{MAX_TOPIC_SHARE}).")
    for lim in (0.35, 0.60):
        if ok.any() or lim <= MAX_TOPIC_SHARE:
            break
        ok = grid["max_share"] <= lim
        notas.append(f"Se relajó max_share≤{lim}.")
    if not ok.any():
        ok = pd.Series(True, index=grid.index)
        notas.append("Ningún k cumplió ni las guardas relajadas; se evaluaron todos "
                     "(revisar el modelo: --max-df, --k-max).")

    sub = grid[ok]
    best = float(sub["composite_score"].max())
    eligible = sorted(int(x) for x in sub.loc[sub["composite_score"] >= best - tol, "k"])
    k = int(min(eligible))
    r = grid.loc[grid["k"] == k].iloc[0]
    excl = sorted(int(x) for x in grid.loc[~ok, "k"])
    just = (
        f"Se eligió k={k} porque maximiza el score compuesto ({r.composite_score:.3f}) "
        f"entre las soluciones NO degeneradas, con coherencia NPMI={r.coherence:.3f}, "
        f"diversidad={r.diversity:.3f}, estabilidad ARI={r.stability:.3f} "
        f"(AMI={r.stability_ami:.3f}), balance entrópico={r.balance_entropy:.3f} "
        f"(≈{r.n_effective:.1f} tópicos efectivos), tópico mayor={100*r.max_share:.1f}% "
        f"de los docs, tamaño mínimo={int(r.min_topic_size)} y "
        f"{100*r.unassigned_share:.1f}% sin asignar. Pesos={WEIGHTS}. "
        f"{' '.join(notas)} k excluidos por degeneración: {excl}. "
        f"Parsimonia (tol={tol}): entre {eligible} se tomó el menor."
    )
    return k, just


# ─────────────────── Modelo base (reutiliza bertopic_modeler) ───────────────────
class JointBertopicModeler(BertopicModeler):
    """Extensión no invasiva: vectorizador sin stopwords + acceso al modelo ajustado."""

    def __init__(self, *, min_df: int = 5, max_df: float = 0.40,
                 ngram_max: int = 2, **kwargs: Any):
        super().__init__(**kwargs)
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_max = ngram_max
        self._model: Any = None

    def _build_model(self, has_embeddings: bool, n_docs: int = 10):
        model = super()._build_model(has_embeddings=has_embeddings, n_docs=n_docs)
        model.vectorizer_model = build_count_vectorizer(
            self.min_df, self.max_df, self.ngram_max
        )
        model.verbose = False
        self._model = model
        return model


def _fit_base_cached(
    texts: list[str], emb: np.ndarray, args: argparse.Namespace, cache_dir: Path,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """Ajusta (o reusa) el modelo base del backend elegido.

    Cada backend tiene su propia clave y su propio directorio. Los fallos de
    XidooTopic se propagan: nunca se cae al modelo BERTopic.
    """
    payload = base_cache_payload(emb, args, backend)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f"base_{key}.npz"
    if npz.exists() and not args.force_topics and not getattr(args, "no_cache", False):
        d = np.load(npz)
        print(f"  ✅ Caché hit [topics_joint/{backend}/base] -> {npz}")
        return d["reduction"], d["base_labels"], int(d["mcs"]), int(d["ms"]), key

    if backend == "xidoo":
        from xidoo_topic_modeler import XidooTopicModeler
        print("  🧮 Entrenando modelo base conjunto (XidooTopic, una sola vez)...")
        modeler = XidooTopicModeler(
            random_state=args.seed, use_internal_outlier=True,
            min_cluster_size=args.base_min_topic_size,
        )
        modeler.fit_transform(texts, embeddings=emb)
        base_labels = np.asarray(modeler.base_labels_, dtype=int)
        reduction = np.asarray(modeler.reduction_, dtype=np.float32)
        mcs = int(modeler.min_cluster_size_)
        ms = 1
    else:
        try:
            from hdbscan import HDBSCAN  # noqa: F401  (verifica disponibilidad)
        except ImportError as exc:
            raise ImportError(
                "hdbscan no está disponible y no hay caché del modelo base. "
                "Instálalo en el entorno antes de ejecutar el PASO 9."
            ) from exc
        print("  🧮 Entrenando modelo base conjunto (BERTopic UMAP+HDBSCAN, una sola vez)...")
        modeler = JointBertopicModeler(
            min_df=args.min_df, max_df=args.max_df, ngram_max=args.ngram_max,
            random_state=args.seed, min_topic_size=args.base_min_topic_size,
            min_samples_per_topic=3, top_n_words=10,
        )
        res = modeler.fit_transform(texts, embeddings=emb)
        base_labels = np.asarray(res["topic_id"], dtype=int)
        reduction = np.asarray(modeler._model.umap_model.embedding_, dtype=np.float32)
        mcs = int(getattr(modeler._model.hdbscan_model, "min_cluster_size",
                          args.base_min_topic_size))
        ms_raw = getattr(modeler._model.hdbscan_model, "min_samples", None)
        ms = int(ms_raw) if ms_raw is not None else 1
    np.savez(npz, reduction=reduction, base_labels=base_labels, mcs=mcs, ms=ms)
    return reduction, base_labels, mcs, ms, key


def _stability_runs(
    reduction: np.ndarray, n: int, mcs: int, ms: int, args: argparse.Namespace,
    backend: str, texts: list[str] | None = None, emb: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Submuestras: HDBSCAN (bertopic) o XidooTopic re-ajustado (xidoo)."""
    if args.stability_runs <= 0:
        print("  ⚠️ --stability-runs 0: estabilidad = 0 para todos los k (no recomendado).")
        return []
    if backend == "xidoo":
        from xidoo_topic_modeler import XidooTopicModeler
        if texts is None or emb is None:
            raise ValueError("Estabilidad xidoo requiere textos y embeddings.")
        print(f"  🔁 Estabilidad: {args.stability_runs} re-ajustes de XidooTopic.")
    else:
        try:
            from hdbscan import HDBSCAN
        except ImportError as exc:
            raise ImportError(
                "hdbscan no está disponible; no se puede medir estabilidad. "
                "Instálalo o usa --stability-runs 0."
            ) from exc
    rng = np.random.default_rng(args.seed)
    runs = []
    for _ in range(args.stability_runs):
        idx = rng.choice(n, size=int(args.subsample * n), replace=False)
        if backend == "xidoo":
            modeler = XidooTopicModeler(
                random_state=args.seed, use_internal_outlier=True,
                min_cluster_size=mcs,
            )
            modeler.fit_transform([texts[i] for i in idx], embeddings=emb[idx])
            lab = np.asarray(modeler.base_labels_, dtype=int)
        else:
            lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(reduction[idx])
        full = np.full(n, -2, dtype=int)  # -2 = no muestreado
        full[idx] = lab
        runs.append((idx, full))
    return runs


def _stability_for_k(
    runs: list[tuple[np.ndarray, np.ndarray]], emb: np.ndarray, w: np.ndarray, k: int
) -> tuple[float, float]:
    """ARI/AMI promedio por pares sobre docs asignados en ambas corridas."""
    merged = []
    for idx, full in runs:
        m, _ = merge_to_k(full[idx], emb[idx], k, w[idx])
        f = np.full(len(full), -2, dtype=int)
        f[idx] = m
        merged.append(f)
    aris, amis = [], []
    for a in range(len(merged)):
        for b in range(a + 1, len(merged)):
            va, vb = merged[a], merged[b]
            mask = (va >= 0) & (vb >= 0)
            if mask.sum() >= 30:
                aris.append(adjusted_rand_score(va[mask], vb[mask]))
                amis.append(adjusted_mutual_info_score(va[mask], vb[mask]))
    return (float(np.mean(aris)) if aris else 0.0,
            float(np.mean(amis)) if amis else 0.0)


# ─────────────────────────── Tarea 'topics' ───────────────────────────
def run_topics_task(args: argparse.Namespace, results_dir: Path,
                    backend: str | None = None) -> list[Path]:
    """Tarea P1: modelo conjunto + selección de k en topics_joint/<backend>/."""
    backend = backend or joint_backend(args)
    print_header("PASO 9: MODELADO CONJUNTO DE TÓPICOS (DMO+UGC) — SELECCIÓN DE k")
    print(f"  🤖 Backend (--model): {backend} | versión: {backend_version(backend)}")
    base_dir = results_dir / "comparison" / "base"
    out_dir = topics_joint_dir(args, results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(base_dir / "analysis_base.csv", encoding="utf-8-sig")
    emb = np.load(base_dir / "embeddings_clean.npy")
    if len(base) != emb.shape[0]:
        raise ValueError(
            "analysis_base y embeddings_clean no están alineados; "
            "ejecuta --comparison-tasks base."
        )
    emb = emb[base["embedding_row"].to_numpy(dtype=int)]
    texts = base["text"].fillna("").astype(str).tolist()
    corpus = base[CORPUS_COLUMN].astype(str).to_numpy()
    n = len(base)
    w = np.where(corpus == "DMO", float(args.dmo_weight), 1.0)
    if args.dmo_weight != 1.0:
        print(f"  ⚖️ --dmo-weight={args.dmo_weight}: pondera centroides/c-TF-IDF, SIN remuestreo.")
    print("  ℹ️ Comparación DMO vs UGC: por proporciones con bootstrap (tarea P3).")

    reduction, base_labels, mcs, ms, base_key = _fit_base_cached(
        texts, emb, args, out_dir / "_cache", backend
    )
    n_base = len(set(base_labels.tolist()) - {-1})
    if n_base < 2:
        raise ValueError("El modelo base produjo <2 tópicos; corpus insuficiente para P1.")
    k_max = min(args.k_max, n_base)
    k_min = max(2, min(args.k_min, k_max))
    ks = list(range(k_min, k_max + 1))
    print(f"  Tópicos base: {n_base} | Grid evaluado: k={ks[0]}..{ks[-1]}")
    print(f"  🛡️ Guardas anti-degeneración: max_share≤{MAX_TOPIC_SHARE:.0%}, "
          f"min_topic_size≥{max(MIN_TOPIC_FLOOR, int(0.002 * n))}")

    print(f"  🔤 Vocabulario: token alfabético len>=3, min_df={args.min_df}, "
          f"max_df={args.max_df} (sin listas de stopwords; filtros del propio corpus).")
    vec = build_count_vectorizer(args.min_df, args.max_df, args.ngram_max)
    X = vec.fit_transform(texts)
    vocab = np.asarray(vec.get_feature_names_out())
    B = X.copy()
    B.data[:] = 1

    # ── Grid (con caché propia versionada y validada) ──
    gpayload = grid_cache_payload(ks, args, backend, base_key)
    gkey = hashlib.sha256(json.dumps(gpayload, sort_keys=True).encode()).hexdigest()[:20]
    gcache = out_dir / "_cache" / f"grid_{gkey}.csv"

    grid: pd.DataFrame | None = None
    if (gcache.exists() and not args.force_topics
            and not getattr(args, "no_cache", False)):
        cand = pd.read_csv(gcache, encoding="utf-8-sig")
        if REQUIRED_GRID_COLS.issubset(cand.columns):
            grid = cand
            print(f"  ✅ Caché hit [topics_joint/{backend}/grid] -> {gcache}")
        else:
            faltan = sorted(REQUIRED_GRID_COLS - set(cand.columns))
            print(f"  ⚠️ Caché de grid obsoleta (faltan {faltan}); se recalcula.")

    if grid is None:
        runs = _stability_runs(reduction, n, mcs, ms, args, backend, texts, emb)
        rows = []
        for k in ks:
            merged, _ = merge_to_k(base_labels, emb, k, w)
            merged, _, _, _ = reassign_and_probs(merged, emb, w, args.reassign_threshold)
            terms = ctfidf_terms(merged, X, vocab, w, top_n=10)
            min_size, max_share, ent, n_eff = size_stats(merged, k)
            stab, ami = _stability_for_k(runs, emb, w, k)
            rows.append({
                "k": k,
                "coherence": npmi_coherence(terms, B, vocab),
                "diversity": topic_diversity(terms),
                "stability": stab,
                "stability_ami": ami,
                "min_topic_size": min_size,
                "max_share": max_share,
                "balance_entropy": ent,
                "n_effective": round(n_eff, 3),
                "unassigned_share": float((merged < 0).mean()),
                "n_assigned": int((merged >= 0).sum()),
            })
        grid = pd.DataFrame(rows)
        grid["composite_score"] = composite_score(grid)
        save_csv(grid, gcache)

    k_sel, justification = select_k(grid, n_docs=n)
    grid["selected"] = grid["k"] == k_sel
    grid["composite_score"] = composite_score(grid)  # asegura columna tras caché

    # ── Solución final con k elegido ──
    merged_pre, _ = merge_to_k(base_labels, emb, k_sel, w)
    merged, probs, uniq, cents = reassign_and_probs(
        merged_pre, emb, w, args.reassign_threshold
    )
    terms8 = ctfidf_terms(merged, X, vocab, w, top_n=8)
    names = {t: ", ".join(terms8.get(t, [])[:3]) or f"Topic {t}" for t in uniq}
    names[-1] = UNASSIGNED

    # ── doc_topics.csv (cobertura total verificada) ──
    doc_topics = pd.DataFrame({
        "doc_id": base["doc_id"].to_numpy(dtype=int),
        "topic_id": merged,
        "topic_prob": np.round(probs, 4),
        "topic_name": [names[int(t)] for t in merged],
    })
    if not (len(doc_topics) == n and set(doc_topics.doc_id) == set(base.doc_id)):
        raise ValueError("doc_topics no cubre exactamente los doc_id de analysis_base.")

    # ── topics_summary.csv (proporciones por corpus para P3) ──
    n_dmo = int((corpus == "DMO").sum())
    n_ugc = n - n_dmo
    rows = []
    for t in uniq + [-1]:
        m = merged == t
        nd = int((m & (corpus == "DMO")).sum())
        nu = int((m & (corpus == "UGC")).sum())
        rows.append({
            "topic_id": t,
            "topic_name": names[t],
            "keywords_top8": "; ".join(terms8.get(t, [])),
            "n_docs": int(m.sum()),
            "n_dmo": nd, "n_ugc": nu,
            "share_total": round(float(m.mean()), 6),
            "share_dmo": round(nd / max(n_dmo, 1), 6),
            "share_ugc": round(nu / max(n_ugc, 1), 6),
        })
    summary = pd.DataFrame(rows)

    # ── Consola: defensa de k + narrativa ──
    print("\n  📊 Grid de selección de k:")
    print(grid.round(3).to_string(index=False))
    print(f"\n  🏆 {justification}")

    print("\n  📦 Distribución de tamaños del k elegido (top-10):")
    fin_sizes = np.bincount(merged[merged >= 0], minlength=k_sel)
    for rank, ti in enumerate(np.argsort(-fin_sizes)[:10], 1):
        if fin_sizes[ti] == 0:
            continue
        print(f"    #{rank:<2d} topic {int(ti):<3d} {names.get(int(ti), '?'):<38s} "
              f"{int(fin_sizes[ti]):>7,d} docs ({100 * fin_sizes[ti] / max(n, 1):5.2f}%)")
    mn, mx, ent, neff = size_stats(merged, k_sel)
    print(f"    → tópico mayor {100 * mx:.1f}% | mínimo {mn:,} docs | "
          f"balance {ent:.3f} | tópicos efectivos {neff:.1f} de {k_sel}")
    if mx > MAX_TOPIC_SHARE:
        print(f"    ⚠️ El tópico mayor supera la guarda ({MAX_TOPIC_SHARE:.0%}). "
              f"Considera --max-df 0.20 o ampliar --k-max.")

    print("\n  🔁 Reasignación de outliers del modelo (coseno ≥ "
          f"{args.reassign_threshold}):")
    for corp in ("DMO", "UGC"):
        mc = corpus == corp
        tot = max(int(mc.sum()), 1)
        antes = int(((merged_pre < 0) & mc).sum())
        despues = int(((merged < 0) & mc).sum())
        print(f"    {corp}: {antes} ({100 * antes / tot:.1f}%) → "
              f"{despues} SIN_ASIGNAR ({100 * despues / tot:.1f}%)")

    print("\n  🥇 Top-3 tópicos por corpus (proporción intra-corpus):")
    for corp, col, ncol in (("DMO", "share_dmo", "n_dmo"), ("UGC", "share_ugc", "n_ugc")):
        top = summary[summary.topic_id >= 0].nlargest(3, col)
        for _, r in top.iterrows():
            print(f"    {corp} | {r.topic_name:<40s} {100 * r[col]:5.1f}%  (n={int(r[ncol])})")

    # ── Escritura de artefactos ──
    written: list[Path] = []
    written.append(save_csv(grid, out_dir / "topic_selection_grid.csv"))
    written.append(save_csv(summary, out_dir / "topics_summary.csv"))
    written.append(save_csv(doc_topics, out_dir / "doc_topics.csv"))
    np.save(out_dir / "topic_centroids.npy", cents)
    written.append(out_dir / "topic_centroids.npy")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    colors = palette_colors(palette_to_plot_colors(args.palette))
    panels = [
        ("coherence", "Coherencia NPMI"), ("diversity", "Diversidad top-10"),
        ("stability", "Estabilidad (ARI)"), ("balance_entropy", "Balance (entropía norm.)"),
        ("max_share", "Proporción del tópico mayor"), ("composite_score", "Score compuesto"),
    ]
    fig = make_subplots(rows=3, cols=2, subplot_titles=[t for _, t in panels])
    for i, (col, title) in enumerate(panels):
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(
            x=grid["k"], y=grid[col], mode="lines+markers", name=title,
            line=dict(color=colors[i % len(colors)]), showlegend=False,
        ), row=r, col=c)
        fig.add_vline(x=k_sel, line_dash="dash", line_color="#555", row=r, col=c)
    fig.add_hline(y=MAX_TOPIC_SHARE, line_dash="dot", line_color="#B22222",
                  row=3, col=1)  # guarda visual en el panel de max_share
    fig.update_layout(
        title=f"Selección de k — corpus conjunto DMO+UGC (k elegido = {k_sel})",
        height=950, template="plotly_white",
    )
    html = out_dir / "curvas_k.html"
    fig.write_html(html, include_plotlyjs=True)
    written.append(html)
    print("\n  📈 curvas_k.html (tabla de datos = topic_selection_grid.csv)")

    written.append(append_metadata(
        base_dir / "run_metadata.csv", task="topics",
        params={**vars(args), "backend": backend, "k_selected": k_sel,
                "weights": WEIGHTS,
                "grid_version": GRID_VERSION, "merge_linkage": "ward_on_l2_centroids",
                "max_topic_share_guard": MAX_TOPIC_SHARE,
                "min_topic_floor": MIN_TOPIC_FLOOR,
                "n_base_topics": n_base, "justification": justification},
        repo_dir=Path(__file__).resolve().parent.parent,
    ))
    print(f"\n  ✅ Aceptación: doc_topics cubre {n:,} doc_id (incluye {UNASSIGNED}); "
          f"selected={int(grid['selected'].sum())}")
    return written