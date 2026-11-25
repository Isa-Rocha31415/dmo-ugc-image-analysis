"""Adaptador de XidooTopic con el contrato estable del pipeline.

La API instalada (xidootopic 0.1.0) expone ``XidooTopic`` y componentes
intercambiables; ``fit_transform(texts)`` devuelve
``{"topics", "documents", "embeddings", "labels"}``. Los labels devueltos
solo cubren los documentos que sobreviven al filtro de outliers y al filtro
de clusters, por lo que este adaptador reconstruye el vector completo
(-1 = descartado) y expone la reducción UMAP para el PASO 9.
"""
from __future__ import annotations

import re
from importlib import metadata
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TOKEN_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"


def xidoo_backend_version() -> str:
    """Versión instalada de xidootopic (para huellas de caché)."""
    try:
        return metadata.version("xidootopic")
    except metadata.PackageNotFoundError:
        return "desconocida"


def _normalize_keyword_value(value: Any) -> List[str]:
    """Normaliza keywords (str, lista, tupla, dict) a lista de strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in re.split(r",|;|\|", value) if p.strip()]
    if isinstance(value, dict):
        return [str(k) for k in value.keys()]
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (list, tuple)):
        return [str(x) for x in value[0]]
    if isinstance(value, (list, tuple, set, np.ndarray)):
        out: List[str] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                out.append(str(item[0]))
            else:
                out.append(str(item))
        return out
    return [str(value)]


def normalize_xidoo_keywords(raw: Any) -> Dict[str, List[str]]:
    """Convierte el dict de tópicos de XidooTopic a ``{str(id): [palabras]}``."""
    if not isinstance(raw, dict):
        return {}
    return {str(tid): _normalize_keyword_value(value) for tid, value in raw.items()}


def expand_xidoo_labels(
    labels: Sequence[int],
    outlier_mask: Sequence[bool],
    cluster_mask: Optional[Sequence[bool]],
    n_docs: int,
) -> np.ndarray:
    """Rehila los labels de XidooTopic al corpus completo (-1 = excluido)."""
    full = np.full(int(n_docs), -1, dtype=int)
    kept = np.flatnonzero(np.asarray(outlier_mask, dtype=bool))
    if cluster_mask is not None:
        mask = np.asarray(cluster_mask, dtype=bool)
        if len(mask) != len(kept):
            raise ValueError("cluster_mask no coincide con los docs tras outliers.")
        kept = kept[mask]
    lab = np.asarray(labels, dtype=int)
    if len(lab) != len(kept):
        raise ValueError(
            f"XidooTopic devolvió {len(lab)} labels para {len(kept)} documentos."
        )
    full[kept] = lab
    return full


class _IdentityPreprocessor:
    """Conserva todas las filas y su orden (los textos ya vienen limpios)."""

    def transform(self, reviews: Sequence[str]) -> tuple[List[str], List[int]]:
        texts = [str(t) for t in reviews]
        return texts, list(range(len(texts)))


class _IdentityChunker:
    def recortaTodo(self, texts: Sequence[str]) -> List[str]:
        return list(texts)


class _ArrayEmbedder:
    """Inyecta los embeddings ya calculados por el pipeline."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if len(self.embeddings) != len(texts):
            raise ValueError("Embeddings y textos no están alineados.")
        return self.embeddings


class _MaskedOutlierDetector:
    """Delega en el detector de XidooTopic y guarda su máscara."""

    def __init__(self, wrapped: Any):
        self.wrapped = wrapped
        self.mask_: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray, text: Optional[List[str]] = None) -> Dict[str, Any]:
        out = self.wrapped.fit_transform(X, text)
        self.mask_ = np.asarray(out["mask"], dtype=bool)
        return out


class _NoopOutlierDetector:
    """Desactiva el detector interno sin romper el contrato de componentes."""

    def fit_transform(self, X: np.ndarray, text: Optional[List[str]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "embeddings": X,
            "mask": np.ones(len(X), dtype=bool),
            "scores": np.zeros(len(X), dtype=float),
            "removed_indices": [],
        }
        if text is not None:
            out["texts"] = list(text)
        return out


class _MaskedClusterFilter:
    """Delega en ClusterFilter y guarda la máscara previa al filtrado."""

    def __init__(self, wrapped: Any):
        self.wrapped = wrapped
        self.mask_: Optional[np.ndarray] = None

    def fit_transform(self, embeddings: np.ndarray, texts: Sequence[str],
                      labels: np.ndarray) -> tuple:
        lab = np.asarray(labels)
        out = self.wrapped.fit_transform(embeddings, list(texts), lab)
        self.mask_ = np.array([c in self.wrapped.valid_clusters for c in lab], dtype=bool)
        return out


class _LiteKeywordExtractor:
    """Keywords deterministas por frecuencia, sin KeyBERT ni modelos extra."""

    def __init__(self, top_n: int = 10):
        from sklearn.feature_extraction.text import CountVectorizer
        self.top_n = top_n
        self.vectorizer = CountVectorizer(
            token_pattern=TOKEN_PATTERN, lowercase=True, min_df=1, max_df=0.40
        )

    def extract(self, texts: Sequence[str], embeddings: np.ndarray,
                labels: np.ndarray) -> Dict[Any, tuple]:
        docs = [str(t) for t in texts]
        try:
            X = self.vectorizer.fit_transform(docs)
            vocab = np.asarray(self.vectorizer.get_feature_names_out())
        except ValueError:
            X, vocab = None, np.asarray([])
        out: Dict[Any, tuple] = {}
        for tid in np.unique(labels):
            if X is None:
                terms: List[str] = []
            else:
                counts = np.asarray(X[np.asarray(labels) == tid].sum(axis=0)).ravel()
                order = np.argsort(-counts)[: self.top_n]
                terms = [str(vocab[i]) for i in order if counts[i] > 0]
            out[tid] = (terms, "[KEYWORDS] " + ", ".join(terms))
        return out


class _NoopDocExtractor:
    """Los documentos representativos no se usan en el pipeline."""

    def extract(self, embeddings: np.ndarray, labels: np.ndarray,
                documents: Sequence[str]) -> Dict[Any, Any]:
        return {}


def build_xidoo_model(
    embeddings: np.ndarray,
    random_state: int,
    min_cluster_size: int,
    use_internal_outlier: bool = True,
) -> Any:
    """Construye ``XidooTopic`` con componentes reales y semillas explícitas."""
    from xidootopic import (
        ClusterFilter, ClusterMerger, KNNGraphBuilder, LeidenClusterer,
        LeidenResolutionOptimizer, UMAPReducer,
    )
    from xidootopic.xidootopic import XidooTopic

    from xidootopic.modeling.outliers import IsolationForestOutlierRemover

    n = len(embeddings)
    outlier = (_MaskedOutlierDetector(
        IsolationForestOutlierRemover(contamination=0.02, random_state=random_state)
    ) if use_internal_outlier else _NoopOutlierDetector())
    return XidooTopic(
        preprocessor=_IdentityPreprocessor(),
        chunker=_IdentityChunker(),
        embedder=_ArrayEmbedder(embeddings),
        reducer=UMAPReducer(
            n_neighbors=min(30, max(2, n - 1)),
            n_components=min(5, max(2, n - 1)),
            metric="cosine", random_state=random_state,
        ),
        outlier_detector=outlier,
        graph_builder=KNNGraphBuilder(k=min(15, max(2, n - 1)), verbose=False),
        optimal_resolution=LeidenResolutionOptimizer(seed=random_state, verbose=False),
        clusterer=LeidenClusterer(random_state=random_state, verbose=False),
        cluster_filter=_MaskedClusterFilter(
            ClusterFilter(min_cluster_size=min_cluster_size, reindex=False, verbose=False)
        ),
        cluster_merger=ClusterMerger(verbose=False, reindex=False),
        keyword_model=_LiteKeywordExtractor(),
        doc_model=_NoopDocExtractor(),
        verbose=False,
    )


class XidooTopicModeler:
    """Adaptador de XidooTopic con salida alineada al corpus de entrada."""

    def __init__(
        self,
        model: Optional[Any] = None,
        use_internal_outlier: bool = False,
        random_state: int = 42,
        min_cluster_size: int = 5,
    ):
        self.model = model
        self.use_internal_outlier = use_internal_outlier
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size
        self.embeddings_: Optional[np.ndarray] = None
        self.reduction_: Optional[np.ndarray] = None
        self.base_labels_: Optional[np.ndarray] = None
        self.min_cluster_size_: int = min_cluster_size
        self.fitted_model_: Optional[Any] = None

    def fit_transform(
        self,
        texts: Sequence[str],
        metadata: Optional[pd.DataFrame] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Ejecuta XidooTopic; expone labels completos y reducción UMAP."""
        texts = [str(t) for t in texts]
        if metadata is None:
            metadata = pd.DataFrame({"text": texts})
        else:
            metadata = metadata.copy().reset_index(drop=True)
        if len(metadata) != len(texts):
            raise ValueError(
                f"metadata tiene {len(metadata)} filas, pero hay {len(texts)} textos."
            )

        if embeddings is None:
            if texts:
                from xidootopic import TextEmbedder
                embeddings = np.asarray(
                    TextEmbedder.default().embed_texts(texts), dtype=np.float32
                )
            else:
                embeddings = np.empty((0, 0), dtype=np.float32)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings y textos no están alineados.")
        self.embeddings_ = embeddings

        if not texts:
            empty = metadata.copy()
            empty["topic_id"] = []
            empty["topic_name"] = []
            self.base_labels_ = np.empty(0, dtype=int)
            self.reduction_ = np.empty((0, 0), dtype=np.float32)
            return {"df": empty, "topic_id": [], "keywords": {},
                    "topic_names": {}, "embeddings": embeddings}

        model = self.model if self.model is not None else build_xidoo_model(
            embeddings, self.random_state, self.min_cluster_size,
            use_internal_outlier=self.use_internal_outlier,
        )
        result = model.fit_transform(texts)
        self.fitted_model_ = model

        outlier = getattr(model, "outlier_detector", None)
        filter_ = getattr(model, "cluster_filter", None)
        self.base_labels_ = expand_xidoo_labels(
            result["labels"],
            outlier.mask_ if outlier is not None and hasattr(outlier, "mask_")
            else np.ones(len(texts), dtype=bool),
            filter_.mask_ if filter_ is not None and hasattr(filter_, "mask_")
            else None,
            len(texts),
        )
        reducer_model = getattr(getattr(model, "reducer", None), "model_", None)
        self.reduction_ = np.asarray(
            getattr(reducer_model, "embedding_", np.empty((0, 0))), dtype=np.float32
        )
        self.min_cluster_size_ = int(getattr(filter_, "min_cluster_size",
                                             self.min_cluster_size))

        keywords = normalize_xidoo_keywords(result.get("topics", {}))
        topic_names = {
            tid: ", ".join(words[:3]) or f"Topic {tid}"
            for tid, words in keywords.items()
        }
        out_df = metadata.copy()
        out_df["topic_id"] = self.base_labels_
        out_df["topic_name"] = [
            topic_names.get(str(tid), f"Topic {tid}") for tid in self.base_labels_
        ]
        return {
            "df": out_df,
            "topic_id": self.base_labels_.tolist(),
            "keywords": keywords,
            "topic_names": topic_names,
            "embeddings": embeddings,
        }
