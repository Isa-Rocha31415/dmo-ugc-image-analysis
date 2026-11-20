"""
bertopic_modeler.py
──────────────────────────────────────────────────────────────
Adapter de BERTopic con la misma interfaz que `XidooTopicModeler`
(`fit_transform(texts, metadata, embeddings)`), para que ambos backends
sean intercambiables en `topic_modeling.run_topic_modeling`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SOURCE_FALLBACK = "text"


class BertopicModeler:
    """Envoltorio de BERTopic con contrato estable para el pipeline."""

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        random_state: int = 42,
        min_topic_size: int = 10,
        top_n_words: int = 10,
        min_samples_per_topic: int = 3,
    ):
        self.embedding_model = embedding_model
        self.random_state = random_state
        self.min_topic_size = min_topic_size
        self.top_n_words = top_n_words
        self.min_samples_per_topic = min_samples_per_topic

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        texts: Sequence[str],
        metadata: Optional[pd.DataFrame] = None,
        embeddings: Optional[np.ndarray] = None,
        precomputed_knn: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Ajusta BERTopic sobre `texts` y devuelve:
          - df con topic_id y topic_name
          - keywords por tópico
          - topic_names
        """
        from bertopic import BERTopic

        texts = [str(t) for t in texts]

        if metadata is None:
            metadata = pd.DataFrame({SOURCE_FALLBACK: texts})
        else:
            metadata = metadata.copy().reset_index(drop=True)

        if len(metadata) != len(texts):
            raise ValueError(
                f"metadata tiene {len(metadata)} filas, "
                f"pero hay {len(texts)} textos."
            )

        if len(texts) == 0:
            empty = metadata.copy()
            empty["topic_id"] = []
            empty["topic_name"] = []
            return {
                "df": empty,
                "topic_id": [],
                "keywords": {},
                "topic_names": {},
                "embeddings": embeddings,
            }

        model = self._build_model(
            has_embeddings=embeddings is not None, n_docs=len(texts),
            precomputed_knn=precomputed_knn,
        )
        topics, _ = model.fit_transform(texts, embeddings=embeddings)

        topic_ids = [int(t) for t in topics]
        topic_names, keywords = self._extract_topics(model, topic_ids)

        out_df = metadata.copy()
        out_df["topic_id"] = topic_ids
        out_df["topic_name"] = [
            topic_names.get(tid, f"Topic {tid}") for tid in topic_ids
        ]

        return {
            "df": out_df,
            "topic_id": topic_ids,
            "keywords": keywords,
            "topic_names": topic_names,
            "embeddings": embeddings,
        }

    # ------------------------------------------------------------------
    def _build_model(self, has_embeddings: bool, n_docs: int = 10,
                     precomputed_knn: Optional[tuple[np.ndarray, np.ndarray]] = None):
        """Instancia BERTopic con semillas explícitas."""
        from bertopic import BERTopic

        kwargs: Dict[str, Any] = {
            "min_topic_size": self.min_topic_size,
            "top_n_words": self.top_n_words,
            "calculate_probabilities": False,
            "verbose": True,
        }

        # Si ya tenemos embeddings no hace falta cargar el modelo de sentences.
        kwargs["embedding_model"] = None if has_embeddings else self.embedding_model

        try:
            from hdbscan import HDBSCAN

            # OJO: `random_state` no se pasa a HDBSCAN (su version instalada
            # lo reenvia a sklearn.KDTree y revienta). El azar se controla
            # con la semilla de UMAP.
            kwargs["hdbscan_model"] = HDBSCAN(
                min_cluster_size=self.min_topic_size,
                min_samples=self.min_samples_per_topic,
            )
        except ImportError:
            pass

        try:
            import umap

            umap_kwargs: Dict[str, Any] = dict(
                n_neighbors=15,
                n_components=min(5, max(2, n_docs)),
                min_dist=0.0,
                metric="cosine",
                random_state=self.random_state,
            )
            if precomputed_knn is not None:
                umap_kwargs["precomputed_knn"] = precomputed_knn
            kwargs["umap_model"] = umap.UMAP(**umap_kwargs)
        except ImportError:
            pass

        return BERTopic(**kwargs)

    def _extract_topics(
        self,
        model: Any,
        topic_ids: Sequence[int],
    ) -> tuple[Dict[int, str], Dict[int, List[str]]]:
        """Nombres y keywords por tópico (incluye el tópico -1 de outliers)."""
        names: Dict[int, str] = {}
        keywords: Dict[int, List[str]] = {}

        info = None
        try:
            info = model.get_topic_info()
        except Exception:
            info = None

        for tid in sorted(set(topic_ids)):
            words = self._topic_words(model, tid)

            if tid == -1:
                names[tid] = "Topic -1: outliers"
                keywords[tid] = words
                continue

            label = self._info_name(info, tid)
            if not label:
                label = ", ".join(words[:3]) if words else ""
            names[tid] = f"Topic {tid}: {label}" if label else f"Topic {tid}"
            keywords[tid] = words

        return names, keywords

    def _topic_words(self, model: Any, topic_id: int) -> List[str]:
        try:
            raw = model.get_topic(topic_id)
        except Exception:
            raw = None
        if not raw:
            return []
        return [str(word) for word, _weight in raw][: self.top_n_words]

    def _info_name(self, info: Optional[pd.DataFrame], topic_id: int) -> str:
        if info is None or info.empty or "Name" not in info.columns:
            return ""
        try:
            row = info[info["Topic"] == topic_id]
        except Exception:
            return ""
        if row.empty:
            return ""
        name = str(row.iloc[0]["Name"]).strip()
        return "" if name in ("", "-") else name
