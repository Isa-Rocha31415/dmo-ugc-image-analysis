from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _normalize_keyword_value(value: Any) -> List[str]:
    """
    Normaliza keywords que pueden venir como:
      - "playa, centro, comida"
      - ["playa", "centro", "comida"]
      - [("playa", 0.8), ("centro", 0.7)]
      - {"playa": 0.8, "centro": 0.7}
    """
    if value is None:
        return []

    if isinstance(value, str):
        parts = re.split(r",|;|\|", value)
        return [part.strip() for part in parts if part.strip()]

    if isinstance(value, dict):
        return [str(key) for key in value.keys()]

    if isinstance(value, (list, tuple, set, np.ndarray)):
        normalized = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                normalized.append(str(item[0]))
            else:
                normalized.append(str(item))
        return normalized

    return [str(value)]


class XidooTopicModeler:
    """
    Adaptador para XidooTopic.

    Recomendación para tu proyecto:
      - Ya detectaste outliers antes, entonces usa use_internal_outlier=False.
      - Mantén metadata con source_column para no perder la fuente.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        use_internal_outlier: bool = False,
        random_state: int = 42,
    ):
        self.model = model
        self.use_internal_outlier = use_internal_outlier
        self.random_state = random_state

    def fit_transform(
        self,
        texts: Sequence[str],
        metadata: Optional[pd.DataFrame] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta XidooTopic y devuelve:
          - df con topic_id y topic_name
          - keywords por tópico
          - topic_names
        """
        texts = [str(text) for text in texts]

        if metadata is None:
            metadata = pd.DataFrame({"text": texts})
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

        model = self.model if self.model is not None else self._build_model()

        result = model.fit_transform(texts)

        topic_ids = self._extract_topic_ids(result, expected_len=len(texts))
        raw_keywords = self._extract_keywords(result)
        keywords = self._normalize_keywords(raw_keywords)
        topic_names = self._build_topic_names(topic_ids, keywords)

        out_df = metadata.copy()
        out_df["topic_id"] = topic_ids
        out_df["topic_name"] = [
            topic_names.get(topic_id, f"Topic {topic_id}")
            for topic_id in topic_ids
        ]

        return {
            "df": out_df,
            "topic_id": topic_ids,
            "keywords": keywords,
            "topic_names": topic_names,
            "embeddings": embeddings,
        }

    def _build_model(self):
        """
        Construye el modelo XidooTopic.

        Intenta desactivar el outlier detector interno si ya detectaste
        outliers en tu propio pipeline.
        """
        try:
            from xidootopic.xidootopic import xidootopic
        except ImportError as exc:
            raise ImportError(
                "No tienes instalado xidootopic. "
                "Instálalo con: pip install xidootopic"
            ) from exc

        # Intento 1: desactivar outliers internos.
        if not self.use_internal_outlier:
            try:
                return xidootopic(outlier_detector=None)
            except TypeError:
                pass

        # Intento 2: constructor simple.
        try:
            return xidootopic()
        except TypeError:
            pass

        # Intento 3: si xidootopic requiere más parámetros,
        # deberás configurarlo manualmente aquí según la versión instalada.
        raise RuntimeError(
            "No se pudo instanciar xidootopic automáticamente. "
            "Revisa la API de la versión instalada y configura el modelo "
            "manualmente en XidooTopicModeler._build_model()."
        )

    def _extract_topic_ids(self, result: Any, expected_len: int) -> List[Any]:
        """
        Extrae los labels de tópicos del resultado de XidooTopic.
        """
        # Caso DataFrame
        if isinstance(result, pd.DataFrame):
            topic_ids = self._topic_ids_from_dataframe(result, expected_len)
            if topic_ids is not None:
                return topic_ids

        # Caso diccionario
        if isinstance(result, dict):
            candidates = [
                "topics",
                "topic_ids",
                "topic_labels",
                "labels",
                "assignments",
                "clusters",
                "community",
                "communities",
            ]

            for key in candidates:
                if key in result:
                    values = result[key]
                    if self._is_valid_label_sequence(values, expected_len):
                        return list(values)

        # Caso objeto con atributos
        attr_candidates = [
            "topics_",
            "topic_ids_",
            "topic_labels_",
            "labels_",
            "assignments_",
            "clusters_",
            "communities_",
        ]

        for attr in attr_candidates:
            if hasattr(result, attr):
                values = getattr(result, attr)
                if self._is_valid_label_sequence(values, expected_len):
                    return list(values)

        # Caso tupla/lista: primer elemento puede ser labels
        if isinstance(result, (tuple, list)):
            for item in result:
                if self._is_valid_label_sequence(item, expected_len):
                    return list(item)

        # Caso numpy array directo
        if isinstance(result, np.ndarray):
            if len(result) == expected_len:
                return result.tolist()

        raise RuntimeError(
            "No se pudieron extraer topic IDs de XidooTopic. "
            f"Tipo de resultado: {type(result)}. "
            "Debes revisar el objeto retornado por model.fit_transform() "
            "y adaptar XidooTopicModeler._extract_topic_ids()."
        )

    def _topic_ids_from_dataframe(
        self,
        df: pd.DataFrame,
        expected_len: int,
    ) -> Optional[List[Any]]:
        """
        Busca columnas típicas de tópico en un DataFrame.
        """
        if len(df) != expected_len:
            return None

        candidates = [
            "topic_id",
            "topic",
            "topic_label",
            "label",
            "cluster",
            "cluster_id",
            "community",
            "community_id",
        ]

        lower_columns = {str(col).lower(): col for col in df.columns}

        for candidate in candidates:
            if candidate in df.columns:
                return df[candidate].tolist()

            if candidate.lower() in lower_columns:
                return df[lower_columns[candidate.lower()]].tolist()

        return None

    def _is_valid_label_sequence(self, values: Any, expected_len: int) -> bool:
        """
        Verifica si algo parece una secuencia de labels del largo esperado.
        """
        if values is None:
            return False

        if isinstance(values, pd.Series):
            return len(values) == expected_len

        if isinstance(values, np.ndarray):
            return len(values) == expected_len

        if isinstance(values, (list, tuple)):
            return len(values) == expected_len

        return False

    def _extract_keywords(self, result: Any) -> Dict[Any, Any]:
        """
        Extrae keywords por tópico desde el resultado.
        """
        # Diccionario directo
        if isinstance(result, dict):
            candidates = [
                "keywords",
                "topic_keywords",
                "representations",
                "topic_representations",
                "words",
                "topic_words",
            ]

            for key in candidates:
                if key in result and isinstance(result[key], dict):
                    return result[key]

        # DataFrame con topic + keywords
        if isinstance(result, pd.DataFrame):
            keywords = self._keywords_from_dataframe(result)
            if keywords:
                return keywords

        # Objeto con atributos
        attr_candidates = [
            "keywords_",
            "topic_keywords_",
            "representations_",
            "topic_representations_",
            "keywords",
            "topic_representations",
        ]

        for attr in attr_candidates:
            if hasattr(result, attr):
                value = getattr(result, attr)
                if isinstance(value, dict):
                    return value

                if isinstance(value, pd.DataFrame):
                    keywords = self._keywords_from_dataframe(value)
                    if keywords:
                        return keywords

        # Tupla/lista: segundo elemento puede ser keywords
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            second = result[1]
            if isinstance(second, dict):
                return second

            if isinstance(second, pd.DataFrame):
                keywords = self._keywords_from_dataframe(second)
                if keywords:
                    return keywords

        return {}

    def _keywords_from_dataframe(self, df: pd.DataFrame) -> Dict[Any, List[str]]:
        """
        Extrae keywords desde un DataFrame que puede tener columnas como:
          - topic / keywords
          - topic_id / representation
          - cluster / words
        """
        if df.empty:
            return {}

        topic_col = None
        keyword_col = None

        topic_candidates = [
            "topic_id",
            "topic",
            "topic_label",
            "label",
            "cluster",
            "cluster_id",
            "community",
        ]

        keyword_candidates = [
            "keywords",
            "keyword",
            "words",
            "terms",
            "representation",
            "topic_representation",
            "name",
            "topic_name",
        ]

        lower_columns = {str(col).lower(): col for col in df.columns}

        for candidate in topic_candidates:
            if candidate in df.columns:
                topic_col = candidate
                break

            if candidate.lower() in lower_columns:
                topic_col = lower_columns[candidate.lower()]
                break

        for candidate in keyword_candidates:
            if candidate in df.columns:
                keyword_col = candidate
                break

            if candidate.lower() in lower_columns:
                keyword_col = lower_columns[candidate.lower()]
                break

        if topic_col is None or keyword_col is None:
            return {}

        keywords = {}

        for topic_id, keyword_value in zip(df[topic_col], df[keyword_col]):
            keywords[topic_id] = _normalize_keyword_value(keyword_value)

        return keywords

    def _normalize_keywords(self, raw_keywords: Dict[Any, Any]) -> Dict[str, List[str]]:
        """
        Deja keywords con keys string y valores como lista de strings.
        """
        if not isinstance(raw_keywords, dict):
            return {}

        normalized = {}

        for topic_id, value in raw_keywords.items():
            normalized[str(topic_id)] = _normalize_keyword_value(value)

        return normalized

    def _build_topic_names(
        self,
        topic_ids: Sequence[Any],
        keywords: Dict[str, List[str]],
        max_words: int = 3,
    ) -> Dict[Any, str]:
        """
        Construye nombres simples para tópicos:
          Topic 1: playa, centro, comida
        """
        names = {}

        unique_topics = []
        seen = set()

        for topic_id in topic_ids:
            if topic_id not in seen:
                seen.add(topic_id)
                unique_topics.append(topic_id)

        for topic_id in unique_topics:
            topic_key = str(topic_id)
            topic_keywords = keywords.get(topic_key, [])

            if topic_keywords:
                label = ", ".join(topic_keywords[:max_words])
                names[topic_id] = f"Topic {topic_id}: {label}"
            else:
                names[topic_id] = f"Topic {topic_id}"

        return names