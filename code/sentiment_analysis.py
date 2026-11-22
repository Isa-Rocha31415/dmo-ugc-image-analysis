from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from sentiment_labels import NEUTRAL_TAU, label_from_score


DEFAULT_MULTILINGUAL_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


# ---------------------------------------------------------------------------
# Helpers para transformers
# ---------------------------------------------------------------------------

def _normalize_label(label: str) -> str:
    """
    Normaliza etiquetas comunes de modelos de sentimiento.
    """
    label = str(label).lower().strip().replace(" ", "_")

    positive_markers = [
        "pos",
        "positive",
        "positif",
        "positivo",
        "label_pos",
        "5_stars",
        "4_stars",
    ]

    negative_markers = [
        "neg",
        "negative",
        "negatif",
        "negativo",
        "label_neg",
        "1_star",
        "2_stars",
    ]

    neutral_markers = [
        "neu",
        "neutral",
        "neutre",
        "3_stars",
    ]

    if any(marker in label for marker in positive_markers):
        return "positive"

    if any(marker in label for marker in negative_markers):
        return "negative"

    if any(marker in label for marker in neutral_markers):
        return "neutral"

    return label


_STAR_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _star_from_label(label: str) -> Optional[float]:
    """
    Detecta etiquetas tipo:
      - '1 star'
      - '2 stars'
      - 'LABEL_4'
      - '4'
      - '5_stars'
    """
    label = str(label).lower()

    if label.strip().isdigit():
        value = float(label.strip())
        return value if 1 <= value <= 5 else None

    if any(marker in label for marker in ["star", "estrella", "etoile", "estrela", "label_"]):
        match = _STAR_RE.search(label)
        if match:
            value = float(match.group(1))
            return value if 1 <= value <= 5 else None

    return None


def _parse_star_mapping(mapping: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Si el modelo devuelve probabilidades para 1-5 estrellas,
    convierte eso en un score continuo [-1, 1].
    """
    pairs = []

    for label, prob in mapping.items():
        star = _star_from_label(label)
        if star is None:
            return None
        pairs.append((star, float(prob)))

    if not pairs:
        return None

    expected_star = sum(star * prob for star, prob in pairs)

    # 1 estrella => -1, 3 estrellas => 0, 5 estrellas => +1
    normalized = (expected_star - 3.0) / 2.0

    return {
        "score": max(-1.0, min(1.0, normalized)),
        "positive_prob": sum(prob for star, prob in pairs if star >= 4),
        "negative_prob": sum(prob for star, prob in pairs if star <= 2),
        "neutral_prob": sum(prob for star, prob in pairs if 2.5 < star < 3.5),
    }


def _parse_classification_output(pred: Any) -> Dict[str, float]:
    """
    Convierte la salida de un pipeline de text-classification
    en un score de polaridad.

    Devuelve:
      score: [-1, 1]
      positive_prob
      negative_prob
      neutral_prob
    """
    output = {
        "score": 0.0,
        "positive_prob": 0.0,
        "negative_prob": 0.0,
        "neutral_prob": 0.0,
    }

    if pred is None:
        return output

    if isinstance(pred, dict):
        pred = [pred]

    if not isinstance(pred, (list, tuple)):
        return output

    mapping = {}

    for item in pred:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", ""))
        try:
            score = float(item.get("score", 0.0))
        except Exception:
            score = 0.0

        mapping[label] = score

    if not mapping:
        return output

    # Caso 1: modelo de estrellas 1-5
    star_output = _parse_star_mapping(mapping)
    if star_output is not None:
        output.update(star_output)
        return output

    # Caso 2: positive / negative / neutral
    normalized_map = {}
    for key, value in mapping.items():
        normalized_map[_normalize_label(key)] = float(value)

    positive_prob = normalized_map.get("positive", 0.0)
    negative_prob = normalized_map.get("negative", 0.0)
    neutral_prob = normalized_map.get("neutral", 0.0)

    if positive_prob or negative_prob or neutral_prob:
        output.update(
            {
                "score": positive_prob - negative_prob,
                "positive_prob": positive_prob,
                "negative_prob": negative_prob,
                "neutral_prob": neutral_prob,
            }
        )
        return output

    # Caso 3: salida binaria simple
    if len(mapping) == 1:
        label, score = next(iter(mapping.items()))
        normalized = _normalize_label(label)

        if normalized == "positive":
            output["score"] = float(score)
            output["positive_prob"] = float(score)

        elif normalized == "negative":
            output["score"] = -float(score)
            output["negative_prob"] = float(score)

    return output


def _transformer_scores(
    texts: List[str],
    model_name: str,
    device: int = -1,
    batch_size: int = 32,
    max_length: int = 512,
    cache_dir: Optional[str] = None,
) -> List[float]:
    """
    Calcula scores de sentimiento usando un modelo local de Transformers.

    Importante:
      - La primera vez puede descargar el modelo.
      - Después puede funcionar offline si el modelo queda en caché.
      - Si necesitas offline estricto, apunta model_name a una carpeta local.
    """
    from transformers import pipeline

    # Evitar textos vacíos, que pueden romper algunos pipelines.
    safe_texts = [text if str(text).strip() else "n/a" for text in texts]

    classifier = pipeline(
        task="sentiment-analysis",
        model=model_name,
        device=device,
        cache_dir=cache_dir,
    )

    call_kwargs = {
        "batch_size": batch_size,
        "truncation": True,
        "max_length": max_length,
    }

    # Intentamos obtener todas las etiquetas.
    try:
        predictions = classifier(safe_texts, top_k=None, **call_kwargs)
    except Exception:
        predictions = classifier(safe_texts, **call_kwargs)

    scores = []

    for pred in predictions:
        parsed = _parse_classification_output(pred)
        scores.append(parsed["score"])

    return scores


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def annotate_sentiment(
    df: pd.DataFrame,
    text_column: str,
    source_column: Optional[str] = None,
    lang: str = "es",
    backend: str = "auto",
    threshold: float = 0.0,
    neutral_tau: float = NEUTRAL_TAU,
    model_name: Optional[str] = None,
    device: int = -1,
    batch_size: int = 32,
    max_length: int = 512,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Agrega columnas:
      - sentiment_score
      - sentiment_label (positive/neutral/negative, banda |score| < neutral_tau)
      - sentiment_label_id (1/0/-1)
      - sentiment_backend

    ``threshold`` se conserva por compatibilidad; la polaridad usa ``neutral_tau``.
    La etiqueta de fuente no se modifica.
    
    Usa exclusivamente modelo transformer (BERT/XLM-RoBERTa).
    """
    if text_column not in df.columns:
        raise KeyError(f"No existe la columna de texto: {text_column}")

    if source_column is not None and source_column not in df.columns:
        raise KeyError(f"No existe la columna de fuente: {source_column}")

    out = df.copy()

    if out.empty:
        out["sentiment_score"] = []
        out["sentiment_label"] = []
        out["sentiment_label_id"] = []
        out["sentiment_backend"] = []
        return out

    texts = out[text_column].fillna("").astype(str).tolist()
    selected_model = model_name or DEFAULT_MULTILINGUAL_MODEL

    # Solo se usa transformers, sin fallback de léxico
    if backend not in {"auto", "transformers"}:
        raise ValueError(
            f"Backend '{backend}' no soportado. "
            "Solo se permite 'transformers' o 'auto' (que usa transformers)."
        )

    try:
        scores = _transformer_scores(
            texts=texts,
            model_name=selected_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
        )
        backend_used = "transformers"

    except Exception as exc:
        raise RuntimeError(
            f"Falló el análisis de sentimientos con el modelo transformer. "
            f"Modelo: {selected_model}. Error: {exc}"
        ) from exc

    out["sentiment_score"] = scores

    labels = [label_from_score(score, neutral_tau) for score in scores]
    out["sentiment_label"] = labels
    out["sentiment_label_id"] = [1 if x == "positive" else (-1 if x == "negative" else 0)
                                 for x in labels]

    out["sentiment_backend"] = backend_used

    return out