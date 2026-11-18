"""
metrics.sentiment_metrics
──────────────────────────────────────────────────────────────
Métricas puras de clasificación binaria para validar el análisis de
sentimiento. Sin efectos secundarios, sin prints, sin dependencias
obligatorias (sklearn/scipy solo en los módulos que las necesitan).

Contrato de datos:
  y_true / y_pred: secuencias de etiquetas ("positive" / "negative").
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_LABELS: Tuple[str, str] = ("negative", "positive")


def _as_list(values: Iterable[str]) -> List[str]:
    return [str(v) for v in values]


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> List[List[int]]:
    """
    Matriz de confusión con las filas = etiquetas reales y
    las columnas = etiquetas predichas.
    """
    truth = _as_list(y_true)
    preds = _as_list(y_pred)

    if len(truth) != len(preds):
        raise ValueError(
            f"y_true ({len(truth)}) y y_pred ({len(preds)}) tienen distinta longitud."
        )

    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for real, pred in zip(truth, preds):
        if real in index and pred in index:
            matrix[index[real]][index[pred]] += 1

    return matrix


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Exactitud global."""
    truth = _as_list(y_true)
    preds = _as_list(y_pred)

    if not truth:
        return 0.0

    correct = sum(1 for t, p in zip(truth, preds) if t == p)
    return correct / len(truth)


def per_class_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> Dict[str, float]:
    """F1 (harmonic mean of precision and recall) por clase."""
    truth = _as_list(y_true)
    preds = _as_list(y_pred)
    matrix = confusion_matrix(truth, preds, labels)

    scores: Dict[str, float] = {}

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fn = sum(matrix[i]) - tp
        fp = sum(row[i] for row in matrix) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        if precision + recall == 0:
            scores[label] = 0.0
        else:
            scores[label] = 2 * precision * recall / (precision + recall)

    return scores


def macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> float:
    """Macro-F1 (promedio no ponderado del F1 por clase)."""
    scores = per_class_f1(y_true, y_pred, labels)

    if not scores:
        return 0.0

    return sum(scores.values()) / len(scores)


def matthews_corrcoef(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> float:
    """
    Matthews correlation coefficient (MCC).

    Pensado para clases desbalanceadas: 0 = azar, 1 = perfecto, -1 = invertido.
    """
    truth = _as_list(y_true)
    preds = _as_list(y_pred)
    matrix = confusion_matrix(truth, preds, labels)

    if len(labels) != 2:
        raise ValueError("MCC solo está definido para clasificación binaria.")

    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> Dict[str, Any]:
    """
    Paquete completo de métricas binarias.
    """
    truth = _as_list(y_true)
    preds = _as_list(y_pred)

    matrix = confusion_matrix(truth, preds, labels)
    f1_scores = per_class_f1(truth, preds, labels)
    total = len(truth)

    # Soporte real y predicho por clase (indices posicionales).
    real = {label: sum(matrix[i]) for i, label in enumerate(labels)}
    predicted = {
        label: sum(matrix[i][j] for i in range(len(labels)))
        for j, label in enumerate(labels)
    }

    return {
        "n": total,
        "accuracy": accuracy(truth, preds),
        "macro_f1": macro_f1(truth, preds, labels),
        "f1_per_class": f1_scores,
        "mcc": matthews_corrcoef(truth, preds, labels),
        "confusion_matrix": matrix,
        "labels": list(labels),
        "support_real": real,
        "support_pred": predicted,
        "accuracy_balanced_baseline": (
            max(real.values()) / total if total else 0.0
        ),
    }


def evaluate_by_group(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    groups: Sequence[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> Dict[str, Dict[str, Any]]:
    """
    Métricas desglosadas por grupo (p. ej. fuente) para detectar
    *domain shift* entre corpus.
    """
    truth = _as_list(y_true)
    preds = _as_list(y_pred)
    group_list = _as_list(groups)

    if not (len(truth) == len(preds) == len(group_list)):
        raise ValueError("y_true, y_pred y groups deben tener la misma longitud.")

    out: Dict[str, Dict[str, Any]] = {}

    for group in sorted(set(group_list)):
        idx = [i for i, g in enumerate(group_list) if g == group]
        out[group] = evaluate(
            [truth[i] for i in idx],
            [preds[i] for i in idx],
            labels,
        )

    return out
