"""
metrics
──────────────────────────────────────────────────────────────
Funciones puras de métricas (sin efectos secundarios ni prints).
"""
from .sentiment_metrics import (  # noqa: F401
    DEFAULT_LABELS,
    accuracy,
    confusion_matrix,
    evaluate,
    evaluate_by_group,
    macro_f1,
    matthews_corrcoef,
    per_class_f1,
)

__all__ = [
    "DEFAULT_LABELS",
    "accuracy",
    "confusion_matrix",
    "evaluate",
    "evaluate_by_group",
    "macro_f1",
    "matthews_corrcoef",
    "per_class_f1",
]
