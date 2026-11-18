"""
Tests de las métricas puras de validación de sentimiento.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from metrics.sentiment_metrics import (  # noqa: E402
    accuracy,
    confusion_matrix,
    evaluate,
    evaluate_by_group,
    macro_f1,
    matthews_corrcoef,
    per_class_f1,
)

LABELS = ("negative", "positive")


class TestSentimentMetrics(unittest.TestCase):
    def test_confusion_matrix_orientacion(self):
        y_true = ["positive", "positive", "negative", "negative"]
        y_pred = ["positive", "negative", "negative", "positive"]
        cm = confusion_matrix(y_true, y_pred, LABELS)
        # filas = real, columnas = predicho
        self.assertEqual(cm[1][1], 1)  # real positive, pred positive
        self.assertEqual(cm[1][0], 1)  # real positive, pred negative
        self.assertEqual(cm[0][0], 1)  # real negative, pred negative
        self.assertEqual(cm[0][1], 1)  # real negative, pred positive

    def test_accuracy(self):
        self.assertEqual(accuracy(["a", "b"], ["a", "b"]), 1.0)
        self.assertEqual(accuracy(["a", "b"], ["b", "a"]), 0.0)

    def test_per_class_f1_y_macro_f1(self):
        y_true = ["positive"] * 4 + ["negative"] * 4
        y_pred = ["positive"] * 4 + ["negative"] * 2 + ["positive"] * 2
        f1 = per_class_f1(y_true, y_pred, LABELS)
        self.assertAlmostEqual(f1["positive"], 0.8)
        self.assertAlmostEqual(f1["negative"], 2 / 3)
        self.assertAlmostEqual(macro_f1(y_true, y_pred, LABELS), (0.8 + 2 / 3) / 2)

    def test_mcc(self):
        y_true = ["positive"] * 3 + ["negative"] * 2
        self.assertAlmostEqual(matthews_corrcoef(y_true, y_true, LABELS), 1.0)
        self.assertAlmostEqual(
            matthews_corrcoef(["positive", "positive", "negative", "negative"],
                               ["negative", "negative", "positive", "positive"], LABELS),
            -1.0,
        )

    def test_evaluate_devuelve_todas_las_metricas(self):
        result = evaluate(["positive", "negative"], ["positive", "negative"], LABELS)
        for key in ("accuracy", "macro_f1", "mcc", "f1_per_class", "confusion_matrix", "n"):
            self.assertIn(key, result)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accuracy"], 1.0)

    def test_evaluate_by_group(self):
        y_true = ["positive", "negative", "positive", "negative"]
        y_pred = ["positive", "positive", "negative", "negative"]
        groups = ["a", "a", "b", "b"]
        por_grupo = evaluate_by_group(y_true, y_pred, groups, LABELS)
        self.assertEqual(set(por_grupo.keys()), {"a", "b"})
        self.assertEqual(por_grupo["a"]["n"], 2)
        self.assertEqual(por_grupo["a"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
