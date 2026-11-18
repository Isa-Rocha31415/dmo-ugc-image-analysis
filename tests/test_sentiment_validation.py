"""
Tests de la validación de sentimiento (etiquetas débiles, gold y ablación).
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from metrics.sentiment_ablation import (  # noqa: E402
    embeddings_logreg_probe,
    lexicon_predict,
    tfidf_logreg_probe,
)
from sentiment_validation import (  # noqa: E402
    build_weak_labels,
    rating_to_label,
    validate_against_rating,
    validar_sentimiento,
)


def _frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        positivo = i % 2 == 0
        rows.append(
            {
                "text": "excelente hotel muy bueno" if positivo else "hotel terrible muy malo",
                "source": "tripadvisor" if i % 3 else "reddit",
                "rating": 5 if positivo else 1,
                "sentiment_score": 0.9 if positivo else -0.9,
                "sentiment_label": "positive" if positivo else "negative",
                "sentiment_backend": "transformers",
            }
        )
    return pd.DataFrame(rows)


class TestWeakLabels(unittest.TestCase):
    def test_rating_to_label(self):
        self.assertEqual(rating_to_label(1), "negative")
        self.assertEqual(rating_to_label(2), "negative")
        self.assertIsNone(rating_to_label(3))
        self.assertEqual(rating_to_label(4), "positive")
        self.assertEqual(rating_to_label(5), "positive")
        self.assertIsNone(rating_to_label(np.nan))

    def test_build_weak_labels(self):
        df = build_weak_labels(_frame(10), rating_column="rating")
        self.assertIn("weak_label", df.columns)
        self.assertEqual(df["weak_label"].notna().sum(), 10)

    def test_validate_against_rating(self):
        df = build_weak_labels(_frame(30), rating_column="rating")
        ext = validate_against_rating(df)
        self.assertEqual(ext["n"], 30)
        self.assertGreater(ext["spearman"], 0.9)
        self.assertLess(ext["mae_stars"], 1.0)


class TestAblation(unittest.TestCase):
    def test_lexicon_predict(self):
        preds = lexicon_predict([
            "excelente hotel, muy buena experiencia",
            "pésimo servicio, terrible y sucio",
        ])
        self.assertEqual(preds[0], "positive")
        self.assertEqual(preds[1], "negative")

    def test_tfidf_probe(self):
        df = _frame(60)
        result = tfidf_logreg_probe(df["text"].tolist(), df["sentiment_label"].tolist())
        self.assertIn("accuracy", result)
        self.assertGreater(result["accuracy"], 0.8)

    def test_embeddings_probe(self):
        df = _frame(60)
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(60, 8))
        emb[df["sentiment_label"] == "positive"] += 3.0
        result = embeddings_logreg_probe(emb, df["sentiment_label"].tolist())
        self.assertGreater(result["accuracy"], 0.8)


class TestValidarSentimiento(unittest.TestCase):
    def test_escribe_json_y_html(self):
        df = _frame(60)
        with tempfile.TemporaryDirectory() as tmp:
            result = validar_sentimiento(
                df=df,
                text_column="text",
                source_column="source",
                output_dir=tmp,
            )
            self.assertIn("weak_labels", result)
            self.assertIn("global", result["weak_labels"])
            self.assertGreater(result["weak_labels"]["global"]["accuracy"], 0.8)
            self.assertTrue((Path(tmp) / "sentiment_validation.json").exists())
            self.assertTrue((Path(tmp) / "sentiment_validation.html").exists())
            self.assertTrue((Path(tmp) / "metrics_by_source.csv").exists())


if __name__ == "__main__":
    unittest.main()
