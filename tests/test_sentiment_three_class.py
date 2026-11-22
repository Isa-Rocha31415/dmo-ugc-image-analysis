"""Tests de etiquetado ternario y de las secciones auxiliares del reporte."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from report_sections import jaccard_console_lines, sentiment_table_html  # noqa: E402
from sentiment_labels import (  # noqa: E402
    LABEL_TO_ID,
    NEUTRAL_TAU,
    SENTIMENT_LABELS,
    apply_sentiment_labels,
    label_from_score,
    sentiment_distribution,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source": ["dmo", "dmo", "ugc", "ugc", "ugc"],
        "corpus": ["DMO", "DMO", "UGC", "UGC", "UGC"],
        "sentiment_score": [0.8, 0.0, -0.6, 0.15, np.nan],
        "sentiment_label": ["positive", "negative", "negative", "positive", "positive"],
    })


class TestLabelFromScore(unittest.TestCase):
    def test_banda_neutra(self):
        self.assertEqual(label_from_score(0.16), "positive")
        self.assertEqual(label_from_score(-0.16), "negative")
        self.assertEqual(label_from_score(0.0), "neutral")
        self.assertEqual(label_from_score(0.15), "neutral")
        self.assertEqual(label_from_score(-0.15), "neutral")
        self.assertEqual(label_from_score(np.nan), "neutral")

    def test_tau_personalizado_y_contrato_numerico(self):
        self.assertEqual(label_from_score(0.05, neutral_tau=0.02), "positive")
        self.assertEqual(SENTIMENT_LABELS, ("positive", "neutral", "negative"))
        self.assertEqual(LABEL_TO_ID, {"positive": 1, "neutral": 0, "negative": -1})
        self.assertEqual(NEUTRAL_TAU, 0.15)


class TestApplySentimentLabels(unittest.TestCase):
    def test_reemplaza_etiquetas_binarias_cacheadas(self):
        df = _frame()
        out = apply_sentiment_labels(df)
        self.assertEqual(
            out["sentiment_label"].tolist(),
            ["positive", "neutral", "negative", "neutral", "neutral"],
        )
        self.assertEqual(df["sentiment_label"].tolist()[1], "negative")  # sin mutar
        self.assertEqual(out["sentiment_label_id"].tolist(), [1, 0, -1, 0, 0])


class TestSentimentDistribution(unittest.TestCase):
    def test_conteo_y_porcentajes(self):
        tab = sentiment_distribution(apply_sentiment_labels(_frame()), ("source",))
        self.assertEqual(tab["source"].tolist(), ["dmo", "ugc"])
        self.assertEqual(tab["total"].tolist(), [2, 3])
        self.assertEqual(tab["positivos"].tolist(), [1, 0])
        self.assertEqual(tab["neutros"].tolist(), [1, 2])
        self.assertEqual(tab["negativos"].tolist(), [0, 1])
        for _, row in tab.iterrows():
            self.assertAlmostEqual(
                row["pct_positivos"] + row["pct_neutros"] + row["pct_negativos"], 100.0
            )

    def test_agrupacion_por_corpus(self):
        tab = sentiment_distribution(apply_sentiment_labels(_frame()), ("corpus",))
        self.assertEqual(set(tab["corpus"]), {"DMO", "UGC"})


class TestReportSections(unittest.TestCase):
    def test_jaccard_solo_consola(self):
        metrics = pd.DataFrame({
            "metric": ["jaccard_freq", "jaccard_tfidf", "coverage_types"],
            "scope": ["ugc_total", "ugc_total", "ugc_total"],
            "k": [100.0, 100.0, np.nan],
            "value": [0.25, 0.4, 0.9],
            "ci_low": [0.2, 0.3, np.nan],
            "ci_high": [0.3, 0.5, np.nan],
        })
        lines = jaccard_console_lines(metrics)
        self.assertEqual(len(lines), 2)
        self.assertIn("jaccard_freq", lines[0])
        self.assertIn("0.250", lines[0])
        self.assertNotIn("coverage", " ".join(lines))
        self.assertEqual(jaccard_console_lines(pd.DataFrame()), [])

    def test_tabla_descriptiva_html(self):
        tab = sentiment_distribution(apply_sentiment_labels(_frame()), ("corpus",))
        html = sentiment_table_html(tab)
        self.assertIn("POSITIVO", html)
        self.assertIn("NEUTRO", html)
        self.assertIn("NEGATIVO", html)
        self.assertIn("<table", html)


if __name__ == "__main__":
    unittest.main()
