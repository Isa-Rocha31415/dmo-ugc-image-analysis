"""
Tests mínimos del dispatch de backends de modelado de tópicos
(`bertopic` vs backend XidooTopic) y del contrato de `run_topic_modeling`.

Ejecutables con `python -m unittest discover tests` o con `pytest -q`.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from topic_modeling import SUPPORTED_BACKENDS, run_topic_modeling  # noqa: E402

XIDOO = chr(120) + "idoo"  # backend alternativo (XidooTopic)


def _fake_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    textos = []
    for i in range(n):
        centro = "hotel playa servicio comida habitación" if i % 2 == 0 else \
                 "malo较差 restaurante comida terrible ruido"
        textos.append(f"{centro} opinion numero {i} " + "palabra " * (i % 5))
    df = pd.DataFrame(
        {
            "text": textos,
            "source": ["tripadvisor"] * (n // 2) + ["reddit"] * (n // 2),
            "sentiment_label": ["positive"] * (n // 2) + ["negative"] * (n // 2),
        }
    )
    embeddings = rng.normal(size=(n, 16)).astype(np.float32)
    return df, embeddings


class TestBackends(unittest.TestCase):
    def test_backends_soportados(self):
        self.assertIn("bertopic", SUPPORTED_BACKENDS)
        self.assertIn(XIDOO, SUPPORTED_BACKENDS)

    def test_backend_bertonic_requiere_sentimiento(self):
        df = pd.DataFrame({"text": ["hola mundo"], "source": ["x"]})
        with self.assertRaises(ValueError):
            run_topic_modeling(
                df=df, text_column="text", source_column="source", topic_backend="bertopic"
            )

    def test_particion_pequena_usa_frecuencia_de_palabras(self):
        df, _ = _fake_frame(4)
        result = run_topic_modeling(
            df=df,
            text_column="text",
            source_column="source",
            topic_backend="bertopic",
            min_partition_size=50,
        )
        self.assertEqual(result["by_sentiment"]["positive"]["method"], "word_frequency")

    def test_backend_bertopic_produce_topicos(self):
        df, embeddings = _fake_frame(60)
        result = run_topic_modeling(
            df=df,
            text_column="text",
            source_column="source",
            topic_backend="bertopic",
            min_partition_size=10,
            embeddings=embeddings,
        )
        by_sentiment = result["by_sentiment"]
        self.assertIn("positive", by_sentiment)
        self.assertIn("negative", by_sentiment)
        self.assertEqual(by_sentiment["positive"]["method"], "bertopic")
        self.assertEqual(by_sentiment["negative"]["method"], "bertopic")
        topics_df = result["topics_df"]
        self.assertFalse(topics_df.empty)
        self.assertIn("topic_name", topics_df.columns)
        self.assertIn("sentiment_label", topics_df.columns)


if __name__ == "__main__":
    unittest.main()
