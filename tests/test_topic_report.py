"""Tests del bloque de tópicos para la comparación DMO vs UGC."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from topic_report import (  # noqa: E402
    build_topic_cards_html,
    build_topic_report_block,
    topic_summary_frame,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame({
        "topic_id": [0, 1, -1],
        "topic_name": ["playa limpia", "precio justo", "SIN_ASIGNAR"],
        "keywords_top8": [
            "playa; limpia; arena; familia; centro; sol; agua; arena",
            "precio; justo; calidad; hotel; barato; valor; coste; tarifa",
            "otro; comentario; sin; asignar; topic; texto; palabra; final",
        ],
        "n_docs": [30, 15, 5],
    })


def _doc_topics() -> pd.DataFrame:
    return pd.DataFrame({
        "doc_id": range(50),
        "topic_id": [0] * 30 + [1] * 15 + [-1] * 5,
        "topic_name": (["playa limpia"] * 30 + ["precio justo"] * 15
                       + ["SIN_ASIGNAR"] * 5),
    })


class TestTopicSummaryFrame(unittest.TestCase):
    def test_columnas_exactas_y_valores(self):
        rows = topic_summary_frame(_summary(), _doc_topics())
        self.assertEqual(
            list(rows.columns),
            ["topic_id", "title", "top5_words", "all_keywords", "coverage_pct", "n_docs"],
        )
        self.assertEqual(rows["title"].tolist(),
                         ["playa limpia", "precio justo", "SIN_ASIGNAR"])
        self.assertEqual(rows["top5_words"].iloc[0],
                         "playa, limpia, arena, familia, centro")
        self.assertEqual(len(rows["all_keywords"].iloc[0].split(", ")), 8)
        self.assertEqual(rows["coverage_pct"].tolist(), [60.0, 30.0, 10.0])
        self.assertEqual(rows["n_docs"].tolist(), [30, 15, 5])

    def test_fallback_sin_doc_topics(self):
        rows = topic_summary_frame(_summary(), None)
        self.assertEqual(rows["coverage_pct"].tolist(), [60.0, 30.0, 10.0])

    def test_titulo_cae_a_top_word(self):
        summary = _summary().drop(columns=["topic_name"])
        rows = topic_summary_frame(summary, _doc_topics())
        self.assertEqual(rows["title"].iloc[0], "playa")


class TestBuildTopicReportBlock(unittest.TestCase):
    def test_formato_exacto_consola_y_html(self):
        block = build_topic_report_block(_summary(), _doc_topics())
        console = block["console"]
        self.assertIn("┌", console)
        self.assertIn("└", console)
        self.assertIn("Tópico 0: playa limpia", console)
        self.assertIn("Top 5 palabras: playa, limpia, arena, familia, centro", console)
        self.assertIn("Keywords completas:", console)
        self.assertIn("Cobertura: 60.00% de los comentarios (n=30)", console)
        self.assertEqual(console.count("┌"), 3)
        self.assertEqual(console.count("└"), 3)
        html_out = block["html"]
        self.assertIn("topic-cards", html_out)
        self.assertIn("Tópico 0", html_out)
        self.assertIn("60.00%", html_out)
        self.assertNotIn("┌", html_out)
        self.assertNotIn("SIN_ASIGNAR", html_out)

    def test_tarjetas_sin_top5_ni_barra_de_cobertura(self):
        table = topic_summary_frame(_summary(), _doc_topics())
        cards = build_topic_cards_html(table)
        self.assertNotIn("Top 5 palabras", cards)
        self.assertNotIn("coverage__bar", cards)
        self.assertIn("Tópico 0", cards)
        self.assertIn('<h3 class="topic-card__title">playa, limpia, arena</h3>', cards)
        for word in ("familia", "centro", "sol", "agua", "arena"):
            with self.subTest(word=word):
                self.assertIn(f'<span class="pill-muted">{word}</span>', cards)
        self.assertIn("60.00%", cards)

    def test_csv_se_escribe_con_codificacion_correcta(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topics_summary.csv"
            block = build_topic_report_block(_summary(), _doc_topics(), csv_path=path)
            self.assertEqual(block["csv"], path)
            self.assertTrue(path.exists())
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            df = pd.read_csv(path, encoding="utf-8-sig")
            self.assertEqual(len(df), 3)


if __name__ == "__main__":
    unittest.main()
