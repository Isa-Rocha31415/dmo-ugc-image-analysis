"""Regresión: wordclouds HTML de log-odds (capa léxica)."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics_lexical import _wordcloud_html  # noqa: E402


class TestWordcloudHtml(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(__file__).resolve().parent / "_tmp_wordcloud_html"
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for path in self.tmp.glob("*"):
            path.unlink()
        self.tmp.rmdir()

    def test_con_terminos_escribe_html_y_tabla(self):
        terms = pd.DataFrame({
            "term": ["playa", "precio", "limpieza"],
            "logodds_z": [1.2, -0.4, 3.1],
        })
        path = self.tmp / "nube.html"
        palette = {"categorical": ["#123456", "#abcdef"]}
        result = _wordcloud_html(terms, path, "Términos distintivos", palette)

        self.assertEqual(result, path)
        self.assertTrue(path.exists())
        data = pd.read_csv(path.with_name("nube_data.csv"), encoding="utf-8-sig")
        self.assertEqual(list(data.columns), ["term", "logodds_z", "size", "x", "y"])
        self.assertEqual(data["term"].tolist(), terms["term"].tolist())
        self.assertTrue((data["size"].to_numpy() > 0).all())
        self.assertFalse(data[["x", "y"]].isna().any().any())
        html = path.read_text(encoding="utf-8")
        self.assertIn("distintivos", html)  # Plotly JSON-escapa los acentos
        self.assertIn("playa", html)

    def test_sin_terminos_tambien_escribe_tabla_vacia(self):
        path = self.tmp / "vacia.html"
        result = _wordcloud_html(pd.DataFrame({"term": [], "logodds_z": []}),
                                 path, "Sin términos", ["#123456"])
        self.assertEqual(result, path)
        data = pd.read_csv(path.with_name("vacia_data.csv"), encoding="utf-8-sig")
        self.assertTrue(data.empty)
        self.assertEqual(list(data.columns), ["term", "logodds_z", "size", "x", "y"])


if __name__ == "__main__":
    unittest.main()
