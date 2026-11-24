"""Tests de la visualización interactiva de n-gramas (hover con comentarios)."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from ngram_visualization import (  # noqa: E402
    _ensure_text_norm,
    build_all_ngram_figures,
    build_ngram_interactive_plot,
    map_ngrams_to_comments,
)
from report_sections import ngram_section_html  # noqa: E402


NORMAL_TEXTOS = [
    "playa bonita y limpia",
    "playa bonita para familias",
    "playa bonita cerca del centro",
    "hotel cómodo y limpio",
    "hotel cómodo con piscina",
    "hotel cómodo y barato",
    "precio justo para la calidad",
    "precio justo y buen servicio",
    "servicio rápido y amable",
    "habitación amplia y luminosa",
]
OUTLIER_TEXTOS = [
    "playa sucia y llena de basura",
    "playa sucia otro comentario",
    "hotel horrible nunca vuelvo",
    "hotel horrible y caro",
    "pésima atención del personal",
]


def _frame(textos: list[str], source: str) -> pd.DataFrame:
    return pd.DataFrame({"text": textos, "text_norm": textos, "source": [source] * len(textos)})


class TestExtraerNgramasLegacy(unittest.TestCase):
    def test_usa_tokenize_de_metrics_lexical(self):
        from visualization import extraer_ngramas

        counter = extraer_ngramas(["La playa es bonita y limpia 123"], n=2)
        self.assertIn("playa bonita", counter)
        self.assertIn("bonita limpia", counter)
        self.assertFalse(any(k.startswith("la ") or "123" in k for k in counter))


class TestMapNgramsToComments(unittest.TestCase):
    def test_un_grama_con_al_menos_un_comentario(self):
        mapping = map_ngrams_to_comments(_frame(NORMAL_TEXTOS, "ugc"), n=2)
        self.assertIn("playa bonita", mapping)
        self.assertGreaterEqual(len(mapping["playa bonita"]), 1)
        self.assertTrue(all(isinstance(v, list) and v for v in mapping.values()))

    def test_trunca_y_escapea_xml(self):
        df = pd.DataFrame({"text_norm": ["aaa <bbb> & ccc"] * 4})
        mapping = map_ngrams_to_comments(df, n=1, max_samples=3)
        samples = mapping["bbb"]
        self.assertEqual(len(samples), 3)
        self.assertIn("&lt;bbb&gt;", samples[0])
        self.assertIn("&amp;", samples[0])


class TestBuildNgramInteractivePlot(unittest.TestCase):
    def test_figura_unica_con_seis_trazas_y_dos_menus(self):
        fig = build_ngram_interactive_plot(
            _frame(NORMAL_TEXTOS, "ugc"), _frame(OUTLIER_TEXTOS, "outliers"), n=2
        )
        self.assertEqual(len(fig.data), 6)
        self.assertEqual(sum(bool(t.visible) for t in fig.data), 1)
        self.assertTrue(fig.data[2].visible)
        self.assertEqual(len(fig.layout.updatemenus), 2)
        top, group = fig.layout.updatemenus
        self.assertEqual([b.label for b in top.buttons],
                         ["Unigramas", "Bigramas", "Trigramas"])
        self.assertEqual([b.label for b in group.buttons],
                         ["Comentarios normales", "Comentarios outliers"])

    def test_cada_boton_deja_una_sola_traza_visible(self):
        fig = build_ngram_interactive_plot(
            _frame(NORMAL_TEXTOS, "ugc"), _frame(OUTLIER_TEXTOS, "outliers")
        )
        for menu in fig.layout.updatemenus:
            for button in menu.buttons:
                mask = button.args[0]["visible"]
                self.assertEqual(len(mask), 6)
                self.assertEqual(mask.count(True), 1)

    def test_customdata_y_hovertemplate_poblados(self):
        for n in (1, 2, 3):
            with self.subTest(n=n):
                fig = build_ngram_interactive_plot(
                    _frame(NORMAL_TEXTOS, "ugc"), _frame(OUTLIER_TEXTOS, "outliers"), n=n
                )
                for trace in fig.data:
                    self.assertIsNotNone(trace.customdata)
                    self.assertEqual(len(trace.customdata), len(trace.x))
                    self.assertEqual(len(trace.customdata[0]), 3)
                    self.assertTrue(str(trace.customdata[0][0]).strip())
                    self.assertIn("<br>", trace.hovertemplate)
                    self.assertIn("<extra></extra>", trace.hovertemplate)

    def test_hover_muestra_comentario_normal_y_outlier(self):
        fig = build_ngram_interactive_plot(
            _frame(NORMAL_TEXTOS, "ugc"), _frame(OUTLIER_TEXTOS, "outliers"), n=2
        )
        normal_cells = " ".join(str(v) for row in fig.data[0].customdata for v in row)
        outlier_cells = " ".join(str(v) for row in fig.data[1].customdata for v in row)
        self.assertIn("playa", normal_cells)
        self.assertIn("playa", outlier_cells)
        self.assertIn("sucia", outlier_cells)
        self.assertNotIn("sucia", normal_cells)

    def test_figura_vacia_no_explota(self):
        fig = build_ngram_interactive_plot(
            pd.DataFrame(columns=["text_norm"]), pd.DataFrame(columns=["text_norm"]), n=1
        )
        self.assertEqual(len(fig.data), 6)
        self.assertEqual(len(fig.data[0].x), 0)


class TestEnsureTextNorm(unittest.TestCase):
    def test_genera_text_norm_en_memoria(self):
        df = pd.DataFrame({"text": ["La playa es bonita"]})
        out = _ensure_text_norm(df)
        self.assertIn("text_norm", out.columns)
        self.assertTrue(out["text_norm"].iloc[0])


class TestNgramSectionHtml(unittest.TestCase):
    def test_incrusta_una_figura_autocontenida(self):
        frame = _frame(NORMAL_TEXTOS, "ugc")
        figs = build_all_ngram_figures(frame, frame)
        markup = ngram_section_html(figs)
        self.assertEqual(markup.count("plotly-graph-div"), 1)
        self.assertIn("Unigramas", markup)
        self.assertIn("Bigramas", markup)
        self.assertIn("Trigramas", markup)


class TestIntegracionHTML(unittest.TestCase):
    def test_html_final_contiene_hover_con_comentarios(self):
        import contextlib
        import io
        import tempfile
        from argparse import Namespace
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "legacy"))
        from report_generator import run_report_task  # noqa: E402

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            base_dir = results / "comparison" / "base"
            base_dir.mkdir(parents=True)
            pd.DataFrame({
                "doc_id": [0, 1],
                "text": ["playa bonita", "hotel cómodo"],
                "corpus": ["DMO", "UGC"],
                "source": ["sectur", "ugc"],
                "sentiment_label": ["positive", "neutral"],
            }).to_csv(base_dir / "analysis_base.csv", index=False, encoding="utf-8-sig")

            figs = {
                n: build_ngram_interactive_plot(
                    _frame(NORMAL_TEXTOS, "ugc"), _frame(OUTLIER_TEXTOS, "outliers"), n=n
                )
                for n in (1, 2, 3)
            }
            (results / "ngram_interactive.html").write_text(
                ngram_section_html(figs), encoding="utf-8"
            )
            args = Namespace(
                title="Informe", palette="ejecutiva", csv_dmo="dmo.csv",
                model="bertopic", seed=42, bootstrap=200, comparison_tasks=["report"],
                embedding_model="modelo-local", verbose_jaccard=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_report_task(args, results)

            final = (results / "informe_final.html").read_text(encoding="utf-8")
            self.assertNotIn("{{NGRAM_SECTION}}", final)
            self.assertIn("Análisis de N-gramas", final)
            self.assertGreaterEqual(final.count("plotly-graph-div"), 1)
            self.assertIn("customdata", final)
            self.assertIn("playa", final)

    def test_report_no_inyecta_bloque_viejo_si_flag_off(self):
        import contextlib
        import io
        import tempfile
        from argparse import Namespace

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "legacy"))
        from report_generator import run_report_task  # noqa: E402

        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "ngram_interactive.html").write_text(
                "<div>gramas viejos</div>", encoding="utf-8"
            )
            args = Namespace(
                title="Informe", palette="ejecutiva", csv_dmo="dmo.csv",
                model="bertopic", seed=42, bootstrap=200, comparison_tasks=["report"],
                embedding_model="modelo-local", verbose_jaccard=False,
                ngram_interactive=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_report_task(args, results)
            final = (results / "informe_final.html").read_text(encoding="utf-8")
            self.assertNotIn("gramas viejos", final)
            self.assertIn("no disponible", final)

    def test_flag_off_no_escribe_nada(self):
        import tempfile
        from argparse import Namespace

        from run_analysis import _write_ngram_interactive  # noqa: E402

        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(ngram_interactive=False, palette="ejecutiva", seed=42)
            written = _write_ngram_interactive(
                args, Path(tmp), _frame(NORMAL_TEXTOS, "ugc"), _frame(OUTLIER_TEXTOS, "out")
            )
            self.assertEqual(written, [])
            self.assertFalse((Path(tmp) / "ngram_interactive.html").exists())


if __name__ == "__main__":
    unittest.main()
