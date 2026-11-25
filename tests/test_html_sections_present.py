"""Contrato de secciones del informe HTML final (capa de presentación)."""
import contextlib
import io
import re
import sys
import unittest
from argparse import Namespace
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "legacy"))

from report_generator import run_report_task  # noqa: E402

REQUIRED_IDS = (
    "kpis", "polaridad-descriptiva", "ngramas", "topicos", "umap-3d",
    "tematica", "brechas", "lexico", "precio",
)
PRICE_IDS = (
    "precio-lectura", "precio-frecuencia", "precio-scatter", "precio-contexto",
    "precio-ejemplos",
)
REMOVED_IDS = (
    "ejecutivo", "ejemplos", "apendice", "metodo", "metadatos-run",
    "pendientes", "semantico-pendiente", "afectivo-pendiente",
    "figuras-pendiente", "pruebas-estadisticas",
)
BOX_CHARS = "┌─┐│└┘"


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.links: list[str] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        if data.get("id"):
            self.ids.append(str(data["id"]))
        if tag == "a" and data.get("href"):
            self.links.append(str(data["href"]))
        if tag in {"section", "aside", "nav"}:
            self.stack.append(str(data.get("id") or tag))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"section", "aside", "nav"} and self.stack:
            self.stack.pop()


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    _write(path)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _fixture_results(root: Path) -> None:
    for sub in ("comparison/base", "comparison/topics_joint", "comparison/thematic",
                "comparison/lexical", "comparison/price", "sentiment", "topics"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame({
        "doc_id": [0, 1, 2], "text": ["playa bonita", "hotel cómodo", "precio caro"],
        "text_norm": ["playa bonita", "hotel cómodo", "precio caro"],
        "corpus": ["DMO", "UGC", "UGC"],
        "source": ["sectur", "tripadvisor", "reddit"],
        "sentiment_label": ["positive", "neutral", "negative"],
    })
    _csv(root / "comparison/base/analysis_base.csv", base)
    _csv(root / "comparison/base/outliers_base.csv", pd.DataFrame({"doc_id": [9], "text": ["x"]}))
    _csv(root / "comparison/topics_joint/bert/topics_summary.csv", pd.DataFrame({
        "topic_id": [0, 1, -1], "topic_name": ["playa limpia", "precio justo", "SIN_ASIGNAR"],
        "keywords_top8": ["playa; limpia; arena; sol; mar; familia; centro; hotel"] * 3,
        "n_docs": [2, 1, 1],
    }))
    _csv(root / "comparison/topics_joint/bert/doc_topics.csv", pd.DataFrame({
        "doc_id": [0, 1, 2], "topic_id": [0, 0, 1],
        "topic_name": ["playa limpia", "playa limpia", "precio justo"],
    }))
    _csv(root / "comparison/topics_joint/bert/topic_selection_grid.csv",
         pd.DataFrame({"k": [2], "selected": [True]}))
    _csv(root / "comparison/thematic/distribution_distances.csv",
         pd.DataFrame({"scope": ["ugc_total"], "jsd_base2": [0.01], "emd_cosine": [0.02]}))
    _csv(root / "comparison/thematic/gaps_table.csv", pd.DataFrame({
        "scope": ["ugc_total", "ugc_total"],
        "topic_name": ["playa", "precio"],
        "p_dmo": [0.10, 0.40], "p_ugc": [0.30, 0.20],
        "gap_pp": [20.0, -20.0], "verdict": ["SOBRE_UGC", "SOBRE_DMO"],
    }))
    _csv(root / "comparison/lexical/metrics_lexical.csv", pd.DataFrame({
        "metric": ["coverage"], "scope": ["ugc_total"], "k": [100], "value": [0.5],
    }))
    _csv(root / "comparison/price/mention_rate_by_source.csv",
         pd.DataFrame({"scope": ["ugc_total"], "rate": [0.2]}))
    _csv(root / "comparison/base/run_metadata.csv",
         pd.DataFrame({"task": ["base"], "timestamp": ["2026-01-01"]}))

    price = pd.DataFrame({
        "scope": ["DMO_vs_UGC_total", "DMO_vs_instagram"],
        "n_mentions_dmo": [0, 0], "n_docs_dmo": [475, 475],
        "ci_low_dmo": [-0.01, -0.01], "ci_high_dmo": [0.01, 0.01],
        "n_mentions_ugc": [10, 2], "n_docs_ugc": [200, 50],
        "ci_low_ugc": [0.02, 0.01], "ci_high_ugc": [0.08, 0.07],
        "gap_pp": [-4.43, -2.09], "cliffs_delta": [-0.569, -0.3],
        "p_adj_bh": [0.0000054, 0.0488], "verdict": ["SOBRE_UGC"] * 2,
        "anchors_sha256": ["abc"] * 2, "seed": [42, 42],
    })
    price["rate_dmo"] = [0.0, 0.0]
    price["rate_ugc"] = [0.0443, 0.0209]
    _csv(root / "comparison/price/price_comparison.csv", price)
    _csv(root / "comparison/price/price_distribution_tests.csv", pd.DataFrame({
        "scope": ["DMO_vs_UGC_total"], "emd_wasserstein": [0.1234],
        "jsd_base2": [0.0567], "n_a": [475], "n_b": [200], "seed": [42],
    }))
    _csv(root / "comparison/price/price_topic_crosstab.csv", pd.DataFrame({
        "topic_id": [0, 0, 1], "topic_name": ["playa", "playa", "precio"],
        "source": ["instagram", "reddit", "instagram"],
        "n": [4, 2, 1], "denominator": [20, 10, 20],
        "pct_within_source": [0.2, 0.2, 0.05], "seed": [42, 42, 42],
    }))
    ex_rows = []
    for scope, source in (("dmo", "sectur"), ("tripadvisor", "tripadvisor"),
                          ("instagram", "instagram"), ("reddit", "reddit")):
        for rank in (1, 2, 3):
            raw = (f'&lt;span style=&quot;color:red&quot;&gt;{source}&lt;/span&gt;'
                   if source == "tripadvisor" else source)
            ex_rows.append({
                "rank": rank, "scope": scope, "doc_id": rank, "source": raw,
                "corpus": "DMO" if scope == "dmo" else "UGC",
                "cos_semantic": 0.12345, "sentiment_label": ["positive", "neutral", "negative"][rank - 1],
                "topic_name": "playa", "registro_dominante": "opinion",
                "text_norm": " texto largo " * 40, "seed": 42,
            })
    _csv(root / "comparison/price/price_examples.csv", pd.DataFrame(ex_rows))
    _csv(root / "comparison/price/price_alignment.csv", pd.DataFrame({
        "scope": ["DMO", "UGC"], "mean_cosine": [-0.1816, -0.0072],
        "gap_cosine": [-0.1743, -0.1743],
    }))
    _csv(root / "comparison/price/price_density.csv", pd.DataFrame({
        "source": ["DMO", "UGC"], "bin_center": [-0.2, 0.1],
        "density": [0.1, 0.2], "seed": [42, 42],
    }))
    _write(root / "ngram_interactive.html", "<div>gramas viejos</div>")
    from ngram_visualization import build_all_ngram_figures
    from report_sections import ngram_section_html
    frame = pd.DataFrame({"text_norm": ["playa bonita"] * 4})
    figs = build_all_ngram_figures(frame, frame)
    _write(root / "ngram_interactive.html", ngram_section_html(figs))
    from topic_report import build_topic_report_block
    tsum = pd.read_csv(root / "comparison/topics_joint/bert/topics_summary.csv", encoding="utf-8-sig")
    dt = pd.read_csv(root / "comparison/topics_joint/bert/doc_topics.csv", encoding="utf-8-sig")
    _write(root / "topic_report_section.html",
           build_topic_report_block(tsum, dt, total_docs=3)["html"])
    _write(root / "umap_section.html", "<div>umap</div>")
    for rel in (
        "sentiment/sentiment_by_source_stacked.html", "sentiment/sentiment_score_hist.html",
        "comparison/lexical/wordcloud_logodds_dmo.html",
        "comparison/lexical/wordcloud_logodds_ugc.html",
        "topics/topics_3d_umap.html", "comparison/price/price_umap_concept.html",
    ):
        _write(root / rel, "<div>figura</div>")
    _write(root / "comparison/price/price_scatter_similarity.html", """
    <html><body><div id="scatter" class="plotly-graph-div"></div>
    <script>Plotly.newPlot("scatter", [], {"updatemenus":[{"buttons":[]}]});</script>
    </body></html>
    """)


def _args() -> Namespace:
    return Namespace(
        title="Informe DMO vs UGC", palette="ejecutiva", csv_dmo="dmo.csv",
        model="bertopic", seed=42, bootstrap=200, comparison_tasks=["report"],
        embedding_model="modelo-local", verbose_jaccard=False, ngram_interactive=True,
        umap_3d=True, generate_wordclouds=False,
    )


def _render(return_path: bool = False):
    import tempfile
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _fixture_results(root)
    with contextlib.redirect_stdout(io.StringIO()):
        run_report_task(_args(), root)
    report = next(root.glob("informe_*_*.html"))
    markup = report.read_text(encoding="utf-8")
    tmp.cleanup()
    return (markup, report.name) if return_path else markup


def _section(markup: str, sid: str) -> str:
    start = markup.index(f'id="{sid}"')
    tail = markup[start:]
    nxt = re.search(r"<section\b", tail[10:])
    return tail[:nxt.start() + 10] if nxt else tail


class TestHtmlSectionsPresent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.markup = _render()
        parser = _Parser()
        parser.feed(cls.markup)
        cls.ids = parser.ids
        cls.links = parser.links

    def test_ids_en_orden_y_sin_placeholders(self):
        seen = [sid for sid in self.ids if sid in REQUIRED_IDS + PRICE_IDS]
        self.assertEqual(seen, list(REQUIRED_IDS) + list(PRICE_IDS))
        self.assertNotIn("{{", self.markup)

    def test_ids_eliminados_ausentes_y_nav_sin_enlaces_muertos(self):
        for sid in REMOVED_IDS + ("k-selection", "semantico", "afectivo", "figuras", "metadata"):
            self.assertNotIn(sid, self.ids)
        for href in self.links:
            if href.startswith("#"):
                self.assertIn(href[1:], self.ids)
        for sid in REMOVED_IDS:
            self.assertNotIn(f'href="#{sid}"', self.markup)

    def test_topicos_sin_dibujo_de_caja_y_sin_topic_menos_uno(self):
        topicos = _section(self.markup, "topicos")
        self.assertFalse(any(ch in topicos for ch in BOX_CHARS))
        self.assertIn("Tópico 0", topicos)
        self.assertNotIn("SIN_ASIGNAR", topicos)
        self.assertIn("topic-cards", topicos)

    def test_brechas_cuatro_columnas_y_negrita_en_mayor(self):
        brechas = _section(self.markup, "brechas")
        self.assertEqual(len(re.findall(r"<th(?:\s|>)", brechas)), 4)
        self.assertNotIn("veredicto", brechas.lower())
        self.assertIn("bold-cell", brechas)
        self.assertIn("gap-pos", brechas)
        self.assertIn("gap-neg", brechas)

    def test_tablas_precio_cabeceras(self):
        expected = {
            "precio-frecuencia": 12, "precio-distribucion": 5,
            "precio-contexto": 6, "precio-ejemplos": 10,
        }
        for sid, count in expected.items():
            with self.subTest(section=sid):
                self.assertEqual(len(re.findall(r"<th(?:\s|>)", _section(self.markup, sid))), count)

    def test_ejemplos_badges_y_titulo(self):
        ejemplos = _section(self.markup, "precio-ejemplos")
        self.assertIn("Ejemplos de comentarios con Precio, valor y costo</h3>", ejemplos)
        self.assertIn('<span class="src-badge', ejemplos)
        self.assertNotIn("&lt;span", ejemplos)
        for source in ("tripadvisor", "instagram", "reddit", "sectur"):
            with self.subTest(source=source):
                self.assertLessEqual(ejemplos.count(f'src-{source}"'), 3)

    def test_figuras_sin_botones_en_scatter_y_dos_menus_en_ngramas(self):
        self.assertNotIn("updatemenus", _section(self.markup, "precio-scatter"))
        ngramas = _section(self.markup, "ngramas")
        self.assertEqual(ngramas.count("plotly-graph-div"), 1)
        self.assertEqual(len(re.findall(r'"name":"(?:Normales|Outliers)"', ngramas)), 6)
        self.assertEqual(re.findall(r'"label":"([^"]+)"', ngramas),
                         ["Unigramas", "Bigramas", "Trigramas",
                          "Comentarios normales", "Comentarios outliers"])

    def test_lectura_tres_kpis_sin_caveat_ni_hairline(self):
        lectura = _section(self.markup, "precio-lectura")
        self.assertEqual(lectura.count('class="kpi kpi--price"'), 3)
        self.assertNotIn('<p class="caveat">', lectura)
        self.assertNotIn('<div class="kpi__hairline">', lectura)
        self.assertIn("Comentarios con este concepto", lectura)
        self.assertIn("-0.1743", lectura)
        self.assertLess(self.markup.index('id="precio-lectura"'),
                        self.markup.index('id="precio-frecuencia"'))

    def test_sin_notas_ni_figure_boxes_anidadas(self):
        self.assertNotIn("Pestañas de orden y grupo", self.markup)
        self.assertNotIn('<div class="figure-box"><div class="figure-box"', self.markup)
        self.assertIn(".updatemenu-item-rect", self.markup)
        self.assertIn("#tematica .figure-box iframe", self.markup)

    def test_ejemplos_con_una_sola_celda_de_texto(self):
        ejemplos = _section(self.markup, "precio-ejemplos")
        self.assertGreater(ejemplos.count("<summary"), 0)
        self.assertEqual(ejemplos.count('class="cell-full"'), 0)
        self.assertEqual(ejemplos.count("<summary"), ejemplos.count("</summary>"))

    def test_strip_updatemenus_no_deja_coma_doble(self):
        from report_sections import strip_plotly_updatemenus

        fragment = 'Plotly.newPlot("x", [], {"height":720,"updatemenus":[{"buttons":[]}],"shapes":[]});'
        result = strip_plotly_updatemenus(fragment)
        self.assertNotIn("updatemenus", result)
        self.assertNotIn(",,", result)
        self.assertIn('"height":720,"shapes":[]', result)

    def test_nombre_fechado_sin_informe_final(self):
        _, name = _render(return_path=True)
        self.assertRegex(name, r"^informe_[a-z0-9_]+_\d{4}-\d{2}-\d{2}\.html$")
        self.assertNotEqual(name, "informe_final.html")


class TestFlagsWordclouds(unittest.TestCase):
    def test_llamada_wordclouds_detras_del_flag(self):
        source = (ROOT / "code" / "run_analysis.py").read_text(encoding="utf-8")
        self.assertIn("[OPTIMIZACIÓN-2026]", source)
        self.assertIn("if args.generate_wordclouds:", source)

    def test_generacion_por_defecto_activa(self):
        source = (ROOT / "code" / "run_analysis.py").read_text(encoding="utf-8")
        self.assertIn('"--generate-wordclouds", action=argparse.BooleanOptionalAction,\n'
                      '                   default=True', source)
        self.assertIn('Wordclouds omitidas', source)


if __name__ == "__main__":
    unittest.main()
