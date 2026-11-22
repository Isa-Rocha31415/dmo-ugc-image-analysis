"""Verifica que el informe HTML final contenga las 8 secciones obligatorias."""
import contextlib
import io
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
    "ejecutivo",
    "polaridad-descriptiva",
    "lexico",
    "ngramas",
    "topicos",
    "umap-3d",
    "pruebas-estadisticas",
    "metadatos-run",
)


class _IdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_results(root: Path) -> None:
    for sub in ("comparison/base", "comparison/topics_joint", "comparison/thematic",
                "comparison/lexical", "comparison/price", "sentiment", "topics"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame({
        "doc_id": [0, 1],
        "text": ["playa bonita", "hotel cómodo"],
        "text_norm": ["playa bonita", "hotel cómodo"],
        "corpus": ["DMO", "UGC"],
        "source": ["sectur", "ugc"],
        "sentiment_label": ["positive", "neutral"],
    })
    base.to_csv(root / "comparison/base/analysis_base.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"doc_id": [9], "text": ["outlier"]}).to_csv(
        root / "comparison/base/outliers_base.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "topic_id": [0], "topic_name": ["playa limpia"],
        "keywords_top8": ["playa; limpia; arena; sol; mar; familia; centro; hotel"],
        "n_docs": [2],
    }).to_csv(root / "comparison/topics_joint/topics_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"k": [4], "selected": [True]}).to_csv(
        root / "comparison/topics_joint/topic_selection_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"doc_id": [0, 1], "topic_id": [0, 0], "topic_name": ["playa limpia"] * 2}).to_csv(
        root / "comparison/topics_joint/doc_topics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"scope": ["ugc_total"], "jsd_base2": [0.01], "emd_cosine": [0.02]}).to_csv(
        root / "comparison/thematic/distribution_distances.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "scope": ["ugc_total"], "topic_name": ["playa"], "p_dmo": [0.1], "p_ugc": [0.2],
        "gap_pp": [10.0], "verdict": ["SOBRE_UGC"],
    }).to_csv(root / "comparison/thematic/gaps_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"metric": ["coverage"], "scope": ["ugc_total"], "k": [100], "value": [0.5]}).to_csv(
        root / "comparison/lexical/metrics_lexical.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"scope": ["ugc_total"], "rate": [0.2]}).to_csv(
        root / "comparison/price/mention_rate_by_source.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"tau": [0.5]}).to_csv(
        root / "comparison/price/tau_calibration.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"task": ["base"], "timestamp": ["2026-01-01"]}).to_csv(
        root / "comparison/base/run_metadata.csv", index=False, encoding="utf-8-sig")

    for rel in (
        "sentiment/sentiment_by_source_stacked.html",
        "sentiment/sentiment_score_hist.html",
        "comparison/lexical/wordcloud_logodds_dmo.html",
        "comparison/lexical/wordcloud_logodds_ugc.html",
        "comparison/topics_joint/curvas_k.html",
        "comparison/price/price_scatter_valence.html",
        "topics/topics_3d_umap.html",
        "ngram_interactive.html",
        "umap_section.html",
        "topic_report_section.html",
    ):
        _write(root / rel, "<div>figura</div>")
    (root / "ngram_interactive.html").write_text(
        '<section id="ngramas"><div>figura</div></section>', encoding="utf-8")

    from topic_report import build_topic_report_block

    tsum = pd.read_csv(
        root / "comparison" / "topics_joint" / "topics_summary.csv",
        encoding="utf-8-sig")
    dt = pd.read_csv(
        root / "comparison" / "topics_joint" / "doc_topics.csv",
        encoding="utf-8-sig")
    topic_block = build_topic_report_block(tsum, dt, total_docs=2)
    (root / "topic_report_section.html").write_text(
        topic_block["html"], encoding="utf-8")


def _args() -> Namespace:
    return Namespace(
        title="Informe DMO vs UGC", palette="ejecutiva", csv_dmo="dmo.csv",
        model="bertopic", seed=42, bootstrap=200, comparison_tasks=["report"],
        embedding_model="modelo-local", verbose_jaccard=False, ngram_interactive=True,
        umap_3d=True, generate_wordclouds=False,
    )


class TestHtmlSectionsPresent(unittest.TestCase):
    def test_ocho_secciones_con_sus_ids(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture_results(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run_report_task(_args(), root)
            markup = (root / "informe_final.html").read_text(encoding="utf-8")
            parser = _IdParser()
            parser.feed(markup)
            missing = [sid for sid in REQUIRED_IDS if sid not in parser.ids]
            self.assertEqual(missing, [], f"Faltan secciones: {missing}")
            self.assertNotIn("{{", markup)
            self.assertIn("Tópico 0: playa limpia", markup)
            self.assertIn("Cobertura:", markup)

    def test_secciones_pendientes_tienen_placeholder(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture_results(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run_report_task(_args(), root)
            markup = (root / "informe_final.html").read_text(encoding="utf-8")
            self.assertIn("id=\"semantico-pendiente\"", markup)
            self.assertIn("id=\"afectivo-pendiente\"", markup)
            self.assertIn("id=\"figuras-pendiente\"", markup)


class TestFlagsWordclouds(unittest.TestCase):
    def test_llamada_wordclouds_detras_del_flag(self):
        source = (ROOT / "code" / "run_analysis.py").read_text(encoding="utf-8")
        self.assertIn("[OPTIMIZACIÓN-2026]", source)
        self.assertIn("if args.generate_wordclouds:", source)


if __name__ == "__main__":
    unittest.main()
