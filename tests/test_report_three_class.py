"""Tests de integración ligera: reporte de polaridad y HTML sin Jaccard."""
import contextlib
import io
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "legacy"))

from comparison.base_builder import run_base_task  # noqa: E402
from report_generator import run_report_task  # noqa: E402
from sentiment_report import reporte_sentimiento_por_fuente  # noqa: E402
from utils import palette_to_plot_colors  # noqa: E402


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "doc_id": range(6),
        "text": ["texto de prueba"] * 6,
        "corpus": ["DMO", "DMO", "UGC", "UGC", "UGC", "UGC"],
        "source": ["sectur", "sectur", "tripadvisor", "tripadvisor", "reddit", "reddit"],
        "sentiment_label": [
            "positive", "neutral", "negative", "neutral", "positive", "negative",
        ],
        "sentiment_score": [0.8, 0.0, -0.8, 0.1, 0.9, -0.4],
    })


class TestReporteSentimientoTernario(unittest.TestCase):
    def test_escribe_tres_clases_y_tablas_csv(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = reporte_sentimiento_por_fuente(
                _base_frame(), source_column="source", output_dir=tmp,
                palette=palette_to_plot_colors("ejecutiva"),
            )
            summary = result["summary"]
            self.assertEqual(summary["positivos"].sum(), 2)
            self.assertEqual(summary["neutros"].sum(), 2)
            self.assertEqual(summary["negativos"].sum(), 2)
            for name in ("sentiment_by_source_summary.csv",
                         "sentiment_by_source_counts.csv",
                         "sentiment_by_source_stacked_data.csv"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            html = (Path(tmp) / "sentiment_by_source_stacked.html").read_text(encoding="utf-8")
            self.assertIn("POSITIVO", html)
            self.assertIn("NEUTRO", html)
            self.assertIn("NEGATIVO", html)


class TestBaseSoloNormales(unittest.TestCase):
    def test_analysis_base_excluye_outliers_y_usa_tres_clases(self):
        import tempfile

        df = _base_frame()
        clean = df.iloc[[0, 1, 2, 4]].reset_index(drop=True).copy()
        outliers = df.iloc[[3, 5]].reset_index(drop=True).copy()
        args = Namespace(
            text_col="text", lang="es", csv_dmo="dmo.csv", csv_ugc=["ugc.csv"],
            outlier_method="IF", contamination="auto", comparison_tasks=["base"],
            model="bertopic", embedding_model="modelo-local", seed=42, bootstrap=200,
            palette="ejecutiva", sentiment_neutral_tau=0.15, allow_recompute=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                run_base_task(
                    args, Path(tmp),
                    inputs={
                        "df": df,
                        "embeddings": __import__("numpy").zeros((len(df), 3)),
                        "df_clean": clean,
                        "df_outliers": outliers,
                    },
                )
            base = pd.read_csv(
                Path(tmp) / "comparison" / "base" / "analysis_base.csv",
                encoding="utf-8-sig",
            )
            self.assertEqual(len(base), 4)
            self.assertFalse(bool(base["is_outlier"].any()))
            self.assertEqual(set(base["sentiment_label"]), {"positive", "negative", "neutral"})


class TestReporteHTMLSinJaccard(unittest.TestCase):
    def test_jaccard_solo_en_consola_y_polaridad_separada(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "comparison" / "base"
            lex_dir = root / "comparison" / "lexical"
            base_dir.mkdir(parents=True)
            lex_dir.mkdir(parents=True)
            _base_frame().to_csv(base_dir / "analysis_base.csv", index=False,
                                 encoding="utf-8-sig")
            pd.DataFrame({
                "metric": ["jaccard_freq", "jaccard_tfidf"],
                "scope": ["ugc_total", "ugc_total"],
                "k": [100, 100],
                "value": [0.25, 0.4],
                "ci_low": [0.2, 0.3],
                "ci_high": [0.3, 0.5],
            }).to_csv(lex_dir / "metrics_lexical.csv", index=False, encoding="utf-8-sig")

            args = Namespace(
                title="Informe DMO vs UGC", palette="ejecutiva", csv_dmo="dmo.csv",
                model="bertopic", seed=42, bootstrap=200, comparison_tasks=["base"],
                embedding_model="modelo-local", verbose_jaccard=True,
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                written = run_report_task(args, root)

            report = next(root.glob("informe_*_*.html"))
            html = report.read_text(encoding="utf-8")
            self.assertIn("Métricas descriptivas de polaridad", html)
            self.assertNotIn("POSITIVO", html)
            self.assertNotIn("Jaccard", html)
            self.assertIn("jaccard_freq", buf.getvalue())
            self.assertTrue((root / "comparison" / "report" / "sentiment_descriptive.csv").exists())
            self.assertIn(report, written)


if __name__ == "__main__":
    unittest.main()
