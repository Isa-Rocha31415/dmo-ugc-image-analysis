"""
Tests mínimos para la carga de corpus DMO + UGC (`io_corpus`).

Ejecutables con `python -m unittest discover tests` o con `pytest -q`.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from io_corpus import (  # noqa: E402
    CORPUS_COLUMN,
    DMO_LABEL,
    SOURCE_COLUMN,
    build_unified_frame,
    resolve_text_column,
)


class TestResolveTextColumn(unittest.TestCase):
    def test_usa_la_columna_pedida(self):
        df = pd.DataFrame({"text": ["hola"], "source": ["x"]})
        self.assertEqual(resolve_text_column(df, "text"), "text")

    def test_cae_en_alias_comunes(self):
        df = pd.DataFrame({"texto": ["hola"]})
        self.assertEqual(resolve_text_column(df, "text"), "texto")

    def test_error_si_no_existe(self):
        df = pd.DataFrame({"foo": [1]})
        with self.assertRaises(KeyError):
            resolve_text_column(df, "text")


class TestBuildUnifiedFrame(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)

        dmo = pd.DataFrame(
            {
                "text": ["El hotel fue muy bueno", "Habitación amplia y limpia"],
                "source": [DMO_LABEL, DMO_LABEL],
            }
        )
        ugc = pd.DataFrame(
            {
                "text": ["Pésima experiencia, no volvería", "Hotelorusado, servicio lento"],
                "source": ["tripadvisor", "reddit"],
            }
        )
        self.dmo_path = base / "dmo.csv"
        self.ugc_path = base / "ugc.csv"
        dmo.to_csv(self.dmo_path, index=False)
        ugc.to_csv(self.ugc_path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_concatena_dmo_y_ugc(self):
        df = build_unified_frame(
            dmo_csv=str(self.dmo_path),
            ugc_csvs=[str(self.ugc_path)],
            text_column="text",
        )
        self.assertEqual(len(df), 4)
        self.assertEqual(df[CORPUS_COLUMN].value_counts().to_dict(), {"UGC": 2, "DMO": 2})

    def test_dmo_usa_la_etiqueta_fija(self):
        df = build_unified_frame(
            dmo_csv=str(self.dmo_path),
            ugc_csvs=[str(self.ugc_path)],
            text_column="text",
        )
        dmo_rows = df[df[CORPUS_COLUMN] == "DMO"]
        self.assertTrue((dmo_rows[SOURCE_COLUMN] == DMO_LABEL).all())

    def test_no_duplica_filas_dmo_en_ugc(self):
        """Si el mismo archivo se pasa en ambos lados, no se duplica DMO."""
        df = build_unified_frame(
            dmo_csv=str(self.dmo_path),
            ugc_csvs=[str(self.dmo_path)],
            text_column="text",
        )
        self.assertEqual(len(df), 2)
        self.assertEqual(df[CORPUS_COLUMN].value_counts().to_dict(), {"DMO": 2})

    def test_ugc_conserva_su_fuente(self):
        df = build_unified_frame(
            dmo_csv=str(self.dmo_path),
            ugc_csvs=[str(self.ugc_path)],
            text_column="text",
        )
        ugc_rows = df[df[CORPUS_COLUMN] == "UGC"]
        self.assertEqual(set(ugc_rows[SOURCE_COLUMN]), {"tripadvisor", "reddit"})


if __name__ == "__main__":
    unittest.main()
