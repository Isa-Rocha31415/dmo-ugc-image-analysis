"""Tests del scatter 3D UMAP de comentarios con tópicos."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from visualization import build_umap_3d_plot  # noqa: E402


def _fixture(n: int = 50) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    textos = [f"Comentario sintético número {i} sobre la playa y el hotel" for i in range(n)]
    df = pd.DataFrame({
        "doc_id": range(n),
        "text": textos,
        "text_norm": textos,
        "topic_id": [i % 3 for i in range(n)],
        "topic_name": [f"tópico {i % 3}" for i in range(n)],
        "source": ["DMO" if i % 2 else "UGC" for i in range(n)],
        "sentiment_label": [["positive", "neutral", "negative"][i % 3] for i in range(n)],
    })
    embeddings = rng.normal(size=(n, 8)).astype(np.float32)
    coords = rng.normal(size=(n, 3)).astype(np.float32)
    return df, embeddings, coords


class TestBuildUmap3DPlot(unittest.TestCase):
    def test_scatter3d_con_un_trace_por_topico_y_50_puntos(self):
        df, emb, coords = _fixture()
        fig = build_umap_3d_plot(df, emb, coords_3d=coords)
        self.assertEqual(len(fig.data), 3)
        self.assertTrue(all(t.type == "scatter3d" for t in fig.data))
        total = sum(len(t.x) for t in fig.data)
        self.assertEqual(total, len(df))
        self.assertTrue(all(t.showlegend for t in fig.data))
        self.assertEqual(
            sorted(t.name for t in fig.data),
            ["tópico 0", "tópico 1", "tópico 2"],
        )

    def test_hover_contiene_topico_fuente_sentimiento_y_texto(self):
        df, emb, coords = _fixture()
        fig = build_umap_3d_plot(df, emb, coords_3d=coords)
        trace = fig.data[0]
        self.assertIsNotNone(trace.customdata)
        self.assertEqual(len(trace.customdata), len(trace.x))
        self.assertEqual(len(trace.customdata[0]), 4)
        cell = " ".join(str(v) for v in trace.customdata[0])
        self.assertIn("tópico", cell)
        self.assertIn("Comentario sintético", cell)
        for marker in ("<br>", "<extra></extra>", "%{customdata[3]}"):
            self.assertIn(marker, trace.hovertemplate)

    def test_texto_se_trunca_a_200_caracteres_y_se_escapea(self):
        df, emb, coords = _fixture(n=3)
        df["text_norm"] = ["<" + "a" * 300 + ">"] * 3
        fig = build_umap_3d_plot(df, emb, coords_3d=coords)
        sample = str(fig.data[0].customdata[0][3])
        self.assertLessEqual(len(sample), 200)
        self.assertIn("&lt;", sample)
        self.assertNotIn("<a", sample)

    def test_umap_se_llama_con_nueve_componentes_no(self):
        df, emb, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "3 componentes"):
            build_umap_3d_plot(df, emb, coords_3d=np.zeros((3, 2)))

    def test_error_claro_si_falta_umap(self):
        import builtins
        from unittest.mock import patch

        df, emb, _ = _fixture(n=3)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "umap":
                raise ImportError("no umap")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ImportError, "umap-learn"):
                build_umap_3d_plot(df, emb)

    def test_desbalance_longitudes(self):
        df, emb, coords = _fixture(n=5)
        with self.assertRaises(ValueError):
            build_umap_3d_plot(df, emb[:3], coords_3d=coords)


if __name__ == "__main__":
    unittest.main()
