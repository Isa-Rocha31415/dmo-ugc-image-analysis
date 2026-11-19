"""PASO 1: doble texto (fixtures sintéticos, sin datos reales)."""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from text_preprocessing import clean_natural, get_language_resources, prepare_corpus  # noqa: E402
def _fx():
    return pd.DataFrame({"text": ["¡Los SERVICIOS son excelentes! 😀 https://x.es 123 #calidad ¶",
                                  "Muy rápido y cómodo, sin duda 45€", "x", ""],
                         "source": ["sectur", "tripadvisor", "sectur", "google"],
                         "corpus": ["DMO", "UGC", "DMO", "UGC"], "rating": [5, 4, 3, 2]})
def test_limpieza_natural():
    t = clean_natural("¡Los SERVICIOS son excelentes! 😀 https://x.es 123 #calidad ¶")
    assert all(s not in t for s in ("http", "😀", "123", "¶", "#")) and "servicios" in t
def test_doble_texto():
    df, st = prepare_corpus(_fx(), "text", "es")
    f = df.iloc[0]
    assert len(df) == 2 and (f["source"], f["corpus"], f["rating"]) == ("sectur", "DMO", 5)
    assert "los" in f["text"] and "excelentes" in f["text"]  # natural, sin lematizar
    assert "los" not in f["text_norm"] and "servicio" in f["text_norm"]
    assert st["removed_by_corpus"] == {"DMO": 1, "UGC": 1} and "sectur" in st["removed_by_source"]
def test_filtros_y_dmo():
    with pytest.raises(ValueError, match="descartadas"):
        prepare_corpus(pd.DataFrame({"text": ["x", ""], "corpus": ["UGC"] * 2}), "text", "es")
    with pytest.raises(ValueError, match="vacío"):
        prepare_corpus(pd.DataFrame(columns=["text", "corpus"]), "text", "es")
    d = _fx(); d.loc[d.corpus == "DMO", "text"] = "x"
    with pytest.raises(ValueError, match="DMO"):
        prepare_corpus(d, "text", "es")
def test_determinismo_y_lang():
    a, _ = prepare_corpus(_fx(), "text", "es")
    b, _ = prepare_corpus(_fx(), "text", "ES")
    pd.testing.assert_frame_equal(a, b)  # --lang case-insensitive
    with pytest.raises(ValueError, match="Idioma no soportado"):
        get_language_resources("xx")
    with pytest.raises(RuntimeError, match="spacy"):
        get_language_resources("fr")  # sin descarga: error claro
