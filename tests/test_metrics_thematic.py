import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from metrics_thematic import (bh_adjust, bootstrap_gap_ci, cohens_h,  # noqa: E402
                              emd_cosine, jsd_base2, two_by_two_test, verdict)

def test_bh_conocido():
    q = bh_adjust(np.array([0.01, 0.04, 0.03, 0.002]))
    assert np.all(q >= np.array([0.01, 0.04, 0.03, 0.002]) - 1e-12) and q.max() <= 1

def test_fisher_automatico():
    assert two_by_two_test(1, 474, 50, 22000)[0] == "fisher"
    assert two_by_two_test(100, 375, 5000, 17000)[0] == "chi2"

def test_veredicto():
    assert verdict(5.0, 0.01) == "SOBRE_DMO"
    assert verdict(-5.0, 0.01) == "SOBRE_UGC"
    assert verdict(5.0, 0.2) == "SIN_DIF"

def test_jsd_extremos():
    p = np.array([0.5, 0.5]); q = np.array([0.5, 0.5])
    assert jsd_base2(p, q) < 1e-12
    assert abs(jsd_base2(np.array([1, 0]), np.array([0, 1])) - 1) < 1e-9

def test_emd_identica_cero():
    C = np.array([[0, 1], [1, 0]], float)
    assert emd_cosine(np.array([0.5, 0.5]), np.array([0.5, 0.5]), C) < 1e-9
    assert emd_cosine(np.array([1, 0]), np.array([0, 1]), C) > 0.9

def test_cohens_h_y_ci():
    assert cohens_h(0.5, 0.5) == 0
    ci = bootstrap_gap_ci(np.array([0, 0, 1]), np.array([0, 1, 1, 1]), 2, B=50, seed=42)
    assert ci.shape == (2, 2) and np.all(ci[0] <= ci[1])