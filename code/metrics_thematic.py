"""
metrics_thematic.py — TAREA P3: capa temática L2 (gaps de prevalencia con inferencia).

- Prevalencia p_t = docs_tópico / docs_ASIGNADOS del scope (SIN_ASIGNAR se reporta aparte).
- Gaps DMO vs UGC_total y por fuente: gap_pp, gap_ratio, CI bootstrap estratificado
  por corpus (B=--bootstrap), χ² si esperados>=5 / Fisher exact si no, p_adj BH (q=0.05
  dentro de cada scope), Cohen's h, veredicto SOBRE_DMO / SOBRE_UGC / SIN_DIF.
- gaps_by_sentiment: DMO-pos vs UGC-pos y DMO-neg vs UGC-neg (controla confusor tema/sentimiento).
- distribution_distances: JSD base 2 sobre distribuciones completas (incluye SIN_ASIGNAR)
  y EMD con costo = distancia coseno entre centroides de embeddings (transporte exacto
  vía linprog HiGHS; sin dependencias nuevas).
Outputs en ../results/comparison/thematic/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import chi2_contingency, entropy, fisher_exact
from sklearn.preprocessing import normalize

from comparison.common import append_metadata, print_header, save_csv
from io_corpus import CORPUS_COLUMN, SOURCE_COLUMN

Q_LEVEL = 0.05
GAPS_COLUMNS = [
    "topic_id", "topic_name", "scope", "p_dmo", "p_ugc", "gap_pp", "gap_ratio",
    "ci_low_pp", "ci_high_pp", "test", "p_value", "p_adj", "cohens_h", "verdict",
]
DIST_COLUMNS = ["scope", "jsd_base2", "emd_cosine", "n_topics"]


# ─────────────────────────── Funciones puras ───────────────────────────
def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (q=0.05 se aplica al veredicto, no al ajuste)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def cohens_h(p1: float, p2: float) -> float:
    """h = 2·arcsin(√p1) − 2·arcsin(√p2)."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def two_by_two_test(a: int, b: int, c: int, d: int) -> tuple[str, float]:
    """χ² (sin corrección) si todos los esperados>=5; Fisher exact en caso contrario."""
    table = np.array([[a, b], [c, d]])
    n = table.sum()
    if n == 0:
        return "fisher", 1.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    if (expected >= 5).all():
        return "chi2", float(chi2_contingency(table, correction=False)[1])
    return "fisher", float(fisher_exact(table)[1])


def verdict(gap_pp: float, p_adj: float, q: float = Q_LEVEL) -> str:
    """Veredicto con p ajustada (BH) dentro del scope."""
    if p_adj < q and gap_pp > 0:
        return "SOBRE_DMO"
    if p_adj < q and gap_pp < 0:
        return "SOBRE_UGC"
    return "SIN_DIF"


def jsd_base2(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence base 2 (rango [0,1])."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2)))


def emd_cosine(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> float:
    """Earth Mover's Distance exacta con matriz de costo (linprog HiGHS)."""
    T = len(p)
    I, ones = sparse.eye(T), sparse.csr_matrix(np.ones((1, T)))
    A = sparse.vstack([sparse.kron(I, ones), sparse.kron(ones, I)], format="csr")
    res = linprog(cost.ravel(), A_eq=A, b_eq=np.concatenate([p, q]),
                  bounds=(0, None), method="highs")
    return float(res.fun) if res.success else float("nan")


def bootstrap_gap_ci(lab_d: np.ndarray, lab_u: np.ndarray, T: int,
                     B: int, seed: int) -> np.ndarray:
    """CI 95% de gap_pp por tópico; remuestreo independiente dentro de cada corpus."""
    nd, nu = len(lab_d), len(lab_u)
    if B <= 0 or nd == 0 or nu == 0:
        return np.full((2, T), np.nan)
    rng = np.random.default_rng(seed)
    gaps = np.empty((B, T))
    for b in range(B):
        cd = np.bincount(lab_d[rng.integers(0, nd, nd)], minlength=T) / nd
        cu = np.bincount(lab_u[rng.integers(0, nu, nu)], minlength=T) / nu
        gaps[b] = 100 * (cd - cu)
    return np.percentile(gaps, [2.5, 97.5], axis=0)


def _remap(labels: np.ndarray, topics: list[int]) -> np.ndarray:
    """Reindexa etiquetas de tópico a 0..T-1 (los labels llegan ya filtrados a >=0)."""
    m = {t: i for i, t in enumerate(topics)}
    return np.array([m[int(t)] for t in labels], dtype=int)


def _gaps_for_scope(lab_d_raw, lab_u_raw, topics, names, scope, B, seed) -> list[dict]:
    """Filas de gaps_table para un scope (labels ya filtrados a asignados)."""
    T = len(topics)
    lab_d, lab_u = _remap(lab_d_raw, topics), _remap(lab_u_raw, topics)
    nd, nu = len(lab_d), len(lab_u)
    cd, cu = np.bincount(lab_d, minlength=T), np.bincount(lab_u, minlength=T)
    p_d, p_u = cd / max(nd, 1), cu / max(nu, 1)
    ci = bootstrap_gap_ci(lab_d, lab_u, T, B, seed)
    tests = [two_by_two_test(cd[t], nd - cd[t], cu[t], nu - cu[t]) for t in range(T)]
    p_adj = bh_adjust(np.array([p for _, p in tests]))
    rows = []
    for t, tid in enumerate(topics):
        gap = 100 * (p_d[t] - p_u[t])
        rows.append({
            "topic_id": tid, "topic_name": names[tid], "scope": scope,
            "p_dmo": round(p_d[t], 6), "p_ugc": round(p_u[t], 6),
            "gap_pp": round(gap, 4),
            "gap_ratio": round(p_d[t] / p_u[t], 4) if p_u[t] > 0 else np.nan,
            "ci_low_pp": round(ci[0, t], 4), "ci_high_pp": round(ci[1, t], 4),
            "test": tests[t][0], "p_value": tests[t][1], "p_adj": p_adj[t],
            "cohens_h": round(cohens_h(p_d[t], p_u[t]), 4),
            "verdict": verdict(gap, p_adj[t]),
        })
    return rows


# ─────────────────────────── Tarea 'thematic' ───────────────────────────
def run_thematic_task(args: argparse.Namespace, results_dir: Path) -> list[Path]:
    """Tarea P3: gaps de prevalencia + distancias distribucionales."""
    print_header("PASO 11: CAPA TEMÁTICA L2 — GAPS DE PREVALENCIA DMO vs UGC")
    base_dir = results_dir / "comparison" / "base"
    tj_dir = results_dir / "comparison" / "topics_joint"
    out_dir = results_dir / "comparison" / "thematic"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(base_dir / "analysis_base.csv", encoding="utf-8-sig")
    doc = pd.read_csv(tj_dir / "doc_topics.csv", encoding="utf-8-sig")
    df = base.merge(doc[["doc_id", "topic_id", "topic_name"]], on="doc_id",
                    validate="one_to_one")
    if len(df) != len(base):
        raise ValueError(
            "doc_topics no cubre analysis_base; ejecuta --comparison-tasks base topics."
        )
    cents = np.load(tj_dir / "topic_centroids.npy")
    topics = sorted(int(t) for t in df.topic_id.unique() if t >= 0)
    T = len(topics)
    if T == 0:
        raise ValueError("No hay tópicos asignados (todo SIN_ASIGNAR); revisa el PASO 9.")
    if cents.shape[0] != T:
        raise ValueError("topic_centroids.npy no coincide con los tópicos asignados.")
    names = dict(zip(doc.topic_id, doc.topic_name))
    corpus = df[CORPUS_COLUMN].astype(str).to_numpy()
    source = df[SOURCE_COLUMN].astype(str).to_numpy()
    sent = df["sentiment_label"].astype(str).to_numpy()
    lab = df["topic_id"].to_numpy(dtype=int)
    assigned = lab >= 0

    # ── Vistas DMO reutilizables (C2) ──
    dmo_mask = corpus == "DMO"
    dmo_all = lab[dmo_mask]
    dmo_asg = lab[dmo_mask & assigned]

    print("  ℹ️ Prevalencia: denominador = docs ASIGNADOS; SIN_ASIGNAR se reporta como sensibilidad.")
    print("  ℹ️ EMD: costo = distancia coseno entre centroides de embeddings (espacio semántico "
          "del modelo conjunto); transporte exacto con linprog (HiGHS).")

    scopes = [("DMO", dmo_mask), ("UGC_total", corpus == "UGC")] + [
        (s, (corpus == "UGC") & (source == s))
        for s in sorted(set(source[corpus == "UGC"]))
    ]
    written: list[Path] = []

    # ── prevalence_by_source.csv + sensibilidad SIN_ASIGNAR ──
    prev_rows, un_rows = [], []
    for scope, mask in scopes:
        n_scope = int((mask & assigned).sum())
        n_un = int((mask & ~assigned).sum())
        un_rows.append(dict(scope=scope, n_docs=int(mask.sum()), n_unassigned=n_un,
                            share_unassigned=round(float(n_un / max(mask.sum(), 1)), 6)))
        for tid in topics:
            n_t = int((mask & (lab == tid)).sum())
            prev_rows.append(dict(topic_id=tid, topic_name=names[tid], scope=scope,
                                  n_docs_topic=n_t, n_docs_scope=n_scope,
                                  prevalence=round(n_t / max(n_scope, 1), 6)))
    written.append(save_csv(pd.DataFrame(prev_rows), out_dir / "prevalence_by_source.csv"))
    written.append(save_csv(pd.DataFrame(un_rows), out_dir / "unassigned_by_scope.csv"))

    # ── C2: sin DMO asignada no hay comparación posible ──
    if dmo_all.size == 0 or dmo_asg.size == 0:
        raise ValueError("La DMO no tiene documentos (asignados); revisa el PASO 9.")

    # ── gaps_table.csv (DMO vs UGC_total y por fuente) ──
    gap_rows = []
    for scope, mask in scopes[1:]:
        u_lab = lab[mask & assigned]
        if len(u_lab) == 0:
            print(f"  ⚠️ scope={scope} sin documentos asignados; se omite.")
            continue
        gap_rows += _gaps_for_scope(dmo_asg, u_lab, topics, names,
                                    "ugc_total" if scope == "UGC_total" else scope,
                                    args.bootstrap, args.seed)
    gaps = pd.DataFrame(gap_rows) if gap_rows else pd.DataFrame(columns=GAPS_COLUMNS)
    written.append(save_csv(gaps, out_dir / "gaps_table.csv"))
    # ── C1 ──
    if gaps.empty:
        raise ValueError(
            "No se pudieron calcular gaps: ninguna fuente UGC tiene docs asignados."
        )

    # ── gaps_by_sentiment.csv (controla confusor tema/sentimiento) ──
    sent_rows = []
    for s in ("positive", "negative"):
        ld = lab[dmo_mask & assigned & (sent == s)]
        lu = lab[(corpus == "UGC") & assigned & (sent == s)]
        if len(ld) == 0 or len(lu) == 0:
            print(f"  ⚠️ Sin documentos suficientes para el estrato '{s}'; se omite.")
            continue
        sent_rows += _gaps_for_scope(ld, lu, topics, names, f"ugc_total_{s}",
                                     args.bootstrap, args.seed)
    # ── C3 ──
    sent_df = pd.DataFrame(sent_rows) if sent_rows else pd.DataFrame(columns=GAPS_COLUMNS)
    written.append(save_csv(sent_df, out_dir / "gaps_by_sentiment.csv"))

    # ── distribution_distances.csv ──
    cost = np.clip(1.0 - normalize(cents) @ normalize(cents).T, 0, 2)
    cat_all = topics + [-1]
    full_d = np.array([(dmo_all == t).mean() for t in cat_all])
    p = np.array([(dmo_asg == t).mean() for t in topics])
    p = p / max(p.sum(), 1e-12)
    dist_rows = []
    for scope, mask in scopes[1:]:
        if not mask.any():
            continue
        full_u = np.array([(lab[mask] == t).mean() for t in cat_all])
        q = np.array([(lab[mask & assigned] == t).mean() for t in topics])
        q = q / max(q.sum(), 1e-12)
        dist_rows.append(dict(scope="ugc_total" if scope == "UGC_total" else scope,
                              jsd_base2=round(jsd_base2(full_d, full_u), 6),
                              emd_cosine=round(emd_cosine(p, q, cost), 6), n_topics=T))
    distances = pd.DataFrame(dist_rows) if dist_rows else pd.DataFrame(columns=DIST_COLUMNS)
    written.append(save_csv(distances, out_dir / "distribution_distances.csv"))

    # ── Consola ──
    print("\n  🔍 Sensibilidad SIN_ASIGNAR por corpus/fuente:")
    for r in un_rows:
        print(f"    {r['scope']:<14s} {100 * r['share_unassigned']:5.1f}%  "
              f"({r['n_unassigned']}/{r['n_docs']})")

    g = gaps[gaps.scope == "ugc_total"].copy()
    cols = ["topic_name", "gap_pp", "ci_low_pp", "ci_high_pp",
            "test", "p_adj", "cohens_h", "verdict"]
    if not g.empty:
        g["abs_gap"] = g.gap_pp.abs()
        print("\n  📊 Top-10 gaps DMO vs UGC_total (|gap_pp|):")
        print(g.nlargest(10, "abs_gap")[cols].round(4).to_string(index=False))
    else:
        print("\n  ⚠️ No hay gaps para el scope 'ugc_total'.")

    n_fisher = int((gaps.test == "fisher").sum())
    print(f"\n  🧪 Fisher exact usado automáticamente en {n_fisher}/{len(gaps)} comparaciones "
          f"(esperados<5 por el tamaño reducido de la DMO).")

    print("\n  🌐 Distancias distribucionales globales:")
    print(distances.to_string(index=False))
    # ── C4 ──
    dsel = distances[distances.scope == "ugc_total"] if not distances.empty else distances
    if not dsel.empty:
        d0 = dsel.iloc[0]
        frase = ("muy alineadas" if d0.jsd_base2 < 0.05 else
                 "moderadamente distintas" if d0.jsd_base2 < 0.15 else
                 "marcadamente distintas")
        print(f"\n  💬 Las distribuciones temáticas DMO y UGC están {frase} "
              f"(JSD₂={d0.jsd_base2:.3f}; EMD-coseno={d0.emd_cosine:.3f}).")
    else:
        print("\n  ⚠️ No hay distancias para 'ugc_total'; revisa la asignación de tópicos.")

    written.append(append_metadata(
        base_dir / "run_metadata.csv", task="thematic",
        params={**vars(args), "q_level": Q_LEVEL, "n_topics": T,
                "denominator": "assigned_docs", "emd_cost": "cosine_centroids"},
        repo_dir=Path(__file__).resolve().parent.parent,
    ))
    return written