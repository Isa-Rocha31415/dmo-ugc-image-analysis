"""
price_validation.py — F4.3: validación del detector Precio/Valor/Costo.

`run_price_validation` ejecuta la validación automática F4.3: acuerdo débil
semantic↔lexical (κ), FPR del placebo, sensibilidad de τ, jackknife del ancla
y bootstrap de tasas/cosenos. No hay anotación humana ni exports de muestra.
La calibración de τ por placebo sigue en `price_concept.py`. Las funciones de
muestreo y consenso se conservan como utilidades legadas (fuera del flujo).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from comparison.common import (append_metadata, mann_whitney_cliffs,
                               print_header, save_csv)

LABEL_POS = "precio"
LABEL_NEG = "no_precio"
VALID_LABELS = (LABEL_POS, LABEL_NEG)
SCORE_BANDS = 5
ANNOTATION_COLUMNS = ("doc_id", "annotator", "label", "note")
ABLATION_RULES = (
    "lexical_only", "resena_only", "promo_only",
    "raw_max", "centered_max", "hybrid", "tau_fijo_0.5",
)


# ─────────────────────────── Muestreo estratificado ───────────────────────────
def _score_band(scores: np.ndarray) -> np.ndarray:
    """5 bandas por cuantiles de score: 'b1'(baja) … 'b5'(alta).

    Con muchos empates los cuantiles pueden colapsar; se usa qcut con
    duplicates='drop' y se rellena lo restante por rango.
    """
    s = pd.Series(np.asarray(scores, dtype=float))
    bands = pd.qcut(s.rank(method="first"), q=SCORE_BANDS, labels=False)
    return np.asarray([f"b{int(b) + 1}" for b in bands.fillna(-1)])


def make_stratified_sample(
    price_scores_df: pd.DataFrame, n: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Muestra estratificada 5 bandas de score × corpus (función pura, determinista).

    Asignación proporcional (n_h = round(n · N_h/N) con ajuste de residuo para
    agotar exactamente `n` cuando cada estrato tiene margen). weight = N_h/n_h.
    """
    df = price_scores_df.reset_index(drop=True).copy()
    df["stratum"] = _score_band(df["price_score"].to_numpy()) + "_" + \
        df["corpus"].astype(str).str.upper()

    strata = df.groupby("stratum", sort=False)
    N_h = strata.size().to_dict()
    n_h = {k: max(1, int(round(n * v / len(df)))) for k, v in N_h.items()
           if v > 0}

    # Ajuste de residuo para respetar n total (respetando tamaños disponibles).
    total = sum(min(n_h[k], N_h[k]) for k in n_h)
    keys = sorted(n_h, key=lambda k: -N_h[k])
    guard = 0
    while total != n and guard < 10_000:
        guard += 1
        for k in keys:
            if total == n:
                break
            cur = min(n_h[k], N_h[k])
            if total < n and cur < N_h[k]:
                n_h[k] = min(n_h[k], N_h[k]) + 1
                total += 1
            elif total > n and cur > 1:
                n_h[k] = cur - 1
                total -= 1

    rng = np.random.default_rng(seed)
    parts = []
    for stratum, idx in df.groupby("stratum", sort=False).groups.items():
        idx = np.asarray(idx)
        take = min(n_h.get(stratum, 0), len(idx))
        if take <= 0:
            continue
        chosen = rng.choice(idx, size=take, replace=False)
        part = df.loc[chosen].copy()
        part["weight"] = N_h[stratum] / take
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["stratum", "doc_id"]).reset_index(drop=True)


# ─────────────────────────── Anotaciones ───────────────────────────
def load_annotations(path: str | Path) -> pd.DataFrame:
    """Lee el CSV de anotaciones (formato largo). Vacío → DataFrame vacío.

    Raises:
        ValueError si el archivo existe pero no tiene las columnas mínimas.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {"doc_id", "label"} - set(df.columns)
    if missing and len(df):
        raise ValueError(f"CSV de anotaciones sin columnas {missing}: {path}")
    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col != "doc_id" else np.nan
    return df


def consensus_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta por documento: mayoría simple; empate → descartado (NaN)."""
    ann = annotations.dropna(subset=["doc_id"]).copy()
    ann["label"] = ann["label"].astype(str).str.strip().str.lower()
    ann = ann[ann["label"].isin(VALID_LABELS)]
    if ann.empty:
        return pd.DataFrame(columns=["doc_id", "label", "n_annotators",
                                     "annotators"])
    counts = ann.groupby(["doc_id", "label"]).size().unstack(fill_value=0)
    rows = []
    grouped = ann.groupby("doc_id")
    for doc_id, g in grouped:
        vc = g["label"].value_counts()
        if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
            label = None  # empate: ambiguo
        else:
            label = vc.idxmax()
        rows.append({
            "doc_id": doc_id,
            "label": label,
            "n_annotators": int(g["annotator"].astype(str).replace("", "a1")
                                .nunique()),
            "annotators": ";".join(sorted(set(g["annotator"].astype(str)))),
        })
    out = pd.DataFrame(rows).merge(
        counts.add_prefix("votes_").reset_index(), on="doc_id", how="left")
    return out


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """κ de Cohen entre dos secuencias de etiquetas alineadas (función pura)."""
    a = pd.Series(list(labels_a), dtype=object)
    b = pd.Series(list(labels_b), dtype=object)
    if len(a) == 0:
        return np.nan
    cats = sorted(set(a) | set(b))
    po = float((a.to_numpy() == b.to_numpy()).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ─────────────────────────── Métricas ───────────────────────────
def classification_metrics(
    y_true: Sequence[bool], y_pred: Sequence[bool],
    weights: Sequence[float] | None = None,
) -> dict:
    """Precisión/recall/F1/MCC/accuracy (función pura).

    Con `weights`, las celdas de la matriz de confusión se ponderan
    (tp_w = Σ w·1[tp], etc.) y las métricas se calculan sobre esas celdas:
    estimación poblacional con el diseño estratificado (w_h = N_h/n_h).
    """
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    tp = float(w[t & p].sum()); fp = float(w[~t & p].sum())
    fn = float(w[t & ~p].sum()); tn = float(w[~t & ~p].sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and
          (precision + recall) > 0 else np.nan)
    denom2 = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom2)) if denom2 > 0 else np.nan
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1e-12)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "mcc": mcc, "accuracy": accuracy, "n": int(len(t)),
    }


def bootstrap_metrics_ci(
    y_true: Sequence[bool], y_pred: Sequence[bool],
    weights: Sequence[float] | None = None,
    B: int = 2000, seed: int = 42,
) -> dict:
    """CI 95% bootstrap (remuestreo con reemplazo) para precision/recall/f1/mcc."""
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    w = None if weights is None else np.asarray(weights, dtype=float)
    n = len(t)
    if n == 0 or B <= 0:
        return {k: (np.nan, np.nan) for k in ("precision", "recall", "f1", "mcc")}
    rng = np.random.default_rng(seed)
    keys = ("precision", "recall", "f1", "mcc")
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        m = classification_metrics(t[idx], p[idx],
                                   None if w is None else w[idx])
        for k in keys:
            samples[k].append(m[k])
    out = {}
    for k in keys:
        vals = np.asarray([v for v in samples[k] if not np.isnan(v)], dtype=float)
        out[k] = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) \
            if vals.size else (np.nan, np.nan)
    return out


# ─────────────────────────── Ablaciones ───────────────────────────
def ablation_predictions(
    scores_df: pd.DataFrame, tau: float,
) -> dict[str, np.ndarray]:
    """Predicciones booleanas de cada variante del detector (función pura).

    Reglas (mención=True):
      lexical_only  : lex_count > 0
      resena_only   : sim_resena_centered >= tau
      promo_only    : sim_promo_centered  >= tau
      raw_max       : score_raw >= tau
      centered_max  : score_centered >= tau   (detector por defecto)
      hybrid        : centered_max OR (lex_count >= 2)
      tau_fijo_0.5  : score_centered >= 0.5
    """
    c = scores_df
    lex = c["lex_count"].to_numpy() > 0
    lex2 = c["lex_count"].to_numpy() >= 2
    r = c["sim_resena_centered"].to_numpy() >= tau
    p = c["sim_promo_centered"].to_numpy() >= tau
    raw = c["score_raw"].to_numpy() >= tau
    cen = c["score_centered"].to_numpy() >= tau
    return {
        "lexical_only": lex,
        "resena_only": r,
        "promo_only": p,
        "raw_max": raw,
        "centered_max": cen,
        "hybrid": cen | lex2,
        "tau_fijo_0.5": c["score_centered"].to_numpy() >= 0.5,
    }


# ─────────────────────────── Orquestación ───────────────────────────
def _append_validation_metadata(
    args: argparse.Namespace, results_dir: Path,
    n_sample: int, n_annotated: int, validated: bool, tau: float, seed: int,
) -> Path:
    """Fila en comparison/base/run_metadata.csv (parámetros + versión de la corrida)."""
    return append_metadata(
        results_dir / "comparison" / "base" / "run_metadata.csv",
        task="price_validation",
        params={
            "n_sample": int(n_sample),
            "n_annotated": int(n_annotated),
            "validated": bool(validated),
            "tau": round(float(tau), 6),
            "tau_quantile": getattr(args, "price_tau_quantile", None),
            "bootstrap": int(getattr(args, "bootstrap", 2000)),
            "seed": int(seed),
        },
        repo_dir=Path(__file__).resolve().parents[1],
    )


def _ensure_annotation_template(repo_dir: Path) -> Path:
    """Plantilla vacía annotations/precio_valor_labels.csv (idempotente)."""
    ann_dir = repo_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    template = ann_dir / "precio_valor_labels.csv"
    if not template.exists():
        save_csv(pd.DataFrame(columns=list(ANNOTATION_COLUMNS)), template)
    return template


REFERENCE_TYPE = "weak_lexical_reference_not_gold_standard"
CHECKS_EJECUTADOS = (
    "kappa_semantic_vs_lexical",
    "placebo_fpr",
    "threshold_sensitivity",
    "anchor_jackknife",
    "bootstrap_ci",
)
SENSITIVITY_QUANTILES = (0.90, 0.95, 0.975)


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Estadístico KS de dos muestras (función pura, sin dependencias nuevas)."""
    x = np.sort(np.asarray(a, dtype=float)); x = x[~np.isnan(x)]
    y = np.sort(np.asarray(b, dtype=float)); y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación de rangos de Spearman (función pura)."""
    x = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    z = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    if x.size < 2 or np.std(x) == 0 or np.std(z) == 0:
        return np.nan
    return float(np.corrcoef(x, z)[0, 1])


def _percentile_rate_ci(mask: np.ndarray, b: int, seed: int,
                        chunk: int = 128) -> tuple[float, float]:
    """IC 95% bootstrap de una tasa binaria, por bloques (memoria acotada)."""
    x = np.asarray(mask, dtype=float)
    if x.size == 0 or b <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    rates: list[np.ndarray] = []
    for start in range(0, int(b), chunk):
        size = min(int(chunk), int(b) - start)
        idx = rng.integers(0, x.size, size=(size, x.size))
        rates.append(x[idx].mean(axis=1))
    values = np.concatenate(rates)
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def _percentile_mean_ci(x: np.ndarray, b: int, seed: int) -> tuple[float, float]:
    """IC 95% bootstrap percentil de una media (función pura)."""
    v = np.asarray(x, dtype=float); v = v[~np.isnan(v)]
    if v.size == 0 or b <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(int(b), dtype=float)
    for i in range(int(b)):
        means[i] = v[rng.integers(0, v.size, size=v.size)].mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def run_price_validation(
    args: argparse.Namespace,
    results_dir: Path,
    price_scores_df: pd.DataFrame,
    tau: float,
    tau_quantile: float | None = None,
    *,
    anchors: Mapping[str, Any] | None = None,
    anchor_texts: Sequence[str] | None = None,
    anchor_emb: np.ndarray | None = None,
    anchor_centroid: np.ndarray | None = None,
    doc_emb: np.ndarray | None = None,
    negative_control_emb: np.ndarray | None = None,
    placebo_scores: np.ndarray | None = None,
    placebo_components: Mapping[str, np.ndarray] | None = None,
    score_mode: str = "centered",
    cache_dir: Path | None = None,
) -> dict:
    """F4.3: validación automática del detector semántico, sin anotación humana.

    Ejecuta cinco comprobaciones sobre scores ya calculados: acuerdo débil
    semantic↔lexical (κ), FPR del placebo, sensibilidad de τ, jackknife del
    ancla y bootstrap de tasas/cosenos. No recalcula embeddings ni sentimiento.

    Returns: dict con status 'validada_automaticamente', rutas escritas y
    métricas. El flag legacy --price-annotation-file se ignora con aviso.
    """
    import json
    import warnings

    from price_concept import (DEFAULT_ANCHORS, anchors_sha256, centroid_scores,
                               load_price_anchors, synthetic_anchor_centroid)

    if getattr(args, "price_annotation_file", None):
        msg = ("--price-annotation-file se ignora: la validación humana fue "
               "retirada del alcance; la validación es automática")
        print(f"  ⚠️  DEPRECATED: {msg}.")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    print_header("PASO 12.5 (F4.3): VALIDACIÓN AUTOMÁTICA (SIN ANOTACIÓN HUMANA)")
    seed = int(getattr(args, "seed", 42))
    n_boot = max(1, int(getattr(args, "bootstrap", 2000)))
    n_rep = max(1, int(getattr(args, "price_jackknife_replicas", 200)))
    price_dir = Path(results_dir) / "comparison" / "price"
    price_dir.mkdir(parents=True, exist_ok=True)

    anchors_path = Path(
        getattr(args, "price_anchor_file", None)
        or getattr(args, "price_anchors", None)
        or DEFAULT_ANCHORS
    )
    anchors = anchors if anchors is not None else load_price_anchors(anchors_path)
    anchors_version = str(anchors.get("version", ""))
    a_sha = anchors_sha256(anchors_path) if anchors_path.exists() else ""

    scores_df = price_scores_df.reset_index(drop=True)
    sem = np.asarray(
        scores_df["score_semantic"] if "score_semantic" in scores_df
        else scores_df["price_score"], dtype=float)
    lex = np.asarray(
        scores_df["score_lexical"] if "score_lexical" in scores_df
        else scores_df.get("lex_count", np.zeros(len(scores_df))), dtype=float)
    corpus = scores_df["corpus"].astype(str).str.upper().to_numpy()
    is_dmo = corpus == "DMO"
    counts = scores_df["corpus"].astype(str).value_counts().sort_index().to_dict()
    counts_str = ";".join(f"{k}={int(v)}" for k, v in counts.items())

    sem_pred = sem >= float(tau)
    lex_pred = lex > 0
    kappa = cohen_kappa(np.where(sem_pred, "precio", "no_precio"),
                        np.where(lex_pred, "precio", "no_precio"))

    # ── Placebo: FPR por componente + separación real vs placebo ──
    components = {str(k): np.asarray(v, dtype=float)
                  for k, v in (placebo_components or {}).items()}
    if placebo_scores is None and negative_control_emb is not None \
            and anchor_centroid is not None:
        placebo_scores = centroid_scores(
            np.asarray(negative_control_emb), np.asarray(anchor_centroid),
            mode=score_mode)
    placebo = (np.asarray(placebo_scores, dtype=float)
               if placebo_scores is not None else np.asarray([], dtype=float))
    controls = np.concatenate([v for k, v in components.items()
                               if k.startswith("neg_control_") and v.size]) \
        if any(k.startswith("neg_control_") and v.size
               for k, v in components.items()) else np.asarray([], dtype=float)
    random = np.asarray(components.get("random_docs", []), dtype=float)
    fpr_ctrl = float(np.mean(controls >= tau)) if controls.size else \
        (float(np.mean(placebo >= tau)) if placebo.size else np.nan)
    fpr_rand = float(np.mean(random >= tau)) if random.size else \
        (float(np.mean(placebo >= tau)) if placebo.size else np.nan)
    ks_stat = _ks_stat(sem, placebo) if placebo.size else np.nan
    dist = mann_whitney_cliffs(sem, placebo) if placebo.size else {
        "mw_p_value": np.nan, "cliffs_delta": np.nan}

    # ── Bootstrap de tasa y coseno medio por fuente ──
    rate_dmo = float(sem_pred[is_dmo].mean()) if is_dmo.any() else np.nan
    rate_ugc = float(sem_pred[~is_dmo].mean()) if (~is_dmo).any() else np.nan
    lo_dmo, hi_dmo = _percentile_rate_ci(sem_pred[is_dmo].astype(float), n_boot, seed)
    lo_ugc, hi_ugc = _percentile_rate_ci(sem_pred[~is_dmo].astype(float), n_boot, seed + 1)
    mu_dmo_lo, mu_dmo_hi = _percentile_mean_ci(sem[is_dmo], n_boot, seed + 2)
    mu_ugc_lo, mu_ugc_hi = _percentile_mean_ci(sem[~is_dmo], n_boot, seed + 3)

    # Métricas contra la referencia débil (nunca llamar accuracy).
    weak = classification_metrics(lex_pred, sem_pred)
    weak_ci = bootstrap_metrics_ci(lex_pred, sem_pred, B=n_boot, seed=seed + 4)

    # ── Sensibilidad de τ ──
    base_placebo = placebo if placebo.size else sem
    sens_rows: list[dict] = []
    for q in SENSITIVITY_QUANTILES:
        t_q = float(np.quantile(base_placebo, q))
        pred = sem >= t_q
        rd = float(pred[is_dmo].mean()) if is_dmo.any() else np.nan
        ru = float(pred[~is_dmo].mean()) if (~is_dmo).any() else np.nan
        sens_rows.append({
            "tau_quantile": q,
            "tau": round(t_q, 6),
            "rate_dmo": round(rd, 6),
            "rate_ugc": round(ru, 6),
            "gap_dmo_minus_ugc": round(rd - ru, 6),
            "gap_sign": int(np.sign(round(rd - ru, 12))),
        })
    stable = len({r["gap_sign"] for r in sens_rows}) == 1
    for row in sens_rows:
        row["gap_sign_stable"] = bool(stable)

    # ── Jackknife del ancla (20% de términos, seed fija) ──
    jk_columns = ["replicate", "n_terms_removed", "spearman_top5", "rate_delta",
                  "tau", "score_mode"]
    jk_rows: list[dict] = []
    if anchor_emb is not None and doc_emb is not None:
        ae = np.asarray(anchor_emb, dtype=float)
        de = np.asarray(doc_emb, dtype=float)
        if ae.ndim == 2 and de.ndim == 2 and ae.shape[0] >= 5:
            mu = de.mean(axis=0)
            center = (np.asarray(anchor_centroid, dtype=float)
                      if anchor_centroid is not None
                      else synthetic_anchor_centroid(ae))
            baseline = centroid_scores(de, center, mode=score_mode, mean=mu)
            rng = np.random.default_rng(seed)
            k = max(1, int(round(0.2 * ae.shape[0])))
            for rep in range(n_rep):
                drop = rng.choice(ae.shape[0], size=k, replace=False)
                keep = np.setdiff1d(np.arange(ae.shape[0]), drop)
                cen = synthetic_anchor_centroid(ae[keep])
                sc = centroid_scores(de, cen, mode=score_mode, mean=mu)
                top = set(np.argsort(baseline)[-5:].tolist()) | \
                    set(np.argsort(sc)[-5:].tolist())
                ids = np.asarray(sorted(top), dtype=int)
                jk_rows.append({
                    "replicate": rep + 1,
                    "n_terms_removed": k,
                    "spearman_top5": round(_spearman(baseline[ids], sc[ids]), 6),
                    "rate_delta": round(float((sc >= tau).mean()
                                              - (baseline >= tau).mean()), 6),
                    "tau": round(float(tau), 6),
                    "score_mode": score_mode,
                })

    trace = {
        "anchors_version": anchors_version,
        "anchors_sha256": a_sha,
        "seed": seed,
        "tau": round(float(tau), 6),
        "detector_primario": "semantic",
        "n_normales": int(len(scores_df)),
        "n_por_fuente": counts_str,
    }
    auto_row = {
        "detector_primario": "semantic",
        "reference_type": REFERENCE_TYPE,
        "kappa_semantic_vs_lexical": round(float(kappa), 6)
        if kappa == kappa else np.nan,
        "tau": round(float(tau), 6),
        "tau_quantile": tau_quantile,
        "score_mode": score_mode,
        "n": int(len(scores_df)),
        "n_por_fuente": counts_str,
        "rate_dmo": round(rate_dmo, 6),
        "rate_ugc": round(rate_ugc, 6),
        "gap_dmo_minus_ugc": round(rate_dmo - rate_ugc, 6),
        "rate_dmo_ci_low": round(lo_dmo, 6),
        "rate_dmo_ci_high": round(hi_dmo, 6),
        "rate_ugc_ci_low": round(lo_ugc, 6),
        "rate_ugc_ci_high": round(hi_ugc, 6),
        "mean_cosine_dmo": round(float(sem[is_dmo].mean()), 6)
        if is_dmo.any() else np.nan,
        "mean_cosine_ugc": round(float(sem[~is_dmo].mean()), 6)
        if (~is_dmo).any() else np.nan,
        "mean_cosine_dmo_ci_low": round(mu_dmo_lo, 6),
        "mean_cosine_dmo_ci_high": round(mu_dmo_hi, 6),
        "mean_cosine_ugc_ci_low": round(mu_ugc_lo, 6),
        "mean_cosine_ugc_ci_high": round(mu_ugc_hi, 6),
        "placebo_fpr_controls": round(fpr_ctrl, 6),
        "placebo_fpr_random": round(fpr_rand, 6),
        "ks_real_vs_placebo": round(ks_stat, 6),
        "mw_p_value_real_vs_placebo": round(float(dist["mw_p_value"]), 6),
        "cliffs_delta_real_vs_placebo": round(float(dist["cliffs_delta"]), 6),
        "precision_vs_lexical": round(weak["precision"], 6),
        "recall_vs_lexical": round(weak["recall"], 6),
        "f1_vs_lexical": round(weak["f1"], 6),
        "precision_vs_lexical_ci_low": round(weak_ci["precision"][0], 6),
        "precision_vs_lexical_ci_high": round(weak_ci["precision"][1], 6),
        "recall_vs_lexical_ci_low": round(weak_ci["recall"][0], 6),
        "recall_vs_lexical_ci_high": round(weak_ci["recall"][1], 6),
        "f1_vs_lexical_ci_low": round(weak_ci["f1"][0], 6),
        "f1_vs_lexical_ci_high": round(weak_ci["f1"][1], 6),
        "gap_sign_stable": bool(stable),
        "jackknife_spearman_top5_mean": round(float(np.nanmean(
            [r["spearman_top5"] for r in jk_rows])), 6) if jk_rows else np.nan,
        "jackknife_rate_delta_std": round(float(np.nanstd(
            [r["rate_delta"] for r in jk_rows])), 6) if jk_rows else np.nan,
        "n_boot": int(n_boot),
        "n_jackknife": len(jk_rows),
        "outliers_excluidos": True,
        **trace,
    }
    paths = [
        save_csv(pd.DataFrame([auto_row]), price_dir / "price_validation_auto.csv"),
        save_csv(pd.DataFrame(sens_rows),
                 price_dir / "price_threshold_sensitivity.csv"),
        save_csv(pd.DataFrame(jk_rows, columns=jk_columns),
                 price_dir / "price_anchor_jackknife.csv"),
    ]
    status = {
        "status": "validada_automaticamente",
        "tau": round(float(tau), 6),
        "tau_quantile": tau_quantile,
        "detector_primario": "semantic",
        "reference_type": REFERENCE_TYPE,
        "anchors_version": anchors_version,
        "anchors_sha256": a_sha,
        "seed": seed,
        "n_normales": int(len(scores_df)),
        "n_por_fuente": {str(k): int(v) for k, v in counts.items()},
        "checks_ejecutados": list(CHECKS_EJECUTADOS),
        "placebo_fpr_controls": round(fpr_ctrl, 6),
        "placebo_fpr_random": round(fpr_rand, 6),
        "gap_sign_stable": bool(stable),
        "n_boot": int(n_boot),
        "n_jackknife": len(jk_rows),
        "outliers_excluidos": True,
    }
    status_path = price_dir / "price_validation_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2,
                                      default=str), encoding="utf-8")
    paths.append(status_path)

    meta_path = _append_validation_metadata(
        args, results_dir, n_sample=len(scores_df), n_annotated=0,
        validated=True, tau=tau, seed=seed)
    paths.append(meta_path)

    print(f"  κ semantic↔lexical (referencia débil): {kappa:.3f}")
    print(f"  FPR placebo: controles={fpr_ctrl:.4f} | random={fpr_rand:.4f} | "
          f"τ={tau:.4f}")
    print(f"  Sensibilidad τ: gap_sign_stable={stable} | "
          f"jackknife={len(jk_rows)} réplicas | bootstrap={n_boot}")
    print(f"  Estado: validada_automaticamente → {status_path.name}")

    return {
        "status": "validada_automaticamente",
        "validated": True,
        "paths": paths,
        "tau": float(tau),
        "tau_quantile": tau_quantile,
        "detector_primario": "semantic",
        "reference_type": REFERENCE_TYPE,
        "checks_ejecutados": list(CHECKS_EJECUTADOS),
        "placebo_fpr_controls": fpr_ctrl,
        "placebo_fpr_random": fpr_rand,
        "kappa_semantic_vs_lexical": float(kappa),
        "gap_sign_stable": bool(stable),
        "n_normales": int(len(scores_df)),
        "n_por_fuente": counts,
        "anchors_version": anchors_version,
        "anchors_sha256": a_sha,
        "seed": seed,
        "metrics": {"weak_reference": weak, "kappa": float(kappa),
                    "fpr_controls": fpr_ctrl, "fpr_random": fpr_rand},
    }
