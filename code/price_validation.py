"""
price_validation.py — F4.3: validación humana del detector Precio/Valor/Costo.

Flujo
-----
1. `make_stratified_sample`: muestreo estratificado (5 bandas de score × corpus)
   de `price_scores.csv`, con pesos w_h = N_h/n_h para estimaciones poblacionales.
2. Plantilla de anotación: annotations/precio_valor_labels.csv
   (doc_id, annotator, label, note). Guía: docs/ANNOTATION_GUIDE.md.
3. Si el archivo de anotaciones está vacío o no existe → se escribe la muestra,
   se imprime "validación pendiente de anotación" y NO se falla.
4. Si hay etiquetas: precisión/recall/F1/MCC (muestrales y ponderados por w_h),
   matriz de confusión, κ de Cohen si hay ≥2 anotadores, CI bootstrap y
   ablaciones (lexical-only, resena-only, promo-only, raw-max, centered-max,
   hybrid, tau_fijo_0.5).

Etiquetas válidas: "precio" (mención), "no_precio" (no mención). Otras
("ambiguo", errores de dedo) se descartan del conteo y se reportan.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from comparison.common import print_header, save_csv

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
def _ensure_annotation_template(repo_dir: Path) -> Path:
    """Plantilla vacía annotations/precio_valor_labels.csv (idempotente)."""
    ann_dir = repo_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    template = ann_dir / "precio_valor_labels.csv"
    if not template.exists():
        save_csv(pd.DataFrame(columns=list(ANNOTATION_COLUMNS)), template)
    return template


def run_price_validation(
    args: argparse.Namespace,
    results_dir: Path,
    price_scores_df: pd.DataFrame,
    tau: float,
    tau_quantile: float | None = None,
) -> dict:
    """Tarea F4.3: muestreo + (si hay etiquetas) métricas y ablaciones.

    Nunca lanza error por falta de anotaciones: degrada elegantemente.
    Returns: dict con keys 'sample_path', 'template_path', 'validated', 'paths'.
    """
    print_header("PASO 12.5 (F4.3): VALIDACIÓN DEL DETECTOR PRECIO/VALOR")
    price_dir = results_dir / "comparison" / "price"
    price_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(__file__).resolve().parents[1]
    template_path = _ensure_annotation_template(repo_dir)
    annotations_path = template_path

    n_sample = int(getattr(args, "price_validation_n", 100))
    seed = int(getattr(args, "seed", 42))
    sample = make_stratified_sample(price_scores_df, n=n_sample, seed=seed)
    sample_out = sample[[
        "doc_id", "corpus", "source", "text", "price_score",
        "sentiment_label", "stratum", "weight",
    ]].copy()
    sample_out["annotator"] = ""
    sample_out["label"] = ""
    sample_out["note"] = ""
    sample_path = save_csv(sample_out, price_dir / "annotation_sample.csv")
    print(f"  Muestra de anotación: {len(sample_out)} docs "
          f"({sample_out['stratum'].nunique()} estratos) → {sample_path.name}")
    print(f"  Plantilla de etiquetas: {template_path}")

    annotations = load_annotations(annotations_path)
    ann_valid = annotations.dropna(subset=["doc_id"])
    ann_valid = ann_valid[ann_valid["label"].astype(str).str.strip().ne("")]
    paths: list[Path] = [sample_path, template_path]
    result = {"sample_path": sample_path, "template_path": template_path,
              "validated": False, "paths": paths}

    if ann_valid.empty:
        print("  ⏳ Validación pendiente de anotación: rellena "
              f"{template_path} (docs/ANNOTATION_GUIDE.md) y reejecuta.")
        return result

    # ── Hay etiquetas: métricas del detector por defecto ──
    consensus = consensus_labels(annotations).dropna(subset=["label"])
    if consensus.empty:
        print("  ⚠️ Anotaciones sin consenso (todos empates); se omite la métrica.")
        return result
    w_map = dict(zip(sample["doc_id"], sample["weight"])) \
        if "weight" in sample.columns else {}
    merged = consensus.merge(
        price_scores_df, on="doc_id", how="inner", suffixes=("", "_score"))
    if merged.empty:
        print("  ⚠️ Ningún doc_id anotado coincide con price_scores.csv.")
        return result
    merged["weight"] = merged["doc_id"].map(w_map).fillna(1.0)

    y_true = (merged["label"] == LABEL_POS).to_numpy()
    y_pred = (merged["price_score"] >= tau).to_numpy()
    w = merged["weight"].to_numpy()
    m_sampled = classification_metrics(y_true, y_pred)
    m_weighted = classification_metrics(y_true, y_pred, weights=w)
    ci = bootstrap_metrics_ci(
        y_true, y_pred, weights=w,
        B=int(getattr(args, "bootstrap", 2000)), seed=seed)

    # ── κ de Cohen (primer par de anotadores con solape) ──
    kappa = np.nan
    ann_long = ann_valid.copy()
    ann_long["annotator"] = ann_long["annotator"].astype(str).replace("", "a1")
    docs_per_annotator = ann_long.groupby("annotator")["doc_id"].apply(set)
    annotators = sorted(docs_per_annotator.index)
    if len(annotators) >= 2:
        common = docs_per_annotator[annotators[0]] & docs_per_annotator[annotators[1]]
        if common:
            a = ann_long[(ann_long["annotator"] == annotators[0]) &
                         (ann_long["doc_id"].isin(common))] \
                .set_index("doc_id")["label"]
            b = ann_long[(ann_long["annotator"] == annotators[1]) &
                         (ann_long["doc_id"].isin(common))] \
                .set_index("doc_id")["label"]
            kappa = cohen_kappa(a.loc[sorted(common)], b.loc[sorted(common)])

    row = {
        "detector": "centered_max", "tau": tau,
        "tau_quantile": tau_quantile, "n_annotated": int(len(merged)),
        "n_annotators": int(ann_long["annotator"].nunique()),
        "cohen_kappa": round(kappa, 4) if kappa == kappa else np.nan,
        "weighted": False, **{k: round(v, 4) if isinstance(v, float) else v
                               for k, v in m_sampled.items()},
    }
    row_w = {"detector": "centered_max", "weighted": True,
             **{k: round(v, 4) if isinstance(v, float) else v
                for k, v in m_weighted.items()},
             **{f"{k}_ci_low": round(v[0], 4) if v[0] == v[0] else np.nan
                for k, v in ci.items()},
             **{f"{k}_ci_high": round(v[1], 4) if v[1] == v[1] else np.nan
                for k, v in ci.items()}}
    paths.append(save_csv(pd.DataFrame([row, row_w]),
                          price_dir / "price_validation.csv"))

    # ── Confusión (ponderada, escala original de conteos ponderados) ──
    cm = pd.DataFrame({
        "pred_no_precio": [m_weighted["tn"], m_weighted["fn"]],
        "pred_precio": [m_weighted["fp"], m_weighted["tp"]],
    }, index=["true_no_precio", "true_precio"])
    cm.index.name = "true\\pred"
    paths.append(save_csv(cm.reset_index(), price_dir / "confusion_matrix.csv"))

    # ── Ablaciones ──
    preds = ablation_predictions(merged, tau)
    abl_rows = []
    for name, pred in preds.items():
        ms = classification_metrics(y_true, pred)
        mw = classification_metrics(y_true, pred, weights=w)
        abl_rows.append({"ablation": name, **{
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in ms.items()}})
        abl_rows.append({"ablation": name, "weighted": True, **{
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in mw.items()}})
    paths.append(save_csv(pd.DataFrame(abl_rows),
                          price_dir / "price_ablations.csv"))

    print(f"  ✅ Validación: n={len(merged)} | κ={kappa:.3f} | "
          f"F1 muestra={m_sampled['f1']:.3f} | F1 ponderado={m_weighted['f1']:.3f} | "
          f"MCC={m_weighted['mcc']:.3f}")
    result["validated"] = True
    result["metrics"] = {"sampled": m_sampled, "weighted": m_weighted, "kappa": kappa}
    return result
