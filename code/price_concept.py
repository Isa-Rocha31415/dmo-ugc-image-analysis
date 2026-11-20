"""
price_concept.py — F4.2: núcleo de scoring semántico del concepto Precio/Valor/Costo.

Diseño
------
- Los anchors (prototipos, ancla léxica, controles negativos) son un recurso
  ESTÁTICO versionado: code/resources/anchors/precio_valor.json (docs/ANCHORS.md).
  El pipeline solo lee el JSON; nunca lo regenera ni llama APIs externas.
- Embeddings de prototipos/controles: sentence-transformers local ya cacheado
  (mismo modelo que el PASO 2); se fuerza modo offline (HF_HUB_OFFLINE).
- Score por documento:
    sim_resena  = max coseno contra prototipos de registro reseña (es+en)
    sim_promo   = max coseno contra prototipos de registro promocional (es+en)
    score       = max(sim_resena, sim_promo)
  Dos modos: "raw" (espacio original) y "centered" (centrado por media del
  corpus y renormalizado; default) → se guardan ambos para ablación.
- Calibración de τ por placebo: τ = cuantil q de los scores placebo
  (controles negativos + N documentos aleatorios del corpus, seed fija).
  No hay etiquetas humanas ni leakage del corpus en los anchors.

Outputs (results/comparison/price/): price_scores.csv, tau_calibration.csv,
tau_sensitivity.csv, panel F4.4 (mention_rate_by_source, score_distribution_tests,
valence_conditioned, price_context_logodds, price_top_examples, scatters, summary),
F4.4b (price_comparison, price_context_terms, price_kpis, price_examples_topk),
F4.3 (annotation_sample + price_validation si hay etiquetas) + caché propia
(<cache-dir>/price/manifest.json, idempotente) + filas en run_metadata.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from comparison.common import (append_metadata, bh_adjust, bootstrap_rate_ci,
                               l2_normalize, mann_whitney_cliffs, max_cosine,
                               palette_colors, print_header, save_csv)
from io_corpus import CORPUS_COLUMN, SOURCE_COLUMN
from metrics_lexical import logodds_z
from metrics_thematic import cohens_h, two_by_two_test

_l2 = l2_normalize  # compatibilidad con el nombre privado histórico
_max_cosine = max_cosine

from utils import palette_to_plot_colors

REQUIRED_PROTOTYPE_GROUPS = ("resena_es", "promo_es", "resena_en", "promo_en")
REQUIRED_NEGATIVE_CONTROLS = ("seguridad", "clima", "transporte", "gastronomia")
REQUIRED_LEXICAL_KEYS = ("terms_es", "terms_en", "patterns")
SCORE_MODES = ("centered", "raw")
DETECTOR_MODES = ("semantic", "lexical", "hybrid")
SENSITIVITY_LEVELS = (0.50, 0.90, 0.95, 0.99)


def detector_mention(
    score: Sequence[float] | np.ndarray,
    lex_count: Sequence[int] | np.ndarray,
    tau: float,
    detector: str = "semantic",
) -> np.ndarray:
    """Máscara booleana de mención de precio según la regla del detector.

    - semantic: score >= τ
    - lexical : lex_count > 0
    - hybrid  : score >= τ O lex_count >= 2 (misma regla que F4.3)
    """
    sc = np.asarray(score, dtype=float)
    lx = np.asarray(lex_count, dtype=int)
    if detector == "lexical":
        return lx > 0
    if detector == "hybrid":
        return (sc >= tau) | (lx >= 2)
    if detector != "semantic":
        raise ValueError(f"detector desconocido: {detector!r}")
    return sc >= tau

DEFAULT_ANCHORS = str(Path(__file__).resolve().parent / "resources" / "anchors" / "precio_valor.json")
PRICE_OUTPUT_SUBDIR = "price"

# ─────────────────────────── Anchors (esquema + carga) ───────────────────────────
def load_price_anchors(path: str | Path) -> dict:
    """Lee y valida el esquema mínimo del JSON de anchors.

    Returns:
        dict con claves concept_id, version, provenance, prototypes,
        negative_controls y lexical_anchor.
    Raises:
        FileNotFoundError / ValueError si falta el archivo o el esquema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Anchors no encontrados: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("concept_id") != "precio_valor":
        raise ValueError(f"concept_id inesperado: {data.get('concept_id')!r}")
    if not data.get("version"):
        raise ValueError("El JSON de anchors no declara 'version'.")
    provenance = data.get("provenance") or {}
    if not provenance.get("prompt_sha256"):
        raise ValueError("provenance.prompt_sha256 ausente (trazabilidad obligatoria).")
    for group in REQUIRED_PROTOTYPE_GROUPS:
        texts = (data.get("prototypes") or {}).get(group) or []
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"prototypes.{group} vacío o ausente.")
    for ctrl in REQUIRED_NEGATIVE_CONTROLS:
        texts = (data.get("negative_controls") or {}).get(ctrl) or []
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"negative_controls.{ctrl} vacío o ausente.")
    lex = data.get("lexical_anchor") or {}
    for key in REQUIRED_LEXICAL_KEYS:
        if not (lex.get(key) or []):
            raise ValueError(f"lexical_anchor.{key} vacío o ausente.")
    return data


def prototype_registry(anchors: Mapping[str, Any]) -> list[dict]:
    """Aplana los prototipos a [(register, group, text), ...] (función pura).

    register: 'resena' | 'promo' (derivado del nombre del grupo).
    """
    registry: list[dict] = []
    for group, texts in anchors["prototypes"].items():
        register = "resena" if group.startswith("resena") else "promo"
        for text in texts:
            registry.append({"register": register, "group": group, "text": str(text)})
    return registry


# ─────────────────────────── Ancla léxica (pura) ───────────────────────────
def _fold(text: str) -> str:
    """Minúsculas sin acentos (compara 'costó' con 'costo')."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", str(text).lower())
        if not unicodedata.combining(ch)
    )


def lexical_hits(texts: Sequence[str], lexical_anchor: Mapping[str, Any]) -> list[dict]:
    """Hits de la ancla léxica por documento (función pura, sin red).

    Cuenta TÉRMINOS distintos (terms_es/terms_en, \b delimitados) y coincidencias
    de `patterns` sobre el texto plegado (sin acentos, minúsculas).
    Returns:
        lista alineada con `texts`: {lex_count, terms, patterns_matched}.
    """
    terms = [_fold(t) for t in list(lexical_anchor.get("terms_es", []))
             + list(lexical_anchor.get("terms_en", []))]
    patterns = [re.compile(p) for p in lexical_anchor.get("patterns", [])]
    out: list[dict] = []
    for text in texts:
        folded = _fold(text)
        matched: list[str] = []
        for term in set(terms):
            if re.search(rf"\b{re.escape(term)}\b", folded):
                matched.append(term)
        patterns_matched = [p.pattern for p in patterns if p.search(folded)]
        out.append({
            "lex_count": len(matched) + len(patterns_matched),
            "terms": ";".join(sorted(set(matched))),
            "patterns_matched": ";".join(patterns_matched),
        })
    return out


# ─────────────────────────── Embeddings de anchors ───────────────────────────
def _anchor_cache_key(model_name: str, texts: Sequence[str], anchors: Mapping[str, Any]) -> str:
    """Clave determinista: modelo + sha256(textos) + versión del anchor."""
    joined = "\n".join(str(t) for t in texts)
    payload = f"{model_name}|{hashlib.sha256(joined.encode('utf-8')).hexdigest()}|{anchors.get('version')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def embed_anchor_texts(
    texts: Sequence[str],
    model_name: str,
    cache_dir: str | Path | None = None,
    anchors: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Embedea textos de anchors con el modelo local ya cacheado (offline).

    Prohibido descargar: se fuerza HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE antes de
    cargar sentence-transformers. Opcionalmente cachea el resultado en disco
    (cache_dir/price_anchor_embeds/<key>.npy) para no re-embedear cada corrida.
    """
    texts = [str(t) for t in texts]
    cache_file: Path | None = None
    if cache_dir is not None and anchors is not None:
        cache_file = Path(cache_dir) / "price_anchor_embeds" / (
            f"{_anchor_cache_key(model_name, texts, anchors)}.npy"
        )
        if cache_file.exists():
            return np.load(cache_file)

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4)
        model = SentenceTransformer(model_name)
        emb = np.asarray(
            model.encode(texts, batch_size=128, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float32,
        )
    except ImportError:
        # Fallback al mismo path del PASO 2 (transformers puro, modelo local).
        from embeddings_comparison import obtener_embeddings_bert
        emb = obtener_embeddings_bert(texts, model_name=model_name, batch_size=64)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, emb)
    return emb


# ─────────────────────────── Scoring (funciones puras) ───────────────────────────
def raw_cosine_scores(doc_emb: np.ndarray, anchor_emb: np.ndarray) -> np.ndarray:
    """Coseno directo doc↔anclas; máximo sobre anclas. docs (N,D), anclas (M,D)."""
    doc_emb = np.asarray(doc_emb, dtype=float)
    anchor_emb = np.asarray(anchor_emb, dtype=float)
    return _max_cosine(doc_emb, anchor_emb)


def centered_cosine_scores(
    doc_emb: np.ndarray,
    anchor_emb: np.ndarray,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """Coseno tras centrar por la media del corpus y renormalizar.

    Restar la media del corpus quita el 'centro de masa' temático (por ejemplo,
    el sesgo turístico común a todos los documentos), de modo que el score
    mide afinidad RELATIVA al corpus y no abundancia de vocabulario general.
    La media puede inyectarse (p. ej. la del corpus completo) para aplicar el
    mismo centrado a otros conjuntos; por defecto se estima de `doc_emb`.
    """
    doc_emb = np.asarray(doc_emb, dtype=float)
    anchor_emb = np.asarray(anchor_emb, dtype=float)
    mu = np.asarray(doc_emb.mean(axis=0)) if mean is None else np.asarray(mean, dtype=float)
    return _max_cosine(doc_emb - mu, anchor_emb - mu)


def compute_price_scores(
    doc_emb: np.ndarray,
    anchor_emb_by_register: Mapping[str, np.ndarray],
    mode: str = "centered",
) -> dict[str, np.ndarray]:
    """sim_resena / sim_promo / score para un modo dado (función pura).

    Args:
        anchor_emb_by_register: {'resena': (M1,D), 'promo': (M2,D)}.
        mode: 'centered' (default) o 'raw'.
    """
    if mode not in SCORE_MODES:
        raise ValueError(f"mode debe ser uno de {SCORE_MODES}: {mode!r}")
    scorer = centered_cosine_scores if mode == "centered" else raw_cosine_scores
    sim_resena = scorer(doc_emb, anchor_emb_by_register["resena"])
    sim_promo = scorer(doc_emb, anchor_emb_by_register["promo"])
    return {
        "sim_resena": sim_resena,
        "sim_promo": sim_promo,
        "score": np.fmax(sim_resena, sim_promo),
    }


# ─────────────────────────── Calibración τ ───────────────────────────
def calibrate_tau(placebo_scores: np.ndarray, quantile: float) -> float:
    """τ = cuantil `quantile` de los scores placebo (función pura)."""
    scores = np.asarray(placebo_scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        raise ValueError("Placebo vacío: no se puede calibrar tau.")
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile fuera de (0,1]: {quantile}")
    return float(np.quantile(scores, quantile))


# ─────────────────────────── Orquestación ───────────────────────────
def _load_base_artifacts(results_dir: Path) -> tuple[pd.DataFrame, np.ndarray, Path]:
    """Lee analysis_base.csv + embeddings_clean.npy del PASO 8 (base)."""
    base_dir = results_dir / "comparison" / "base"
    base_csv = base_dir / "analysis_base.csv"
    emb_npy = base_dir / "embeddings_clean.npy"
    if not base_csv.exists() or not emb_npy.exists():
        raise FileNotFoundError(
            f"Faltan artefactos base ({base_csv} / {emb_npy}). "
            "Ejecuta primero la tarea 'base' del PASO 8."
        )
    base = pd.read_csv(base_csv, encoding="utf-8-sig")
    emb = np.load(emb_npy)
    if len(base) != emb.shape[0]:
        raise ValueError(
            f"analysis_base.csv ({len(base)}) no alinea con embeddings_clean.npy "
            f"({emb.shape[0]}). Regenera la tarea 'base'."
        )
    return base, emb, base_dir


def _sensitivity_table(
    placebo_scores: np.ndarray, scores_by_mode: dict[str, np.ndarray],
    base: pd.DataFrame,
) -> pd.DataFrame:
    """τ a varios cuantiles × modos + cobertura por corpus (para tau_sensitivity.csv)."""
    rows: list[dict] = []
    is_dmo = (base[CORPUS_COLUMN].astype(str).str.upper() == "DMO").to_numpy()
    for mode, scores in scores_by_mode.items():
        for level in SENSITIVITY_LEVELS:
            tau = calibrate_tau(placebo_scores, level)
            above = np.asarray(scores) >= tau
            n_above = int(above.sum())
            rows.append({
                "mode": mode,
                "level": level,
                "tau": round(tau, 6),
                "n_docs": int(len(scores)),
                "n_docs_above": n_above,
                "share_docs_above": round(n_above / max(1, len(scores)), 6),
                "share_dmo_above": round(float(above[is_dmo].mean()) if is_dmo.any() else np.nan, 6),
                "share_ugc_above": round(float(above[~is_dmo].mean()) if (~is_dmo).any() else np.nan, 6),
            })
    return pd.DataFrame(rows)


def run_price_core(args: argparse.Namespace, results_dir: Path) -> list[Path]:
    """Tarea 'price_core' (F4.2): scores + calibración τ. Sin figuras.

    Args requeridos vía getattr con defaults: price_anchor_file (o legacy
    price_anchors), price_tau_quantile, price_placebo_docs, price_mode
    (semantic|lexical|hybrid|centered|raw), price_center, embedding_model,
    seed, no_cache, force_steps.
    """
    print_header("PASO 12 (F4.2): SCORE SEMÁNTICO PRECIO/VALOR + CALIBRACIÓN τ")
    out_dir = results_dir / "comparison" / PRICE_OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors_path = Path(
        getattr(args, "price_anchor_file", None)
        or getattr(args, "price_anchors", None)
        or DEFAULT_ANCHORS
    )
    anchors = load_price_anchors(anchors_path)
    tau_quantile = float(getattr(args, "price_tau_quantile", 0.95))
    placebo_docs_n = int(getattr(args, "price_placebo_docs", 300))
    model_name = str(getattr(args, "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"))
    seed = int(getattr(args, "seed", 42))
    cache_dir = Path(getattr(args, "cache_dir", None) or (results_dir / "cache"))

    base, emb, _ = _load_base_artifacts(results_dir)
    print(f"  Anchors: {anchors_path} (v{anchors['version']})")
    print(f"  Corpus base: {len(base):,} docs | embeddings {emb.shape}")

    # ── Embeddings de prototipos y controles (recurso estático, cacheado) ──
    registry = prototype_registry(anchors)
    anchor_emb = embed_anchor_texts(
        [r["text"] for r in registry], model_name, cache_dir=cache_dir, anchors=anchors
    )
    emb_by_register: dict[str, list[np.ndarray]] = {"resena": [], "promo": []}
    for row, vec in zip(registry, anchor_emb):
        emb_by_register[row["register"]].append(vec)
    anchor_by_register = {k: np.asarray(v) for k, v in emb_by_register.items()}

    neg_texts = [str(t) for texts in anchors["negative_controls"].values() for t in texts]
    neg_emb = embed_anchor_texts(neg_texts, model_name, cache_dir=cache_dir, anchors=anchors)

    # ── Scores por documento (ambos modos → ablación) ──
    scores_by_mode = {
        mode: compute_price_scores(emb, anchor_by_register, mode=mode)
        for mode in SCORE_MODES
    }
    lexical = lexical_hits(base["text"].astype(str).tolist(), anchors["lexical_anchor"])

    # Detector (F4.5 CLI): semantic|lexical|hybrid + score centrado o raw.
    # Acepta además el valor legacy "raw"/"centered" en price_mode.
    raw_mode = str(getattr(args, "price_mode", "semantic"))
    if raw_mode in SCORE_MODES:
        detector, score_mode = "semantic", raw_mode
    elif raw_mode in DETECTOR_MODES:
        detector = raw_mode
        score_mode = "centered" if getattr(args, "price_center", True) else "raw"
    else:
        raise ValueError(
            f"--price-mode debe ser uno de {DETECTOR_MODES + SCORE_MODES}: {raw_mode!r}")
    default_mode = score_mode

    # ── Placebo: controles negativos + N docs aleatorios del corpus ──
    rng = np.random.default_rng(seed)
    rand_idx = np.sort(rng.choice(len(base), size=min(placebo_docs_n, len(base)),
                                  replace=False))
    placebo_by_component: dict[str, np.ndarray] = {}
    for i, name in enumerate(anchors["negative_controls"]):
        rows = [t for t in anchors["negative_controls"][name]]
        vecs = neg_emb[[neg_texts.index(t) for t in rows]]
        neg_scores = compute_price_scores(vecs, anchor_by_register, mode=default_mode)["score"]
        placebo_by_component[f"neg_control_{name}"] = neg_scores
    rand_scores = compute_price_scores(emb[rand_idx], anchor_by_register,
                                       mode=default_mode)["score"]
    placebo_by_component["random_docs"] = rand_scores
    placebo_scores = np.concatenate([v for v in placebo_by_component.values() if len(v)])

    tau = calibrate_tau(placebo_scores, tau_quantile)
    print(f"  Placebo: {len(placebo_scores)} scores | τ(q={tau_quantile}) = {tau:.4f}")

    # ── price_scores.csv ──
    scores = scores_by_mode[default_mode]
    out = pd.DataFrame({
        "doc_id": base["doc_id"].to_numpy(),
        "text": base["text"].astype(str).to_numpy(),
        "text_norm": (base["text_norm"].astype(str).to_numpy()
                      if "text_norm" in base.columns else base["text"].astype(str).to_numpy()),
        CORPUS_COLUMN: base[CORPUS_COLUMN].to_numpy(),
        SOURCE_COLUMN: base[SOURCE_COLUMN].to_numpy(),
        "sentiment_label": base["sentiment_label"].to_numpy(),
        "sentiment_score": base["sentiment_score"].astype(float).to_numpy(),
        "lex_count": [h["lex_count"] for h in lexical],
        "lex_terms": [h["terms"] for h in lexical],
        "lex_patterns": [h["patterns_matched"] for h in lexical],
        "sim_resena_raw": scores_by_mode["raw"]["sim_resena"],
        "sim_promo_raw": scores_by_mode["raw"]["sim_promo"],
        "score_raw": scores_by_mode["raw"]["score"],
        "sim_resena_centered": scores_by_mode["centered"]["sim_resena"],
        "sim_promo_centered": scores_by_mode["centered"]["sim_promo"],
        "score_centered": scores_by_mode["centered"]["score"],
        "price_score": scores["score"],
    })
    # Mención = regla del detector (semantic/lexical/hybrid) sobre τ.
    out["above_tau"] = detector_mention(
        out["price_score"], out["lex_count"], tau, detector)
    out["detector"] = detector
    written = [save_csv(out, out_dir / "price_scores.csv")]

    # ── tau_calibration.csv (componentes del placebo, p90/p95/p99) ──
    cal_rows: list[dict] = []
    for component, vals in placebo_by_component.items():
        v = np.asarray(vals, dtype=float)
        v = v[~np.isnan(v)]
        cal_rows.append({
            "component": component,
            "n": int(v.size),
            "mean": round(float(v.mean()), 6) if v.size else np.nan,
            "p50": round(float(np.quantile(v, 0.50)), 6) if v.size else np.nan,
            "p90": round(float(np.quantile(v, 0.90)), 6) if v.size else np.nan,
            "p95": round(float(np.quantile(v, 0.95)), 6) if v.size else np.nan,
            "p99": round(float(np.quantile(v, 0.99)), 6) if v.size else np.nan,
            "max": round(float(v.max()), 6) if v.size else np.nan,
        })
    cal_rows.append({
        "component": "ALL_PLACEBO",
        "n": int(placebo_scores.size),
        "mean": round(float(placebo_scores.mean()), 6),
        "p50": round(float(np.quantile(placebo_scores, 0.50)), 6),
        "p90": round(float(np.quantile(placebo_scores, 0.90)), 6),
        "p95": round(float(np.quantile(placebo_scores, 0.95)), 6),
        "p99": round(float(np.quantile(placebo_scores, 0.99)), 6),
        "max": round(float(placebo_scores.max()), 6),
    })
    written.append(save_csv(pd.DataFrame(cal_rows), out_dir / "tau_calibration.csv"))

    # ── tau_sensitivity.csv (p50/p90/p95/p99 × raw/centered) ──
    written.append(save_csv(
        _sensitivity_table(placebo_scores, {m: s["score"] for m, s in scores_by_mode.items()}, base),
        out_dir / "tau_sensitivity.csv",
    ))

    # ── Metadata ──
    written.append(append_metadata(
        results_dir / "comparison" / "base" / "run_metadata.csv",
        task="price_core",
        params={
            "concept_id": anchors["concept_id"],
            "anchors_version": anchors["version"],
            "anchors_path": str(anchors_path),
            "model_name": model_name,
            "mode": default_mode,
            "detector": detector,
            "tau_quantile": tau_quantile,
            "tau": round(tau, 6),
            "placebo_docs": int(len(rand_idx)),
            "placebo_controls": int(sum(len(v) for v in
                                        anchors["negative_controls"].values())),
            "seed": seed,
        },
        repo_dir=Path(__file__).resolve().parents[1],
    ))

    print(f"  Docs sobre τ (detector={detector}): {int(out['above_tau'].sum()):,} / {len(out):,} "
          f"({100 * out['above_tau'].mean():.1f}%)")
    print(f"  Archivos: {[str(p) for p in written]}")

    # ── F4.4: panel comparativo (no rompe si falta algo; errores a consola) ──
    try:
        written += run_price_panel(
            args, results_dir, out, tau, tau_quantile, anchors, model_name)
    except Exception as exc:
        print(f"  ⚠️ Panel F4.4 omitido por error: {exc}")
        import traceback; traceback.print_exc()

    # ── F4.3: validación (degrada elegantemente si no hay etiquetas) ──
    valid: dict = {}
    try:
        from price_validation import run_price_validation
        valid = run_price_validation(args, results_dir, out, tau,
                                     tau_quantile=tau_quantile)
        written += list(valid.get("paths", []))
    except Exception as exc:
        print(f"  ⚠️ Validación F4.3 omitida por error: {exc}")
        import traceback; traceback.print_exc()

    # ── F4.4b: comparación DMO vs percibido + contexto + KPIs + anotación ──
    try:
        written += run_price_comparison_step(
            args, results_dir, out, tau, tau_quantile, detector, score_mode, valid)
    except Exception as exc:
        print(f"  ⚠️ Comparación F4.4b omitida por error: {exc}")
        import traceback; traceback.print_exc()

    # ── Parámetros de la corrida (insumo de la caché propia de price) ──
    params_path = out_dir / "price_params.json"
    params_path.write_text(json.dumps({
        "mode": detector, "score_mode": score_mode,
        "tau": round(float(tau), 6), "tau_quantile": tau_quantile,
        "seed": seed, "bootstrap": int(getattr(args, "bootstrap", 2000)),
        "permutations": int(getattr(args, "permutations", 1000)),
        "top_k": int(getattr(args, "price_top_k", 5)),
        "prior_strength": float(getattr(args, "prior_strength", 1000.0)),
        "sample_size": int(getattr(args, "price_sample_size", 100) or 100),
        "palette": str(getattr(args, "palette", "ejecutiva")),
        "model_name": model_name,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    written.append(params_path)

    print(f"  Archivos totales PASO 12: {len(written)}")
    return written


def run_price_task(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    """Entrada del PASO 12 en run_analysis.py: F4.2 → F4.4 → F4.3.

    Normaliza los flags nuevos del CLI (price_anchor_file, price_sample_size,
    price_annotation_file) a los nombres que consumen run_price_core y
    run_price_validation, y delega en la misma cadena existente.
    """
    # --price-sample-size (alias --price-validation-n) → price_validation_n
    sample = getattr(args, "price_sample_size", None)
    if sample is not None:
        args.price_validation_n = int(sample)
    # --price-annotation-file → ruta de anotaciones de F4.3
    ann = getattr(args, "price_annotation_file", None)
    if ann:
        args.price_annotation_file = str(Path(ann))
    return run_price_core(args, Path(output_dir))


# ══════════════════════════════════════════════════════════════════════════
# F4.4: PANEL COMPARATIVO PRECIO/VALOR (tasas, distribuciones, valencia,
# log-odds en contexto, ejemplos y scatters). Usa score_centered (default).
# ══════════════════════════════════════════════════════════════════════════
_WILSON_Z = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = _WILSON_Z) -> tuple[float, float]:
    """CI 95% de Wilson para una proporción (función pura)."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mention_rate_table(scores: pd.DataFrame, dmo_label: str) -> pd.DataFrame:
    """Tasas de mención (above_tau) DMO vs UGC_total y DMO vs cada fuente.

    Una fila por scope con tasas + CI de Wilson por grupo, gap_pp, χ²/Fisher,
    p_value, p_adj (BH dentro de la tabla), h de Cohen y veredicto
    (SOBRE_DMO = el gap favorece a DMO con p_adj<0.05).
    """
    mention = scores["above_tau"].to_numpy()
    is_dmo = scores[CORPUS_COLUMN].astype(str).str.upper() == "DMO"

    def _group_stats(mask: np.ndarray) -> dict:
        n = int(mask.sum()); k = int(mention[mask].sum())
        lo, hi = wilson_ci(k, n)
        return {"n_docs": n, "n_mentions": k, "rate": k / n if n else np.nan,
                "ci_low": lo, "ci_high": hi}

    ugc_mask = ~is_dmo
    scopes: dict[str, dict[str, np.ndarray]] = {
        "DMO_vs_UGC_total": {"DMO": is_dmo, "UGC": ugc_mask},
    }
    for source in sorted(scores.loc[ugc_mask, SOURCE_COLUMN].astype(str).unique()):
        scopes[f"DMO_vs_{source}"] = {
            "DMO": is_dmo,
            source: ugc_mask & (scores[SOURCE_COLUMN].astype(str) == source),
        }

    rows: list[dict] = []
    for scope, groups in scopes.items():
        (g_dmo, g_ugc) = list(groups.items())
        stats = {name: _group_stats(mask) for name, mask in groups.items()}
        (n1, k1) = (stats[g_dmo[0]]["n_docs"], stats[g_dmo[0]]["n_mentions"])
        (n2, k2) = (stats[g_ugc[0]]["n_docs"], stats[g_ugc[0]]["n_mentions"])
        test, p_value = two_by_two_test(k1, n1 - k1, k2, n2 - k2)
        rows.append({
            "scope": scope,
            "group_dmo": g_dmo[0], "group_ugc": g_ugc[0],
            "n_docs_dmo": n1, "n_mentions_dmo": k1,
            "rate_dmo": round(stats[g_dmo[0]]["rate"], 6),
            "ci_low_dmo": round(stats[g_dmo[0]]["ci_low"], 6),
            "ci_high_dmo": round(stats[g_dmo[0]]["ci_high"], 6),
            "n_docs_ugc": n2, "n_mentions_ugc": k2,
            "rate_ugc": round(stats[g_ugc[0]]["rate"], 6),
            "ci_low_ugc": round(stats[g_ugc[0]]["ci_low"], 6),
            "ci_high_ugc": round(stats[g_ugc[0]]["ci_high"], 6),
            "gap_pp": round(100 * (stats[g_dmo[0]]["rate"] - stats[g_ugc[0]]["rate"]), 4),
            "test": test, "p_value": p_value,
            "cohens_h": round(cohens_h(stats[g_dmo[0]]["rate"], stats[g_ugc[0]]["rate"]), 6),
        })
    tbl = pd.DataFrame(rows)
    if len(tbl):
        tbl["p_adj"] = bh_adjust(tbl["p_value"].to_numpy())
        tbl["verdict"] = [
            ("SOBRE_DMO" if g > 0 else "SOBRE_UGC") if p < 0.05 else "SIN_DIF"
            for g, p in zip(tbl["gap_pp"], tbl["p_adj"])
        ]
    return tbl


def score_distribution_tests(
    dmo_scores: np.ndarray, ugc_scores: np.ndarray,
    bootstrap: int = 2000, permutations: int = 1000, seed: int = 42,
) -> dict:
    """KS, Wasserstein-1D, diff de medias con CI bootstrap y p por permutación."""
    from scipy.stats import ks_2samp, wasserstein_distance

    dmo = np.asarray(dmo_scores, dtype=float)
    dmo = dmo[~np.isnan(dmo)]
    ugc = np.asarray(ugc_scores, dtype=float)
    ugc = ugc[~np.isnan(ugc)]
    rng = np.random.default_rng(seed)
    diff = float(dmo.mean() - ugc.mean()) if len(dmo) and len(ugc) else np.nan

    boots = np.empty(max(0, bootstrap))
    for start in range(0, len(boots), 256):
        b = min(256, len(boots) - start)
        idx_d = rng.integers(0, len(dmo), size=(b, len(dmo)))
        idx_u = rng.integers(0, len(ugc), size=(b, len(ugc)))
        boots[start:start + b] = dmo[idx_d].mean(axis=1) - ugc[idx_u].mean(axis=1)
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) \
        if boots.size else (np.nan, np.nan)

    pool = np.concatenate([dmo, ugc]); n_d = len(dmo)
    perms = np.empty(max(0, permutations))
    for start in range(0, len(perms), 128):
        b = min(128, len(perms) - start)
        order = np.argsort(rng.random((b, len(pool))), axis=1)
        perms[start:start + b] = (pool[order[:, :n_d]].mean(axis=1)
                                  - pool[order[:, n_d:]].mean(axis=1))
    p_perm = float((np.sum(np.abs(perms) >= abs(diff)) + 1) / (len(perms) + 1)) \
        if perms.size else np.nan

    return {
        "n_dmo": int(len(dmo)), "n_ugc": int(len(ugc)),
        "mean_dmo": round(float(dmo.mean()), 6) if len(dmo) else np.nan,
        "mean_ugc": round(float(ugc.mean()), 6) if len(ugc) else np.nan,
        "mean_diff": round(diff, 6) if diff == diff else np.nan,
        "ci_low": round(ci[0], 6), "ci_high": round(ci[1], 6),
        "ks_statistic": round(float(ks_2samp(dmo, ugc).statistic), 6) if len(dmo) and len(ugc) else np.nan,
        "ks_p_value": round(float(ks_2samp(dmo, ugc).pvalue), 6) if len(dmo) and len(ugc) else np.nan,
        "wasserstein_1d": round(float(wasserstein_distance(dmo, ugc)), 6) if len(dmo) and len(ugc) else np.nan,
        "perm_p_value": round(p_perm, 6) if p_perm == p_perm else np.nan,
        "bootstrap": int(bootstrap), "permutations": int(permutations),
    }


def valence_conditioned_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Sentimiento condicionado a mención (1) vs no mención (0) + banda neutra."""
    rows = []
    is_dmo = scores[CORPUS_COLUMN].astype(str).str.upper() == "DMO"
    sent = scores["sentiment_score"].astype(float).to_numpy()
    neutral = np.abs(sent) < 0.15
    for corpus, mask in (("DMO", is_dmo), ("UGC_total", ~is_dmo)):
        for mention in (1, 0):
            m = mask & (scores["above_tau"].to_numpy() == bool(mention))
            sub = sent[m]; nb = sent[m & neutral]
            rows.append({
                "corpus": corpus, "mention": mention, "n": int(m.sum()),
                "mean_sentiment": round(float(np.mean(sub)), 6) if sub.size else np.nan,
                "share_positive": round(float((scores.loc[m, "sentiment_label"] == "positive").mean()), 6) if m.sum() else np.nan,
                "n_neutral_band": int((m & neutral).sum()),
                "share_neutral": round(float((m & neutral).sum() / m.sum()), 6) if m.sum() else np.nan,
                "mean_sentiment_neutral": round(float(np.mean(nb)), 6) if nb.size else np.nan,
            })
    return pd.DataFrame(rows)


def price_context_logodds(
    scores: pd.DataFrame, prior_strength: float = 1000.0, min_count: int = 3,
) -> pd.DataFrame:
    """Log-odds Monroe (2008) dentro de docs que mencionan precio, DMO vs UGC."""
    from sklearn.feature_extraction.text import CountVectorizer

    mention = scores[scores["above_tau"]].copy()
    is_dmo = mention[CORPUS_COLUMN].astype(str).str.upper() == "DMO"
    if is_dmo.sum() == 0 or (~is_dmo).sum() == 0:
        return pd.DataFrame(columns=["term", "freq_dmo", "freq_ugc", "z", "direction"])
    for md, xd in ((min_count, 1.0), (1, 1.0)):
        try:
            vec = CountVectorizer(token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
                                  lowercase=True, min_df=md, max_df=xd)
            X = vec.fit_transform(mention["text"].astype(str)).astype(np.float64)
            if X.shape[1]:
                break
        except ValueError:
            continue
    fd = np.asarray(X[is_dmo.to_numpy()].sum(axis=0)).ravel()
    fu = np.asarray(X[~is_dmo.to_numpy()].sum(axis=0)).ravel()
    z = logodds_z(fd, fu, alpha0=float(prior_strength))
    terms = np.asarray(vec.get_feature_names_out())
    keep = (fd + fu) >= min_count
    out = pd.DataFrame({
        "term": terms[keep],
        "freq_dmo": fd[keep].astype(int),
        "freq_ugc": fu[keep].astype(int),
        "z": np.round(z[keep], 4),
    }).sort_values("z", ascending=False).reset_index(drop=True)
    out["direction"] = np.where(out["z"] > 0, "DMO", "UGC")
    return out


def price_top_examples(
    scores: pd.DataFrame, top_k: int = 10, max_chars: int = 300,
) -> pd.DataFrame:
    """Top-k docs por score dentro de cada corpus y cada fuente."""
    frames = []
    keys = [CORPUS_COLUMN, SOURCE_COLUMN]
    for keys_values, group in scores.groupby(keys, sort=False):
        corpus, source = keys_values
        top = group.sort_values("price_score", ascending=False).head(top_k).copy()
        top.insert(0, "rank", np.arange(1, len(top) + 1))
        top["corpus_scope"] = f"{corpus}|{source}"
        top["text_excerpt"] = top["text"].astype(str).str.slice(0, max_chars)
        frames.append(top)
    out = pd.concat(frames, ignore_index=True)
    cols = ["corpus_scope", "rank", "doc_id", CORPUS_COLUMN, SOURCE_COLUMN,
            "price_score", "sim_resena_centered", "sim_promo_centered",
            "lex_count", "sentiment_label", "sentiment_score", "text_excerpt"]
    return out[cols]


def _hover_text(df: pd.DataFrame, max_chars: int = 140) -> list[str]:
    """Hover compacto (score + sentimiento + texto truncado)."""
    return [
        f"#{cid} · {src}<br>score={s:.3f} (R={r:.3f}, P={p:.3f}) · lex={lx}"
        f"<br>sent={se} ({sv:.2f})<br>{tx[:max_chars]}"
        for cid, src, s, r, p, lx, se, sv, tx in zip(
            df[CORPUS_COLUMN], df[SOURCE_COLUMN], df["price_score"],
            df["sim_resena_centered"], df["sim_promo_centered"],
            df["lex_count"], df["sentiment_label"], df["sentiment_score"],
            df["text"].astype(str))
    ]


def price_scatter_prototypes(scores: pd.DataFrame, tau: float,
                             colors: list[str]) -> "go.Figure":
    """Scatter (x=sim_resena, y=sim_promo) Scattergl; color=corpus, símbolo=fuente."""
    import plotly.graph_objects as go

    symbols = ["circle", "square", "diamond", "cross", "triangle-up", "x",
               "star", "pentagon", "hexagram", "asterisk"]
    sources = sorted(scores[SOURCE_COLUMN].astype(str).unique())
    sym_map = {s: symbols[i % len(symbols)] for i, s in enumerate(sources)}
    fig = go.Figure()
    for i, corpus in enumerate(["DMO", "UGC"]):
        sub_all = scores[scores[CORPUS_COLUMN].astype(str).str.upper() == corpus]
        for j, source in enumerate(sources):
            sub = sub_all[sub_all[SOURCE_COLUMN].astype(str) == source]
            if sub.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=sub["sim_resena_centered"], y=sub["sim_promo_centered"],
                mode="markers", name=f"{corpus} · {source}",
                marker={"color": colors[i % len(colors)],
                        "size": np.clip(4 + 4 * sub["lex_count"].to_numpy(), 4, 20),
                        "opacity": 0.55, "symbol": sym_map[source],
                        "line": {"width": 0}},
                text=_hover_text(sub), hoverinfo="text"))
    tau_line = {"color": "#444", "width": 1, "dash": "dash"}
    fig.add_hline(y=tau, line=tau_line, annotation_text=f"τ promo={tau:.3f}")
    fig.add_vline(x=tau, line=tau_line, annotation_text=f"τ reseña={tau:.3f}")
    fig.update_layout(
        title="Precio/Valor: reseña vs promocional (score centrado)",
        xaxis_title="sim_resena (centered)", yaxis_title="sim_promo (centered)",
        template="plotly_white", legend={"title": "corpus · fuente"})
    return fig


def price_scatter_valence(scores: pd.DataFrame, tau: float,
                          colors: list[str]) -> "go.Figure":
    """Scatter score vs sentimiento; línea vertical τ y horizontal 0."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, corpus in enumerate(["DMO", "UGC"]):
        sub = scores[scores[CORPUS_COLUMN].astype(str).str.upper() == corpus]
        if sub.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=sub["price_score"], y=sub["sentiment_score"].astype(float),
            mode="markers", name=corpus,
            marker={"color": colors[i % len(colors)], "size": 6, "opacity": 0.55},
            text=_hover_text(sub), hoverinfo="text"))
    fig.add_vline(x=tau, line={"color": "#444", "width": 1, "dash": "dash"},
                  annotation_text=f"τ={tau:.3f}")
    fig.add_hline(y=0.0, line={"color": "#999", "width": 1})
    fig.update_layout(
        title="Precio/Valor: score vs sentimiento",
        xaxis_title="price_score (centered)", yaxis_title="sentiment_score",
        template="plotly_white")
    return fig


def price_summary_figure(
    rates: pd.DataFrame, scores: pd.DataFrame, terms: pd.DataFrame,
    tau: float, colors: list[str],
) -> "go.Figure":
    """Panel resumen: tasas con CI, densidades de score, valencia condicionada."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2, subplot_titles=(
            "Tasa de mención por scope (CI 95%)", "Distribución de score",
            "Sentimiento condicionado a mención", "Términos distintivos (|z| top-12)"),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "table"}]])

    n_rows = len(rates)
    for i, row in rates.iterrows():
        for suffix, color_idx in (("dmo", 0), ("ugc", 1)):
            fig.add_trace(go.Bar(
                x=[row["scope"]], y=[row[f"rate_{suffix}"]],
                name=f"{'DMO' if suffix == 'dmo' else row['group_ugc']}",
                marker_color=colors[color_idx % len(colors)],
                error_y={"type": "data", "symmetric": False,
                         "array": [row[f"ci_high_{suffix}"] - row[f"rate_{suffix}"]],
                         "arrayminus": [row[f"rate_{suffix}"] - row[f"ci_low_{suffix}"]]},
                showlegend=i == 0), row=1, col=1)

    for i, corpus in enumerate(["DMO", "UGC"]):
        sub = scores[scores[CORPUS_COLUMN].astype(str).str.upper() == corpus]
        fig.add_trace(go.Histogram(
            x=sub["price_score"], name=corpus, histnorm="probability density",
            opacity=0.6, marker_color=colors[i % len(colors)], nbinsx=40,
            showlegend=i == 0), row=1, col=2)

    vc = valence_conditioned_table(scores)
    for j, mention in enumerate([1, 0]):
        sub = vc[vc["mention"] == mention]
        fig.add_trace(go.Bar(
            x=sub["corpus"], y=sub["mean_sentiment"],
            name=f"mention={mention}",
            marker_color=colors[j % len(colors)], showlegend=False), row=2, col=1)

    top_terms = pd.concat(
        [terms.head(6), terms.tail(6)]).drop_duplicates() if len(terms) else terms
    fig.add_trace(go.Table(
        header={"values": ["término", "z", "dirección"]},
        cells={"values": [top_terms["term"], top_terms["z"], top_terms["direction"]]},
        columnwidth=[120, 50, 70]), row=2, col=2)

    # Línea de τ sobre el histograma (x2): add_vline falla con Table en layout.
    fig.add_shape(type="line", xref="x2", yref="paper", x0=tau, x1=tau, y0=0, y1=1,
                  line={"color": "#444", "width": 1, "dash": "dash"})
    fig.add_annotation(xref="x2", yref="paper", x=tau, y=0.98,
                       text=f"τ={tau:.3f}", showarrow=False, font={"size": 10})
    fig.update_layout(
        title="Panel Precio/Valor/Costo (DMO vs UGC)", template="plotly_white",
        barmode="group", height=850)
    return fig


def run_price_panel(
    args: argparse.Namespace, results_dir: Path,
    scores: pd.DataFrame, tau: float, tau_quantile: float,
    anchors: dict, model_name: str,
) -> list[Path]:
    """F4.4: CSVs + figuras HTML del panel Precio/Valor (sin validación)."""
    print_header("PASO 12.6 (F4.4): PANEL PRECIO/VALOR DMO vs UGC")
    out_dir = results_dir / "comparison" / PRICE_OUTPUT_SUBDIR
    colors = palette_colors(palette_to_plot_colors(
        getattr(args, "palette", "ejecutiva")))
    dmo_label = str(scores.loc[scores[CORPUS_COLUMN].astype(str).str.upper() == "DMO",
                               SOURCE_COLUMN].astype(str).iloc[0]) \
        if (scores[CORPUS_COLUMN].astype(str).str.upper() == "DMO").any() else "DMO"
    top_k = int(getattr(args, "price_top_k", 10))
    written: list[Path] = []

    # 1) Tasas de mención + gap + inferencia
    rates = mention_rate_table(scores, dmo_label)
    written.append(save_csv(rates, out_dir / "mention_rate_by_source.csv"))

    # 2) Distribuciones
    is_dmo = scores[CORPUS_COLUMN].astype(str).str.upper() == "DMO"
    dist = score_distribution_tests(
        scores.loc[is_dmo, "price_score"].to_numpy(),
        scores.loc[~is_dmo, "price_score"].to_numpy(),
        bootstrap=int(getattr(args, "bootstrap", 2000)),
        permutations=int(getattr(args, "permutations", 1000)),
        seed=int(getattr(args, "seed", 42)))
    written.append(save_csv(pd.DataFrame([dist]), out_dir / "score_distribution_tests.csv"))

    # 3) Valencia condicionada
    valence = valence_conditioned_table(scores)
    written.append(save_csv(valence, out_dir / "valence_conditioned.csv"))

    # 4) Log-odds en contexto de mención
    terms = price_context_logodds(
        scores, prior_strength=float(getattr(args, "prior_strength", 1000.0)))
    written.append(save_csv(terms, out_dir / "price_context_logodds.csv"))

    # 5) Ejemplos
    examples = price_top_examples(scores, top_k=top_k)
    written.append(save_csv(examples, out_dir / "price_top_examples.csv"))

    # 6) Scatter prototipos
    fig_proto = price_scatter_prototypes(scores, tau, colors)
    path_proto = out_dir / "price_scatter_prototypes.html"
    fig_proto.write_html(str(path_proto), include_plotlyjs="inline", full_html=True)
    written.append(path_proto)
    written.append(save_csv(
        scores[[c for c in scores.columns if c != "text"]],
        out_dir / "price_scatter_prototypes_data.csv"))

    # 7) Scatter valencia
    fig_val = price_scatter_valence(scores, tau, colors)
    path_val = out_dir / "price_scatter_valence.html"
    fig_val.write_html(str(path_val), include_plotlyjs="inline", full_html=True)
    written.append(path_val)
    written.append(save_csv(
        scores[["doc_id", CORPUS_COLUMN, SOURCE_COLUMN, "price_score",
                "sentiment_label", "sentiment_score", "lex_count"]],
        out_dir / "price_scatter_valence_data.csv"))

    # 8) Panel resumen + CSV de densidades
    fig_sum = price_summary_figure(rates, scores, terms, tau, colors)
    path_sum = out_dir / "price_summary.html"
    fig_sum.write_html(str(path_sum), include_plotlyjs="inline", full_html=True)
    written.append(path_sum)
    bins = np.linspace(0, max(1e-9, float(scores["price_score"].max())), 41)
    dens_rows = []
    for corpus in ["DMO", "UGC"]:
        sub = scores[scores[CORPUS_COLUMN].astype(str).str.upper() == corpus]["price_score"]
        hist, _ = np.histogram(sub, bins=bins, density=True)
        for lo, cnt in zip(bins[:-1], hist):
            dens_rows.append({"bin_low": round(float(lo), 6), "corpus": corpus,
                              "density": round(float(cnt), 8)})
    written.append(save_csv(pd.DataFrame(dens_rows), out_dir / "price_density.csv"))

    # ── Consola ──
    prov = anchors.get("provenance", {})
    print(f"  Anchors: {anchors.get('concept_id')} v{anchors.get('version')} | "
          f"generator={prov.get('generator')} | corpus_seen={prov.get('corpus_seen')}")
    print(f"  τ = {tau:.4f} (cuantil {tau_quantile} del placebo)")
    r0 = rates.iloc[0] if len(rates) else None
    if r0 is not None:
        print(f"  Tasa DMO: {r0['rate_dmo']:.3f} [{r0['ci_low_dmo']:.3f}, {r0['ci_high_dmo']:.3f}]"
              f" | UGC: {r0['rate_ugc']:.3f} [{r0['ci_low_ugc']:.3f}, {r0['ci_high_ugc']:.3f}]"
              f" | gap={r0['gap_pp']:.1f}pp ({r0['test']}, p_adj={r0['p_adj']:.4f}, {r0['verdict']})")
    print(f"  KS={dist['ks_statistic']:.3f} (p={dist['ks_p_value']:.4f}) | "
          f"W1={dist['wasserstein_1d']:.4f} | Δmedia={dist['mean_diff']:.4f} "
          f"[{dist['ci_low']:.4f}, {dist['ci_high']:.4f}] | perm p={dist['perm_p_value']:.4f}")
    pos = valence[valence["mention"] == 1].set_index("corpus")
    if "DMO" in pos.index and "UGC_total" in pos.index:
        print(f"  Valencia | mención: DMO={pos.loc['DMO', 'mean_sentiment']:.3f} "
              f"(pos={100 * pos.loc['DMO', 'share_positive']:.1f}%) vs "
              f"UGC={pos.loc['UGC_total', 'mean_sentiment']:.3f} "
              f"(pos={100 * pos.loc['UGC_total', 'share_positive']:.1f}%)")
    if len(examples):
        print("  Top-5 ejemplos:")
        for _, ex in examples[examples["rank"] <= 5].head(5).iterrows():
            print(f"    [{ex['corpus_scope']}] {ex['price_score']:.3f} · "
                  f"{ex['text_excerpt'][:90]!r}")
    if r0 is not None:
        verdict = r0["verdict"]
        frase = ("DMO proyecta relativamente MÁS precio/valor del que percibe la UGC"
                 if verdict == "SOBRE_DMO" else
                 "la UGC percibe relativamente MÁS precio/valor del que proyecta la DMO"
                 if verdict == "SOBRE_UGC" else
                 "no hay diferencia significativa en la saliencia de precio/valor")
        print(f"  📌 Interpretación: {frase} (gap={r0['gap_pp']:.1f}pp, "
              f"p_adj={r0['p_adj']:.4f}, h={r0['cohens_h']:.2f}).")
    print(f"  Figuras: {[p.name for p in written if str(p).endswith('.html')]}")

    written.append(append_metadata(
        results_dir / "comparison" / "base" / "run_metadata.csv",
        task="price_panel",
        params={
            "tau": round(float(tau), 6),
            "tau_quantile": tau_quantile,
            "n_docs": int(len(scores)),
            "top_k": int(top_k),
            "permutations": int(getattr(args, "permutations", 1000)),
            "bootstrap": int(getattr(args, "bootstrap", 2000)),
            "seed": int(getattr(args, "seed", 42)),
        },
        repo_dir=Path(__file__).resolve().parents[1],
    ))
    return written


# ══════════════════════════════════════════════════════════════════════════
# F4.4b: DMO vs percibido (tasas bootstrap, χ²/Fisher+BH, MW/KS+Cliff),
# contexto, KPIs del dashboard, ejemplos top-k y scatter de valencia.
# ══════════════════════════════════════════════════════════════════════════
PRICE_OUTPUTS = ("price_comparison.csv", "price_context_terms.csv",
                 "price_kpis.csv", "price_examples_topk.csv",
                 "price_scatter_valence.html")


def price_comparison_table(scores: pd.DataFrame, bootstrap: int = 2000,
                           seed: int = 42) -> pd.DataFrame:
    """DMO vs UGC_total y DMO vs cada fuente: tasas + CI bootstrap (seed fija),
    χ²/Fisher + BH, Mann-Whitney/KS + Cliff's delta y banda neutra."""
    from scipy.stats import ks_2samp

    mention = scores["above_tau"].to_numpy(dtype=bool)
    dmo = scores[CORPUS_COLUMN].astype(str).str.upper() == "DMO"
    ugc_all = (~dmo).to_numpy()
    sc = scores["price_score"].to_numpy(dtype=float)
    sent = scores["sentiment_score"].astype(float).to_numpy()
    neutral = np.abs(sent) < 0.15
    scopes: dict[str, tuple[np.ndarray, str]] = {
        "DMO_vs_UGC_total": (ugc_all, "UGC_total")}
    for src in sorted(scores.loc[~dmo, SOURCE_COLUMN].astype(str).unique()):
        scopes[f"DMO_vs_{src}"] = (
            ugc_all & (scores[SOURCE_COLUMN].astype(str) == src).to_numpy(), src)
    dmo_np = dmo.to_numpy()
    rows: list[dict] = []
    for i, (scope, (ugc, ugc_name)) in enumerate(scopes.items()):
        n1, k1 = int(dmo_np.sum()), int((mention & dmo_np).sum())
        n2, k2 = int(ugc.sum()), int((mention & ugc).sum())
        r1 = k1 / n1 if n1 else np.nan
        r2 = k2 / n2 if n2 else np.nan
        lo1, hi1 = bootstrap_rate_ci(mention[dmo_np], bootstrap, seed + i)
        lo2, hi2 = bootstrap_rate_ci(mention[ugc], bootstrap, seed + 1000 + i)
        test, p_value = two_by_two_test(k1, n1 - k1, k2, n2 - k2)
        mw = mann_whitney_cliffs(sc[dmo_np], sc[ugc])
        ks = ks_2samp(sc[dmo_np], sc[ugc]) if n1 and n2 else None
        rows.append({
            "scope": scope, "group_dmo": "DMO", "group_ugc": ugc_name,
            "n_docs_dmo": n1, "n_mentions_dmo": k1,
            "rate_dmo": round(r1, 6), "ci_low_dmo": round(lo1, 6),
            "ci_high_dmo": round(hi1, 6),
            "n_docs_ugc": n2, "n_mentions_ugc": k2,
            "rate_ugc": round(r2, 6), "ci_low_ugc": round(lo2, 6),
            "ci_high_ugc": round(hi2, 6),
            "gap_pp": round(100 * (r1 - r2), 4) if r1 == r1 and r2 == r2 else np.nan,
            "test": test, "p_value": p_value,
            "mw_u": mw["mw_u"], "mw_p_value": mw["mw_p_value"],
            "cliffs_delta": mw["cliffs_delta"], "cliffs_interp": mw["cliffs_interp"],
            "ks_statistic": round(float(ks.statistic), 6) if ks is not None else np.nan,
            "ks_p_value": round(float(ks.pvalue), 6) if ks is not None else np.nan,
            "neutral_share_dmo": round(float(neutral[dmo_np].mean()), 6) if n1 else np.nan,
            "neutral_share_ugc": round(float(neutral[ugc].mean()), 6) if n2 else np.nan,
        })
    tbl = pd.DataFrame(rows)
    if len(tbl):
        tbl["p_adj"] = bh_adjust(tbl["p_value"].to_numpy())
        tbl["verdict"] = [
            ("SOBRE_DMO" if g > 0 else "SOBRE_UGC") if p < 0.05 else "SIN_DIF"
            for g, p in zip(tbl["gap_pp"], tbl["p_adj"])]
    return tbl


def price_context_cooccur(scores: pd.DataFrame, prior_strength: float = 1000.0,
                          top_k: int = 20, min_count: int = 3) -> pd.DataFrame:
    """Términos que co-ocurren con la mención de precio (tokenize + log-odds)."""
    from collections import Counter

    from metrics_lexical import tokenize

    mention = scores[scores["above_tau"]]
    is_dmo = mention[CORPUS_COLUMN].astype(str).str.upper() == "DMO"
    cols = ["term", "docs_dmo", "docs_ugc", "freq_dmo", "freq_ugc", "z", "direction"]
    if is_dmo.sum() == 0 or (~is_dmo).sum() == 0:
        return pd.DataFrame(columns=cols)
    dd, du, cd, cu = Counter(), Counter(), Counter(), Counter()
    tcol = "text_norm" if "text_norm" in scores.columns else "text"
    for tokens, dmo in zip((tokenize(t) for t in mention[tcol].astype(str)),
                           is_dmo.to_numpy()):
        docs, cnts = (dd, cd) if dmo else (du, cu)
        docs.update(set(tokens)); cnts.update(tokens)
    terms = np.asarray(sorted(set(dd) | set(du)), dtype=object)
    fd = np.asarray([dd[t] for t in terms], dtype=float)
    fu = np.asarray([du[t] for t in terms], dtype=float)
    z = logodds_z(fd, fu, alpha0=float(prior_strength))
    keep = (fd + fu) >= min_count
    out = pd.DataFrame({
        "term": terms[keep], "docs_dmo": fd[keep].astype(int),
        "docs_ugc": fu[keep].astype(int),
        "freq_dmo": [cd[t] for t in terms[keep]],
        "freq_ugc": [cu[t] for t in terms[keep]], "z": np.round(z[keep], 4)})
    out = out.iloc[np.argsort(-np.abs(out["z"].to_numpy()))[:top_k]]
    out["direction"] = np.where(out["z"] > 0, "DMO", "UGC")
    return out.reset_index(drop=True)


def price_kpi_table(scores: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """KPIs planos (kpi,value,unit,scope) para el dashboard. Sin anotaciones."""
    is_dmo = scores[CORPUS_COLUMN].astype(str).str.upper() == "DMO"
    sc = scores["price_score"].astype(float)
    neutral = np.abs(scores["sentiment_score"].astype(float)) < 0.15
    r0 = rates.iloc[0] if len(rates) else None
    rows: list[dict] = []

    def add(kpi: str, value: Any, unit: str, scope: str = "total") -> None:
        rows.append({"kpi": kpi, "value": value, "unit": unit, "scope": scope})

    add("n_docs", int(len(scores)), "docs")
    add("n_docs_dmo", int(is_dmo.sum()), "docs", "DMO")
    add("n_docs_ugc", int((~is_dmo).sum()), "docs", "UGC")
    add("mean_score_dmo", round(float(sc[is_dmo].mean()), 6), "score", "DMO")
    add("mean_score_ugc", round(float(sc[~is_dmo].mean()), 6), "score", "UGC")
    add("share_neutral_dmo", round(float(neutral[is_dmo].mean()), 6), "share", "DMO")
    add("share_neutral_ugc", round(float(neutral[~is_dmo].mean()), 6), "share", "UGC")
    if r0 is not None:
        add("mention_rate_dmo", float(r0["rate_dmo"]), "share", "DMO")
        add("mention_rate_ugc", float(r0["rate_ugc"]), "share", "UGC")
        add("mention_gap_pp", float(r0["gap_pp"]), "pp", "DMO_vs_UGC")
        add("cliffs_delta", float(r0["cliffs_delta"]), "delta", "DMO_vs_UGC")
        add("p_adj", float(r0["p_adj"]), "p", "DMO_vs_UGC")
        add("verdict", str(r0["verdict"]), "texto", "DMO_vs_UGC")
    return pd.DataFrame(rows, columns=["kpi", "value", "unit", "scope"])


def price_scatter_valence_sources(scores: pd.DataFrame, tau: float,
                                  colors: list[str]) -> "go.Figure":
    """x=proximidad (price_score), y=sentimiento; color=corpus, símbolo=fuente."""
    import plotly.graph_objects as go

    symbols = ["circle", "square", "diamond", "cross", "triangle-up", "x",
               "star", "pentagon", "hexagram", "asterisk"]
    sources = sorted(scores[SOURCE_COLUMN].astype(str).unique())
    sym_map = {s: symbols[i % len(symbols)] for i, s in enumerate(sources)}
    fig = go.Figure()
    for i, corpus in enumerate(["DMO", "UGC"]):
        sub_all = scores[scores[CORPUS_COLUMN].astype(str).str.upper() == corpus]
        for source in sources:
            sub = sub_all[sub_all[SOURCE_COLUMN].astype(str) == source]
            if sub.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=sub["price_score"], y=sub["sentiment_score"].astype(float),
                mode="markers", name=f"{corpus} · {source}",
                marker={"color": colors[i % len(colors)], "size": 7,
                        "opacity": 0.55, "symbol": sym_map[source],
                        "line": {"width": 0}},
                text=_hover_text(sub), hoverinfo="text"))
    fig.add_vline(x=tau, line={"color": "#444", "width": 1, "dash": "dash"},
                  annotation_text=f"τ={tau:.3f}")
    fig.add_hline(y=0.0, line={"color": "#999", "width": 1})
    fig.update_layout(
        title="Precio/Valor: proximidad vs valencia (DMO vs UGC)",
        xaxis_title="price_score (proximidad al concepto)",
        yaxis_title="sentiment_score", template="plotly_white",
        legend={"title": "corpus · fuente"})
    return fig


def _price_fingerprint(scores: pd.DataFrame, params: dict) -> str:
    """Hash estable de parámetros + scores (fingerprint de la caché propia)."""
    cols = [c for c in ("doc_id", CORPUS_COLUMN, SOURCE_COLUMN, "price_score",
                        "above_tau", "sentiment_score") if c in scores.columns]
    params = {**params, "text_norm_v": 2}  # invalidate caché al cambiar text_norm
    digest = hashlib.sha256(json.dumps(
        params, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(
        scores[cols].reset_index(drop=True), index=False).to_numpy().tobytes())
    return digest.hexdigest()


def run_price_comparison_step(
    args: argparse.Namespace, results_dir: Path, scores: pd.DataFrame,
    tau: float, tau_quantile: float, detector: str, score_mode: str,
    valid: dict,
) -> list[Path]:
    """F4.4b: comparación DMO vs percibido + contexto + KPIs + dashboard.

    Caché propia idempotente: <cache-dir>/price/manifest.json guarda un
    fingerprint de (parámetros, scores); si no cambia y los outputs existen,
    la segunda corrida no recalcula nada. `--no-cache` fuerza el recálculo.
    """
    out_dir = results_dir / "comparison" / PRICE_OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(getattr(args, "cache_dir", None) or (results_dir / "cache"))
    params = {
        "mode": detector, "score_mode": score_mode,
        "tau": round(float(tau), 6), "tau_quantile": float(tau_quantile),
        "seed": int(getattr(args, "seed", 42)),
        "bootstrap": int(getattr(args, "bootstrap", 2000)),
        "top_k": int(getattr(args, "price_top_k", 5)),
        "prior_strength": float(getattr(args, "prior_strength", 1000.0)),
        "palette": str(getattr(args, "palette", "ejecutiva")),
        "n_docs": int(len(scores)),
    }
    fp = _price_fingerprint(scores, params)
    manifest_path = cache_dir / "price" / "manifest.json"
    outputs = [out_dir / name for name in PRICE_OUTPUTS]
    try:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        cached = {}
    if (cached.get("fingerprint") == fp and all(p.exists() for p in outputs)
            and not getattr(args, "no_cache", False)):
        print_header("PASO 12.7 (F4.4b): COMPARACIÓN DMO vs PERCIBIDO (caché)")
        print(f"  Sin cambios (fingerprint {fp[:12]}): no se recalcula.")
        print(f"  Archivos reutilizados: {[p.name for p in outputs]}")
        return outputs

    print_header("PASO 12.7 (F4.4b): COMPARACIÓN DMO vs PERCIBIDO + CONTEXTO")
    rates = price_comparison_table(
        scores, bootstrap=params["bootstrap"], seed=params["seed"])
    written = [save_csv(rates, out_dir / "price_comparison.csv")]
    terms = price_context_cooccur(
        scores, prior_strength=params["prior_strength"], top_k=20)
    written.append(save_csv(terms, out_dir / "price_context_terms.csv"))
    written.append(save_csv(price_kpi_table(scores, rates),
                            out_dir / "price_kpis.csv"))
    written.append(save_csv(
        price_top_examples(scores, top_k=params["top_k"], max_chars=280),
        out_dir / "price_examples_topk.csv"))

    colors = palette_colors(palette_to_plot_colors(params["palette"]))
    fig = price_scatter_valence_sources(scores, tau, colors)
    path_val = out_dir / "price_scatter_valence.html"
    fig.write_html(str(path_val), include_plotlyjs="inline", full_html=True)
    written.append(path_val)
    written.append(save_csv(
        scores[["doc_id", CORPUS_COLUMN, SOURCE_COLUMN, "price_score",
                "sentiment_label", "sentiment_score", "lex_count", "above_tau"]],
        out_dir / "price_scatter_valence_data.csv"))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(
        {"fingerprint": fp, "params": params,
         "outputs": [p.name for p in written]},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    written.append(manifest_path)
    written.append(append_metadata(
        results_dir / "comparison" / "base" / "run_metadata.csv",
        task="price_comparison",
        params={"mode": detector, "score_mode": score_mode,
                "tau": round(float(tau), 6), "tau_quantile": tau_quantile,
                "seed": params["seed"], "bootstrap": params["bootstrap"],
                "top_k": params["top_k"], "n_docs": params["n_docs"]},
        repo_dir=Path(__file__).resolve().parents[1]))

    r0 = rates.iloc[0] if len(rates) else None
    if r0 is not None:
        print(f"  Mención DMO={r0['rate_dmo']:.3f} "
              f"[{r0['ci_low_dmo']:.3f}, {r0['ci_high_dmo']:.3f}] vs "
              f"UGC={r0['rate_ugc']:.3f} [{r0['ci_low_ugc']:.3f}, "
              f"{r0['ci_high_ugc']:.3f}] | gap={r0['gap_pp']:.1f}pp | "
              f"{r0['test']} p_adj={r0['p_adj']:.4f} | "
              f"MW p={r0['mw_p_value']:.4f} Cliff δ={r0['cliffs_delta']:.3f} "
              f"({r0['cliffs_interp']})")
        print(f"  Banda neutra (|sentiment|<0.15): "
              f"DMO={100 * r0['neutral_share_dmo']:.1f}% | "
              f"UGC={100 * r0['neutral_share_ugc']:.1f}%")
    print(f"  Contexto top-5: {terms['term'].head(5).tolist() if len(terms) else []}")
    print(f"  F4.3={valid.get('status', 'pendiente')} | "
          f"figura={path_val.name} | caché={manifest_path} (fp {fp[:12]})")
    print(f"  Archivos: {[p.name for p in written]}")
    return written

