"""
Tarea 'base' (P0): base analítica consolidada DMO vs UGC.

Reutiliza la caché del pipeline (step_load_corpus, step_embeddings_bert,
step_outliers_ngram, step_sentiment) SIN recalcular nada. Reconstruye el
doc_id original por contenido (los CSV de caché no conservan el índice)
y verifica la alineación documento ↔ embedding antes de escribir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from io_corpus import CORPUS_COLUMN, DMO_LABEL, SOURCE_COLUMN
from comparison.common import append_metadata, print_header, save_csv

NA_SENTINEL = "__NA__"


# ─────────────────────────── Funciones puras ───────────────────────────
def replicate_pipeline_filter(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Réplica EXACTA del PASO 1 de run_analysis.py (fillna/str/len>5/reset_index)."""
    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    return df[df[text_column].str.strip().str.len() > 5].reset_index(drop=True)


def binarize_sentiment(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Misma regla binaria que run_analysis._asegurar_sentimiento_binario."""
    df = df.copy()
    if "sentiment_score" in df.columns:
        df["sentiment_label"] = np.where(
            df["sentiment_score"] >= threshold, "positive", "negative"
        )
    else:
        df["sentiment_label"] = (
            df["sentiment_label"].astype(str).str.lower()
            .map(lambda x: "positive" if x.startswith(("pos", "posi")) else "negative")
        )
    return df


def _row_keys(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """Claves hashables por fila (NaN normalizado a un sentinel)."""
    sub = df[cols].astype(object)
    sub = sub.where(sub.notna(), other=NA_SENTINEL)
    return [tuple(map(str, row)) for row in sub.itertuples(index=False, name=None)]


def _two_pointer_match(
    keys_u: list[tuple], keys_c: list[tuple], keys_o: list[tuple]
) -> tuple[np.ndarray, np.ndarray]:
    """Método 1: asume orden relativo preservado; verifica contra df_outliers."""
    ids_c: list[int] = []
    ids_o: list[int] = []
    j = 0
    for i, key in enumerate(keys_u):
        if j < len(keys_c) and key == keys_c[j]:
            ids_c.append(i)
            j += 1
        else:
            ids_o.append(i)
    if j != len(keys_c):
        raise ValueError("two-pointer: no se emparejaron todas las filas limpias.")
    if [keys_u[i] for i in ids_o] != keys_o:
        raise ValueError("two-pointer: los outliers reconstruidos no coinciden.")
    return np.asarray(ids_c, dtype=int), np.asarray(ids_o, dtype=int)


def _content_match(
    unified: pd.DataFrame, clean: pd.DataFrame, outliers: pd.DataFrame,
    key_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Método 2 (fallback): match por (clave de fila, ocurrencia). tolera reordenamiento.

    Filas 100% idénticas son intercambiables: mismo texto -> mismo embedding.
    """
    u = pd.DataFrame({"_k": _row_keys(unified, key_cols)})
    u["doc_id"] = np.arange(len(u))
    u["_occ"] = u.groupby("_k").cumcount()

    o = pd.DataFrame({"_k": _row_keys(outliers, key_cols)})
    o["_occ"] = o.groupby("_k").cumcount()
    mo = o.merge(u, on=["_k", "_occ"], how="left", validate="one_to_one")
    if mo["doc_id"].isna().any():
        raise ValueError("match por contenido: hay outliers inexistentes en el corpus.")
    ids_o = mo["doc_id"].astype(int).to_numpy()

    rest = u[~u["doc_id"].isin(ids_o)].copy()
    rest["_occ"] = rest.groupby("_k").cumcount()
    c = pd.DataFrame({"_k": _row_keys(clean, key_cols)})
    c["_occ"] = c.groupby("_k").cumcount()
    mc = c.merge(rest, on=["_k", "_occ"], how="left", validate="one_to_one")
    if mc["doc_id"].isna().any():
        raise ValueError("match por contenido: hay filas limpias inexistentes en el corpus.")
    ids_c = mc["doc_id"].astype(int).to_numpy()
    return ids_c, ids_o


def assign_doc_ids(
    unified: pd.DataFrame, clean: pd.DataFrame, outliers: pd.DataFrame,
    key_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruye el doc_id original (índice en el corpus unificado filtrado).

    Returns:
        (doc_ids_clean, doc_ids_outliers) en el orden de las filas de cada DataFrame.
    Raises:
        ValueError: si tamaños no cuadran o la partición no cubre el corpus.
    """
    if len(clean) + len(outliers) != len(unified):
        raise ValueError(
            f"Tamaños inconsistentes: clean({len(clean)}) + outliers({len(outliers)}) "
            f"!= corpus({len(unified)}). ¿Los parámetros apuntan a otra corrida?"
        )
    try:
        ids_c, ids_o = _two_pointer_match(
            _row_keys(unified, key_cols),
            _row_keys(clean, key_cols),
            _row_keys(outliers, key_cols),
        )
    except ValueError:
        ids_c, ids_o = _content_match(unified, clean, outliers, key_cols)

    all_ids = np.concatenate([ids_c, ids_o])
    if len(np.unique(all_ids)) != len(unified):
        raise ValueError(
            "La partición clean/outliers no cubre exactamente el corpus unificado "
            "(doc_ids duplicados o faltantes). Abortando para no desalinear embeddings."
        )
    return ids_c, ids_o


# ─────────────────────────── Orquestación ───────────────────────────
def _require_cache_hit(meta: dict, step: str, allow_recompute: bool) -> None:
    """Fail-fast: la tarea base debe reutilizar la caché, no recalcular."""
    if not meta.get("cache_hit") and not allow_recompute:
        raise RuntimeError(
            f"El paso '{step}' NO tuvo cache hit. Los parámetros de run_comparison "
            "no coinciden con los del run original (revisa --outlier-method, "
            "--contamination, --text-col, rutas de CSV, etc.) o ejecuta desde code/. "
            "Usa --allow-recompute si el recálculo es intencional."
        )


def _annotate_outlier_sentiment(df_outliers: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Sentimiento de outliers (no está cacheado en el pipeline; solo ~88 docs)."""
    try:
        from sentiment_analysis import annotate_sentiment
        df = annotate_sentiment(
            df_outliers, text_column=args.text_col, source_column=SOURCE_COLUMN,
            lang=args.lang, backend="transformers", model_name=None,
        )
        return binarize_sentiment(df)
    except Exception as exc:
        print(f"  ⚠️ No se pudo anotar sentimiento de outliers ({exc}); se guarda NaN.")
        df = df_outliers.copy()
        df["sentiment_label"] = pd.NA
        df["sentiment_score"] = np.nan
        return df


def _print_summary(base: pd.DataFrame, out_base: pd.DataFrame, emb: np.ndarray) -> None:
    """Resumen legible en español + verificación de alineación."""
    print(f"\n  ✅ Alineación verificada: {len(base):,} documentos limpios ↔ "
          f"embeddings_clean {emb.shape}")
    print(f"  ✅ Outliers etiquetados: {len(out_base):,}")
    print("\n  Distribución por corpus:")
    for corpus, cnt in base["corpus"].value_counts().items():
        print(f"    {corpus:<10s} {cnt:>8,d}")
    print("\n  Distribución por fuente y sentimiento:")
    tab = base.groupby("source").agg(
        total=("doc_id", "size"),
        pos=("sentiment_label", lambda s: (s == "positive").sum()),
        neg=("sentiment_label", lambda s: (s == "negative").sum()),
    )
    tab["%pos"] = (100 * tab["pos"] / tab["total"]).round(1)
    print(tab.to_string())
    print("\n  Distribución global por sentimiento:")
    for lbl, cnt in base["sentiment_label"].value_counts().items():
        print(f"    {lbl:<10s} {cnt:>8,d} ({100 * cnt / len(base):.1f}%)")

def run_base_task(
    args: argparse.Namespace,
    results_dir: Path,
    *,
    inputs: dict | None = None,
) -> list[Path]:
    """Tarea 'base': consolida analysis_base.csv, embeddings_clean.npy, outliers_base.csv
    y run_metadata.csv.

    Args:
        inputs: objetos ya calculados por run_analysis.py (df unificado filtrado,
            embeddings, df_clean con sentimiento, df_outliers con sentimiento).
            Si se proporcionan, NO se re-lee la caché ni se re-anota sentimiento.
            Si es None, se reconstruye todo desde la caché del pipeline.
    """
    print_header("PASO 8: BASE ANALÍTICA CONSOLIDADA (COMPARACIÓN DMO vs UGC)")
    base_dir = results_dir / "comparison" / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    if inputs is not None:
        print("  ♻️ Usando objetos en memoria del pipeline (pasos 0–4); sin recargar caché.")
        df = inputs["df"]
        embeddings = np.asarray(inputs["embeddings"])
        df_sent = binarize_sentiment(inputs["df_clean"].reset_index(drop=True))
        df_out = inputs["df_outliers"].reset_index(drop=True)
        if "sentiment_label" in df_out.columns:
            df_out = binarize_sentiment(df_out)
        else:
            df_out["sentiment_label"] = pd.NA
            df_out["sentiment_score"] = np.nan
        if embeddings.shape[0] != len(df):
            raise ValueError(
                f"Embeddings ({embeddings.shape[0]}) != corpus filtrado ({len(df)})."
            )
        key_cols = [c for c in df.columns
                    if c in df_sent.columns and c in df_out.columns]
    else:
        from cache import (
            step_embeddings_bert, step_load_corpus, step_outliers_ngram, step_sentiment,
        )
        cache_dir = str(results_dir / "cache")
        df, load_meta = step_load_corpus(
            dmo_csv=args.csv_dmo, ugc_csvs=args.csv_ugc, text_column=args.text_col,
            source_column=SOURCE_COLUMN, dmo_label=DMO_LABEL,
            cache_dir=cache_dir, use_cache=True, force=False,
        )
        _require_cache_hit(load_meta, "load_corpus", args.allow_recompute)
        df = replicate_pipeline_filter(df, args.text_col)

        embeddings, emb_meta = step_embeddings_bert(
            df=df, text_column=args.text_col, model_name=args.embedding_model,
            cache_dir=cache_dir, parent_key=load_meta["key"], use_cache=True, force=False,
        )
        _require_cache_hit(emb_meta, "embeddings_bert", args.allow_recompute)
        if embeddings.shape[0] != len(df):
            raise ValueError("La caché de embeddings corresponde a otro corpus; abortando.")

        df_clean, df_outliers, _analysis, out_meta = step_outliers_ngram(
            df=df, embeddings=embeddings, text_column=args.text_col,
            source_column=SOURCE_COLUMN, method=args.outlier_method,
            contamination=args.contamination, per_source=args.per_source,
            top_n=args.top_n, min_samples_per_source=10,
            output_dir=str(results_dir / "outliers"),
            cache_dir=cache_dir, parent_key=emb_meta["key"], use_cache=True, force=False,
        )
        _require_cache_hit(out_meta, "outliers_ngram", args.allow_recompute)

        df_sent, sent_meta = step_sentiment(
            df_clean=df_clean, text_column=args.text_col, source_column=SOURCE_COLUMN,
            lang=args.lang, sentiment_backend="transformers", sentiment_threshold=0.0,
            sentiment_model=None, cache_dir=cache_dir, parent_key=out_meta["key"],
            use_cache=True, force=False,
        )
        _require_cache_hit(sent_meta, "sentiment", args.allow_recompute)
        df_sent = binarize_sentiment(df_sent)
        key_cols = [c for c in df.columns
                    if c in df_sent.columns and c in df_outliers.columns]
        if not df_sent[key_cols].reset_index(drop=True).equals(
            df_clean[key_cols].reset_index(drop=True)
        ):
            raise ValueError("df_sentiment no conserva el orden de df_clean; abortando.")
        df_out = _annotate_outlier_sentiment(df_outliers, args)

    ids_clean, ids_out = assign_doc_ids(df, df_sent, df_out, key_cols)

    # ── Corpus limpio ──
    base = pd.DataFrame({
        "doc_id": ids_clean,
        "text": df_sent[args.text_col].to_numpy(),
        "text_norm": (df_sent["text_norm"].to_numpy() if "text_norm" in df_sent
                      else df_sent[args.text_col].to_numpy()),
        "corpus": df_sent[CORPUS_COLUMN].to_numpy(),
        "source": df_sent[SOURCE_COLUMN].to_numpy(),
        "sentiment_label": df_sent["sentiment_label"].to_numpy(),
        "sentiment_score": df_sent["sentiment_score"].astype(float).to_numpy(),
        "is_outlier": False,
    }).sort_values("doc_id").reset_index(drop=True)
    base["embedding_row"] = np.arange(len(base), dtype=int)
    emb_clean = embeddings[base["doc_id"].to_numpy()]
    if len(base) != emb_clean.shape[0]:
        raise ValueError("Fallo de alineación final: analysis_base vs embeddings_clean.")

    # ── Outliers (análisis separado) ──
    out_base = pd.DataFrame({
        "doc_id": ids_out,
        "text": df_out[args.text_col].to_numpy(),
        "corpus": df_out[CORPUS_COLUMN].to_numpy(),
        "source": df_out[SOURCE_COLUMN].to_numpy(),
        "sentiment_label": df_out["sentiment_label"].to_numpy(),
        "sentiment_score": df_out["sentiment_score"].astype(float).to_numpy(),
        "is_outlier": True,
    }).sort_values("doc_id").reset_index(drop=True)
    out_base["embedding_row"] = np.arange(len(out_base), dtype=int)
    emb_out = embeddings[out_base["doc_id"].to_numpy()]

    # ── Artefactos ──
    written: list[Path] = []
    written.append(save_csv(base, base_dir / "analysis_base.csv"))
    np.save(base_dir / "embeddings_clean.npy", emb_clean)
    written.append(base_dir / "embeddings_clean.npy")
    written.append(save_csv(out_base, base_dir / "outliers_base.csv"))
    np.save(base_dir / "embeddings_outliers.npy", emb_out)
    written.append(base_dir / "embeddings_outliers.npy")
    written.append(append_metadata(
        base_dir / "run_metadata.csv", task="base",
        params={**vars(args), "source_of_truth": "in_memory" if inputs else "cache"},
        repo_dir=Path(__file__).resolve().parents[2],
    ))

    _print_summary(base, out_base, emb_clean)
    return written