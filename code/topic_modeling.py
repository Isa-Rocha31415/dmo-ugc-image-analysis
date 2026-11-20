from __future__ import annotations

import contextlib
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from topic_selection import select_num_topics
from xidoo_topic_modeler import XidooTopicModeler
from bertopic_modeler import BertopicModeler

# Backends de modelado de tópicos soportados.
SUPPORTED_BACKENDS = ("xidoo", "bertopic")


# ---------------------------------------------------------------------------
# Utilidades de frecuencia de palabras
# ---------------------------------------------------------------------------


def get_top_terms(
    texts: List[str],
    top_n: int = 30,
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 1,
) -> List[Tuple[str, int]]:
    """
    Devuelve los términos más frecuentes como:
      [(termino, conteo), ...]
    """
    cleaned_texts = [str(text) for text in texts if str(text).strip()]

    if not cleaned_texts:
        return []

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        lowercase=True,
        token_pattern=r"(?u)\b\w{2,}\b",
        min_df=min_df,
    )

    try:
        matrix = vectorizer.fit_transform(cleaned_texts)
    except ValueError:
        return []

    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    if len(terms) == 0:
        return []

    top_indices = np.argsort(counts)[::-1][:top_n]

    return [
        (terms[idx], int(counts[idx]))
        for idx in top_indices
        if counts[idx] > 0
    ]


def _word_frequency_output(
    subset: pd.DataFrame,
    text_column: str,
    sentiment_label: str,
    top_n: int = 30,
) -> Dict[str, Any]:
    """
    Salida para particiones pequeñas donde no se hace topic modeling,
    sino frecuencia de palabras.
    """
    texts = subset[text_column].fillna("").astype(str).tolist()

    unigrams = get_top_terms(texts, top_n=top_n, ngram_range=(1, 1))
    bigrams = get_top_terms(texts, top_n=top_n, ngram_range=(2, 2))
    trigrams = get_top_terms(texts, top_n=top_n, ngram_range=(3, 3))

    out_df = subset.copy()
    out_df["topic_id"] = "word_frequency"
    out_df["topic_name"] = f"{sentiment_label}: frecuencia de palabras"

    return {
        "method": "word_frequency",
        "sentiment_label": sentiment_label,
        "count": len(out_df),
        "df": out_df,
        "top_terms_unigram": unigrams,
        "top_terms_bigram": bigrams,
        "top_terms_trigram": trigrams,
        "keywords": {
            "word_frequency": [term for term, _ in unigrams]
        },
        "topic_names": {
            "word_frequency": f"{sentiment_label}: frecuencia de palabras"
        },
    }


# ---------------------------------------------------------------------------
# Modelado de tópicos por sentimiento
# ---------------------------------------------------------------------------


def _worker_fit(payload: tuple) -> tuple[Dict[str, Any], str, float]:
    """Ajusta una partición en el proceso hijo; devuelve resultado, logs y tiempo."""
    (sub, label, text_col, src_col, min_size, backend, use_internal,
     seed, top_n, part_emb, num_topics, knn, mult) = payload
    import numba
    import torch

    numba.set_num_threads(2)
    torch.set_num_threads(2)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        res = run_topic_modeling(
            sub, text_col, src_col, min_size, backend, use_internal,
            seed, top_n, part_emb, num_topics, labels=[label],
            parallel=False, precomputed_knn=knn, knn_k_mult=mult,
        )
    return res, buf.getvalue(), time.perf_counter() - t0


def run_topic_modeling(
    df: pd.DataFrame,
    text_column: str,
    source_column: str,
    min_partition_size: int = 50,
    topic_backend: str = "xidoo",
    use_internal_outlier: bool = False,
    random_state: int = 42,
    top_n_terms: int = 30,
    embeddings: Optional[np.ndarray] = None,
    num_topics: Any = "auto",
    labels: Optional[List[str]] = None,
    parallel: bool = True,
    precomputed_knn: Optional[tuple] = None,
    knn_k_mult: float = 1.3,
) -> Dict[str, Any]:
    """
    Aplica modelado de tópicos para comentarios positivos y negativos.

    Si una partición tiene menos de min_partition_size comentarios,
    usa frecuencia de palabras.

    Backends: 'xidoo' (XidooTopic) o 'bertopic' (BERTopic).

    Args:
        embeddings: Embeddings alineados fila a fila con `df` (opcional).
                    Si se proveen, BERTopic los reutiliza sin recalcular.
        num_topics: entero >= 1 para fijar el número de tópicos por
                    sentimiento, o ``"auto"``/``"automatico"`` para
                    seleccionarlo con `topic_selection.select_num_topics`.

    Devuelve:
      by_sentiment: diccionario con resultados para positive/negative
      topics_df: DataFrame combinado con topic_id, topic_name, etc.
    """
    if "sentiment_label" not in df.columns:
        raise ValueError(
            "El DataFrame debe tener sentiment_label. "
            "Primero ejecuta annotate_sentiment()."
        )

    if source_column not in df.columns:
        raise ValueError(f"No existe la columna de fuente: {source_column}")

    print(f"\n  🔧 Configuración de topic modeling:")
    print(f"     topic_backend: {topic_backend}")
    print(f"     min_partition_size: {min_partition_size}")
    print(f"     num_topicos: {num_topics}")

    labels = list(labels or ["positive", "negative"])
    if parallel and not os.environ.get("PERF2_KNN_FORCE"):
        print("  ⚠️ Puerta ARI<1: kNN interna en PASO 7 (grafo compartido inactivo).")
    if (parallel and len(labels) > 1 and embeddings is not None
            and not os.environ.get("PERF2_SERIAL")):
        graph = None
        if os.environ.get("PERF2_KNN_FORCE"):
            from knn_shared import build_global_knn, knn_k

            frac = min(float((df["sentiment_label"] == c).mean()) for c in labels)
            graph = build_global_knn(
                embeddings, knn_k(15, frac, knn_k_mult), random_state)
        payloads = []
        for label in labels:
            pos = np.flatnonzero((df["sentiment_label"] == label).to_numpy())
            knn = None
            if graph is not None:
                from knn_shared import subset_knn

                knn, cov = subset_knn(graph, pos, 15)
                print(f"  🔗 kNN {label}: cobertura min={cov.min_survivors}/15 "
                      f"({'ok' if cov.ok else 'fallback interno'})")
                if not cov.ok:
                    print("  ⚠️ Cobertura insuficiente: UMAP usa kNN interna.")
            payloads.append((
                df[df["sentiment_label"] == label].copy(), label, text_column,
                source_column, min_partition_size, topic_backend,
                use_internal_outlier, random_state, top_n_terms,
                embeddings[pos], num_topics, knn, knn_k_mult,
            ))
        print("  ⚡ PASO 7 en paralelo: 2 procesos, 2 hilos por proceso.")
        try:
            with ProcessPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(_worker_fit, payloads))
        except Exception as exc:
            print(f"  ⚠️ Fallo el pool paralelo ({exc}); reintento secuencial.")
            results = [_worker_fit(p) for p in payloads]
        outputs, frames = {}, []
        for label, (res, logs, secs) in zip(labels, results):
            print(f"\n  📊 {label} (paralelo {secs:.1f}s)")
            print(logs, end="")
            out = res["by_sentiment"].get(label, {})
            outputs[label] = out
            if "df" in out:
                frames.append(out["df"])
        topics_df = pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0].copy()
        if not topics_df.empty and "topic_id" in topics_df:
            topics_df["topic_id"] = topics_df["topic_id"].astype(str)
        return {"by_sentiment": outputs, "topics_df": topics_df}

    if os.environ.get("PERF2_SERIAL"):
        print("  🐢 PASO 7 secuencial (PERF2_SERIAL=1).")

    outputs: Dict[str, Any] = {}
    frames: List[pd.DataFrame] = []

    for sentiment_label in labels:
        subset = df[df["sentiment_label"] == sentiment_label].copy()
        print(f"\n  📊 Procesando {sentiment_label}: {len(subset)} comentarios")

        if subset.empty:
            outputs[sentiment_label] = {
                "method": "empty",
                "sentiment_label": sentiment_label,
                "count": 0,
                "message": "No hay comentarios para esta partición.",
            }
            continue

        # Verificar si usar word_frequency o el backend elegido
        use_word_freq = (
            len(subset) < min_partition_size
            or topic_backend not in SUPPORTED_BACKENDS
        )

        if use_word_freq:
            reason = (
                f"tamaño pequeño ({len(subset)} < {min_partition_size})"
                if len(subset) < min_partition_size
                else f"backend '{topic_backend}' no soportado"
            )
            print(f"     ⚠️ Usando word_frequency: {reason}")

            output = _word_frequency_output(
                subset=subset,
                text_column=text_column,
                sentiment_label=sentiment_label,
                top_n=top_n_terms,
            )
            output["note"] = f"Se usó frecuencia de palabras: {reason}"
            outputs[sentiment_label] = output

            if "df" in output:
                frames.append(output["df"])
            continue

        # Embeddings de esta particion (si el pipeline los proveyo)
        part_embeddings = None
        if embeddings is not None:
            positions = np.flatnonzero(
                (df["sentiment_label"] == sentiment_label).to_numpy()
            )
            part_embeddings = embeddings[positions]

        # Intentar el backend de modelado elegido
        if topic_backend == "bertopic":
            backend_label = "BERTopic"
            modeler = BertopicModeler(random_state=random_state)
        else:
            backend_label = "XidooTopic"
            modeler = XidooTopicModeler(
                use_internal_outlier=use_internal_outlier,
                random_state=random_state,
            )

        print(f"     ✅ Intentando {backend_label} con {len(subset)} comentarios...")
        try:
            fit_kwargs = (
                {"precomputed_knn": precomputed_knn}
                if topic_backend == "bertopic" and precomputed_knn is not None else {}
            )
            result = modeler.fit_transform(
                texts=subset[text_column].astype(str).tolist(),
                metadata=subset.copy(),
                embeddings=part_embeddings,
                **fit_kwargs,
            )

            out_df = result["df"].copy()
            base_keywords = result.get("keywords", {})
            base_topic_names = result.get("topic_names", {})
            selection_summary = None

            # Selección/fijado de k sobre la solución fina del backend.
            try:
                selected = select_num_topics(
                    texts=subset[text_column].astype(str).tolist(),
                    topics=out_df["topic_id"].astype(str).tolist(),
                    embeddings=part_embeddings,
                    keywords=base_keywords,
                    requested=num_topics,
                    random_state=random_state,
                )
                out_df["base_topic_id"] = out_df["topic_id"].astype(str)
                out_df["topic_id"] = (
                    out_df["topic_id"].astype(str)
                    .map(selected["mapping"])
                    .fillna(-1)
                    .astype(int)
                )
                out_df["topic_name"] = out_df["topic_id"].map(
                    selected["topic_names"]
                )
                out_df["topic_name"] = out_df["topic_name"].fillna(
                    "Topic -1: outliers"
                )
                base_keywords = selected["keywords"]
                base_topic_names = selected["topic_names"]
                selection_summary = selected["summary"]
                print(
                    f"     ✅ Selección de k: {selection_summary['selected_k']} "
                    f"tópicos (+ {selection_summary['n_outliers']} outliers) "
                    f"desde {selection_summary['base_topics']} tópicos base"
                )
            except Exception as exc:
                print(f"     ⚠️ Selección de número de tópicos omitida: {exc}")

            out_df["sentiment_label"] = sentiment_label
            out_df["sentiment_topic_id"] = (
                sentiment_label + "_" + out_df["topic_id"].astype(str)
            )

            if "topic_name" not in out_df:
                out_df["topic_name"] = out_df["topic_id"].map(
                    lambda tid: base_topic_names.get(tid, f"Topic {tid}")
                )

            out_df["topic_name"] = (
                sentiment_label + ": " + out_df["topic_name"].astype(str)
            )

            n_topics = len(base_topic_names)
            print(f"     ✅ {backend_label} exitoso: {n_topics} tópicos identificados")

            outputs[sentiment_label] = {
                "method": topic_backend,
                "sentiment_label": sentiment_label,
                "count": len(out_df),
                "df": out_df,
                "keywords": base_keywords,
                "topic_names": base_topic_names,
                "embeddings": result.get("embeddings", None),
                "topic_selection": selection_summary,
            }

            frames.append(out_df)

        except Exception as exc:
            print(f"     ❌ {backend_label} falló: {exc}")
            print(f"     ⚠️ Cayendo a word_frequency como fallback")

            output = _word_frequency_output(
                subset=subset,
                text_column=text_column,
                sentiment_label=sentiment_label,
                top_n=top_n_terms,
            )
            output["warning"] = f"{backend_label} falló y se usó fallback. Error: {exc}"
            outputs[sentiment_label] = output

            if "df" in output:
                frames.append(output["df"])

    # Combinar todos los resultados
    if frames:
        topics_df = pd.concat(frames, ignore_index=True)
    else:
        topics_df = df.iloc[0:0].copy()

    if not topics_df.empty:
        if "topic_id" in topics_df:
            topics_df["topic_id"] = topics_df["topic_id"].astype(str)

        if "topic_name" not in topics_df:
            topics_df["topic_name"] = "Topic " + topics_df["topic_id"].astype(str)

    return {
        "by_sentiment": outputs,
        "topics_df": topics_df,
    }


# ---------------------------------------------------------------------------
# Resúmenes para visualización / comparación
# ---------------------------------------------------------------------------


def summarize_topics_by_source(
    topics_df: pd.DataFrame,
    source_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - conteos de tópicos por fuente
      - porcentajes de tópicos por fuente
    """
    if topics_df.empty or "topic_name" not in topics_df:
        empty = pd.DataFrame()
        return empty, empty

    counts = (
        topics_df.groupby([source_column, "topic_name"])
        .size()
        .unstack(fill_value=0)
    )

    percentages = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100

    return counts, percentages


def topic_keywords_to_dataframe(keywords: Dict[Any, List[str]]) -> pd.DataFrame:
    """
    Convierte keywords por tópico en DataFrame.
    """
    rows = []

    for topic_id, terms in keywords.items():
        rows.append(
            {
                "topic_id": str(topic_id),
                "keywords": ", ".join([str(term) for term in terms]),
            }
        )

    return pd.DataFrame(rows)