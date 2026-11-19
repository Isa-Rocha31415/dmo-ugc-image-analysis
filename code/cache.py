import os
import json
import hashlib
import shutil
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:
    nltk_stopwords = None

from text_preprocessing import preprocess_dataframe
from outlier_detector import detectar_y_analizar_outliers
from sentiment_analysis import annotate_sentiment
from topic_modeling import run_topic_modeling, summarize_topics_by_source
import visualization as viz


# =============================================================================
# Versiones de caché
# =============================================================================
CACHE_VERSION = 1
PREPROCESS_VERSION = 1
EMBEDDINGS_VERSION = 1

# Se sube versión porque ahora outliers también guarda embeddings limpios.
OUTLIER_VERSION = 2

SENTIMENT_VERSION = 1
# v2: selección/fijado de número de tópicos (num_topicos).
TOPICS_VERSION = 2
VISUALS_VERSION = 1


# =============================================================================
# Fallback de embeddings
# =============================================================================
def generar_embeddings_fallback(textos: List[str]) -> np.ndarray:
    """
    Fallback de embeddings usando TF-IDF cuando el generador principal
    no está disponible.
    """
    spanish_stopwords = None

    if nltk_stopwords is not None:
        try:
            spanish_stopwords = list(nltk_stopwords.words("spanish"))
        except Exception:
            spanish_stopwords = None

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=spanish_stopwords,
    )

    return vectorizer.fit_transform(textos).toarray()


# =============================================================================
# Utilidades generales
# =============================================================================
def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ensure_dir(path: str) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _hash_obj(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _force_step(step_name: str, force_steps: set) -> bool:
    return ("all" in force_steps) or (step_name in force_steps)


def _file_fingerprint(path: str) -> dict:
    """
    Genera una huella del archivo CSV.
    """
    path = str(path)
    st = os.stat(path)
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return {
        "path": os.path.abspath(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "sha256": sha.hexdigest(),
    }


def _manifest_path(step_dir: str) -> str:
    return str(Path(step_dir) / "manifest.json")


def _load_manifest(step_dir: str) -> Optional[dict]:
    path = _manifest_path(step_dir)

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_manifest(step_dir: str, manifest: dict) -> None:
    _ensure_dir(step_dir)

    with open(_manifest_path(step_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)


def _load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(str(path), compression="gzip")


def _save_df(df: pd.DataFrame, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=False, compression="gzip")


def _load_json(path: str) -> dict:
    path = str(path)

    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(obj: dict, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _copy_tree(src: str, dst: str) -> None:
    src = str(src)
    dst = str(dst)

    if not os.path.isdir(src):
        return

    _ensure_dir(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _step_meta(
    step: str,
    key: str,
    step_dir: str,
    cache_hit: bool,
    extra: Optional[dict] = None,
) -> dict:
    meta = {
        "step": step,
        "key": key,
        "dir": step_dir,
        "cache_hit": bool(cache_hit),
        "timestamp": _now_iso(),
    }

    if extra:
        meta.update(extra)

    return meta


def _pick_text_column(df: pd.DataFrame, default_text_column: str) -> str:
    """
    Prioriza processed_text si existe.
    """
    if "processed_text" in df.columns:
        return "processed_text"

    if default_text_column in df.columns:
        return default_text_column

    raise KeyError(
        f"No existe columna de texto válida. "
        f"Se esperaba '{default_text_column}' o 'processed_text'."
    )


def _topic_outputs_for_json(topic_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quita DataFrames de topic_outputs para poder guardar el resumen en JSON.
    """
    summary = {}

    by_sentiment = topic_outputs.get("by_sentiment", {})

    for sentiment_label, output in by_sentiment.items():
        if not isinstance(output, dict):
            summary[sentiment_label] = {"method": "unknown"}
            continue

        output_copy = {
            key: value
            for key, value in output.items()
            if key not in ("df", "embeddings")
        }

        summary[sentiment_label] = output_copy

    return summary


def _safe_plot(name: str, fn, *args, **kwargs):
    """
    Wrapper para que un gráfico falle no tumbe todo el pipeline.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"⚠️ No se pudo generar el gráfico [{name}]: {exc}")
        return None


# =============================================================================
# Paso 1: Load
# =============================================================================
def step_load(
    input_csv: str,
    cache_dir: str,
    use_cache: bool = True,
    force: bool = False,
):
    step_name = "load"

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "csv": os.path.abspath(input_csv),
        "file": _file_fingerprint(input_csv),
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)
    artifact = str(Path(step_dir) / "data.csv.gz")
    manifest = _load_manifest(step_dir)

    if use_cache and not force and manifest and os.path.exists(artifact):
        df = _load_df(artifact)
        print(f"✅ Caché hit [{step_name}] -> {step_dir}")

        return df, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "rows": int(len(df)),
                "artifact": artifact,
            },
        )

    print(f"🧮 Ejecutando [{step_name}]...")

    df = pd.read_csv(input_csv)
    _save_df(df, artifact)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {"data": artifact},
        "rows": int(len(df)),
    }

    _save_manifest(step_dir, manifest)

    return df, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "rows": int(len(df)),
            "artifact": artifact,
        },
    )


# =============================================================================
# Paso 2: Preprocess
# =============================================================================
def step_preprocess(
    df: pd.DataFrame,
    text_column: str,
    lang: str,
    cache_dir: str,
    parent_key: str,
    use_cache: bool = True,
    force: bool = False,
):
    step_name = "preprocess"

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "parent_key": parent_key,
        "text_column": text_column,
        "lang": lang,
        "input_rows": int(len(df)),
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)
    artifact = str(Path(step_dir) / "data.csv.gz")
    manifest = _load_manifest(step_dir)

    if use_cache and not force and manifest and os.path.exists(artifact):
        df_proc = _load_df(artifact)
        emb_model = manifest.get("emb_model")

        print(f"✅ Caché hit [{step_name}] -> {step_dir}")

        return df_proc, emb_model, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "rows": int(len(df_proc)),
                "emb_model": emb_model,
                "artifact": artifact,
            },
        )

    print(f"🧮 Ejecutando [{step_name}]...")

    df_proc, emb_model = preprocess_dataframe(
        df=df,
        text_column=text_column,
        lang=lang,
    )

    if df_proc.empty:
        raise ValueError("No quedaron textos válidos después del preprocesamiento.")

    _save_df(df_proc, artifact)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {"data": artifact},
        "rows": int(len(df_proc)),
        "emb_model": str(emb_model),
    }

    _save_manifest(step_dir, manifest)

    return df_proc, emb_model, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "rows": int(len(df_proc)),
            "emb_model": str(emb_model),
            "artifact": artifact,
        },
    )


# =============================================================================
# Paso 3: Embeddings
# =============================================================================
def step_embeddings(
    df_proc: pd.DataFrame,
    emb_model: str,
    cache_dir: str,
    parent_key: str,
    embedding_backend: str = "auto",
    use_cache: bool = True,
    force: bool = False,
):
    step_name = "embeddings"

    generator_fn = None

    try:
        from embeddings_generator import generate_embeddings
        generator_fn = generate_embeddings
    except Exception:
        generator_fn = None

    if embedding_backend == "fallback":
        backend = "tfidf_fallback"

    elif embedding_backend == "generator":
        if generator_fn is None:
            raise ImportError(
                "No se pudo importar embeddings_generator. "
                "Usa --embedding-backend fallback."
            )
        backend = "embeddings_generator"

    else:
        backend = "embeddings_generator" if generator_fn is not None else "tfidf_fallback"

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "embeddings_version": EMBEDDINGS_VERSION,
        "parent_key": parent_key,
        "backend": backend,
        "emb_model": str(emb_model),
        "rows": int(len(df_proc)),
        "embedding_backend_requested": embedding_backend,
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)
    artifact = str(Path(step_dir) / "embeddings.npy")
    manifest = _load_manifest(step_dir)

    if use_cache and not force and manifest and os.path.exists(artifact):
        embeddings = np.load(artifact)

        print(f"✅ Caché hit [{step_name}] -> {step_dir}")

        return embeddings, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "rows": int(embeddings.shape[0]),
                "shape": list(embeddings.shape),
                "backend": manifest.get("backend", backend),
                "artifact": artifact,
            },
        )

    print(f"🧮 Ejecutando [{step_name}] con backend [{backend}]...")
    _ensure_dir(step_dir)

    if backend == "embeddings_generator":
        embeddings = generator_fn(
            df_proc,
            text_column="processed_text",
            model_name=emb_model,
            output_path=artifact,
        )

        if embeddings is None:
            embeddings = np.load(artifact)
        else:
            if hasattr(embeddings, "cpu"):
                embeddings = embeddings.cpu()

            if hasattr(embeddings, "numpy"):
                embeddings = embeddings.numpy()

            if hasattr(embeddings, "toarray"):
                embeddings = embeddings.toarray()

            embeddings = np.asarray(embeddings)
            np.save(artifact, embeddings)

    else:
        textos = df_proc["processed_text"].astype(str).tolist()
        embeddings = generar_embeddings_fallback(textos)
        np.save(artifact, embeddings)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {"embeddings": artifact},
        "backend": backend,
        "emb_model": str(emb_model),
        "shape": list(embeddings.shape),
    }

    _save_manifest(step_dir, manifest)

    return embeddings, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "rows": int(embeddings.shape[0]),
            "shape": list(embeddings.shape),
            "backend": backend,
            "artifact": artifact,
        },
    )


# =============================================================================
# Paso 4: Outliers
# =============================================================================
def _align_embeddings_after_outliers(
    df_with_id: pd.DataFrame,
    df_clean: pd.DataFrame,
    embeddings: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Alinea los embeddings originales con el DataFrame limpio después de outliers.

    La estrategia principal usa una columna auxiliar `_row_id`.
    """
    if embeddings is None:
        return None

    if len(df_clean) == 0:
        return np.empty((0, embeddings.shape[1]), dtype=embeddings.dtype)

    # Caso ideal: df_clean conserva _row_id.
    if "_row_id" in df_clean.columns:
        row_ids = df_clean["_row_id"].astype(int).to_numpy()

        if len(row_ids) > 0 and row_ids.min() >= 0 and row_ids.max() < len(embeddings):
            return embeddings[row_ids]

    # Fallback: usar índice original si parece confiable.
    idx = df_clean.index

    if pd.api.types.is_integer_dtype(idx) and idx.is_unique:
        idx_arr = idx.to_numpy(dtype=int)

        if len(idx_arr) == len(embeddings):
            return embeddings[idx_arr]

        if len(idx_arr) < len(embeddings):
            # Si el índice es 0,1,2,...,n-1 después de filtrar, no es confiable.
            # Solo lo usamos si parece conservar índices originales no contiguos.
            if not np.array_equal(idx_arr, np.arange(len(idx_arr))):
                if len(idx_arr) > 0 and idx_arr.min() >= 0 and idx_arr.max() < len(embeddings):
                    return embeddings[idx_arr]

    if len(df_clean) == len(embeddings):
        return embeddings

    raise ValueError(
        "No fue posible alinear embeddings con df_clean después de outliers. "
        "Asegúrate de que tu detector de outliers conserve la columna auxiliar "
        "`_row_id`, o devuelve una máscara/índices limpios."
    )


def step_outliers(
    df_proc: pd.DataFrame,
    embeddings: np.ndarray,
    source_column: str,
    dmo_label: str,
    outlier_method: str,
    contamination: float,
    palette: str,
    final_results_dir: str,
    cache_dir: str,
    parent_key: str,
    use_cache: bool = True,
    force: bool = False,
):
    step_name = "outliers"

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "outlier_version": OUTLIER_VERSION,
        "parent_key": parent_key,
        "source_column": source_column,
        "dmo_label": dmo_label,
        "method": outlier_method,
        "contamination": float(contamination),
        "palette": palette,
        "vision_type": "normal",
        "rows": int(len(df_proc)),
        "embedding_shape": list(embeddings.shape),
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)

    clean_artifact = str(Path(step_dir) / "df_clean.csv.gz")
    embeddings_clean_artifact = str(Path(step_dir) / "embeddings_clean.npy")
    analysis_artifact = str(Path(step_dir) / "analysis.json")
    cached_plots_dir = str(Path(step_dir) / "plots")

    manifest = _load_manifest(step_dir)

    if use_cache and not force and manifest and os.path.exists(clean_artifact):
        df_clean = _load_df(clean_artifact)

        embeddings_clean = None
        if os.path.exists(embeddings_clean_artifact):
            embeddings_clean = np.load(embeddings_clean_artifact)

        analysis = _load_json(analysis_artifact).get("analysis", {})

        _copy_tree(cached_plots_dir, final_results_dir)

        print(f"✅ Caché hit [{step_name}] -> {step_dir}")

        return df_clean, embeddings_clean, analysis, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "rows_clean": int(len(df_clean)),
                "method": outlier_method,
                "palette": palette,
                "clean_artifact": clean_artifact,
                "embeddings_clean_artifact": embeddings_clean_artifact,
                "analysis_artifact": analysis_artifact,
                "plots_dir": cached_plots_dir,
            },
        )

    print(f"🧮 Ejecutando [{step_name}]...")
    _ensure_dir(cached_plots_dir)

    # Agregamos un identificador de fila para poder alinear embeddings después.
    df_with_id = df_proc.copy()
    df_with_id["_row_id"] = np.arange(len(df_with_id))

    df_clean, analisis_ngramas = detectar_y_analizar_outliers(
        df=df_with_id,
        embeddings=embeddings,
        texto_col="processed_text",
        fuente_col=source_column,
        dmo_value=dmo_label,
        metodo=outlier_method,
        contamination=contamination,
        results_dir=cached_plots_dir,
        palette_name=palette,
        vision_type="normal",
    )

    embeddings_clean = _align_embeddings_after_outliers(
        df_with_id=df_with_id,
        df_clean=df_clean,
        embeddings=embeddings,
    )

    # Limpiamos columna auxiliar y dejamos índice consistente.
    df_clean = df_clean.drop(columns=["_row_id"], errors="ignore")
    df_clean = df_clean.reset_index(drop=True)

    _save_df(df_clean, clean_artifact)

    if embeddings_clean is not None:
        np.save(embeddings_clean_artifact, embeddings_clean)

    _save_json({"analysis": analisis_ngramas}, analysis_artifact)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {
            "df_clean": clean_artifact,
            "embeddings_clean": embeddings_clean_artifact,
            "analysis": analysis_artifact,
            "plots": cached_plots_dir,
        },
        "rows_clean": int(len(df_clean)),
        "method": outlier_method,
        "palette": palette,
    }

    _save_manifest(step_dir, manifest)
    _copy_tree(cached_plots_dir, final_results_dir)

    return df_clean, embeddings_clean, analisis_ngramas, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "rows_clean": int(len(df_clean)),
            "method": outlier_method,
            "palette": palette,
            "clean_artifact": clean_artifact,
            "embeddings_clean_artifact": embeddings_clean_artifact,
            "analysis_artifact": analysis_artifact,
            "plots_dir": cached_plots_dir,
        },
    )


# =============================================================================
# Paso 5: Sentiment
# =============================================================================
def step_sentiment(
    df_clean: pd.DataFrame,
    text_column: str,
    source_column: str,
    lang: str,
    sentiment_backend: str,
    sentiment_threshold: float,
    sentiment_model: Optional[str],
    cache_dir: str,
    parent_key: str,
    use_cache: bool = True,
    force: bool = False,
):
    step_name = "sentiment"

    sentiment_text_column = _pick_text_column(df_clean, text_column)

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "sentiment_version": SENTIMENT_VERSION,
        "parent_key": parent_key,
        "text_column": sentiment_text_column,
        "source_column": source_column,
        "lang": lang,
        "backend": sentiment_backend,
        "threshold": float(sentiment_threshold),
        "model_name": str(sentiment_model or ""),
        "rows": int(len(df_clean)),
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)
    artifact = str(Path(step_dir) / "df_sentiment.csv.gz")
    manifest = _load_manifest(step_dir)

    if use_cache and not force and manifest and os.path.exists(artifact):
        df_sentiment = _load_df(artifact)

        print(f"✅ Caché hit [{step_name}] -> {step_dir}")

        return df_sentiment, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "rows": int(len(df_sentiment)),
                "artifact": artifact,
            },
        )

    print(f"🧮 Ejecutando [{step_name}] con backend [{sentiment_backend}]...")

    df_sentiment = annotate_sentiment(
        df=df_clean,
        text_column=sentiment_text_column,
        source_column=source_column,
        lang=lang,
        backend=sentiment_backend,
        threshold=sentiment_threshold,
        model_name=sentiment_model,
        cache_dir=cache_dir,
    )

    _save_df(df_sentiment, artifact)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {"df_sentiment": artifact},
        "rows": int(len(df_sentiment)),
    }

    _save_manifest(step_dir, manifest)

    return df_sentiment, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "rows": int(len(df_sentiment)),
            "artifact": artifact,
        },
    )


# =============================================================================
# Paso 6: Topics
# =============================================================================
def step_topics(
    df_sentiment: pd.DataFrame,
    text_column: str,
    source_column: str,
    min_partition_size: int,
    topic_backend: str,
    use_xidoo_outliers: bool,
    random_state: int,
    cache_dir: str,
    parent_key: str,
    use_cache: bool = True,
    force: bool = False,
    embeddings: Optional[np.ndarray] = None,
    num_topics: Any = "auto",
):
    step_name = "topics"

    from topic_selection import parse_num_topics

    num_topics = parse_num_topics(num_topics)
    topic_text_column = _pick_text_column(df_sentiment, text_column)

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "topics_version": TOPICS_VERSION,
        "parent_key": parent_key,
        "text_column": topic_text_column,
        "source_column": source_column,
        "min_partition_size": int(min_partition_size),
        "topic_backend": topic_backend,
        "use_xidoo_outliers": bool(use_xidoo_outliers),
        "random_state": int(random_state),
        "num_topics": str(num_topics),
        "rows": int(len(df_sentiment)),
        "embeddings_shape": list(embeddings.shape) if embeddings is not None else None,
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)

    topics_artifact = str(Path(step_dir) / "topics_df.csv.gz")
    summary_artifact = str(Path(step_dir) / "topic_outputs_summary.json")

    manifest = _load_manifest(step_dir)

    if (
        use_cache
        and not force
        and manifest
        and os.path.exists(topics_artifact)
        and os.path.exists(summary_artifact)
    ):
        topics_df = _load_df(topics_artifact)
        summary = _load_json(summary_artifact)

        topic_outputs = {
            "by_sentiment": summary.get("by_sentiment", {}),
            "topics_df": topics_df,
        }

        print(f"✅ Caché hit [{step_name}] -> {step_dir}")

        return topic_outputs, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "rows_topics": int(len(topics_df)),
                "topics_artifact": topics_artifact,
                "summary_artifact": summary_artifact,
            },
        )

    print(f"🧮 Ejecutando [{step_name}] con backend [{topic_backend}]...")

    topic_outputs = run_topic_modeling(
        df=df_sentiment,
        text_column=topic_text_column,
        source_column=source_column,
        min_partition_size=min_partition_size,
        topic_backend=topic_backend,
        use_internal_outlier=use_xidoo_outliers,
        random_state=random_state,
        embeddings=embeddings,
        num_topics=num_topics,
    )

    topics_df = topic_outputs.get("topics_df", pd.DataFrame())

    _save_df(topics_df, topics_artifact)

    summary = {
        "by_sentiment": _topic_outputs_for_json(topic_outputs),
    }

    _save_json(summary, summary_artifact)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {
            "topics_df": topics_artifact,
            "summary": summary_artifact,
        },
        "rows_topics": int(len(topics_df)),
    }

    _save_manifest(step_dir, manifest)

    return topic_outputs, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "rows_topics": int(len(topics_df)),
            "topics_artifact": topics_artifact,
            "summary_artifact": summary_artifact,
        },
    )


# =============================================================================
# Paso 7: Visuals
# =============================================================================
def _generate_visuals(
    df_sentiment: pd.DataFrame,
    embeddings_clean: Optional[np.ndarray],
    topic_outputs: Dict[str, Any],
    topics_df: pd.DataFrame,
    source_column: str,
    dmo_label: str,
    palette: str,
    text_column: str,
    umap_method: str,
    max_plot_points: int,
    plots_dir: Path,
) -> None:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    hover_text_column = None

    if text_column in df_sentiment.columns:
        hover_text_column = text_column
    elif "processed_text" in df_sentiment.columns:
        hover_text_column = "processed_text"

    # -----------------------------------------------------------------------
    # 1. Espacio semántico
    # -----------------------------------------------------------------------
    if embeddings_clean is not None and len(df_sentiment) == len(embeddings_clean):
        _safe_plot(
            "UMAP por fuente",
            viz.plot_embedding_space,
            df=df_sentiment,
            embeddings=embeddings_clean,
            color_col=source_column,
            title="Espacio semántico por fuente",
            output_html=plots_dir / "07_umap_por_fuente.html",
            palette=palette,
            text_column=hover_text_column,
            max_points=max_plot_points,
            method=umap_method,
        )

        if "sentiment_label" in df_sentiment.columns:
            _safe_plot(
                "UMAP por sentimiento",
                viz.plot_embedding_space,
                df=df_sentiment,
                embeddings=embeddings_clean,
                color_col="sentiment_label",
                title="Espacio semántico por sentimiento",
                output_html=plots_dir / "08_umap_por_sentimiento.html",
                palette=palette,
                text_column=hover_text_column,
                max_points=max_plot_points,
                method=umap_method,
            )
    else:
        print("⚠️ No hay embeddings limpios alineados. Se omiten gráficos UMAP.")

    # -----------------------------------------------------------------------
    # 2. Sentimiento
    # -----------------------------------------------------------------------
    if "sentiment_label" in df_sentiment.columns:
        _safe_plot(
            "Sentimiento por fuente",
            viz.plot_sentiment_by_source,
            df=df_sentiment,
            source_column=source_column,
            output_html=plots_dir / "09_sentimiento_por_fuente.html",
            palette=palette,
        )

        _safe_plot(
            "Distribución de sentimiento",
            viz.plot_sentiment_score_distribution,
            df=df_sentiment,
            source_column=source_column,
            output_html=plots_dir / "10_distribucion_sentimiento.html",
            palette=palette,
        )

    # -----------------------------------------------------------------------
    # 3. Tópicos
    # -----------------------------------------------------------------------
    if topics_df is not None and not topics_df.empty:
        _safe_plot(
            "Distribución de tópicos por fuente",
            viz.plot_topic_distribution_by_source,
            topics_df=topics_df,
            source_column=source_column,
            output_html=plots_dir / "11_topicos_por_fuente.html",
            palette=palette,
            title="Distribución de tópicos por fuente",
        )

        _safe_plot(
            "Sunburst fuente-sentimiento-tópico",
            viz.plot_topic_sentiment_sunburst,
            topics_df=topics_df,
            source_column=source_column,
            output_html=plots_dir / "12_sunburst_fuente_sentimiento_topico.html",
            palette=palette,
        )

        _safe_plot(
            "Brecha DMO vs percibido",
            viz.plot_dmo_vs_perceived_topic_gap,
            topics_df=topics_df,
            source_column=source_column,
            dmo_label=dmo_label,
            output_html=plots_dir / "13_brecha_topicos_dmo_vs_percibido.html",
            palette=palette,
        )

        if (
            embeddings_clean is not None
            and "topic_name" in topics_df.columns
            and len(topics_df) == len(embeddings_clean)
        ):
            _safe_plot(
                "UMAP por tópico",
                viz.plot_embedding_space,
                df=topics_df,
                embeddings=embeddings_clean,
                color_col="topic_name",
                title="Espacio semántico por tópico",
                output_html=plots_dir / "14_umap_por_topico.html",
                palette=palette,
                text_column=hover_text_column,
                max_points=max_plot_points,
                method=umap_method,
            )

    # -----------------------------------------------------------------------
    # 4. Keywords / fallback de frecuencia
    # -----------------------------------------------------------------------
    by_sentiment = topic_outputs.get("by_sentiment", {})

    for sentiment_label, output in by_sentiment.items():
        if not isinstance(output, dict):
            continue

        method = output.get("method")

        if method == "word_frequency":
            unigrams = output.get("top_terms_unigram", [])
            bigrams = output.get("top_terms_bigram", [])
            trigrams = output.get("top_terms_trigram", [])

            if unigrams:
                _safe_plot(
                    f"Unigramas {sentiment_label}",
                    viz.plot_top_terms_bar,
                    terms=unigrams,
                    title=f"Unigramas frecuentes - {sentiment_label}",
                    output_html=plots_dir / f"15_unigramas_{sentiment_label}.html",
                    palette=palette,
                )

            if bigrams:
                _safe_plot(
                    f"Bigramas {sentiment_label}",
                    viz.plot_top_terms_bar,
                    terms=bigrams,
                    title=f"Bigramas frecuentes - {sentiment_label}",
                    output_html=plots_dir / f"16_bigramas_{sentiment_label}.html",
                    palette=palette,
                )

            if trigrams:
                _safe_plot(
                    f"Trigramas {sentiment_label}",
                    viz.plot_top_terms_bar,
                    terms=trigrams,
                    title=f"Trigramas frecuentes - {sentiment_label}",
                    output_html=plots_dir / f"17_trigramas_{sentiment_label}.html",
                    palette=palette,
                )

        elif method == "xidoo":
            keywords = output.get("keywords", {})

            if keywords:
                _safe_plot(
                    f"Keywords {sentiment_label}",
                    viz.plot_topic_keywords_table,
                    keywords=keywords,
                    output_html=plots_dir / f"18_keywords_topicos_{sentiment_label}.html",
                    title=f"Palabras clave por tópico - {sentiment_label}",
                    palette=palette,
                )


def step_visuals(
    df_sentiment: pd.DataFrame,
    embeddings_clean: Optional[np.ndarray],
    topic_outputs: Dict[str, Any],
    topics_df: pd.DataFrame,
    source_column: str,
    dmo_label: str,
    palette: str,
    text_column: str,
    umap_method: str,
    max_plot_points: int,
    final_results_dir: str,
    cache_dir: str,
    parent_key: str,
    use_cache: bool = True,
    force: bool = False,
):
    step_name = "visuals"

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "visuals_version": VISUALS_VERSION,
        "parent_key": parent_key,
        "rows": int(len(df_sentiment)),
        "embeddings_shape": list(embeddings_clean.shape) if embeddings_clean is not None else None,
        "palette": palette,
        "umap_method": umap_method,
        "max_plot_points": int(max_plot_points),
        "has_topics": bool(topics_df is not None and not topics_df.empty),
        "has_sentiment": bool("sentiment_label" in df_sentiment.columns),
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)
    cached_plots_dir = str(Path(step_dir) / "plots")

    manifest = _load_manifest(step_dir)

    if (
        use_cache
        and not force
        and manifest
        and os.path.isdir(cached_plots_dir)
        and any(os.scandir(cached_plots_dir))
    ):
        _copy_tree(cached_plots_dir, final_results_dir)

        print(f"✅ Caché hit [{step_name}] -> {step_dir}")

        return _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "plots_dir": cached_plots_dir,
            },
        )

    print(f"🧮 Ejecutando [{step_name}]...")
    _ensure_dir(cached_plots_dir)

    _generate_visuals(
        df_sentiment=df_sentiment,
        embeddings_clean=embeddings_clean,
        topic_outputs=topic_outputs,
        topics_df=topics_df,
        source_column=source_column,
        dmo_label=dmo_label,
        palette=palette,
        text_column=text_column,
        umap_method=umap_method,
        max_plot_points=max_plot_points,
        plots_dir=Path(cached_plots_dir),
    )

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {
            "plots": cached_plots_dir,
        },
    }

    _save_manifest(step_dir, manifest)
    _copy_tree(cached_plots_dir, final_results_dir)

    return _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "plots_dir": cached_plots_dir,
        },
    )


# =============================================================================
# Pipeline principal
# =============================================================================
def run_pipeline(
    input_csv: str,
    text_column: str,
    source_column: str,
    dmo_label: str,
    lang: str,
    title: str,
    palette: str,
    output_dir: str = "results",
    outlier_method: str = "LOF",
    contamination: float = 0.05,
    n_topics: Optional[int] = None,
    embedding_backend: str = "auto",
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    force_steps: Optional[List[str]] = None,
    topic_backend: str = "xidoo",
    sentiment_backend: str = "auto",
    sentiment_model: Optional[str] = None,
    sentiment_threshold: float = 0.0,
    min_partition_size: int = 50,
    use_xidoo_outliers: bool = False,
    umap_method: str = "umap",
    max_plot_points: int = 12000,
    cli_args: Optional[dict] = None,
) -> str:
    force_steps = set(force_steps or [])

    output_dir_path = Path(_ensure_dir(output_dir))
    results_dir = output_dir_path / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_dir_path = Path(cache_dir) if cache_dir else output_dir_path / "cache"
    cache_dir_str = _ensure_dir(str(cache_dir_path))

    history_path = output_dir_path / "history.jsonl"
    latest_run_path = output_dir_path / "latest_run.json"

    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    steps_meta = []

    # -----------------------------------------------------------------------
    # 1. Load
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\n[1/7] Cargando CSV unificado: {input_csv}\n{'='*60}")

    df, load_meta = step_load(
        input_csv=input_csv,
        cache_dir=cache_dir_str,
        use_cache=use_cache,
        force=_force_step("load", force_steps),
    )

    steps_meta.append(load_meta)

    df_dmo = df[df[source_column] == dmo_label].copy()
    df_ugc = df[df[source_column] != dmo_label].copy()

    print(f"DMO ({dmo_label}): {len(df_dmo):,} rows | UGC: {len(df_ugc):,} rows")

    # -----------------------------------------------------------------------
    # 2. Preprocess
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\n[2/7] Preprocesando textos (idioma: {lang})...\n{'='*60}")

    df_proc, emb_model, preprocess_meta = step_preprocess(
        df=df,
        text_column=text_column,
        lang=lang,
        cache_dir=cache_dir_str,
        parent_key=load_meta["key"],
        use_cache=use_cache,
        force=_force_step("preprocess", force_steps),
    )

    steps_meta.append(preprocess_meta)

    if df_proc.empty:
        raise ValueError("No quedaron textos válidos después del preprocesamiento.")

    print(f"Textos válidos tras preprocesamiento: {len(df_proc):,}")

    # -----------------------------------------------------------------------
    # 3. Embeddings
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\n[3/7] Generando embeddings...\n{'='*60}")

    embeddings, embeddings_meta = step_embeddings(
        df_proc=df_proc,
        emb_model=emb_model,
        cache_dir=cache_dir_str,
        parent_key=preprocess_meta["key"],
        embedding_backend=embedding_backend,
        use_cache=use_cache,
        force=_force_step("embeddings", force_steps),
    )

    steps_meta.append(embeddings_meta)

    print(f"✅ Embeddings listos: shape={embeddings.shape}")

    # -----------------------------------------------------------------------
    # 4. Outliers
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\n[4/7] Detectando outliers (método: {outlier_method})...\n{'='*60}")

    df_limpio, embeddings_clean, analisis_ngramas, outliers_meta = step_outliers(
        df_proc=df_proc,
        embeddings=embeddings,
        source_column=source_column,
        dmo_label=dmo_label,
        outlier_method=outlier_method,
        contamination=contamination,
        palette=palette,
        final_results_dir=str(results_dir),
        cache_dir=cache_dir_str,
        parent_key=embeddings_meta["key"],
        use_cache=use_cache,
        force=_force_step("outliers", force_steps),
    )

    steps_meta.append(outliers_meta)

    out_csv = output_dir_path / "df_clean_after_outliers.csv"
    df_limpio.to_csv(out_csv, index=False)

    print(f"✅ DataFrame limpio: {len(df_limpio):,} rows")

    # -----------------------------------------------------------------------
    # 5. Sentiment
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\n[5/7] Análisis de sentimientos...\n{'='*60}")

    df_sentiment, sentiment_meta = step_sentiment(
        df_clean=df_limpio,
        text_column=text_column,
        source_column=source_column,
        lang=lang,
        sentiment_backend=sentiment_backend,
        sentiment_threshold=sentiment_threshold,
        sentiment_model=sentiment_model,
        cache_dir=cache_dir_str,
        parent_key=outliers_meta["key"],
        use_cache=use_cache,
        force=_force_step("sentiment", force_steps),
    )

    steps_meta.append(sentiment_meta)

    sentiment_csv = output_dir_path / "df_with_sentiment.csv"
    df_sentiment.to_csv(sentiment_csv, index=False)

    print(f"✅ Sentimiento guardado: {sentiment_csv}")

    # -----------------------------------------------------------------------
    # 6. Topics
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\n[6/7] Modelado de tópicos por sentimiento...\n{'='*60}")

    topic_outputs, topics_meta = step_topics(
        df_sentiment=df_sentiment,
        text_column=text_column,
        source_column=source_column,
        min_partition_size=min_partition_size,
        topic_backend=topic_backend,
        use_xidoo_outliers=use_xidoo_outliers,
        random_state=42,
        cache_dir=cache_dir_str,
        parent_key=sentiment_meta["key"],
        use_cache=use_cache,
        force=_force_step("topics", force_steps),
        num_topics=n_topics if n_topics is not None else "auto",
    )

    steps_meta.append(topics_meta)

    topics_df = topic_outputs.get("topics_df", pd.DataFrame())

    topics_csv = output_dir_path / "topics_by_sentiment.csv"
    topics_df.to_csv(topics_csv, index=False)

    print(f"✅ Tópicos guardados: {topics_csv}")

    # Resumen de tópicos por fuente
    if not topics_df.empty:
        topic_counts, topic_percentages = summarize_topics_by_source(
            topics_df=topics_df,
            source_column=source_column,
        )

        if not topic_counts.empty:
            topic_counts.to_csv(output_dir_path / "topic_counts_by_source.csv")
            topic_percentages.to_csv(output_dir_path / "topic_percentages_by_source.csv")

    # -----------------------------------------------------------------------
    # 7. Visuals
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\n[7/7] Generando visualizaciones...\n{'='*60}")

    visuals_meta = step_visuals(
        df_sentiment=df_sentiment,
        embeddings_clean=embeddings_clean,
        topic_outputs=topic_outputs,
        topics_df=topics_df,
        source_column=source_column,
        dmo_label=dmo_label,
        palette=palette,
        text_column=text_column,
        umap_method=umap_method,
        max_plot_points=max_plot_points,
        final_results_dir=str(results_dir),
        cache_dir=cache_dir_str,
        parent_key=topics_meta["key"],
        use_cache=use_cache,
        force=_force_step("visuals", force_steps),
    )

    steps_meta.append(visuals_meta)

    # -----------------------------------------------------------------------
    # History / latest run
    # -----------------------------------------------------------------------
    record = {
        "run_id": run_id,
        "timestamp": _now_iso(),
        "params": {
            "input_csv": input_csv,
            "text_column": text_column,
            "source_column": source_column,
            "dmo_label": dmo_label,
            "lang": lang,
            "title": title,
            "palette": palette,
            "output_dir": str(output_dir_path),
            "outlier_method": outlier_method,
            "contamination": contamination,
            "n_topics": n_topics,
            "embedding_backend": embedding_backend,
            "topic_backend": topic_backend,
            "sentiment_backend": sentiment_backend,
            "sentiment_model": sentiment_model,
            "sentiment_threshold": sentiment_threshold,
            "min_partition_size": min_partition_size,
            "use_xidoo_outliers": use_xidoo_outliers,
            "umap_method": umap_method,
            "max_plot_points": max_plot_points,
            "use_cache": use_cache,
            "force_steps": sorted(list(force_steps)),
        },
        "steps": steps_meta,
        "outputs": {
            "clean_csv": str(out_csv),
            "sentiment_csv": str(sentiment_csv),
            "topics_csv": str(topics_csv),
            "plots_dir": str(results_dir),
            "cache_dir": cache_dir_str,
        },
        "cli_args": cli_args or {},
    }

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    _save_json(record, latest_run_path)

    print(f"\n{'='*60}")
    print("✅ Pipeline completado.")
    print(f"✅ DataFrame limpio: {out_csv}")
    print(f"✅ DataFrame con sentimiento: {sentiment_csv}")
    print(f"✅ Tópicos: {topics_csv}")
    print(f"✅ Gráficos: {results_dir}")
    print(f"✅ Caché: {cache_dir_str}")
    print(f"🕘 Histórico guardado en: {history_path}")
    print(f"{'='*60}\n")

    return str(sentiment_csv)


# =============================================================================
# Pasos adicionales (pipeline DMO + UGC de run_analysis.py)
# =============================================================================
def step_load_corpus(
    dmo_csv: str,
    ugc_csvs: List[str],
    text_column: str,
    source_column: str,
    dmo_label: str,
    cache_dir: str,
    use_cache: bool = True,
    force: bool = False,
):
    """
    Une el corpus DMO y el/los corpus UGC en un solo DataFrame y lo cachea.
    """
    from io_corpus import build_unified_frame

    step_name = "load_corpus"

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "dmo": _file_fingerprint(dmo_csv),
        "ugc": [_file_fingerprint(path) for path in ugc_csvs],
        "text_column": text_column,
        "source_column": source_column,
        "dmo_label": dmo_label,
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)
    artifact = str(Path(step_dir) / "data.csv.gz")
    manifest = _load_manifest(step_dir)

    if use_cache and not force and manifest and os.path.exists(artifact):
        df = _load_df(artifact)
        print(f"\u2705 Cach\u00e9 hit [{step_name}] -> {step_dir}")

        return df, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {"rows": int(len(df)), "artifact": artifact},
        )

    print(f"\U0001f9ee Ejecutando [{step_name}]...")

    df = build_unified_frame(
        dmo_csv=dmo_csv,
        ugc_csvs=list(ugc_csvs),
        text_column=text_column,
        source_column=source_column,
        dmo_label=dmo_label,
    )

    _save_df(df, artifact)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {"data": artifact},
        "rows": int(len(df)),
    }
    _save_manifest(step_dir, manifest)

    return df, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {"rows": int(len(df)), "artifact": artifact},
    )


def step_embeddings_bert(
    df: pd.DataFrame,
    text_column: str,
    model_name: str,
    cache_dir: str,
    parent_key: str,
    use_cache: bool = True,
    force: bool = False,
    preprocess_fingerprint: str = "",
):
    """
    Genera embeddings BERT (sentence-transformers) y los cachea en disco.
    """
    step_name = "embeddings_bert"

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "embeddings_version": EMBEDDINGS_VERSION,
        "parent_key": parent_key,
        "backend": "bert",
        "model_name": str(model_name),
        "text_column": text_column,
        "preprocess_fingerprint": preprocess_fingerprint,
        "rows": int(len(df)),
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)
    artifact = str(Path(step_dir) / "embeddings.npy")
    manifest = _load_manifest(step_dir)

    if use_cache and not force and manifest and os.path.exists(artifact):
        embeddings = np.load(artifact)
        print(f"\u2705 Cach\u00e9 hit [{step_name}] -> {step_dir}")

        return embeddings, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "rows": int(embeddings.shape[0]),
                "shape": list(embeddings.shape),
                "artifact": artifact,
            },
        )

    print(f"\U0001f9ee Ejecutando [{step_name}] con modelo [{model_name}]...")
    _ensure_dir(step_dir)

    from embeddings_comparison import obtener_embeddings_bert

    embeddings = obtener_embeddings_bert(
        df[text_column].fillna("").astype(str).tolist(),
        model_name=model_name,
    )
    np.save(artifact, embeddings)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {"embeddings": artifact},
        "model_name": str(model_name),
        "shape": list(embeddings.shape),
    }
    _save_manifest(step_dir, manifest)

    return embeddings, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "rows": int(embeddings.shape[0]),
            "shape": list(embeddings.shape),
            "artifact": artifact,
        },
    )


def step_outliers_ngram(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_column: str,
    source_column: str,
    method: str,
    contamination,
    per_source: bool,
    top_n: int,
    min_samples_per_source: int,
    output_dir: str,
    cache_dir: str,
    parent_key: str,
    use_cache: bool = True,
    force: bool = False,
):
    """
    Detecta outliers (IF/LOF/zscore) por fuente y calcula n-gramas.
    Cachea df_clean, df_outliers y el resumen de n-gramas.
    """
    from outlier_ngram_analysis import analizar_outliers

    step_name = "outliers_ngram"

    inputs = {
        "step": step_name,
        "cache_version": CACHE_VERSION,
        "outlier_version": OUTLIER_VERSION,
        "parent_key": parent_key,
        "text_column": text_column,
        "source_column": source_column,
        "method": method,
        "contamination": str(contamination),
        "per_source": bool(per_source),
        "top_n": int(top_n),
        "min_samples_per_source": int(min_samples_per_source),
        "rows": int(len(df)),
        "embedding_shape": list(embeddings.shape),
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)

    clean_artifact = str(Path(step_dir) / "df_clean.csv.gz")
    outliers_artifact = str(Path(step_dir) / "df_outliers.csv.gz")
    analysis_artifact = str(Path(step_dir) / "analysis.json")

    manifest = _load_manifest(step_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (
        use_cache
        and not force
        and manifest
        and os.path.exists(clean_artifact)
        and os.path.exists(outliers_artifact)
        and os.path.exists(analysis_artifact)
    ):
        df_clean = _load_df(clean_artifact)
        df_outliers = _load_df(outliers_artifact)
        analysis = _load_json(analysis_artifact)

        # Restaurar los artefactos visibles en results/
        df_clean.to_csv(out_dir / "df_clean.csv", index=False)
        df_outliers.to_csv(out_dir / "df_outliers.csv", index=False)
        _save_json(analysis, str(out_dir / "outlier_analysis.json"))

        print(f"\u2705 Cach\u00e9 hit [{step_name}] -> {step_dir}")

        return df_clean, df_outliers, analysis, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {
                "rows_clean": int(len(df_clean)),
                "rows_outliers": int(len(df_outliers)),
                "clean_artifact": clean_artifact,
                "outliers_artifact": outliers_artifact,
                "analysis_artifact": analysis_artifact,
            },
        )

    print(f"\U0001f9ee Ejecutando [{step_name}]...")

    result = analizar_outliers(
        df=df,
        text_column=text_column,
        source_column=source_column,
        embeddings=embeddings,
        output_dir=str(out_dir),
        method=method,
        contamination=contamination,
        per_source=per_source,
        min_samples_per_source=min_samples_per_source,
        top_n=top_n,
    )

    df_clean = result["df_clean"]
    df_outliers = result["df_outliers"]

    analysis = {
        "unigramas": [list(item) for item in result["unigramas"]],
        "bigramas": [list(item) for item in result["bigramas"]],
        "trigramas": [list(item) for item in result["trigramas"]],
        "stats_por_fuente": result["stats_por_fuente"],
        "resumen": result.get("resumen", {}),
    }

    _save_df(df_clean, clean_artifact)
    _save_df(df_outliers, outliers_artifact)
    _save_json(analysis, analysis_artifact)

    manifest = {
        "step": step_name,
        "created_at": _now_iso(),
        "inputs": inputs,
        "outputs": {
            "df_clean": clean_artifact,
            "df_outliers": outliers_artifact,
            "analysis": analysis_artifact,
        },
        "rows_clean": int(len(df_clean)),
        "rows_outliers": int(len(df_outliers)),
    }
    _save_manifest(step_dir, manifest)

    return df_clean, df_outliers, analysis, _step_meta(
        step_name,
        key,
        step_dir,
        False,
        {
            "rows_clean": int(len(df_clean)),
            "rows_outliers": int(len(df_outliers)),
            "clean_artifact": clean_artifact,
            "outliers_artifact": outliers_artifact,
            "analysis_artifact": analysis_artifact,
        },
    )
