"""Stage B: add cacheable steps used by run_analysis.py (DMO+UGC, BERT, outliers)."""
from pathlib import Path

p = Path(__file__).resolve().parent / "cache.py"
s = p.read_text()


def rep(old, new):
    global s
    assert old in s, f"NOT FOUND -> {old[:80]!r}"
    s = s.replace(old, new, 1)


# --- step_topics: soporte de embeddings (BERTopic) ---------------------------
rep(
    """def step_topics(
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
):""",
    """def step_topics(
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
):""",
)

rep(
    """        "use_xidoo_outliers": bool(use_xidoo_outliers),
        "random_state": int(random_state),
        "rows": int(len(df_sentiment)),
    }""",
    """        "use_xidoo_outliers": bool(use_xidoo_outliers),
        "random_state": int(random_state),
        "rows": int(len(df_sentiment)),
        "embeddings_shape": list(embeddings.shape) if embeddings is not None else None,
    }""",
)

rep(
    """    topic_outputs = run_topic_modeling(
        df=df_sentiment,
        text_column=topic_text_column,
        source_column=source_column,
        min_partition_size=min_partition_size,
        topic_backend=topic_backend,
        use_internal_outlier=use_xidoo_outliers,
        random_state=random_state,
    )""",
    """    topic_outputs = run_topic_modeling(
        df=df_sentiment,
        text_column=topic_text_column,
        source_column=source_column,
        min_partition_size=min_partition_size,
        topic_backend=topic_backend,
        use_internal_outlier=use_xidoo_outliers,
        random_state=random_state,
        embeddings=embeddings,
    )""",
)

# --- Nuevos pasos -------------------------------------------------------------
NEW_STEPS = '''

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
        print(f"\\u2705 Cach\\u00e9 hit [{step_name}] -> {step_dir}")

        return df, _step_meta(
            step_name,
            key,
            step_dir,
            True,
            {"rows": int(len(df)), "artifact": artifact},
        )

    print(f"\\U0001f9ee Ejecutando [{step_name}]...")

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
        "rows": int(len(df)),
    }

    key = _hash_obj(inputs)
    step_dir = str(Path(cache_dir) / step_name / key)
    artifact = str(Path(step_dir) / "embeddings.npy")
    manifest = _load_manifest(step_dir)

    if use_cache and not force and manifest and os.path.exists(artifact):
        embeddings = np.load(artifact)
        print(f"\\u2705 Cach\\u00e9 hit [{step_name}] -> {step_dir}")

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

    print(f"\\U0001f9ee Ejecutando [{step_name}] con modelo [{model_name}]...")
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

        print(f"\\u2705 Cach\\u00e9 hit [{step_name}] -> {step_dir}")

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

    print(f"\\U0001f9ee Ejecutando [{step_name}]...")

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
'''

s = s.rstrip("\n") + "\n" + NEW_STEPS
p.write_text(s)
print("cache.py patched")
