"""Stage C-bis: adapt cache/topic report to the new pipeline (cache + palette)."""
from pathlib import Path

CODE = Path(__file__).resolve().parent


def patch(fname, pairs):
    p = CODE / fname
    s = p.read_text()
    for old, new in pairs:
        assert old in s, f"{fname}: NOT FOUND -> {old[:80]!r}"
        s = s.replace(old, new, 1)
    p.write_text(s)
    print(f"patched {fname}")


# --- cache.py: el resumen JSON no debe llevar DataFrames ni ndarrays --------
patch("cache.py", [(
    """        output_copy = {
            key: value
            for key, value in output.items()
            if key != "df"
        }""",
    """        output_copy = {
            key: value
            for key, value in output.items()
            if key not in ("df", "embeddings")
        }""",
)])


# --- topic_console_report.py: aceptar topic_outputs precomputados -----------
patch("topic_console_report.py", [
    (
        """    embeddings: Optional[np.ndarray] = None,
    palette: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    \"\"\"
    Ejecuta run_topic_modeling y formatea TODO para consola.
    \"\"\"
    from topic_modeling import run_topic_modeling

    print(f"  Backend de modelado de topicos: {topic_backend}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = run_topic_modeling(
        df=df,
        text_column=text_column,
        source_column=source_column,
        min_partition_size=min_partition_size,
        topic_backend=topic_backend,
        top_n_terms=top_n_terms,
        embeddings=embeddings,
    )""",
        """    embeddings: Optional[np.ndarray] = None,
    palette: Optional[Dict[str, str]] = None,
    topic_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    \"\"\"
    Ejecuta run_topic_modeling y formatea TODO para consola.

    Si `topic_outputs` viene dado (por ejemplo, desde la caché), no se
    vuelve a modelar: solo se reconstruyen los DataFrames por sentimiento
    a partir de `topics_df`.
    \"\"\"
    from topic_modeling import run_topic_modeling

    print(f"  Backend de modelado de topicos: {topic_backend}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if topic_outputs is not None:
        result = topic_outputs
        by_sentiment = result.get("by_sentiment", {})
        topics_df = result.get("topics_df", pd.DataFrame())

        # Reconstruir el df de cada partición (no se cachea por tamaño).
        for label, part in by_sentiment.items():
            if isinstance(part, dict) and "df" not in part and not topics_df.empty:
                if "sentiment_label" in topics_df.columns:
                    part["df"] = topics_df[
                        topics_df["sentiment_label"] == label
                    ].reset_index(drop=True)
    else:
        result = run_topic_modeling(
            df=df,
            text_column=text_column,
            source_column=source_column,
            min_partition_size=min_partition_size,
            topic_backend=topic_backend,
            top_n_terms=top_n_terms,
            embeddings=embeddings,
        )""",
    ),
    (
        """    by_sentiment = result["by_sentiment"]
    topics_df = result["topics_df"]

    SEP""",
        """    SEP""",
    ),
    (
        """    # Recolectar embeddings de ambos sentimientos
    embeddings_dict = {}
    for sentiment_label in ["positive", "negative"]:
        part = by_sentiment.get(sentiment_label, {})
        if "embeddings" in part and part["embeddings"] is not None:
            embeddings_dict[sentiment_label] = part["embeddings"]""",
        """    # Recolectar embeddings de ambos sentimientos
    embeddings_dict = {}
    for sentiment_label in ["positive", "negative"]:
        part = by_sentiment.get(sentiment_label, {})
        if "embeddings" in part and part["embeddings"] is not None:
            embeddings_dict[sentiment_label] = part["embeddings"]

    # Si no vienen en el resultado, se alinean desde el DataFrame + embeddings
    if not embeddings_dict and embeddings is not None and len(df) == len(embeddings):
        labels = df["sentiment_label"].astype(str).to_numpy()
        for sentiment_label in ["positive", "negative"]:
            mask = labels == sentiment_label
            if mask.sum() > 0:
                embeddings_dict[sentiment_label] = np.asarray(embeddings)[mask]""",
    ),
])
