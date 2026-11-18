"""
topic_console_report.py
─────────────────────────────────────────────────────────────
Modelado de tópicos por sentimiento (pos / neg)
+ salida a consola para IA.
Usa topic_modeling.run_topic_modeling() existente.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def reporte_topicos_consola(
    df: pd.DataFrame,
    text_column: str,
    source_column: str,
    output_dir: str = "results/topics",
    topic_backend: str = "xidoo",
    min_partition_size: int = 50,
    top_n_terms: int = 20,
    embeddings: Optional[np.ndarray] = None,
    palette: Optional[Dict[str, str]] = None,
    topic_outputs: Optional[Dict[str, Any]] = None,
    num_topics: Any = "auto",
) -> Dict[str, Any]:
    """
    Ejecuta run_topic_modeling y formatea TODO para consola.

    Si `topic_outputs` viene dado (por ejemplo, desde la caché), no se
    vuelve a modelar: solo se reconstruyen los DataFrames por sentimiento
    a partir de `topics_df`.
    """
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
            num_topics=num_topics,
        )

    SEP = "═" * 70

    # ──────────────────────────────────
    # POSITIVOS
    # ──────────────────────────────────
    print(f"\n{SEP}")
    print("   🟢  MODELADO DE TÓPICOS – COMENTARIOS POSITIVOS")
    print(f"{SEP}")
    pos = by_sentiment.get("positive", {})
    _imprimir_particion(pos, source_column, "POSITIVO")

    # ──────────────────────────────────
    # NEGATIVOS
    # ──────────────────────────────────
    print(f"\n{SEP}")
    print("   🔴  MODELADO DE TÓPICOS – COMENTARIOS NEGATIVOS")
    print(f"{SEP}")
    neg = by_sentiment.get("negative", {})
    _imprimir_particion(neg, source_column, "NEGATIVO")

    # ──────────────────────────────────
    # DISTRIBUCIÓN DE TÓPICOS POR FUENTE
    # ──────────────────────────────────
    if not topics_df.empty and "topic_name" in topics_df.columns:
        print(f"\n{SEP}")
        print("   📊  DISTRIBUCIÓN DE TÓPICOS POR FUENTE")
        print(f"{SEP}")
        counts = (
            topics_df.groupby([source_column, "topic_name"])
            .size()
            .unstack(fill_value=0)
        )
        pcts = counts.div(counts.sum(axis=1), axis=0) * 100

        print(f"\n  Conteos:")
        print(counts.to_string())
        print(f"\n  Porcentajes (%):")
        print(pcts.round(1).to_string())
        print(f"{SEP}\n")

        # Guardar CSV
        counts.to_csv(out / "topic_counts_by_source.csv")
        pcts.round(2).to_csv(out / "topic_pcts_by_source.csv")

    # Guardar topics_df
    if not topics_df.empty:
        topics_df.to_csv(out / "topics_df.csv", index=False)
        print(f"  ✅ topics_df.csv guardado en {out}/")

    # ──────────────────────────────────
    # SELECCIÓN DEL NÚMERO DE TÓPICOS
    # ──────────────────────────────────
    _reportar_seleccion_topicos(by_sentiment, out, palette)

    # Generar visualizaciones 3D
    from visualization import plot_topics_3d_umap, plot_topic_distribution_by_source_3d

    # Recolectar embeddings de ambos sentimientos
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
                embeddings_dict[sentiment_label] = np.asarray(embeddings)[mask]

    # Generar gráficos
    plot_topics_3d_umap(
        topics_df=topics_df,
        embeddings_dict=embeddings_dict,
        text_column=text_column,
        source_column=source_column,
        output_dir=output_dir,
        palette=palette,
    )

    plot_topic_distribution_by_source_3d(
        topics_df=topics_df,
        source_column=source_column,
        output_dir=output_dir,
        palette=palette,
    )


    return {"by_sentiment": by_sentiment, "topics_df": topics_df}


def _imprimir_particion(particion: Dict[str, Any], source_column: str, label: str):
    """Imprime una partición (positive o negative) en consola."""
    method = particion.get("method", "N/A")
    count = particion.get("count", 0)
    print(f"  Método: {method}")
    print(f"  Cantidad de comentarios: {count:,}")

    if method == "empty":
        print(f"  ⚠️  {particion.get('message', 'Sin datos')}")
        return

    # Keywords / tópicos
    keywords = particion.get("keywords", {})
    topic_names = particion.get("topic_names", {})

    if method == "word_frequency":
        # Frecuencia de palabras (fallback)
        print(f"  Nota: {particion.get('note', '')}")
        for ngram_type in ["top_terms_unigram", "top_terms_bigram", "top_terms_trigram"]:
            terms = particion.get(ngram_type, [])
            if terms:
                print(f"\n  ── {ngram_type.replace('top_terms_', '').upper()}S (top {len(terms)}) ──")
                for i, (w, c) in enumerate(terms, 1):
                    print(f"    {i:>3}. {w:<40s} {c:>5,d}")
    else:
        # Topic modeling real
        print(f"\n  ── Tópicos identificados ──")
        for tid in sorted(topic_names.keys(), key=str):
            tname = topic_names.get(tid, f"Topic {tid}")
            kws = keywords.get(tid, keywords.get(str(tid), []))
            if isinstance(kws, list):
                kws_str = ", ".join(str(k) for k in kws[:10])
            else:
                kws_str = str(kws)
            print(f"    📌 {tname}")
            print(f"       Keywords: {kws_str}")

        # Distribución por fuente dentro de la partición
        df_part = particion.get("df")
        if df_part is not None and not df_part.empty:
            print(f"\n  ── Distribución por fuente ({label}) ──")
            if "topic_name" in df_part.columns and source_column in df_part.columns:
                dist = df_part.groupby([source_column, "topic_name"]).size().reset_index(name="n")
                for _, row in dist.iterrows():
                    print(f"    {row[source_column]:<20s} | {row['topic_name']:<45s} | n={row['n']}")


def _reportar_seleccion_topicos(
    by_sentiment: Dict[str, Any],
    out: Path,
    palette: Optional[Dict[str, str]],
) -> None:
    """Imprime y guarda la evidencia de por qué se eligió ese número de tópicos."""
    for label in ("positive", "negative"):
        part = by_sentiment.get(label, {})
        selection = part.get("topic_selection") if isinstance(part, dict) else None
        if not isinstance(selection, dict) or not selection.get("candidates"):
            continue

        SEP = "═" * 70
        print(f"\n{SEP}")
        print(f"   🎯  SELECCIÓN DE NÚMERO DE TÓPICOS – {label.upper()}")
        print(f"{SEP}")
        print(
            f"  Base: {selection.get('base_topics', 0)} tópicos | "
            f"Elegido: k={selection.get('selected_k', 0)} | "
            f"Outliers: {selection.get('n_outliers', 0):,}"
        )
        print(
            f"  {'k':>4s} {'cNPMI':>8s} {'Div':>7s} {'ARI':>7s} "
            f"{'Red':>7s} {'Bal':>7s} {'Score':>7s} {'n mín':>7s}  Estado"
        )
        for row in sorted(selection["candidates"], key=lambda item: item.get("k", 0)):
            mark = "✅" if row.get("selected") else "  "
            print(
                f"  {row.get('k', 0):>4d} "
                f"{_fmt_metric(row.get('coherence')):>8s} "
                f"{_fmt_metric(row.get('diversity')):>7s} "
                f"{_fmt_metric(row.get('stability')):>7s} "
                f"{_fmt_metric(row.get('redundancy')):>7s} "
                f"{_fmt_metric(row.get('balance')):>7s} "
                f"{_fmt_metric(row.get('score')):>7s} "
                f"{int(row.get('min_topic_size', 0)):>7d}  {mark}"
            )
        print(f"\n  Justificación: {selection.get('reason', '')}")
        print(f"{SEP}\n")

        table = pd.DataFrame(selection["candidates"]).rename(
            columns={
                "k": "k",
                "coherence": "coherencia_npmi",
                "diversity": "diversidad",
                "stability": "estabilidad_ari",
                "redundancy": "redundancia",
                "balance": "balance",
                "min_topic_size": "tamano_minimo",
                "eligible": "elegible",
                "selected": "seleccionado",
                "score": "puntaje",
                "note": "nota",
            }
        )
        table.to_csv(out / f"topic_selection_{label}.csv", index=False)

        try:
            from visualization import plot_topic_selection_curve

            plot_topic_selection_curve(
                selection=selection,
                output_html=out / f"topic_selection_{label}.html",
                palette=palette,
                sentiment_label=label,
            )
            print(
                f"  ✅ Evidencia guardada: topic_selection_{label}.csv "
                f"y topic_selection_{label}.html"
            )
        except Exception as exc:
            print(f"  ⚠️ No se pudo generar el gráfico de selección: {exc}")


def _fmt_metric(value: Any) -> str:
    try:
        if value is None:
            return "n/d"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/d"