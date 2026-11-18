#!/usr/bin/env python3
"""
run_analysis.py
─────────────────────────────────────────────────────────────
Pipeline COMPLETO en una sola ejecución:
  Bloque I  (pasos 0–7): carga, embeddings, outliers, sentimiento, validación,
                         reportes, wordclouds, tópicos por sentimiento.
  Bloque II (pasos 8–11): comparación imagen proyectada (DMO) vs percibida (UGC)
                         8  base analítica consolidada
                         9  tópicos conjuntos DMO+UGC con selección de k
                         10 capa léxica L1 (Jaccard, cobertura, log-odds Monroe 2008)
                         11 capa temática L2 (gaps, χ²/Fisher, BH, JSD/EMD)
                         (semantic, affective, ias, figures, qualitative, report: P4+)

Uso (una sola corrida, desde ~/ac_4/code/):
    python run_analysis.py \
        --csv-dmo "../data/processed/df_dmo.csv" \
        --csv-ugc "../data/processed/df_ugc.csv" \
        --text-col text --lang es \
        --outlier-method IF --contamination auto \
        --min-partition 50 \
        --output-dir ../results \
        --model bertopic --palette ejecutiva --num-topicos automatico

Caché: <output-dir>/cache/<paso>/<clave>/. La segunda corrida reutiliza lo que no cambió.
"""
from __future__ import annotations

import argparse
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from cache import (
    step_embeddings_bert,
    step_load_corpus,
    step_outliers_ngram,
    step_sentiment,
    step_topics,
)
from io_corpus import CORPUS_COLUMN, DMO_LABEL, SOURCE_COLUMN
from utils import PALETAS_ACCESIBLES, palette_to_plot_colors

BACKEND_BERTOPIC = "bertopic"
BACKEND_XIDOO = "x" + "idoo"
MODEL_CHOICES = (BACKEND_BERTOPIC, BACKEND_XIDOO + "topic", BACKEND_XIDOO)

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CACHE_STEPS = ("load", "embeddings", "outliers", "sentiment", "topics", "all")

COMPARISON_TASKS = [
    "base", "topics", "lexical", "thematic", "price",
    "semantic", "affective", "ias", "figures", "qualitative", "report",
]
COMPARISON_STEP = {
    "base": 8, "topics": 9, "lexical": 10, "thematic": 11, "price": 12,
    "semantic": 13, "affective": 14, "ias": 15, "figures": 16,
    "qualitative": 17, "report": 18,
}


def _parse_contamination(v: str):
    """Permite pasar un float o la palabra 'auto'."""
    if v.strip().lower() == "auto":
        return "auto"
    return float(v)


def _parse_num_topics(v: str):
    """Permite un entero >= 1 o 'automatico' para seleccionar k."""
    from topic_selection import parse_num_topics

    try:
        return parse_num_topics(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _normalize_model(value: str) -> str:
    """Normaliza el backend elegido: 'bertopic' o el backend XidooTopic."""
    v = (value or "").strip().lower()
    if v.startswith("bert"):
        return BACKEND_BERTOPIC
    if v.startswith(BACKEND_XIDOO):
        return BACKEND_XIDOO
    raise ValueError(f"Modelo de topicos no soportado: {value}")


def _asegurar_sentimiento_binario(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Garantiza etiquetas binarias positive/negative."""
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


def _alinear_embeddings_clean(
    df: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_outliers: pd.DataFrame,
    embeddings: np.ndarray | None,
) -> np.ndarray | None:
    """Alinea embeddings con df_clean por CONTENIDO, no por índice.

    El índice original se pierde al leer los CSV de caché (index=False), por lo que
    `embeddings[df_clean.index]` queda desplazado en corridas con cache hit. Aquí se
    reconstruye el doc_id con la misma función que usa el PASO 8.
    """
    if embeddings is None or not len(df_clean):
        return None
    try:
        from comparison.base_builder import assign_doc_ids

        key_cols = [c for c in df.columns
                    if c in df_clean.columns and c in df_outliers.columns]
        ids_clean, _ = assign_doc_ids(
            df, df_clean.reset_index(drop=True),
            df_outliers.reset_index(drop=True), key_cols,
        )
        return embeddings[np.asarray(ids_clean, dtype=int)]
    except Exception as exc:
        print(f"  ⚠️ Alineación por contenido no disponible ({exc}); se usa el índice.")
        pos = df_clean.index.to_numpy()
        return embeddings[pos] if len(pos) and pos.max() < len(embeddings) else None


def _run_comparison(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    df_clean: pd.DataFrame,
    df_outliers: pd.DataFrame,
) -> tuple[list[Path], list[str]]:
    """Ejecuta las tareas de comparación (PASO 8+) con los objetos ya calculados."""
    from comparison.common import print_header

    tasks = [t for t in COMPARISON_TASKS if t in set(args.comparison_tasks)]
    written: list[Path] = []
    failed: list[str] = []

    for task in tasks:
        try:
            if task == "base":
                from comparison.base_builder import run_base_task
                written += run_base_task(args, output_dir, inputs={
                    "df": df, "embeddings": embeddings,
                    "df_clean": df_clean, "df_outliers": df_outliers,
                })
            elif task == "topics":
                from joint_topic_model import run_topics_task
                written += run_topics_task(args, output_dir)
            elif task == "lexical":
                from metrics_lexical import run_lexical_task
                written += run_lexical_task(args, output_dir)
            elif task == "thematic":
                from metrics_thematic import run_thematic_task
                written += run_thematic_task(args, output_dir)
            elif task == "price":
                from price_concept import run_price_task
                written += run_price_task(args, output_dir)
            else:
                print_header(f"PASO {COMPARISON_STEP[task]}: {task.upper()} (PENDIENTE)")
                print("  ⏳ Se implementará en la siguiente fase del plan (P4+).")
        except Exception as exc:
            failed.append(task)
            print(f"\n  ❌ Error en la tarea '{task}': {exc}")
            traceback.print_exc()
    return written, failed


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo DMO (proyectada) vs UGC (percibida) en una sola corrida"
    )
    parser.add_argument("--csv-dmo", required=True, help="CSV del corpus DMO (imagen proyectada)")
    parser.add_argument("--csv-ugc", required=True, action="append",
                        help="CSV del corpus UGC (imagen percibida). Puede repetirse.")
    parser.add_argument("--text-col", required=True, help="Columna de texto")
    parser.add_argument("--lang", default="es")
    parser.add_argument("--outlier-method", default="IF", choices=["IF", "LOF", "zscore"])
    parser.add_argument("--contamination", type=_parse_contamination, default="auto",
                        help="float (0-0.5) o 'auto'")
    parser.add_argument("--min-partition", type=int, default=50)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--top-n", type=int, default=50, help="Top n-gramas a imprimir")
    parser.add_argument("--global-outliers", dest="per_source", action="store_false",
                        help="Detección global (no por fuente)")
    parser.add_argument("--model", default=BACKEND_BERTOPIC, choices=list(MODEL_CHOICES),
                        help="Backend de modelado de tópicos: bertopic o xidootopic")
    parser.add_argument("--num-topicos", "--num-topics", dest="num_topicos",
                        type=_parse_num_topics, default="auto",
                        help="Tópicos por sentimiento: entero >=1 o 'automatico'")
    parser.add_argument("--palette", default="ejecutiva", choices=PALETAS_ACCESIBLES,
                        help="Paleta de colores de los gráficos HTML")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                        help="Modelo de sentence-transformers para los embeddings")
    parser.add_argument("--sentiment-model", default=None,
                        help="Modelo HF para el análisis de sentimiento")

    # ── Caché ──
    parser.add_argument("--cache-dir", default=None,
                        help="Directorio de caché (por defecto <output-dir>/cache)")
    parser.add_argument("--no-cache", action="store_true", help="Ignorar la caché")
    parser.add_argument("--force-steps", nargs="+", default=[], choices=list(CACHE_STEPS),
                        help="Pasos a recalcular aunque exista caché")

    # ── Validación de sentimiento ──
    parser.add_argument("--no-sentiment-validation", dest="sentiment_validation",
                        action="store_false", help="Omitir la validación del sentimiento")
    parser.add_argument("--gold-csv", default=None, help="CSV con etiquetas humanas")
    parser.add_argument("--sentiment-ablation", action="store_true",
                        help="Ejecutar la ablación (léxico / TF-IDF / linear probe)")
    parser.add_argument("--sentiment-finetune", action="store_true",
                        help="Incluir transformer afinado en la ablación (costoso)")
    parser.add_argument("--finetune-model", default="pysentimiento/robertuito-sentiment-analysis",
                        help="Modelo a afinar en la ablación")

    # ── BLOQUE II: comparación DMO vs UGC (pasos 8+) ──
    g = parser.add_argument_group("Comparación DMO vs UGC (pasos 8+)")
    g.add_argument("--skip-comparison", action="store_true",
                   help="Ejecuta solo el bloque I (pasos 0–7)")
    g.add_argument("--comparison-tasks", nargs="+", choices=COMPARISON_TASKS,
                   default=list(COMPARISON_TASKS), help="Tareas de comparación a ejecutar")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--bootstrap", type=int, default=2000)
    g.add_argument("--permutations", type=int, default=1000)
    g.add_argument("--allow-recompute", action="store_true",
                   help="Permite recalcular pasos si la base no encuentra caché")
    # P1: tópicos conjuntos
    g.add_argument("--k-min", type=int, default=4)
    g.add_argument("--k-max", type=int, default=30)
    g.add_argument("--stability-runs", type=int, default=5)
    g.add_argument("--subsample", type=float, default=0.8)
    g.add_argument("--dmo-weight", type=float, default=1.0,
                   help="Peso DMO en centroides/c-TF-IDF (sin remuestreo). Default 1.0")
    g.add_argument("--reassign-threshold", type=float, default=0.30)
    g.add_argument("--base-min-topic-size", type=int, default=10)
    g.add_argument("--ngram-max", type=int, default=2, choices=[1, 2])
    g.add_argument("--min-df", type=int, default=5)
    g.add_argument("--max-df", type=float, default=0.40,
                   help="Descarta tipos presentes en > esta proporción de docs del corpus "
                        "combinado (filtro estadístico, sin listas de stopwords). 1.0 = off")
    g.add_argument("--force-topics", action="store_true",
                   help="Ignora la caché propia de topics_joint")
    # P2: capa léxica
    g.add_argument("--prior-strength", type=float, default=1000.0,
                   help="alpha0 del prior dirichlet (Monroe et al. 2008)")
    g.add_argument("--disable-lemma", action="store_true",
                   help="Fuerza tokenización sin lematización (determinista offline)")

    # Grupo propio: Precio/Valor/Costo (PASO 12, F4.2–F4.4)
    pg = parser.add_argument_group("Precio/Valor/Costo")
    pg.add_argument("--price-anchor-file", "--price-anchors", dest="price_anchor_file",
                    default="resources/anchors/precio_valor.json",
                    help="JSON de anchors Precio/Valor "
                         "(default: resources/anchors/precio_valor.json)")
    pg.add_argument("--price-mode", default="semantic",
                    choices=["semantic", "lexical", "hybrid"],
                    help="Regla del detector: semantic (>=τ), lexical (lex>0), "
                         "hybrid (>=τ o lex>=2)")
    pg.add_argument("--price-center", action=argparse.BooleanOptionalAction, default=True,
                    help="Score semántico centrado por la media del corpus "
                         "(--no-price-center = raw; default: centrado)")
    pg.add_argument("--price-tau-quantile", type=float, default=0.95,
                    help="Cuantil del placebo usado como τ (F4.2)")
    pg.add_argument("--price-placebo-docs", type=int, default=200,
                    help="Docs aleatorios del corpus en el placebo de τ")
    pg.add_argument("--price-sample-size", "--price-validation-n", dest="price_sample_size",
                    type=int, default=100,
                    help="Tamaño de la muestra estratificada de anotación (F4.3)")
    pg.add_argument("--price-annotation-file",
                    default="../annotations/precio_valor_labels.csv",
                    help="CSV de anotaciones F4.3 "
                         "(default: ../annotations/precio_valor_labels.csv)")
    pg.add_argument("--price-top-k", type=int, default=5,
                    help="Top-k ejemplos por corpus/fuente (F4.4)")

    parser.set_defaults(per_source=True, sentiment_validation=True)
    args = parser.parse_args()

    backend = _normalize_model(args.model)
    force_steps = set(args.force_steps or [])
    use_cache = not args.no_cache

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(args.cache_dir) if args.cache_dir else output_dir / "cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    colors = palette_to_plot_colors(args.palette)
    SEP = "█" * 70

    def _force(step: str) -> bool:
        return ("all" in force_steps) or (step in force_steps)

    # ══ PASO 0: Cargar datos ══
    print(f"\n{SEP}")
    print("   PASO 0: CARGA DE DATOS")
    print(SEP)
    df, load_meta = step_load_corpus(
        dmo_csv=args.csv_dmo,
        ugc_csvs=args.csv_ugc,
        text_column=args.text_col,
        source_column=SOURCE_COLUMN,
        dmo_label=DMO_LABEL,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force=_force("load"),
    )
    print(f"  Corpus DMO: {args.csv_dmo}")
    print(f"  Corpus UGC: {', '.join(args.csv_ugc)}")
    print(f"  Filas: {len(df):,}")
    print(f"  Columnas: {list(df.columns)}")
    if CORPUS_COLUMN in df.columns:
        print("  Distribución por corpus:")
        for corpus, cnt in df[CORPUS_COLUMN].value_counts().items():
            print(f"    {corpus:<25s} {cnt:>6,d}")
    print("  Distribución por fuente:")
    for src, cnt in df[SOURCE_COLUMN].value_counts().items():
        print(f"    {src:<25s} {cnt:>6,d}")

    # ══ PASO 1: Preprocesamiento ══
    print(f"\n{SEP}")
    print("   PASO 1: PREPROCESAMIENTO DE TEXTO")
    print(SEP)
    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    df = df[df[args.text_col].str.strip().str.len() > 5].reset_index(drop=True)
    print(f"  Comentarios válidos tras limpieza: {len(df):,}")

    # ══ PASO 2: Embeddings ══
    print(f"\n{SEP}")
    print("   PASO 2: GENERACIÓN DE EMBEDDINGS")
    print(SEP)
    embeddings, embeddings_meta = step_embeddings_bert(
        df=df,
        text_column=args.text_col,
        model_name=args.embedding_model,
        cache_dir=cache_dir,
        parent_key=load_meta["key"],
        use_cache=use_cache,
        force=_force("embeddings"),
    )
    print(f"  Embeddings generados: {embeddings.shape}")

    # ══ PASO 3: Outliers + N-gramas ══
    print(f"\n{SEP}")
    print("   PASO 3: DETECCIÓN DE OUTLIERS + N-GRAMAS")
    print(SEP)
    df_clean, df_outliers, outlier_analysis, outliers_meta = step_outliers_ngram(
        df=df,
        embeddings=embeddings,
        text_column=args.text_col,
        source_column=SOURCE_COLUMN,
        method=args.outlier_method,
        contamination=args.contamination,
        per_source=args.per_source,
        top_n=args.top_n,
        min_samples_per_source=10,
        output_dir=f"{args.output_dir}/outliers",
        cache_dir=cache_dir,
        parent_key=embeddings_meta["key"],
        use_cache=use_cache,
        force=_force("outliers"),
    )

    # ══ PASO 4: Análisis de Sentimiento ══
    print(f"\n{SEP}")
    print("   PASO 4: ANÁLISIS DE SENTIMIENTOS")
    print(SEP)
    from sentiment_analysis import annotate_sentiment

    df_clean, sentiment_meta = step_sentiment(
        df_clean=df_clean,
        text_column=args.text_col,
        source_column=SOURCE_COLUMN,
        lang=args.lang,
        sentiment_backend="transformers",
        sentiment_threshold=0.0,
        sentiment_model=args.sentiment_model,
        cache_dir=cache_dir,
        parent_key=outliers_meta["key"],
        use_cache=use_cache,
        force=_force("sentiment"),
    )
    df_clean = _asegurar_sentimiento_binario(df_clean)
    print(f"  Sentimiento aplicado a {len(df_clean):,} comentarios normales.")
    print(f"  Rango sentiment_score: [{df_clean['sentiment_score'].min():.3f}, "
          f"{df_clean['sentiment_score'].max():.3f}]")
    print(f"  Distribución: {df_clean['sentiment_label'].value_counts().to_dict()}")

    df_outliers = annotate_sentiment(
        df_outliers, text_column=args.text_col,
        source_column=SOURCE_COLUMN, lang=args.lang, backend="transformers",
        model_name=args.sentiment_model,
    )
    df_outliers = _asegurar_sentimiento_binario(df_outliers)
    print(f"  Sentimiento aplicado a {len(df_outliers):,} outliers.")
    print(f"  Distribución outliers: {df_outliers['sentiment_label'].value_counts().to_dict()}")

    # ── Alineación documento↔embedding (por contenido; robusta ante cache hit) ──
    embeddings_clean = _alinear_embeddings_clean(df, df_clean, df_outliers, embeddings)
    if embeddings_clean is not None:
        print(f"  ✅ Embeddings alineados con df_clean: {embeddings_clean.shape}")

    # ══ PASO 4.5: Validación del sentimiento ══
    if args.sentiment_validation:
        print(f"\n{SEP}")
        print("   PASO 4.5: VALIDACIÓN DEL ANÁLISIS DE SENTIMIENTO")
        print(SEP)
        from sentiment_validation import validar_sentimiento

        validacion = validar_sentimiento(
            df=df_clean,
            text_column=args.text_col,
            source_column=SOURCE_COLUMN,
            label_column="sentiment_label",
            score_column="sentiment_score",
            rating_column="rating",
            gold_csv=args.gold_csv,
            embeddings=embeddings_clean,
            with_ablation=args.sentiment_ablation,
            with_finetune=args.sentiment_finetune,
            finetune_model=args.finetune_model,
            output_dir=f"{args.output_dir}/sentiment_validation",
        )

        weak = validacion.get("weak_labels", {})
        if "global" in weak:
            g_ = weak["global"]
            print(f"  Accuracy (etiqueta débil externa): {g_['accuracy']:.3f}")
            print(f"  Macro-F1: {g_['macro_f1']:.3f}   MCC: {g_['mcc']:.3f}")
            print(f"  Filas evaluadas: {g_['n']:,}")
        ext = validacion.get("validacion_externa", {})
        if ext.get("spearman") is not None:
            print(f"  Spearman score↔rating: {ext['spearman']:.4f}   "
                  f"MAE (estrellas): {ext['mae_stars']:.3f}")

    # ══ PASO 5: Reporte sentimiento por fuente ══
    from sentiment_report import reporte_sentimiento_por_fuente, reporte_sentimiento_outliers

    reporte_sentimiento_por_fuente(
        df_clean,
        source_column=SOURCE_COLUMN,
        output_dir=f"{args.output_dir}/sentiment",
        palette=colors,
        title="Sentimiento por Fuente (comentarios normales)",
    )

    # ══ PASO 6: Sentimiento de outliers ══
    reporte_sentimiento_outliers(
        df_outliers,
        source_column=SOURCE_COLUMN,
        output_dir=f"{args.output_dir}/sentiment",
        palette=colors,
        title="Sentimiento de Outliers",
    )

    # ══ PASO 6.5: Wordclouds por sentimiento ══
    print(f"\n{SEP}")
    print("   PASO 6.5: WORDCLOUDS POR SENTIMIENTO")
    print(SEP)
    from sentiment_wordclouds import generar_wordclouds_sentimiento

    wordcloud_results = generar_wordclouds_sentimiento(
        df_clean=df_clean,
        df_outliers=df_outliers,
        text_column=args.text_col,
        source_column=SOURCE_COLUMN,
        output_dir=f"{args.output_dir}/sentiment/wordclouds",
        lang=args.lang,
        colormap_pos="Greens",
        colormap_neg="Reds",
        top_n=80,
        palette=colors,
    )

    # ══ PASO 7: Modelado de tópicos por sentimiento ══
    print(f"\n{SEP}")
    print("   PASO 7: MODELADO DE TÓPICOS POR SENTIMIENTO")
    print(f"   Modelo: {backend}")
    print(f"   Número de tópicos: {args.num_topicos}")
    print(SEP)
    from topic_console_report import reporte_topicos_consola

    topic_outputs, topics_meta = step_topics(
        df_sentiment=df_clean,
        text_column=args.text_col,
        source_column=SOURCE_COLUMN,
        min_partition_size=args.min_partition,
        topic_backend=backend,
        use_xidoo_outliers=False,
        random_state=args.seed,
        cache_dir=cache_dir,
        parent_key=sentiment_meta["key"],
        use_cache=use_cache,
        force=_force("topics"),
        embeddings=embeddings_clean,
        num_topics=args.num_topicos,
    )

    reporte_topicos_consola(
        df=df_clean,
        text_column=args.text_col,
        source_column=SOURCE_COLUMN,
        output_dir=f"{args.output_dir}/topics",
        topic_backend=backend,
        min_partition_size=args.min_partition,
        embeddings=embeddings_clean,
        palette=colors,
        topic_outputs=topic_outputs,
        num_topics=args.num_topicos,
    )

    # ══════════════════════════════════════════════════════
    # ══ BLOQUE II: COMPARACIÓN DMO vs IMAGEN PERCIBIDA ══
    # ══════════════════════════════════════════════════════
    written: list[Path] = []
    failed: list[str] = []
    if not args.skip_comparison:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"\n{SEP}")
        print("   BLOQUE II: COMPARACIÓN DMO (proyectada) vs UGC (percibida)")
        print(f"   Tareas: {', '.join(args.comparison_tasks)}")
        print(f"   seed={args.seed} | bootstrap={args.bootstrap} | paleta={args.palette}")
        print(SEP)
        written, failed = _run_comparison(
            args, output_dir, df, embeddings, df_clean, df_outliers
        )

    # ══ RESUMEN FINAL ══
    print(f"\n{SEP}")
    print("   ✅  PIPELINE COMPLETADO" if not failed else "   ⚠️  PIPELINE COMPLETADO CON ERRORES")
    print(SEP)
    print(f"  Modelo de tópicos : {backend}")
    print(f"  Paleta            : {args.palette}")
    print(f"  Caché             : {cache_dir}")
    print(f"  Resultados en     : {args.output_dir}/")
    if written:
        print("\n  📁 Archivos de comparación escritos:")
        for path in written:
            print(f"     • {path}")
    if failed:
        print(f"\n  ❌ Tareas con error: {', '.join(failed)}")
    print("\n  🔁 Para reejecutar el análisis completo:")
    print(f"     cd ~/ac_4/code && python run_analysis.py \\")
    print(f"       --csv-dmo {args.csv_dmo} \\")
    for u in args.csv_ugc:
        print(f"       --csv-ugc {u} \\")
    print(f"       --text-col {args.text_col} --lang {args.lang} \\")
    print(f"       --outlier-method {args.outlier_method} --contamination {args.contamination} \\")
    print(f"       --output-dir {args.output_dir} --model {args.model} \\")
    print(f"       --palette {args.palette} --num-topicos {args.num_topicos}")
    print(f"{SEP}\n")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()