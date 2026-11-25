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
                         10 capa léxica L1 (cobertura/log-odds; Jaccard solo en consola)
                         11 capa temática L2 (gaps, χ²/Fisher, BH, JSD/EMD)
                         18 informe HTML (incluye n-gramas interactivos)
                         (semantic, affective, figures: PENDIENTE; huecos 15/17)

Uso (una sola corrida, desde ~/ac_4/code/):
    python run_analysis.py \
        --csv-dmo "../data/df_dmo.csv" \
        --csv-ugc "../data/df_ugc.csv" \
        --text-col text --lang es \
        --outlier-method IF --contamination auto \
        --min-partition 50 \
        --output-dir ../results \
        --model xidoo --palette ejecutiva --num-topicos automatico

Caché: <output-dir>/cache/<paso>/<clave>/. La segunda corrida reutiliza lo que no cambió.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
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

# Tareas de comparación. Los pasos 15 y 17 quedan como hueco documentado.
# Crossover 5k: 0.58x con pools -> paralelo solo bajo petición explícita.
AUTO_PARALLEL = False

COMPARISON_TASKS = [
    "base", "topics", "lexical", "thematic", "price",
    "semantic", "affective", "figures", "report",
]
COMPARISON_STEP = {
    "base": 8, "topics": 9, "lexical": 10, "thematic": 11, "price": 12,
    "semantic": 13, "affective": 14, "figures": 16, "report": 18,
}

_TIMING_MARKS: list[tuple[str, float]] = []


def _mark_step(step: str) -> None:
    """Registra un hito monotónico para el informe de tiempos por paso."""
    _TIMING_MARKS.append((step, time.perf_counter()))


def _write_timings(output_dir: str) -> Path:
    """Escribe <output-dir>/run_timings.csv con columnas (step,seconds)."""
    rows = [
        {"step": name, "seconds": round(t1 - t0, 3)}
        for (name, t0), (_, t1) in zip(_TIMING_MARKS, _TIMING_MARKS[1:])
    ]
    path = Path(output_dir) / "run_timings.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["step", "seconds"]).to_csv(
        path, index=False, encoding="utf-8-sig")
    print(f"  ⏱️  Tiempos por paso -> {path}")
    return path


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


def _asegurar_sentimiento_ternario(df: pd.DataFrame, neutral_tau: float = 0.15) -> pd.DataFrame:
    """Reetiqueta la caché binaria como POS/NEU/NEG sin recalcular el modelo."""
    from sentiment_labels import apply_sentiment_labels

    return apply_sentiment_labels(df, neutral_tau=neutral_tau)


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


def _comparison_worker(task: str, args: argparse.Namespace, output_dir: Path):
    """Ejecuta una tarea de comparación en un proceso hijo y devuelve logs + metadata."""
    import time

    import torch
    from comparison.common import take_metadata

    torch.set_num_threads(2)
    os.environ["PERF2_BUFFER_METADATA"] = "1"
    os.environ["PERF2_COMPARISON_SERIAL"] = "1"
    subargs = copy.copy(args)
    subargs.comparison_tasks = [task]
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        written, failed = _run_comparison(subargs, output_dir, None, None, None, None)
    return task, written, failed, buf.getvalue(), time.perf_counter() - t0, take_metadata()


def _write_ngram_interactive(
    args: argparse.Namespace,
    output_dir: Path,
    df_clean: pd.DataFrame,
    df_outliers: pd.DataFrame,
) -> list[Path]:
    """Genera el bloque HTML de n-gramas con hover de comentarios normalizados."""
    if not args.ngram_interactive:
        return []

    from comparison.common import append_metadata
    from ngram_visualization import build_all_ngram_figures
    from report_sections import ngram_section_html

    sep = "█" * 70
    print(f"\n{sep}")
    print("   PASO 7.5: VISUALIZACIÓN INTERACTIVA DE N-GRAMAS")
    print(sep)

    figures = build_all_ngram_figures(
        df_clean, df_outliers, palette=args.palette
    )
    block = ngram_section_html(figures)
    path = output_dir / "ngram_interactive.html"
    path.write_text(block, encoding="utf-8")

    print(f"  ✅ Uni/bi/trigramas con muestras en hover -> {path}")
    print(f"     Normales: {len(df_clean):,} | Outliers: {len(df_outliers):,}")
    written = [path]
    written.append(append_metadata(
        output_dir / "comparison" / "base" / "run_metadata.csv",
        task="ngram_interactive",
        params={
            "top_n": 15, "max_samples": 3, "sample_chars": 120,
            "text_column": "text_norm", "palette": args.palette, "seed": args.seed,
        },
        repo_dir=Path(__file__).resolve().parents[1],
    ))
    return written


def _write_topic_report_block(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    """Bloques de tópicos (consola, HTML y CSV canónico) desde artefactos cacheados."""
    from comparison.common import append_metadata, save_csv
    from topic_report import build_topic_report_block

    topics_dir = output_dir / "comparison" / "topics_joint"
    summary_path = topics_dir / "topics_summary.csv"
    if not summary_path.exists():
        print("  ⚠️ Sin topics_summary.csv; bloque de tópicos omitido.")
        return []

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    doc_path = topics_dir / "doc_topics.csv"
    doc_topics = (pd.read_csv(doc_path, encoding="utf-8-sig")
                  if doc_path.exists() else None)
    base_path = output_dir / "comparison" / "base" / "analysis_base.csv"
    total_docs = None
    if base_path.exists():
        total_docs = len(pd.read_csv(base_path, encoding="utf-8-sig"))

    block = build_topic_report_block(summary, doc_topics, total_docs=total_docs)
    sep = "█" * 70
    print(f"\n{sep}")
    print("   PASO 9.5: BLOQUES DE TÓPICOS")
    print(sep)
    print(block["console"])

    csv_path = save_csv(
        block["table"], output_dir / "comparison" / "report" / "topics_summary.csv")
    html_path = output_dir / "topic_report_section.html"
    html_path.write_text(block["html"], encoding="utf-8")
    print(f"  ✅ Bloques de tópicos -> {html_path}")
    print(f"  ✅ Tabla canónica -> {csv_path}")
    meta = append_metadata(
        output_dir / "comparison" / "base" / "run_metadata.csv",
        task="topic_report",
        params={"seed": args.seed, "n_topics": len(block["table"]),
                "total_docs": total_docs},
        repo_dir=Path(__file__).resolve().parents[1],
    )
    return [csv_path, html_path, meta]


def _write_umap_section(
    args: argparse.Namespace,
    output_dir: Path,
    embeddings_clean: np.ndarray | None,
) -> list[Path]:
    """Scatter UMAP 3D con embeddings cacheados y tabla de puntos exportada."""
    import plotly.io as pio
    from comparison.common import append_metadata, save_csv
    from visualization import build_umap_3d_plot

    sep = "█" * 70
    html_path = output_dir / "umap_section.html"
    if not args.umap_3d:
        html_path.write_text(
            '<div class="placeholder">UMAP 3D desactivado (--no-umap-3d)</div>',
            encoding="utf-8",
        )
        return [html_path]

    print(f"\n{sep}")
    print("   PASO 9.6: DISTRIBUCIÓN SEMÁNTICA (UMAP 3D)")
    print(sep)

    base_path = output_dir / "comparison" / "base" / "analysis_base.csv"
    doc_path = output_dir / "comparison" / "topics_joint" / "doc_topics.csv"
    if not base_path.exists() or not doc_path.exists():
        html_path.write_text(
            '<div class="placeholder">UMAP 3D no disponible: faltan base o tópicos</div>',
            encoding="utf-8",
        )
        print("  ⚠️ UMAP 3D omitido: faltan analysis_base.csv o doc_topics.csv.")
        return [html_path]
    if embeddings_clean is None:
        npy_path = output_dir / "comparison" / "base" / "embeddings_clean.npy"
        if not npy_path.exists():
            html_path.write_text(
                '<div class="placeholder">UMAP 3D no disponible: faltan embeddings</div>',
                encoding="utf-8",
            )
            print("  ⚠️ UMAP 3D omitido: no hay embeddings cacheados.")
            return [html_path]
        embeddings_clean = np.load(npy_path)

    base = pd.read_csv(base_path, encoding="utf-8-sig")
    doc_topics = pd.read_csv(doc_path, encoding="utf-8-sig")
    if "topic_name" not in doc_topics.columns:
        doc_topics["topic_name"] = doc_topics["topic_id"].astype(str)
    df = base.merge(doc_topics[["doc_id", "topic_name"]], on="doc_id", how="inner")
    emb = np.asarray(embeddings_clean)
    if "embedding_row" in df.columns:
        rows = df["embedding_row"].astype(int).to_numpy()
        if rows.max() >= len(emb):
            raise ValueError("embedding_row fuera de rango en embeddings_clean.npy")
        emb = emb[rows]
    elif len(emb) >= len(df):
        emb = emb[:len(df)]
    else:
        raise ValueError("Los embeddings no cubren los comentarios del análisis.")

    fig = build_umap_3d_plot(
        df,
        emb,
        text_column="text_norm" if "text_norm" in df.columns else "text",
        source_column="source" if "source" in df.columns else "corpus",
        topic_column="topic_name",
        sentiment_column="sentiment_label" if "sentiment_label" in df.columns else "corpus",
        random_state=args.seed,
    )
    html_path.write_text(
        pio.to_html(fig, full_html=False, include_plotlyjs="cdn"), encoding="utf-8")

    records = []
    for trace in fig.data:
        for x, y, z, custom in zip(trace.x, trace.y, trace.z, trace.customdata):
            records.append({
                "topic": custom[0], "source": custom[1],
                "sentiment": custom[2], "text": custom[3],
                "umap_1": float(x), "umap_2": float(y), "umap_3": float(z),
            })
    points = pd.DataFrame(records, columns=[
        "topic", "source", "sentiment", "text", "umap_1", "umap_2", "umap_3"])
    csv_path = save_csv(
        points, output_dir / "comparison" / "report" / "umap_3d_points.csv")

    print(f"  ✅ Scatter 3D -> {html_path}")
    print(f"  ✅ Tabla de puntos -> {csv_path}")
    print(f"     Comentarios proyectados: {len(df):,} | Tópicos: {len(fig.data)}")
    meta = append_metadata(
        output_dir / "comparison" / "base" / "run_metadata.csv",
        task="umap_3d",
        params={"n_components": 3, "random_state": args.seed,
                "n_neighbors": 15, "n_points": len(df)},
        repo_dir=Path(__file__).resolve().parents[1],
    )
    return [html_path, csv_path, meta]


def _run_comparison(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    df_clean: pd.DataFrame,
    df_outliers: pd.DataFrame,
    embeddings_clean: np.ndarray | None = None,
) -> tuple[list[Path], list[str]]:
    """Ejecuta las tareas de comparación (PASO 8+) con los objetos ya calculados."""
    from comparison.common import print_header

    tasks = [t for t in COMPARISON_TASKS if t in set(args.comparison_tasks)]
    written: list[Path] = []
    failed: list[str] = []

    if (not os.environ.get("PERF2_COMPARISON_SERIAL")
            and not os.environ.get("PERF2_SERIAL")
            and bool({t for t in tasks if t in ("lexical", "thematic", "price")})
            and ((AUTO_PARALLEL and len(df_clean) >= 5000)
                 or getattr(args, "force_parallel", False))):
        par = [t for t in tasks if t in ("lexical", "thematic", "price")]
        seq = [t for t in tasks if t not in par]
        if par:
            os.environ["PERF2_COMPARISON_SERIAL"] = "1"
            try:
                if seq:
                    sub = copy.copy(args)
                    sub.comparison_tasks = seq
                    w, f = _run_comparison(
                        sub, output_dir, df, embeddings, df_clean, df_outliers)
                    written += w
                    failed += f
                workers = min(3, max(1, (os.cpu_count() or 1) // 2))
                print(f"  ⚡ Comparación en paralelo: {workers} procesos "
                      "(léxica, temática, precio).")
                try:
                    with ProcessPoolExecutor(
                            max_workers=workers,
                            mp_context=get_context("spawn")) as pool:
                        results = list(pool.map(
                            _comparison_worker, par,
                            [args] * len(par), [output_dir] * len(par)))
                except Exception as exc:
                    print(f"  ⚠️ Fallo el pool de comparación ({exc}); reintento secuencial.")
                    results = [_comparison_worker(t, args, output_dir) for t in par]
                from comparison.common import replay_metadata

                for t, ww, ff, logs, secs, meta in results:
                    print(f"\n  📄 {t.upper()} (paralelo {secs:.1f}s)")
                    print(logs, end="")
                    written += ww
                    failed += ff
                    replay_metadata(meta)
                return written, failed
            finally:
                os.environ.pop("PERF2_COMPARISON_SERIAL", None)
    elif (tasks and not os.environ.get("PERF2_SERIAL")
          and not os.environ.get("PERF2_COMPARISON_SERIAL")):
        print("  🐢 Comparación secuencial: corpus pequeño: modo serial "
              "(overhead de pools no justificado).")

    for task in tasks:
        try:
            _mark_step(f"PASO {COMPARISON_STEP[task]} {task}")
            if task == "base":
                from comparison.base_builder import run_base_task
                written += run_base_task(args, output_dir, inputs={
                    "df": df, "embeddings": embeddings,
                    "df_clean": df_clean, "df_outliers": df_outliers,
                })
            elif task == "topics":
                from joint_topic_model import run_topics_task
                written += run_topics_task(args, output_dir)
                written += _write_topic_report_block(args, output_dir)
                written += _write_umap_section(args, output_dir, embeddings_clean)
            elif task == "lexical":
                from metrics_lexical import run_lexical_task
                written += run_lexical_task(args, output_dir)
            elif task == "thematic":
                from metrics_thematic import run_thematic_task
                written += run_thematic_task(args, output_dir)
            elif task == "price":
                from price_concept import run_price_task
                written += run_price_task(args, output_dir)
            elif task == "report":
                legacy_dir = Path(__file__).resolve().parents[1] / "legacy"
                if str(legacy_dir) not in sys.path:
                    sys.path.insert(0, str(legacy_dir))
                from report_generator import run_report_task
                written += run_report_task(args, output_dir)
            else:
                print_header(f"PASO {COMPARISON_STEP[task]}: {task.upper()} (PENDIENTE)")
                print("  ⏳ Se implementará en la siguiente fase del plan (P4+).")
        except Exception as exc:
            failed.append(task)
            print(f"\n  ❌ Error en la tarea '{task}': {exc}")
            traceback.print_exc()
    _mark_step("PASO comparison-fin")
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
    parser.add_argument("--title", default="Imagen proyectada (DMO) vs imagen percibida (UGC)",
                        help="Título del informe HTML")
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
    parser.add_argument("--sent-max-length", type=int, default=512,
                        help="Máximo de tokens para sentimiento (default científico 512)")
    parser.add_argument("--sentiment-model", default=None,
                        help="Modelo HF para el análisis de sentimiento")
    parser.add_argument("--sentiment-neutral-tau", type=float, default=0.15,
                        help="Banda |score| < tau para la clase NEUTRO (default 0.15; "
                             "no recalcula la caché de sentimiento)")
    parser.add_argument("--verbose-jaccard", action="store_true",
                        help="Imprime Jaccard auxiliar en consola al generar el informe; "
                             "nunca se muestra en el HTML")

    # ── Caché ──
    parser.add_argument("--cache-dir", default=None,
                        help="Directorio de caché (por defecto <output-dir>/cache)")
    parser.add_argument("--no-cache", action="store_true", help="Ignorar la caché")
    parser.add_argument("--force-steps", nargs="+", default=[], choices=list(CACHE_STEPS),
                        help="Pasos a recalcular aunque exista caché")

    # ── Validación de sentimiento ──
    parser.add_argument("--no-sentiment-validation", dest="sentiment_validation",
                        action="store_false", help="Omitir la validación del sentimiento")
    parser.add_argument("--gold-csv", default=None,
                        help="Opcional, OFF: CSV con etiquetas humanas")
    parser.add_argument("--sentiment-ablation", action="store_true",
                        help="Opcional, OFF: ablación léxico / TF-IDF / linear probe")
    parser.add_argument("--sentiment-finetune", action="store_true",
                        help="Opcional, OFF: transformer afinado (costoso)")
    parser.add_argument("--finetune-model", default="pysentimiento/robertuito-sentiment-analysis",
                        help="Modelo a afinar en la ablación")

    # ── BLOQUE II: comparación DMO vs UGC (pasos 8+) ──
    g = parser.add_argument_group("Comparación DMO vs UGC (pasos 8+)")
    g.add_argument("--ngram-interactive", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Genera n-gramas interactivos con muestras en hover (default ON). "
                        "Desactivar con --no-ngram-interactive")
    g.add_argument("--umap-3d", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Scatter UMAP 3D de comentarios por tópico (default ON). "
                        "Requiere umap-learn; desactivar con --no-umap-3d")
    g.add_argument("--generate-wordclouds", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Genera las imágenes PNG de wordclouds (default ON). "
                        "Desactivar con --no-wordclouds")
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
    g.add_argument("--stability-runs", type=int, default=3,
                   help="Corridas de estabilidad (antes 5; pasar 5 para restaurar)")
    g.add_argument("--subsample", type=float, default=0.75,
                   help="Fracción por corrida (antes 0.8; pasar 0.8 para restaurar)")
    g.add_argument("--force-parallel", action="store_true",
                   help="Fuerza pools aunque el corpus sea pequeño")
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
    pg.add_argument("--price-placebo-docs", type=int, default=500,
                    help="Alias legacy de --price-sample-size (docs aleatorios del placebo)")
    pg.add_argument("--price-sample-size", "--price-validation-n", dest="price_sample_size",
                    type=int, default=500,
                    help="Docs aleatorios del corpus para calibrar τ por placebo (F4.3)")
    pg.add_argument("--price-annotation-file", default=None,
                    help="Legacy ignorado: la validación es automática, sin anotación humana")
    pg.add_argument("--price-top-k", type=int, default=5,
                    help="Top-k ejemplos por corpus/fuente (F4.4)")

    parser.set_defaults(per_source=True, sentiment_validation=True)
    args = parser.parse_args()
    if args.force_parallel:
        os.environ["PERF2_FORCE_PARALLEL"] = "1"

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
    _mark_step("PASO 0")
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
    _mark_step("PASO 1")
    print(f"\n{SEP}")
    print("   PASO 1: PREPROCESAMIENTO DE TEXTO")
    print(SEP)
    from text_preprocessing import prepare_corpus

    df, prep_stats = prepare_corpus(df, args.text_col, args.lang)
    # Huella del módulo + idioma + params: cambia el texto y caduca la caché aguas abajo.
    prep_fp = hashlib.sha256(
        (Path(__file__).with_name("text_preprocessing.py").read_text()
         + args.lang + "|min_len=5|natural+norm").encode("utf-8")
    ).hexdigest()[:12]
    print(f"  Comentarios válidos tras limpieza: {len(df):,} "
          f"(de {prep_stats['input_rows']:,})")
    print("  Filas descartadas por fuente:")
    for k, v in (prep_stats["removed_by_source"] or {"(ninguna)": 0}).items():
        print(f"    {k:<25s} {v:>6,d}")
    print("  Filas descartadas por corpus:")
    for k, v in (prep_stats["removed_by_corpus"] or {"(ninguna)": 0}).items():
        print(f"    {k:<25s} {v:>6,d}")
    print(f"  Columna text_norm creada para la capa léxica (huella {prep_fp}).")

    # ══ PASO 2: Embeddings ══
    _mark_step("PASO 2")
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
        preprocess_fingerprint=prep_fp,
    )
    print(f"  Embeddings generados: {embeddings.shape}")

    # ══ PASO 3: Outliers + N-gramas ══
    _mark_step("PASO 3")
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
    _mark_step("PASO 4")
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
        sentiment_max_length=args.sent_max_length,
        cache_dir=cache_dir,
        parent_key=outliers_meta["key"],
        use_cache=use_cache,
        force=_force("sentiment"),
    )
    df_clean = _asegurar_sentimiento_ternario(df_clean, args.sentiment_neutral_tau)
    print(f"  Sentimiento aplicado a {len(df_clean):,} comentarios normales.")
    print(f"  Rango sentiment_score: [{df_clean['sentiment_score'].min():.3f}, "
          f"{df_clean['sentiment_score'].max():.3f}]")
    print(f"  Distribución: {df_clean['sentiment_label'].value_counts().to_dict()}")

    df_outliers = annotate_sentiment(
        df_outliers, text_column=args.text_col,
        source_column=SOURCE_COLUMN, lang=args.lang, backend="transformers",
        model_name=args.sentiment_model, max_length=args.sent_max_length,
        neutral_tau=args.sentiment_neutral_tau,
    )
    df_outliers = _asegurar_sentimiento_ternario(df_outliers, args.sentiment_neutral_tau)
    print(f"  Sentimiento aplicado a {len(df_outliers):,} outliers.")
    print(f"  Distribución outliers: {df_outliers['sentiment_label'].value_counts().to_dict()}")

    # ── Alineación documento↔embedding (por contenido; robusta ante cache hit) ──
    embeddings_clean = _alinear_embeddings_clean(df, df_clean, df_outliers, embeddings)
    if embeddings_clean is not None:
        print(f"  ✅ Embeddings alineados con df_clean: {embeddings_clean.shape}")

    # ══ PASO 4.5: Validación del sentimiento ══
    _mark_step("PASO 4.5")
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
    _mark_step("PASO 5")
    from sentiment_report import reporte_sentimiento_por_fuente, reporte_sentimiento_outliers

    print("  ℹ️  Polaridad POS/NEU/NEG: métrica descriptiva, NO prueba de la brecha DMO vs UGC.")
    reporte_sentimiento_por_fuente(
        df_clean,
        source_column=SOURCE_COLUMN,
        output_dir=f"{args.output_dir}/sentiment",
        palette=colors,
        title="Sentimiento por Fuente (comentarios normales)",
    )

    # ══ PASO 6: Sentimiento de outliers ══
    _mark_step("PASO 6")
    reporte_sentimiento_outliers(
        df_outliers,
        source_column=SOURCE_COLUMN,
        output_dir=f"{args.output_dir}/sentiment",
        palette=colors,
        title="Sentimiento de Outliers",
    )

    # ══ PASO 6.5: Wordclouds por sentimiento ══
    _mark_step("PASO 6.5")
    print(f"\n{SEP}")
    print("   PASO 6.5: WORDCLOUDS POR SENTIMIENTO")
    print(SEP)
    # [OPTIMIZACIÓN-2026] Wordclouds activadas por defecto; --no-wordclouds las omite.
    if args.generate_wordclouds:
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
        print(f"  ✅ Wordclouds PNG -> {args.output_dir}/sentiment/wordclouds")
    else:
        print("  ⏭️  Wordclouds omitidas (--no-wordclouds)")

    # ══ PASO 7: Modelado de tópicos por sentimiento ══
    _mark_step("PASO 7")
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
        ngram_written = _write_ngram_interactive(args, output_dir, df_clean, df_outliers)
        comparison_written, failed = _run_comparison(
            args, output_dir, df, embeddings, df_clean, df_outliers, embeddings_clean
        )
        written += ngram_written + comparison_written

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
    print(f"       --text-col {args.text_col} --lang {args.lang} --title \"{args.title}\" \\")
    print(f"       --sentiment-neutral-tau {args.sentiment_neutral_tau} \\")
    print("       --ngram-interactive \\" if args.ngram_interactive
          else "       --no-ngram-interactive \\")
    print(f"       --outlier-method {args.outlier_method} --contamination {args.contamination} \\")
    print(f"       --output-dir {args.output_dir} --model {args.model} \\")
    print(f"       --palette {args.palette} --num-topicos {args.num_topicos}")
    print(f"{SEP}\n")

    _mark_step("FIN")
    _write_timings(args.output_dir)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()