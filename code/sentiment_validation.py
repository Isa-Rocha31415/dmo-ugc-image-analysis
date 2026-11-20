"""
sentiment_validation.py
──────────────────────────────────────────────────────────────
TAREA 3 — Validación del análisis de sentimiento (PASO 4/5/6/7).

Tres fuentes de evidencia:

1. Etiquetas débiles externas (rating de TripAdvisor 1–5):
   1–2 → negative, 3 → neutral (se excluye), 4–5 → positive.
   Accuracy, macro-F1, F1 por clase, matriz de confusión y MCC
   (global y desglosado por fuente → domain shift).

2. Gold set humano opcional (`--gold-csv`, OFF): desactivado por defecto;
   la corrida estándar usa solo validación automática por rating.

3. Correlación score↔rating (Spearman) y MAE en escala de 1–5 estrellas:
   evidencia de que el score ordena bien las reseñas, sin anotar nada.

4. Ablación (opcional): léxico / TF-IDF+LogReg / embeddings+LogReg
   (linear probe) / transformer afinado.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from metrics.sentiment_ablation import DEFAULT_FINETUNE_MODEL, run_ablation
from metrics.sentiment_metrics import evaluate, evaluate_by_group

NEGATIVE_MAX_RATING = 2
POSITIVE_MIN_RATING = 4
LABELS = ("negative", "positive")


# ══════════════════════════════════════════════════════════════════════
#  Construcción de etiquetas
# ══════════════════════════════════════════════════════════════════════
def rating_to_label(rating: Any) -> Optional[str]:
    """Convierte un rating 1–5 en etiqueta binaria (None = neutral)."""
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None

    if np.isnan(value):
        return None
    if value <= NEGATIVE_MAX_RATING:
        return "negative"
    if value >= POSITIVE_MIN_RATING:
        return "positive"
    return None


def build_weak_labels(
    df: pd.DataFrame,
    rating_column: str = "rating",
) -> pd.DataFrame:
    """
    Añade la columna `weak_label` a partir del rating externo.
    Solo para las fuentes que tienen rating (p. ej. TripAdvisor).
    """
    out = df.copy()

    if rating_column not in out.columns:
        out["weak_label"] = None
        out["rating_numeric"] = np.nan
        return out

    ratings = pd.to_numeric(out[rating_column], errors="coerce")
    out["rating_numeric"] = ratings
    out["weak_label"] = ratings.map(rating_to_label)

    return out


def load_gold_labels(
    gold_csv: str,
    text_column: str = "text",
    label_column: str = "sentiment_label",
) -> pd.DataFrame:
    """
    Lee un CSV de etiquetas humanas (columnas de texto y etiqueta).
    """
    gold = pd.read_csv(gold_csv)

    text_col = text_column if text_column in gold.columns else (
        "text" if "text" in gold.columns else gold.columns[0]
    )
    label_col = label_column if label_column in gold.columns else (
        "sentiment_label" if "sentiment_label" in gold.columns else gold.columns[1]
    )

    out = pd.DataFrame(
        {
            "gold_text": gold[text_col].astype(str),
            "gold_label": gold[label_col].astype(str).str.lower().map(
                lambda x: "positive" if x.startswith(("pos", "posi")) else "negative"
            ),
        }
    )
    return out


# ══════════════════════════════════════════════════════════════════════
#  Validación externa (score vs rating)
# ══════════════════════════════════════════════════════════════════════
def validate_against_rating(
    df: pd.DataFrame,
    score_column: str = "sentiment_score",
    rating_column: str = "rating_numeric",
) -> Dict[str, Any]:
    """
    Spearman (¿el score ordena bien?) y MAE en escala 1–5 estrellas.
    """
    if score_column not in df.columns or rating_column not in df.columns:
        return {"n": 0, "spearman": None, "mae_stars": None}

    scores = pd.to_numeric(df[score_column], errors="coerce")
    ratings = pd.to_numeric(df[rating_column], errors="coerce")

    mask = scores.notna() & ratings.notna()
    n = int(mask.sum())

    if n < 3:
        return {"n": n, "spearman": None, "mae_stars": None}

    x = scores[mask].to_numpy(dtype=float)
    y = ratings[mask].to_numpy(dtype=float)

    # score (-1..1) → estrellas (1..5)
    stars = np.clip(3.0 + 2.0 * x, 1.0, 5.0)
    mae = float(np.mean(np.abs(stars - y)))

    try:
        from scipy.stats import spearmanr

        rho, p_value = spearmanr(x, y)
        spearman = float(rho)
        p = float(p_value)
    except Exception:
        spearman, p = None, None

    return {
        "n": n,
        "spearman": spearman,
        "p_value": p,
        "mae_stars": mae,
        "score_mean": float(np.mean(x)),
        "rating_mean": float(np.mean(y)),
    }


# ══════════════════════════════════════════════════════════════════════
#  Orquestador
# ══════════════════════════════════════════════════════════════════════
def validar_sentimiento(
    df: pd.DataFrame,
    text_column: str,
    source_column: str,
    label_column: str = "sentiment_label",
    score_column: str = "sentiment_score",
    rating_column: str = "rating",
    gold_csv: Optional[str] = None,
    embeddings: Optional[np.ndarray] = None,
    with_ablation: bool = False,
    with_finetune: bool = False,
    finetune_model: str = DEFAULT_FINETUNE_MODEL,
    output_dir: str = "results/sentiment_validation",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Ejecuta la validación completa y escribe JSON + CSV + HTML.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Índice compacto: los embeddings que llegan están alineados a estas filas.
    df_val = build_weak_labels(df.reset_index(drop=True), rating_column=rating_column)

    results: Dict[str, Any] = {
        "n_rows": int(len(df_val)),
        "modelo_pipeline": (
            df_val["sentiment_backend"].iloc[0]
            if "sentiment_backend" in df_val.columns and len(df_val)
            else "desconocido"
        ),
    }

    # ── 1. Validación con etiquetas débiles externas ──
    weak = df_val[df_val["weak_label"].notna()].copy()

    if len(weak) >= 10:
        y_true = weak["weak_label"].tolist()
        y_pred = weak[label_column].tolist()

        results["weak_labels"] = {
            "descripcion": (
                "Etiqueta débil externa: rating 1-2 = negative, "
                "4-5 = positive (3 = neutral, excluido)."
            ),
            "global": evaluate(y_true, y_pred, LABELS),
            "por_fuente": evaluate_by_group(y_true, y_pred, weak[source_column].tolist(), LABELS),
        }
    else:
        results["weak_labels"] = {
            "descripcion": "Sin suficientes filas con rating externo.",
            "n": int(len(weak)),
        }

    # ── 2. Gold set humano (opcional) ──
    if gold_csv:
        gold = load_gold_labels(gold_csv, text_column=text_column)
        merged = df_val.assign(_text=df_val[text_column].astype(str)).merge(
            gold, left_on="_text", right_on="gold_text", how="inner"
        )

        if len(merged) >= 5:
            results["gold"] = {
                "csv": str(gold_csv),
                "n": int(len(merged)),
                "global": evaluate(
                    merged["gold_label"].tolist(),
                    merged[label_column].tolist(),
                    LABELS,
                ),
                "por_fuente": evaluate_by_group(
                    merged["gold_label"].tolist(),
                    merged[label_column].tolist(),
                    merged[source_column].tolist(),
                    LABELS,
                ),
            }
        else:
            results["gold"] = {
                "csv": str(gold_csv),
                "n": int(len(merged)),
                "mensaje": "No hubo suficientes coincidencias texto→etiqueta.",
            }

    # ── 3. Validación externa score↔rating ──
    results["validacion_externa"] = validate_against_rating(df_val, score_column)

    # ── 4. Ablación ──
    if with_ablation and len(weak) >= 50:
        textos = weak[text_column].astype(str).tolist()
        labels = weak["weak_label"].tolist()

        emb = None
        if embeddings is not None:
            positions = weak.index.to_numpy()
            if positions.max() < len(embeddings):
                emb = np.asarray(embeddings)[positions]

        results["ablacion"] = run_ablation(
            texts=textos,
            labels=labels,
            embeddings=emb,
            include_finetune=with_finetune,
            finetune_model=finetune_model,
            seed=random_state,
        )

    # ── 5. Persistencia ──
    _guardar_json(results, out_dir / "sentiment_validation.json")
    _guardar_csvs(results, out_dir)
    _guardar_html(results, out_dir)

    return results


# ══════════════════════════════════════════════════════════════════════
#  Persistencia
# ══════════════════════════════════════════════════════════════════════
def _guardar_json(results: Dict[str, Any], path: Path) -> None:
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_default)


def _guardar_csvs(results: Dict[str, Any], out_dir: Path) -> None:
    weak = results.get("weak_labels", {})

    if "global" in weak:
        matrix = weak["global"]["confusion_matrix"]
        labels = weak["global"]["labels"]
        df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
        df_cm.to_csv(out_dir / "confusion_matrix_weak_labels.csv")

        rows = []
        for fuente, metrics in weak.get("por_fuente", {}).items():
            rows.append(
                {
                    "fuente": fuente,
                    "n": metrics["n"],
                    "accuracy": round(metrics["accuracy"], 4),
                    "macro_f1": round(metrics["macro_f1"], 4),
                    "mcc": round(metrics["mcc"], 4),
                    "f1_negative": round(metrics["f1_per_class"].get("negative", 0.0), 4),
                    "f1_positive": round(metrics["f1_per_class"].get("positive", 0.0), 4),
                }
            )
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "metrics_by_source.csv", index=False)

    ablation = results.get("ablacion", [])
    if ablation:
        pd.DataFrame(ablation).to_csv(out_dir / "ablation.csv", index=False)


def _guardar_html(results: Dict[str, Any], out_dir: Path) -> None:
    """Reporte HTML autocontenido (tablas + matriz de confusión)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    weak = results.get("weak_labels", {})
    html_path = out_dir / "sentiment_validation.html"

    if "global" not in weak:
        html_path.write_text(
            "<html><body><h1>Validación de sentimiento</h1>"
            "<p>No hay datos suficientes con rating externo para validar.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        return

    global_metrics = weak["global"]
    labels = global_metrics["labels"]
    matrix = global_metrics["confusion_matrix"]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"pred: {lab}" for lab in labels],
            y=[f"real: {lab}" for lab in labels],
            text=[[str(v) for v in row] for row in matrix],
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig_cm.update_layout(
        title="Matriz de confusión (etiqueta débil externa)",
        xaxis_title="Predicho",
        yaxis_title="Real",
    )

    rows = []
    for fuente, metrics in weak.get("por_fuente", {}).items():
        rows.append(
            f"<tr><td>{fuente}</td><td>{metrics['n']:,}</td>"
            f"<td>{metrics['accuracy']:.3f}</td><td>{metrics['macro_f1']:.3f}</td>"
            f"<td>{metrics['mcc']:.3f}</td>"
            f"<td>{metrics['f1_per_class'].get('negative', 0):.3f}</td>"
            f"<td>{metrics['f1_per_class'].get('positive', 0):.3f}</td></tr>"
        )

    ext = results.get("validacion_externa", {})
    ext_rows = "".join(
        f"<li>{key}: <b>{value}</b></li>"
        for key, value in ext.items()
        if key in {"n", "spearman", "p_value", "mae_stars"}
    )

    ablation_html = ""
    ablation = results.get("ablacion", [])
    if ablation:
        body = "".join(
            f"<tr><td>{row.get('modelo', '?')}</td>"
            f"<td>{row.get('accuracy', float('nan')):.3f}</td>"
            f"<td>{row.get('macro_f1', float('nan')):.3f}</td>"
            f"<td>{row.get('mcc', float('nan')):.3f}</td></tr>"
            for row in ablation
        )
        ablation_html = f"""
        <h2>Ablación (accuracy del embedding)</h2>
        <table><tr><th>Modelo</th><th>Accuracy</th><th>Macro-F1</th><th>MCC</th></tr>
        {body}</table>
        """

    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Validación del análisis de sentimiento</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 32px; color: #1A202C; }}
 table {{ border-collapse: collapse; margin: 12px 0 28px 0; }}
 th, td {{ border: 1px solid #CBD5E0; padding: 6px 12px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 h1, h2 {{ color: #2B6CB0; }}
 .meta {{ background: #F7FAFC; border-left: 4px solid #2B6CB0; padding: 10px 16px; }}
</style></head><body>
<h1>Validación del análisis de sentimiento</h1>
<div class="meta">
  <p><b>Modelo del pipeline:</b> {results.get('modelo_pipeline', '?')} &nbsp;|&nbsp;
     <b>Filas evaluadas:</b> {results.get('n_rows', 0):,}</p>
  <p>{weak.get('descripcion', '')}</p>
</div>

<h2>Métricas globales (etiqueta débil externa)</h2>
<table>
 <tr><th>Métrica</th><th>Valor</th></tr>
 <tr><td>Accuracy</td><td>{global_metrics['accuracy']:.4f}</td></tr>
 <tr><td>Macro-F1</td><td>{global_metrics['macro_f1']:.4f}</td></tr>
 <tr><td>MCC</td><td>{global_metrics['mcc']:.4f}</td></tr>
 <tr><td>F1 negative</td><td>{global_metrics['f1_per_class'].get('negative', 0):.4f}</td></tr>
 <tr><td>F1 positive</td><td>{global_metrics['f1_per_class'].get('positive', 0):.4f}</td></tr>
 <tr><td>Accuracy (mayoría)</td><td>{global_metrics['accuracy_balanced_baseline']:.4f}</td></tr>
 <tr><td>n</td><td>{global_metrics['n']:,}</td></tr>
</table>

<h2>Por fuente (domain shift)</h2>
<table>
 <tr><th>Fuente</th><th>n</th><th>Accuracy</th><th>Macro-F1</th><th>MCC</th>
     <th>F1 neg.</th><th>F1 pos.</th></tr>
 {''.join(rows)}
</table>

<h2>Validación externa (score vs rating)</h2>
<ul>{ext_rows}</ul>
{ablation_html}

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<div id="cm"></div>
<script>
  var cm = {fig_cm.to_json()};
  Plotly.newPlot('cm', cm.data, cm.layout);
</script>
</body></html>
"""

    html_path.write_text(html, encoding="utf-8")
