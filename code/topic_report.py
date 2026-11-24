"""Bloque de reporte de tópicos para la comparación DMO vs UGC.

Funciones puras: no imprimen y no escriben salvo ``build_topic_report_block``
cuando recibe una ruta CSV explícita.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

from comparison.common import save_csv

CSV_COLUMNS: tuple[str, ...] = (
    "topic_id", "title", "top5_words", "all_keywords", "coverage_pct", "n_docs",
)
MIN_WIDTH: int = 49


def _keywords(value: Any) -> list[str]:
    """Normaliza `keywords_top8` (separado por ';' o ',') a lista de palabras."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    raw = str(value)
    parts = raw.split(";") if ";" in raw else raw.split(",")
    return [p.strip() for p in parts if p.strip()]


def topic_summary_frame(
    topics_summary: pd.DataFrame,
    doc_topics: pd.DataFrame | None = None,
    total_docs: int | None = None,
) -> pd.DataFrame:
    """Construye la tabla canónica del reporte con cobertura por tópico."""
    if topics_summary is None or topics_summary.empty:
        return pd.DataFrame(columns=list(CSV_COLUMNS))

    if doc_topics is not None and not doc_topics.empty:
        counts = doc_topics["topic_id"].value_counts().to_dict()
        n_total = int(len(doc_topics))
    else:
        counts = topics_summary.set_index("topic_id")["n_docs"].to_dict()
        n_total = int(topics_summary["n_docs"].sum())
    if total_docs:
        n_total = int(total_docs)

    rows: list[dict[str, Any]] = []
    for _, row in topics_summary.iterrows():
        words = _keywords(row.get("keywords_top8"))
        topic_id = int(row["topic_id"])
        n_docs = int(counts.get(topic_id, row.get("n_docs", 0) or 0))
        title = str(row.get("topic_name") or (words[0] if words else f"Tópico {topic_id}"))
        rows.append({
            "topic_id": topic_id,
            "title": title,
            "top5_words": ", ".join(words[:5]),
            "all_keywords": ", ".join(words),
            "coverage_pct": round(100 * n_docs / n_total, 2) if n_total else 0.0,
            "n_docs": n_docs,
        })
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def format_topic_console(table: pd.DataFrame) -> str:
    """Renderiza los bloques con el formato exacto ┌─ / │ / └─."""
    if table.empty:
        return "No hay tópicos retenidos para la comparación."

    boxes: list[str] = []
    for _, row in table.iterrows():
        lines = [
            f"Tópico {int(row['topic_id'])}: {row['title']}",
            f"Top 5 palabras: {row['top5_words']}",
            f"Keywords completas: {row['all_keywords']}",
            f"Cobertura: {float(row['coverage_pct']):.2f}% de los comentarios "
            f"(n={int(row['n_docs'])})",
        ]
        width = max(MIN_WIDTH, max(len(line) for line in lines) + 4)
        top = "┌" + "─" * width + "┐"
        bottom = "└" + "─" * width + "┘"
        body = "\n".join(f"│ {line:<{width - 2}} │" for line in lines)
        boxes.append("\n".join([top, body, bottom]))
    return "\n\n".join(boxes)


def topic_report_html(table: pd.DataFrame) -> str:
    """HTML del bloque: <pre> conserva el dibujo de caja del formato pedido."""
    if table.empty:
        return '<div class="placeholder">No hay tópicos retenidos</div>'
    return '<pre class="topic-block">' + html.escape(format_topic_console(table)) + "</pre>"


def build_topic_cards_html(table: pd.DataFrame) -> str:
    """Tarjetas HTML de tópicos; sin caracteres de dibujo de caja."""
    rows = table[table["topic_id"] != -1]
    if rows.empty:
        return '<div class="placeholder">No hay tópicos retenidos</div>'
    cards: list[str] = []
    for _, row in rows.iterrows():
        words = [p.strip() for p in str(row["all_keywords"]).split(",") if p.strip()]
        title = ", ".join(words[:3]) or str(row["title"])
        allkw = "".join(f'<span class="pill-muted">{html.escape(p)}</span>' for p in words)
        cov = float(row["coverage_pct"])
        cards.append(
            '<article class="topic-card">'
            f'<span class="topic-card__chip">Tópico {int(row["topic_id"])}</span>'
            f'<h3 class="topic-card__title">{html.escape(title)}</h3>'
            '<span class="muted">Keywords completas</span>'
            f'<div class="pills">{allkw}</div>'
            '<div class="coverage">'
            f'<div class="coverage__value">{cov:.2f}%</div>'
            f'<div class="coverage__n">(n={int(row["n_docs"])})</div>'
            '</div></article>'
        )
    return '<div class="topic-cards">' + "".join(cards) + "</div>"


def build_topic_report_block(
    topics_summary: pd.DataFrame,
    doc_topics: pd.DataFrame | None = None,
    total_docs: int | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Genera tabla, consola, HTML y (opcionalmente) el CSV del reporte."""
    table = topic_summary_frame(topics_summary, doc_topics, total_docs)
    written = save_csv(table, Path(csv_path)) if csv_path is not None else None
    return {
        "table": table,
        "console": format_topic_console(table),
        "html": build_topic_cards_html(table),
        "csv": written,
    }
