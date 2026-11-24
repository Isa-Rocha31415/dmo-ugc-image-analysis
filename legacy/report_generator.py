"""
report_generator.py — Tarea 'report': genera el informe final HTML.

Lee artefactos existentes sin recalcular modelos. El sentimiento va en una
sección descriptiva separada; Jaccard solo se imprime en consola con
--verbose-jaccard y nunca entra en el bloque principal del dashboard.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from comparison.common import append_metadata, print_header, save_csv  # noqa: E402
from io_corpus import CORPUS_COLUMN, SOURCE_COLUMN  # noqa: E402
from report_sections import (  # noqa: E402
    density_figure_html,
    gaps_sidebar_html,
    inline_plotly_fragment,
    jaccard_console_lines,
    price_section_html,
    sentiment_table_html,
    strip_plotly_updatemenus,
)
from sentiment_labels import sentiment_distribution  # noqa: E402
from utils import palette_to_plot_colors  # noqa: E402


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Lee CSV si existe, None si no."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None


def _make_figure_box(rel_path: Path, results_dir: Path, placeholder: str) -> str:
    """Genera HTML para embeber figura o placeholder."""
    full = results_dir / rel_path
    if full.exists():
        return f'<div class="figure-box"><iframe src="{rel_path}"></iframe></div>'
    return f'<div class="figure-box placeholder">{placeholder}</div>'


def _make_kpi_card(value: str, label: str, sub: str = "") -> str:
    """Tarjeta KPI."""
    sub_html = f'<div class="kpi__sub">{sub}</div>' if sub else ""
    return ('<div class="kpi"><div class="kpi__value">'
            f'{value}</div><div class="kpi__label">{label}</div>{sub_html}</div>')


def _bullet(text: str) -> str:
    return f"<li>{text}</li>"


def _fmt_num(value: object) -> str:
    """Formatea un número para la tabla de métricas globales."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def run_report_task(args: argparse.Namespace, results_dir: Path) -> list[Path]:
    """Genera results/informe_final.html y la tabla descriptiva de polaridad."""
    print_header("PASO 18: REPORTE FINAL HTML")

    template_path = CODE_DIR / "templates" / "informe_final.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Plantilla no encontrada: {template_path}")

    template = template_path.read_text(encoding="utf-8")

    # Leer artefactos (sin recalcular nada)
    base = _safe_read_csv(results_dir / "comparison" / "base" / "analysis_base.csv")
    outliers = _safe_read_csv(results_dir / "comparison" / "base" / "outliers_base.csv")
    summary = _safe_read_csv(results_dir / "comparison" / "topics_joint" / "topics_summary.csv")
    grid = _safe_read_csv(results_dir / "comparison" / "topics_joint" / "topic_selection_grid.csv")
    dist = _safe_read_csv(results_dir / "comparison" / "thematic" / "distribution_distances.csv")
    gaps = _safe_read_csv(results_dir / "comparison" / "thematic" / "gaps_table.csv")
    lex = _safe_read_csv(results_dir / "comparison" / "lexical" / "metrics_lexical.csv")
    price_scores = _safe_read_csv(results_dir / "comparison" / "price" / "price_scores.csv")
    price_rates = _safe_read_csv(results_dir / "comparison" / "price" / "mention_rate_by_source.csv")
    price_comparison = _safe_read_csv(results_dir / "comparison" / "price" / "price_comparison.csv")
    price_dist = _safe_read_csv(results_dir / "comparison" / "price" / "price_distribution_tests.csv")
    price_ct = _safe_read_csv(results_dir / "comparison" / "price" / "price_topic_crosstab.csv")
    price_examples = _safe_read_csv(results_dir / "comparison" / "price" / "price_examples.csv")
    price_alignment = _safe_read_csv(results_dir / "comparison" / "price" / "price_alignment.csv")
    price_density = _safe_read_csv(results_dir / "comparison" / "price" / "price_density.csv")
    gaps_sidebar = _safe_read_csv(results_dir / "comparison" / "thematic" / "gaps_table.csv")

    # Bloques generados por run_analysis.py (n-gramas, tópicos y UMAP 3D).
    def _read_block(name: str) -> str:
        path = results_dir / name
        return (path.read_text(encoding="utf-8") if path.exists()
                else '<div class="placeholder">Sección no disponible en esta corrida</div>')

    topic_report_block = _read_block("topic_report_section.html")
    umap_block = _read_block("umap_section.html")

    kpis: dict[str, str] = {}
    n_total = 0
    n_dmo = 0
    n_ugc = 0
    written: list[Path] = []

    if base is not None:
        n_total = len(base)
        n_dmo = int((base[CORPUS_COLUMN] == "DMO").sum())
        n_ugc = n_total - n_dmo
        kpis["n_docs"] = _make_kpi_card(
            f"{n_total:,}", "Documentos Totales", f"DMO: {n_dmo:,} | UGC: {n_ugc:,}"
        )

    # La tabla descriptiva se sigue guardando como CSV; no se renderiza en el HTML.
    if base is not None and "sentiment_label" in base.columns:
        sent_table = sentiment_distribution(base, ("corpus", SOURCE_COLUMN))
        report_dir = results_dir / "comparison" / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        written.append(save_csv(sent_table, report_dir / "sentiment_descriptive.csv"))
        neutral_pct = 100 * (base["sentiment_label"] == "neutral").mean()
        kpis["sent_neu"] = _make_kpi_card(
            f"{neutral_pct:.1f}%", "Polaridad neutra (descriptiva)",
            "No se usa como prueba de brecha",
        )

    if outliers is not None:
        pct_out = (len(outliers) / n_total * 100) if n_total else 0.0
        kpis["outliers"] = _make_kpi_card(
            f"{len(outliers):,}", "Outliers", f"{pct_out:.1f}% del corpus"
        )

    if dist is not None and not dist.empty:
        d_ugc = dist[dist["scope"] == "ugc_total"]
        if not d_ugc.empty:
            jsd = d_ugc.iloc[0]["jsd_base2"]
            emd = d_ugc.iloc[0]["emd_cosine"]
            kpis["alignment"] = _make_kpi_card(
                f"{(1 - jsd) * 100:.1f}%", "Alineación Semántica", f"JSD₂={jsd:.3f} | EMD={emd:.3f}"
            )

    if grid is not None and not grid.empty:
        sel = grid[grid["selected"] == True]
        if not sel.empty:
            kpis["k_selected"] = _make_kpi_card(
                str(int(sel.iloc[0]["k"])), "Tópicos (k)", "Selección automática"
            )

    # ── Jaccard: consola auxiliar, NUNCA KPI ni bullet del HTML ──
    if lex is not None and not lex.empty and getattr(args, "verbose_jaccard", False):
        print("\n  📎 Validación léxica auxiliar (Jaccard, no incluida en el dashboard):")
        for line in jaccard_console_lines(lex):
            print(line)

    if price_rates is not None and not price_rates.empty:
        p_ugc = price_rates[price_rates["scope"] == "ugc_total"]
        if not p_ugc.empty:
            kpis["price"] = _make_kpi_card(
                f"{p_ugc.iloc[0]['rate'] * 100:.1f}%", "Mención de Precio", "UGC total"
            )

    kpi_cards = "\n".join(kpis.values()) or '<div class="placeholder">Sin datos</div>'

    sent_fig = _make_figure_box(
        Path("sentiment/sentiment_by_source_stacked.html"), results_dir,
        "Figura de sentimiento no disponible",
    )
    sent_hist = _make_figure_box(
        Path("sentiment/sentiment_score_hist.html"), results_dir,
        "Histograma no disponible",
    )
    ngram_html_path = results_dir / "ngram_interactive.html"
    ngram_section = (
        ngram_html_path.read_text(encoding="utf-8")
        if getattr(args, "ngram_interactive", True) and ngram_html_path.exists()
        else '<div class="placeholder">Visualización de n-gramas no disponible</div>'
    )
    topics_fig = _make_figure_box(
        Path("topics/topics_3d_umap.html"), results_dir, "Mapa 3D no disponible",
    )
    lex_dmo = _make_figure_box(
        Path("comparison/lexical/wordcloud_logodds_dmo.html"), results_dir,
        "Wordcloud DMO no disponible",
    )
    lex_ugc = _make_figure_box(
        Path("comparison/lexical/wordcloud_logodds_ugc.html"), results_dir,
        "Wordcloud UGC no disponible",
    )

    doc_topics = _safe_read_csv(results_dir / "comparison" / "topics_joint" / "doc_topics.csv")
    if (gaps_sidebar is None or gaps_sidebar.empty) and doc_topics is not None and base is not None:
        merged = base.merge(doc_topics[["doc_id", "topic_id", "topic_name"]], on="doc_id")
        kept = merged[merged["topic_id"] != -1]
        d_total = int((kept[CORPUS_COLUMN] == "DMO").sum())
        u_total = int((kept[CORPUS_COLUMN] != "DMO").sum())
        rows = []
        for tid in sorted(kept["topic_id"].unique()):
            subset = kept[kept["topic_id"] == tid]
            p_dmo = int((subset[CORPUS_COLUMN] == "DMO").sum()) / d_total if d_total else 0.0
            p_ugc = int((subset[CORPUS_COLUMN] != "DMO").sum()) / u_total if u_total else 0.0
            rows.append({
                "topic_name": subset.iloc[0]["topic_name"],
                "p_dmo": p_dmo, "p_ugc": p_ugc,
                "gap_pp": 100 * (p_ugc - p_dmo),
            })
        gaps_sidebar = pd.DataFrame(rows)
    gaps_html = gaps_sidebar_html(gaps_sidebar)

    scatter_path = results_dir / "comparison" / "price" / "price_scatter_similarity.html"
    scatter_html = ""
    if scatter_path.exists():
        fragment = inline_plotly_fragment(scatter_path.read_text(encoding="utf-8"))
        scatter_html = strip_plotly_updatemenus(fragment)
    umap_html = _make_figure_box(
        Path("comparison/price/price_umap_concept.html"), results_dir,
        "UMAP 3D no disponible",
    )
    price_html = price_section_html(
        price_comparison, price_dist, price_ct, price_examples, price_alignment,
        density_html=density_figure_html(price_density),
        scatter_html=scatter_html, umap_html=umap_html,
    )

    colors = palette_to_plot_colors(args.palette)
    if isinstance(colors, dict):
        primary = colors.get("primary", "#3e6fc0")
        secondary = colors.get("secondary", "#2a9d8f")
        accent = colors.get("accent", "#e0605e")
        dark = colors.get("dark", "#16225e")
    elif isinstance(colors, list) and len(colors) >= 4:
        primary, secondary, accent, dark = colors[:4]
    else:
        primary, secondary, accent, dark = "#3e6fc0", "#2a9d8f", "#e0605e", "#16225e"

    replacements = {
        "{{TITLE}}": args.title,
        "{{DATE}}": datetime.now().strftime("%Y-%m-%d"),
        "{{INPUT_PATH}}": str(getattr(args, "csv_dmo", "DMO")),
        "{{PALETTE_NAME}}": args.palette,
        "{{N_DOCS}}": f"{n_total:,}" if base is not None else "0",
        "{{PRIMARY}}": primary,
        "{{SECONDARY}}": secondary,
        "{{ACCENT}}": accent,
        "{{DARK}}": dark,
        "{{KPI_CARDS}}": kpi_cards,
        "{{SENTIMENT_FIGURE}}": sent_fig,
        "{{SENTIMENT_HIST}}": sent_hist,
        "{{NGRAM_SECTION}}": ngram_section,
        "{{TOPIC_REPORT_BLOCK}}": topic_report_block,
        "{{UMAP_3D_FIGURE}}": umap_block,
        "{{TOPICS_FIGURE}}": topics_fig,
        "{{GAPS_TABLE}}": gaps_html,
        "{{LEXICAL_FIGURES}}": lex_dmo + lex_ugc,
        "{{PRICE_SECTION}}": price_html,
    }
    html_out = template
    for marker, value in replacements.items():
        html_out = html_out.replace(marker, str(value))

    html_path = results_dir / "informe_final.html"
    html_path.write_text(html_out, encoding="utf-8")
    written.append(html_path)
    written.append(append_metadata(
        results_dir / "comparison" / "report" / "run_metadata.csv",
        task="report",
        params={**vars(args), "jaccard_in_html": False,
                "jaccard_verbose": bool(getattr(args, "verbose_jaccard", False)),
                "ngram_interactive": bool(getattr(args, "ngram_interactive", True))},
        repo_dir=CODE_DIR.parent,
    ))
    print(f"  ✅ Informe HTML -> {html_path}")
    for path in written:
        print(f"     • {path}")
    return written
