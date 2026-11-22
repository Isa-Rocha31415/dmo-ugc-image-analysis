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
    generic_table_html,
    jaccard_console_lines,
    run_metadata_html,
    sentiment_table_html,
    stat_tests_html,
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
    tau_cal = _safe_read_csv(results_dir / "comparison" / "price" / "tau_calibration.csv")
    price_valence = _safe_read_csv(results_dir / "comparison" / "price" / "price_valence_conditioned.csv")
    run_meta = _safe_read_csv(results_dir / "comparison" / "base" / "run_metadata.csv")

    # Bloques generados por run_analysis.py (n-gramas, tópicos y UMAP 3D).
    def _read_block(name: str) -> str:
        path = results_dir / name
        return (path.read_text(encoding="utf-8") if path.exists()
                else '<div class="placeholder">Sección no disponible en esta corrida</div>')

    topic_report_block = _read_block("topic_report_section.html")
    umap_block = _read_block("umap_section.html")

    kpis: dict[str, str] = {}
    bullets: list[str] = []
    n_total = 0
    n_dmo = 0
    n_ugc = 0
    k_value = "—"
    jsd_value: object = None
    emd_value: object = None
    rate_value: object = None

    if base is not None:
        n_total = len(base)
        n_dmo = int((base[CORPUS_COLUMN] == "DMO").sum())
        n_ugc = n_total - n_dmo
        kpis["n_docs"] = _make_kpi_card(
            f"{n_total:,}", "Documentos Totales", f"DMO: {n_dmo:,} | UGC: {n_ugc:,}"
        )
        bullets.append(
            f"Se analizaron <b>{n_total:,}</b> documentos: <b>{n_dmo:,}</b> de la DMO "
            f"y <b>{n_ugc:,}</b> de fuentes percibidas."
        )

    # ── Sentimiento: SOLO tabla descriptiva, nunca evidencia de brecha ──
    sent_html = '<div class="placeholder">Sin datos de sentimiento</div>'
    written: list[Path] = []
    if base is not None and "sentiment_label" in base.columns:
        sent_table = sentiment_distribution(base, ("corpus", SOURCE_COLUMN))
        sent_html = sentiment_table_html(sent_table)
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
            jsd_value, emd_value = jsd, emd
            kpis["alignment"] = _make_kpi_card(
                f"{(1 - jsd) * 100:.1f}%", "Alineación Semántica", f"JSD₂={jsd:.3f} | EMD={emd:.3f}"
            )
            if jsd < 0.05:
                bullets.append("Las distribuciones temáticas DMO y UGC están <b>muy alineadas</b> (JSD₂ &lt; 0.05).")
            elif jsd < 0.15:
                bullets.append("Las distribuciones temáticas DMO y UGC están <b>moderadamente distintas</b> (JSD₂ entre 0.05 y 0.15).")
            else:
                bullets.append("Las distribuciones temáticas DMO y UGC están <b>marcadamente distintas</b> (JSD₂ &gt; 0.15).")

    if grid is not None and not grid.empty:
        sel = grid[grid["selected"] == True]
        if not sel.empty:
            k_value = str(int(sel.iloc[0]["k"]))
            kpis["k_selected"] = _make_kpi_card(k_value, "Tópicos (k)", "Selección automática")

    if gaps is not None and not gaps.empty:
        g_ugc = gaps[gaps["scope"] == "ugc_total"]
        if not g_ugc.empty:
            bullets.append(
                f"De los tópicos analizados: <b>{int((g_ugc['verdict'] == 'SOBRE_DMO').sum())}</b> "
                f"sobre-representados en DMO, <b>{int((g_ugc['verdict'] == 'SOBRE_UGC').sum())}</b> "
                f"sobre-representados en UGC, <b>{int((g_ugc['verdict'] == 'SIN_DIF').sum())}</b> "
                "sin diferencia significativa."
            )

    # ── Jaccard: consola auxiliar, NUNCA KPI ni bullet del HTML ──
    if lex is not None and not lex.empty and getattr(args, "verbose_jaccard", False):
        print("\n  📎 Validación léxica auxiliar (Jaccard, no incluida en el dashboard):")
        for line in jaccard_console_lines(lex):
            print(line)

    if price_rates is not None and not price_rates.empty:
        p_ugc = price_rates[price_rates["scope"] == "ugc_total"]
        if not p_ugc.empty:
            rate = p_ugc.iloc[0]["rate"] * 100
            rate_value = p_ugc.iloc[0]["rate"]
            kpis["price"] = _make_kpi_card(f"{rate:.1f}%", "Mención de Precio", "UGC total")
            bullets.append(
                f"El concepto precio/valor/costo es mencionado en <b>{rate:.1f}%</b> del UGC."
            )

    kpi_cards = "\n".join(kpis.values()) or '<div class="placeholder">Sin datos</div>'
    exec_bullets = "\n".join(_bullet(b) for b in bullets) or '<li class="placeholder">Sin datos</li>'

    # ── a) Tabla de métricas globales del resumen ejecutivo ──
    exec_metrics = pd.DataFrame([
        {"Métrica": "Documentos en análisis", "Valor": f"{n_total:,}"},
        {"Métrica": "Documentos DMO", "Valor": f"{n_dmo:,}"},
        {"Métrica": "Documentos UGC", "Valor": f"{n_ugc:,}"},
        {"Métrica": "Outliers excluidos", "Valor": f"{len(outliers) if outliers is not None else 0:,}"},
        {"Métrica": "Tópicos seleccionados (k)", "Valor": k_value},
        {"Métrica": "JSD₂ (temático)", "Valor": _fmt_num(jsd_value)},
        {"Métrica": "EMD coseno", "Valor": _fmt_num(emd_value)},
        {"Métrica": "Tasa de mención de precio", "Valor": _fmt_num(rate_value)},
    ])
    exec_metrics_html = generic_table_html(
        exec_metrics, {"Métrica": "Métrica", "Valor": "Valor"}
    )

    stat_tests = stat_tests_html(gaps, price_valence)
    run_metadata_section = run_metadata_html(
        run_meta,
        flags={
            "paleta": args.palette,
            "seed": args.seed,
            "bootstrap": args.bootstrap,
            "ngram_interactive": getattr(args, "ngram_interactive", True),
            "umap_3d": getattr(args, "umap_3d", True),
            "generate_wordclouds": getattr(args, "generate_wordclouds", False),
            "verbose_jaccard": getattr(args, "verbose_jaccard", False),
        },
        sample_sizes={
            "documentos": n_total,
            "DMO": n_dmo,
            "UGC": n_ugc,
            "outliers": len(outliers) if outliers is not None else 0,
        },
    )

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
    k_sel_fig = _make_figure_box(
        Path("comparison/topics_joint/curvas_k.html"), results_dir,
        "Curvas de selección no disponibles",
    )
    lex_dmo = _make_figure_box(
        Path("comparison/lexical/wordcloud_logodds_dmo.html"), results_dir,
        "Wordcloud DMO no disponible",
    )
    lex_ugc = _make_figure_box(
        Path("comparison/lexical/wordcloud_logodds_ugc.html"), results_dir,
        "Wordcloud UGC no disponible",
    )

    gaps_html = '<div class="placeholder">No hay datos de brechas temáticas</div>'
    if gaps is not None and not gaps.empty:
        g_ugc = gaps[gaps["scope"] == "ugc_total"].copy()
        if not g_ugc.empty:
            g_ugc["abs_gap"] = g_ugc["gap_pp"].abs()
            rows = []
            for _, r in g_ugc.nlargest(10, "abs_gap").iterrows():
                rows.append(
                    f"<tr><td>{r['topic_name']}</td><td>{r['p_dmo']:.4f}</td>"
                    f"<td>{r['p_ugc']:.4f}</td><td>{r['gap_pp']:+.4f}</td>"
                    f"<td>{r['verdict']}</td></tr>"
                )
            gaps_html = (
                '<div class="table-wrap"><table class="table"><thead><tr>'
                '<th>Tópico</th><th>% DMO</th><th>% UGC</th><th>GAP</th><th>Veredicto</th>'
                f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            )

    price_html = '<div class="placeholder">No hay datos de precio/valor</div>'
    if price_scores is not None and not price_scores.empty:
        price_fig = _make_figure_box(
            Path("comparison/price/price_scatter_valence.html"), results_dir,
            "Scatter de valencia no disponible",
        )
        tau_html = ""
        if tau_cal is not None and not tau_cal.empty:
            tau_html = f"<p><b>τ (umbral):</b> {tau_cal.iloc[0]['tau']:.4f}</p>"
        price_html = tau_html + price_fig

    rep_html = '<div class="placeholder">No hay comentarios representativos disponibles</div>'
    doc_topics = _safe_read_csv(results_dir / "comparison" / "topics_joint" / "doc_topics.csv")
    if doc_topics is not None and base is not None and not doc_topics.empty:
        merged = base.merge(doc_topics[["doc_id", "topic_id", "topic_name"]], on="doc_id")
        if not merged.empty:
            rep_rows = []
            for tid in merged["topic_id"].unique()[:5]:
                subset = merged[merged["topic_id"] == tid]
                if subset.empty:
                    continue
                row = subset.iloc[0]
                text = str(row["text"])
                text = text[:200] + "..." if len(text) > 200 else text
                rep_rows.append(
                    f"<tr><td>{row['topic_name']}</td><td>{row[SOURCE_COLUMN]}</td>"
                    f"<td>{row.get('sentiment_label', 'N/A')}</td><td>{text}</td></tr>"
                )
            if rep_rows:
                rep_html = (
                    '<div class="table-wrap"><table class="table"><thead><tr>'
                    '<th>Tópico</th><th>Fuente</th><th>Sentimiento</th><th>Texto (truncado)</th>'
                    f'</tr></thead><tbody>{"".join(rep_rows)}</tbody></table></div>'
                )

    all_figs = []
    for html_file in sorted(results_dir.rglob("*.html")):
        if "_cache" not in str(html_file) and html_file.name != "informe_final.html":
            rel = html_file.relative_to(results_dir)
            csv_file = html_file.with_name(html_file.stem + "_data.csv")
            csv_link = (f'<a href="{csv_file.relative_to(results_dir)}">CSV</a>'
                        if csv_file.exists() else "")
            all_figs.append(
                f'<tr><td>{rel.parent}</td><td>{html_file.name}</td>'
                f'<td><a href="{rel}">Abrir</a></td><td>{csv_link}</td></tr>'
            )
    all_figures_html = (
        '<div class="table-wrap"><table class="table"><thead><tr>'
        '<th>Sección</th><th>Archivo</th><th>Enlace</th><th>Datos</th>'
        f'</tr></thead><tbody>{"".join(all_figs)}</tbody></table></div>'
        if all_figs else '<div class="placeholder">No hay figuras generadas</div>'
    )

    meth_html = (
        '<ul class="bullets">'
        f'<li><b>Embeddings:</b> {args.embedding_model}</li>'
        f'<li><b>Modelo de tópicos:</b> {args.model}</li>'
        f'<li><b>Seed:</b> {args.seed}</li>'
        f'<li><b>Bootstrap:</b> B={args.bootstrap}</li>'
        f'<li><b>Tareas ejecutadas:</b> {", ".join(args.comparison_tasks)}</li>'
        f'<li><b>Fecha:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</li>'
        '</ul>'
    )
    lim_html = (
        '<ul class="bullets">'
        '<li>La polaridad POS/NEU/NEG es <b>descriptiva</b>: no prueba la brecha DMO vs UGC.</li>'
        '<li>El análisis comparativo usa únicamente comentarios normales (sin outliers).</li>'
        '</ul>'
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
        "{{EXECUTIVE_BULLETS}}": exec_bullets,
        "{{EXECUTIVE_METRICS_TABLE}}": exec_metrics_html,
        "{{SENTIMENT_FIGURE}}": sent_fig,
        "{{SENTIMENT_HIST}}": sent_hist,
        "{{SENTIMENT_DESCRIPTIVE_TABLE}}": sent_html,
        "{{NGRAM_SECTION}}": ngram_section,
        "{{TOPIC_REPORT_BLOCK}}": topic_report_block,
        "{{UMAP_3D_FIGURE}}": umap_block,
        "{{TOPICS_FIGURE}}": topics_fig,
        "{{K_SELECTION_FIGURE}}": k_sel_fig,
        "{{GAPS_TABLE}}": gaps_html,
        "{{STAT_TESTS_TABLE}}": stat_tests,
        "{{RUN_METADATA_SECTION}}": run_metadata_section,
        "{{LEXICAL_FIGURES}}": lex_dmo + lex_ugc,
        "{{PRICE_SECTION}}": price_html,
        "{{REPRESENTATIVE_COMMENTS}}": rep_html,
        "{{ALL_FIGURES_TABLE}}": all_figures_html,
        "{{METHODOLOGY_SECTION}}": meth_html,
        "{{LIMITATIONS_SECTION}}": lim_html,
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
