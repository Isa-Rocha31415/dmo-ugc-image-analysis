"""
final_report.py — Tarea 'report': genera el informe final HTML.

Lee artefactos existentes sin recalcular modelos. Genera KPIs, tablas y
visualizaciones prioritarias. Funciona offline y se degrada elegantemente
si faltan artefactos.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from comparison.common import append_metadata, print_header, save_csv
from io_corpus import CORPUS_COLUMN, SOURCE_COLUMN
from utils import palette_to_plot_colors


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
    return f'<div class="kpi"><div class="kpi__value">{value}</div><div class="kpi__label">{label}</div>{sub_html}</div>'


def _bullet(text: str) -> str:
    return f"<li>{text}</li>"


def run_report_task(args: argparse.Namespace, results_dir: Path) -> list[Path]:
    """Genera results/informe_final.html."""
    print_header("PASO 18: REPORTE FINAL HTML")
    
    template_path = Path(__file__).resolve().parent / "templates" / "informe_final.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Plantilla no encontrada: {template_path}")
    
    template = template_path.read_text(encoding="utf-8")
    
    # Leer artefactos
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
    
    # Calcular KPIs
    kpis = {}
    bullets = []
    
    if base is not None:
        n_total = len(base)
        n_dmo = int((base[CORPUS_COLUMN] == "DMO").sum())
        n_ugc = n_total - n_dmo
        kpis["n_docs"] = _make_kpi_card(f"{n_total:,}", "Documentos Totales", f"DMO: {n_dmo:,} | UGC: {n_ugc:,}")
        bullets.append(f"Se analizaron <b>{n_total:,}</b> documentos: <b>{n_dmo:,}</b> de la DMO y <b>{n_ugc:,}</b> de fuentes percibidas.")
        
        if "sentiment_label" in base.columns:
            dmo = base[base[CORPUS_COLUMN] == "DMO"]
            ugc = base[base[CORPUS_COLUMN] == "UGC"]
            pos_dmo = (dmo["sentiment_label"] == "positive").mean() * 100
            pos_ugc = (ugc["sentiment_label"] == "positive").mean() * 100
            neg_dmo = (dmo["sentiment_label"] == "negative").mean() * 100
            neg_ugc = (ugc["sentiment_label"] == "negative").mean() * 100
            kpis["sent_pos"] = _make_kpi_card(f"{pos_dmo:.1f}%", "Sentimiento Positivo", f"DMO vs UGC: {pos_ugc:.1f}%")
            kpis["sent_neg"] = _make_kpi_card(f"{neg_dmo:.1f}%", "Sentimiento Negativo", f"DMO vs UGC: {neg_ugc:.1f}%")
            gap = pos_dmo - pos_ugc
            bullets.append(f"La imagen proyectada tiene <b>{pos_dmo:.1f}%</b> de comentarios positivos frente a <b>{pos_ugc:.1f}%</b> del UGC (brecha: {gap:+.1f} puntos).")
    
    if outliers is not None:
        n_out = len(outliers)
        pct_out = (n_out / (n_total if base is not None else 1)) * 100
        kpis["outliers"] = _make_kpi_card(f"{n_out:,}", "Outliers", f"{pct_out:.1f}% del corpus")
    
    if dist is not None and not dist.empty:
        d_ugc = dist[dist["scope"] == "ugc_total"]
        if not d_ugc.empty:
            jsd = d_ugc.iloc[0]["jsd_base2"]
            emd = d_ugc.iloc[0]["emd_cosine"]
            alignment = (1 - jsd) * 100
            kpis["alignment"] = _make_kpi_card(f"{alignment:.1f}%", "Alineación Semántica", f"JSD₂={jsd:.3f} | EMD={emd:.3f}")
            if jsd < 0.05:
                bullets.append("Las distribuciones temáticas DMO y UGC están <b>muy alineadas</b> (JSD₂ < 0.05).")
            elif jsd < 0.15:
                bullets.append("Las distribuciones temáticas DMO y UGC están <b>moderadamente distintas</b> (JSD₂ entre 0.05 y 0.15).")
            else:
                bullets.append("Las distribuciones temáticas DMO y UGC están <b>marcadamente distintas</b> (JSD₂ > 0.15).")
    
    if grid is not None and not grid.empty:
        sel = grid[grid["selected"] == True]
        if not sel.empty:
            k_sel = int(sel.iloc[0]["k"])
            kpis["k_selected"] = _make_kpi_card(str(k_sel), "Tópicos (k)", "Selección automática")
    
    if gaps is not None and not gaps.empty:
        g_ugc = gaps[gaps["scope"] == "ugc_total"]
        if not g_ugc.empty:
            n_sobre_dmo = int((g_ugc["verdict"] == "SOBRE_DMO").sum())
            n_sobre_ugc = int((g_ugc["verdict"] == "SOBRE_UGC").sum())
            n_sin_dif = int((g_ugc["verdict"] == "SIN_DIF").sum())
            bullets.append(f"De los tópicos analizados: <b>{n_sobre_dmo}</b> sobre-representados en DMO, <b>{n_sobre_ugc}</b> sobre-representados en UGC, <b>{n_sin_dif}</b> sin diferencia significativa.")
    
    if lex is not None and not lex.empty:
        l_ugc = lex[lex["scope"] == "ugc_total"]
        if not l_ugc.empty:
            j100_freq = l_ugc[(l_ugc["metric"] == "jaccard_freq") & (l_ugc["k"] == 100)]
            j100_tfidf = l_ugc[(l_ugc["metric"] == "jaccard_tfidf") & (l_ugc["k"] == 100)]
            if not j100_freq.empty:
                j = j100_freq.iloc[0]["value"]
                kpis["jaccard"] = _make_kpi_card(f"{j:.3f}", "Jaccard@100 (freq)")
            if not j100_tfidf.empty:
                j_t = j100_tfidf.iloc[0]["value"]
                bullets.append(f"La similitud léxica (Jaccard@100 TF-IDF) entre DMO y UGC es <b>{j_t:.3f}</b>.")
    
    if price_rates is not None and not price_rates.empty:
        p_ugc = price_rates[price_rates["scope"] == "ugc_total"]
        if not p_ugc.empty:
            rate_dmo = p_ugc.iloc[0]["rate"] * 100
            rate_ugc = p_ugc.iloc[0]["rate"] * 100
            kpis["price"] = _make_kpi_card(f"{rate_dmo:.1f}%", "Mención de Precio", f"DMO vs UGC: {rate_ugc:.1f}%")
            bullets.append(f"El concepto precio/valor/costo es mencionado en <b>{rate_dmo:.1f}%</b> de la DMO y <b>{rate_ugc:.1f}%</b> del UGC.")
    
    # Generar HTML
    kpi_cards = "\n".join(kpis.values()) if kpis else '<div class="placeholder">Sin datos</div>'
    exec_bullets = "\n".join([_bullet(b) for b in bullets]) if bullets else '<li class="placeholder">Sin datos</li>'
    
    sent_fig = _make_figure_box(
        Path("sentiment/sentiment_by_source_stacked.html"),
        results_dir,
        "Figura de sentimiento no disponible"
    )
    sent_hist = _make_figure_box(
        Path("sentiment/sentiment_score_hist.html"),
        results_dir,
        "Histograma no disponible"
    )
    topics_fig = _make_figure_box(
        Path("topics/topics_3d_umap.html"),
        results_dir,
        "Mapa 3D no disponible"
    )
    k_sel_fig = _make_figure_box(
        Path("comparison/topics_joint/curvas_k.html"),
        results_dir,
        "Curvas de selección no disponibles"
    )
    lex_dmo = _make_figure_box(
        Path("comparison/lexical/wordcloud_logodds_dmo.html"),
        results_dir,
        "Wordcloud DMO no disponible"
    )
    lex_ugc = _make_figure_box(
        Path("comparison/lexical/wordcloud_logodds_ugc.html"),
        results_dir,
        "Wordcloud UGC no disponible"
    )
    
    # Tabla de gaps
    gaps_html = '<div class="placeholder">No hay datos de brechas temáticas</div>'
    if gaps is not None and not gaps.empty:
        g_ugc = gaps[gaps["scope"] == "ugc_total"].copy()
        if not g_ugc.empty:
            g_ugc["abs_gap"] = g_ugc["gap_pp"].abs()
            top = g_ugc.nlargest(10, "abs_gap")
            rows = []
            for _, r in top.iterrows():
                rows.append(f"""<tr>
                    <td>{r['topic_name']}</td>
                    <td>{r['p_dmo']:.4f}</td>
                    <td>{r['p_ugc']:.4f}</td>
                    <td>{r['gap_pp']:+.4f}</td>
                    <td>{r['verdict']}</td>
                </tr>""")
            gaps_html = f"""
            <div class="table-wrap">
                <table class="table">
                    <thead><tr><th>Tópico</th><th>% DMO</th><th>% UGC</th><th>GAP</th><th>Veredicto</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
            """
    
    # Sección precio
    price_html = '<div class="placeholder">No hay datos de precio/valor</div>'
    if price_scores is not None and not price_scores.empty:
        price_fig = _make_figure_box(
            Path("comparison/price/price_scatter_valence.html"),
            results_dir,
            "Scatter de valencia no disponible"
        )
        tau_html = ""
        if tau_cal is not None and not tau_cal.empty:
            tau_val = tau_cal.iloc[0]["tau"]
            tau_html = f"<p><b>τ (umbral):</b> {tau_val:.4f}</p>"
        price_html = tau_html + price_fig
    
    # Comentarios representativos
    rep_html = '<div class="placeholder">No hay comentarios representativos disponibles</div>'
    doc_topics = _safe_read_csv(results_dir / "comparison" / "topics_joint" / "doc_topics.csv")
    if doc_topics is not None and base is not None and not doc_topics.empty:
        merged = base.merge(doc_topics[["doc_id", "topic_id", "topic_name"]], on="doc_id")
        if not merged.empty:
            rep_rows = []
            for tid in merged["topic_id"].unique()[:5]:
                subset = merged[merged["topic_id"] == tid]
                if not subset.empty:
                    row = subset.iloc[0]
                    text = str(row["text"])[:200] + "..." if len(str(row["text"])) > 200 else str(row["text"])
                    rep_rows.append(f"""<tr>
                        <td>{row['topic_name']}</td>
                        <td>{row[SOURCE_COLUMN]}</td>
                        <td>{row.get('sentiment_label', 'N/A')}</td>
                        <td>{text}</td>
                    </tr>""")
            if rep_rows:
                rep_html = f"""
                <div class="table-wrap">
                    <table class="table">
                        <thead><tr><th>Tópico</th><th>Fuente</th><th>Sentimiento</th><th>Texto (truncado)</th></tr></thead>
                        <tbody>{''.join(rep_rows)}</tbody>
                    </table>
                </div>
                """
    
    # Apéndice: todas las figuras
    all_figs = []
    for html_file in results_dir.rglob("*.html"):
        if "_cache" not in str(html_file) and html_file.name != "informe_final.html":
            rel = html_file.relative_to(results_dir)
            csv_file = html_file.with_name(html_file.stem + "_data.csv")
            csv_link = f'<a href="{csv_file.relative_to(results_dir)}">CSV</a>' if csv_file.exists() else ""
            all_figs.append(f'<tr><td>{rel.parent}</td><td>{html_file.name}</td><td><a href="{rel}">Abrir</a></td><td>{csv_link}</td></tr>')
    
    all_figures_html = '<div class="placeholder">No hay figuras generadas</div>'
    if all_figs:
        all_figures_html = f"""
        <div class="table-wrap">
            <table class="table">
                <thead><tr><th>Sección</th><th>Archivo</th><th>Enlace</th><th>Datos</th></tr></thead>
                <tbody>{''.join(all_figs)}</tbody>
            </table>
        </div>
        """
    
    # Metodología
    meth_html = f"""
    <ul class="bullets">
        <li><b>Embeddings:</b> {args.embedding_model}</li>
        <li><b>Modelo de tópicos:</b> {args.model} con selección automática de k</li>
        <li><b>Seed:</b> {args.seed}</li>
        <li><b>Bootstrap:</b> B={args.bootstrap}</li>
        <li><b>Tareas ejecutadas:</b> {', '.join(args.comparison_tasks)}</li>
        <li><b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
    </ul>
    """
    
    # Paleta
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
    
    # Reemplazar marcadores
    html_out = template
    html_out = html_out.replace("{{TITLE}}", args.title)
    html_out = html_out.replace("{{DATE}}", datetime.now().strftime('%Y-%m-%d'))
    html_out = html_out.replace("{{N_DOCS}}", f"{n_total:,}" if base is not None else "0")
    html_out = html_out.replace("{{PRIMARY}}", primary)
    html_out = html_out.replace("{{SECONDARY}}", secondary)
    html_out = html_out.replace("{{ACCENT}}", accent)
    html_out = html_out.replace("{{DARK}}", dark)
    html_out = html_out.replace("{{KPI_CARDS}}", kpi_cards)
    html_out = html_out.replace("{{EXECUTIVE_BULLETS}}", exec_bullets)
    html_out = html_out.replace("{{SENTIMENT_FIGURE}}", sent_fig)
    html_out = html_out.replace("{{SENTIMENT_HIST}}", sent_hist)
    html_out = html_out.replace("{{TOPICS_FIGURE}}", topics_fig)
    html_out = html_out.replace("{{K_SELECTION}}", k_sel_fig)
    html_out = html_out.replace("{{GAPS_TABLE}}", gaps_html)
    html_out =