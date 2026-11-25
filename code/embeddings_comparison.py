"""
embeddings_comparison.py
─────────────────────────────────────────────────────────────
Comparativa de embeddings:
  • BERT (sentence-transformers / transformers)
  • XidooTopic (embeddings internos)
Se comparan con:
  - Similitud coseno DMO vs percibido
  - Gráfico UMAP/PCA coloreado por fuente
  - Métricas de separación
Todo se imprime en consola.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# ──────────────────────────────────────────────
# Generar embeddings BERT
# ──────────────────────────────────────────────
def obtener_embeddings_bert(
    textos: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
    device: int = -1,
) -> np.ndarray:
    """
    Genera embeddings con sentence-transformers (offline si ya está descargado).
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embs = model.encode(
            textos,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(embs, dtype=np.float32)
    except ImportError:
        # Fallback: transformers puro
        from transformers import AutoTokenizer, AutoModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        all_embs = []
        for i in range(0, len(textos), batch_size):
            batch = textos[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            )
            with torch.no_grad():
                out = model(**enc)
            # Mean pooling
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            # Normalizar
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.numpy())
        return np.vstack(all_embs).astype(np.float32)


# ──────────────────────────────────────────────
# Obtener embeddings internos de XidooTopic
# ──────────────────────────────────────────────
def obtener_embeddings_xidoo(
    textos: list[str],
    random_state: int = 42,
) -> np.ndarray:
    """Embeddings reales de XidooTopic; cualquier fallo termina la corrida."""
    import traceback

    from xidoo_topic_modeler import XidooTopicModeler

    try:
        model = XidooTopicModeler(random_state=random_state)
        model.fit_transform(textos)
        return np.asarray(model.embeddings_, dtype=np.float32)
    except Exception as exc:
        print(f"  ❌ XidooTopic (embeddings) falló: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise RuntimeError("No se obtuvo embeddings de XidooTopic; sin fallback.") from exc

# ──────────────────────────────────────────────
# Reducción 2D
# ──────────────────────────────────────────────
def reducir_2d(
    embeddings: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
) -> np.ndarray:
    """Reduce a 2D con UMAP o PCA."""
    if method == "umap":
        try:
            import umap
            red = umap.UMAP(n_components=2, random_state=random_state)
            return red.fit_transform(embeddings)
        except ImportError:
            pass
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(embeddings)


# ──────────────────────────────────────────────
# Comparativa completa
# ──────────────────────────────────────────────
def comparar_embeddings(
    df: pd.DataFrame,
    text_column: str,
    source_column: str,
    dmo_label: str,
    output_dir: str = "results/embeddings_comparison",
    bert_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    reduction: str = "pca",
) -> Dict[str, Any]:
    """
    Compara BERT vs XidooTopic e imprime todo en consola.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    textos = df[text_column].fillna("").astype(str).tolist()
    fuentes = df[source_column].astype(str).tolist()
    mask_dmo = df[source_column].astype(str) == str(dmo_label)

    SEP = "═" * 70

    # ──────────────────────────────────
    # A) EMBEDDINGS BERT
    # ──────────────────────────────────
    print(f"\n{SEP}")
    print("   🤖  GENERANDO EMBEDDINGS BERT")
    print(f"{SEP}")
    emb_bert = obtener_embeddings_bert(textos, model_name=bert_model)
    print(f"  Modelo: {bert_model}")
    print(f"  Forma: {emb_bert.shape}")

    # Similitud coseno DMO vs percibido
    emb_dmo_bert = emb_bert[mask_dmo.values]
    emb_perc_bert = emb_bert[~mask_dmo.values]
    mean_dmo_bert = emb_dmo_bert.mean(axis=0).reshape(1, -1)
    mean_perc_bert = emb_perc_bert.mean(axis=0).reshape(1, -1)
    cos_bert = cosine_similarity(mean_dmo_bert, mean_perc_bert)[0, 0]

    print(f"  Comentarios DMO: {len(emb_dmo_bert):,}")
    print(f"  Comentarios percibidos: {len(emb_perc_bert):,}")
    print(f"  Similitud coseno (DMO vs percibido): {cos_bert:.4f}")

    # Similitud por fuente vs DMO
    print(f"\n  ── Similitud coseno de cada fuente vs DMO (BERT) ──")
    fuentes_unicas = sorted(set(fuentes))
    cos_por_fuente_bert = {}
    for f in fuentes_unicas:
        if f == str(dmo_label):
            continue
        mask_f = np.array([fu == f for fu in fuentes])
        if mask_f.sum() == 0:
            continue
        mean_f = emb_bert[mask_f].mean(axis=0).reshape(1, -1)
        sim = cosine_similarity(mean_dmo_bert, mean_f)[0, 0]
        cos_por_fuente_bert[f] = sim
        print(f"    {f:<25s} vs DMO: {sim:.4f}")

    # ──────────────────────────────────
    # B) EMBEDDINGS XIDOOTOPIC
    # ──────────────────────────────────
    print(f"\n{SEP}")
    print("   🔬  GENERANDO EMBEDDINGS XIDOOTOPIC")
    print(f"{SEP}")
    emb_xidoo = obtener_embeddings_xidoo(textos)
    print(f"  Forma: {emb_xidoo.shape}")

    emb_dmo_xidoo = emb_xidoo[mask_dmo.values]
    emb_perc_xidoo = emb_xidoo[~mask_dmo.values]
    mean_dmo_xidoo = emb_dmo_xidoo.mean(axis=0).reshape(1, -1)
    mean_perc_xidoo = emb_perc_xidoo.mean(axis=0).reshape(1, -1)
    cos_xidoo = cosine_similarity(mean_dmo_xidoo, mean_perc_xidoo)[0, 0]

    print(f"  Similitud coseno (DMO vs percibido): {cos_xidoo:.4f}")

    print(f"\n  ── Similitud coseno de cada fuente vs DMO (XidooTopic) ──")
    cos_por_fuente_xidoo = {}
    for f in fuentes_unicas:
        if f == str(dmo_label):
            continue
        mask_f = np.array([fu == f for fu in fuentes])
        if mask_f.sum() == 0:
            continue
        mean_f = emb_xidoo[mask_f].mean(axis=0).reshape(1, -1)
        sim = cosine_similarity(mean_dmo_xidoo, mean_f)[0, 0]
        cos_por_fuente_xidoo[f] = sim
        print(f"    {f:<25s} vs DMO: {sim:.4f}")

    # ──────────────────────────────────
    # C) TABLA COMPARATIVA
    # ──────────────────────────────────
    print(f"\n{SEP}")
    print("   📋  TABLA COMPARATIVA: BERT vs XIDOOTOPIC")
    print(f"{SEP}")
    print(f"  {'Métrica':<40s} {'BERT':>10s} {'XidooTopic':>12s}")
    print(f"  {'─'*40} {'─'*10} {'─'*12}")
    print(f"  {'Dimensiones':<40s} {emb_bert.shape[1]:>10,d} {emb_xidoo.shape[1]:>12,d}")
    print(f"  {'Similitud DMO vs Percibido':<40s} {cos_bert:>10.4f} {cos_xidoo:>12.4f}")
    print(f"  {'|Δ| (1 - similitud)':<40s} {1-cos_bert:>10.4f} {1-cos_xidoo:>12.4f}")

    for f in fuentes_unicas:
        if f == str(dmo_label):
            continue
        cb = cos_por_fuente_bert.get(f, 0)
        cx = cos_por_fuente_xidoo.get(f, 0)
        print(f"  {'Fuente: ' + f:<40s} {cb:>10.4f} {cx:>12.4f}")

    print(f"{SEP}\n")

    # ──────────────────────────────────
    # D) VISUALIZACIONES
    # ──────────────────────────────────
    df_viz = df[[source_column, text_column]].copy()
    df_viz["is_dmo"] = df_viz[source_column].astype(str).apply(
        lambda x: "DMO" if x == str(dmo_label) else "Percibido"
    )

    # BERT 2D
    coords_bert = reducir_2d(emb_bert, method=reduction)
    df_viz["bert_x"] = coords_bert[:, 0]
    df_viz["bert_y"] = coords_bert[:, 1]

    fig_bert = px.scatter(
        df_viz, x="bert_x", y="bert_y",
        color=source_column,
        symbol="is_dmo",
        title=f"Embeddings BERT ({reduction.upper()}) – {bert_model}",
        labels={"bert_x": "Dim 1", "bert_y": "Dim 2"},
        hover_data=[text_column],
        height=550,
    )
    fig_bert.update_traces(marker=dict(size=6, opacity=0.7))
    fig_bert.update_layout(template="plotly_white")
    fig_bert.write_html(str(out / "embeddings_bert_2d.html"), include_plotlyjs="cdn")

    # XidooTopic 2D
    coords_xidoo = reducir_2d(emb_xidoo, method=reduction)
    df_viz["xidoo_x"] = coords_xidoo[:, 0]
    df_viz["xidoo_y"] = coords_xidoo[:, 1]

    fig_xidoo = px.scatter(
        df_viz, x="xidoo_x", y="xidoo_y",
        color=source_column,
        symbol="is_dmo",
        title=f"Embeddings XidooTopic ({reduction.upper()})",
        labels={"xidoo_x": "Dim 1", "xidoo_y": "Dim 2"},
        hover_data=[text_column],
        height=550,
    )
    fig_xidoo.update_traces(marker=dict(size=6, opacity=0.7))
    fig_xidoo.update_layout(template="plotly_white")
    fig_xidoo.write_html(str(out / "embeddings_xidoo_2d.html"), include_plotlyjs="cdn")

    # ── Gráfico comparativo de similitudes ──
    comp_df = pd.DataFrame({
        "Fuente": list(cos_por_fuente_bert.keys()),
        "BERT": list(cos_por_fuente_bert.values()),
        "XidooTopic": list(cos_por_fuente_xidoo.values()),
    })
    comp_df["DMO_global_BERT"] = cos_bert
    comp_df["DMO_global_Xidoo"] = cos_xidoo

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="BERT", x=comp_df["Fuente"], y=comp_df["BERT"],
        marker_color="#0072B2",
    ))
    fig_comp.add_trace(go.Bar(
        name="XidooTopic", x=comp_df["Fuente"], y=comp_df["XidooTopic"],
        marker_color="#D55E00",
    ))
    fig_comp.update_layout(
        title="Similitud coseno vs DMO: BERT vs XidooTopic",
        barmode="group",
        yaxis_title="Similitud coseno",
        template="plotly_white",
        height=450,
    )
    fig_comp.write_html(str(out / "comparacion_similitudes.html"), include_plotlyjs="cdn")

    print(f"  ✅ Gráficos guardados en: {out}/")
    print(f"     • embeddings_bert_2d.html")
    print(f"     • embeddings_xidoo_2d.html")
    print(f"     • comparacion_similitudes.html\n")

    return {
        "emb_bert": emb_bert,
        "emb_xidoo": emb_xidoo,
        "cos_bert_global": cos_bert,
        "cos_xidoo_global": cos_xidoo,
        "cos_por_fuente_bert": cos_por_fuente_bert,
        "cos_por_fuente_xidoo": cos_por_fuente_xidoo,
    }