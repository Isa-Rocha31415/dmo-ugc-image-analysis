"""
outlier_ngram_analysis.py
─────────────────────────────────────────────────────────────
Detección de outliers (Isolation Forest / LOF / Z-Score)
+ análisis de unigramas, bigramas y trigramas de los outliers
+ detección GLOBAL o POR FUENTE
+ salidas a consola para pasar a una IA.
"""
from __future__ import annotations

import re
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# Helpers de n-gramas
# ──────────────────────────────────────────────
STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "con", "por", "para", "que", "es",
    "se", "no", "lo", "al", "a", "o", "y", "e", "pero",
    "más", "muy", "me", "mi", "su", "le", "les", "nos",
    "te", "ya", "también", "como", "fue", "ser", "está",
    "este", "esta", "eso", "esa", "hay", "han", "he",
    "son", "tiene", "todo", "todos", "una", "cuando",
    "sobre", "entre", "desde", "hasta", "sin", "pues",
    "porque", "donde", "cual", "quien", "sus", "mis",
    "tu", "yo", "él", "ella", "ellos", "ellas", "usted",
    "nosotros", "eso", "esto", "aquel", "aquella",
    "otro", "otra", "cada", "mismo", "mucha", "mucho",
    "bien", "aquí", "ahí", "allí", "así", "tan", "the",
    "and", "of", "to", "in", "is", "it", "was", "for",
    "on", "are", "that", "this", "with", "at", "an",
    "be", "or", "as", "from", "by", "had", "have",
    "he", "she", "they", "we", "you", "not", "but",
    "can", "all", "will", "there", "their", "what",
    "so", "if", "my", "your", "his", "her", "its",
}


def _limpiar(texto: str) -> List[str]:
    """Tokeniza y limpia un texto."""
    texto = re.sub(r"http\S+|www\S+", "", str(texto))
    texto = re.sub(r"[^a-záéíóúüñ]", " ", texto.lower())
    tokens = texto.split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS_ES]


def extraer_ngramas_lista(textos: List[str], n: int, top: int = 25) -> List[Tuple[str, int]]:
    """Devuelve los n-gramas más frecuentes como lista [(ngrama, conteo), ...]."""
    counter: Counter = Counter()
    for texto in textos:
        tokens = _limpiar(texto)
        for i in range(len(tokens) - n + 1):
            counter[" ".join(tokens[i : i + n])] += 1
    return counter.most_common(top)


# ──────────────────────────────────────────────
# Detección de outliers (núcleo)
# ──────────────────────────────────────────────
def detectar_outliers(
    embeddings: np.ndarray,
    method: str = "LOF",
    contamination: Union[float, str] = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    """
    Devuelve array booleano: True = outlier.
    method ∈ {"IF", "LOF", "zscore"}
    contamination: float (proporción fija) o "auto" (detección automática).
    """
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if method == "IF":
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        preds = model.fit_predict(X)

    elif method == "LOF":
        n_neigh = min(20, max(2, len(X) - 1))
        model = LocalOutlierFactor(
            n_neighbors=n_neigh,
            contamination=contamination,
        )
        preds = model.fit_predict(X)

    elif method == "zscore":
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
        scores = np.abs(Z).mean(axis=1)
        if contamination == "auto":
            # Con "auto" usa 2 desviaciones estándar como umbral
            threshold = scores.mean() + 2 * scores.std()
        else:
            threshold = np.percentile(scores, (1 - float(contamination)) * 100)
        preds = np.where(scores > threshold, -1, 1)

    else:
        raise ValueError(f"Método no soportado: {method}")

    return preds == -1  # True = outlier


# ──────────────────────────────────────────────
# Detección POR FUENTE
# ──────────────────────────────────────────────
def detectar_outliers_por_fuente(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    source_column: str,
    method: str = "IF",
    contamination: Union[float, str] = "auto",
    min_samples: int = 10,
) -> np.ndarray:
    """
    Detecta outliers DENTRO de cada fuente por separado.
    Esto evita que fuentes con distribuciones muy diferentes
    (ej: TripAdvisor vs Instagram) se marquen como outliers
    entre sí.

    Args:
        df: DataFrame con los comentarios.
        embeddings: Matriz de embeddings (n_docs, n_dims).
        source_column: Nombre de la columna de fuente.
        method: "IF", "LOF" o "zscore".
        contamination: float o "auto".
        min_samples: Mínimo de comentarios por fuente para
                     detectar outliers. Fuentes con menos se
                     consideran todas normales.

    Returns:
        Array booleano global: True = outlier.
    """
    mask_global = np.zeros(len(df), dtype=bool)
    fuentes = df[source_column].unique()

    for fuente in fuentes:
        mask_fuente = (df[source_column] == fuente).values
        emb_fuente = embeddings[mask_fuente]
        n_fuente = len(emb_fuente)

        if n_fuente < min_samples:
            # Muy pocos datos → no detectar outliers
            continue

        mask_out = detectar_outliers(
            emb_fuente,
            method=method,
            contamination=contamination,
        )
        mask_global[mask_fuente] = mask_out

    return mask_global


# ──────────────────────────────────────────────
# Análisis completo + consola
# ──────────────────────────────────────────────
def analizar_outliers(
    df: pd.DataFrame,
    text_column: str,
    source_column: str,
    embeddings: np.ndarray,
    output_dir: str = "results/outliers",
    method: str = "LOF",
    contamination: Union[float, str] = 0.05,
    top_n: int = 50,
    per_source: bool = True,
    min_samples_per_source: int = 10,
) -> Dict[str, Any]:
    """
    Ejecuta todo el análisis de outliers e imprime en consola.

    Args:
        per_source: Si True, detecta outliers DENTRO de cada
                    fuente por separado. Si False, detección global.
        min_samples_per_source: Mínimo de comentarios por fuente
                                para aplicar detección.

    Devuelve diccionario con resultados y df limpio.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Detectar ──
    if per_source:
        mask_outlier = detectar_outliers_por_fuente(
            df=df,
            embeddings=embeddings,
            source_column=source_column,
            method=method,
            contamination=contamination,
            min_samples=min_samples_per_source,
        )
        modo = f"POR FUENTE ({method})"
    else:
        mask_outlier = detectar_outliers(
            embeddings, method=method, contamination=contamination
        )
        modo = f"GLOBAL ({method})"

    n_total = len(df)
    n_outliers = int(mask_outlier.sum())
    n_normales = n_total - n_outliers

    df_outliers = df.loc[mask_outlier].copy()
    df_clean = df.loc[~mask_outlier].copy()

    # ── 2. N-gramas de outliers ──
    textos_out = df_outliers[text_column].fillna("").astype(str).tolist()
    unigramas = extraer_ngramas_lista(textos_out, 1, top_n)
    bigramas = extraer_ngramas_lista(textos_out, 2, top_n)
    trigramas = extraer_ngramas_lista(textos_out, 3, top_n)

    # ── 3. Distribución de outliers por fuente ──
    fuente_out = df_outliers[source_column].value_counts()

    # ── 4. Estadísticas por fuente ──
    stats_fuente = []
    for fuente in df[source_column].unique():
        mask_f = (df[source_column] == fuente).values
        n_f = int(mask_f.sum())
        n_out_f = int(mask_outlier[mask_f].sum())
        pct_f = n_out_f / n_f * 100 if n_f > 0 else 0
        stats_fuente.append({
            "fuente": fuente,
            "total": n_f,
            "outliers": n_out_f,
            "normales": n_f - n_out_f,
            "pct_outliers": round(pct_f, 2),
        })

    # ── 5. Imprimir en consola ──
    SEP = "═" * 70
    cont_str = f"{contamination:.2%}" if isinstance(contamination, float) else contamination

    print(f"\n{SEP}")
    print("   📊  ANÁLISIS DE OUTLIERS")
    print(f"{SEP}")
    print(f"  Modo de detección      : {modo}")
    print(f"  Método                 : {method}")
    print(f"  Contaminación          : {cont_str}")
    print(f"  Total comentarios      : {n_total:,}")
    print(f"  Outliers detectados    : {n_outliers:,} ({n_outliers/n_total:.2%})")
    print(f"  Comentarios normales   : {n_normales:,} ({n_normales/n_total:.2%})")

    # Tabla detallada por fuente
    print(f"\n  ── Estadísticas de outliers POR FUENTE ──")
    print(f"  {'Fuente':<22s} {'Total':>7s} {'Out':>7s} {'Norm':>7s} {'%Out':>7s}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for s in stats_fuente:
        print(
            f"  {s['fuente']:<22s} "
            f"{s['total']:>7,d} "
            f"{s['outliers']:>7,d} "
            f"{s['normales']:>7,d} "
            f"{s['pct_outliers']:>6.1f}%"
        )

    # ── 6. N-gramas ──
    print(f"\n  ── UNIGRAMAS más frecuentes en outliers (top {top_n}) ──")
    for i, (w, c) in enumerate(unigramas, 1):
        print(f"    {i:>3}. {w:<35s} {c:>5,d}")

    print(f"\n  ── BIGRAMAS más frecuentes en outliers (top {top_n}) ──")
    for i, (w, c) in enumerate(bigramas, 1):
        print(f"    {i:>3}. {w:<35s} {c:>5,d}")

    print(f"\n  ── TRIGRAMAS más frecuentes en outliers (top {top_n}) ──")
    for i, (w, c) in enumerate(trigramas, 1):
        print(f"    {i:>3}. {w:<35s} {c:>5,d}")

    # ── 7. Ejemplos de comentarios outlier ──
    n_ejemplos = min(10, len(df_outliers))
    print(f"\n  ── Ejemplos de comentarios OUTLIER ({n_ejemplos}) ──")
    for idx, (_, row) in enumerate(df_outliers.head(n_ejemplos).iterrows()):
        txt = str(row[text_column])[:180]
        src = row[source_column]
        print(f"    [{idx+1}] ({src}) {txt}")

    print(f"\n{SEP}\n")

    # ── 8. Guardar JSON resumen ──
    resumen = {
        "modo": modo,
        "method": method,
        "contamination": str(contamination),
        "per_source": per_source,
        "n_total": n_total,
        "n_outliers": n_outliers,
        "n_normales": n_normales,
        "stats_por_fuente": stats_fuente,
        "unigramas": unigramas[:20],
        "bigramas": bigramas[:20],
        "trigramas": trigramas[:20],
    }
    with open(out / "outlier_analysis.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    # ── 9. Guardar CSVs ──
    df_outliers.to_csv(out / "df_outliers.csv", index=False)
    df_clean.to_csv(out / "df_clean.csv", index=False)

    print(f"  ✅ Resultados guardados en: {out}/")
    print(f"     • outlier_analysis.json")
    print(f"     • df_outliers.csv ({len(df_outliers):,} filas)")
    print(f"     • df_clean.csv ({len(df_clean):,} filas)")

    return {
        "mask_outlier": mask_outlier,
        "df_outliers": df_outliers,
        "df_clean": df_clean,
        "unigramas": unigramas,
        "bigramas": bigramas,
        "trigramas": trigramas,
        "stats_por_fuente": stats_fuente,
        "resumen": resumen,
    }