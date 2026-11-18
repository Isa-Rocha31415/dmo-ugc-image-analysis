import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from visualization import extraer_ngramas, graficar_top_ngramas, graficar_comparacion_metodos_outliers


# ============================================================
# DETECCIÓN DE OUTLIERS
# ============================================================
def aplicar_isolation_forest(embeddings: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """Aplica Isolation Forest para detectar outliers globales."""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    return iso_forest.fit_predict(embeddings)


def aplicar_lof(embeddings: np.ndarray, n_neighbors: int = 20, contamination: float = 0.05) -> np.ndarray:
    """Aplica Local Outlier Factor para detectar outliers locales."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    return lof.fit_predict(embeddings)


# ============================================================
# ANÁLISIS DE N-GRAMAS EN OUTLIERS
# ============================================================
def analizar_ngramas_outliers(
    df_outliers: pd.DataFrame,
    texto_col: str,
    results_dir: str = "code/results",
    palette_name: str = "ejecutiva",
    vision_type: str = "normal"
) -> Dict[str, Counter]:
    """Extrae y visualiza n-gramas (1, 2, 3) de comentarios atípicos."""
    textos_outliers = df_outliers[texto_col].tolist()
    
    unigramas = extraer_ngramas(textos_outliers, 1)
    bigramas = extraer_ngramas(textos_outliers, 2)
    trigramas = extraer_ngramas(textos_outliers, 3)
    
    graficar_top_ngramas(
        unigramas, 1,
        "Unigramas más frecuentes en comentarios atípicos (Outliers)",
        f"{results_dir}/06_unigramas_outliers.html",
        palette_name=palette_name,
        vision_type=vision_type
    )
    
    graficar_top_ngramas(
        bigramas, 2,
        "Bigramas más frecuentes en comentarios atípicos (Outliers)",
        f"{results_dir}/06_bigramas_outliers.html",
        palette_name=palette_name,
        vision_type=vision_type
    )
    
    graficar_top_ngramas(
        trigramas, 3,
        "Trigramas más frecuentes en comentarios atípicos (Outliers)",
        f"{results_dir}/06_trigramas_outliers.html",
        palette_name=palette_name,
        vision_type=vision_type
    )
    
    return {'unigramas': unigramas, 'bigramas': bigramas, 'trigramas': trigramas}


# ============================================================
# PIPELINE COMPLETO DE DETECCIÓN DE OUTLIERS
# ============================================================
def detectar_y_analizar_outliers(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    texto_col: str,
    fuente_col: str,
    dmo_value: str,
    metodo: str = 'LOF',
    contamination: float = 0.05,
    results_dir: str = "code/results",
    palette_name: str = "ejecutiva",
    vision_type: str = "normal"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compara Isolation Forest y LOF, analiza n-gramas de outliers y retorna DataFrame limpio.
    """
    df = df.copy()
    
    df['outlier_IF'] = aplicar_isolation_forest(embeddings, contamination)
    df['outlier_LOF'] = aplicar_lof(embeddings, contamination=contamination)
    
    graficar_comparacion_metodos_outliers(
        df,
        "Comparación de detección de outliers: Isolation Forest vs LOF",
        f"{results_dir}/06_comparacion_outliers_IF_vs_LOF.html",
        palette_name=palette_name,
        vision_type=vision_type
    )
    
    col_outlier = 'outlier_LOF' if metodo == 'LOF' else 'outlier_IF'
    
    df_outliers = df[df[col_outlier] == -1]
    df_normales = df[df[col_outlier] == 1].copy()
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE OUTLIERS (Método: {metodo})")
    print(f"{'='*60}")
    print(f"Total de comentarios: {len(df)}")
    print(f"Comentarios atípicos (outliers): {len(df_outliers)} ({len(df_outliers)/len(df)*100:.2f}%)")
    print(f"Comentarios normales: {len(df_normales)} ({len(df_normales)/len(df)*100:.2f}%)")
    
    print(f"\nDistribución de outliers por fuente:")
    print(df_outliers[fuente_col].value_counts())
    
    print(f"\nDistribución de datos normales por fuente:")
    print(df_normales[fuente_col].value_counts())
    
    ngramas_analysis = analizar_ngramas_outliers(
        df_outliers, texto_col, results_dir,
        palette_name=palette_name,
        vision_type=vision_type
    )
    
    df_limpio = df_normales.drop(columns=['outlier_IF', 'outlier_LOF'])
    
    print(f"\n✅ DataFrame limpio generado: {len(df_limpio)} comentarios listos para el siguiente paso")
    print(f"✅ Gráficos guardados en: {results_dir}/")
    
    return df_limpio, ngramas_analysis