import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. CARGA DE DATAFRAMES
# ============================================================
ruta_base = Path("../data/raw")  # ajusta a tu estructura

df_insta   = pd.read_csv(ruta_base / "instaUGC.csv")
df_reddit  = pd.read_csv(ruta_base / "redditcdmx.csv")
df_trip    = pd.read_csv(ruta_base / "TripAdvisor.csv")
df_sectur  = pd.read_csv(ruta_base / "sectur.csv")

# ============================================================
# 2. NORMALIZACIÓN: cada DF → esquema común
#    Columnas unificadas: text, source, date, rating, category, num_comments
# ============================================================

# --- Instagram (UGC) ---
df_insta_norm = pd.DataFrame({
    "text":         df_insta["description"].astype(str),
    "source":       "instagram",
    "date":         np.nan,
    "rating":       np.nan,
    "category":     np.nan,
    "num_comments": np.nan
})

# --- Reddit (UGC): combinamos title + text para mayor contexto ---
df_reddit_norm = pd.DataFrame({
    "text": (df_reddit["title"].fillna("") + " . " + df_reddit["text"].fillna("")).str.strip(" ."),
    "source":       "reddit",
    "date":         np.nan,
    "rating":       np.nan,
    "category":     np.nan,
    "num_comments": df_reddit["num_comments"]
})

# --- TripAdvisor (UGC) ---
df_trip_norm = pd.DataFrame({
    "text":         df_trip["texto"].astype(str),
    "source":       "tripadvisor",
    "date":         df_trip["date"],
    "rating":       pd.to_numeric(df_trip["target"], errors="coerce"),
    "category":     np.nan,
    "num_comments": np.nan
})

# --- Sectur (DMO) ---
df_sectur_norm = pd.DataFrame({
    "text":         df_sectur["texto"].astype(str),
    "source":       "sectur",          # etiqueta DMO
    "date":         np.nan,
    "rating":       np.nan,
    "category":     df_sectur["clase"].astype(str),
    "num_comments": np.nan
})

# ============================================================
# 3. CONCATENACIÓN POR TIPO DE FUENTE
# ============================================================

# UGC = Instagram + Reddit + TripAdvisor
df_ugc = pd.concat(
    [df_insta_norm, df_reddit_norm, df_trip_norm],
    ignore_index=True
)

# DMO = Sectur (ya está normalizado con el mismo esquema)
df_dmo = df_sectur_norm.copy()

# ============================================================
# 4. LIMPIEZA BÁSICA (aplicada a ambos)
# ============================================================
# Eliminar filas sin texto
df_ugc = df_ugc.dropna(subset=["text"]).reset_index(drop=True)
df_dmo = df_dmo.dropna(subset=["text"]).reset_index(drop=True)

# ============================================================
# 5. GUARDADO
# ============================================================
ruta_salida = Path("../data/processed")
ruta_salida.mkdir(parents=True, exist_ok=True)

ruta_ugc = ruta_salida / "df_ugc.csv"
ruta_dmo = ruta_salida / "df_dmo.csv"

df_ugc.to_csv(ruta_ugc, index=False, encoding="utf-8")
df_dmo.to_csv(ruta_dmo, index=False, encoding="utf-8")

print(f"✅ df_ugc.csv guardado en: {ruta_ugc}")
print(f"   📊 Total de registros UGC: {len(df_ugc):,}")
print(f"   📂 Distribución por fuente UGC:\n{df_ugc['source'].value_counts()}\n")

print(f"✅ df_dmo.csv guardado en: {ruta_dmo}")
print(f"   📊 Total de registros DMO: {len(df_dmo):,}")
print(f"   📂 Distribución por categoría DMO:\n{df_dmo['category'].value_counts()}")