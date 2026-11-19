import re
import emoji
import pandas as pd
import numpy as np
from langdetect import detect, LangDetectException
from tqdm import tqdm
import nltk
import spacy
from nltk.corpus import stopwords
import os

# Descargar recursos de NLTK de forma silenciosa
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# --------------------- CONFIGURACIÓN -----------------------
# Modelos de spaCy para lematización según el idioma
SPACY_MODELS = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "pt": "pt_core_news_sm",
}

# Modelos de embeddings (BERT) para representación semántica
EMBEDDING_MODELS = {
    "es": "dccuchile/bert-base-spanish-wwm-uncased",
    "en": "bert-base-uncased",
    "fr": "dbmdz/bert-base-french-europeana-cased",
    "pt": "neuralmind/bert-base-portuguese-cased",
}

# Mapeo de idiomas para la librería NLTK (stopwords)
LANG_MAP_SW = {"es": "spanish", "en": "english", "fr": "french", "pt": "portuguese"}

# --------------------- RECURSOS DE IDIOMA -----------------------
def get_language_resources(lang: str):
    """
    Carga las stopwords, el modelo de spaCy y el nombre del modelo de embeddings
    para el idioma especificado.
    """
    lang = lang.lower()
    if lang not in LANG_MAP_SW:
        raise ValueError(f"Idioma no soportado: {lang}. Usa 'es', 'en', 'fr' o 'pt'.")

    stop_words = set(stopwords.words(LANG_MAP_SW[lang]))

    model_name = SPACY_MODELS[lang]
    try:
        # Deshabilitamos NER y parser para acelerar el procesamiento
        nlp = spacy.load(model_name, disable=["ner", "parser"])
    except OSError:
        print(f"Descargando modelo de spaCy: {model_name}")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name, disable=["ner", "parser"])

    emb_model = EMBEDDING_MODELS[lang]
    return stop_words, nlp, emb_model

# --------------------- DETECCIÓN DE IDIOMA -----------------------
def detect_language(text: str) -> str:
    """
    Detecta el idioma de un texto. Limpia ruido básico antes de detectar.
    """
    if not isinstance(text, str):
        return "unknown"
    try:
        # Eliminar URLs, menciones y hashtags para no sesgar la detección
        clean = re.sub(r"http\S+|@\w+|#\w+", "", text)
        return detect(clean)
    except LangDetectException:
        return "unknown"

# --------------------- LIMPIEZA DE TEXTO (MENOS AGRESIVA) -----------------------
def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando SOLO el ruido de redes sociales (URLs, menciones).
    Conserva números, puntuación y caracteres especiales para no perder 
    información semántica relevante (ej. "5 estrellas", "¡excelente!", "$100").
    """
    if not isinstance(text, str):
        return ""

    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Eliminar menciones (@usuario)
    text = re.sub(r"@\w+", "", text)
    # Convertir hashtags a texto normal (eliminar el símbolo #)
    text = re.sub(r"#(\w+)", r"\1", text)
    # Convertir emojis a su representación en texto (ej. :smile:)
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # NO eliminamos números ni caracteres de puntuación/simbolos 
    # para conservar el contexto y sentimiento.
    
    # Convertir a minúsculas
    text = text.lower()
    # Normalizar espacios en blanco
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# --------------------- LEMATIZACIÓN -----------------------
def lemmatize_text(text: str, nlp, stop_words: set) -> str:
    """
    Lematiza el texto y elimina stopwords.
    Conserva números y palabras alfabéticas.
    """
    if not text:
        return ""

    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in stop_words
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 1
    ]
    return " ".join(tokens)

# --------------------- PIPELINE PRINCIPAL -----------------------
def preprocess_dataframe(df: pd.DataFrame, text_column: str, lang: str) -> tuple:
    print(f"\nIniciando preprocesamiento | idioma objetivo: {lang.upper()} | filas: {len(df):,}")
    df_proc = df.copy()

    # Paso 1: detectar idioma
    print("Detectando idioma por fila...")
    df_proc["lang"] = [detect_language(t) for t in tqdm(df_proc[text_column], desc="Detectando idioma")]
    print(df_proc["lang"].value_counts())

    # Paso 2: filtrar por idioma objetivo
    initial_count = len(df_proc)
    df_proc = df_proc[df_proc["lang"] == lang].copy()
    print(f"Filtrado por idioma '{lang}': {initial_count:,} -> {len(df_proc):,}")

    if df_proc.empty:
        raise ValueError(f"No se encontraron textos para el idioma '{lang}'.")

    stop_words, nlp, emb_model = get_language_resources(lang)

    df_proc["text_clean"] = [clean_text(t) for t in tqdm(df_proc[text_column], desc="Limpiando")]
    
    # ============================================================
    # 🔍 DIAGNÓSTICO DE REDUCCIÓN DE TEXTO (clean_text)
    # ============================================================
    original_lengths = df_proc[text_column].astype(str).str.len()
    cleaned_lengths = df_proc["text_clean"].str.len()
    
    total_orig = int(original_lengths.sum())
    total_clean = int(cleaned_lengths.sum())
    chars_eliminated = total_orig - total_clean
    pct_eliminated = (chars_eliminated / total_orig * 100) if total_orig > 0 else 0
    
    # Filas que se eliminarán por el filtro de longitud mínima
    rows_filtered_out = (df_proc["text_clean"].str.len() <= 10).sum()
    pct_rows_filtered = (rows_filtered_out / len(df_proc) * 100) if len(df_proc) > 0 else 0
    
    print("\n" + "="*60)
    print(f"🔍 DIAGNÓSTICO DE REDUCCIÓN DE TEXTO - {lang.upper()}")
    print("="*60)
    print(f"Registros analizados: {len(df_proc):,}")
    print(f"Caracteres totales originales: {total_orig:,}")
    print(f"Caracteres totales después de clean_text: {total_clean:,}")
    print(f"Caracteres eliminados en limpieza: {chars_eliminated:,} ({pct_eliminated:.2f}%)")
    print(f"Filas descartadas por longitud <= 10: {rows_filtered_out:,} ({pct_rows_filtered:.2f}%)")
    print("-" * 60)
    print(f"Longitud promedio original por registro: {original_lengths.mean():.1f} caracteres")
    print(f"Longitud promedio después de limpieza: {cleaned_lengths.mean():.1f} caracteres")
    print("="*60 + "\n")
    # ============================================================

    # Paso 5: lematizar
    print("Lematizando...")
    df_proc["processed_text"] = [
        lemmatize_text(t, nlp, stop_words)
        for t in tqdm(df_proc["text_clean"], desc="Lematizando")
    ]
    df_proc = df_proc[df_proc["processed_text"].str.len() > 0].copy()

    df_proc = df_proc.reset_index(drop=True)
    print(f"Listo. Filas finales: {len(df_proc):,} | modelo de embeddings: {emb_model}\n")
    
    return df_proc, emb_model


