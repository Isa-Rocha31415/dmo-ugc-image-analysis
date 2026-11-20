"""
download_models.py
Descarga todos los modelos necesarios para el pipeline en caché local.
Ejecutar UNA VEZ en el nodo de login (con internet).
"""
import os

# Asegurar modo online
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

def main():
    print("=" * 60)
    print("  DESCARGA DE MODELOS PARA PIPELINE NLP")
    print("=" * 60)

    # 1) Modelo de sentimiento
    print("\n[1/3] Descargando modelo de sentimiento...")
    print("      cardiffnlp/twitter-xlm-roberta-base-sentiment")
    from transformers import pipeline
    _ = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    )
    print("  ✅ Modelo de sentimiento descargado.")

    # 2) Modelo de embeddings BERT (sentence-transformers)
    print("\n[2/3] Descargando modelo de embeddings BERT...")
    print("      paraphrase-multilingual-MiniLM-L12-v2")
    from sentence_transformers import SentenceTransformer
    _ = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("  ✅ Modelo BERT (sentence-transformers) descargado.")

    # 3) Tokenizer y config de BERT vía transformers (por si el fallback lo usa)
    print("\n[3/3] Descargando tokenizer y config de BERT (transformers)...")
    from transformers import AutoModel, AutoTokenizer
    _ = AutoModel.from_pretrained("paraphrase-multilingual-MiniLM-L12-v2")
    _ = AutoTokenizer.from_pretrained("paraphrase-multilingual-MiniLM-L12-v2")
    print("  ✅ Tokenizer y config descargados.")

    print("\n" + "=" * 60)
    print("  ✅ TODOS LOS MODELOS DESCARGADOS EN CACHÉ")
    print("=" * 60)
    print("\nAhora puedes ejecutar tu job Slurm en modo offline sin problemas.")
    print("Ubicación del caché:", os.path.expanduser("~/.cache/huggingface"))


if __name__ == "__main__":
    main()