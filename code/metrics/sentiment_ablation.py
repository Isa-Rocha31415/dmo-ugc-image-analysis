"""
metrics.sentiment_ablation
──────────────────────────────────────────────────────────────
Ablación de modelos para responder "¿qué tan bueno es el embedding?":

  (a) Léxico            → baseline tonto (sin embeddings)
  (b) TF-IDF + LogReg   → baseline clásico de bolsa de palabras
  (c) Embeddings + LogReg (linear probe) → calidad real del embedding
  (d) Transformer afinado → techo del modelo (opcional, costoso)

Todas las funciones son puras salvo `run_ablation`, que entrena modelos.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .sentiment_metrics import evaluate

SEED = 42

# Léxico mínimo español/inglés para la línea base "tonta" (a).
LEXICON_POSITIVE = {
    "excelente", "excellent", "bueno", "buena", "buenos", "buenas", "great",
    "good", "love", "amo", "encanta", "perfecto", "perfecta", "perfect",
    "increible", "amazing", "fantastico", "fantástica", "wonderful",
    "recomendado", "recomienda", "recommend", "limpio", "limpia", "clean",
    "acogedor", "acogedora", "comfortable", "bonito", "beautiful", "delicioso",
    "delicious", "rápido", "rapido", "fast", "amable", "friendly", "servicio",
    "habitación", "habitacion", "hotel", "playa", "paisaje", "location",
}

LEXICON_NEGATIVE = {
    "malo", "mala", "malos", "malas", "bad", "terrible", "horrible", "worst",
    "peor", "pobre", "poor", "sucio", "sucia", "dirty", "caro", "caros",
    "expensive", "lento", "slow", "ruidoso", "noisy", "frio", "frío", "cold",
    "decepcion", "decepción", "disappointing", "-evitar", "nunca", "never",
    "problema", "problemas", "problem", "queja", "quejas", "complaint",
    "roto", "rota", "broken", "sucio", "fuga", "leak", "ignorante", "rude",
}

DEFAULT_FINETUNE_MODEL = "pysentimiento/robertuito-sentiment-analysis"


def _tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"[a-záéíóúüñ]+", str(text).lower())


def lexicon_predict(texts: Sequence[str]) -> List[str]:
    """
    (a) Predicción por léxico: gana la clase con más coincidencias.
    """
    preds: List[str] = []

    for text in texts:
        tokens = set(_tokenize(text))
        n_pos = len(tokens & LEXICON_POSITIVE)
        n_neg = len(tokens & LEXICON_NEGATIVE)
        preds.append("positive" if n_pos > n_neg else "negative")

    return preds


def _split(
    texts: Sequence[str],
    labels: Sequence[str],
    seed: int = SEED,
    test_size: float = 0.3,
):
    """Split estratificado y determinista."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(texts))
    y = np.asarray(labels)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    return train_idx, test_idx


def _probe_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = evaluate(y_true, y_pred)
    if extra:
        metrics.update(extra)
    return metrics


def tfidf_logreg_probe(
    texts: Sequence[str],
    labels: Sequence[str],
    seed: int = SEED,
    test_size: float = 0.3,
    max_features: int = 20000,
) -> Dict[str, Any]:
    """
    (b) TF-IDF + Regresión Logística.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    train_idx, test_idx = _split(texts, labels, seed=seed, test_size=test_size)

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    model = make_pipeline(
        TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
        ),
        LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
    )
    model.fit(train_texts, y_train)
    y_pred = model.predict(test_texts)

    return _probe_metrics(
        y_test,
        y_pred,
        {
            "modelo": "TF-IDF + LogReg",
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        },
    )


def embeddings_logreg_probe(
    embeddings: np.ndarray,
    labels: Sequence[str],
    seed: int = SEED,
    test_size: float = 0.3,
) -> Dict[str, Any]:
    """
    (c) Embeddings congelados + Regresión Logística (linear probe).

    Esta es la métrica que dice "qué tan bueno es el embedding".
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    emb = np.asarray(embeddings)
    train_idx, test_idx = _split(list(range(len(labels))), labels, seed=seed, test_size=test_size)

    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
    )
    model.fit(emb[train_idx], y_train)
    y_pred = model.predict(emb[test_idx])

    return _probe_metrics(
        y_test,
        y_pred,
        {
            "modelo": "Embeddings + LogReg (linear probe)",
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "dim_embeddings": int(emb.shape[1]),
        },
    )


def finetune_transformer(
    texts: Sequence[str],
    labels: Sequence[str],
    model_name: str = DEFAULT_FINETUNE_MODEL,
    seed: int = SEED,
    test_size: float = 0.3,
    epochs: int = 1,
    max_length: int = 256,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    (d) Transformer afinado (fine-tuning) sobre las etiquetas disponibles.

    Es la opción más costosa: se deja fuera por defecto.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    train_idx, test_idx = _split(texts, labels, seed=seed, test_size=test_size)

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    label_values = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(label_values)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_values)
    )

    class _Dataset(Dataset):
        def __init__(self, texts_, labels_):
            self.texts = texts_
            self.labels = labels_

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.texts[idx], label2id[self.labels[idx]]

    def _collate(batch):
        raw_texts = [b[0] for b in batch]
        raw_labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(
            raw_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc, raw_labels

    torch.manual_seed(seed)

    loader = DataLoader(
        _Dataset(train_texts, y_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for _ in range(epochs):
        for enc, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(**enc, labels=batch_labels)
            outputs.loss.backward()
            optimizer.step()

    model.eval()
    preds: List[str] = []

    with torch.no_grad():
        for start in range(0, len(test_texts), batch_size):
            chunk = test_texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            preds.extend(label_values[i] for i in logits.argmax(dim=-1).tolist())

    return _probe_metrics(
        y_test,
        preds,
        {
            "modelo": f"Transformer afinado ({model_name})",
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        },
    )


def run_ablation(
    texts: Sequence[str],
    labels: Sequence[str],
    embeddings: Optional[np.ndarray] = None,
    include_finetune: bool = False,
    finetune_model: str = DEFAULT_FINETUNE_MODEL,
    seed: int = SEED,
    test_size: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Ejecuta la ablación completa y devuelve una lista de resultados.
    """
    results: List[Dict[str, Any]] = []

    y_true = [str(label) for label in labels]

    lexicon_pred = lexicon_predict(texts)
    results.append(
        _probe_metrics(
            y_true,
            lexicon_pred,
            {
                "modelo": "Léxico (baseline)",
                "n_train": 0,
                "n_test": len(y_true),
            },
        )
    )

    try:
        results.append(tfidf_logreg_probe(texts, y_true, seed=seed, test_size=test_size))
    except Exception as exc:  # pragma: no cover - defensivo
        results.append({"modelo": "TF-IDF + LogReg", "error": str(exc)})

    if embeddings is not None:
        try:
            results.append(
                embeddings_logreg_probe(embeddings, y_true, seed=seed, test_size=test_size)
            )
        except Exception as exc:  # pragma: no cover - defensivo
            results.append({"modelo": "Embeddings + LogReg", "error": str(exc)})

    if include_finetune:
        try:
            results.append(
                finetune_transformer(
                    texts,
                    y_true,
                    model_name=finetune_model,
                    seed=seed,
                    test_size=test_size,
                )
            )
        except Exception as exc:  # pragma: no cover - defensivo
            results.append({"modelo": "Transformer afinado", "error": str(exc)})

    return results
