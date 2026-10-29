"""Optional local neural adapters. No model is downloaded during a request."""
import json
import threading
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import MODELS

MODEL_LOCK = threading.RLock()

def selection_manifest():
    for name in ("selected.json","default-selected.json"):
        candidate=MODELS/name
        if candidate.exists():
            selected=json.loads(candidate.read_text(encoding="utf-8"))
            if (MODELS/selected["path"]).exists():
                return candidate
    return None

@lru_cache(maxsize=1)
def encoder():
    if (MODELS / "embeddings" / "modules.json").exists():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(str(MODELS / "embeddings"), device="cpu")
    return None

@lru_cache(maxsize=1)
def ner():
    if (MODELS / "ner").exists():
        import spacy
        return spacy.load(MODELS / "ner")
    return None

@lru_cache(maxsize=1)
def nli():
    if (MODELS / "nli" / "config.json").exists():
        from transformers import pipeline
        return pipeline("text-classification", model=str(MODELS / "nli"), tokenizer=str(MODELS / "nli"), device=-1)
    return None

@lru_cache(maxsize=1)
def bias():
    manifest = selection_manifest()
    if manifest is None:
        return None, "Lexicon fallback"
    selected = json.loads(manifest.read_text(encoding="utf-8"))
    if selected["kind"] == "sklearn":
        import joblib
        return joblib.load(MODELS / selected["path"]), selected["name"]
    from transformers import pipeline
    return pipeline("text-classification", model=str(MODELS / selected["path"]), tokenizer=str(MODELS / selected["path"]), device=-1), selected["name"]

def similarities(texts):
    model = encoder()
    if model is not None:
        with MODEL_LOCK:
            vectors = model.encode(texts, normalize_embeddings=True).astype("float32")
        return vectors @ vectors.T, "Sentence embeddings"
    vectors = TfidfVectorizer(stop_words="english", ngram_range=(1,2)).fit_transform(texts)
    return (vectors @ vectors.T).toarray(), "TF-IDF fallback"

def status():
    selected = selection_manifest()
    report = MODELS / "metrics.json"
    return {
        "bias":json.loads(selected.read_text())["name"] if selected else "Lexicon fallback — no trained classifier",
        "embeddings":"all-MiniLM-L6-v2 / FAISS" if (MODELS/"embeddings"/"modules.json").exists() else "TF-IDF retrieval fallback",
        "ner":"spaCy en_core_web_sm" if (MODELS/"ner").exists() else "Capitalized-name heuristic",
        "nli":"Pretrained NLI" if (MODELS/"nli"/"config.json").exists() else "Unavailable — aligned passages only",
        "metrics":json.loads(report.read_text()) if report.exists() else None,
    }

