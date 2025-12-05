# app/embeddings.py
from sentence_transformers import SentenceTransformer
import os

FINE_TUNED = "models/fine_tuned_embedding"
BASE = "all-MiniLM-L6-v2"

if os.path.isdir(FINE_TUNED):
    print(f"[EMBEDDING] Loading fine-tuned model from {FINE_TUNED}")
    _model = SentenceTransformer(FINE_TUNED)
else:
    print(f"[EMBEDDING] Using base model {BASE}")
    _model = SentenceTransformer(BASE)


def get_embeddings(texts: list) -> list:
    return _model.encode(texts, convert_to_numpy=True).tolist()


# =============================================================================
# Key Features

# Safe fallback: Uses fine-tuned model if exists, else base
# Lazy loading: Model loaded once at startup
# 384-dimensional vectors (compatible with FAISS)