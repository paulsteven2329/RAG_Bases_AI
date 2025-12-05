# app/vectorstore.py
import faiss
import json
import os
import numpy as np
from typing import List, Dict
from .config import settings

INDEX_PATH = settings.INDEX_PATH
META_PATH = settings.META_PATH
DIM = 384  # size of each embedding vector 

_vector_store = None
_meta: List[Dict] = []


def _load_or_create():
    global _vector_store, _meta
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        _vector_store = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta.extend([json.loads(line) for line in f])
    else:
        _vector_store = faiss.IndexFlatL2(DIM)
        _meta.clear()


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _load_or_create()
    return _vector_store


def add_vectors(vectors: List[List[float]], docs: List[Dict]):
    global _vector_store, _meta
    if _vector_store is None:
        _load_or_create()

    _vector_store.add(np.array(vectors, dtype="float32"))
    _meta.extend(docs)

    faiss.write_index(_vector_store, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for d in _meta:
            f.write(json.dumps(d) + "\n")


def search(query_vec, k: int = 6) -> List[Dict]:
    global _vector_store, _meta
    if _vector_store is None or _vector_store.ntotal == 0:
        return []
    D, I = _vector_store.search(np.array([query_vec], dtype="float32"), k)
    return [_meta[i] for i in I[0] if i < len(_meta)]