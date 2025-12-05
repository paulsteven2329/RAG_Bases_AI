# app/ingest.py
import os
from typing import List
from .extractors import extract_text
from .embeddings import get_embeddings
from .vectorstore import add_vectors
from .config import settings
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
CHUNK_SIZE = 1000
OVERLAP = 200


def _chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - OVERLAP
    return chunks


def _summarize(text: str) -> str:
    if len(text) < 200:
        return text
    try:
        return summarizer(text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
    except:
        return text[:1000]


def ingest_file(path: str, source_id: str) -> None:
    try:
        raw_text = extract_text(path)
        if not raw_text.strip():
            print(f"[INGEST] Empty content: {source_id}")
            return

        if len(raw_text) > 2000:
            raw_text = _summarize(raw_text)

        chunks = _chunk_text(raw_text)
        embeddings = get_embeddings(chunks)

        abs_source = os.path.join(settings.UPLOAD_DIR, source_id)

        docs = [
            {
                "text": c,
                "source": abs_source,
                "chunk_index": idx,
                "source_id": source_id,
                "is_image": source_id.lower().endswith(('.png', '.jpg', '.jpeg'))
            }
            for idx, c in enumerate(chunks)
        ]

        add_vectors(embeddings, docs)
        print(f"[INGEST] {source_id}: {len(chunks)} chunks")

    except Exception as e:
        print(f"[INGEST ERROR] {source_id}: {e}")
        raise
    
# ==============================================================
# Ingestion Steps

# Extract → unified via extractors.py
# Summarize → if >2000 chars
# Chunk → 1000 words, 200 overlap
# Embed → all-MiniLM-L6-v2
# Store → FAISS + metadata