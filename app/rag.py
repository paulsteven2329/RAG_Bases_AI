# app/rag.py
"""
RAG pipeline with:
- Dual LLM support: Hugging Face Router OR Ollama
- Image-aware retrieval
- Anti-hallucination (T=0.0, strict prompt, <think> removal)
- Full error handling (402, 503, 401, 403, 500)
"""
import httpx
import requests
import re
from typing import Dict, List
from .config import settings
from .vectorstore import search
from .embeddings import get_embeddings

# Hugging Face Router
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HF_HEADERS = {
    "Authorization": f"Bearer {settings.HF_TOKEN}",
    "Content-Type": "application/json"
}

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"

print(f"[RAG] Using {settings.LLM_PROVIDER.upper()} → "
      f"{getattr(settings, 'HF_MODEL', settings.OLLAMA_MODEL)}")


def answer_question(question: str, top_k: int = 6) -> Dict:
    """
    Retrieve relevant chunks and generate a clean, factual answer.
    """
    # 1. Embed query
    q_vec = get_embeddings([question])[0]
    results = search(q_vec, k=top_k * 3)

    if not results:
        return {"answer": "No documents indexed yet.", "sources": []}

    # 2. Detect image query
    img_kw = ["image", "photo", "picture", "screenshot", "what is in"]
    is_image_query = any(kw in question.lower() for kw in img_kw)

    # 3. Identify latest source
    src_ids = [r.get("source_id") for r in results if r.get("source_id")]
    latest_src = max(src_ids, key=lambda x: x) if src_ids else None

    # 4. Score each result
    scored: List[tuple] = []
    for r in results:
        score = 1.0
        if is_image_query and r.get("is_image", False):
            score = 3.0
        elif r.get("source_id") == latest_src:
            score = 1.5
        scored.append((r, score))

    # 5. Select top-k
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [r for r, _ in scored[:top_k]]

    # 6. Build context and sources
    context = "\n\n".join(r.get("text", "").strip() for r in selected)
    sources = [
        {"source": r["source"], "chunk": r.get("chunk_index", 0)}
        for r in selected[:3]
    ]

    # 7. Anti-hallucination prompt
    system_prompt = (
        "Answer ONLY using the context. NEVER use <think>. NEVER speculate. "
        "If unsure, say 'I don't know'."
    )

    # 8. Call LLM (HF or Ollama)
    answer = "I don't know."
    try:
        if settings.LLM_PROVIDER == "ollama":
            # Ollama: Simple generate endpoint
            payload = {
                "model": settings.OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 250
            }
            r = requests.post(OLLAMA_URL, json=payload, timeout=60.0)
            r.raise_for_status()
            raw = r.json()["response"].strip()

        else:
            # Hugging Face Router
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            payload = {
                "model": settings.HF_MODEL,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 250
            }
            with httpx.Client(timeout=60.0) as client:
                r = client.post(HF_CHAT_URL, json=payload, headers=HF_HEADERS)
                print(f"[HF] Status: {r.status_code}")

                if r.status_code == 503:
                    print("[HF] Warming up – retrying...")
                    r = client.post(HF_CHAT_URL, json=payload, headers=HF_HEADERS)

                r.raise_for_status()
                data = r.json()
                raw = data["choices"][0]["message"]["content"].strip()

        # 9. Clean output
        answer = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        answer = re.sub(r'\n\s*\n', '\n', answer)
        if len(answer) < 3:
            answer = "I don't know."

    except requests.exceptions.RequestException as e:
        print(f"[OLLAMA ERROR] {e}")
        answer = "Ollama not running. Start with: ollama serve"
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        msg = e.response.text[:200]
        print(f"[HF ERROR] {status}: {msg}")
        if status == 402:
            answer = "HF free quota exceeded. Upgrade to PRO."
        elif status == 401:
            answer = "Invalid HF token."
        elif status == 403:
            answer = "HF token lacks Inference Provider role."
        else:
            answer = f"LLM Error {status}"
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}")
        answer = "Error: LLM failed"

    return {"answer": answer, "sources": sources}