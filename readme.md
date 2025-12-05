[User]
   ↓
Frontend (localhost:3000)
   ↓ POST /ingest
FastAPI Backend (localhost:8000)
   ├── Save file → data/uploads/
   └── Background Task:
        ├── extract_text(file)
        │    ├── PDF → PyPDF2
        │    ├── TXT → read()
        │    ├── DOCX → python-docx
        │    └── IMAGE → OCR (TrOCR) → text
        ├── Summarize (if >2000 chars)
        ├── Chunk (1000 words, 200 overlap)
        ├── Embed (sentence-transformers)
        └── Store in FAISS + metadata
   ↓
User asks: "What is the course name?"
   ↓
Frontend → POST /query
   ├── Embed question
   ├── Search FAISS (top-k=18)
   ├── Prioritize: Image chunks if query about image
   ├── Build context
   ├── Call HF Router → LLM
   └── Clean answer → return JSON


=================================================================================================
   RAG Flow (Core Logic)

question → embed → FAISS search (k=18)
↓
Score chunks:
  - Image query + is_image → +3.0
  - Latest file → +1.5
↓
Select top 6 → context
↓
Prompt:
  system: "Answer ONLY using context. No <think>."
  user: "Context: ...\nQuestion: ..."
↓
HF Router → LLM → clean answer

====================================================================================================
Key Files & Responsibilities

File,Role
main.py,"API endpoints, CORS, background tasks"
ingest.py,Extract → Summarize → Chunk → Embed → Store
extractors.py,Unified PDF/TXT/DOCX/OCR
ocr.py,Image preprocessing + TrOCR
vectorstore.py,FAISS index + metadata
rag.py,Smart retrieval + LLM call
embeddings.py,Safe embedding model loading