# app/main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .ingest import ingest_file
from .rag import answer_question
from .schemas import QueryRequest, QueryResponse
from .config import settings

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bg(path: str, name: str):
    try:
        ingest_file(path, source_id=name)
    except Exception as e:
        print(f"[INGEST FAILED] {name}: {e}")


@app.post("/ingest")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".txt", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(400, "Unsupported file")
    dest = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    background.add_task(_bg, dest, file.filename)
    return {"status": "ingestion started", "file": file.filename}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    return QueryResponse(**answer_question(req.question))


@app.get("/")
async def root():
    return {"message": "Tesla RAG ready"}