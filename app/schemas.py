# app/schemas.py
from pydantic import BaseModel
from typing import List


class Source(BaseModel):
    source: str
    chunk: int = 0


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]