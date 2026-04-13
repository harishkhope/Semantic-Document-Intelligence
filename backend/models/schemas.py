from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filename_filter: Optional[str] = None


class SourceChunk(BaseModel):
    text: str
    filename: str
    chunk_index: int
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    total_sources_found: int


class IngestResponse(BaseModel):
    filename: str
    file_type: str
    total_chunks: int
    total_vectors_stored: int
