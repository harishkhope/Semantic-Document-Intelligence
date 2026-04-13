from fastapi import APIRouter, HTTPException
from google import genai

from config import GEMINI_API_KEY, LLM_MODEL
from models.schemas import QueryRequest, QueryResponse, SourceChunk
from services import embedder, vector_store

_client = genai.Client(api_key=GEMINI_API_KEY)

router = APIRouter()


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = [
        f"[Source: {chunk['filename']}, chunk {chunk['chunk_index']}]\n{chunk['text']}"
        for chunk in chunks
    ]
    context = "\n\n---\n\n".join(context_parts)

    return (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the answer is not in the context, say "
        '"I could not find relevant information in the uploaded documents." '
        "Always be concise and accurate.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    query_embedding = embedder.generate_embedding(request.question)

    hits = vector_store.semantic_search(
        query_embedding=query_embedding,
        top_k=request.top_k,
        filename_filter=request.filename_filter,
    )

    if not hits:
        return QueryResponse(
            answer="I could not find relevant information in the uploaded documents.",
            sources=[],
            total_sources_found=0,
        )

    prompt = _build_prompt(request.question, hits)

    response = _client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    answer = response.text

    sources = [
        SourceChunk(
            text=hit["text"][:200],
            filename=hit["filename"],
            chunk_index=hit["chunk_index"],
            similarity_score=round(hit["score"], 4),
        )
        for hit in hits
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        total_sources_found=len(hits),
    )
