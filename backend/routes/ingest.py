from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from config import CHUNK_SIZE, CHUNK_OVERLAP
from models.schemas import IngestResponse
from services import parser, chunker, embedder, vector_store

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    file_bytes = await file.read()

    try:
        text = parser.parse_file(file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded file.")

    file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"

    chunks = chunker.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=422, detail="File produced no valid chunks after parsing.")

    texts = [c["text"] for c in chunks]
    embeddings = embedder.generate_embeddings_batch(texts)

    stored = vector_store.store_chunks(chunks, embeddings, filename, file_ext)

    return IngestResponse(
        filename=filename,
        file_type=file_ext,
        total_chunks=len(chunks),
        total_vectors_stored=stored,
    )


@router.get("/documents")
def list_documents():
    return vector_store.list_documents()


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    # Count before deleting
    docs = vector_store.list_documents()
    found = next((d for d in docs if d["filename"] == filename), None)
    chunk_count = found["chunk_count"] if found else 0

    vector_store.delete_document(filename)

    return {"filename": filename, "deleted_vectors": chunk_count}


@router.get("/collection/info")
def collection_info():
    return vector_store.get_collection_info()
