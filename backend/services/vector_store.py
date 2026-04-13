import time
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import COLLECTION_NAME, EMBEDDING_DIMENSION, QDRANT_HOST, QDRANT_PORT

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def init_collection() -> None:
    client = None
    last_error: Exception | None = None

    for attempt in range(1, 6):
        try:
            client = _get_client()
            existing = [c.name for c in client.get_collections().collections]
            if COLLECTION_NAME in existing:
                try:
                    info = client.get_collection(COLLECTION_NAME)
                    # Navigate safely — structure varies across qdrant-client versions
                    vectors_config = info.config.params.vectors
                    if hasattr(vectors_config, "size"):
                        actual_size = vectors_config.size
                    else:
                        # Mapping of named vectors: grab the first entry
                        actual_size = next(iter(vectors_config.values())).size
                    if actual_size != EMBEDDING_DIMENSION:
                        print(
                            f"[vector_store] Dimension mismatch (stored={actual_size}, "
                            f"expected={EMBEDDING_DIMENSION}). Recreating collection."
                        )
                        client.delete_collection(COLLECTION_NAME)
                        existing = []
                except Exception as inspect_exc:
                    print(f"[vector_store] Could not inspect collection, recreating: {inspect_exc}")
                    client.delete_collection(COLLECTION_NAME)
                    existing = []

            if COLLECTION_NAME not in existing:
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"[vector_store] Collection '{COLLECTION_NAME}' created.")
            else:
                print(f"[vector_store] Collection '{COLLECTION_NAME}' already exists.")
            return
        except Exception as exc:
            last_error = exc
            print(f"[vector_store] Qdrant not ready (attempt {attempt}/5): {exc}")
            time.sleep(2)

    raise RuntimeError(f"Could not connect to Qdrant after 5 attempts: {last_error}")


def store_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    filename: str,
    file_type: str,
) -> int:
    client = _get_client()
    now = datetime.now(timezone.utc).isoformat()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "chunk_index": chunk["chunk_index"],
                "filename": filename,
                "file_type": file_type,
                "upload_timestamp": now,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def semantic_search(
    query_embedding: list[float],
    top_k: int = 5,
    filename_filter: str | None = None,
) -> list[dict]:
    client = _get_client()

    query_filter = None
    if filename_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename_filter),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload["text"],
            "filename": hit.payload["filename"],
            "chunk_index": hit.payload["chunk_index"],
            "score": hit.score,
        }
        for hit in results.points
    ]


def get_collection_info() -> dict:
    client = _get_client()
    info = client.get_collection(COLLECTION_NAME)
    total = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
    return {
        "total_vectors": total,
        "status": str(info.status),
    }


def delete_document(filename: str) -> int:
    client = _get_client()
    result = client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename),
                )
            ]
        ),
    )
    # Count deleted by checking difference; simpler: scroll to count before delete.
    # We return the operation result info instead.
    return result.operation_id if result else 0


def list_documents() -> list[dict]:
    client = _get_client()
    offset = None
    all_payloads: list[dict] = []

    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_payloads.extend(r.payload for r in records)
        if next_offset is None:
            break
        offset = next_offset

    counts: dict[str, int] = {}
    for payload in all_payloads:
        fname = payload.get("filename", "unknown")
        counts[fname] = counts.get(fname, 0) + 1

    return [{"filename": fname, "chunk_count": count} for fname, count in counts.items()]
