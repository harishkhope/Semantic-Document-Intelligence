import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME: str = "rag_documents"
EMBEDDING_MODEL: str = "gemini-embedding-001"
LLM_MODEL: str = "gemini-2.0-flash"

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

EMBEDDING_DIMENSION: int = 768
