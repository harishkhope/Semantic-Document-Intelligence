import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME: str = "rag_documents"
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
