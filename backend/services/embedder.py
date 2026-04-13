import time
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION

_client = genai.Client(api_key=GEMINI_API_KEY)


def generate_embedding(text: str) -> list[float]:
    response = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    return response.embeddings[0].values


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        embeddings.append(generate_embedding(text))
        time.sleep(0.5)
    return embeddings
