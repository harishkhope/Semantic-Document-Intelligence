import time
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION

_client = OpenAI(api_key=OPENAI_API_KEY)


def generate_embedding(text: str) -> list[float]:
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSION,
        encoding_format="float",
    )
    return response.data[0].embedding


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        embeddings.append(generate_embedding(text))
        time.sleep(0.5)
    return embeddings
