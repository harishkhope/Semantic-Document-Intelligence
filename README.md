# Semantic Document Intelligence

A production-ready RAG (Retrieval Augmented Generation) system that lets you upload PDF and TXT documents, store them as vector embeddings in Qdrant, and ask natural-language questions answered by OpenAI models — all running locally in Docker.

---

## Demo Video

Project demo is included in-repo at `assets/document-intelligence.mp4`.

<video controls width="100%" preload="metadata">
  <source src="assets/document-intelligence.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

Fallback link: [Watch demo video](assets/document-intelligence.mp4)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.11 · FastAPI |
| Vector Database | Qdrant |
| Embeddings | OpenAI (`text-embedding-3-small`) |
| LLM | OpenAI (`gpt-4o-mini`) |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | Streamlit |
| Containerisation | Docker · docker-compose |

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd semantic-document-intelligence
```

### 2. Add your OpenAI API key

```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY=<your key>
```

Get your API key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).

### 3. Start everything

```bash
docker-compose up --build
```

Docker will:
- Pull and start a Qdrant container (with persistent volume)
- Build and start the FastAPI backend (waits for Qdrant to be healthy)
- Build and start the Streamlit frontend

### 4. Open the apps

| App | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs (Swagger) | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

---

## Architecture

```
User uploads file
       │
       ▼
  [Streamlit UI]
       │  POST /ingest
       ▼
  [FastAPI Backend]
       │
       ├─► parser.py      — extract raw text from PDF / TXT
       ├─► chunker.py     — sliding-window overlap chunking (500 chars, 50 overlap)
       ├─► embedder.py    — OpenAI embedding API → 1536-dim float vectors
       └─► vector_store.py ─► [Qdrant]  store PointStructs with payload

User asks a question
       │
       ▼
  [Streamlit UI]
       │  POST /query
       ▼
  [FastAPI Backend]
       │
       ├─► embedder.py    — embed the question
       ├─► vector_store.py ─► [Qdrant]  cosine similarity search → top-k chunks
       ├─► build prompt   — system instructions + retrieved context + question
       └─► OpenAI LLM     — generate grounded answer
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a PDF or TXT file, parse, chunk, embed, and store |
| `GET` | `/documents` | List all ingested documents with chunk counts |
| `DELETE` | `/documents/{filename}` | Remove all vectors for a given filename |
| `GET` | `/collection/info` | Qdrant collection stats (total vectors, status) |
| `POST` | `/query` | Ask a question; returns LLM answer + ranked source chunks |
| `GET` | `/` | Health check |

### POST /query — request body

```json
{
  "question": "What are the main findings?",
  "top_k": 5,
  "filename_filter": "report.pdf"
}
```

`filename_filter` is optional — omit it to search across all documents.

---

## Key Implementation Details

- **Qdrant collection** uses `Distance.COSINE` so all similarity scores are in `[0, 1]`.
- **Embedding dimension** defaults to `1536` (OpenAI `text-embedding-3-small`) and is configurable with `EMBEDDING_DIMENSION`.
- **Overlapping chunks** (500-char window, 50-char overlap) ensure context isn't lost at chunk boundaries.
- **Similarity scores** are surfaced in the Streamlit UI as labelled progress bars so you can visually verify that vector search is working.
- **Startup retry loop** in `vector_store.init_collection()` retries up to 5 times with 2 s sleep so the backend never crashes when Qdrant is still initialising.
- **Rate-limit guard** — a 0.5 s sleep between embedding API calls helps avoid rate-limit bursts during bulk ingestion.
