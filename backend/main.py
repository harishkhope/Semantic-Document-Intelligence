from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import ingest, query
from services.vector_store import get_collection_info, init_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_collection()
    yield


app = FastAPI(title="RAG Vector Memory API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/")
def health_check():
    try:
        info = get_collection_info()
    except Exception as exc:
        info = {"error": str(exc)}
    return {"status": "ok", "collection_info": info}
