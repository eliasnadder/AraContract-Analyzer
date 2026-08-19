"""Minimal Colab entry point for the public RAG routes.

It intentionally avoids importing the main application, whose protected routes
require Firebase configuration.  The RAG router is the same router used by the
normal API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.rag import router as rag_router


app = FastAPI(title="AraContract RAG (Colab)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rag_router, prefix="/api/contract")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}
