"""
Qdrant store module for AraContract RAG system.
Handles collection creation and document ingestion only.
Retrieval is handled by retriever.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from typing import List, Dict, Any
import uuid
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


VECTOR_SIZE = settings.EMBEDDING_VECTOR_SIZE

class RAGStore:
    def __init__(self):
        self._client = QdrantClient(":memory:")
        logger.info("تم إنشاء Qdrant client في الذاكرة")

    @property
    def client(self) -> QdrantClient:
        """يكشف الـ client للـ retriever"""
        return self._client

    def create_collection(self, session_id: str) -> str:
        collection_name = f"contract_{session_id}"

        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name in existing:
            self._client.delete_collection(collection_name)
            logger.info(f"حذف collection قديم: {collection_name}")

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        logger.info(f"تم إنشاء collection: {collection_name}")
        return collection_name

    def store_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"عدد الـ chunks ({len(chunks)}) لا يساوي عدد الـ embeddings ({len(embeddings)})"
            )

        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "parent_text": chunk["parent_text"],
                        "clause_index": chunk["clause_index"],
                        "chunk_index": chunk["chunk_index"],
                        "is_chunked": chunk["is_chunked"],
                    }
                )
            )

        self._client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"تم تخزين {len(points)} chunk في {collection_name}")
        return len(points)

    def delete_collection(self, collection_name: str) -> bool:
        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name in existing:
            self._client.delete_collection(collection_name)
            logger.info(f"تم حذف collection: {collection_name}")
            return True
        return False

    def collection_exists(self, collection_name: str) -> bool:
        existing = [c.name for c in self._client.get_collections().collections]
        return collection_name in existing


_store_instance = None

def get_rag_store() -> RAGStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = RAGStore()
    return _store_instance