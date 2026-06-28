"""
Retriever — يجمع كل الـ steps في دالة واحدة نظيفة
"""

from qdrant_client import QdrantClient
from typing import List, Dict, Any
import logging
from app.services.rag.retrieval.service import RetrieverService

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, qdrant_client: QdrantClient):
        self._service = RetrieverService(qdrant_client)

    def retrieve(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Pipeline كامل للاسترجاع:
        1. Similarity Search
        2. Score Threshold
        3. Deduplication
        4. Cross-Encoder Re-ranking
        5. إرجاع أفضل top_k

        Args:
            collection_name: اسم الـ collection في Qdrant
            query_vector: ناتج embedder.embed_query()
            query_text: نص السؤال الأصلي — للـ Cross-Encoder
            top_k: عدد النتائج النهائية

        Returns:
            List of dicts:
            {
                "text": نص الـ chunk,
                "parent_text": النص الكامل للبند,
                "clause_index": رقم البند,
                "score": درجة التشابه النهائية
            }
        """
        # Step 1: Similarity Search
        results = self._service.similarity_search(collection_name, query_vector)
        logger.info(f"Step 1 — Similarity Search: {len(results)} نتيجة")
        if not results:
            return []

        # Step 2: Score Threshold
        results = self._service.apply_score_threshold(results)
        if not results:
            logger.warning("جميع النتائج تحت الـ threshold")
            return []

        # Step 3: Deduplication
        results = self._service.deduplicate(results)

        # Step 4: Cross-Encoder Re-ranking
        results = self._service.rerank(results, query_text)

        # Step 5: أفضل top_k
        final = results[:top_k]
        logger.info(f"النتائج النهائية: {len(final)} بند")
        return final


# Factory
def get_retriever(qdrant_client: QdrantClient) -> Retriever:
    return Retriever(qdrant_client)