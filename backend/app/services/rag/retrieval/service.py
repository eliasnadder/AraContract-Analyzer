"""
Retrieval service — كل الـ logic هنا:
Similarity Search → Score Threshold → Deduplication → Cross-Encoder Re-ranking
"""

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from app.core.config import settings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


CROSS_ENCODER_PATH = Path(settings.CROSS_ENCODER_MODEL_PATH)
SCORE_THRESHOLD = settings.RETRIEVAL_SCORE_THRESHOLD
INITIAL_TOP_K = settings.RETRIEVAL_INITIAL_TOP_K


class RetrieverService:
    def __init__(self, qdrant_client: QdrantClient):
        self._client = qdrant_client
        self._cross_encoder = None

    def _load_cross_encoder(self):
        """Lazy Loading للـ Cross-Encoder"""
        if self._cross_encoder is None:
            if not CROSS_ENCODER_PATH.exists():
                raise FileNotFoundError(
                    f"Cross-Encoder غير موجود في: {CROSS_ENCODER_PATH}\n"
                    f"ضعه في: backend/models_local/cross-encoder-ms-marco-MiniLM-L-6-v2/"
                )
            logger.info(f"تحميل Cross-Encoder من: {CROSS_ENCODER_PATH}")
            self._cross_encoder = CrossEncoder(str(CROSS_ENCODER_PATH))
            logger.info("Cross-Encoder جاهز")

    def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float]
    ) -> List[Dict[str, Any]]:
        """البحث الأولي في Qdrant"""
        results = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=INITIAL_TOP_K,
            with_payload=True
        )

        return [
            {
                "text": hit.payload["text"],
                "parent_text": hit.payload["parent_text"],
                "clause_index": hit.payload["clause_index"],
                "chunk_index": hit.payload["chunk_index"],
                "is_chunked": hit.payload["is_chunked"],
                "score": hit.score
            }
            for hit in results.points
        ]

    def apply_score_threshold(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """يحذف النتائج تحت SCORE_THRESHOLD"""
        filtered = [r for r in results if r["score"] >= SCORE_THRESHOLD]
        logger.info(
            f"Score Threshold (>={SCORE_THRESHOLD}): "
            f"{len(results)} → {len(filtered)} نتيجة"
        )
        return filtered

    def deduplicate(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        إذا رجع أكثر من chunk من نفس البند:
        يحتفظ بالـ chunk صاحب أعلى score فقط
        """
        seen = {}
        for result in results:
            clause_idx = result["clause_index"]
            if clause_idx not in seen:
                seen[clause_idx] = result
            else:
                if result["score"] > seen[clause_idx]["score"]:
                    seen[clause_idx] = result

        deduplicated = list(seen.values())
        logger.info(
            f"Deduplication: {len(results)} → {len(deduplicated)} بند فريد"
        )
        return deduplicated

    def rerank(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Cross-Encoder Re-ranking:
        يأخذ السؤال + parent_text معاً ويعيد حساب الـ score بدقة أعلى
        """
        self._load_cross_encoder()

        if not results:
            return results

        # Cross-Encoder يقيّم كل (query, parent_text) معاً
        pairs = [[query, r["parent_text"]] for r in results]
        scores = self._cross_encoder.predict(pairs)

        for result, score in zip(results, scores):
            result["score"] = round(float(score), 4)

        reranked = sorted(results, key=lambda x: x["score"], reverse=True)
        logger.info(f"Cross-Encoder Re-ranking: تم تقييم {len(reranked)} بند")
        return reranked