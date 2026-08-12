"""
Embedder module for AraContract RAG system.
Uses LangChain RecursiveCharacterTextSplitter + HuggingFaceEmbeddings (local model).
Hybrid Strategy: short clauses → single embedding, long clauses → recursive chunking with parent_text.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from typing import List, Dict, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


MODEL_PATH = Path(settings.EMBEDDING_MODEL_PATH)
MAX_CHARS = settings.CHUNK_MAX_CHARS
OVERLAP_CHARS = settings.CHUNK_OVERLAP_CHARS


class Embedder:
    def __init__(self):
        self._embeddings = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHARS,
            chunk_overlap=OVERLAP_CHARS,
            separators=[".\n", "؟\n", "!\n", "،", "؛", " ", ""]
        )

    def _load_model(self):
        """تحميل النموذج عند الاستخدام الأول فقط — Lazy Loading"""
        if self._embeddings is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"النموذج غير موجود في: {MODEL_PATH}\n"
                    f"ضع النموذج في: backend/models_local/paraphrase-multilingual-MiniLM-L12-v2/"
                )
            logger.info(f"تحميل نموذج الـ Embedding من: {MODEL_PATH}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=str(MODEL_PATH),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("تم تحميل النموذج بنجاح")

    def prepare_chunks(self, clauses: List[str]) -> List[Dict[str, Any]]:
        all_chunks = []

        for clause_idx, clause in enumerate(clauses):
            clause = clause.strip()
            if not clause:
                continue

            if len(clause) <= MAX_CHARS:
                all_chunks.append({
                    "text": clause,
                    "parent_text": clause,
                    "clause_index": clause_idx,
                    "chunk_index": 0,
                    "is_chunked": False
                })
            else:
                sub_chunks = self._splitter.split_text(clause)
                # احذف الـ chunks المكررة أو الفارغة
                seen_chunks = set()
                clean_chunks = []
                for c in sub_chunks:
                    c = c.strip()
                    if c and c not in seen_chunks:
                        seen_chunks.add(c)
                        clean_chunks.append(c)

                logger.info(
                    f"البند {clause_idx} ({len(clause)} حرف) "
                    f"→ {len(clean_chunks)} chunks فريدة"
                )
                for chunk_idx, chunk_text in enumerate(clean_chunks):
                    all_chunks.append({
                        "text": chunk_text,
                        "parent_text": clause,
                        "clause_index": clause_idx,
                        "chunk_index": chunk_idx,
                        "is_chunked": True
                    })

        logger.info(
            f"إجمالي: {len(clauses)} بند → {len(all_chunks)} chunk"
        )
        return all_chunks

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        يولّد embeddings لقائمة نصوص.

        Args:
            texts: قائمة النصوص

        Returns:
            List of vectors — كل vector حجمه 384
        """
        self._load_model()
        if not texts:
            raise ValueError("قائمة النصوص فارغة")
        logger.info(f"توليد embeddings لـ {len(texts)} نص...")
        vectors = self._embeddings.embed_documents(texts)
        logger.info(f"تم — عدد المتجهات: {len(vectors)}")
        return vectors

    def embed_query(self, query: str) -> List[float]:
        """
        يحوّل سؤال المستخدم إلى embedding.

        Args:
            query: سؤال المستخدم بالعربية

        Returns:
            vector حجمه 384
        """
        self._load_model()
        if not query or not query.strip():
            raise ValueError("السؤال فارغ")
        return self._embeddings.embed_query(query.strip())


# Singleton
_embedder_instance = None

def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance