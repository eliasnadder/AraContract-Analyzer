"""
RAG Pipeline — يجمع كل الـ modules في workflow واحد نظيف.
هذا هو الملف الوحيد الذي يستورده باقي المشروع.

Workflow:
    Ingestion: clauses → chunks → embeddings → Qdrant
    Query:     question → embedding → retrieve → prompt → LLM → answer
    Summary:   clauses + analyzed_clauses → prompt → LLM → summary
"""

from typing import List, Dict, Any, Optional
import uuid
import logging

from app.services.rag.embedder import get_embedder
from app.services.rag.rag_store import get_rag_store
from app.services.rag.retrieval import get_retriever
from app.services.rag.prompt_builder import get_prompt_builder
from app.services.rag.llm import get_llm

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self._embedder = get_embedder()
        self._store = get_rag_store()
        self._prompt_builder = get_prompt_builder()
        self._llm = get_llm()
        self._retriever = get_retriever(self._store.client)

    def ingest(
        self,
        clauses: List[str],
        session_id: Optional[str] = None
    ) -> str:
        """
        Ingestion Pipeline:
        clauses → chunks → embeddings → Qdrant

        يُستدعى مرة واحدة عند رفع العقد.

        Args:
            clauses: ناتج segment_arabic_text()
            session_id: معرف الجلسة — إذا لم يُمرر يُنشأ تلقائياً

        Returns:
            session_id — يُحفظ عند المستخدم لاستخدامه في ask() لاحقاً
        """
        session_id = session_id or str(uuid.uuid4())
        logger.info(f"بدء الـ Ingestion — session: {session_id}")

        # Step 1: Prepare chunks
        chunks = self._embedder.prepare_chunks(clauses)
        logger.info(f"Step 1 — Chunks: {len(clauses)} بند → {len(chunks)} chunk")

        # Step 2: Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._embedder.embed_texts(texts)
        logger.info(f"Step 2 — Embeddings: {len(embeddings)} vector")

        # Step 3: Create Qdrant collection
        collection_name = self._store.create_collection(session_id)
        logger.info(f"Step 3 — Collection: {collection_name}")

        # Step 4: Store in Qdrant
        stored = self._store.store_chunks(collection_name, chunks, embeddings)
        logger.info(f"Step 4 — Stored: {stored} chunk في Qdrant")

        logger.info(f"Ingestion اكتمل — session_id: {session_id}")
        return session_id

    def ask(
        self,
        session_id: str,
        question: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Query Pipeline:
        question → embedding → retrieve → prompt → LLM → answer

        Args:
            session_id: من ingest()
            question: سؤال المستخدم بالعربية
            top_k: عدد البنود المسترجعة

        Returns:
            {
                "answer": إجابة الـ LLM,
                "retrieved_clauses": البنود المستخدمة في الإجابة,
                "session_id": session_id
            }
        """
        collection_name = f"contract_{session_id}"

        if not self._store.collection_exists(collection_name):
            raise ValueError(
                f"الجلسة غير موجودة: {session_id}\n"
                f"يجب رفع العقد أولاً عبر ingest()"
            )

        if not question or not question.strip():
            raise ValueError("السؤال فارغ")

        logger.info(f"بدء الـ Query — session: {session_id}")

        # Step 1: Embed query
        query_vector = self._embedder.embed_query(question)
        logger.info("Step 1 — Query Embedding: تم")

        # Step 2: Retrieve relevant clauses
        retrieved = self._retriever.retrieve(
            collection_name=collection_name,
            query_vector=query_vector,
            query_text=question,
            top_k=top_k
        )
        logger.info(f"Step 2 — Retrieval: {len(retrieved)} بند")

        if not retrieved:
            return {
                "answer": "لم أتمكن من العثور على معلومات ذات صلة بسؤالك في هذا العقد.",
                "retrieved_clauses": [],
                "session_id": session_id
            }

        # Step 3: Build prompt
        prompt = self._prompt_builder.build_qa_prompt(retrieved, question)
        logger.info(f"Step 3 — Prompt: {len(prompt)} حرف")

        # Step 4: Generate answer
        answer = self._llm.generate(prompt, max_tokens=512)
        logger.info("Step 4 — Generation: تم")

        return {
            "answer": answer,
            "retrieved_clauses": [
                {
                    "clause_index": r["clause_index"],
                    "parent_text": r["parent_text"],
                    "score": r["score"]
                }
                for r in retrieved
            ],
            "session_id": session_id
        }

    def summarize(
        self,
        clauses: List[str],
        analyzed_clauses: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Summary Pipeline (FR-6):
        clauses + analyzed_clauses → prompt → LLM → summary

        Args:
            clauses: كل بنود العقد
            analyzed_clauses: ناتج المصنّف (اختياري)
                              يحتوي على predicted_risk_level, warning

        Returns:
            ملخص تنفيذي 3-5 جمل بالعربية
        """
        if not clauses:
            raise ValueError("قائمة البنود فارغة")

        logger.info(f"بدء الـ Summary — {len(clauses)} بند")

        # Step 1: Build summary prompt
        prompt = self._prompt_builder.build_summary_prompt(
            clauses=clauses,
            analyzed_clauses=analyzed_clauses
        )
        logger.info(f"Step 1 — Summary Prompt: {len(prompt)} حرف")

        # Step 2: Generate summary
        summary = self._llm.generate(prompt, max_tokens=300)
        logger.info("Step 2 — Summary Generation: تم")

        return summary

    def cleanup(self, session_id: str) -> bool:
        """
        يحذف الـ collection عند انتهاء الجلسة.

        Args:
            session_id: من ingest()

        Returns:
            True إذا تم الحذف
        """
        collection_name = f"contract_{session_id}"
        deleted = self._store.delete_collection(collection_name)
        if deleted:
            logger.info(f"تم حذف جلسة: {session_id}")
        return deleted


# Singleton
_pipeline_instance = None

def get_rag_pipeline() -> RAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance