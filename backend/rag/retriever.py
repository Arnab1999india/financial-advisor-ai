"""
Enhanced RAG Retriever
======================
Uses FAISS with LangChain Document objects (including metadata) and supports
Maximal Marginal Relevance (MMR) search for diverse, high-quality context
retrieval.
"""

from __future__ import annotations

from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Prefer the dedicated package; fall back to community if unavailable
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

from config import settings
from logger import logger


class FinancialRetriever:
    """
    Manages the FAISS vector store for financial Q&A retrieval.

    Lifecycle
    ---------
    1. Call ``initialize(qa_data)`` once on application startup.
    2. Use ``retrieve(query)`` to get relevant context for any user query.
    """

    def __init__(self) -> None:
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[FAISS] = None
        self.qa_data: List[Dict] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, qa_data: List[Dict]) -> None:
        """Build embeddings model and FAISS index from Q&A data.

        Args:
            qa_data: List of dicts with keys ``question``, ``answer``, and
                     optionally ``category``.
        """
        self.qa_data = qa_data

        logger.info("Loading HuggingFace embeddings model: %s", settings.embedding_model)
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        # Build Document objects with rich metadata for better retrieval
        documents: List[Document] = []
        for item in qa_data:
            doc = Document(
                # Index both question and answer so that answer-oriented queries
                # can still surface the right entry.
                page_content=f"Q: {item['question']}\nA: {item['answer']}",
                metadata={
                    "question": item["question"],
                    "answer": item["answer"],
                    "category": item.get("category", "general"),
                },
            )
            documents.append(doc)

        logger.info("Building FAISS index from %d documents …", len(documents))
        self._vectorstore = FAISS.from_documents(documents, self._embeddings)
        logger.info("FAISS index ready.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Return the top-k most relevant Q&A pairs for *query*.

        Uses MMR (Maximal Marginal Relevance) to balance relevance and
        diversity; falls back to plain similarity search if MMR fails.

        Args:
            query: The user's question.
            k:     Number of results (defaults to ``settings.max_results``).

        Returns:
            List of dicts with keys: ``question``, ``answer``, ``category``.
        """
        if not self._vectorstore:
            logger.warning("Retriever not initialized; returning empty results.")
            return []

        k = k or settings.max_results

        try:
            docs = self._vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=min(k * 4, 20)
            )
        except Exception as mmr_err:
            logger.debug("MMR search failed (%s); falling back to similarity search.", mmr_err)
            docs = self._vectorstore.similarity_search(query, k=k)

        results = [
            {
                "question": doc.metadata.get("question", ""),
                "answer": doc.metadata.get("answer", ""),
                "category": doc.metadata.get("category", "general"),
            }
            for doc in docs
        ]

        logger.info("Retrieved %d document(s) for query.", len(results))
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True once the FAISS index has been built."""
        return self._vectorstore is not None


# ---------------------------------------------------------------------------
# Module-level singleton — imported by nodes and main.py
# ---------------------------------------------------------------------------
retriever = FinancialRetriever()
