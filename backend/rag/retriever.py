"""
Enhanced RAG Retriever
======================
Uses FAISS with LangChain Document objects (including metadata) and supports
Maximal Marginal Relevance (MMR) search for diverse, high-quality context
retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone
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

# ---------------------------------------------------------------------------
# Module-level query cache — avoids re-embedding and re-searching FAISS for
# repeated or near-duplicate queries.  Bounded FIFO; keys are (query, k) tuples.
# ---------------------------------------------------------------------------
_RETRIEVE_CACHE: dict = {}
_RETRIEVE_CACHE_MAX = 128


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
        self._indexed_at: str = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

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

        self._indexed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        logger.info("Loading HuggingFace embeddings model: %s", settings.embedding_model)
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        # Build Document objects with rich metadata for better retrieval
        documents: List[Document] = []
        for index, item in enumerate(qa_data):
            chunk_id = str(item.get("id") or f"kb-{index}")
            doc = Document(
                # Index both question and answer so that answer-oriented queries
                # can still surface the right entry.
                page_content=f"Q: {item['question']}\nA: {item['answer']}",
                metadata={
                    "chunk_id": chunk_id,
                    "question": item["question"],
                    "answer": item["answer"],
                    "category": item.get("category", "general"),
                    "updated_at": item.get("updated_at") or self._indexed_at,
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

        Results are cached by (normalised_query, k) so repeated or
        near-duplicate questions skip FAISS search and embedding entirely.

        Uses MMR (Maximal Marginal Relevance) to balance relevance and
        diversity; falls back to plain similarity search if MMR fails.

        Args:
            query: The user's question.
            k:     Number of results (defaults to ``settings.max_results``).

        Returns:
            List of dicts with keys: ``question``, ``answer``, ``category``,
            ``chunk_id``, ``confidence``, ``retrieved_at``, ``source_timestamp``,
            ``source_type``.
        """
        if not self._vectorstore:
            logger.warning("Retriever not initialized; returning empty results.")
            return []

        k = k or settings.max_results
        cache_key = (query.strip().lower(), k)

        if cache_key in _RETRIEVE_CACHE:
            logger.info("Retriever cache hit for query.")
            return _RETRIEVE_CACHE[cache_key]

        try:
            scored_docs = self._vectorstore.similarity_search_with_score(query, k=k)
        except Exception as score_err:
            logger.debug("Scored search failed (%s); falling back to MMR.", score_err)
            try:
                plain = self._vectorstore.max_marginal_relevance_search(
                    query, k=k, fetch_k=min(k * 4, 20)
                )
                scored_docs = [(doc, 0.5) for doc in plain]
            except Exception as mmr_err:
                logger.debug("MMR search failed (%s); falling back to similarity search.", mmr_err)
                plain = self._vectorstore.similarity_search(query, k=k)
                scored_docs = [(doc, 0.5) for doc in plain]

        retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        results = []
        for doc, distance in scored_docs:
            dist = float(distance)
            confidence = round(1.0 / (1.0 + max(dist, 0.0)), 4)
            results.append(
                {
                    "chunk_id": doc.metadata.get("chunk_id") or "",
                    "question": doc.metadata.get("question", ""),
                    "answer": doc.metadata.get("answer", ""),
                    "category": doc.metadata.get("category", "general"),
                    "confidence": confidence,
                    "retrieved_at": retrieved_at,
                    "source_timestamp": doc.metadata.get("updated_at") or self._indexed_at,
                    "source_type": "faiss",
                }
            )

        # Evict oldest entry when cache is full (FIFO)
        if len(_RETRIEVE_CACHE) >= _RETRIEVE_CACHE_MAX:
            _RETRIEVE_CACHE.pop(next(iter(_RETRIEVE_CACHE)))
        _RETRIEVE_CACHE[cache_key] = results

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
