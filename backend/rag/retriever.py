"""
Enhanced RAG Retriever — hybrid FAISS + BM25 with score thresholding.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

from cache.embedding_cache import clear_embeddings, get_embedding, put_embedding
from cache.response_cache import set_index_epoch
from config import settings
from logger import logger
from rag.bm25_index import BM25Index
from security.injection_scan import scan_chunk_for_ingestion

_RETRIEVE_CACHE: dict = {}
_RETRIEVE_CACHE_MAX = 128
_TICKER = re.compile(r"\b[A-Z]{1,5}(?:\.(?:NS|BO))?\b")


class FinancialRetriever:
    """FAISS + BM25 hybrid retriever with thresholding and metadata-rich chunks."""

    def __init__(self) -> None:
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[FAISS] = None
        self._bm25: Optional[BM25Index] = None
        self._chunk_ids: List[str] = []
        self._records_by_id: Dict[str, dict] = {}
        self.quarantined_chunks: List[dict] = []
        self.qa_data: List[Dict] = []
        self._indexed_at: str = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self.index_epoch: str = self._indexed_at
        self.init_error: str | None = None

    def initialize(self, qa_data: List[Dict]) -> None:
        self.qa_data = qa_data
        self._indexed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        logger.info("Loading HuggingFace embeddings model: %s", settings.embedding_model)
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        documents: List[Document] = []
        bm25_corpus: List[str] = []
        self._chunk_ids = []
        self._records_by_id = {}
        self.quarantined_chunks = []

        for index, item in enumerate(qa_data):
            finding = scan_chunk_for_ingestion(item.get("question", ""), item.get("answer", ""))
            if finding.blocked:
                chunk_id = str(item.get("id") or f"kb-{index}")
                self.quarantined_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "reason_code": finding.reason_code,
                        "pattern_id": finding.pattern_id,
                    }
                )
                logger.warning(
                    "Quarantined KB chunk %s reason=%s pattern=%s",
                    chunk_id,
                    finding.reason_code,
                    finding.pattern_id,
                )
                continue

            chunk_id = str(item.get("id") or f"kb-{index}")
            doc_date = item.get("document_date") or item.get("updated_at") or self._indexed_at
            meta = {
                "chunk_id": chunk_id,
                "question": item["question"],
                "answer": item["answer"],
                "category": item.get("category", "general"),
                "section": item.get("section") or item.get("category", "general"),
                "source_file": item.get("source_file") or settings.data_file,
                "page": item.get("page", index + 1),
                "document_date": doc_date,
                "updated_at": doc_date,
            }
            page_content = f"Q: {item['question']}\nA: {item['answer']}"
            documents.append(Document(page_content=page_content, metadata=meta))
            bm25_corpus.append(page_content)
            self._chunk_ids.append(chunk_id)
            self._records_by_id[chunk_id] = dict(meta)

        self._bm25 = BM25Index(bm25_corpus) if settings.hybrid_bm25_enabled else None
        logger.info("Building FAISS index from %d documents …", len(documents))
        self._vectorstore = FAISS.from_documents(documents, self._embeddings)
        self.index_epoch = self._indexed_at
        _RETRIEVE_CACHE.clear()
        clear_embeddings()
        set_index_epoch(self.index_epoch)
        self.init_error = None
        logger.info("FAISS index ready (hybrid BM25=%s).", settings.hybrid_bm25_enabled)

    def embed_query_cached(self, query: str) -> list[float]:
        cached = get_embedding(query)
        if cached is not None:
            return cached
        if not self._embeddings:
            raise RuntimeError("Embeddings not initialised")
        from resilience.external import call_external

        vector = call_external(
            "embeddings",
            lambda: self._embeddings.embed_query(query),
            timeout_seconds=settings.embedding_timeout_seconds,
        )
        put_embedding(query, vector)
        return vector

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        if not self._vectorstore:
            logger.warning("Retriever not initialized; returning empty results.")
            return []

        fetch_k = k or settings.retrieval_fetch_k
        final_k = settings.retrieval_final_k
        cache_key = (query.strip().lower(), fetch_k, final_k, settings.retrieval_min_confidence)

        if cache_key in _RETRIEVE_CACHE:
            logger.info("Retriever cache hit for query.")
            return _RETRIEVE_CACHE[cache_key]

        scored = self._hybrid_search(query, fetch_k=fetch_k)
        scored = self._rerank_with_recency(scored)
        scored = [(cid, conf) for cid, conf in scored if conf >= settings.retrieval_min_confidence]
        scored = scored[:final_k]

        retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        results: List[Dict] = []
        seen_chunks: set[str] = set()
        for chunk_id, confidence in scored:
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            meta = self._records_by_id.get(chunk_id, {})
            results.append(
                {
                    "chunk_id": chunk_id,
                    "question": meta.get("question", ""),
                    "answer": (meta.get("answer", "") or "")[: settings.chunk_max_chars],
                    "category": meta.get("category", "general"),
                    "section": meta.get("section", meta.get("category", "general")),
                    "source_file": meta.get("source_file", settings.data_file),
                    "page": meta.get("page"),
                    "document_date": meta.get("document_date") or self._indexed_at,
                    "confidence": round(confidence, 4),
                    "retrieved_at": retrieved_at,
                    "source_timestamp": meta.get("document_date") or self._indexed_at,
                    "source_type": "faiss",
                }
            )

        if len(_RETRIEVE_CACHE) >= _RETRIEVE_CACHE_MAX:
            _RETRIEVE_CACHE.pop(next(iter(_RETRIEVE_CACHE)))
        _RETRIEVE_CACHE[cache_key] = results

        logger.info(
            "Retrieved %d document(s) after threshold (>=%.2f).",
            len(results),
            settings.retrieval_min_confidence,
        )
        return results

    def _hybrid_search(self, query: str, *, fetch_k: int) -> list[tuple[str, float]]:
        assert self._vectorstore is not None
        faiss_rank: dict[str, float] = {}
        try:
            scored_docs = self._vectorstore.similarity_search_with_score(query, k=fetch_k)
            for rank, (doc, distance) in enumerate(scored_docs):
                cid = str(doc.metadata.get("chunk_id") or "")
                if not cid:
                    continue
                confidence = 1.0 / (1.0 + max(float(distance), 0.0))
                faiss_rank[cid] = max(faiss_rank.get(cid, 0.0), confidence)
        except Exception as exc:
            logger.warning("FAISS scored search failed (%s); using BM25 only.", exc)

        bm25_rank: dict[str, float] = {}
        if self._bm25 is not None:
            raw_scores = self._bm25.search(query, k=fetch_k)
            if raw_scores:
                max_bm25 = max(s for _, s in raw_scores) or 1.0
                for doc_idx, score in raw_scores:
                    if doc_idx >= len(self._chunk_ids):
                        continue
                    cid = self._chunk_ids[doc_idx]
                    bm25_rank[cid] = max(bm25_rank.get(cid, 0.0), score / max_bm25)

        ticker_boost = 0.08 if _TICKER.search(query or "") else 0.0
        bm25_weight = settings.hybrid_bm25_weight + ticker_boost
        faiss_weight = 1.0 - bm25_weight

        candidates = set(faiss_rank) | set(bm25_rank)
        fused: list[tuple[str, float]] = []
        for cid in candidates:
            score = faiss_weight * faiss_rank.get(cid, 0.0) + bm25_weight * bm25_rank.get(cid, 0.0)
            fused.append((cid, score))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[: max(fetch_k, settings.retrieval_final_k)]

    def _rerank_with_recency(self, scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
        now = datetime.now(timezone.utc)

        def recency_boost(chunk_id: str) -> float:
            meta = self._records_by_id.get(chunk_id, {})
            raw = str(meta.get("document_date") or "")
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_days = max((now - dt).days, 0)
                return max(0.0, 0.06 * (1.0 - min(age_days, 3650) / 3650))
            except Exception:
                return 0.0

        reranked = [(cid, score + recency_boost(cid)) for cid, score in scored]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None


retriever = FinancialRetriever()
