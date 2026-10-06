"""
Cognitive Memory Store
======================
Per-user episodic memory backed by per-user in-memory FAISS indexes.

Lifecycle
---------
1. ``memory_retrieve_node`` calls ``retrieve_context(user_id, query)`` before the
   LLM prompt is built — the returned string fills ``{long_term_context}``.
2. ``memory_save_node`` calls ``save_episode(user_id, question, answer)`` after
   the LLM responds — turning today's conversation into tomorrow's Long-Term Memory.

Design decisions
----------------
- Per-user isolation: each user_id gets its own FAISS index so one user's
  history never leaks into another user's context.
- Bounded growth: a ``deque(maxlen=N)`` per user caps episode count at
  ``settings.memory_max_episodes_per_user``.  When the cap is hit, the oldest
  episode is evicted and the index is rebuilt from the remaining buffer.
- Shared embeddings: reuses the FinancialRetriever's already-loaded
  HuggingFaceEmbeddings instance to avoid a second model load.
- Incremental adds: ``FAISS.add_documents()`` is used for normal operation so
  rebuilds only happen at the eviction boundary (rare).
"""

from __future__ import annotations

import time
import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import settings
from logger import logger


def _ts_to_date(ts: float) -> str:
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return ""


class CognitiveMemoryStore:
    """
    In-memory cognitive memory store with per-user FAISS indexes.

    Public interface
    ----------------
    retrieve_context(user_id, query)  → str   (formatted context for LLM prompt)
    save_episode(user_id, q, a)       → None  (persist a Q&A turn)
    clear_user_memory(user_id)        → int   (returns episodes cleared)
    episode_count(user_id)            → int
    """

    def __init__(self) -> None:
        self._embeddings = None
        # per-user FAISS indexes
        self._user_stores: Dict[str, FAISS] = {}
        # per-user bounded episode buffers — (page_content, metadata) tuples
        self._episode_buffers: Dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Embeddings (shared with FinancialRetriever to avoid double-loading)
    # ------------------------------------------------------------------

    def _get_embeddings(self):
        # Prefer the already-initialised retriever embeddings (loaded at startup)
        try:
            from rag.retriever import retriever as _retriever
            if _retriever._embeddings is not None:
                return _retriever._embeddings
        except Exception:
            pass

        if self._embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
            except Exception:
                from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
            self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
            logger.info("CognitiveMemoryStore: loaded embeddings model %s.", settings.embedding_model)

        return self._embeddings

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_context(self, user_id: str, query: str) -> str:
        """Return a formatted string of past episodes relevant to *query*.

        Returns an empty string when the user has no history yet.
        The returned string is safe to inject directly into the LLM system prompt.
        """
        if not user_id or user_id not in self._user_stores:
            return ""

        store = self._user_stores[user_id]
        k = min(settings.memory_top_k, self.episode_count(user_id))
        if k == 0:
            return ""

        try:
            docs = store.similarity_search(query, k=k)
        except Exception as exc:
            logger.warning("Memory retrieval error for user %s: %s", user_id, exc)
            return ""

        if not docs:
            return ""

        lines: List[str] = []
        for doc in docs:
            ts = doc.metadata.get("timestamp", 0)
            date_str = f" [{_ts_to_date(ts)}]" if ts else ""
            lines.append(f"- Past interaction{date_str}:\n  {doc.page_content}")

        context = "\n".join(lines)
        logger.info(
            "Memory: retrieved %d episode(s) for user %s (%d chars).",
            len(docs), user_id, len(context),
        )
        return context

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_episode(self, user_id: str, question: str, answer: str) -> None:
        """Persist a Q&A episode into the user's memory store.

        Eviction strategy: when the buffer is at max capacity, the oldest
        episode is automatically dropped (deque behaviour) and the FAISS index
        is rebuilt from the remaining episodes.  For normal operation
        (buffer not full), a cheap incremental ``add_documents`` is used.
        """
        if not user_id:
            return

        content = f"Q: {question}\nA: {answer}"
        meta: dict = {
            "user_id": user_id,
            "timestamp": time.time(),
            "question": question[:200],  # cap to prevent bloated metadata
        }

        maxlen = settings.memory_max_episodes_per_user
        if user_id not in self._episode_buffers:
            self._episode_buffers[user_id] = deque(maxlen=maxlen)

        buffer = self._episode_buffers[user_id]
        will_evict = len(buffer) == maxlen  # next append drops the oldest entry
        buffer.append((content, meta))

        embeddings = self._get_embeddings()

        if will_evict:
            # Rebuild full index from the evicted buffer (oldest entry now gone)
            docs = [Document(page_content=c, metadata=m) for c, m in buffer]
            self._user_stores[user_id] = FAISS.from_documents(docs, embeddings)
            logger.info(
                "Memory store rebuilt for user %s after eviction (%d episodes).",
                user_id, len(buffer),
            )
        elif user_id not in self._user_stores:
            # First episode — initialise the index
            doc = Document(page_content=content, metadata=meta)
            self._user_stores[user_id] = FAISS.from_documents([doc], embeddings)
            logger.info("Memory store created for user %s.", user_id)
        else:
            # Incremental add (most common path — no rebuild needed)
            doc = Document(page_content=content, metadata=meta)
            self._user_stores[user_id].add_documents([doc])
            logger.info(
                "Memory episode added for user %s (total: %d).",
                user_id, len(buffer),
            )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear_user_memory(self, user_id: str) -> int:
        """Delete all memory for *user_id*.  Returns the number of episodes removed."""
        count = self.episode_count(user_id)
        self._user_stores.pop(user_id, None)
        self._episode_buffers.pop(user_id, None)
        logger.info("Memory cleared for user %s (%d episodes removed).", user_id, count)
        return count

    def episode_count(self, user_id: str) -> int:
        """Return the number of stored episodes for *user_id*."""
        buf = self._episode_buffers.get(user_id)
        return len(buf) if buf else 0

    @property
    def known_users(self) -> List[str]:
        return list(self._user_stores.keys())


# ---------------------------------------------------------------------------
# Module-level singleton — imported by nodes.py
# ---------------------------------------------------------------------------
memory_store = CognitiveMemoryStore()
