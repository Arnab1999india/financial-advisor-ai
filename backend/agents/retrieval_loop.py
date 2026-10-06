"""
Retrieval grading loop
======================
After FAISS retrieval, a lightweight Flash grader checks whether the
documents actually answer the user query.

If relevance is low:
1. Query rewriter (Flash) rephrases the question and FAISS is queried again.
2. If still low (or rewrite budget exhausted), an external web search
   (Tavily or DuckDuckGo) supplies context before generation.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from config import settings
from logger import logger
from telemetry import make_token_event
from tools.web_search import format_web_context, search_web

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "whom", "this", "that", "these", "those", "and", "or", "of", "to", "in",
    "on", "for", "with", "how", "why", "does", "do", "did", "can", "could",
    "should", "would", "please", "tell", "me", "about", "explain",
}

_GRADE_SYSTEM = (
    "You are a binary relevance grader for a financial knowledge base. "
    "Decide if the retrieved documents contain enough information to answer "
    "the user question. Reply with ONLY yes or no."
)

_REWRITE_SYSTEM = (
    "You rewrite financial questions to improve vector-store recall. "
    "Keep the same intent. Use standard finance terms. "
    "Output ONLY the rewritten question, one line, no quotes."
)


def _demo_mode() -> bool:
    key = (settings.gemini_api_key or "").strip()
    return not key or key == "your-gemini-api-key-here"


def _content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(tok) > 2 and tok not in _STOPWORDS
    }


def heuristic_docs_answer_query(query: str, docs: list[dict]) -> bool:
    """Lexical overlap check used when Flash is unavailable."""
    if not docs:
        return False
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return bool(docs)
    for doc in docs:
        blob = f"{doc.get('question', '')} {doc.get('answer', '')}"
        overlap = q_tokens & _content_tokens(blob)
        if len(overlap) >= max(1, min(3, len(q_tokens) // 2)):
            return True
    return False


def _format_docs_for_grade(docs: list[dict]) -> str:
    if not docs:
        return "(no documents retrieved)"
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(
            f"{i}. Q: {doc.get('question', '')}\n   A: {(doc.get('answer') or '')[:400]}"
        )
    return "\n".join(parts)


def _parse_yes_no(text: str) -> bool:
    token = (text or "").strip().lower()
    token = re.split(r"[^a-z]+", token, maxsplit=1)[0]
    if token in {"yes", "y", "relevant"}:
        return True
    if token in {"no", "n", "irrelevant"}:
        return False
    lowered = (text or "").lower()
    return "yes" in lowered and "no" not in lowered


def grade_documents_node(state: AgentState) -> dict:
    """Flash grader: do the current FAISS docs actually answer the query?"""
    query = (state.get("query") or "").strip()
    docs = state.get("retrieved_docs") or []

    if not docs:
        logger.info("Retrieval grader: no documents — not_relevant.")
        return {"retrieval_grade": "not_relevant"}

    if _demo_mode():
        relevant = heuristic_docs_answer_query(query, docs)
        grade = "relevant" if relevant else "not_relevant"
        logger.info("Retrieval grader (heuristic): %s.", grade)
        return {"retrieval_grade": grade}

    try:
        from agents.nodes import _get_cheap_llm

        user = (
            f"Question:\n{query}\n\nRetrieved documents:\n{_format_docs_for_grade(docs)}\n\n"
            "Do these documents answer the question? yes or no:"
        )
        response = _get_cheap_llm().invoke(
            [SystemMessage(content=_GRADE_SYSTEM), HumanMessage(content=user)]
        )
        relevant = _parse_yes_no(response.content or "")
        grade = "relevant" if relevant else "not_relevant"
        logger.info("Retrieval grader (Flash): %s.", grade)
        return {
            "retrieval_grade": grade,
            "token_calls": [make_token_event("grade_documents", settings.cheap_model, response)],
        }
    except Exception as exc:
        logger.warning("Retrieval grader failed (%s); using heuristic.", exc)
        relevant = heuristic_docs_answer_query(query, docs)
        return {"retrieval_grade": "relevant" if relevant else "not_relevant"}


def query_rewrite_node(state: AgentState) -> dict:
    """Rephrase the question for a second FAISS pass."""
    original = (state.get("query") or "").strip()
    current = (state.get("retrieval_query") or original).strip()
    attempts = int(state.get("rewrite_attempts") or 0) + 1
    rewritten = current
    token_calls: list[dict] = []

    if _demo_mode():
        extra = " financial definition explanation"
        rewritten = current if extra.strip() in current.lower() else f"{current}{extra}"
    else:
        try:
            from agents.nodes import _get_cheap_llm

            user = (
                f"Original question: {original}\n"
                f"Previous search query: {current}\n"
                "Rewritten search query:"
            )
            response = _get_cheap_llm().invoke(
                [SystemMessage(content=_REWRITE_SYSTEM), HumanMessage(content=user)]
            )
            token_calls.append(make_token_event("query_rewrite", settings.cheap_model, response))
            candidate = (response.content or "").strip().strip('"').splitlines()[0].strip()
            if candidate:
                rewritten = candidate
        except Exception as exc:
            logger.warning("Query rewrite failed (%s); appending finance keywords.", exc)
            rewritten = f"{current} finance investing"

    logger.info("Query rewriter (attempt %d): %r → %r", attempts, current, rewritten)
    payload: dict = {
        "retrieval_query": rewritten,
        "rewrite_attempts": attempts,
        "retrieval_grade": None,
    }
    if token_calls:
        payload["token_calls"] = token_calls
    return payload


def web_search_node(state: AgentState) -> dict:
    """Fill context from Tavily or DuckDuckGo when FAISS cannot answer."""
    query = (state.get("retrieval_query") or state.get("query") or "").strip()
    hits = search_web(query)
    context = format_web_context(hits)
    source_questions = [h["title"] for h in hits if h.get("title")]

    if not context:
        logger.warning("Web search returned no usable context.")
        return {
            "context": state.get("context") or "",
            "source_questions": state.get("source_questions") or [],
            "retrieval_source": "web",
            "retrieval_grade": "relevant",
        }

    logger.info("Web search context built from %d hit(s).", len(hits))
    return {
        "context": context,
        "source_questions": source_questions,
        "retrieved_docs": [
            {
                "chunk_id": h.get("chunk_id") or f"web-{i}",
                "question": h.get("title") or query,
                "answer": h.get("snippet") or "",
                "category": "web",
                "confidence": float(h.get("confidence") or 0.0),
                "retrieved_at": h.get("retrieved_at") or "",
                "source_timestamp": h.get("source_timestamp") or h.get("published_date") or "",
                "source_url": h.get("url") or "",
                "source_type": "web",
            }
            for i, h in enumerate(hits)
        ],
        "retrieval_source": "web",
        "retrieval_grade": "relevant",
    }


def dispatch_retrieval_grade(state: AgentState) -> str:
    """Route after grading: generate, rewrite+retrieve, or web search."""
    grade = state.get("retrieval_grade") or "relevant"
    if grade == "relevant":
        return "generate"
    attempts = int(state.get("rewrite_attempts") or 0)
    if attempts < settings.max_query_rewrites:
        return "rewrite"
    return "web_search"
