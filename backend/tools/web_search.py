"""
External web search fallback
============================
Used when FAISS retrieval is graded as not answering the user query.

Preference order:
1. Tavily  — if ``TAVILY_API_KEY`` is set
2. DuckDuckGo — no key required
"""

from __future__ import annotations

from typing import Any
from rag.citations import utc_now_iso

from config import settings
from logger import logger


def search_web(query: str, max_results: int | None = None) -> list[dict[str, str]]:
    """Return a list of ``{title, url, snippet}`` hits for *query*."""
    limit = max_results or settings.web_search_max_results
    tavily_key = (settings.tavily_api_key or "").strip()
    if tavily_key:
        hits = _search_tavily(query, limit, tavily_key)
        if hits:
            return _annotate_hits(hits)
        logger.warning("Tavily returned no results; falling back to DuckDuckGo.")
    return _annotate_hits(_search_duckduckgo(query, limit))


def format_web_context(hits: list[dict[str, str]]) -> str:
    """Format search hits for injection into the generation prompt."""
    if not hits:
        return ""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[WEB {i}] {hit.get('title', '').strip()}\n"
            f"URL: {hit.get('url', '').strip()}\n"
            f"{hit.get('snippet', '').strip()}"
        )
    return "\n\n".join(blocks)


def _annotate_hits(hits: list[dict[str, str]]) -> list[dict[str, str]]:
    retrieved_at = utc_now_iso()
    annotated = []
    n = max(len(hits), 1)
    for i, hit in enumerate(hits):
        annotated.append(
            {
                **hit,
                "chunk_id": hit.get("chunk_id") or f"web-{i}",
                "confidence": round(1.0 - (i / (n + 1)), 4),
                "retrieved_at": retrieved_at,
                "source_timestamp": hit.get("published_date") or retrieved_at,
                "source_type": "web",
            }
        )
    return annotated


def _search_tavily(query: str, limit: int, api_key: str) -> list[dict[str, str]]:
    try:
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=api_key)
        payload: dict[str, Any] = client.search(query, max_results=limit)
        results = payload.get("results") or []
        hits = []
        for item in results[:limit]:
            hits.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "snippet": str(item.get("content") or item.get("snippet") or ""),
                    "published_date": str(item.get("published_date") or item.get("published_at") or ""),
                }
            )
        logger.info("Tavily web search returned %d hit(s).", len(hits))
        return hits
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []


def _search_duckduckgo(query: str, limit: int) -> list[dict[str, str]]:
    try:
        from duckduckgo_search import DDGS  # type: ignore

        hits: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=limit):
                hits.append(
                    {
                        "title": str(item.get("title") or ""),
                        "url": str(item.get("href") or item.get("url") or ""),
                        "snippet": str(item.get("body") or item.get("snippet") or ""),
                    }
                )
        logger.info("DuckDuckGo web search returned %d hit(s).", len(hits))
        return hits
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []
