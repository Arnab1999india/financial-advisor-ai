"""
Sentence-level grounding
========================
Maps each sentence of the generated answer onto the closest sentence in a
retrieved knowledge-base chunk or web snippet, with a 0–1 confidence score.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9]+")
_MIN_MAP_CONFIDENCE = 0.28


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def split_sentences(text: str) -> list[str]:
    """Split *text* into non-empty sentences, preserving order."""
    if not (text or "").strip():
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").lower()) if len(t) > 2}


def sentence_similarity(left: str, right: str) -> float:
    """Blend Jaccard overlap with sequence ratio for short financial sentences."""
    if not left.strip() or not right.strip():
        return 0.0
    a, b = _tokens(left), _tokens(right)
    jaccard = (len(a & b) / len(a | b)) if a and b else 0.0
    ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return round(0.65 * jaccard + 0.35 * ratio, 4)


def map_answer_to_chunks(answer: str, chunks: list[dict]) -> list[dict]:
    """Return exact sentence mappings from *answer* onto *chunks*.

    Each mapping includes character offsets into the answer so the frontend
    can highlight the grounded span.
    """
    mappings: list[dict] = []
    if not answer or not chunks:
        return mappings

    cursor = 0
    for sentence in split_sentences(answer):
        start = answer.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        cursor = end

        best: dict | None = None
        best_score = _MIN_MAP_CONFIDENCE
        for chunk in chunks:
            source_body = (chunk.get("answer") or chunk.get("snippet") or "").strip()
            source_sentences = split_sentences(source_body) or [source_body]
            for src in source_sentences:
                score = sentence_similarity(sentence, src)
                if score > best_score:
                    best_score = score
                    best = {
                        "answer_sentence": sentence,
                        "answer_start": start,
                        "answer_end": end,
                        "chunk_id": chunk.get("chunk_id") or "",
                        "source_title": chunk.get("question") or chunk.get("title") or "",
                        "source_sentence": src,
                        "source_type": chunk.get("source_type") or "faiss",
                        "confidence": round(score, 4),
                        "source_timestamp": chunk.get("source_timestamp")
                        or chunk.get("retrieved_at")
                        or "",
                    }
        if best:
            mappings.append(best)
    return mappings


def source_chunks_payload(chunks: list[dict]) -> list[dict]:
    """Normalise retrieved docs/web hits for the API and UI."""
    payload = []
    for chunk in chunks or []:
        payload.append(
            {
                "chunk_id": chunk.get("chunk_id") or "",
                "title": chunk.get("question") or chunk.get("title") or "",
                "snippet": chunk.get("answer") or chunk.get("snippet") or "",
                "category": chunk.get("category") or "general",
                "confidence": float(chunk.get("confidence") or 0.0),
                "retrieved_at": chunk.get("retrieved_at") or "",
                "source_timestamp": chunk.get("source_timestamp") or chunk.get("retrieved_at") or "",
                "source_url": chunk.get("source_url") or chunk.get("url") or None,
                "source_type": chunk.get("source_type") or "faiss",
            }
        )
    return payload
