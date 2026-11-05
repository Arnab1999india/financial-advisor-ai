"""Validate structured model output before returning to clients."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.domain_classifier import load_domain_config
from agents.outbound_guardrail import _ADVICE_PATTERNS
from rag.citations import sentence_similarity, split_sentences
from validation.structured_output import StructuredAnswer

_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
_AADHAAR = re.compile(r"\b(?:\d{4}\s?){3}\d{4}\b")
_NUMBER = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?%?\b",
)
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r"\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_TICKER = re.compile(r"\b[A-Z]{1,5}(?:\.(?:NS|BO))?\b")
_LEAKAGE = re.compile(
    r"(gemini_api|api[_ -]?key|sk-[a-z0-9]{10,}|INSTRUCTION HIERARCHY|"
    r"GROUNDING RULES|<context_data>|</context_data>|<tool_data>|"
    r"\\backend\\|/etc/|C:\\Users\\)",
    re.IGNORECASE,
)
_GUARANTEE = re.compile(r"\bguarantee(?:d)?\s+(?:returns?|profit|income)\b", re.IGNORECASE)


@dataclass
class OutputValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    sanitized_answer: str = ""


def build_evidence_corpus(docs: list[dict], context: str) -> str:
    parts = [context or ""]
    for doc in docs or []:
        parts.append(doc.get("answer") or doc.get("snippet") or "")
        parts.append(str(doc.get("source_file") or ""))
        parts.append(str(doc.get("source_url") or ""))
        parts.append(str(doc.get("page") or ""))
        parts.append(str(doc.get("document_date") or doc.get("source_timestamp") or ""))
        parts.append(doc.get("question") or doc.get("title") or "")
    return "\n".join(p for p in parts if p).lower()


def _known_sources(docs: list[dict]) -> set[str]:
    known: set[str] = set()
    for doc in docs or []:
        for key in ("source_file", "source_url", "question", "title"):
            val = (doc.get(key) or "").strip().lower()
            if val:
                known.add(val)
    return known


def _known_pages(docs: list[dict]) -> set[int]:
    pages: set[int] = set()
    for doc in docs or []:
        page = doc.get("page")
        if isinstance(page, int) and page > 0:
            pages.add(page)
    return pages


def validate_structured_output(
    structured: StructuredAnswer,
    *,
    docs: list[dict],
    context: str,
    used_market_data: bool,
) -> OutputValidationResult:
    reasons: list[str] = []
    answer = structured.answer.strip()

    if structured.used_market_data and not used_market_data:
        reasons.append("used_market_data_mismatch")
    if used_market_data and not structured.used_market_data:
        reasons.append("missing_used_market_data_flag")

    known = _known_sources(docs)
    pages = _known_pages(docs)
    for cite in structured.citations:
        src = cite.source.strip().lower()
        if not any(src in k or k in src for k in known):
            reasons.append(f"citation_unknown_source:{cite.source[:40]}")
        if cite.page is not None and cite.page not in pages:
            reasons.append(f"citation_unknown_page:{cite.page}")

    corpus = build_evidence_corpus(docs, context)
    for match in _NUMBER.findall(answer):
        token = match.replace(",", "").lower()
        if token in {"0", "1", "2", "3", "4", "5"}:
            continue
        if token.rstrip("%") not in corpus and match.lower() not in corpus:
            reasons.append(f"untracked_number:{match}")
    for match in _DATE.findall(answer):
        if match.lower() not in corpus:
            reasons.append(f"untracked_date:{match}")
    for match in _TICKER.findall(answer):
        if match.lower() not in corpus and match not in corpus.upper():
            reasons.append(f"untracked_ticker:{match}")

    if docs or context.strip():
        supported = 0
        for sentence in split_sentences(answer):
            if len(sentence) < 12:
                supported += 1
                continue
            best = 0.0
            for doc in docs:
                body = doc.get("answer") or doc.get("snippet") or ""
                best = max(best, sentence_similarity(sentence, body))
            if best >= 0.22:
                supported += 1
        if supported == 0 and len(split_sentences(answer)) > 0:
            reasons.append("grounding_overlap_low")

    cfg = load_domain_config()
    for pat in cfg.get("off_topic_deny_patterns") or []:
        if re.search(pat, answer, re.IGNORECASE):
            reasons.append("domain_off_topic")
            break

    if _LEAKAGE.search(answer):
        reasons.append("leakage_detected")
    if _PAN.search(answer) or _AADHAAR.search(answer):
        reasons.append("pii_echo")
    if any(p.search(answer) for p in _ADVICE_PATTERNS):
        reasons.append("directive_advice")
    if _GUARANTEE.search(answer):
        reasons.append("guaranteed_returns")

    if reasons:
        return OutputValidationResult(ok=False, reasons=reasons, sanitized_answer=answer)
    return OutputValidationResult(ok=True, sanitized_answer=answer)


def strip_untracked_sentences(answer: str, docs: list[dict], context: str) -> str:
    """Remove sentences containing numbers that cannot be traced to evidence."""
    corpus = build_evidence_corpus(docs, context)
    kept: list[str] = []
    for sentence in split_sentences(answer):
        nums = _NUMBER.findall(sentence) + _DATE.findall(sentence)
        if not nums:
            kept.append(sentence)
            continue
        untracked = False
        for token in nums:
            norm = token.replace(",", "").lower()
            if norm in {"0", "1", "2", "3", "4", "5"}:
                continue
            if norm.rstrip("%") not in corpus and token.lower() not in corpus:
                untracked = True
                break
        if not untracked:
            kept.append(sentence)
    return " ".join(kept).strip() or answer
