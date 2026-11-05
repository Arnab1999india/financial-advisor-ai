"""Format retrieved chunks for grounded LLM prompts with source metadata."""

from __future__ import annotations

from typing import Iterable

from security.delimiters import sanitize_untrusted_text


def source_citation_label(doc: dict) -> str:
    if doc.get("source_type") == "web":
        url = doc.get("source_url") or doc.get("url") or "web"
        fetched = doc.get("retrieved_at") or doc.get("source_timestamp") or ""
        return f"[Source: {url}, fetched {fetched}]"
    if doc.get("source_type") == "market_data":
        provider = doc.get("provider") or "market data provider"
        ts = doc.get("source_timestamp") or doc.get("as_of") or ""
        return f"[Market data: {provider}, {ts}]"
    src_file = doc.get("source_file") or "knowledge_base"
    page = doc.get("page")
    page_label = f", page {page}" if page is not None else ""
    doc_date = doc.get("document_date") or doc.get("source_timestamp") or ""
    date_suffix = f", document date {doc_date}" if doc_date else ""
    section = doc.get("section") or doc.get("category")
    section_suffix = f", section {section}" if section else ""
    return f"[Source: {src_file}{page_label}{date_suffix}{section_suffix}]"


def format_grounded_context(docs: Iterable[dict], *, web_preamble: str = "") -> str:
    blocks: list[str] = []
    for doc in docs:
        cite = source_citation_label(doc)
        body = sanitize_untrusted_text((doc.get("answer") or doc.get("snippet") or "").strip())
        question = sanitize_untrusted_text((doc.get("question") or doc.get("title") or "").strip())
        if doc.get("source_type") == "web" and web_preamble:
            blocks.append(f"{web_preamble}\n{cite}\nQ: {question}\n{body}")
        else:
            blocks.append(f"{cite}\nQ: {question}\nA: {body}")
    return "\n\n".join(blocks)
