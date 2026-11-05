"""Ingestion- and query-time injection detection."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Literal, Optional

from security.policy import compiled_patterns

ReasonCode = Literal[
    "INGESTION_INJECTION",
    "QUERY_OVERRIDE",
    "QUERY_JAILBREAK",
    "QUERY_EXFILTRATION",
    "QUERY_ENCODED_PAYLOAD",
    "QUERY_BORDERLINE",
]

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_HTML_CMD_COMMENT = re.compile(r"<!--[\s\S]*?(?:ignore|instruction|system)[\s\S]*?-->", re.IGNORECASE)
_B64_BLOB = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")


@dataclass
class InjectionFinding:
    blocked: bool
    reason_code: Optional[ReasonCode] = None
    pattern_id: str = ""


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> Optional[str]:
    for pat in patterns:
        if pat.search(text):
            return pat.pattern
    return None


def scan_chunk_for_ingestion(question: str, answer: str) -> InjectionFinding:
    blob = f"{question}\n{answer}"
    if _ZERO_WIDTH.search(blob):
        return InjectionFinding(True, "INGESTION_INJECTION", "zero_width")
    if _HTML_CMD_COMMENT.search(blob):
        return InjectionFinding(True, "INGESTION_INJECTION", "html_comment")
    if _B64_BLOB.search(blob):
        return InjectionFinding(True, "INGESTION_INJECTION", "base64_blob")
    hit = _first_match(compiled_patterns("ingestion_injection_patterns"), blob)
    if hit:
        return InjectionFinding(True, "INGESTION_INJECTION", hit[:80])
    return InjectionFinding(False)


def _looks_encoded_exfiltration(text: str) -> bool:
    lowered = text.lower()
    if "base64" in lowered and ("prompt" in lowered or "key" in lowered):
        return True
    for token in re.findall(r"\b[A-Za-z0-9+/]{32,}={0,2}\b", text):
        try:
            padded = token + "=" * (-len(token) % 4)
            decoded = base64.b64decode(padded, validate=True).decode("utf-8", errors="ignore").lower()
            if any(k in decoded for k in ("instruction", "system prompt", "api key", "ignore")):
                return True
        except Exception:
            continue
    return False


def scan_user_query(text: str, *, borderline: bool = False) -> InjectionFinding:
    blob = text or ""
    if _looks_encoded_exfiltration(blob):
        return InjectionFinding(True, "QUERY_ENCODED_PAYLOAD", "encoded_exfiltration")

    exfil = _first_match(compiled_patterns("prompt_exfiltration_patterns"), blob)
    if exfil:
        return InjectionFinding(True, "QUERY_EXFILTRATION", exfil[:80])

    jailbreak_tokens = ("jailbreak", "developer mode", "dan mode", "unrestricted", "bypass")
    if any(t in blob.lower() for t in jailbreak_tokens):
        hit = _first_match(compiled_patterns("query_injection_patterns"), blob)
        return InjectionFinding(True, "QUERY_JAILBREAK", (hit or "jailbreak_keyword")[:80])

    hit = _first_match(compiled_patterns("query_injection_patterns"), blob)
    if hit:
        if "instruction" in hit or "disregard" in hit or "ignore" in hit:
            return InjectionFinding(True, "QUERY_OVERRIDE", hit[:80])
        return InjectionFinding(True, "QUERY_JAILBREAK", hit[:80])

    if borderline:
        return InjectionFinding(True, "QUERY_BORDERLINE", "borderline_classifier")
    return InjectionFinding(False)


def is_borderline_query(text: str) -> bool:
    """Heuristic triggers for optional classifier (not implemented as LLM here)."""
    lowered = (text or "").lower()
    hints = ("instruction", "prompt", "rule", "system", "uda do", "batao", "bhool")
    return any(h in lowered for h in hints)
