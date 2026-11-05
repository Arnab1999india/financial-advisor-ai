"""Normalize and validate user text fields."""

from __future__ import annotations

import re
import unicodedata

from security.policy import load_security_policy

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_REPEAT = re.compile(r"(.+?)\1{4,}", re.DOTALL)
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")


class TextValidationError(ValueError):
    pass


def normalize_user_text(text: str) -> str:
    raw = unicodedata.normalize("NFKC", (text or ""))
    raw = _ZERO_WIDTH.sub("", raw)
    raw = _CONTROL.sub("", raw)
    return raw.strip()


def reject_extreme_repetition(text: str) -> None:
    if _REPEAT.search(text):
        raise TextValidationError("Query contains extreme repetition.")


def reject_non_text_junk(text: str) -> None:
    if not text:
        raise TextValidationError("Query is empty.")
    printable = sum(1 for c in text if c.isprintable() or c.isspace())
    if printable / max(len(text), 1) < 0.85:
        raise TextValidationError("Query contains non-text characters.")


def validate_language(text: str) -> None:
    policy = load_security_policy()
    allowed = set(policy.get("allowed_languages") or ["en", "hinglish"])
    if "en" in allowed and "hinglish" in allowed:
        allowed_punct = set('.,!?\'"-:;@#%&()/+=[]₹$€£')
        disallowed_count = 0
        for ch in text:
            if ch.isdigit() or ch.isspace() or ch.isascii() and ch.isalpha():
                continue
            if "\u0900" <= ch <= "\u097f":
                continue
            if ch in allowed_punct:
                continue
            disallowed_count += 1
        ratio = disallowed_count / max(len(text), 1)
        if ratio > 0.15:
            raise TextValidationError("Query language is not supported.")
        return
    if "en" in allowed and not (_LATIN.search(text) or text.isascii()):
        if not _DEVANAGARI.search(text):
            raise TextValidationError("Query language is not supported.")


def validate_user_query(text: str, *, min_len: int, max_len: int) -> str:
    normalized = normalize_user_text(text)
    if len(normalized) < min_len:
        raise TextValidationError(f"Query must be at least {min_len} characters.")
    if len(normalized) > max_len:
        raise TextValidationError(f"Query must be at most {max_len} characters.")
    reject_non_text_junk(normalized)
    reject_extreme_repetition(normalized)
    validate_language(normalized)
    return normalized
