"""Detect and mask common Indian financial PII before logs / LLM."""

from __future__ import annotations

import re

_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
_AADHAAR = re.compile(r"\b(?:\d{4}\s?){3}\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_BANK_ACCT = re.compile(r"\b\d{9,18}\b")
_OTP = re.compile(r"\b(?:otp|one[- ]time password)[:\s]*\d{4,8}\b", re.IGNORECASE)
_PHONE_IN = re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")


def mask_pii(text: str) -> str:
    if not text:
        return text
    masked = _OTP.sub("otp [REDACTED]", text)
    masked = _PAN.sub("[PAN_REDACTED]", masked)
    masked = _AADHAAR.sub("[AADHAAR_REDACTED]", masked)
    masked = _CARD.sub("[CARD_REDACTED]", masked)
    masked = _PHONE_IN.sub("[PHONE_REDACTED]", masked)
    masked = _BANK_ACCT.sub("[ACCOUNT_REDACTED]", masked)
    return masked
