"""Load grounding policy (messages, web allow-list) from JSON."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from config import settings


def _policy_path() -> Path:
    return Path(__file__).resolve().parent.parent / settings.grounding_policy_file


@lru_cache(maxsize=1)
def load_grounding_policy() -> dict:
    with open(_policy_path(), encoding="utf-8") as fh:
        return json.load(fh)


def insufficient_evidence_message() -> str:
    return str(load_grounding_policy().get("insufficient_evidence_message", "")).strip()


def web_untrusted_preamble() -> str:
    return str(load_grounding_policy().get("web_untrusted_preamble", "")).strip()


def allowed_web_domains() -> list[str]:
    raw = load_grounding_policy().get("web_allowed_domains") or []
    return [str(d).lower().strip().lstrip(".") for d in raw if str(d).strip()]


def url_on_allowlist(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    if not host:
        return False
    for domain in allowed_web_domains():
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False
