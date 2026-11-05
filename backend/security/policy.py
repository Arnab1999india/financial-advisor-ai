from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from config import settings


def _path() -> Path:
    return Path(__file__).resolve().parent.parent / settings.security_policy_file


@lru_cache(maxsize=1)
def load_security_policy() -> dict:
    with open(_path(), encoding="utf-8") as fh:
        return json.load(fh)


def injection_refusal_message() -> str:
    return str(load_security_policy().get("injection_refusal_message", "")).strip()


def compiled_patterns(key: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in load_security_policy().get(key, []))


def output_link_allow_domains() -> list[str]:
    return [str(d).lower() for d in load_security_policy().get("output_link_allow_domains", [])]
