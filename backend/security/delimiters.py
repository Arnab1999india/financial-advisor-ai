"""Wrap untrusted payloads so delimiter-closing tags cannot break out."""

from __future__ import annotations

import re

_CONTEXT_OPEN = "<context_data>"
_CONTEXT_CLOSE = "</context_data>"
_TOOL_OPEN = "<tool_data>"
_TOOL_CLOSE = "</tool_data>"

_FAKE_TAG = re.compile(
    r"</?\s*(?:context_data|tool_data|system|assistant|user)\s*>",
    re.IGNORECASE,
)
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")


def sanitize_untrusted_text(text: str) -> str:
    cleaned = _ZERO_WIDTH.sub("", text or "")
    cleaned = _FAKE_TAG.sub("[tag_removed]", cleaned)
    return cleaned.replace(_CONTEXT_CLOSE, "[/context_removed]").replace(
        _CONTEXT_OPEN, "[context_removed]"
    ).replace(_TOOL_CLOSE, "[/tool_removed]").replace(_TOOL_OPEN, "[tool_removed]")


def wrap_context_data(text: str) -> str:
    body = sanitize_untrusted_text(text)
    return f"{_CONTEXT_OPEN}\n{body}\n{_CONTEXT_CLOSE}"


def wrap_tool_data(text: str) -> str:
    body = sanitize_untrusted_text(text)
    return f"{_TOOL_OPEN}\n{body}\n{_TOOL_CLOSE}"
