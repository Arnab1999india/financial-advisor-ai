"""Block markdown links/images to non-allow-listed domains in model output."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from rag.grounding_policy import url_on_allowlist
from security.policy import output_link_allow_domains

_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_BARE_URL = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)


def _host_allowed(url: str) -> bool:
    if url_on_allowlist(url):
        return True
    host = (urlparse(url).netloc or "").lower()
    for domain in output_link_allow_domains():
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False


def filter_untrusted_links(text: str) -> str:
    if not text:
        return text

    def repl_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2).strip()
        return match.group(0) if _host_allowed(url) else label or "[link removed]"

    def repl_image(match: re.Match[str]) -> str:
        url = match.group(2).strip()
        return match.group(0) if _host_allowed(url) else "[image removed]"

    out = _MD_IMAGE.sub(repl_image, text)
    out = _MD_LINK.sub(repl_link, out)

    def repl_bare(match: re.Match[str]) -> str:
        url = match.group(0)
        return url if _host_allowed(url) else "[link removed]"

    out = _BARE_URL.sub(repl_bare, out)
    return out
