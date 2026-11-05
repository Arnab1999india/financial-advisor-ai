from __future__ import annotations

import threading
import time
from typing import Any, Optional

from config import settings

_lock = threading.Lock()
_quotes: dict[str, tuple[float, dict[str, Any]]] = {}


def get_quote(key: str) -> Optional[dict[str, Any]]:
    now = time.time()
    with _lock:
        item = _quotes.get(key)
        if not item:
            return None
        expires, payload = item
        if expires < now:
            _quotes.pop(key, None)
            return None
        return payload


def put_quote(key: str, payload: dict[str, Any]) -> None:
    with _lock:
        _quotes[key] = (time.time() + settings.market_data_cache_ttl_seconds, payload)
