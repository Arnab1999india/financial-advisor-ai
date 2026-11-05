from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

from config import settings

_lock = threading.Lock()
_exact: dict[str, tuple[float, Any]] = {}
_semantic: list[tuple[list[float], str, float, Any]] = {}
_index_epoch: str = ""


def set_index_epoch(epoch: str) -> None:
    global _index_epoch
    with _lock:
        _index_epoch = epoch
        _exact.clear()
        _semantic.clear()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def get_exact(key: str) -> Optional[Any]:
    now = time.time()
    with _lock:
        item = _exact.get(key)
        if not item:
            return None
        expires, value = item
        if expires < now:
            _exact.pop(key, None)
            return None
        return value


def put_exact(key: str, value: Any, ttl_seconds: int) -> None:
    with _lock:
        if len(_exact) >= settings.response_cache_size:
            _exact.pop(next(iter(_exact)))
        _exact[key] = (time.time() + ttl_seconds, value)


def get_semantic(query_vec: list[float]) -> Optional[Any]:
    now = time.time()
    best = 0.0
    best_val = None
    with _lock:
        for vec, _text, expires, value in _semantic:
            if expires < now:
                continue
            score = _cosine(query_vec, vec)
            if score > best:
                best = score
                best_val = value
    if best >= settings.semantic_cache_similarity_threshold:
        return best_val
    return None


def put_semantic(query_vec: list[float], query_text: str, value: Any, ttl_seconds: int) -> None:
    with _lock:
        if len(_semantic) >= settings.semantic_cache_max_entries:
            _semantic.pop(0)
        _semantic.append((query_vec, query_text, time.time() + ttl_seconds, value))
