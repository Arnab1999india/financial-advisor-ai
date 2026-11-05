from __future__ import annotations

import threading
from typing import Optional

_lock = threading.Lock()
_vectors: dict[str, list[float]] = {}


def get_embedding(text: str) -> Optional[list[float]]:
    key = text.strip().lower()
    with _lock:
        return _vectors.get(key)


def put_embedding(text: str, vector: list[float]) -> None:
    key = text.strip().lower()
    with _lock:
        _vectors[key] = vector


def clear_embeddings() -> None:
    with _lock:
        _vectors.clear()
