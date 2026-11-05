from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from logger import logger


def _hash_user(user_id: str | None) -> str:
    raw = (user_id or "anonymous").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def log_event(
    *,
    request_id: str,
    stage: str,
    user_id: str | None = None,
    latency_ms: float | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cache_hit: bool | None = None,
    guardrail: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "request_id": request_id,
        "stage": stage,
        "user_hash": _hash_user(user_id),
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cache_hit": cache_hit,
        "guardrail": guardrail,
    }
    if extra:
        payload.update(extra)
    logger.info("structured %s", json.dumps({k: v for k, v in payload.items() if v is not None}))


class StageTimer:
    def __init__(self, request_id: str, stage: str, user_id: str | None = None) -> None:
        self.request_id = request_id
        self.stage = stage
        self.user_id = user_id
        self._start = time.perf_counter()

    def finish(self, **kwargs: Any) -> None:
        latency_ms = (time.perf_counter() - self._start) * 1000
        log_event(
            request_id=self.request_id,
            stage=self.stage,
            user_id=self.user_id,
            latency_ms=latency_ms,
            **kwargs,
        )
