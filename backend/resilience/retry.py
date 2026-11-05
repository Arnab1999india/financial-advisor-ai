from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def is_retryable_http_status(status: int) -> bool:
    return status == 429 or status >= 500


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int,
    base_delay_seconds: float,
    retry_predicate: Callable[[Exception], bool],
) -> T:
    attempt = 0
    last_exc: Exception | None = None
    while attempt < max_attempts:
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            attempt += 1
            if attempt >= max_attempts or not retry_predicate(exc):
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            delay += random.uniform(0, base_delay_seconds)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def retryable_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if any(x in text for x in ("429", "503", "502", "504", "timeout", "timed out", "connection")):
        return True
    if any(x in text for x in ("400", "401", "403", "404", "invalid", "validation")):
        return False
    return False
