from __future__ import annotations

import concurrent.futures
from typing import Callable, TypeVar

from config import settings
from logger import logger
from resilience.circuit_breaker import get_breaker
from resilience.retry import call_with_retry, retryable_llm_error

T = TypeVar("T")


class ProviderUnavailableError(RuntimeError):
    pass


def call_external(
    provider: str,
    fn: Callable[[], T],
    *,
    timeout_seconds: float | None = None,
    retry: bool = True,
) -> T:
    breaker = get_breaker(provider)
    if not breaker.allow():
        raise ProviderUnavailableError(f"{provider} circuit open")

    timeout = timeout_seconds or settings.external_default_timeout_seconds
    attempts = settings.external_retry_max_attempts if retry else 1

    def _run() -> T:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn)
            try:
                result = future.result(timeout=timeout)
                breaker.record_success()
                return result
            except concurrent.futures.TimeoutError as exc:
                breaker.record_failure()
                raise TimeoutError(f"{provider} timed out after {timeout}s") from exc
            except Exception:
                breaker.record_failure()
                raise

    try:
        if retry:
            return call_with_retry(
                _run,
                max_attempts=attempts,
                base_delay_seconds=settings.external_retry_base_seconds,
                retry_predicate=retryable_llm_error,
            )
        return _run()
    except ProviderUnavailableError:
        raise
    except Exception as exc:
        logger.warning("External call failed provider=%s error=%s", provider, type(exc).__name__)
        raise
