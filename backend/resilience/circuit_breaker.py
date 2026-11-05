from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from config import settings
from logger import logger


@dataclass
class CircuitState:
    failures: int = 0
    opened_until: float = 0.0


class CircuitBreaker:
    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._state = CircuitState()

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            if now < self._state.opened_until:
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state.failures = 0
            self._state.opened_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._state.failures += 1
            if self._state.failures >= settings.circuit_breaker_failure_threshold:
                self._state.opened_until = time.time() + settings.circuit_breaker_open_seconds
                logger.warning("Circuit open for provider=%s", self.name)


_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name)
    return _breakers[name]
