"""In-memory abuse scores per user / client id."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from config import settings
from logger import logger
from security.policy import load_security_policy


class AbuseTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scores: dict[str, int] = defaultdict(int)
        self._cooldown_until: dict[str, float] = {}

    def record(self, subject: str, reason_code: str) -> int:
        key = (subject or "anonymous").strip() or "anonymous"
        with self._lock:
            self._scores[key] += 1
            score = self._scores[key]
            threshold = int(
                load_security_policy().get("abuse_score_threshold", settings.abuse_score_threshold)
            )
            if score >= threshold:
                self._cooldown_until[key] = time.time() + settings.abuse_cooldown_seconds
        logger.warning("Abuse event reason=%s subject=%s score=%d", reason_code, key, score)
        return score

    def score(self, subject: str) -> int:
        key = (subject or "anonymous").strip() or "anonymous"
        with self._lock:
            return self._scores[key]

    def over_threshold(self, subject: str) -> bool:
        threshold = int(load_security_policy().get("abuse_score_threshold", settings.abuse_score_threshold))
        return self.score(subject) >= threshold

    def in_cooldown(self, subject: str) -> bool:
        key = (subject or "anonymous").strip() or "anonymous"
        with self._lock:
            until = self._cooldown_until.get(key, 0.0)
        return time.time() < until

    def cooldown_remaining(self, subject: str) -> int:
        key = (subject or "anonymous").strip() or "anonymous"
        with self._lock:
            until = self._cooldown_until.get(key, 0.0)
        return max(0, int(until - time.time()))


abuse_tracker = AbuseTracker()
