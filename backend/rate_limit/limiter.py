"""Sliding-window limits with optional Redis and in-memory fallback."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Literal

from config import settings
from logger import logger
from security.abuse_tracker import abuse_tracker

LimitKind = Literal[
    "rpm_ip",
    "rpm_user",
    "rph_ip",
    "rph_user",
    "daily_tokens_user",
    "upload_rpm",
    "web_rpm",
    "inflight_user",
    "global_inflight",
    "abuse_cooldown",
]

_REDIS = None
_REDIS_TRIED = False


def _redis_client():
    global _REDIS, _REDIS_TRIED
    if _REDIS_TRIED:
        return _REDIS
    _REDIS_TRIED = True
    url = (settings.rate_limit_redis_url or "").strip()
    if not url:
        return None
    try:
        import redis  # type: ignore

        _REDIS = redis.Redis.from_url(url, decode_responses=True)
        _REDIS.ping()
        logger.info("Rate limiter using Redis backend.")
    except Exception as exc:
        logger.warning("Redis unavailable for rate limiting (%s); using in-memory.", exc)
        _REDIS = None
    return _REDIS


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0
    limit_hit: str = ""


class _MemoryWindow:
    def __init__(self) -> None:
        self._events: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def count(self, key: str, window_seconds: int, now: float) -> int:
        with self._lock:
            dq = self._events[key]
            cutoff = now - window_seconds
            while dq and dq[0] < cutoff:
                dq.popleft()
            return len(dq)

    def add(self, key: str, now: float) -> None:
        with self._lock:
            self._events[key].append(now)


class RateLimiter:
    def __init__(self) -> None:
        self._memory = _MemoryWindow()
        self._user_inflight: dict[str, int] = defaultdict(int)
        self._global_inflight = 0
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {}

    def _sliding_allow(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        now = time.time()
        r = _redis_client()
        if r is not None:
            try:
                bucket = f"rl:{key}:{window_seconds}"
                pipe = r.pipeline()
                pipe.zremrangebyscore(bucket, 0, now - window_seconds)
                pipe.zadd(bucket, {str(now): now})
                pipe.zcard(bucket)
                pipe.expire(bucket, window_seconds + 5)
                _, _, count, _ = pipe.execute()
                if int(count) > limit:
                    return RateLimitDecision(False, window_seconds, f"window:{window_seconds}")
                return RateLimitDecision(True)
            except Exception as exc:
                logger.warning("Redis rate limit failed (%s); falling back to memory.", exc)

        count = self._memory.count(key, window_seconds, now)
        if count >= limit:
            return RateLimitDecision(False, window_seconds, f"window:{window_seconds}")
        self._memory.add(key, now)
        return RateLimitDecision(True)

    def check_chat(self, *, ip: str, user_id: str | None) -> RateLimitDecision:
        if not settings.rate_limit_enabled:
            return RateLimitDecision(True)
        subject = (user_id or ip or "anonymous").strip()
        if abuse_tracker.in_cooldown(subject):
            return RateLimitDecision(False, abuse_tracker.cooldown_remaining(subject), "abuse_cooldown")

        checks = [
            ("rpm_ip", f"ip:{ip}", settings.rate_limit_rpm_per_ip, 60),
            ("rpm_user", f"user:{subject}", settings.rate_limit_rpm_per_user, 60),
            ("rph_ip", f"ip:{ip}:h", settings.rate_limit_rph_per_ip, 3600),
            ("rph_user", f"user:{subject}:h", settings.rate_limit_rph_per_user, 3600),
        ]
        for name, key, limit, window in checks:
            decision = self._sliding_allow(key, limit, window)
            if not decision.allowed:
                decision.limit_hit = name
                return decision
        return RateLimitDecision(True)

    def check_daily_tokens(self, user_id: str | None, tokens: int) -> RateLimitDecision:
        if not settings.rate_limit_enabled or not user_id:
            return RateLimitDecision(True)
        key = f"tokens:{user_id}:{time.strftime('%Y-%m-%d')}"
        now = time.time()
        r = _redis_client()
        if r is not None:
            try:
                total = int(r.incrby(key, tokens))
                r.expire(key, 86_400)
                if total > settings.rate_limit_daily_tokens_per_user:
                    return RateLimitDecision(False, 3600, "daily_tokens_user")
                return RateLimitDecision(True)
            except Exception:
                pass
        # memory fallback: approximate using event weights
        count_key = f"{key}:mem"
        current = self._memory.count(count_key, 86_400, now)
        if current + tokens > settings.rate_limit_daily_tokens_per_user:
            return RateLimitDecision(False, 3600, "daily_tokens_user")
        for _ in range(tokens):
            self._memory.add(count_key, now)
        return RateLimitDecision(True)

    def check_upload(self, ip: str) -> RateLimitDecision:
        if not settings.rate_limit_enabled:
            return RateLimitDecision(True)
        decision = self._sliding_allow(f"upload:{ip}", settings.rate_limit_upload_rpm, 60)
        if not decision.allowed:
            decision.limit_hit = "upload_rpm"
        return decision

    def record_web_search(self, subject: str) -> RateLimitDecision:
        if not settings.rate_limit_enabled:
            return RateLimitDecision(True)
        decision = self._sliding_allow(f"web:{subject}", settings.rate_limit_web_rpm, 60)
        if not decision.allowed:
            decision.limit_hit = "web_rpm"
        return decision

    def acquire_inflight(self, user_id: str | None) -> RateLimitDecision:
        if not settings.rate_limit_enabled:
            return RateLimitDecision(True)
        subject = (user_id or "anonymous").strip()
        with self._lock:
            if self._global_inflight >= settings.rate_limit_global_inflight:
                return RateLimitDecision(False, 5, "global_inflight")
            if self._user_inflight[subject] >= settings.rate_limit_max_inflight_per_user:
                return RateLimitDecision(False, 5, "inflight_user")
            self._global_inflight += 1
            self._user_inflight[subject] += 1
        return RateLimitDecision(True)

    def release_inflight(self, user_id: str | None) -> None:
        subject = (user_id or "anonymous").strip()
        with self._lock:
            self._global_inflight = max(0, self._global_inflight - 1)
            self._user_inflight[subject] = max(0, self._user_inflight[subject] - 1)


rate_limiter = RateLimiter()
