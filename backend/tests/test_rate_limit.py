import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from rate_limit.limiter import RateLimiter


def test_sliding_window_blocks_after_limit():
    limiter = RateLimiter()
    settings.rate_limit_enabled = True
    ip = "test-ip-1"
    allowed = 0
    for _ in range(settings.rate_limit_rpm_per_ip + 2):
        decision = limiter.check_chat(ip=ip, user_id=None)
        if decision.allowed:
            allowed += 1
    assert allowed <= settings.rate_limit_rpm_per_ip


def test_inflight_concurrency_cap():
    limiter = RateLimiter()
    settings.rate_limit_enabled = True
    settings.rate_limit_max_inflight_per_user = 1
    first = limiter.acquire_inflight("user-a")
    second = limiter.acquire_inflight("user-a")
    assert first.allowed is True
    assert second.allowed is False
    limiter.release_inflight("user-a")
    third = limiter.acquire_inflight("user-a")
    assert third.allowed is True
    limiter.release_inflight("user-a")
