from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict

from config import settings
from logger import logger


class CostTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._daily_global: dict[str, float] = defaultdict(float)
        self._daily_user: dict[str, float] = defaultdict(float)
        self._alerted: set[str] = set()

    @staticmethod
    def _day_key() -> str:
        return time.strftime("%Y-%m-%d")

    @staticmethod
    def hash_user(user_id: str | None) -> str:
        raw = (user_id or "anonymous").encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def record(self, user_id: str | None, cost_usd: float) -> None:
        day = self._day_key()
        with self._lock:
            self._daily_global[day] += cost_usd
            user_key = f"{day}:{self.hash_user(user_id)}"
            self._daily_user[user_key] += cost_usd
            global_total = self._daily_global[day]
            user_total = self._daily_user[user_key]
        if global_total >= settings.cost_alert_global_usd_daily:
            key = f"global:{day}"
            if key not in self._alerted:
                self._alerted.add(key)
                logger.warning("Cost alert: global daily spend reached %.4f USD", global_total)
        if user_id and user_total >= settings.cost_alert_user_usd_daily:
            key = f"user:{user_key}"
            if key not in self._alerted:
                self._alerted.add(key)
                logger.warning("Cost alert: user daily spend reached %.4f USD", user_total)

    def snapshot(self) -> dict:
        day = self._day_key()
        with self._lock:
            return {
                "day": day,
                "global_usd": round(self._daily_global[day], 6),
                "users_tracked": sum(1 for k in self._daily_user if k.startswith(day)),
            }


cost_tracker = CostTracker()
