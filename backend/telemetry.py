"""
Live Token Telemetry Tracker
============================
Extracts Gemini / LangChain usage metadata from LLM responses, estimates
USD cost, and keeps a process-wide running total that FastAPI and Streamlit
can read without extra instrumentation.

Per-request events live on ``AgentState.token_calls`` and are rolled into
this module's singleton after each ``/chat`` invocation.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Iterable

from config import settings

_RECENT_LIMIT = 50


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_usage(response: Any) -> dict[str, int]:
    """Pull input/output/total token counts from a LangChain message.

    Handles both the standardised ``usage_metadata`` attribute and Gemini's
    ``response_metadata['usage_metadata']`` shape.
    """
    input_tokens = output_tokens = total_tokens = 0

    usage = getattr(response, "usage_metadata", None)
    if usage:
        if isinstance(usage, dict):
            input_tokens = _as_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
            output_tokens = _as_int(usage.get("output_tokens") or usage.get("completion_tokens"))
            total_tokens = _as_int(usage.get("total_tokens"))
        else:
            input_tokens = _as_int(getattr(usage, "input_tokens", 0))
            output_tokens = _as_int(getattr(usage, "output_tokens", 0))
            total_tokens = _as_int(getattr(usage, "total_tokens", 0))

    if not (input_tokens or output_tokens or total_tokens):
        meta = getattr(response, "response_metadata", None) or {}
        if isinstance(meta, dict):
            gemini_usage = meta.get("usage_metadata") or meta.get("token_usage") or {}
            if isinstance(gemini_usage, dict):
                input_tokens = _as_int(
                    gemini_usage.get("prompt_token_count")
                    or gemini_usage.get("prompt_tokens")
                    or gemini_usage.get("input_tokens")
                )
                output_tokens = _as_int(
                    gemini_usage.get("candidates_token_count")
                    or gemini_usage.get("completion_tokens")
                    or gemini_usage.get("output_tokens")
                )
                total_tokens = _as_int(
                    gemini_usage.get("total_token_count") or gemini_usage.get("total_tokens")
                )

    if not total_tokens:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _is_flash(model: str) -> bool:
    name = (model or "").lower()
    return "flash" in name or name in {"fallback", "static", "gatekeeper", ""}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from configured per-million-token prices."""
    if _is_flash(model):
        in_price = settings.flash_input_usd_per_million
        out_price = settings.flash_output_usd_per_million
    else:
        in_price = settings.pro_input_usd_per_million
        out_price = settings.pro_output_usd_per_million
    return (input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price


def make_token_event(node: str, model: str, response: Any) -> dict[str, Any]:
    """Build a single telemetry event from an LLM response object."""
    usage = extract_usage(response)
    cost = estimate_cost_usd(model, usage["input_tokens"], usage["output_tokens"])
    return {
        "node": node,
        "model": model,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "estimated_cost_usd": round(cost, 8),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def empty_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def _add_to_bucket(bucket: dict[str, Any], event: dict[str, Any]) -> None:
    bucket["calls"] += 1
    bucket["input_tokens"] += int(event.get("input_tokens") or 0)
    bucket["output_tokens"] += int(event.get("output_tokens") or 0)
    bucket["total_tokens"] += int(event.get("total_tokens") or 0)
    bucket["estimated_cost_usd"] = round(
        bucket["estimated_cost_usd"] + float(event.get("estimated_cost_usd") or 0.0),
        8,
    )


def summarise_calls(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Roll a list of token events into a per-request summary."""
    summary = empty_bucket()
    events: list[dict[str, Any]] = []
    for event in calls or []:
        _add_to_bucket(summary, event)
        events.append(event)
    summary["calls"] = len(events)
    summary["estimated_cost_usd"] = round(float(summary["estimated_cost_usd"]), 8)
    summary["by_node"] = {}
    by_node: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    for event in events:
        _add_to_bucket(by_node[event.get("node") or "unknown"], event)
    summary["by_node"] = dict(by_node)
    summary["events"] = events
    return summary


class TokenTelemetryTracker:
    """Thread-safe in-memory aggregator for process-wide LLM usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_unlocked()

    def _reset_unlocked(self) -> None:
        self.chat_requests = 0
        self.cache_hits = 0
        self.totals = empty_bucket()
        self.by_model: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
        self.by_node: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
        self.recent: deque[dict[str, Any]] = deque(maxlen=_RECENT_LIMIT)
        self.started_at = datetime.now(timezone.utc).isoformat()

    def record_request(self, calls: Iterable[dict[str, Any]], *, cache_hit: bool = False) -> dict[str, Any]:
        summary = summarise_calls(calls)
        with self._lock:
            self.chat_requests += 1
            if cache_hit:
                self.cache_hits += 1
            else:
                for event in summary["events"]:
                    _add_to_bucket(self.totals, event)
                    _add_to_bucket(self.by_model[event.get("model") or "unknown"], event)
                    _add_to_bucket(self.by_node[event.get("node") or "unknown"], event)
                    self.recent.appendleft(event)
        return summary

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started_at": self.started_at,
                "chat_requests": self.chat_requests,
                "cache_hits": self.cache_hits,
                "llm_calls": self.totals["calls"],
                "input_tokens": self.totals["input_tokens"],
                "output_tokens": self.totals["output_tokens"],
                "total_tokens": self.totals["total_tokens"],
                "estimated_cost_usd": round(self.totals["estimated_cost_usd"], 8),
                "by_model": {k: dict(v) for k, v in self.by_model.items()},
                "by_node": {k: dict(v) for k, v in self.by_node.items()},
                "recent_calls": list(self.recent),
            }

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._reset_unlocked()
        return self.snapshot()


tracker = TokenTelemetryTracker()
