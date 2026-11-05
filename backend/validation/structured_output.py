"""Structured LLM answer schema (Phase 5)."""

from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ConfidenceLevel = Literal["high", "medium", "low"]

RELIABLE_ANSWER_FALLBACK = (
    "I couldn't produce a reliable answer from the available sources."
)


class StructuredCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1)
    page: Optional[int] = None

    @field_validator("page")
    @classmethod
    def page_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError("page must be >= 1")
        return value


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1)
    citations: list[StructuredCitation] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"
    used_market_data: bool = False


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_structured_answer(raw: str) -> StructuredAnswer:
    text = (raw or "").strip()
    text = _JSON_FENCE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    payload = json.loads(text[start : end + 1])
    return StructuredAnswer.model_validate(payload)
