"""Lightweight BM25 over knowledge-base chunks for exact-term recall."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._docs = documents
        self._k1 = k1
        self._b = b
        self._doc_tokens = [_tokenize(d) for d in documents]
        self._doc_len = [len(t) or 1 for t in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / max(len(self._doc_len), 1)
        self._df: dict[str, int] = {}
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1
        self._nd = max(len(documents), 1)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self._docs:
            return []
        scores: list[tuple[int, float]] = []
        for idx, tokens in enumerate(self._doc_tokens):
            if not tokens:
                continue
            tf = Counter(tokens)
            dl = self._doc_len[idx]
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (self._nd - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                score += idf * (freq * (self._k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
