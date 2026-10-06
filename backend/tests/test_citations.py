import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.citation_map import citation_map_node
from rag.citations import map_answer_to_chunks, source_chunks_payload, split_sentences


def test_split_sentences_keeps_order():
    parts = split_sentences("PE is price over earnings. It is a valuation ratio.")
    assert parts == ["PE is price over earnings.", "It is a valuation ratio."]


def test_map_answer_to_chunks_links_matching_sentence():
    chunks = [
        {
            "chunk_id": "kb-1",
            "question": "What is P/E ratio?",
            "answer": "The P/E ratio is price divided by earnings per share. It is a valuation metric.",
            "confidence": 0.81,
            "source_timestamp": "2026-08-23T00:00:00Z",
            "source_type": "faiss",
        }
    ]
    answer = "The P/E ratio is price divided by earnings per share. Diversification can reduce risk."
    mappings = map_answer_to_chunks(answer, chunks)
    assert len(mappings) == 1
    assert mappings[0]["chunk_id"] == "kb-1"
    assert mappings[0]["answer_start"] == 0
    assert mappings[0]["confidence"] >= 0.28
    assert "price divided by earnings" in mappings[0]["source_sentence"]


def test_map_skips_unrelated_sentences():
    chunks = [
        {
            "chunk_id": "kb-2",
            "question": "NAV",
            "answer": "Net asset value is fund assets minus liabilities.",
            "source_type": "faiss",
        }
    ]
    mappings = map_answer_to_chunks("Bananas are a yellow fruit.", chunks)
    assert mappings == []


def test_source_chunks_payload_normalises_web_hit():
    payload = source_chunks_payload(
        [
            {
                "chunk_id": "web-0",
                "title": "SEC filing",
                "snippet": "A 10-K is an annual report.",
                "confidence": 0.9,
                "retrieved_at": "2026-08-23T10:00:00Z",
                "source_timestamp": "2026-01-01T00:00:00Z",
                "url": "https://example.com/10k",
                "source_type": "web",
                "category": "web",
            }
        ]
    )
    assert payload[0]["source_url"] == "https://example.com/10k"
    assert payload[0]["title"] == "SEC filing"


def test_citation_map_node_uses_final_answer():
    state = {
        "answer": "A dividend is a distribution of company profits to shareholders.",
        "retrieved_docs": [
            {
                "chunk_id": "kb-div",
                "question": "What is a dividend?",
                "answer": "A dividend is a distribution of company profits to shareholders.",
                "confidence": 0.77,
                "retrieved_at": "2026-08-23T10:00:00Z",
                "source_timestamp": "2025-12-01T00:00:00Z",
                "source_type": "faiss",
            }
        ],
    }
    out = citation_map_node(state)
    assert out["source_chunks"][0]["chunk_id"] == "kb-div"
    assert out["sentence_mappings"][0]["chunk_id"] == "kb-div"
    assert out["sentence_mappings"][0]["source_timestamp"] == "2025-12-01T00:00:00Z"
