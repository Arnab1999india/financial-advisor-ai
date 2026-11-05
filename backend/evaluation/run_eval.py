#!/usr/bin/env python3
"""
Run offline evaluation against eval_set.json.

Usage (from backend/):
  python evaluation/run_eval.py
  python evaluation/run_eval.py --json report.json

Metrics (no Gemini key required for default checks):
  - refusal_accuracy: off-topic + injection cases classified as expected
  - groundedness_rate: in-KB queries with retrieval confidence above threshold
  - citation_accuracy: output fixture checks (synthetic structured answers)
  - avg_tokens_per_request: 0 offline (use /telemetry after live runs)
  - cache_hit_rate: 0 offline (use /info token_telemetry.cache_hits after live runs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from agents.domain_classifier import classify_domain, extract_finance_query  # noqa: E402
from agents.grounding_gate import assess_evidence_strength  # noqa: E402
from agents.injection_guard import injection_guard_node  # noqa: E402
from rag.retriever import FinancialRetriever  # noqa: E402
from security.injection_scan import scan_user_query  # noqa: E402
from validation.output_validator import validate_structured_output  # noqa: E402
from validation.structured_output import StructuredAnswer, StructuredCitation  # noqa: E402


@dataclass
class EvalReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    refusal_accuracy: float = 0.0
    groundedness_rate: float = 0.0
    citation_accuracy: float = 0.0
    avg_tokens_per_request: float = 0.0
    cache_hit_rate: float = 0.0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _load_qa() -> list[dict]:
    path = BACKEND / "sample_qa.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_cases() -> list[dict]:
    path = Path(__file__).with_name("eval_set.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def _retriever() -> FinancialRetriever:
    r = FinancialRetriever()
    r.initialize(_load_qa())
    return r


def _check_domain(case: dict, report: EvalReport) -> bool:
    expect = case.get("expect") or {}
    if "domain" not in expect:
        return True
    decision = classify_domain(case["query"], demo_mode=True)
    ok = decision.status == expect["domain"]
    if expect["domain"] == "authorized" and expect.get("finance_query_excludes"):
        fq = (decision.finance_query or extract_finance_query(case["query"])).lower()
        ok = ok and all(x not in fq for x in expect["finance_query_excludes"])
    if not ok:
        report.failures.append(f"{case['id']}: domain expected {expect['domain']} got {decision.status}")
    return ok


def _check_injection(case: dict, report: EvalReport) -> bool:
    expect = case.get("expect") or {}
    if not expect.get("injection_blocked"):
        return True
    finding = scan_user_query(case["query"])
    ok = finding.blocked
    if not ok:
        report.failures.append(f"{case['id']}: expected injection block")
    return ok


def _check_grounding(case: dict, retriever: FinancialRetriever, report: EvalReport) -> bool:
    expect = case.get("expect") or {}
    ctype = case.get("type", "")
    if ctype not in ("finance_in_kb", "finance_hinglish", "finance_borderline", "finance_complex", "no_kb_answer"):
        return True
    hits = retriever.retrieve(case["query"], k=4)
    top_conf = max((h.get("confidence") or 0.0) for h in hits) if hits else 0.0
    ok = True
    if expect.get("min_top_confidence") is not None:
        ok = top_conf >= float(expect["min_top_confidence"])
    if expect.get("max_top_confidence") is not None:
        ok = ok and top_conf <= float(expect["max_top_confidence"])
    if expect.get("answer_must_include") and hits:
        blob = (hits[0].get("answer") or "").lower()
        ok = ok and all(tok.lower() in blob for tok in expect["answer_must_include"])
    if expect.get("exact_number_in_source") and hits:
        num = expect["exact_number_in_source"]
        corpus = " ".join((h.get("answer") or "") for h in hits)
        ok = ok and num in corpus.replace(",", "")
    strength = assess_evidence_strength({"retrieved_docs": hits})
    if expect.get("grounding") == "none_or_low" and hits and "max_top_confidence" not in expect:
        ok = ok and strength in ("none", "partial")
    if not ok:
        report.failures.append(
            f"{case['id']}: grounding conf={top_conf:.3f} strength={strength} hits={len(hits)}"
        )
    return ok


def _check_output_fixtures(case: dict, report: EvalReport) -> bool:
    if case.get("type") != "output_fixture":
        return True
    expect = case["expect"]
    docs = [{"source_file": "sample_qa.json", "page": 1, "answer": "An ETF is an exchange-traded fund."}]
    if expect.get("reason_contains") == "untracked_number":
        structured = StructuredAnswer(
            answer="The value becomes 9999999 dollars after ten years.",
            citations=[StructuredCitation(source="sample_qa.json", page=1)],
            confidence="high",
            used_market_data=False,
        )
    elif expect.get("reason_contains") == "citation_unknown":
        structured = StructuredAnswer(
            answer="ETFs trade on exchanges.",
            citations=[StructuredCitation(source="fake_source.pdf", page=99)],
            confidence="high",
            used_market_data=False,
        )
    else:
        structured = StructuredAnswer(
            answer="Here is the gemini_api key.",
            citations=[],
            confidence="low",
            used_market_data=False,
        )
    result = validate_structured_output(structured, docs=docs, context="", used_market_data=False)
    want_ok = expect.get("validation_ok", False)
    ok = result.ok == want_ok
    if not want_ok:
        ok = ok and any(expect["reason_contains"] in r for r in result.reasons)
    if not ok:
        report.failures.append(f"{case['id']}: output validation reasons={result.reasons}")
    return ok


def _check_cost(case: dict, report: EvalReport) -> bool:
    expect = case.get("expect") or {}
    if expect.get("max_llm_calls") != 0:
        return True
    state = {
        "query": case["query"],
        "messages": [],
        "user_id": None,
        "token_calls": [],
    }
    inj = injection_guard_node(state)
    if inj.get("injection_blocked"):
        ok = not inj.get("token_calls")
    else:
        decision = classify_domain(case["query"], demo_mode=True)
        ok = decision.status == expect.get("domain", decision.status) and not decision.token_calls
    if not ok:
        report.failures.append(f"{case['id']}: expected zero LLM token_calls on early exit")
    return ok


def run_eval() -> EvalReport:
    report = EvalReport()
    retriever = _retriever()
    cases = _load_cases()

    refusal_total = refusal_ok = 0
    ground_total = ground_ok = 0
    cite_total = cite_ok = 0

    for case in cases:
        report.total += 1
        ctype = case.get("type", "")
        ok = True
        if "query" in case:
            ok = _check_domain(case, report) and ok
            ok = _check_injection(case, report) and ok
            ok = _check_grounding(case, retriever, report) and ok
            ok = _check_cost(case, report) and ok
        ok = _check_output_fixtures(case, report) and ok

        if ctype in ("off_topic", "injection", "injection_hinglish", "injection_encoded", "cost"):
            if ctype.startswith("injection") or case.get("expect", {}).get("injection_blocked"):
                refusal_total += 1
                if scan_user_query(case.get("query", "")).blocked:
                    refusal_ok += 1
            elif case.get("expect", {}).get("domain") == "unauthorized":
                refusal_total += 1
                if classify_domain(case["query"], demo_mode=True).status == "unauthorized":
                    refusal_ok += 1

        if ctype in ("finance_in_kb", "finance_hinglish", "finance_borderline", "finance_complex"):
            ground_total += 1
            hits = retriever.retrieve(case["query"], k=4)
            top = max((h.get("confidence") or 0.0) for h in hits) if hits else 0.0
            if top >= 0.35:
                ground_ok += 1

        if ctype == "output_fixture":
            cite_total += 1
            if _check_output_fixtures(case, EvalReport()):
                cite_ok += 1

        if ok:
            report.passed += 1
        else:
            report.failed += 1

    report.refusal_accuracy = (refusal_ok / refusal_total) if refusal_total else 1.0
    report.groundedness_rate = (ground_ok / ground_total) if ground_total else 1.0
    report.citation_accuracy = (cite_ok / cite_total) if cite_total else 1.0
    report.avg_tokens_per_request = 0.0
    report.cache_hit_rate = 0.0
    report.notes.append(
        "avg_tokens_per_request and cache_hit_rate require a live API run; "
        "read GET /telemetry and token_telemetry.cache_hits on /info after exercising /chat."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation runner")
    parser.add_argument("--json", type=str, help="Write JSON report to this path")
    args = parser.parse_args()
    report = run_eval()
    summary = {
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "refusal_accuracy": round(report.refusal_accuracy, 4),
        "groundedness_rate": round(report.groundedness_rate, 4),
        "citation_accuracy": round(report.citation_accuracy, 4),
        "avg_tokens_per_request": report.avg_tokens_per_request,
        "cache_hit_rate": report.cache_hit_rate,
        "failures": report.failures,
        "notes": report.notes,
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
