"""
Human-in-the-loop approval
==========================
Pauses the LangGraph when:

* the user asks to simulate an automated trade or price alert, or
* generation + retrieval look like an actionable high-probability setup.

``interrupt()`` yields an approval card.  The graph resumes only after
FastAPI receives an explicit approve / reject (with optional param edits).
"""

from __future__ import annotations

import re
from typing import Any

from langgraph.types import interrupt

from agents.state import AgentState
from config import settings
from logger import logger

_TICKER = re.compile(r"\$([A-Za-z]{1,5})\b|(?<![A-Za-z])([A-Z]{2,5})(?![A-Za-z])")
_QTY = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:shares?|qty|units?)?\b", re.I)
_PRICE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)", re.I)

_TICKER_STOP = {
    "THE", "AND", "FOR", "BUY", "SELL", "LONG", "SHORT", "CALL", "PUT",
    "ETF", "USD", "CEO", "IPO", "NAV", "PE", "EPS", "ROI", "API", "PDF",
    "WHAT", "WHEN", "WITH", "FROM", "THIS", "THAT", "HAVE", "WILL",
}

_SIM_REQUEST = re.compile(
    r"\b(?:simulat(?:e|ion)|paper[\s-]?trad(?:e|ing)|auto(?:mated)?[\s-]?(?:trade|alert)"
    r"|place (?:a |an )?(?:paper )?order|set (?:a |an )?(?:price )?alert"
    r"|price alert|notify me|alert me|execute (?:a )?(?:paper )?trade)\b",
    re.I,
)

_ACTIONABLE_SETUP = re.compile(
    r"high[- ]probability|strong (?:buy|sell)|actionable setup|"
    r"enter (?:long|short)|this (?:looks like|is) a (?:buy|sell) setup|"
    r"confluence (?:buy|sell)|high[- ]conviction",
    re.I,
)

_DIRECTIONAL = re.compile(r"\b(?:buy|sell|long|short)\b", re.I)


def is_simulation_request(text: str) -> bool:
    return bool(_SIM_REQUEST.search(text or ""))


def is_high_probability_setup(state: AgentState) -> bool:
    """True when the turn describes an actionable setup above the confidence floor."""
    query = state.get("query") or ""
    answer = state.get("answer") or ""
    blob = f"{query}\n{answer}"
    if not _ACTIONABLE_SETUP.search(blob) or not _DIRECTIONAL.search(blob):
        return False

    scores: list[float] = []
    conf = state.get("confidence")
    if conf is not None:
        scores.append(float(conf))
    for doc in state.get("retrieved_docs") or []:
        try:
            scores.append(float(doc.get("confidence") or 0.0))
        except (TypeError, ValueError):
            continue
    if not scores:
        return "high" in blob.lower() and "probability" in blob.lower()
    return max(scores) >= float(settings.high_probability_threshold)


def extract_symbol(text: str) -> str:
    for match in _TICKER.finditer(text or ""):
        token = (match.group(1) or match.group(2) or "").upper()
        if token and token not in _TICKER_STOP:
            return token
    return "DEMO"


def build_action_proposal(state: AgentState) -> dict[str, Any]:
    query = state.get("query") or ""
    answer = state.get("answer") or ""
    blob = f"{query}\n{answer}"
    lower = blob.lower()

    action_type = "price_alert" if re.search(r"\balert|notify me\b", lower) else "paper_trade"
    side = "SELL" if re.search(r"\b(sell|short)\b", lower) and not re.search(r"\b(buy|long)\b", lower) else "BUY"
    qty_match = _QTY.search(query)
    quantity = float(qty_match.group(1)) if qty_match else 1.0

    prices = [float(p) for p in _PRICE.findall(query)]
    limit_price = prices[0] if prices else None
    stop_loss = prices[1] if len(prices) > 1 else None
    take_profit = prices[2] if len(prices) > 2 else None

    docs = state.get("retrieved_docs") or []
    doc_scores = [float(d.get("confidence") or 0.0) for d in docs]
    probability = float(state.get("confidence") or (max(doc_scores) if doc_scores else 0.0))

    rationale = (answer or query).strip()
    if len(rationale) > 400:
        rationale = rationale[:397] + "..."

    return {
        "action_type": action_type,
        "symbol": extract_symbol(blob),
        "side": side,
        "quantity": quantity,
        "order_type": "limit" if limit_price is not None else "market",
        "limit_price": limit_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "alert_trigger": f"price crosses last close" if action_type == "price_alert" else None,
        "probability": round(probability, 4),
        "rationale": rationale,
    }


def setup_gate_node(state: AgentState) -> dict:
    """Decide whether Q&A output should pause for a human approval card."""
    if is_high_probability_setup(state) or is_simulation_request(state.get("query") or ""):
        proposal = build_action_proposal(state)
        logger.info(
            "HITL gate: pausing for %s %s %s (p=%.2f).",
            proposal["action_type"],
            proposal["side"],
            proposal["symbol"],
            proposal["probability"],
        )
        reason = (
            "user_requested_simulation"
            if is_simulation_request(state.get("query") or "")
            else "high_probability_setup"
        )
        return {
            "approval_required": True,
            "approval_reason": reason,
            "pending_action": proposal,
        }
    return {"approval_required": False}


def dispatch_approval(state: AgentState) -> str:
    if state.get("approval_required"):
        return "human_approval"
    return "memory_save"


def human_approval_node(state: AgentState) -> dict:
    """Pause the graph until the UI posts an approve/reject decision."""
    proposal = state.get("pending_action") or build_action_proposal(state)
    card = {
        "reason": state.get("approval_reason") or "user_requested_simulation",
        "draft_answer": state.get("answer") or "",
        "proposal": proposal,
    }
    logger.info("HITL interrupt: waiting for approval of %s.", proposal.get("symbol"))
    decision = interrupt(card)
    if not isinstance(decision, dict):
        decision = {"approved": bool(decision), "decision": "approve" if decision else "reject"}

    params = dict(proposal)
    incoming = decision.get("params") or {}
    if isinstance(incoming, dict):
        for key, value in incoming.items():
            if value is not None and value != "":
                params[key] = value

    approved = bool(decision.get("approved"))
    if str(decision.get("decision") or "").lower() in {"approve", "approved", "edit", "edit_approve"}:
        approved = True
    if str(decision.get("decision") or "").lower() in {"reject", "rejected", "deny"}:
        approved = False

    return {
        "pending_action": params,
        "approval_required": True,
        "approval_decision": {
            "approved": approved,
            "decision": "approve" if approved else "reject",
            "notes": str(decision.get("notes") or ""),
            "params": params,
        },
    }


def execute_simulation_node(state: AgentState) -> dict:
    """Apply the approved (or rejected) paper-trade / alert simulation."""
    decision = state.get("approval_decision") or {}
    params = decision.get("params") or state.get("pending_action") or {}
    prior = (state.get("answer") or "").strip()
    approved = bool(decision.get("approved"))

    if not approved:
        note = (
            "Human approval **rejected** this simulated trade/alert. "
            "No paper order or alert was recorded."
        )
        combined = f"{prior}\n\n---\n{note}".strip() if prior else note
        return {"answer": combined, "model_used": state.get("model_used") or "simulation"}

    action = str(params.get("action_type") or "paper_trade")
    symbol = str(params.get("symbol") or "DEMO")
    side = str(params.get("side") or "BUY")
    quantity = params.get("quantity") or 1
    order_type = str(params.get("order_type") or "market")
    probability = float(params.get("probability") or 0.0)
    notes = str(decision.get("notes") or "").strip()

    if action == "price_alert":
        trigger = params.get("alert_trigger") or "price crosses last close"
        body = (
            f"**Alert simulation recorded (not a live notification).**\n\n"
            f"- Symbol: `{symbol}`\n"
            f"- Side bias: `{side}`\n"
            f"- Trigger: {trigger}\n"
            f"- Stated probability: {probability:.0%}\n"
            f"- Status: `ARMED` (simulated)\n"
        )
    else:
        body = (
            f"**Paper-trade simulation recorded (not a live broker order).**\n\n"
            f"- Symbol: `{symbol}`\n"
            f"- Side: `{side}`\n"
            f"- Quantity: `{quantity}`\n"
            f"- Order type: `{order_type}`\n"
            f"- Limit: {params.get('limit_price') or '—'}\n"
            f"- Stop loss: {params.get('stop_loss') or '—'}\n"
            f"- Take profit: {params.get('take_profit') or '—'}\n"
            f"- Stated probability: {probability:.0%}\n"
            f"- Status: `FILLED` (simulated)\n"
        )
    if notes:
        body += f"\nOperator notes: {notes}\n"
    body += (
        "\nThis is an educational simulation only. It does not transmit orders "
        "to any exchange or broker."
    )
    combined = f"{prior}\n\n---\n{body}".strip() if prior else body
    logger.info("HITL simulation executed: %s %s %s.", action, side, symbol)
    return {
        "answer": combined,
        "pending_action": params,
        "model_used": state.get("model_used") or "simulation",
    }
