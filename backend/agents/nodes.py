"""
LangGraph Node Functions
========================
Each function here represents one node in the StateGraph.  Nodes receive the
current ``AgentState``, perform their work, and return a *partial* state dict
that LangGraph merges into the global state.

Node catalogue
--------------
summarize_history_node — Prune messages > threshold; rolling summary via Flash
gatekeeper_node        — Domain guardrail: authorize finance queries or reject
out_of_scope_node      — Static refusal for non-finance queries
router_node            — Classify intent: financial_qa | chitchat
rag_retrieve_node      — FAISS MMR retrieval for relevant Q&A context
grade_documents_node   — Flash grader: do FAISS docs actually answer the query? (retrieval_loop.py)
query_rewrite_node     — Rephrase the question and retry FAISS (retrieval_loop.py)
web_search_node        — Tavily/DuckDuckGo fallback when FAISS is not relevant (retrieval_loop.py)
gemini_generate_node   — Tier-aware generation: Flash when RAG-grounded, Pro otherwise
outbound_guardrail_node — Post-generation intercept of direct financial advice (see outbound_guardrail.py)
chitchat_node          — Fast static replies for greetings / small-talk
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from agents.state import AgentState
from config import settings
from logger import logger
from memory.cognitive_store import memory_store
from rag.retriever import retriever as financial_retriever
from telemetry import make_token_event
from tools.market_data import get_market_quote, is_market_data_query

# ---------------------------------------------------------------------------
# Module-level keyword sets — used for free pre-classification in gatekeeper
# and router so Flash/Pro calls are skipped for obvious cases.
# ---------------------------------------------------------------------------

_FINANCE_KEYWORDS = {
    "stock", "stocks", "share", "shares", "market", "invest", "investment",
    "portfolio", "trading", "trade", "fund", "etf", "bond", "equity",
    "dividend", "profit", "loss", "economy", "inflation", "interest",
    "rate", "currency", "forex", "crypto", "bitcoin", "financial",
    "finance", "revenue", "earnings", "index", "nasdaq", "s&p", "dow",
    "rsi", "macd", "sma", "ema", "bollinger", "vwap", "atr", "candlestick",
    "mutual", "nifty", "sensex", "option", "futures", "derivative",
    "rebalancing", "diversification", "nav", "yield", "liquidity",
    "simulate", "simulation", "alert", "paper",
}

_CHITCHAT_TRIGGERS = {
    "hello", "hi", "hey", "howdy",
    "thanks", "thank you", "thx",
    "bye", "goodbye", "see you",
    "how are you",
}

# ---------------------------------------------------------------------------
# Lazy LLM singletons — created once, reused across all node invocations.
# Previously _build_cheap_llm() / _build_power_llm() created a new instance
# on every call, wasting allocation overhead.
# ---------------------------------------------------------------------------

_cheap_llm = None
_power_llm = None


def _get_cheap_llm():
    """Return the shared Flash model instance, initialising it on first use."""
    global _cheap_llm
    if _cheap_llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        _cheap_llm = ChatGoogleGenerativeAI(
            model=settings.cheap_model,
            temperature=0.0,
            google_api_key=settings.gemini_api_key,
        )
    return _cheap_llm


def _get_power_llm():
    """Return the shared Pro model instance, initialising it on first use."""
    global _power_llm
    if _power_llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        _power_llm = ChatGoogleGenerativeAI(
            model=settings.power_model,
            temperature=settings.gemini_temperature,
            google_api_key=settings.gemini_api_key,
        )
    return _power_llm


# ---------------------------------------------------------------------------
# Middleware / Wrapper Nodes
# ---------------------------------------------------------------------------

# Raised from 5 → settings.summary_threshold (default 10) to halve the
# frequency of Flash compression calls on typical conversations.
_SUMMARY_THRESHOLD: int = settings.summary_threshold


def summarize_history_node(state: AgentState) -> dict:
    """Prune the conversation history when it grows too long.

    If ``len(messages) > _SUMMARY_THRESHOLD``:
    * Uses the cheap Flash model to produce a rolling summary that incorporates
      any previous summary plus the oldest messages.
    * Removes the old messages from the state via ``RemoveMessage``.
    * Keeps only the 2 most-recent messages (the last user turn + AI reply).

    This keeps input-token counts low on every subsequent call.
    """
    messages = state.get("messages") or []
    if len(messages) <= _SUMMARY_THRESHOLD:
        return {}  # nothing to prune

    api_key = (settings.gemini_api_key or "").strip()
    is_demo = not api_key or api_key == "your-gemini-api-key-here"

    # Keep the two most-recent messages; summarise everything older.
    messages_to_summarise = messages[:-2]
    existing_summary = (state.get("summary") or "").strip()

    if is_demo:
        # Demo mode: just drop old messages without calling the LLM
        new_summary = existing_summary or "Previous conversation summarised (demo mode)."
        logger.info("summarize_history_node (demo): pruning %d old messages.", len(messages_to_summarise))
        return {
            "summary": new_summary,
            "messages": [RemoveMessage(id=m.id) for m in messages_to_summarise],
        }

    # Build the text to summarise
    conv_text = "\n".join(
        f"{type(m).__name__}: {m.content}" for m in messages_to_summarise
    )

    system_content = (
        "You are a concise summariser for a financial chatbot.  "
        "Your job is to condense conversation history into a brief, factual summary "
        "that preserves all important details (questions asked, answers given, topics discussed).  "
        "Output ONLY the summary paragraph — no preamble, no headings."
    )
    if existing_summary:
        user_content = (
            f"Existing summary:\n{existing_summary}\n\n"
            f"New conversation turns to incorporate:\n{conv_text}\n\n"
            "Produce an updated combined summary."
        )
    else:
        user_content = f"Conversation to summarise:\n{conv_text}"

    try:
        response = _get_cheap_llm().invoke(
            [SystemMessage(content=system_content), HumanMessage(content=user_content)]
        )
        new_summary = (response.content or "").strip()
        if not new_summary:
            new_summary = existing_summary or "Conversation summary unavailable."
        token_event = make_token_event("summarize_history", settings.cheap_model, response)
    except Exception as exc:
        logger.warning("summarize_history_node error: %s — skipping prune.", exc)
        return {}

    logger.info(
        "summarize_history_node: pruned %d messages; summary updated.",
        len(messages_to_summarise),
    )
    return {
        "summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_summarise],
        "token_calls": [token_event],
    }


def gatekeeper_node(state: AgentState) -> dict:
    """Domain guardrail — reject non-finance queries before any expensive work.

    Three-tier classification (cheapest first):
    1. Chitchat pre-check  — free keyword match; short-circuits to chitchat intent.
    2. Finance keyword check — free; approves obvious finance queries without LLM.
    3. Flash LLM fallback   — only called for genuinely ambiguous queries.

    Returns ``domain_status = "authorized"`` for finance topics and
    ``domain_status = "unauthorized"`` for everything else.
    """
    query = state["query"]
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())

    # Tier 1 — free: chitchat pre-check.
    # If it's a greeting or pleasantry, mark as authorized + chitchat so the
    # router skips RAG/LLM entirely.  No Flash call needed.
    if len(query_lower) < 30 and any(t in query_lower for t in _CHITCHAT_TRIGGERS):
        logger.info("Gatekeeper: chitchat pre-check matched — skipping LLM.")
        return {"domain_status": "authorized", "intent": "chitchat"}

    # Tier 2 — free: obvious finance keywords.
    # Covers ~80 % of real traffic without spending any tokens.
    # Recognise ticker-only requests such as "price of TCS" before the
    # generic finance-keyword rule, so they never fall through to static RAG.
    if is_market_data_query(query):
        logger.info("Gatekeeper: market-data query matched â€” skipping LLM.")
        return {"domain_status": "authorized"}

    if query_words & _FINANCE_KEYWORDS:
        logger.info("Gatekeeper: finance keyword matched — skipping LLM.")
        return {"domain_status": "authorized"}

    api_key = (settings.gemini_api_key or "").strip()
    is_demo = not api_key or api_key == "your-gemini-api-key-here"

    if is_demo:
        # Demo mode: no API key available — treat unrecognised queries as out-of-scope.
        logger.info("Gatekeeper (demo, no keyword match): unauthorized.")
        return {"domain_status": "unauthorized"}

    # Tier 3 — Flash LLM: only reached for genuinely ambiguous queries
    # (e.g. "What happened in 2008?" could be finance or history).
    system_content = (
        "You are a domain classifier for a financial assistant chatbot. "
        "Determine if the user's query is related to finance, stock markets, "
        "trading, or economics. "
        "Respond with ONLY the single word 'authorized' if it is finance-related, "
        "or 'unauthorized' if it is not."
    )
    user_content = f"User query: {query}"

    try:
        response = _get_cheap_llm().invoke(
            [SystemMessage(content=system_content), HumanMessage(content=user_content)]
        )
        result = (response.content or "").strip().lower()
        # Check "unauthorized" first because "authorized" is a substring of it
        domain_status = "unauthorized" if "unauthorized" in result else "authorized"
        token_event = make_token_event("gatekeeper", settings.cheap_model, response)
    except Exception as exc:
        logger.error("gatekeeper_node error: %s — defaulting to authorized.", exc)
        domain_status = "authorized"
        token_event = None

    logger.info("Gatekeeper (LLM fallback) classified query as: %s", domain_status)
    payload: dict = {"domain_status": domain_status}
    if token_event:
        payload["token_calls"] = [token_event]
    return payload


def out_of_scope_node(state: AgentState) -> dict:
    """Return a polite refusal when the query is outside the finance domain."""
    return {
        "answer": (
            "I am not able to answer this. "
            "I am only specialized in the financial domain and trading strategies. "
            "Please ask a finance-related question."
        ),
        "confidence": 0.0,
        "source_questions": [],
        "model_used": "gatekeeper",
    }


# ---------------------------------------------------------------------------
# Core Nodes
# ---------------------------------------------------------------------------

_CHITCHAT_REPLIES = {
    "hello": "Hello! I'm your financial assistant. What financial topic can I help you with?",
    "hi": "Hi there! Ask me anything about finance, investments, or trading strategies.",
    "hey": "Hey! Ready to talk finance. What's on your mind?",
    "howdy": "Howdy! Fire away with your financial questions.",
    "thanks": "You're welcome! Feel free to ask more questions.",
    "thank you": "Happy to help! Let me know if you need anything else.",
    "thx": "No problem! Anything else I can help with?",
    "bye": "Goodbye! Come back anytime you have financial questions.",
    "goodbye": "Goodbye! Have a great day!",
    "see you": "See you! Good luck with your investments.",
    "how are you": "I'm doing great and ready to help with your financial questions!",
}


def router_node(state: AgentState) -> dict:
    """Classify the user query as financial Q&A, a quote, chitchat, or a simulation.

    Explicit paper-trade / alert requests win over a prior chitchat shortcut
    so the graph can pause on an approval card.  Otherwise, if gatekeeper
    already set intent, this node is a no-op.
    """
    from agents.human_approval import is_simulation_request

    query = state.get("query") or ""
    if is_simulation_request(query):
        logger.info("Router classified intent: trade_sim")
        return {"intent": "trade_sim"}

    if is_market_data_query(query):
        logger.info("Router classified intent: market_data")
        return {"intent": "market_data"}

    if state.get("intent"):
        logger.info("Router: intent already set to '%s' by gatekeeper.", state["intent"])
        return {}

    query_lower = query.lower().strip()
    is_chitchat = (
        len(query_lower) < 30
        and any(trigger in query_lower for trigger in _CHITCHAT_TRIGGERS)
    )

    intent = "chitchat" if is_chitchat else "financial_qa"
    logger.info("Router classified intent: %s", intent)
    return {"intent": intent}


def market_data_node(state: AgentState) -> dict:
    """Return a provider-backed quote without sending a price question to RAG."""
    quote = get_market_quote(state.get("query") or "")
    if not quote.get("ok"):
        return {
            "answer": quote.get("error") or "Unable to fetch the requested live market data.",
            "confidence": 0.0,
            "model_used": "market_data_error",
            "retrieval_source": "market_data",
            "retrieved_docs": [],
            "source_questions": [],
        }

    change = quote.get("change_percent")
    change_text = f" Change versus the prior returned close: {change:+.2f}%." if change is not None else ""
    answer = (
        f"{quote['name']} ({quote['symbol']}) latest available price: "
        f"{quote['currency']} {quote['price']:,.2f}. "
        f"Provider timestamp: {quote['as_of']}.{change_text} "
        "Data may be delayed; verify before making a trading decision."
    )
    document = {
        "chunk_id": f"market-{quote['symbol']}",
        "question": state.get("query") or "Live market quote",
        "answer": answer,
        "category": "live_market_data",
        "confidence": 1.0,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_timestamp": quote["as_of"],
        "source_url": f"https://finance.yahoo.com/quote/{quote['symbol']}",
        "source_type": "market_data",
    }
    return {
        "answer": answer,
        "confidence": None,
        "model_used": "yfinance",
        "retrieval_source": "market_data",
        "retrieved_docs": [document],
        "source_questions": [document["question"]],
    }


# ---------------------------------------------------------------------------
# Cognitive Memory Nodes
# ---------------------------------------------------------------------------

# Master System Prompt — injected into every financial_qa generation call.
# {long_term_context} is replaced at runtime with retrieved memory episodes.
# When a user has no history the placeholder reads "No previous interactions
# found for this user." so the LLM always sees a well-formed prompt.
_MASTER_SYSTEM_PROMPT = """\
You are a Senior Financial Intelligence Agent. You possess a Cognitive Memory \
System that allows you to recall historical user data, past financial decisions, \
and long-term risk preferences.

### CONTEXTUAL LAYERS
1. **EPISODIC MEMORY (Current Session):** The current conversation is the ground truth.
2. **SEMANTIC MEMORY (Long-Term Context):** Past interactions retrieved for this \
user are listed below. Use them to personalise your response.

### [LONG-TERM MEMORY]
{long_term_context}

### OPERATIONAL GUIDELINES
- **Cross-reference:** Always check the user's current query against their \
[LONG-TERM MEMORY] before answering.
- **Conflict Resolution:** If the request contradicts a known past constraint \
(e.g. the user previously said they avoid high-volatility assets but now asks \
about meme coins), respectfully highlight the discrepancy before proceeding.
- **Personalisation:** Reference memory explicitly — say "Based on your stated \
preference for dividend-paying stocks..." rather than generic advice.
- **Hallucination Guard:** If [LONG-TERM MEMORY] is empty or irrelevant, rely \
strictly on the Knowledge Base Context and your general financial knowledge. \
Do NOT invent past behaviours or preferences.

### ANSWER GUIDELINES
- Use the Knowledge Base Context provided as your primary source where available.
- Never fabricate specific numbers, dates, statistics, or product names.
- If uncertain about a fact, say so clearly.
- Maintain a professional, data-centric tone.
- Use Markdown tables or bullet points for structured financial information.
- Do not give personalised investment advice (no "you should buy/sell X"). \
Explain concepts and options in educational terms only.\
"""

# Models that indicate no real LLM output was produced — these should NOT be
# persisted to long-term memory as they are either static replies or fallbacks.
_NON_MEMORABLE_MODELS = {"fallback", "static", "gatekeeper"}


def memory_retrieve_node(state: AgentState) -> dict:
    """Retrieve the user's relevant past episodes before LLM generation.

    Queries the CognitiveMemoryStore with the current user query and writes
    the formatted result into ``state.long_term_context``.

    No-op for anonymous requests (user_id is None or empty).
    """
    user_id = (state.get("user_id") or "").strip()
    if not user_id:
        return {"long_term_context": ""}

    query = state["query"]
    context = memory_store.retrieve_context(user_id, query)
    return {"long_term_context": context}


def memory_save_node(state: AgentState) -> dict:
    """Persist the current Q&A turn to the user's long-term memory store.

    Only saves when all three conditions are met:
    1. A user_id is present (anonymous sessions are ephemeral).
    2. The answer is a real LLM-generated response (not a fallback or static reply).
    3. There is a non-empty answer to save.

    Failures are swallowed — a memory write error must never break the response.
    """
    user_id = (state.get("user_id") or "").strip()
    answer = (state.get("answer") or "").strip()
    query = (state.get("query") or "").strip()
    model_used = (state.get("model_used") or "").strip()

    if not user_id or not answer or not query:
        return {}
    if model_used in _NON_MEMORABLE_MODELS:
        return {}

    try:
        memory_store.save_episode(user_id, query, answer)
    except Exception as exc:
        logger.warning("memory_save_node: failed to save episode for user %s: %s", user_id, exc)

    return {}


def rag_retrieve_node(state: AgentState) -> dict:
    """Retrieve top-k relevant Q&A pairs from the FAISS knowledge base."""
    query = (state.get("retrieval_query") or state["query"]).strip()
    source = "faiss_rewritten" if int(state.get("rewrite_attempts") or 0) > 0 else "faiss"

    try:
        docs = financial_retriever.retrieve(query, k=settings.max_results)

        if docs:
            context_blocks = [
                f"[{d.get('category', 'general').upper()}]\nQ: {d['question']}\nA: {d['answer']}"
                for d in docs
            ]
            context = "\n\n".join(context_blocks)
            source_questions = [d["question"] for d in docs]
        else:
            context = ""
            source_questions = []
            logger.warning("No documents retrieved; will attempt direct Gemini answer.")

        return {
            "retrieved_docs": docs,
            "context": context,
            "source_questions": source_questions,
            "retrieval_source": source,
        }

    except Exception as exc:
        logger.error("RAG retrieval error: %s", exc)
        return {
            "retrieved_docs": [],
            "context": "",
            "source_questions": [],
            "error": str(exc),
            "retrieval_source": source,
        }


def gemini_generate_node(state: AgentState) -> dict:
    """Generate a grounded answer using automatic Flash/Pro tier selection.

    Model selection (cheapest-first):
    - Flash  — when RAG returned full coverage (``len(retrieved_docs) >= max_results``).
              The context already constrains the answer, so the cheaper model
              is sufficient and costs ~47x less than Pro.
    - Pro    — when RAG coverage is partial or zero; open-ended generation needs
              the stronger model's reasoning.

    Prompt is built from three low-token sources only (no raw history):
    1. ``summary``  — rolling summary of older turns (may be empty).
    2. ``context``  — RAG-retrieved knowledge-base snippets (may be empty).
    3. ``query``    — current user question.
    """
    query = state["query"]
    context = state.get("context", "")
    summary = (state.get("summary") or "").strip()
    retrieved_docs = state.get("retrieved_docs") or []
    api_key = (settings.gemini_api_key or "").strip()
    is_demo = not api_key or api_key == "your-gemini-api-key-here"

    # ── Fallback path (no API key) ─────────────────────────────────────────
    if is_demo:
        logger.info("Gemini API key not set; using fallback answer from retrieved docs.")
        if retrieved_docs:
            return {"answer": retrieved_docs[0]["answer"], "confidence": 0.8, "model_used": "fallback"}
        return {
            "answer": (
                "I don't have information on this topic in my knowledge base. "
                "Set GEMINI_API_KEY in your .env to enable AI-powered answers."
            ),
            "confidence": 0.0,
            "model_used": "fallback",
        }

    # ── Model tier selection ───────────────────────────────────────────────
    # Flash when we have full RAG coverage; Pro for open-ended/low-coverage queries.
    use_flash = settings.use_flash_for_grounded_answers and len(retrieved_docs) >= settings.max_results
    llm = _get_cheap_llm() if use_flash else _get_power_llm()
    model_label = settings.cheap_model if use_flash else settings.power_model
    logger.info(
        "Generation model: %s (RAG docs: %d/%d).",
        model_label, len(retrieved_docs), settings.max_results,
    )

    # ── Build Master Prompt with injected long-term memory ────────────────
    try:
        long_term_context = (state.get("long_term_context") or "").strip()
        memory_display = long_term_context or "No previous interactions found for this user."
        system_content = _MASTER_SYSTEM_PROMPT.format(long_term_context=memory_display)

        user_parts: list[str] = []

        if summary:
            user_parts.append(f"Conversation Summary (for context):\n{summary}")

        if context:
            source_label = "Web Search Context" if state.get("retrieval_source") == "web" else "Knowledge Base Context"
            user_parts.append(f"{source_label}:\n{context}")

        user_parts.append(f"Question: {query}")
        user_parts.append("Please provide an accurate and helpful answer.")

        user_content = "\n\n".join(user_parts)

        response = llm.invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ])
        answer = (response.content or "").strip()

        if not answer:
            raise ValueError("Gemini returned an empty response.")

        logger.info("Answer generated successfully (model=%s).", model_label)
        return {
            "answer": answer,
            "confidence": None,
            "messages": [AIMessage(content=answer)],
            "model_used": model_label,
            "token_calls": [make_token_event("gemini_generate", model_label, response)],
        }

    except Exception as exc:
        logger.error("Gemini generation error: %s", exc)
        fallback = retrieved_docs[0]["answer"] if retrieved_docs else "Sorry, I was unable to process your request."
        return {
            "answer": fallback,
            "confidence": 0.6,
            "error": str(exc),
            "model_used": "fallback",
        }


def chitchat_node(state: AgentState) -> dict:
    """Return a canned reply for greetings and small-talk."""
    query_lower = state["query"].lower().strip()

    reply = next(
        (v for k, v in _CHITCHAT_REPLIES.items() if k in query_lower),
        "Hello! I'm your financial assistant. Ask me anything about finance and investments!",
    )

    return {
        "answer": reply,
        "confidence": 1.0,
        "source_questions": [],
        "model_used": "static",
    }
