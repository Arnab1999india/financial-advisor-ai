"""
LangGraph Node Functions
========================
Each function here represents one node in the StateGraph.  Nodes receive the
current ``AgentState``, perform their work, and return a *partial* state dict
that LangGraph merges into the global state.

Node catalogue
--------------
router_node         — Classify intent: financial_qa | chitchat
rag_retrieve_node   — FAISS MMR retrieval for relevant Q&A context
gemini_generate_node — Generate answer via Gemini with (or without) context
chitchat_node       — Fast static replies for greetings / small-talk
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from config import settings
from logger import logger
from rag.retriever import retriever as financial_retriever

# ---------------------------------------------------------------------------
# LLM factory — lazy import so the app boots even without langchain-google-genai
# ---------------------------------------------------------------------------

def _build_llm():
    """Instantiate ChatGoogleGenerativeAI with current settings."""
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        google_api_key=settings.gemini_api_key,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_CHITCHAT_TRIGGERS = {
    "hello", "hi", "hey", "howdy",
    "thanks", "thank you", "thx",
    "bye", "goodbye", "see you",
    "how are you",
}

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
    """Classify the user query as *financial_qa* or *chitchat*.

    Uses a lightweight keyword check — fast and predictable for this
    domain-specific bot.  All substantive questions fall through to
    ``financial_qa`` regardless of phrasing.
    """
    query_lower = state["query"].lower().strip()
    words = set(query_lower.split())

    # Treat very short messages that match chitchat keywords as chitchat
    is_chitchat = (
        len(query_lower) < 30
        and any(trigger in query_lower for trigger in _CHITCHAT_TRIGGERS)
    )

    intent = "chitchat" if is_chitchat else "financial_qa"
    logger.info("Router classified intent: %s", intent)
    return {"intent": intent}


def rag_retrieve_node(state: AgentState) -> dict:
    """Retrieve top-k relevant Q&A pairs from the FAISS knowledge base."""
    query = state["query"]

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
        }

    except Exception as exc:
        logger.error("RAG retrieval error: %s", exc)
        return {
            "retrieved_docs": [],
            "context": "",
            "source_questions": [],
            "error": str(exc),
        }


def gemini_generate_node(state: AgentState) -> dict:
    """Generate a grounded answer via Gemini, falling back gracefully."""
    query = state["query"]
    context = state.get("context", "")
    api_key = (settings.gemini_api_key or "").strip()
    is_demo = not api_key or api_key == "your-gemini-api-key-here"

    # ── Fallback path (no API key) ─────────────────────────────────────────
    if is_demo:
        logger.info("Gemini API key not set; using fallback answer from retrieved docs.")
        docs = state.get("retrieved_docs") or []
        if docs:
            return {"answer": docs[0]["answer"], "confidence": 0.8, "model_used": "fallback"}
        return {
            "answer": (
                "I don't have information on this topic in my knowledge base. "
                "Set GEMINI_API_KEY in your .env to enable AI-powered answers."
            ),
            "confidence": 0.0,
            "model_used": "fallback",
        }

    # ── Gemini path ────────────────────────────────────────────────────────
    try:
        llm = _build_llm()

        if context:
            system_content = (
                "You are a knowledgeable and trustworthy financial assistant. "
                "Answer the user's question using the provided context as your primary source. "
                "You may supplement with your general financial knowledge, but never fabricate "
                "specific numbers, dates, statistics, or product names. "
                "Be concise and clear."
            )
            user_content = (
                f"Question: {query}\n\n"
                f"Knowledge Base Context:\n{context}\n\n"
                "Please provide an accurate and helpful answer."
            )
        else:
            system_content = (
                "You are a knowledgeable and trustworthy financial assistant. "
                "Answer financial questions accurately and concisely. "
                "If you are uncertain about a specific fact, say so clearly."
            )
            user_content = f"Question: {query}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        answer = (response.content or "").strip()

        if not answer:
            raise ValueError("Gemini returned an empty response.")

        logger.info("Gemini answered successfully (model=%s).", settings.gemini_model)
        return {
            "answer": answer,
            "confidence": None,
            "messages": [AIMessage(content=answer)],
            "model_used": settings.gemini_model,
        }

    except Exception as exc:
        logger.error("Gemini generation error: %s", exc)
        # Graceful fallback to best retrieved doc
        docs = state.get("retrieved_docs") or []
        fallback = docs[0]["answer"] if docs else "Sorry, I was unable to process your request."
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
