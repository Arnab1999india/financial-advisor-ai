"""
LangChain Tools — Financial Bot
================================
Defines @tool-decorated functions that the Gemini agent can invoke via
function calling.  These are OpenAI-function-calling-compatible through
LangChain's unified tool interface, meaning the same tool definitions work
with any LangChain-supported LLM (Gemini, OpenAI, Anthropic, etc.).
"""

import json
from langchain_core.tools import tool

from rag.retriever import retriever as financial_retriever


@tool
def search_financial_knowledge(query: str) -> str:
    """Search the internal financial knowledge base for relevant Q&A context.

    Use this tool when answering questions about stocks, investments, trading
    strategies, mutual funds, bonds, portfolio management, or any financial topic.

    Args:
        query: The financial question or topic to search for.

    Returns:
        Formatted Q&A context string, or a message indicating nothing was found.
    """
    results = financial_retriever.retrieve(query)

    if not results:
        return "No relevant information found in the knowledge base."

    context_parts = [
        f"[{r.get('category', 'general').upper()}]\nQ: {r['question']}\nA: {r['answer']}"
        for r in results
    ]
    return "\n\n".join(context_parts)


@tool
def get_available_trading_strategies() -> str:
    """Return the list of available trading strategies and their descriptions.

    Use this tool when a user asks which trading strategies are supported or
    wants to know what kind of analysis can be performed on uploaded data.

    Returns:
        JSON string mapping strategy names to descriptions.
    """
    strategies = {
        "swing_trading": (
            "Moving Average Crossover — Buy when the short-term MA (20-period) crosses "
            "above the long-term MA (50-period); sell on the inverse crossover."
        ),
        "scalping_trading": (
            "RSI-Based Scalping — Buy when RSI drops below 30 (oversold); "
            "sell when RSI rises above 70 (overbought)."
        ),
        "44_moving_average": (
            "44-Period MA — Buy when the closing price is above the 44-period MA; "
            "sell when it falls below."
        ),
        "trend_trading": (
            "MACD Crossover — Buy on a bullish MACD/signal-line crossover; "
            "sell on a bearish crossover."
        ),
        "range_trading": (
            "Bollinger Bands — Buy when price touches the lower band; "
            "sell when price touches the upper band."
        ),
    }
    return json.dumps(strategies, indent=2)


@tool
def search_web_for_finance(query: str) -> str:
    """Search the public web for financial context when the knowledge base is insufficient.

    Use this after internal retrieval fails to cover the question.

    Args:
        query: The financial question to search for.

    Returns:
        Formatted web snippets, or a message indicating nothing was found.
    """
    from tools.web_search import format_web_context, search_web

    hits = search_web(query)
    context = format_web_context(hits)
    return context or "No relevant web results found."


# Exported list used when building the tool-calling agent
FINANCIAL_TOOLS = [
    search_financial_knowledge,
    get_available_trading_strategies,
    search_web_for_finance,
]
