"""
Tool safety policy — the chat graph does not expose tool-calling to the LLM.

Only read-only, allow-listed helpers may be invoked by application code
(not by model-generated instructions inside CONTEXT/TOOL_DATA).
"""

from __future__ import annotations

READ_ONLY_TOOLS = frozenset(
    {
        "get_market_quote",
        "search_web",
        "financial_retriever.retrieve",
    }
)


def assert_read_only_tool(tool_name: str) -> None:
    if tool_name not in READ_ONLY_TOOLS:
        raise PermissionError(f"Tool '{tool_name}' is not allow-listed for read-only use.")
