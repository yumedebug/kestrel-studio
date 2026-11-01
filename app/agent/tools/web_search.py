"""Agent tool wrapper for the API-key-free read-only web search."""

from __future__ import annotations

from typing import Any

from app.agent.tools.registry import ToolContext, ToolError, ToolRegistry, ToolSpec
from app.agent.web_search import WebSearchError, search_web


def web_search(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    try:
        results = search_web(
            str(arguments.get("query") or ""),
            limit=int(arguments.get("limit") or 10),
            timeout=float(arguments.get("timeout") or 10),
        )
    except (WebSearchError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    data = {"query": str(arguments.get("query") or ""), "count": len(results),
            "results": [result.to_dict() for result in results]}
    return {
        "ok": True,
        "summary": f"Found {len(results)} web result(s).",
        "data": data,
    }


def register(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="web_search",
            description=(
                "Search the public web with DuckDuckGo HTML. Read-only, no API key, "
                "and limited to approved search domains."
            ),
            kind="web_search",
            parameters={
                "query": {"type": "string", "required": True, "description": "Search terms."},
                "limit": {"type": "integer", "description": "Maximum results, 1-20."},
                "timeout": {"type": "number", "description": "Request timeout in seconds."},
            },
            modes=("plan", "build", "autonomous"),
        ),
        web_search,
    )


__all__ = ["register", "web_search"]
