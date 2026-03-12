"""Example search tools."""

from agentscope.tool import ToolResponse


def web_search(query: str, api_key: str | None = None) -> ToolResponse:
    """Search the web for information.

    Args:
        query: The search query.
        api_key: Optional API key for search service.
    """
    # TODO: Implement actual search logic
    # This is a placeholder
    return ToolResponse(
        content=f"Search results for '{query}' (placeholder - implement actual search)",
    )


def search_documentation(query: str) -> ToolResponse:
    """Search documentation.

    Args:
        query: The search query.
    """
    # TODO: Implement documentation search
    return ToolResponse(
        content=f"Documentation results for '{query}' (placeholder)",
    )
