"""Web search backends behind one interface, plus helpers to run many queries."""
from __future__ import annotations

import logging
from typing import Protocol

import httpx
from pydantic import BaseModel

from research_system.config.settings import Settings, get_settings
from research_system.core.models import make_source_id
from research_system.tools.errors import ToolError

log = logging.getLogger(__name__)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class SearchTool(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class DuckDuckGoSearch:
    """Free, no API key. Can be rate limited, so failures raise ToolError."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        from ddgs import DDGS

        try:
            raw = DDGS().text(query, max_results=max_results)
        except Exception as e:  # the library raises several different exception types
            raise ToolError(f"DuckDuckGo search failed: {e}") from e

        results: list[SearchResult] = []
        for item in raw or []:
            url = item.get("href") or item.get("url")
            if url:
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=url,
                        snippet=item.get("body", ""),
                    )
                )
        return results


class TavilySearch:
    """Better quality results; needs TAVILY_API_KEY (free tier available)."""

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._http = httpx.Client(
            base_url="https://api.tavily.com",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            resp = self._http.post("/search", json={"query": query, "max_results": max_results})
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ToolError(f"Tavily search failed: {e}") from e

        return [
            SearchResult(
                title=item.get("title", ""),
                url=item["url"],
                snippet=item.get("content", ""),
            )
            for item in resp.json().get("results", [])
            if item.get("url")
        ]


def get_search_tool(settings: Settings | None = None) -> SearchTool:
    settings = settings or get_settings()
    if settings.tavily_api_key:
        return TavilySearch(settings.tavily_api_key)
    return DuckDuckGoSearch()


def search_many(
    tool: SearchTool, queries: list[str], max_results: int = 5
) -> tuple[list[SearchResult], list[str]]:
    """Run several queries, dedupe by URL, and collect errors instead of raising."""
    seen: set[str] = set()
    results: list[SearchResult] = []
    errors: list[str] = []
    for query in queries:
        try:
            found = tool.search(query, max_results=max_results)
        except ToolError as e:
            errors.append(f"search '{query}': {e}")
            continue
        for result in found:
            sid = make_source_id(result.url)
            if sid not in seen:
                seen.add(sid)
                results.append(result)
    return results, errors