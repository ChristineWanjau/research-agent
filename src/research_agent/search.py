"""Web search behind one small interface: Tavily first, DuckDuckGo as fallback."""

import logging
import time
from typing import Protocol, TypedDict

log = logging.getLogger(__name__)


class Source(TypedDict):
    title: str
    url: str
    content: str


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[Source]: ...


class TavilySearch:
    def __init__(self, api_key: str, max_chars: int = 1500):
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)
        self._max_chars = max_chars

    def search(self, query: str, max_results: int) -> list[Source]:
        data = self._client.search(query, max_results=max_results)
        return [
            Source(title=r.get("title", ""), url=r["url"], content=(r.get("content") or "")[: self._max_chars])
            for r in data.get("results", [])
        ]


class DuckDuckGoSearch:
    def __init__(self, max_chars: int = 1500, retries: int = 3):
        self._max_chars = max_chars
        self._retries = retries

    def search(self, query: str, max_results: int) -> list[Source]:
        from ddgs import DDGS

        for attempt in range(self._retries):
            try:
                results = DDGS().text(query, max_results=max_results) or []
                return [
                    Source(title=r.get("title", ""), url=r["href"], content=(r.get("body") or "")[: self._max_chars])
                    for r in results
                ]
            except Exception as exc:  # ddgs raises on rate limits
                log.warning("DuckDuckGo attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2**attempt)
        return []


class FallbackSearch:
    """Try each provider in order; use the first one that succeeds."""

    def __init__(self, *providers: SearchProvider):
        self._providers = providers

    def search(self, query: str, max_results: int) -> list[Source]:
        for provider in self._providers:
            try:
                return provider.search(query, max_results)
            except Exception as exc:
                log.warning("%s failed, falling back: %s", type(provider).__name__, exc)
        return []


def make_search(settings) -> SearchProvider:
    ddg = DuckDuckGoSearch(settings.max_source_chars)
    if settings.tavily_api_key:
        return FallbackSearch(TavilySearch(settings.tavily_api_key, settings.max_source_chars), ddg)
    return ddg
