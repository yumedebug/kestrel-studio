"""API-key-free, read-only web search using DuckDuckGo's HTML endpoint."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
ALLOWED_SEARCH_HOSTS = frozenset({"html.duckduckgo.com", "duckduckgo.com"})
MAX_QUERY_LENGTH = 500
MAX_RESULTS = 20
MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_TIMEOUT = 10.0
USER_AGENT = "KestrelStudio/0.2 (read-only search)"


class WebSearchError(RuntimeError):
    """Raised when a safe search cannot be completed."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class _DuckParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._tag = ""
        self._classes: set[str] = set()
        self._href = ""
        self._title = ""
        self._snippet = ""
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._in_title = True
            self._href = attributes.get("href") or ""
            self._title = ""
        elif "result__snippet" in classes:
            self._in_snippet = True
            self._snippet = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title += data
        if self._in_snippet:
            self._snippet += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            url = _safe_result_url(self._href)
            title = _clean(self._title)
            if url and title:
                self.results.append(SearchResult(title, url, ""))
            self._in_title = False
        if self._in_snippet and tag in {"a", "div", "td"}:
            self._in_snippet = False
            snippet = _clean(self._snippet)
            if snippet and self.results and not self.results[-1].snippet:
                self.results[-1] = SearchResult(
                    self.results[-1].title, self.results[-1].url, snippet
                )


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _safe_result_url(raw: str) -> str:
    value = html.unescape(raw).strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value
    if parsed.path.startswith("/l/"):
        encoded = parse_qs(parsed.query).get("uddg", [""])[0]
        target = urlparse(encoded)
        if target.scheme in {"http", "https"} and target.netloc:
            return encoded
    return ""


def search_target() -> str:
    """Permission target used by ToolExecutor; never accepts a user URL."""
    return SEARCH_ENDPOINT


def search_web(
    query: str,
    *,
    limit: int = 10,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Callable[..., object] | None = None,
) -> list[SearchResult]:
    text = str(query or "").strip()
    if not text:
        raise WebSearchError("A search query is required.")
    if len(text) > MAX_QUERY_LENGTH:
        raise WebSearchError(f"Search queries are limited to {MAX_QUERY_LENGTH} characters.")
    count = max(1, min(MAX_RESULTS, int(limit)))
    try:
        timeout_value = max(1.0, min(30.0, float(timeout)))
    except (TypeError, ValueError):
        timeout_value = DEFAULT_TIMEOUT
    endpoint = urlparse(SEARCH_ENDPOINT)
    if endpoint.hostname not in ALLOWED_SEARCH_HOSTS:
        raise WebSearchError("The configured search endpoint is not allowed.")
    request = Request(
        f"{SEARCH_ENDPOINT}?q={quote_plus(text)}",
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        method="GET",
    )
    try:
        with (opener or urlopen)(request, timeout=timeout_value) as response:
            final_url = getattr(response, "geturl", lambda: SEARCH_ENDPOINT)()
            final = urlparse(str(final_url))
            if final.scheme not in {"http", "https"} or final.hostname not in ALLOWED_SEARCH_HOSTS:
                raise WebSearchError("Search redirected to an unapproved domain.")
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except WebSearchError:
        raise
    except Exception as exc:
        raise WebSearchError(f"Web search failed: {type(exc).__name__}: {exc}") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        raise WebSearchError("The search response was too large.")
    parser = _DuckParser()
    try:
        parser.feed(body.decode("utf-8", "replace"))
    except Exception as exc:
        raise WebSearchError(f"Could not parse search results: {exc}") from exc
    return parser.results[:count]


__all__ = [
    "ALLOWED_SEARCH_HOSTS",
    "SearchResult",
    "WebSearchError",
    "search_target",
    "search_web",
]
