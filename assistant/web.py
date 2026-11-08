"""
assistant/web.py
================
The research agent's web search: Wikipedia's public API.

It needs no key, returns article summaries with their URLs (so answers can
cite sources), and identifies itself with a generic User-Agent as Wikimedia
asks of automated clients.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = "GateOrchestra/1.0 (research assistant)"
TIMEOUT = 15


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def search_wikipedia(query: str, limit: int = 3) -> str:
    """Summaries of the top Wikipedia articles for ``query``, each with its URL."""
    query = query.strip()
    if not query:
        return "Error: empty search query."
    try:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": limit,
            }
        )
        hits = _get(f"{API}?{params}").get("query", {}).get("search", [])
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
        return f"Error: Wikipedia search failed ({type(err).__name__})."
    if not hits:
        return f"No Wikipedia article matches {query!r}."
    parts = []
    for hit in hits[:limit]:
        title = hit["title"]
        try:
            summary = _get(SUMMARY + urllib.parse.quote(title.replace(" ", "_"), safe=""))
            text = summary.get("extract") or ""
            url = (summary.get("content_urls") or {}).get("desktop", {}).get("page") or (
                "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            text, url = "", "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
                title.replace(" ", "_")
            )
        parts.append(f"## {title}\n{url}\n{text[:1200]}")
    return "\n\n".join(parts)
