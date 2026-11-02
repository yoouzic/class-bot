"""Web search for answers beyond local materials.

Primary provider is the DeepSeek Search API (``POST /v1/tools/search``,
structured JSON, billed per successful request) using the configured API key
and base URL; keyless HTML scraping (Bing, then DuckDuckGo) remains as a
degradation layer for deployments without a key. Results are always
presented to the model as untrusted, clearly labeled candidates — they never
enter the local evidence numbering system. Any failure returns an empty
tuple so callers degrade to their existing behavior.
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx

_LOGGER = logging.getLogger(__name__)

_DEEPSEEK_SEARCH_PATH = "/v1/tools/search"
# The Search API caps the query at 128 Unicode characters.
_QUERY_LIMIT = 128
_BING_URL = "https://www.bing.com/search"
_DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_BING_BLOCK_SPLIT = re.compile(r'<li class="b_algo"')
_BING_TITLE_RE = re.compile(
    r'<h2[^>]*><a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_BING_SNIPPET_RE = re.compile(r"<p[^>]*>(?P<snippet>.*?)</p>", re.DOTALL)
_DUCKDUCKGO_RESULT_RE = re.compile(
    r'class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r".*?class=\"result__snippet\"[^>]*>(?P<snippet>.*?)</a>",
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class WebResult:
    title: str
    url: str
    snippet: str


def _clean(fragment: str) -> str:
    return _WHITESPACE_RE.sub(" ", _TAG_RE.sub("", html.unescape(fragment))).strip()


def _direct_url(url: str) -> str:
    # DuckDuckGo wraps results in /l/?uddg=<encoded> redirects.
    if "uddg=" in url:
        query = parse_qs(urlparse(url.replace("&amp;", "&")).query)
        targets = query.get("uddg")
        if targets:
            return unquote(targets[0])
    return url.replace("&amp;", "&")


def _parse_bing(html_text: str, limit: int) -> tuple[WebResult, ...]:
    results: list[WebResult] = []
    for block in _BING_BLOCK_SPLIT.split(html_text)[1:]:
        title_match = _BING_TITLE_RE.search(block)
        if title_match is None:
            continue
        snippet_match = _BING_SNIPPET_RE.search(block)
        result = WebResult(
            title=_clean(title_match["title"]),
            url=_direct_url(_clean(title_match["url"])),
            snippet=_clean(snippet_match["snippet"]) if snippet_match else "",
        )
        if result.title and result.url.startswith("http") and result.snippet:
            results.append(result)
        if len(results) >= limit:
            break
    return tuple(results)


def _parse(
    html_text: str, pattern: re.Pattern[str], limit: int
) -> tuple[WebResult, ...]:
    results = []
    for match in pattern.finditer(html_text):
        result = WebResult(
            title=_clean(match["title"]),
            url=_direct_url(_clean(match["url"])),
            snippet=_clean(match["snippet"]),
        )
        if result.title and result.snippet and result.url.startswith("http"):
            results.append(result)
        if len(results) >= limit:
            break
    return tuple(results)


async def _search_deepseek(
    query: str,
    http_client: httpx.AsyncClient,
    api_key: str,
    limit: int,
    timeout: float,
) -> tuple[WebResult, ...]:
    """Call the DeepSeek Search API through the shared base URL and key."""
    try:
        response = await http_client.post(
            _DEEPSEEK_SEARCH_PATH,
            json={
                "query": query,
                "count": limit,
                "need_content": False,
                "ori_question": query,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, OSError) as error:
        _LOGGER.debug("deepseek web search failed: %s", error)
        return ()
    items = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return ()
    results: list[WebResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result = WebResult(
            title=_clean(str(item.get("title") or "")),
            url=_clean(str(item.get("url") or "")),
            snippet=_clean(
                str(item.get("snippet") or item.get("summary") or "")
            ),
        )
        if result.title and result.url.startswith("http") and result.snippet:
            results.append(result)
        if len(results) >= limit:
            break
    return tuple(results)


async def search_web(
    query: str,
    http_client: httpx.AsyncClient,
    *,
    api_key: str | None = None,
    limit: int = 4,
    timeout: float = 6.0,
) -> tuple[WebResult, ...]:
    """DeepSeek Search first, then keyless providers; () when all fail."""
    query = " ".join(query.split())[:_QUERY_LIMIT]
    if not query:
        return ()
    if api_key:
        results = await _search_deepseek(
            query, http_client, api_key, limit, timeout
        )
        if results:
            return results
    headers = {"User-Agent": _USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"}
    for provider, url in (
        ("bing", _BING_URL),
        ("duckduckgo", _DUCKDUCKGO_URL),
    ):
        try:
            response = await http_client.get(
                url,
                params={"q": query},
                headers=headers,
                timeout=timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            if provider == "bing":
                results = _parse_bing(response.text, limit)
            else:
                results = _parse(response.text, _DUCKDUCKGO_RESULT_RE, limit)
        except (httpx.HTTPError, OSError) as error:
            _LOGGER.debug("web search provider %s failed: %s", provider, error)
            continue
        if results:
            return results
        _LOGGER.debug("web search provider %s returned no parseable results", provider)
    return ()
