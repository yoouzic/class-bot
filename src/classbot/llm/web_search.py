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

import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass, replace
from urllib.parse import parse_qs, unquote, urlparse

import httpx

_LOGGER = logging.getLogger(__name__)

_DEEPSEEK_SEARCH_PATH = "/v1/tools/search"
_DEEPSEEK_FETCH_PATH = "/v1/tools/fetch"
_URL_LIMIT = 8192
_FETCH_CONTENT_LIMIT = 4000
_DETAIL_MARKERS = (
    "多少", "几个", "版本", "年份", "价格", "最新",
    "参数", "配置", "步骤", "代码", "对比",
)
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
    content: str = ""
    fetched: bool = False


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
            content=_clean(str(item.get("content") or "")),
        )
        if result.title and result.url.startswith("http") and result.snippet:
            results.append(result)
        if len(results) >= limit:
            break
    return tuple(results)


async def fetch_web(
    url: str,
    http_client: httpx.AsyncClient,
    *,
    api_key: str | None,
    timeout: float = 1.0,
) -> WebResult | None:
    parsed = urlparse(url)
    if (
        not api_key
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or len(url) > _URL_LIMIT
    ):
        return None
    try:
        response = await http_client.post(
            _DEEPSEEK_FETCH_PATH,
            json={"url": url},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, OSError) as error:
        _LOGGER.debug("deepseek fetch failed: %s", error)
        return None
    if not isinstance(payload, dict):
        return None
    title = _clean(str(payload.get("title") or ""))
    content = str(payload.get("content") or "")
    if not content.strip():
        return None
    return WebResult(
        title=title or url,
        url=url,
        snippet="",
        content=content[:_FETCH_CONTENT_LIMIT],
        fetched=True,
    )


def _needs_fetch(results: tuple[WebResult, ...], query: str) -> bool:
    snippets = [result.snippet for result in results if result.snippet]
    if not snippets:
        return True
    if sum(len(snippet) for snippet in snippets) < 240:
        return True
    if max(len(snippet) for snippet in snippets) < 80:
        return True
    return any(marker in query for marker in _DETAIL_MARKERS)


def fetch_mode_from_env() -> str:
    value = os.getenv("CLASSBOT_WEB_FETCH", "auto").strip().lower()
    return value if value in {"auto", "off", "always"} else "auto"


def timeout_from_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


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


async def research_web(
    query: str,
    http_client: httpx.AsyncClient,
    *,
    api_key: str | None = None,
    limit: int = 4,
    search_timeout: float = 1.8,
    fetch_timeout: float = 1.0,
    fetch_mode: str = "auto",
) -> tuple[WebResult, ...]:
    results = await search_web(
        query,
        http_client,
        api_key=api_key,
        limit=limit,
        timeout=search_timeout,
    )
    mode = fetch_mode if fetch_mode in {"auto", "off", "always"} else "auto"
    if not results or not api_key or mode == "off":
        return results
    if mode != "always" and not _needs_fetch(results, query):
        return results
    candidates = [
        result for result in results
        if result.url.startswith(("http://", "https://"))
    ][:2]
    if not candidates:
        return results
    tasks = [
        asyncio.create_task(
            fetch_web(
                result.url,
                http_client,
                api_key=api_key,
                timeout=fetch_timeout,
            )
        )
        for result in candidates
    ]
    done, pending = await asyncio.wait(tasks, timeout=fetch_timeout)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    fetched: dict[str, WebResult] = {}
    for task, candidate in zip(tasks, candidates):
        if task not in done:
            continue
        try:
            page = task.result()
        except Exception:
            page = None
        if page is not None:
            fetched[candidate.url] = page
    return tuple(
        replace(result, content=page.content, fetched=True)
        if (page := fetched.get(result.url)) is not None
        else result
        for result in results
    )
