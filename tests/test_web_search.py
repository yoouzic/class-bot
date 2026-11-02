"""Web search: DeepSeek Search API first, keyless providers as fallback."""
from __future__ import annotations

import json

import httpx
import pytest

from classbot.llm.web_search import search_web

_DEEPSEEK_BODY = {
    "results": [
        {
            "rerank_score": 2.4,
            "title": "剩余<b>价值</b>的定义",
            "url": "https://example.com/surplus",
            "snippet": "剩余价值是劳动者创造的被资产阶级无偿占有的价值。",
            "date": "2026-01-01T00:00:00+00:00",
            "authority": 3,
        },
        {"title": "", "url": "https://example.com/empty", "snippet": "无标题"},
    ]
}

_BING_HTML = """
<html><body>
<ol id="b_results">
<li class="b_algo"><div><h2><a
 href="https://example.com/one">第一条标题</a></h2>
<div><p>第一条摘要内容。</p></div></div></li>
</ol>
</body></html>
"""


def _client(handler, *, base_url: str | None = None) -> httpx.AsyncClient:
    kwargs: dict = {"transport": httpx.MockTransport(handler)}
    if base_url:
        kwargs["base_url"] = base_url
    return httpx.AsyncClient(**kwargs)


@pytest.mark.asyncio
async def test_deepseek_search_api_is_the_primary_provider():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/tools/search"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["query"] == "剩余价值 是什么"
        assert body["count"] == 4
        assert body["need_content"] is False
        return httpx.Response(200, json=_DEEPSEEK_BODY)

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        results = await search_web(
            "剩余价值 是什么", client, api_key="test-key",
        )

    assert len(requests) == 1
    assert [item.title for item in results] == ["剩余价值的定义"]
    assert results[0].url == "https://example.com/surplus"
    assert "无偿占有" in results[0].snippet


@pytest.mark.asyncio
async def test_deepseek_failure_falls_back_to_keyless_providers():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/tools/search":
            return httpx.Response(401)
        assert "bing.com" in str(request.url)
        return httpx.Response(200, text=_BING_HTML)

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        results = await search_web("查询", client, api_key="test-key")

    assert [item.title for item in results] == ["第一条标题"]


@pytest.mark.asyncio
async def test_missing_api_key_skips_the_deepseek_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/v1/tools/search" not in request.url.path
        return httpx.Response(200, text=_BING_HTML)

    async with _client(handler) as client:
        results = await search_web("查询", client, api_key=None)

    assert [item.title for item in results] == ["第一条标题"]


@pytest.mark.asyncio
async def test_query_is_truncated_to_the_api_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert len(body["query"]) == 128
        return httpx.Response(200, json=_DEEPSEEK_BODY)

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        results = await search_web("长" * 200, client, api_key="test-key")

    assert results


@pytest.mark.asyncio
async def test_duckduckgo_fallback_when_bing_fails_and_redirects_are_unwrapped():
    _DDG_HTML = (
        '<html><body><div class="result">'
        '<a rel="nofollow" class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fddg">嵌套标题</a>'
        '<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">嵌套<b>摘要</b>内容</a>'
        "</div></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "bing.com" in str(request.url):
            return httpx.Response(503)
        assert "duckduckgo" in str(request.url)
        return httpx.Response(200, text=_DDG_HTML)

    async with _client(handler) as client:
        results = await search_web("测试查询", client)

    assert results[0].title == "嵌套标题"
    assert results[0].url == "https://example.com/ddg"
    assert results[0].snippet == "嵌套摘要内容"


@pytest.mark.asyncio
async def test_all_provider_failures_return_empty_tuple():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        results = await search_web("测试查询", client, api_key="test-key")

    assert results == ()


@pytest.mark.asyncio
async def test_blank_query_skips_the_network():
    async with _client(
        lambda request: pytest.fail("network must not be called")
    ) as client:
        results = await search_web("   ", client, api_key="test-key")

    assert results == ()


from classbot.llm import web_search as web_module
from classbot.llm.web_search import fetch_web, research_web

_FETCH_BODY = {
    "title": "页面标题",
    "url": "https://example.com/page",
    "content": "# 标题\n正文内容",
}


@pytest.mark.asyncio
async def test_fetch_web_returns_markdown_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/tools/fetch"
        assert json.loads(request.content) == {"url": "https://example.com/page"}
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json=_FETCH_BODY)

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        result = await fetch_web(
            "https://example.com/page", client, api_key="test-key"
        )

    assert result is not None
    assert result.content.startswith("# 标题")
    assert result.fetched is True


@pytest.mark.asyncio
async def test_fetch_web_rejects_non_http_url():
    async with _client(
        lambda request: pytest.fail("network must not be called")
    ) as client:
        assert await fetch_web("file:///etc/passwd", client, api_key="k") is None


@pytest.mark.asyncio
async def test_research_web_fetches_short_snippets():
    fetched = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/tools/search":
            return httpx.Response(
                200,
                json={"results": [{
                    "title": "结果",
                    "url": "https://example.com/page",
                    "snippet": "短",
                }]},
            )
        assert request.url.path == "/v1/tools/fetch"
        fetched.append(request)
        return httpx.Response(200, json=_FETCH_BODY)

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        results = await research_web(
            "细节问题", client, api_key="test-key", fetch_mode="auto"
        )

    assert fetched
    assert results[0].content.startswith("# 标题")
    assert results[0].fetched is True


@pytest.mark.asyncio
async def test_research_web_skips_fetch_for_rich_snippets():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/tools/fetch":
            pytest.fail("rich snippets should not trigger fetch")
        return httpx.Response(
            200,
            json={"results": [
                {"title": "一", "url": "https://example.com/1", "snippet": "长" * 200},
                {"title": "二", "url": "https://example.com/2", "snippet": "长" * 200},
            ]},
        )

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        results = await research_web(
            "普通问题", client, api_key="test-key", fetch_mode="auto"
        )

    assert len(results) == 2
    assert all(not result.fetched for result in results)


@pytest.mark.asyncio
async def test_research_web_fetch_mode_off_skips_fetch():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/tools/fetch":
            pytest.fail("fetch mode off must not fetch")
        return httpx.Response(
            200,
            json={"results": [{
                "title": "结果", "url": "https://example.com/page",
                "snippet": "短",
            }]},
        )

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        results = await research_web(
            "问题", client, api_key="test-key", fetch_mode="off"
        )

    assert results and not results[0].fetched


@pytest.mark.asyncio
async def test_research_web_fetch_failure_keeps_snippets():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/tools/search":
            return httpx.Response(
                200,
                json={"results": [{
                    "title": "结果", "url": "https://example.com/page",
                    "snippet": "短",
                }]},
            )
        return httpx.Response(500)

    async with _client(handler, base_url="https://api.deepseek.example") as client:
        results = await research_web(
            "细节问题", client, api_key="test-key", fetch_mode="always"
        )

    assert results[0].snippet == "短"
    assert results[0].fetched is False


@pytest.mark.asyncio
async def test_research_web_fetch_timeout_keeps_results(monkeypatch):
    async def slow_fetch(*args, **kwargs):
        import asyncio
        await asyncio.sleep(1)
        return None

    async def fake_search(*args, **kwargs):
        return (web_module.WebResult("结果", "https://example.com", "短"),)

    monkeypatch.setattr(web_module, "fetch_web", slow_fetch)
    monkeypatch.setattr(web_module, "search_web", fake_search)

    async with _client(
        lambda request: pytest.fail("network must not be called")
    ) as client:
        results = await research_web(
            "细节问题", client, api_key="test-key",
            fetch_mode="always", fetch_timeout=0.01,
        )

    assert results[0].url == "https://example.com"
    assert results[0].fetched is False
