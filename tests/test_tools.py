import httpx
import pytest

from research_system.config.settings import Settings
from research_system.core.models import Source
from research_system.tools import scraper as scraper_module
from research_system.tools.citations import (
    find_citation_ids,
    number_citations,
    reference_list,
    score_reliability,
    unknown_citations,
)
from research_system.tools.errors import ToolError
from research_system.tools.scraper import PageScraper
from research_system.tools.search import (
    DuckDuckGoSearch,
    SearchResult,
    TavilySearch,
    get_search_tool,
    search_many,
)


# ------------------------------------------------------------------ helpers
def html_response(ctype: str = "text/html; charset=utf-8", status: int = 200) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": ctype}, content=b"<html></html>")


def make_scraper(handler, **kwargs) -> PageScraper:
    return PageScraper(transport=httpx.MockTransport(handler), **kwargs)


def make_source(sid: str, url: str, title: str = "") -> Source:
    return Source(id=sid, sub_question_id="q1", url=url, title=title)


# ------------------------------------------------------------------ scraper
def test_scraper_returns_truncated_text(monkeypatch):
    monkeypatch.setattr(scraper_module.trafilatura, "extract", lambda *a, **k: "word " * 400)
    monkeypatch.setattr(scraper_module.trafilatura, "extract_metadata", lambda *a, **k: None)
    page = make_scraper(lambda request: html_response(), max_chars=500).fetch("https://x.org/a")
    assert 0 < len(page.text) <= 500
    assert page.title == "x.org"


def test_scraper_rejects_non_html():
    scraper = make_scraper(lambda request: html_response(ctype="application/pdf"))
    with pytest.raises(ToolError):
        scraper.fetch("https://x.org/file.pdf")


def test_scraper_wraps_http_errors():
    scraper = make_scraper(lambda request: html_response(status=404))
    with pytest.raises(ToolError):
        scraper.fetch("https://x.org/missing")


def test_scraper_rejects_empty_extraction(monkeypatch):
    monkeypatch.setattr(scraper_module.trafilatura, "extract", lambda *a, **k: None)
    with pytest.raises(ToolError):
        make_scraper(lambda request: html_response()).fetch("https://x.org/a")


# ------------------------------------------------------------------ search
def test_tavily_parses_results():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer key"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "https://a.org", "content": "snippet"},
                    {"title": "no url"},
                ]
            },
        )

    tool = TavilySearch("key", transport=httpx.MockTransport(handler))
    results = tool.search("q")
    assert [r.url for r in results] == ["https://a.org"]


def test_tavily_wraps_errors():
    tool = TavilySearch("key", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(ToolError):
        tool.search("q")


def test_get_search_tool_picks_backend_by_key():
    with_key = Settings(_env_file=None, tavily_api_key="k")
    without_key = Settings(_env_file=None, tavily_api_key="")
    assert isinstance(get_search_tool(with_key), TavilySearch)
    assert isinstance(get_search_tool(without_key), DuckDuckGoSearch)


def test_search_many_dedupes_and_collects_errors():
    class FakeTool:
        def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
            if query == "bad":
                raise ToolError("boom")
            return [
                SearchResult(title="A", url="https://a.org/x"),
                SearchResult(title="A again", url="https://A.org/x/"),
            ]

    results, errors = search_many(FakeTool(), ["good", "bad", "good2"])
    assert len(results) == 1
    assert len(errors) == 1
    assert "bad" in errors[0]


# ------------------------------------------------------------------ citations
def test_reliability_ordering():
    assert score_reliability("https://www.who.int/x") > score_reliability("https://blog.example.com")
    assert score_reliability("https://energy.gov/x") == 0.9
    assert score_reliability("https://www.reddit.com/r/x") < 0.5


def test_find_citation_ids_keeps_order_and_dedupes():
    text = "A [s_bbbbbbbb] B [s_aaaaaaaa] C [s_bbbbbbbb]"
    assert find_citation_ids(text) == ["s_bbbbbbbb", "s_aaaaaaaa"]


def test_number_citations_and_unknowns():
    sources = [
        make_source("s_aaaaaaaa", "https://a.org", "Paper A"),
        make_source("s_bbbbbbbb", "https://b.org", "Paper B"),
    ]
    text = "One [s_bbbbbbbb]. Two [s_aaaaaaaa]. Again [s_bbbbbbbb]. Ghost [s_cccccccc]."
    numbered, used = number_citations(text, sources)
    assert numbered == "One [1]. Two [2]. Again [1]. Ghost [s_cccccccc]."
    assert [s.id for s in used] == ["s_bbbbbbbb", "s_aaaaaaaa"]
    assert unknown_citations(text, sources) == ["s_cccccccc"]


def test_reference_list_format():
    refs = reference_list([make_source("s_aaaaaaaa", "https://a.org", "Paper A")])
    assert refs.startswith("[1] Paper A. https://a.org")