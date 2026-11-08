"""Fetch a web page and reduce it to clean article text."""
from __future__ import annotations

from urllib.parse import urlparse

import httpx
import trafilatura
from pydantic import BaseModel

from research_system.tools.errors import ToolError

USER_AGENT = "Mozilla/5.0 (compatible; ResearchSystemBot/0.1; educational project)"
MIN_TEXT_CHARS = 100


class FetchedPage(BaseModel):
    url: str
    title: str
    text: str


class PageScraper:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        max_chars: int = 12_000,
        max_bytes: int = 2_000_000,
        timeout: float = 20.0,
    ) -> None:
        self.max_chars = max_chars
        self.max_bytes = max_bytes
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=timeout,
            transport=transport,
        )

    def fetch(self, url: str) -> FetchedPage:
        try:
            with self._http.stream("GET", url) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").lower()
                if ctype and "html" not in ctype and "text" not in ctype:
                    raise ToolError(f"Unsupported content type '{ctype}' at {url}")
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > self.max_bytes:
                        break
                html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
        except httpx.HTTPError as e:
            raise ToolError(f"Could not fetch {url}: {e}") from e

        text = trafilatura.extract(
            html, include_comments=False, include_tables=False, favor_precision=True
        )
        if not text or len(text) < MIN_TEXT_CHARS:
            raise ToolError(f"No readable article text at {url}")

        if len(text) > self.max_chars:
            text = text[: self.max_chars].rsplit(" ", 1)[0]

        meta = trafilatura.extract_metadata(html)
        title = (meta.title if meta and meta.title else "") or (urlparse(url).hostname or url)
        return FetchedPage(url=url, title=title.strip(), text=text)