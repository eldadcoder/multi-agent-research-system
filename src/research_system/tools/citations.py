"""Citation helpers: source reliability heuristic, [s_id] markers -> [1], reference list."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from research_system.core.models import Source

CITATION_RE = re.compile(r"\[(s_[0-9a-f]{8})\]")

HIGH_TRUST_DOMAINS = {
    "arxiv.org", "nature.com", "science.org", "who.int", "worldbank.org",
    "un.org", "oecd.org", "ieee.org", "acm.org", "nih.gov", "europa.eu",
}  # fmt: skip
LOW_TRUST_DOMAINS = {
    "reddit.com", "quora.com", "pinterest.com", "medium.com", "facebook.com", "x.com",
}  # fmt: skip


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _matches(host: str, domains: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def score_reliability(url: str) -> float:
    """Rough 0-1 heuristic from the domain only. A starting signal, not a verdict."""
    host = _host(url)
    if _matches(host, HIGH_TRUST_DOMAINS):
        return 0.9
    if host.endswith((".gov", ".edu")) or ".gov." in host or ".edu." in host or ".ac." in host:
        return 0.9
    if _matches(host, LOW_TRUST_DOMAINS):
        return 0.3
    if host.endswith(".org"):
        return 0.7
    return 0.5


def find_citation_ids(text: str) -> list[str]:
    """Citation ids in order of first appearance, without duplicates."""
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def unknown_citations(text: str, sources: list[Source]) -> list[str]:
    known = {s.id for s in sources}
    return [sid for sid in find_citation_ids(text) if sid not in known]


def number_citations(markdown: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """Replace [s_xxxxxxxx] with [1], [2]... by first appearance.

    Returns the new text and the used sources in numbered order.
    Unknown ids are left untouched so unknown_citations() can flag them.
    """
    by_id = {s.id: s for s in sources}
    order: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        sid = match.group(1)
        if sid not in by_id:
            return match.group(0)
        if sid not in order:
            order[sid] = len(order) + 1
        return f"[{order[sid]}]"

    numbered = CITATION_RE.sub(replace, markdown)
    return numbered, [by_id[sid] for sid in order]


def reference_list(used: list[Source]) -> str:
    lines = [
        f"[{i}] {s.title or s.url}. {s.url} (retrieved {s.retrieved_at:%Y-%m-%d})"
        for i, s in enumerate(used, start=1)
    ]
    return "\n".join(lines)