"""Protocols: agents depend on these, so engines and fakes are swappable."""
from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from research_system.core.models import AnalysisResult, CritiqueResult
from research_system.core.state import ResearchState

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def structured(
        self, system: str, user: str, schema: type[T], *, model: str | None = None
    ) -> T: ...

    def text(self, system: str, user: str, *, model: str | None = None) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Analyst(Protocol):
    """Implemented by the CrewAI crew (and a plain LangGraph version for comparison)."""

    def run(self, state: ResearchState) -> AnalysisResult: ...


class Critic(Protocol):
    """Implemented by the AutoGen debate (and a plain version for comparison)."""

    def run(self, state: ResearchState) -> CritiqueResult: ...