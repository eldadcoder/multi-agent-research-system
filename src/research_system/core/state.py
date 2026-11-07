"""Shared pipeline state. Every agent reads from and writes to this."""
import operator
from typing import Annotated, TypeVar

from typing_extensions import TypedDict

from research_system.core.models import (
    AnalysisResult,
    CritiqueResult,
    Finding,
    ResearchPlan,
    Source,
)

T = TypeVar("T")


def merge_by_id(left: list[T] | None, right: list[T] | None) -> list[T]:
    """Reducer: merge two lists of objects with an `id`, newest wins.

    Needed because parallel searchers write to the same key, and loop-backs
    may rediscover sources we already have.
    """
    merged = {item.id: item for item in (left or [])}  # type: ignore[attr-defined]
    for item in right or []:
        merged[item.id] = item  # type: ignore[attr-defined]
    return list(merged.values())


class ResearchState(TypedDict, total=False):
    question: str
    plan: ResearchPlan
    sources: Annotated[list[Source], merge_by_id]
    findings: Annotated[list[Finding], merge_by_id]
    analysis: AnalysisResult
    critique: CritiqueResult
    draft: str
    final_report: str
    iteration: int
    errors: Annotated[list[str], operator.add]


def new_state(question: str) -> ResearchState:
    return ResearchState(
        question=question,
        sources=[],
        findings=[],
        errors=[],
        iteration=0,
    )