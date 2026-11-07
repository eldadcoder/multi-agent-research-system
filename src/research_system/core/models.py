"""Domain models shared by every agent and every framework adapter.

Two kinds of models live here:
  * LLM-facing schemas: small and flat, with descriptions; the model must produce these.
  * Domain objects: richer; ids, timestamps and long text are filled in by OUR code.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


def make_source_id(url: str) -> str:
    """Stable id from the URL, so the same page found twice dedupes to one source."""
    normalized = url.strip().rstrip("/").lower()
    return "s_" + hashlib.sha1(normalized.encode()).hexdigest()[:8]


def make_finding_id(source_id: str, index: int) -> str:
    return f"f_{source_id[2:]}_{index}"


class UsageStats(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: UsageStats) -> UsageStats:
        return UsageStats(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

class SubQuestion(BaseModel):
    id: str = Field(default="", description="Leave empty. Assigned automatically.")
    question: str = Field(description="A specific, answerable research sub-question.")
    search_queries: list[str] = Field(
        min_length=1,
        max_length=4,
        description="1-4 short web search queries that would help answer it.",
    )


class ResearchPlan(BaseModel):
    objective: str = Field(description="One-sentence restatement of the research goal.")
    sub_questions: list[SubQuestion] = Field(
        min_length=1,
        max_length=6,
        description="3-5 sub-questions that together cover the objective.",
    )

    @model_validator(mode="after")
    def _assign_ids(self) -> ResearchPlan:
        # Small models are unreliable at ids, so we always overwrite them.
        for i, sq in enumerate(self.sub_questions, start=1):
            sq.id = f"q{i}"
        return self


class Source(BaseModel):
    id: str
    sub_question_id: str
    url: str
    title: str = ""
    text: str = ""  # cleaned page text; can be long, never produced by the LLM
    retrieved_at: datetime = Field(default_factory=utcnow)
    reliability: float = Field(default=0.5, ge=0, le=1)



class ExtractedFact(BaseModel):  # LLM-facing
    claim: str = Field(description="One factual claim, in your own words.")
    quote: str = Field(
        description="The exact sentence copied from the text that supports the claim."
    )
    confidence: float = Field(ge=0, le=1, description="0 to 1: how well the text supports it.")


class ExtractionResult(BaseModel):  # LLM-facing
    facts: list[ExtractedFact] = Field(
        max_length=8, description="Up to 8 facts relevant to the sub-question."
    )


class Finding(BaseModel):  # domain
    id: str
    sub_question_id: str
    source_id: str
    claim: str
    quote: str
    confidence: float = Field(ge=0, le=1)


class Conflict(BaseModel):
    description: str = Field(description="What the sources disagree about.")
    finding_ids: list[str] = Field(description="Ids of the findings that conflict.")


class AnalysisResult(BaseModel):
    summary: str = Field(description="A short synthesis of what the evidence says.")
    key_points: list[str] = Field(description="The most important conclusions.")
    conflicts: list[Conflict] = Field(default_factory=list)
    gaps: list[str] = Field(
        default_factory=list, description="Questions the evidence does not answer."
    )


class CritiqueIssue(BaseModel):
    severity: Literal["low", "medium", "high"]
    description: str
    finding_id: str | None = None


class CritiqueResult(BaseModel):
    issues: list[CritiqueIssue] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, description="Overall confidence in the evidence.")
    needs_more_research: bool = Field(
        description="True if important gaps or weak claims justify another search round."
    )
    follow_up_queries: list[str] = Field(
        default_factory=list, description="Search queries to fill the gaps."
    )



class Report(BaseModel):
    title: str
    markdown: str