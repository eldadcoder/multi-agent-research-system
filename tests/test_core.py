import json

import httpx
import pytest

from research_system.config.settings import Settings
from research_system.core.llm import LLMError, OllamaClient
from research_system.core.models import ResearchPlan, Source, make_source_id
from research_system.core.state import merge_by_id

PLAN = {
    "objective": "Study X",
    "sub_questions": [
        {"id": "weird", "question": "What is X?", "search_queries": ["what is X"]}
    ],
}


def client_with(replies: list[str]) -> OllamaClient:
    """An OllamaClient whose server returns the given replies in order."""
    it = iter(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": next(it)},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    return OllamaClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))


def test_plan_ids_are_always_reassigned():
    plan = ResearchPlan.model_validate(PLAN)
    assert plan.sub_questions[0].id == "q1"


def test_structured_returns_valid_output():
    client = client_with([json.dumps(PLAN)])
    plan = client.structured("sys", "user", ResearchPlan)
    assert plan.objective == "Study X"
    assert client.usage.calls == 1


def test_structured_retries_after_invalid_output():
    client = client_with(["not json at all", json.dumps(PLAN)])
    plan = client.structured("sys", "user", ResearchPlan)
    assert plan.sub_questions[0].id == "q1"
    assert client.usage.calls == 2
    assert client.usage.prompt_tokens == 20


def test_structured_gives_up_after_max_attempts():
    client = client_with(["nope"] * 3)
    with pytest.raises(LLMError):
        client.structured("sys", "user", ResearchPlan)


def test_source_id_is_stable():
    assert make_source_id("https://x.org/a") == make_source_id("https://X.org/a/")


def test_merge_by_id_dedupes_and_newest_wins():
    sid = make_source_id("https://x.org/a")
    old = Source(id=sid, sub_question_id="q1", url="https://x.org/a")
    new = old.model_copy(update={"title": "newer"})
    merged = merge_by_id([old], [new])
    assert len(merged) == 1
    assert merged[0].title == "newer"