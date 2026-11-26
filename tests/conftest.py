"""Fake LLM and search providers so tests run with no keys and no quota."""

import os

os.environ["LANGSMITH_TRACING"] = "false"  # never send test runs to LangSmith

from collections import deque  # noqa: E402
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.config import Settings
from research_agent.state import Critique, Plan

GOOD_DRAFT = """## Summary
A is faster than B [1].
## Comparison
| | A | B |
|---|---|---|
| Speed | fast [1] | slow [2] |
## What is A?
A is a tool [1].
"""


class FakeLLM:
    """Returns scripted answers in order. Structured calls pop from `structured`, text calls from `texts`."""

    def __init__(self, structured=(), texts=()):
        self.structured = deque(structured)
        self.texts = deque(texts)
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        parent = self

        class _Runner:
            def invoke(self, prompt):
                parent.prompts.append(prompt)
                result = parent.structured.popleft()
                assert isinstance(result, schema), f"expected {schema.__name__}, got {type(result).__name__}"
                return result

        return _Runner()

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.texts.popleft() if self.texts else GOOD_DRAFT)


class FakeSearch:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, max_results):
        self.queries.append(query)
        slug = query.lower().replace(" ", "-")
        return [{"title": f"About {query}", "url": f"https://example.com/{slug}", "content": f"Facts on {query}."}]


def plan(*qs) -> Plan:
    return Plan(sub_questions=list(qs))


def critique(score=8, verdict="pass", answered=True, follow_ups=()) -> Critique:
    return Critique(
        all_sub_questions_answered=answered,
        score=score,
        verdict=verdict,
        issues=[] if verdict == "pass" else ["Needs work"],
        follow_up_questions=list(follow_ups),
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, reports_dir=str(tmp_path / "reports"), max_revisions=2)


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "t1"}}


@pytest.fixture
def saver():
    return InMemorySaver()
