from pathlib import Path

import pytest
from langgraph.types import Command

from conftest import GOOD_DRAFT, FakeLLM, FakeSearch, critique, plan
from research_agent.graph import build_graph
from research_agent.search import FallbackSearch


def start(llm, search, settings, saver, config, question="A vs B"):
    graph = build_graph(llm, search, settings, checkpointer=saver)
    graph.invoke({"question": question}, config)
    return graph


def test_pauses_for_plan_approval_before_searching(settings, saver, config):
    search = FakeSearch()
    graph = start(FakeLLM([plan("What is A?", "What is B?")]), search, settings, saver, config)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert snapshot.values["plan"] == ["What is A?", "What is B?"]
    assert search.queries == []


def test_approved_plan_is_researched_in_parallel_and_report_saved(settings, saver, config):
    search = FakeSearch()
    llm = FakeLLM([plan("What is A?", "What is B?", "Speed?"), critique()])
    graph = start(llm, search, settings, saver, config)

    final = graph.invoke(Command(resume=True), config)

    assert sorted(search.queries) == ["Speed?", "What is A?", "What is B?"]
    assert len(final["findings"]) == 3
    path = Path(final["report_path"])
    text = path.read_text()
    assert path.name == "a-vs-b.md"
    assert text.startswith("# A vs B")
    assert "## Sources" in text and "https://example.com/what-is-a?" in text


def test_edited_plan_replaces_proposed_plan(settings, saver, config):
    search = FakeSearch()
    graph = start(FakeLLM([plan("Old question"), critique()]), search, settings, saver, config)

    final = graph.invoke(Command(resume={"plan": ["New question"]}), config)

    assert search.queries == ["New question"]
    assert final["plan"] == ["New question"]


def test_rewrite_verdict_loops_back_to_writer(settings, saver, config):
    llm = FakeLLM([plan("Q1"), critique(5, "rewrite"), critique()])
    graph = start(llm, FakeSearch(), settings, saver, config)

    final = graph.invoke(Command(resume=True), config)

    assert final["revisions"] == 1
    assert final["passed"] is True
    assert "Needs work" in llm.prompts[-2]  # second writer prompt includes critic feedback


def test_research_verdict_searches_follow_up_questions(settings, saver, config):
    search = FakeSearch()
    llm = FakeLLM([plan("Q1"), critique(5, "research", follow_ups=["Missing fact?"]), critique()])
    graph = start(llm, search, settings, saver, config)

    final = graph.invoke(Command(resume=True), config)

    assert search.queries == ["Q1", "Missing fact?"]
    assert {f["sub_question"] for f in final["findings"]} == {"Q1", "Missing fact?"}


def test_revision_cap_stops_loops_and_still_saves(settings, saver, config):
    llm = FakeLLM([plan("Q1")] + [critique(3, "rewrite")] * 3)
    graph = start(llm, FakeSearch(), settings, saver, config)

    final = graph.invoke(Command(resume=True), config)

    assert final["revisions"] == settings.max_revisions
    assert final["passed"] is False
    assert Path(final["report_path"]).exists()
    assert not llm.structured  # exactly 3 critic passes: initial + 2 revisions


def test_missing_table_fails_checklist_even_with_high_score(settings, saver, config):
    no_table = "## Summary\nA is fast [1].\n"
    llm = FakeLLM([plan("Q1"), critique(9, "pass"), critique()], texts=[no_table, GOOD_DRAFT])
    graph = start(llm, FakeSearch(), settings, saver, config)

    final = graph.invoke(Command(resume=True), config)

    assert final["revisions"] == 1
    assert final["passed"] is True


def test_fallback_search_uses_next_provider_when_first_fails():
    class Broken:
        def search(self, query, max_results):
            raise RuntimeError("quota exceeded")

    backup = FakeSearch()
    results = FallbackSearch(Broken(), backup).search("A vs B", 3)

    assert backup.queries == ["A vs B"]
    assert results[0]["url"] == "https://example.com/a-vs-b"


@pytest.mark.live
def test_live_end_to_end(tmp_path):
    from research_agent.config import Settings
    from research_agent.llm import make_llms
    from research_agent.search import make_search
    from langgraph.checkpoint.memory import InMemorySaver

    s = Settings(reports_dir=str(tmp_path))
    llm, critic = make_llms(s)
    graph = build_graph(llm, make_search(s), s, critic_llm=critic, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "live"}, "max_concurrency": s.max_concurrency}
    graph.invoke({"question": "FastAPI vs Flask for REST APIs"}, cfg)
    final = graph.invoke(Command(resume=True), cfg)
    assert Path(final["report_path"]).exists()
