"""Stage 2: the researcher chooses its own searches through a LangChain tool. No API keys needed."""

from langchain_core.messages import AIMessage
from langgraph.types import Command

from conftest import FakeLLM, FakeSearch, critique, plan
from research_agent.graph import build_graph


class ToolCallingLLM(FakeLLM):
    """FakeLLM that, once bound to tools, replies with scripted tool calls."""

    def __init__(self, structured=(), tool_replies=()):
        super().__init__(structured)
        self.tool_replies = list(tool_replies)
        self.bound: list[str] = []

    def bind_tools(self, tools):
        self.bound = [t.name for t in tools]
        parent = self

        class _Bound:
            def invoke(self, messages):
                return parent.tool_replies.pop(0) if parent.tool_replies else AIMessage("done")

        return _Bound()


def search_call(query, i=0):
    return AIMessage("", tool_calls=[{"name": "web_search", "args": {"query": query}, "id": f"call_{i}"}])


def run(llm, search, settings, saver, config):
    graph = build_graph(llm, search, settings, checkpointer=saver)
    graph.invoke({"question": "A vs B"}, config)
    return graph.invoke(Command(resume=True), config)


def test_researcher_runs_the_queries_the_model_chooses(settings, saver, config):
    search = FakeSearch()
    llm = ToolCallingLLM([plan("Speed?"), critique()], [search_call("A benchmark"), search_call("B benchmark", 1)])

    final = run(llm, search, settings, saver, config)

    assert llm.bound == ["web_search"]
    assert search.queries == ["A benchmark", "B benchmark"]
    urls = [s["url"] for s in final["findings"][0]["sources"]]
    assert urls == ["https://example.com/a-benchmark", "https://example.com/b-benchmark"]


def test_tool_call_limit_is_enforced(settings, saver, config):
    settings.research_tool_calls = 1
    search = FakeSearch()
    llm = ToolCallingLLM([plan("Speed?"), critique()], [search_call("first"), search_call("second", 1)])

    run(llm, search, settings, saver, config)

    assert search.queries == ["first"]


def test_falls_back_to_direct_search_when_model_makes_no_calls(settings, saver, config):
    search = FakeSearch()
    llm = ToolCallingLLM([plan("Speed?"), critique()], [AIMessage("I already know this.")])

    run(llm, search, settings, saver, config)

    assert search.queries == ["Speed?"]


def test_falls_back_to_direct_search_when_tool_calling_errors(settings, saver, config):
    search = FakeSearch()
    llm = ToolCallingLLM([plan("Speed?"), critique()])
    llm.bind_tools = lambda tools: (_ for _ in ()).throw(RuntimeError("model does not support tools"))

    run(llm, search, settings, saver, config)

    assert search.queries == ["Speed?"]


def test_tools_disabled_searches_sub_question_directly(settings, saver, config):
    settings.research_tool_calls = 0
    search = FakeSearch()
    llm = ToolCallingLLM([plan("Speed?"), critique()], [search_call("ignored")])

    run(llm, search, settings, saver, config)

    assert search.queries == ["Speed?"]
