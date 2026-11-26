"""Graph entry point for LangGraph Studio (`uv run langgraph dev`). Studio supplies its own checkpointer."""

from research_agent.cache import setup_cache
from research_agent.config import Settings
from research_agent.graph import build_graph
from research_agent.llm import make_llms
from research_agent.search import make_search

_settings = Settings()
setup_cache(_settings)
_llm, _critic = make_llms(_settings)
graph = build_graph(_llm, make_search(_settings), _settings, critic_llm=_critic)
