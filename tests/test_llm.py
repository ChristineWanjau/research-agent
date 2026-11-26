"""Stage 1 LangChain features: provider switching, fallbacks and the response cache. No API keys needed."""

from langchain_core.globals import set_llm_cache
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from research_agent.cache import SQLiteLLMCache
from research_agent.config import Settings
from research_agent.llm import WithFallback, make_llms


class RateLimited(GenericFakeChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("429 rate limit")


class CountingModel(GenericFakeChatModel):
    calls: int = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return super()._generate(*args, **kwargs)


def reply(text):
    return GenericFakeChatModel(messages=iter([AIMessage(text)]))


def test_fallback_answers_when_primary_is_rate_limited():
    llm = WithFallback(RateLimited(messages=iter([])), reply("from backup"))
    assert llm.invoke("hi").content == "from backup"


def test_fallback_keeps_structured_output_working():
    class Plan(BaseModel):
        steps: list[str]

    class FakeStructured:
        def __init__(self, result):
            self.result = result

        def with_structured_output(self, schema):
            from langchain_core.runnables import RunnableLambda

            def run(_):
                if isinstance(self.result, Exception):
                    raise self.result
                return self.result

            return RunnableLambda(run)

    llm = WithFallback(FakeStructured(RuntimeError("429")), FakeStructured(Plan(steps=["a"])))
    assert llm.with_structured_output(Plan).invoke("q") == Plan(steps=["a"])


def test_make_llms_switches_provider_from_settings():
    s = Settings(_env_file=None, llm_provider="ollama", llm_model="llama3.2", fallback_provider=None)
    main, critic = make_llms(s)
    assert type(main).__name__ == "ChatOllama" and main.model == "llama3.2"
    assert critic is main


def test_make_llms_adds_backup_only_when_its_key_is_set():
    base = dict(
        _env_file=None,
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
        critic_model="llama-3.1-8b-instant",
        groq_api_key="gsk-test",
        google_api_key="",
        fallback_provider="gemini",
    )
    without_key, _ = make_llms(Settings(**base))
    with_key, critic = make_llms(Settings(**{**base, "google_api_key": "g-test"}))

    assert not isinstance(without_key, WithFallback)
    assert isinstance(with_key, WithFallback) and isinstance(critic, WithFallback)
    assert type(with_key.backup).__name__ == "ChatGoogleGenerativeAI"
    assert with_key.primary.model_name == "llama-3.3-70b-versatile"
    assert critic.primary.model_name == "llama-3.1-8b-instant"


def test_cache_returns_stored_answer_without_calling_model(tmp_path):
    set_llm_cache(SQLiteLLMCache(str(tmp_path / "cache.sqlite")))
    try:
        model = CountingModel(messages=iter([AIMessage("first"), AIMessage("second")]))
        prompt = ChatPromptTemplate.from_template("Compare {a}").invoke({"a": "X and Y"})
        assert model.invoke(prompt).content == "first"
        assert model.invoke(prompt).content == "first"
        assert model.calls == 1
    finally:
        set_llm_cache(None)
