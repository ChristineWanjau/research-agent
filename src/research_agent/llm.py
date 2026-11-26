"""Builds chat models for the configured provider, with an optional backup provider."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from research_agent.config import Settings

# Our setting names -> LangChain's provider names.
PROVIDERS = {"groq": "groq", "gemini": "google_genai", "ollama": "ollama"}


def _api_key(settings: Settings, provider: str) -> str | None:
    return {"groq": settings.groq_api_key, "gemini": settings.google_api_key}.get(provider)


def make_llm(settings: Settings, provider: str, model: str, max_retries: int = 5) -> BaseChatModel:
    kwargs: dict = {}
    if key := _api_key(settings, provider):
        kwargs["api_key"] = key
    if provider != "ollama":
        kwargs["max_retries"] = max_retries
    return init_chat_model(model, model_provider=PROVIDERS[provider], temperature=0.2, **kwargs)


class WithFallback:
    """A primary model plus a backup used when the primary errors (e.g. a rate limit).

    Fallbacks are attached per call, because a fallback chain built on the raw models
    no longer supports `with_structured_output`.
    """

    def __init__(self, primary, backup):
        self.primary, self.backup = primary, backup

    def invoke(self, prompt):
        return self.primary.with_fallbacks([self.backup]).invoke(prompt)

    def with_structured_output(self, schema):
        return self.primary.with_structured_output(schema).with_fallbacks([self.backup.with_structured_output(schema)])

    def bind_tools(self, tools):
        return self.primary.bind_tools(tools).with_fallbacks([self.backup.bind_tools(tools)])


def make_llms(settings: Settings):
    """Return (main_llm, critic_llm). The critic model only applies to Groq."""
    backup_provider = settings.fallback_provider
    use_backup = (
        backup_provider is not None
        and backup_provider != settings.llm_provider
        and (backup_provider == "ollama" or bool(_api_key(settings, backup_provider)))
    )
    # With a backup available, give up on the primary sooner instead of waiting out long retries.
    retries = 2 if use_backup else 5

    main = make_llm(settings, settings.llm_provider, settings.llm_model, retries)
    critic_name = settings.critic_model if settings.llm_provider == "groq" else None
    critic = make_llm(settings, settings.llm_provider, critic_name, retries) if critic_name else main

    if use_backup:
        backup = make_llm(settings, backup_provider, settings.fallback_model)
        main, critic = WithFallback(main, backup), WithFallback(critic, backup)
    return main, critic
