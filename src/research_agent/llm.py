"""Builds chat models for the configured free-tier provider."""

from langchain_core.language_models import BaseChatModel

from research_agent.config import Settings


def make_llm(settings: Settings, model: str | None = None) -> BaseChatModel:
    name = model or settings.llm_model
    match settings.llm_provider:
        case "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(model=name, api_key=settings.groq_api_key, temperature=0.2, max_retries=5)
        case "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=name, google_api_key=settings.google_api_key, temperature=0.2, max_retries=5
            )
        case "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=name, temperature=0.2)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def make_llms(settings: Settings) -> tuple[BaseChatModel, BaseChatModel]:
    """Return (main_llm, critic_llm). The critic model only applies to Groq."""
    main = make_llm(settings)
    critic_name = settings.critic_model if settings.llm_provider == "groq" else None
    critic = make_llm(settings, critic_name) if critic_name else main
    return main, critic
