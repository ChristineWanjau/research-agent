from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env into os.environ so libraries that read env vars directly
# (e.g. LangSmith tracing: LANGSMITH_*) see them. Existing env vars win.
load_dotenv()


class Settings(BaseSettings):
    """All runtime settings, loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["groq", "gemini", "ollama"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    critic_model: str | None = "llama-3.1-8b-instant"
    # Backup provider used when the main one errors (e.g. rate limits). Needs its API key.
    fallback_provider: Literal["groq", "gemini", "ollama"] | None = "gemini"
    fallback_model: str = "gemini-2.5-flash"

    # Cache LLM responses on disk. Development only: re-runs reuse answers instead of spending quota.
    llm_cache: bool = False
    llm_cache_path: str = ".llm_cache.sqlite"

    groq_api_key: str | None = None
    google_api_key: str | None = None
    tavily_api_key: str | None = None

    max_revisions: int = 2
    pass_score: int = 7  # critic score needed to pass; raise to 9 or 10 to force revisions
    max_results: int = 3
    max_source_chars: int = 1500
    max_concurrency: int = 2
    # Searches the researcher may choose per sub-question. 0 = always search the sub-question as written.
    research_tool_calls: int = 2

    reports_dir: str = "reports"
    db_path: str = ".checkpoints.sqlite"
