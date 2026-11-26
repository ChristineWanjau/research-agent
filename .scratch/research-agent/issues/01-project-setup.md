# 01: Project setup, settings and fake providers

**What to build:** A `uv` project that installs, loads settings from `.env`, and has fake LLM and search providers so tests run without keys.

**Blocked by:** None (can start immediately)

**Status:** done

- [ ] `uv sync` installs everything; `uv run pytest` passes
- [ ] Settings load from `.env` with sensible defaults; `.env.example` lists every variable
- [ ] LLM factory builds Groq, Gemini or Ollama models from settings
- [ ] Fake chat model and fake search provider exist for tests
