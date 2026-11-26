# Tickets: LangGraph Research Agent

Spec: [spec.md](spec.md). Work top to bottom; each ticket is blocked by the one before it.

Verified so far: `uv run pytest` (8 passed) and one real CLI run with Ollama `qwen3:8b` + Tavily, which produced [reports/sample-report.md](../../reports/sample-report.md) (critic score 8/10, passed first time).

## 01: Project setup, settings and fake providers
**Blocked by:** None · **Status:** done
- [x] `uv sync` installs everything; `uv run pytest` passes
- [x] Settings load from `.env`; `.env.example` lists every variable
- [x] LLM factory builds Groq, Gemini or Ollama models (only Ollama tested for real)
- [x] Fake chat model and fake search provider for tests

## 02: Planner → writer graph from the CLI
**Blocked by:** 01 · **Status:** done
- [x] Planner returns 3–5 sub-questions (structured output)
- [x] Writer produces summary, comparison table, sections, Sources
- [x] Report saved to `reports/<slug>.md`
- [x] Test through `build_graph` with fakes

## 03: Parallel research with Tavily and DuckDuckGo fallback
**Blocked by:** 02 · **Status:** done
- [x] `SearchProvider` with Tavily, DuckDuckGo and fallback implementations (DuckDuckGo not tested for real)
- [x] `Send` fan-out, max 2 concurrent, 3 results each, ~1,500 chars per source
- [x] Sources de-duplicated and numbered; writer cites `[n]`
- [x] Tests: findings merge; fallback used when Tavily fails

## 04: Critic with rewrite and research loops
**Blocked by:** 03 · **Status:** done
- [x] Critic returns checklist, score, verdict, follow-up questions
- [x] Passes only if checklist passes and score ≥ 7
- [x] "rewrite" → writer; "research" → researchers (covered by fake tests; the real run passed without looping)
- [x] Shared cap of 2 revisions; tests cover each route and the cap

## 05: Plan approval and SQLite persistence
**Blocked by:** 04 · **Status:** mostly done
- [x] `interrupt()` after planner; resume with approved or edited plan
- [x] CLI approve/edit prompt
- [ ] CLI resume by thread id: written but not tried yet
- [x] SQLite checkpointer
- [x] Test: resume with an edited plan researches the edited plan

## 06: Streamlit UI
**Blocked by:** 05 · **Status:** written, not run
- [ ] Question input, live step status, editable plan with approve button: code compiles, not opened in a browser yet
- [ ] Report streams in as it is written: not built; the report appears when finished
- [ ] Report can be downloaded: written, not tried

## 07: README, sample report and LangGraph Studio
**Blocked by:** 06 · **Status:** mostly done
- [ ] `langgraph.json` works with `langgraph dev`: graph loads, Studio not opened yet
- [x] README: Mermaid diagram, setup, features demonstrated, GIF placeholder
- [x] Sample report in `reports/`
