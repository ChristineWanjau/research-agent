# Spec: LangGraph Research Agent

**Status:** ready-for-agent

## Problem Statement

I want a portfolio project that shows I can design and build a real multi-step agent with LangGraph, not just a single LLM call. It has to run on free tiers only (no credit card), be easy for anyone who clones the repo to run, and visibly demonstrate LangGraph's core features: typed state, conditional edges, cycles, parallel fan-out, human-in-the-loop, and persistence.

## Solution

A research agent specialised in **tech comparisons** (e.g. "LangGraph vs CrewAI for multi-agent apps"). Given a question it:

1. **Plans** 3–5 sub-questions.
2. **Pauses** so the user can approve or edit the plan.
3. **Researches** every sub-question in parallel using web search.
4. **Writes** a Markdown report with a summary, a comparison table, one section per sub-question, and inline `[n]` citations to a numbered source list.
5. **Critiques** the report against a checklist and a score. If it fails, the critic sends the run back to the writer (writing problems) or to the researchers (missing facts), at most 2 times in total.
6. **Saves** the report to `reports/<slug>.md`.

It runs from a CLI, a Streamlit UI, and LangGraph Studio. Runs are saved in SQLite so they can be resumed.

## User Stories

1. As a user, I want to type a tech comparison question, so that I get a researched report without searching myself.
2. As a user, I want the agent to break my question into 3–5 sub-questions, so that the report covers the topic systematically.
3. As a user, I want to see the plan before any searching happens, so that I don't waste quota on a bad plan.
4. As a user, I want to approve, edit, add, or remove sub-questions, so that the research matches what I care about.
5. As a user, I want sub-questions researched in parallel, so that runs finish quickly.
6. As a user, I want every claim in the report to cite a numbered source, so that I can check the facts.
7. As a user, I want a comparison table in the report, so that I can see the differences at a glance.
8. As a user, I want a short summary at the top, so that I get the answer without reading everything.
9. As a user, I want the agent to review its own draft and fix problems, so that the final report is better than a first draft.
10. As a user, I want the critic to fetch more research when facts are missing, so that gaps get filled rather than glossed over.
11. As a user, I want revisions capped, so that a run can't loop forever or burn my free quota.
12. As a user, I want the report saved as a Markdown file, so that I can keep and share it.
13. As a user, I want to resume a run after closing the app, so that I don't lose work when it pauses for approval.
14. As a user, I want to run it from the command line, so that I can use it without a UI.
15. As a user, I want a Streamlit UI that shows which step is running, so that I can follow the agent's progress.
16. As a user, I want the report to stream into the UI as it is written, so that the demo feels live.
17. As a developer, I want to open the graph in LangGraph Studio, so that I can inspect state and step back through runs.
18. As a developer, I want Tavily search with an automatic DuckDuckGo fallback, so that the project runs even without a Tavily key.
19. As a developer, I want to choose the LLM provider (Groq, Gemini, Ollama) in `.env`, so that I can switch when I hit a free-tier limit.
20. As a developer, I want automatic retry with backoff on rate-limit errors, so that free-tier limits don't crash a run.
21. As a developer, I want sources trimmed before they reach the writer, so that a run stays within Groq's token limits.
22. As a developer, I want unit tests that use fake LLM and search providers, so that tests run without keys or quota.
23. As a developer, I want an opt-in live test, so that I can check the real integrations when I choose to.
24. As a portfolio reviewer, I want a README with a graph diagram, a demo GIF, a sample report and a list of LangGraph features used, so that I can judge the project quickly.

## Implementation Decisions

- **Modules**
  - **config**: loads settings from `.env` (provider, model names, keys, limits). One source of truth for limits.
  - **llm factory**: builds a LangChain chat model for the configured provider (Groq by default; Gemini and Ollama optional). Separate model names are allowed for writer and critic.
  - **search**: a `SearchProvider` interface with one method, `search(query, max_results) -> list[Source]`. Implementations: Tavily, DuckDuckGo, and a fallback wrapper that tries Tavily then DuckDuckGo. Content is trimmed to about 1,500 characters per source.
  - **state**: the typed graph state.
  - **nodes**: planner, human review, researcher, writer, critic, save report.
  - **graph**: `build_graph(llm, search, checkpointer, settings)`, the single seam for tests and all front ends.
  - **front ends**: CLI, Streamlit app, and a `langgraph.json` entry for Studio.
- **State shape**: question; plan (list of sub-questions); findings (sub-question, list of sources), accumulated with a reducer so parallel researchers merge; draft report; critique (passed, score, issues, verdict, follow-up questions); revision count; final report path.
- **Graph**: START → planner → human review (`interrupt`) → fan-out with `Send` (one researcher per sub-question) → writer → critic → conditional route:
  - passed, or revision limit reached → save report → END
  - verdict "rewrite" → writer
  - verdict "research" → fan-out to researchers with the critic's follow-up questions
- **Structured output**: the planner and critic return Pydantic models via `with_structured_output`.
- **Citations**: sources are de-duplicated by URL and numbered globally before the writer sees them; the writer must cite as `[n]`.
- **Critic rule**: passes only if the checklist passes (every sub-question addressed, claims cited, comparison table present) AND score ≥ 7. Revision limit is 2, shared by both loops.
- **Limits**: 3–5 sub-questions, 3 results per search, at most 2 searches running at once (`max_concurrency`).
- **Rate limits**: chat models are wrapped with retry and exponential backoff.
- **Persistence**: `SqliteSaver` in the CLI and Streamlit; Studio supplies its own checkpointer.
- **Report format**: Summary → Comparison table → one section per sub-question → numbered Sources.

## Testing Decisions

- Tests check **external behaviour through the one seam** (`build_graph`), not node internals: given a question, fake LLM output and fake search results, what final state and report come out?
- Fake LLM: a scripted chat model that returns planned structured outputs and text in order. Fake search: returns fixed sources.
- Covered behaviour: plan → interrupt → resume with an edited plan; parallel research merges all findings; critic "rewrite" loops to writer; critic "research" loops to researchers; revision cap stops loops; report file is written; search fallback switches to DuckDuckGo when Tavily fails.
- One live test marked `live`, skipped by default, runs the real graph end to end.
- No prior art yet; this is a new repository.

## Out of Scope

- Paid APIs and LangSmith.
- Docker, cloud deployment, auth, multi-user support.
- PDF or JSON output.
- Topics outside tech comparisons (the agent will still try, but prompts are tuned for comparisons).
- Pausing for approval of the final report.

## Further Notes

- Free-tier limits change; numbers in this spec are as of November 2026.
- DuckDuckGo has no official API and may rate-limit or break; it is the fallback, not the default.
