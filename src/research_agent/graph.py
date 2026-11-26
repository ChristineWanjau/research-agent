"""The research graph: plan → approve → research (parallel) → write → critique ↺ → save."""

import re
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from slugify import slugify

from research_agent.config import Settings
from research_agent.search import SearchProvider, Source
from research_agent.state import Critique, Finding, Plan, ResearchState, ResearchTask

PLANNER_PROMPT = """You plan research for a technology comparison report.
Break the question into 3 to 5 focused sub-questions that can each be answered with a web search.
Cover: what each option is, key features, performance or scalability, ecosystem and community, and when to choose which.

Question: {question}"""

WRITER_PROMPT = """You write a technology comparison report in Markdown, using ONLY the numbered sources below.

Question: {question}

Sub-questions:
{plan}

Sources:
{sources}
{feedback}
Write the report with exactly this structure:
## Summary
(3-5 sentences answering the question)
## Comparison
(a Markdown table comparing the options)
## <one section per sub-question, using the sub-question as the heading>

Rules:
- Cite every factual claim inline with the source number, like [2].
- Do not invent facts or cite numbers that are not in the source list.
- Do NOT write a Sources section; it is added automatically."""

CRITIC_PROMPT = """You review a technology comparison report.

Question: {question}
Sub-questions:
{plan}

Report:
{draft}

Check: does every sub-question have an answered section? Is every claim cited? Is it accurate and useful?
Give a score from 1 to 10. Verdict: "pass" if the score is 7 or more and nothing important is missing;
"rewrite" if the writing, structure or citations need work; "research" if facts are missing,
in which case give 1 to 3 new searchable follow_up_questions."""


def number_sources(findings: list[Finding]) -> list[Source]:
    """De-duplicate sources by URL, keeping first-seen order, so citation numbers are stable."""
    seen: dict[str, Source] = {}
    for f in findings:
        for s in f["sources"]:
            seen.setdefault(s["url"], s)
    return list(seen.values())


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{i}] {s['title']} ({s['url']})\n{s['content']}" for i, s in enumerate(sources, 1))


def bullet(items: list[str]) -> str:
    return "\n".join(f"- {q}" for q in items)


def has_table(text: str) -> bool:
    return bool(re.search(r"^\|.*\|\s*\n\|[\s:|-]+\|", text, re.MULTILINE))


def has_citations(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text))


def build_graph(llm, search: SearchProvider, settings: Settings, critic_llm=None, checkpointer=None):
    critic_llm = critic_llm or llm

    def planner(state: ResearchState) -> dict:
        plan = llm.with_structured_output(Plan).invoke(PLANNER_PROMPT.format(question=state["question"]))
        subs = [q.strip() for q in plan.sub_questions if q.strip()][:5]
        return {"plan": subs, "revisions": 0}

    def human_review(state: ResearchState) -> dict:
        # Pauses the run. Resume with Command(resume={"plan": [...]}) or Command(resume=True) to accept.
        answer = interrupt({"question": state["question"], "plan": state["plan"]})
        plan = state["plan"]
        if isinstance(answer, dict) and answer.get("plan"):
            plan = [q.strip() for q in answer["plan"] if q.strip()]
        return {"plan": plan, "pending": plan}

    def fan_out(state: ResearchState) -> list[Send]:
        return [Send("researcher", ResearchTask(sub_question=q)) for q in state["pending"]]

    def researcher(task: ResearchTask) -> dict:
        sources = search.search(task["sub_question"], settings.max_results)
        return {"findings": [Finding(sub_question=task["sub_question"], sources=sources)]}

    def writer(state: ResearchState) -> dict:
        sources = number_sources(state.get("findings", []))
        feedback = ""
        if crit := state.get("critique"):
            feedback = "\nA reviewer found these problems in your previous draft; fix them:\n" + bullet(crit["issues"]) + "\n"
        prompt = WRITER_PROMPT.format(
            question=state["question"],
            plan=bullet(state["plan"]),
            sources=format_sources(sources) or "(no sources found)",
            feedback=feedback,
        )
        return {"draft": llm.invoke(prompt).content}

    def critic(state: ResearchState) -> dict:
        draft = state["draft"]
        c: Critique = critic_llm.with_structured_output(Critique).invoke(
            CRITIC_PROMPT.format(question=state["question"], plan=bullet(state["plan"]), draft=draft)
        )
        issues = list(c.issues)
        if not has_table(draft):
            issues.append("The comparison table is missing.")
        if not has_citations(draft):
            issues.append("Claims have no [n] citations.")
        checklist_ok = c.all_sub_questions_answered and has_table(draft) and has_citations(draft)
        passed = checklist_ok and c.score >= settings.pass_score
        verdict = "pass" if passed else ("research" if c.verdict == "research" and c.follow_up_questions else "rewrite")
        loop_again = not passed and state.get("revisions", 0) < settings.max_revisions
        update: dict = {
            "critique": {**c.model_dump(), "issues": issues, "verdict": verdict},
            "passed": passed,
            "loop_again": loop_again,
        }
        if loop_again:
            update["revisions"] = state.get("revisions", 0) + 1
            if verdict == "research":
                update["pending"] = c.follow_up_questions[:3]
        return update

    def route_after_critic(state: ResearchState):
        if not state.get("loop_again"):
            return "save_report"
        if state["critique"]["verdict"] == "research":
            return fan_out(state)
        return "writer"

    def save_report(state: ResearchState) -> dict:
        sources = number_sources(state.get("findings", []))
        refs = "\n".join(f"{i}. [{s['title'] or s['url']}]({s['url']})" for i, s in enumerate(sources, 1))
        report = f"# {state['question']}\n\n{state['draft'].strip()}\n\n## Sources\n{refs}\n"
        out_dir = Path(settings.reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{slugify(state['question'])[:80] or 'report'}.md"
        path.write_text(report, encoding="utf-8")
        return {"report_path": str(path), "draft": report}

    g = StateGraph(ResearchState)
    g.add_node("planner", planner)
    g.add_node("human_review", human_review)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_node("save_report", save_report)

    g.add_edge(START, "planner")
    g.add_edge("planner", "human_review")
    g.add_conditional_edges("human_review", fan_out, ["researcher"])
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["writer", "researcher", "save_report"])
    g.add_edge("save_report", END)
    return g.compile(checkpointer=checkpointer)


def run_config(thread_id: str, settings: Settings) -> dict:
    return {"configurable": {"thread_id": thread_id}, "max_concurrency": settings.max_concurrency}
