"""Command-line front end: `uv run research-agent "LangGraph vs CrewAI for multi-agent apps"`."""

import argparse
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from research_agent.config import Settings
from research_agent.graph import build_graph, run_config
from research_agent.llm import make_llms
from research_agent.search import make_search


def review_plan(plan: list[str]) -> dict | bool:
    print("\nProposed plan:")
    for i, q in enumerate(plan, 1):
        print(f"  {i}. {q}")
    answer = input("\nPress Enter to approve, or type 'e' to edit: ").strip().lower()
    if answer != "e":
        return True
    print("Enter one sub-question per line. Blank line to finish.")
    edited = []
    while line := input("> ").strip():
        edited.append(line)
    return {"plan": edited} if edited else True


def stream(graph, payload, config) -> list[str] | None:
    """Run until the graph ends or pauses. Returns the plan if paused for review."""
    for update in graph.stream(payload, config, stream_mode="updates"):
        for node, data in update.items():
            if node == "__interrupt__":
                return data[0].value["plan"]
            print(f"✓ {node}")
            if node == "critic" and data:
                c = data["critique"]
                print(f"   score {c['score']}/10, verdict: {c['verdict']}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Research a tech comparison question.")
    parser.add_argument("question", nargs="?", help="e.g. 'LangGraph vs CrewAI for multi-agent apps'")
    parser.add_argument("--thread", help="Resume a saved run by thread id")
    args = parser.parse_args()

    settings = Settings()
    llm, critic_llm = make_llms(settings)
    search = make_search(settings)

    with SqliteSaver.from_conn_string(settings.db_path) as saver:
        graph = build_graph(llm, search, settings, critic_llm=critic_llm, checkpointer=saver)
        thread_id = args.thread or str(uuid.uuid4())
        config = run_config(thread_id, settings)
        print(f"Thread: {thread_id}")

        if args.thread:
            snapshot = graph.get_state(config)
            if not snapshot.next:
                print("That run already finished:", snapshot.values.get("report_path"))
                return
            plan = snapshot.values.get("plan") if "human_review" in snapshot.next else None
            payload = Command(resume=review_plan(plan)) if plan else None
        else:
            if not args.question:
                parser.error("a question is required unless --thread is given")
            payload = {"question": args.question}

        while (plan := stream(graph, payload, config)) is not None:
            payload = Command(resume=review_plan(plan))

        final = graph.get_state(config).values
        print(f"\nReport saved to {final['report_path']}")


if __name__ == "__main__":
    main()
