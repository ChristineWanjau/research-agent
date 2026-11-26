"""Streamlit front end: `uv run streamlit run app.py`."""

import sqlite3
import uuid

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from research_agent.cache import setup_cache
from research_agent.config import Settings
from research_agent.graph import build_graph, run_config
from research_agent.llm import make_llms
from research_agent.rag import make_retriever
from research_agent.search import make_search

STEPS = {
    "planner": "🧭 Planning sub-questions",
    "researcher": "🔎 Researching",
    "writer": "✍️ Writing report",
    "critic": "🧐 Reviewing draft",
    "save_report": "💾 Saving report",
}

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")


@st.cache_resource
def get_graph():
    settings = Settings()
    setup_cache(settings)
    llm, critic_llm = make_llms(settings)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    graph = build_graph(
        llm,
        make_search(settings),
        settings,
        critic_llm=critic_llm,
        checkpointer=SqliteSaver(conn),
        retriever=make_retriever(settings),
    )
    return graph, settings


graph, settings = get_graph()
ss = st.session_state
ss.setdefault("thread_id", None)

st.title("🔎 Tech Comparison Research Agent")
st.caption("LangGraph · Groq · Tavily: plan → approve → parallel research → write → critique ↺")

with st.sidebar:
    st.subheader("Run")
    resume_id = st.text_input("Resume thread id", value=ss.thread_id or "")
    if st.button("Resume") and resume_id:
        ss.thread_id = resume_id
        st.rerun()
    if st.button("New run"):
        ss.thread_id = None
        ss.node_runs = {}
        st.rerun()
    if ss.thread_id:
        st.code(ss.thread_id, language=None)


def chunk_text(chunk) -> str:
    """Token text from a streamed message chunk (string or list-of-parts content)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


# Node colours: waiting (grey), running (amber), done (green), paused for a human (blue).
COLOURS = {"waiting": "#e5e7eb", "running": "#fbbf24", "done": "#86efac", "paused": "#93c5fd"}


def graph_dot(running: dict[str, int], runs: dict[str, int], paused: str | None = None, last: str | None = None) -> str:
    """Graphviz DOT for the compiled graph, coloured by each node's live state.

    running: node -> number of copies running right now (researchers run in parallel)
    runs:    node -> number of times the node has finished in this session
    """
    drawable = graph.get_graph()
    lines = [
        "digraph G {",
        "rankdir=LR; bgcolor=transparent;",
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        'edge [fontname="Helvetica", fontsize=9, color="#6b7280"];',
    ]
    for name in drawable.nodes:
        if name in ("__start__", "__end__"):
            label = "START" if name == "__start__" else "END"
            lines.append(f'"{name}" [label="{label}", shape=circle, fillcolor="#f3f4f6", width=0.5, fontsize=9];')
            continue
        if running.get(name):
            state = "running"
        elif name == paused:
            state = "paused"
        elif runs.get(name):
            state = "done"
        else:
            state = "waiting"
        label = name
        if running.get(name, 0) > 1:
            label += f"\\n×{running[name]} in parallel"
        elif runs.get(name, 0) > 1:
            label += f"\\nran {runs[name]}×"
        if state == "paused":
            label += "\\n⏸ waiting for you"
        pen = "penwidth=3" if state == "running" else "penwidth=1"
        lines.append(f'"{name}" [label="{label}", fillcolor="{COLOURS[state]}", {pen}];')
    for e in drawable.edges:
        style = "dashed" if e.conditional else "solid"
        hot = ', color="#f59e0b", penwidth=2.5' if last and e.source == last and running.get(e.target) else ""
        lines.append(f'"{e.source}" -> "{e.target}" [style={style}{hot}];')
    lines.append("}")
    return "\n".join(lines)


def run(payload, config):
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Graph (live)**")
        diagram = st.empty()
        st.caption("⬜ waiting · 🟨 running · 🟩 done · 🟦 paused for you · dashed = conditional edge")
    with right:
        status = st.status("Running agent…", expanded=True)
        live_header = st.empty()
        live_report = st.empty()

    runs: dict[str, int] = ss.setdefault("node_runs", {})
    running: dict[str, int] = {}
    last: str | None = None
    draft, draft_id, attempt = "", None, 0
    diagram.graphviz_chart(graph_dot(running, runs))

    # "tasks": an event when each node starts and finishes (drives the diagram).
    # "updates": each node's output (drives the step list).
    # "messages": LLM tokens as they arrive (drives the live report).
    for mode, event in graph.stream(payload, config, stream_mode=["tasks", "updates", "messages"]):
        if mode == "tasks":
            name = event["name"]
            if "input" in event:  # task started
                running[name] = running.get(name, 0) + 1
            else:  # task finished
                running[name] = max(running.get(name, 1) - 1, 0)
                if not event.get("interrupts"):
                    runs[name] = runs.get(name, 0) + 1
                last = name
            diagram.graphviz_chart(graph_dot(running, runs, last=last))
            continue

        if mode == "messages":
            chunk, meta = event
            if meta.get("langgraph_node") != "writer":
                continue
            if chunk.id != draft_id:  # a new writer call (first draft or a revision)
                draft_id, draft, attempt = chunk.id, "", attempt + 1
                live_header.subheader("✍️ Writing report" + (f" (revision {attempt - 1})" if attempt > 1 else ""))
            draft += chunk_text(chunk)
            live_report.markdown(draft + " ▌")
            continue

        for node, data in event.items():
            if node == "__interrupt__":
                diagram.graphviz_chart(graph_dot({}, runs, paused="human_review"))
                status.update(label="Waiting for plan approval", state="complete")
                return
            label = STEPS.get(node, node)
            if node == "researcher" and data:
                label += f": {data['findings'][0]['sub_question']}"
            if node == "critic" and data:
                c = data["critique"]
                label += f": score {c['score']}/10 → {c['verdict']}"
            status.write(f"✓ {label}")
    diagram.graphviz_chart(graph_dot({}, runs))
    status.update(label="Done", state="complete")


if not ss.thread_id:
    question = st.text_input("What should I compare?", placeholder="LangGraph vs CrewAI for multi-agent apps")
    if st.button("Start research", type="primary", disabled=not question):
        ss.thread_id = str(uuid.uuid4())
        ss.node_runs = {}
        run({"question": question}, run_config(ss.thread_id, settings))
        st.rerun()
    with st.expander("How the graph works", expanded=True):
        st.graphviz_chart(graph_dot({}, {}))
    st.stop()

config = run_config(ss.thread_id, settings)
snapshot = graph.get_state(config)
values = snapshot.values
node_runs = ss.get("node_runs", {})

if "human_review" in snapshot.next:
    st.graphviz_chart(graph_dot({}, node_runs or {"planner": 1}, paused="human_review"))
    st.subheader(f"Plan for: {values['question']}")
    st.write("Edit, add or remove sub-questions (one per line), then approve.")
    edited = st.text_area("Sub-questions", value="\n".join(values["plan"]), height=180)
    if st.button("Approve plan", type="primary"):
        plan = [q.strip() for q in edited.splitlines() if q.strip()]
        run(Command(resume={"plan": plan}), config)
        st.rerun()
elif values.get("report_path"):
    crit = values.get("critique", {})
    cols = st.columns(3)
    cols[0].metric("Critic score", f"{crit.get('score', '–')}/10")
    cols[1].metric("Revisions", values.get("revisions", 0))
    cols[2].metric("Sources", len({s["url"] for f in values.get("findings", []) for s in f["sources"]}))
    st.download_button("Download Markdown", values["draft"], file_name=values["report_path"].split("/")[-1])
    if node_runs:
        with st.expander("Path this run took through the graph"):
            st.graphviz_chart(graph_dot({}, node_runs))
    st.markdown(values["draft"])
elif snapshot.next:
    st.info(f"Run paused at: {', '.join(snapshot.next)}")
    if st.button("Continue run"):
        run(None, config)
        st.rerun()
else:
    st.warning("No run found for this thread id.")
