"""Stage 3: retrieval over fetched pages. Uses a fake embedder and fake fetcher, so no network or Ollama."""

from langchain_core.embeddings import Embeddings
from langgraph.types import Command

from conftest import FakeLLM, FakeSearch, critique, plan
from research_agent.graph import build_graph
from research_agent.rag import Retriever


class KeywordEmbeddings(Embeddings):
    """Embeds text as counts of a few keywords, so similarity is predictable."""

    WORDS = ["speed", "price", "community"]

    def _vec(self, text):
        t = text.lower()
        return [float(t.count(w)) + 0.01 for w in self.WORDS]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


PAGES = {
    "https://a.com": "price " * 50 + "community " * 50,
    "https://b.com": "speed " * 80,
}


def make_retriever(settings, fetch=lambda url: PAGES.get(url, "")):
    settings.retrieval_k = 1
    settings.chunk_size = 200
    settings.chunk_overlap = 0
    return Retriever(KeywordEmbeddings(), settings, fetch=fetch)


def src(url, content="snippet"):
    return {"title": url, "url": url, "content": content}


def test_returns_most_relevant_chunk_from_fetched_pages(settings):
    hits = make_retriever(settings).retrieve("What is the speed?", [src("https://a.com"), src("https://b.com")])

    assert len(hits) == 1
    assert hits[0]["url"] == "https://b.com"
    assert "speed" in hits[0]["content"]


def test_chunks_from_one_page_are_merged_into_one_source(settings):
    retriever = make_retriever(settings)
    settings.retrieval_k = 3

    hits = retriever.retrieve("speed", [src("https://b.com")])

    assert [h["url"] for h in hits] == ["https://b.com"]
    assert hits[0]["content"].count("…") == 2


def test_unfetchable_page_falls_back_to_search_snippet(settings):
    hits = make_retriever(settings, fetch=lambda url: "").retrieve("speed", [src("https://x.com", "speed benchmark")])

    assert hits == [src("https://x.com", "speed benchmark")]


def test_embedding_failure_returns_original_sources(settings):
    class Broken(KeywordEmbeddings):
        def embed_documents(self, texts):
            raise ConnectionError("Ollama not running")

    sources = [src("https://a.com")]
    retriever = Retriever(Broken(), settings, fetch=lambda url: "text")

    assert retriever.retrieve("speed", sources) == sources


def test_graph_passes_retrieved_chunks_to_writer(settings, saver, config):
    class PageSearch(FakeSearch):
        def search(self, query, max_results):
            self.queries.append(query)
            return [src("https://a.com"), src("https://b.com")]

    llm = FakeLLM([plan("How is the speed?"), critique()])
    graph = build_graph(llm, PageSearch(), settings, checkpointer=saver, retriever=make_retriever(settings))
    graph.invoke({"question": "A vs B"}, config)
    final = graph.invoke(Command(resume=True), config)

    sources = final["findings"][0]["sources"]
    assert [s["url"] for s in sources] == ["https://b.com"]
    writer_prompt = llm.prompts[1]
    assert "speed speed" in writer_prompt and "snippet" not in writer_prompt
