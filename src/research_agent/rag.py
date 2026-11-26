"""Retrieval: fetch full pages behind search results, chunk, embed locally, and keep the most relevant chunks."""

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_agent.config import Settings
from research_agent.search import Source

log = logging.getLogger(__name__)


def fetch_page(url: str, timeout: float = 10.0) -> str:
    """Download a page and return its visible text. Returns '' on any failure."""
    import httpx
    from bs4 import BeautifulSoup

    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "research-agent/0.1"})
        resp.raise_for_status()
        if "html" not in resp.headers.get("content-type", ""):
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        return " ".join(soup.get_text(" ").split())
    except Exception as exc:
        log.warning("Could not fetch %s: %s", url, exc)
        return ""


class Retriever:
    """Turns a sub-question's search results into the chunks most relevant to it."""

    def __init__(self, embeddings: Embeddings, settings: Settings, fetch=fetch_page):
        self._embeddings = embeddings
        self._settings = settings
        self._fetch = fetch
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    def retrieve(self, question: str, sources: list[Source]) -> list[Source]:
        """Return up to `retrieval_k` sources whose content is the best-matching chunks.

        Falls back to the original snippets if nothing could be fetched or embedded.
        """
        if not sources:
            return sources
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                pages = list(pool.map(self._fetch, [s["url"] for s in sources]))
            docs = []
            for src, page in zip(sources, pages):
                text = page or src["content"]  # unfetchable page: keep the search snippet
                docs.extend(self._splitter.create_documents([text], metadatas=[{"title": src["title"], "url": src["url"]}]))
            store = InMemoryVectorStore(self._embeddings)
            store.add_documents(docs)
            hits: list[Document] = store.similarity_search(question, k=self._settings.retrieval_k)
        except Exception as exc:  # e.g. Ollama not running
            log.warning("Retrieval failed for %r, using search snippets: %s", question, exc)
            return sources
        # Several chunks can come from one page; merge them so each URL is one citable source.
        merged: dict[str, Source] = {}
        for d in hits:
            url = d.metadata["url"]
            if url in merged:
                merged[url]["content"] += "\n…\n" + d.page_content
            else:
                merged[url] = Source(title=d.metadata["title"], url=url, content=d.page_content)
        return list(merged.values())


def make_retriever(settings: Settings) -> Retriever | None:
    if not settings.rag:
        return None
    from langchain_ollama import OllamaEmbeddings

    return Retriever(OllamaEmbeddings(model=settings.embedding_model), settings)
