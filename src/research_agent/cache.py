"""Optional on-disk cache for LLM responses, so re-runs during development don't spend quota."""

import json
import sqlite3
import threading
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumps, loads

from research_agent.config import Settings


class SQLiteLLMCache(BaseCache):
    """Stores generations keyed by (prompt, model settings) in a local SQLite file."""

    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS llm_cache (prompt TEXT, llm TEXT, value TEXT, PRIMARY KEY (prompt, llm))")

    def lookup(self, prompt: str, llm_string: str) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM llm_cache WHERE prompt=? AND llm=?", (prompt, llm_string)).fetchone()
        return [loads(g, allowed_objects="core") for g in json.loads(row[0])] if row else None

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        value = json.dumps([dumps(g) for g in return_val])
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?)", (prompt, llm_string, value))

    def clear(self, **kwargs: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM llm_cache")


def setup_cache(settings: Settings) -> None:
    """Turn on the cache only when LLM_CACHE=true. Real runs get fresh answers by default."""
    if settings.llm_cache:
        set_llm_cache(SQLiteLLMCache(settings.llm_cache_path))
