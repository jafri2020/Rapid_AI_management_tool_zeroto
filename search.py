import time
from typing import List, Dict

import requests

from config import config


class SearchClient:
    """Web search for DE research, backed solely by the Tavily API.

    Tavily returns ranked results plus extracted page content — not just snippets.
    There is no fallback provider: if the API key is missing or the call fails after
    retries, the query returns no results and the failure is logged loudly so the
    agent's evidence gap is visible rather than masked.
    """

    TAVILY_URL = "https://api.tavily.com/search"

    def __init__(self, cfg=None):
        # Per-run config snapshot; falls back to the global singleton.
        self._cfg = cfg or config
        if self._cfg.enable_search and not self._cfg.tavily_api_key:
            print("[search] WARNING: TAVILY_API_KEY not set — searches will return nothing.")

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int | None = None) -> List[Dict]:
        if not self._cfg.enable_search:
            return []
        if not self._cfg.tavily_api_key:
            return []
        n = max_results or self._cfg.search_results_per_query
        try:
            return self._search_tavily(query, n)
        except Exception as e:
            print(f"[search] tavily '{query}': {e}")
            return []

    def multi_search(self, queries: List[str]) -> str:
        seen_urls: set = set()
        all_results: List[Dict] = []
        for q in queries:
            for r in self.search(q):
                if r["url"] and r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        return self._format(all_results)

    # ── Tavily ──────────────────────────────────────────────────────────────────

    def _post_with_retry(self, url: str, *, headers: Dict, json: Dict) -> Dict:
        last_exc: Exception | None = None
        for attempt in range(self._cfg.search_max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=json, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # network error, 4xx/5xx, bad JSON
                last_exc = e
                if attempt < self._cfg.search_max_retries:
                    time.sleep(2 ** attempt)  # 1s, 2s, ...
        raise last_exc  # type: ignore[misc]

    def _search_tavily(self, query: str, n: int) -> List[Dict]:
        body: Dict = {
            "query": query,
            "max_results": n,
            "search_depth": self._cfg.search_depth,
            "include_raw_content": True,
        }
        if self._cfg.search_time_range:
            body["time_range"] = self._cfg.search_time_range
        data = self._post_with_retry(
            self.TAVILY_URL,
            headers={"Authorization": f"Bearer {self._cfg.tavily_api_key}"},
            json=body,
        )
        results = []
        for r in data.get("results", []):
            # Prefer Tavily's curated extract (clean, LLM-oriented); fall back to the
            # raw page dump only when no extract is available.
            content = r.get("content") or r.get("raw_content") or ""
            results.append(
                {
                    "title": r.get("title", ""),
                    "body": r.get("content", ""),
                    "content": self._truncate(content),
                    "url": r.get("url", ""),
                    "score": r.get("score"),
                }
            )
        return results

    # ── Formatting ──────────────────────────────────────────────────────────────

    def _truncate(self, text: str) -> str:
        limit = self._cfg.search_max_content_chars
        text = (text or "").strip()
        return text if len(text) <= limit else text[:limit].rstrip() + " …[truncated]"

    @staticmethod
    def _format(results: List[Dict]) -> str:
        if not results:
            return "No search results available."
        lines = []
        for i, r in enumerate(results, 1):
            # Prefer extracted page content; fall back to the snippet.
            detail = r.get("content") or r.get("body", "")
            lines.append(f"{i}. {r['title']}\n{detail}\nURL: {r['url']}")
        return "\n\n".join(lines)
