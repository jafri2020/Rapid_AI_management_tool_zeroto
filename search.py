import time
from typing import List, Dict

from config import config


class SearchClient:
    def __init__(self, cfg=None):
        # Per-run config snapshot; falls back to the global singleton.
        self._cfg = cfg or config
        self._ddgs = None

    def _get_ddgs(self):
        if self._ddgs is None:
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
        return self._ddgs

    def search(self, query: str, max_results: int | None = None) -> List[Dict]:
        if not self._cfg.enable_search:
            return []
        n = max_results or self._cfg.search_results_per_query
        results = []
        try:
            for r in self._get_ddgs().text(query, max_results=n):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                )
            time.sleep(self._cfg.search_delay_seconds)
        except Exception as e:
            print(f"[search] '{query}': {e}")
        return results

    def multi_search(self, queries: List[str]) -> str:
        seen_urls: set = set()
        all_results: List[Dict] = []
        for q in queries:
            for r in self.search(q):
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        return self._format(all_results)

    @staticmethod
    def _format(results: List[Dict]) -> str:
        if not results:
            return "No search results available."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n{r['body']}\nURL: {r['url']}")
        return "\n\n".join(lines)
