"""Base agent with REASON → SEARCH → GENERATE → CRITIQUE pipeline."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Type, TypeVar

from pydantic import BaseModel

from config import config
from llm import LLMClient
from search import SearchClient

T = TypeVar("T", bound=BaseModel)

SEARCH_EXTRACTION_PROMPT = """
At the end of your analysis, output a JSON block of 2-3 targeted web search queries that would
produce the most useful market data, competitor pricing, or user research evidence for this idea.
Format:
```json
{"search_queries": ["query 1", "query 2", "query 3"]}
```
"""


class BaseAgent:
    name: str = "BaseAgent"

    def __init__(self, llm: LLMClient, search: SearchClient):
        self.llm = llm
        self.search = search

    # ── LLM steps ────────────────────────────────────────────────────────────

    def _reason(self, system: str, user: str) -> str:
        return self.llm.complete(system, user, max_tokens=config.reason_max_tokens)

    def _generate(self, system: str, user: str) -> str:
        return self.llm.complete(system, user, max_tokens=config.generate_max_tokens)

    def _critique(self, system: str, output_text: str) -> str:
        prompt = (
            "Review this JSON output for vagueness, unsupported claims, or missing evidence. "
            "Flag anything that needs strengthening. Be specific.\n\n"
            f"Output to review:\n{output_text}"
        )
        return self.llm.complete(system, prompt, max_tokens=config.critique_max_tokens)

    # ── Search ────────────────────────────────────────────────────────────────

    def _extract_queries(self, text: str) -> List[str]:
        match = re.search(r'```json\s*(\{.*?"search_queries".*?\})\s*```', text, re.DOTALL)
        if not match:
            match = re.search(r'\{[^{}]*"search_queries"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1) if match.lastindex else match.group())
                return data.get("search_queries", [])[:3]
            except Exception:
                pass
        return []

    # ── JSON parsing pipeline ─────────────────────────────────────────────────

    def _parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError(f"[{self.name}] Empty text passed to JSON parser.")

        # 0. Strip common prose wrappers (Kimi/Qwen/DeepSeek often add these)
        stripped = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
        stripped = re.sub(r"<reasoning>.*?</reasoning>", "", stripped, flags=re.DOTALL | re.IGNORECASE)
        stripped = stripped.strip()

        # 1. Direct parse
        try:
            return json.loads(stripped)
        except Exception:
            pass

        # 2. Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", stripped).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 3. Extract outermost { } block (greedy)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        # 4. Repair common issues (trailing commas, Python literals, smart quotes)
        repaired = re.sub(r",\s*([}\]])", r"\1", cleaned)
        repaired = repaired.replace("True", "true").replace("False", "false").replace("None", "null")
        repaired = repaired.replace("“", '"').replace("”", '"')  # smart quotes
        repaired = repaired.replace("‘", "'").replace("’", "'")
        match2 = re.search(r"\{.*\}", repaired, re.DOTALL)
        if match2:
            try:
                return json.loads(match2.group())
            except Exception:
                pass
        try:
            return json.loads(repaired)
        except Exception:
            pass

        # 5. json_repair fallback
        try:
            import json_repair  # type: ignore
            result = json_repair.repair_json(stripped, return_objects=True)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        raise ValueError(
            f"[{self.name}] Could not parse JSON from LLM output. "
            f"First 600 chars of raw output:\n{text[:600]}"
        )

    def _validate_output(self, data: Dict[str, Any], model_class: Type[T]) -> T:
        return model_class.model_validate(data)

    # ── Full pipeline helper ──────────────────────────────────────────────────

    def run_pipeline(
        self,
        system: str,
        reason_prompt: str,
        generate_prompt_template: str,
        model_class: Type[T],
    ) -> T:
        """
        1. REASON  — free-form scratchpad + search query extraction
        2. SEARCH  — execute extracted queries
        3. GENERATE — structured JSON output
        4. CRITIQUE — optional self-audit (logged, not used to modify output)
        """
        # Step 1: Reason
        full_reason_prompt = reason_prompt + "\n\n" + SEARCH_EXTRACTION_PROMPT
        reasoning = self._reason(system, full_reason_prompt)

        # Step 2: Search
        queries = self._extract_queries(reasoning)
        search_context = self.search.multi_search(queries) if queries else "Search disabled or no queries generated."

        # Step 3: Generate
        gen_prompt = generate_prompt_template.format(
            reasoning=reasoning,
            search_results=search_context,
        )
        raw_output = self._generate(system, gen_prompt)

        # Step 4: Critique (log only)
        if config.enable_critique:
            try:
                self._critique(system, raw_output)
            except Exception:
                pass

        data = self._parse_json(raw_output)
        data["search_queries_used"] = queries
        return self._validate_output(data, model_class)
