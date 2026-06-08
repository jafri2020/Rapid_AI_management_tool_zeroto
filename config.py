import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM provider: "kimi" or "ollama"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "kimi"))

    # Kimi (Moonshot AI) — OpenAI-compatible
    kimi_api_key: str = field(default_factory=lambda: os.getenv("KIMI_API_KEY", ""))
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    model: str = field(default_factory=lambda: os.getenv("MODEL", "kimi-k2.6"))

    # Ollama — local models
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))

    # Token budgets per pipeline step
    # kimi-k2 is a thinking model — internal reasoning tokens consume the budget before
    # the visible output is written. 8192 was exhausted entirely by think-blocks.
    # GENERATE has the same problem: reasoning-heavy agents (economics, scale) burned
    # through 4096 on think-blocks before emitting any JSON → finish_reason=length.
    reason_max_tokens: int = 16000
    generate_max_tokens: int = 16000
    critique_max_tokens: int = 4096

    # Ollama: disable built-in thinking (qwen3 <think> blocks eat the token budget;
    # our reason prompts already do explicit step-by-step analysis)
    ollama_disable_thinking: bool = field(default_factory=lambda: os.getenv("OLLAMA_DISABLE_THINKING", "true").lower() == "true")

    # Search — Tavily only (agent-grade results + extracted page content).
    # No fallback provider: if TAVILY_API_KEY is unset or a call fails, the query
    # returns nothing and the gap is logged.
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    search_results_per_query: int = 5
    # Tavily depth: "basic" (cheaper, snippet-level) or "advanced" (richer extracts).
    search_depth: str = field(default_factory=lambda: os.getenv("SEARCH_DEPTH", "advanced").lower())
    # Recency window for results: one of day/week/month/year, or "" for no limit.
    # Market data is time-sensitive, so default to the past year.
    search_time_range: str = field(default_factory=lambda: os.getenv("SEARCH_TIME_RANGE", "year").lower())
    # Cap per-result page content fed to the LLM so the prompt isn't drowned in noise.
    search_max_content_chars: int = 2000
    # Per-query network retries (exponential backoff) before giving up / falling back.
    search_max_retries: int = 2
    enable_search: bool = field(default_factory=lambda: os.getenv("ENABLE_SEARCH", "true").lower() == "true")

    # Pipeline toggles
    enable_critique: bool = field(default_factory=lambda: os.getenv("ENABLE_CRITIQUE", "true").lower() == "true")

    # Storage
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "idea_engine.db"))

    # Gate thresholds
    # Out of 20 — below this, skip DE research. Aligned with the "park" floor in
    # TriageOutput.recompute_total (>=5 park, <5 reject) so anything the triage
    # step classifies as "reject" is gated out instead of burning the full pipeline.
    triage_gate_min: int = 5
    ltv_coca_min_ratio: float = 3.0   # hard gate: external score capped at 55 if below
    data_readiness_floor: int = 30    # feasibility dimension capped if data readiness < 30

    # Decision band thresholds (applied to composite score)
    build_now_min: int = 80
    prototype_min: int = 65
    park_min: int = 50

    # Quick Win flag: score >= 65 AND mvp <= this many days
    quick_win_mvp_days: int = 7

    # Strategic Bet flag: score >= 55 AND secret >= 4 AND timing >= 4
    strategic_bet_min_score: int = 55
    strategic_bet_timing_min: int = 4


config = Config()
