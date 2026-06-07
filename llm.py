from openai import OpenAI
from config import config


def _kimi_client(cfg) -> OpenAI:
    return OpenAI(api_key=cfg.kimi_api_key, base_url=cfg.kimi_base_url)


def _ollama_client(cfg) -> OpenAI:
    # Ollama exposes an OpenAI-compatible endpoint at /v1
    return OpenAI(api_key="ollama", base_url=f"{cfg.ollama_base_url}/v1")


class LLMClient:
    def __init__(self, cfg=None):
        # Per-run config snapshot; falls back to the global singleton so existing
        # callers that don't pass one keep working.
        self._cfg = cfg or config
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if self._cfg.llm_provider == "kimi":
                self._client = _kimi_client(self._cfg)
            elif self._cfg.llm_provider == "ollama":
                self._client = _ollama_client(self._cfg)
            else:
                raise ValueError(f"Unknown LLM provider: {self._cfg.llm_provider!r}. Use 'kimi' or 'ollama'.")
        return self._client

    def _model(self) -> str:
        if self._cfg.llm_provider == "ollama":
            return self._cfg.ollama_model
        return self._cfg.model  # kimi: moonshot-v1-8k / 32k / 128k

    # Hard ceiling for the length-retry escalation below — keeps a runaway
    # think-loop from requesting an absurd budget on every doubling.
    _MAX_TOKENS_CEILING = 32000

    def _call(self, system: str, actual_user: str, max_tokens: int, temperature: float):
        client = self._get_client()
        try:
            return client.chat.completions.create(
                model=self._model(),
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": actual_user},
                ],
                temperature=temperature,
            )
        except Exception as e:
            raise RuntimeError(
                f"LLM call failed (provider={self._cfg.llm_provider}, model={self._model()}): {type(e).__name__}: {e}"
            ) from e

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        # qwen3 on Ollama emits large <think> blocks that eat the token budget;
        # /no_think disables thinking at the prompt level (works across Ollama versions)
        actual_user = (
            f"/no_think\n{user}"
            if self._cfg.llm_provider == "ollama" and self._cfg.ollama_disable_thinking
            else user
        )

        # kimi-k2 only accepts temperature=1
        temperature = 1 if self._cfg.llm_provider == "kimi" and self._model().startswith("kimi-k2") else 0.3

        # Thinking models occasionally burn the whole budget on reasoning and return
        # empty content with finish_reason=length. When that happens, retry with a
        # doubled budget (up to the ceiling) before giving up. A non-empty-but-truncated
        # response is returned as-is — downstream json_repair salvages partial JSON.
        attempt_tokens = max_tokens
        last_finish_reason = "unknown"
        while True:
            response = self._call(system, actual_user, attempt_tokens, temperature)

            if not response.choices:
                raise RuntimeError(f"LLM returned no choices. Raw response: {response}")
            content = response.choices[0].message.content
            if content:
                return content

            last_finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
            if last_finish_reason == "length" and attempt_tokens < self._MAX_TOKENS_CEILING:
                attempt_tokens = min(attempt_tokens * 2, self._MAX_TOKENS_CEILING)
                continue
            break

        raise RuntimeError(
            f"LLM returned empty content after escalating to max_tokens={attempt_tokens}. "
            f"finish_reason={last_finish_reason}. "
            f"This often means the budget was exhausted by reasoning or the model refused. "
            f"Provider={self._cfg.llm_provider}, model={self._model()}."
        )
