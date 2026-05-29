from openai import OpenAI
from config import config


def _kimi_client() -> OpenAI:
    return OpenAI(api_key=config.kimi_api_key, base_url=config.kimi_base_url)


def _ollama_client() -> OpenAI:
    # Ollama exposes an OpenAI-compatible endpoint at /v1
    return OpenAI(api_key="ollama", base_url=f"{config.ollama_base_url}/v1")


class LLMClient:
    def __init__(self):
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if config.llm_provider == "kimi":
                self._client = _kimi_client()
            elif config.llm_provider == "ollama":
                self._client = _ollama_client()
            else:
                raise ValueError(f"Unknown LLM provider: {config.llm_provider!r}. Use 'kimi' or 'ollama'.")
        return self._client

    def _model(self) -> str:
        if config.llm_provider == "ollama":
            return config.ollama_model
        return config.model  # kimi: moonshot-v1-8k / 32k / 128k

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self._model(),
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
            )
        except Exception as e:
            raise RuntimeError(
                f"LLM call failed (provider={config.llm_provider}, model={self._model()}): {type(e).__name__}: {e}"
            ) from e

        if not response.choices:
            raise RuntimeError(f"LLM returned no choices. Raw response: {response}")
        content = response.choices[0].message.content
        if not content:
            finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
            raise RuntimeError(
                f"LLM returned empty content. finish_reason={finish_reason}. "
                f"This often means max_tokens was hit or the model refused. "
                f"Provider={config.llm_provider}, model={self._model()}."
            )
        return content
