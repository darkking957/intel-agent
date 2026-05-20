import logging
from collections.abc import Iterable

from openai import OpenAI

from intel_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

Message = dict[str,any]

class LLMClient:
    """
    Minimal OpenAI-compatible LLM client.
    """

    def __init__(self,settings:Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
        )
    
    def chat(
            self,
            messages: list[Message],
            *,
            temperature: float = 0.2,
            max_tokens: int = 512,
            response_format: dict[str,any] | None = None,
            tools: list[dict[str,any]] | None = None,
            tool_choice: str | list[dict[str,any]] | None = None,
    ):
        return self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

    def stream_text(
            self,
            messages: list[Message],
            *,
            temperature: float = 0.2,
            max_tokens: int = 512,
    ) ->Iterable[str]:
        stream = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta :
                yield delta