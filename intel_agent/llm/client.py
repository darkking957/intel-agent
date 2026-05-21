import logging
from openai import AsyncOpenAI,APIStatusError,APIConnectionError
from tenacity import(
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from intel_agent.config import settings

logger = logging.getLogger(__name__)

"全局client"
_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)

def _should_retry(exc: BaseException) ->bool :
    if isinstance(exc,APIStatusError):
        return exc.status_code
    if isinstance(exc,APIConnectionError):
        return True
    return False

@retry(
    retry=retry_if_exception_type((APIConnectionError,APIStatusError),),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1,min=2,max=30),
    before_sleep=before_sleep_log(logger,logging.WARNING),
    reraise=True,
)

async def chat_complete(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    response_format: dict|None = None,
) ->str :
    """
    基础 Chat Completion 调用。
    返回 assistant message 的文本内容。
    遇到限速/5xx 自动重试最多 3 次。
    """
    kwargs = {
        "model": settings.deepseek_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format
    
    response = await _client.chat.completions.create(**kwargs)

    #记录token 用量
    usage = response.usage
    if usage:
        logger.info(
            "LLM call completed | prompt=%d  completion=%d  total=%d",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
    return response.choices[0].message.content

async def chat_stream(messages: list[dict], temperature: float = 0.7):
    """
    流式输出
    """
    stream = await _client.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content