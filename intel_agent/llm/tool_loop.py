import json
import logging
import asyncio
from openai import AsyncOpenAI
from intel_agent.config import settings
from intel_agent.tools.registry import registry

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
)

async def run_tool_loop(
    user_message: str,
    system_prompt: str = "你是一个信息采集助手，根据需要调用工具获取信息。",
    max_iterations: int = 5,  # 防止无限循环
) -> str:
    """
    完整的 Tool Calling 循环。
    max_iterations：最多工具调用轮次，防止失控的 Agent 无限循环。
    """

    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_message},
        ]
    
    tools = registry.get_openai_tools() #tool name Mapping func name

    for iteration in range(max_iterations):
        response = await _client.chat.completions.create(
                    model=settings.deepseek_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
        
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason=="stop":
            logger.info(f"Tool loop completed in {iteration + 1} iterations")
            return message.content or ""
        
        if finish_reason=="tool_calls" and message.tool_calls:
            messages.append(
                {
                    "role":"assistant",
                    "content":message.content or "",
                    "tool_calls":[{
                        "id":tc.id,
                        "type":"function",
                        "function":{
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                    ]
                }         
            )

            
            tool_results = await asyncio.gather(*[
                registry.dispatch(tc.function.name, tc.function.arguments)
                for tc in message.tool_calls
            ])
                
            for tc, result in zip(message.tool_calls, tool_results):
                logger.info(f"Tool {tc.function.name} called, result length: {len(result)}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,  # 必须匹配！
                    "content": result,
                })

    logger.warning(f"Tool loop reached max iterations ({max_iterations})")
    return "已达到最大工具调用轮次，请重新提问。"
                
