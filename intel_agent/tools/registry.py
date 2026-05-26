import json
import logging
from collections.abc import Callable,Awaitable
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ToolDefinition:
    """single tool calling"""
    def __init__(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        func: callable[...,Awaitable[any]],        
                 ):
        self.name = name
        self.description = description
        self.args_model = args_model
        self.func = func

    def to_openai_schema(self) -> dict:
        """trans to openai Tool Calling schema"""
        return {
            "type":"function",
            "function":{
                "name":self.name,
                "description":self.description,
                "parameters":self.args_model.model_json_schema(),
            }
        }

class ToolRegistry:
    """Tool registry table"""
    def __init__(self):
        self._tools: dict[str,ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.info(f"registered tool: {tool.name}")
    
    def get_openai_tools(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]
    
    async def dispatch(self, tool_name: str, arguments_json: str) -> str:
        """
        Using Tool
        arguments_json:LLM 返回的参数 JSON 字符串
        返回：工具执行结果
        """
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        tool = self._tools[tool_name]

        try:
            args = tool.args_model.model_validate_json(arguments_json)
            result = await tool.func(**args.model_dump())
            if isinstance(result, str):
                return result
            elif isinstance(result, BaseModel):
                return result.model_dump_json
            else:
                return json.dumps(result, ensure_ascii=False , default=str)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed : {e}")
            return json.dumps({"error": str(e), "tool":tool_name})

registry = ToolRegistry()