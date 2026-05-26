import asyncio
# 必须先 import tools 模块，触发工具注册到 registry
import intel_agent.tools.rss_tool  # noqa: F401

from intel_agent.llm.tool_loop import run_tool_loop


async def main():
    print("=== Tool Calling 测试 ===")
    print("（LLM 会自动决定调用 fetch_rss 工具）\n")

    result = await run_tool_loop(
        user_message="帮我查看arXiv cs.AI RSS，告诉我最新的几篇文章标题",
        system_prompt="你是信息助手，当需要获取网络信息时调用 fetch_rss 工具。",
    )
    print(f"最终回复：\n{result}")

    # 验证：结果不为空，且包含一些文章信息
    assert len(result) > 50, "结果太短，可能工具没有正常执行"
    print("\n✅ Day 3 Tool Calling 验收通过")

asyncio.run(main())