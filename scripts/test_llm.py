import asyncio
from intel_agent.llm.client import chat_complete, chat_stream


async def test_basic_call():
    """测试基础调用"""
    print("=== 测试基础调用 ===")
    response = await chat_complete(
        messages=[
            {"role": "system", "content": "你是一个简洁的助手，用一句话回答。"},
            {"role": "user", "content": "LLM 的 temperature 参数有什么作用？"},
        ],
        temperature=0,
    )
    print(f"回复：{response}")
    assert len(response) > 10, "回复太短，可能有问题"
    print("✓ 基础调用通过")


async def test_streaming():
    """测试流式输出"""
    print("\n=== 测试流式输出 ===")
    messages = [{"role": "user", "content": "用三句话解释 Transformer 架构"}]
    print("流式回复：", end="")
    chunks = []
    async for chunk in chat_stream(messages):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print()
    assert len(chunks) > 3, "流式输出块太少"
    print("✓ 流式输出通过")


async def main():
    await test_basic_call()
    await test_streaming()
    print("\n✅ Day 1 全部验收通过")


if __name__ == "__main__":
    asyncio.run(main())