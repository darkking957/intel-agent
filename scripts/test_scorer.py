"""Day 2 验收：测试结构化输出解析"""
import asyncio
from datetime import datetime
from intel_agent.pipeline.models import RawArticle
from intel_agent.pipeline.scorer import score_article, batch_score


async def main():
    # 测试数据：3 篇不同质量的文章
    articles = [
        RawArticle(
            url="https://blog.langchain.dev/langgraph-v1/",
            title="LangGraph 1.0: Production-Ready Agent Orchestration",
            content="We are thrilled to announce LangGraph 1.0... state persistence, human-in-the-loop...",
            source="langchain_blog",
        ),
        RawArticle(
            url="https://example.com/chatgpt-tips",
            title="10 ChatGPT Prompts to Write Better Emails",
            content="Are you looking to improve your email writing? Here are 10 prompts...",
            source="medium",
        ),
        RawArticle(
            url="https://arxiv.org/abs/2501.12345",
            title="Scaling Laws for Reasoning in Large Language Models",
            content="We study the relationship between compute and reasoning ability...",
            source="arxiv",
        ),
    ]

    print("=== 单篇评分测试 ===")
    scored = await score_article(articles[0])
    assert scored.score_result is not None, "评分结果不能为 None"
    assert 8 <= scored.score_result.score <= 10, f"LangGraph 文章应该高分，实际 {scored.score_result.score}"
    print(f"✓ {articles[0].title[:40]}")
    print(f"  分数：{scored.score_result.score}  类别：{scored.score_result.category}")
    print(f"  理由：{scored.score_result.reason}")

    print("\n=== 批量评分测试（3篇并发）===")
    results = await batch_score(articles, concurrency=2)
    assert len(results) == 3
    scores = [r.score_result.score for r in results if r.score_result]
    print(f"✓ 分数：{scores}（LangGraph > Prompts，arXiv 应该高分）")
    assert scores[0] > scores[1], "LangGraph 文章应该比 Prompt 技巧得分高"

    print("\n=== Pydantic 类型安全测试 ===")
    # 故意传错误数据，验证 Pydantic 会拒绝
    try:
        from intel_agent.pipeline.models import ArticleScore
        ArticleScore.model_validate({"score": 15, "category": "unknown", "reason": "x", "importance": "skip"})
        print("❌ 应该报 ValidationError！")
    except Exception as e:
        print(f"✓ Pydantic 正确拒绝了非法数据：{type(e).__name__}")

    print("\n✅ Day 2 全部验收通过")


asyncio.run(main())