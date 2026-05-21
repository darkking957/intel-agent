"""
文章评分器。
核心设计：
1. Prompt 模板函数，支持注入用户兴趣画像
2. LLM 返回的 JSON 用 Pydantic 解析（解析即校验）
3. 失败时有 fallback（不丢数据）
"""
import json
import logging
import asyncio
from intel_agent.llm.client import chat_complete
from intel_agent.pipeline.models import ArticleScore, ContentCategory, ImportanceLevel, RawArticle, ScoredArticle

logger = logging.getLogger(__name__)

# ============================================================
# Prompt 模板
# 设计原则：system prompt 固定角色，user prompt 传入变量
# 用函数而非字符串常量，方便注入动态内容（兴趣画像）
# ============================================================

# 参考 few-shot 样例（固定，用于稳定输出质量）
FEW_SHOT_EXAMPLES = """
[好文示例]
标题：LangGraph 1.0 Production Release: State Persistence and Human-in-the-Loop
打分：9
分类：engineering_tool
理由：LangGraph 正式版发布，直接影响 AI Agent 工程实践，有具体 API 变化

[中等文章示例]
标题：ChatGPT Daily Active Users Hit 100M
打分：5
分类：industry_news
理由：行业数据有参考价值，但无技术深度

[低价值示例]
标题：10 ChatGPT Prompts for Better Writing
打分：2
分类：tutorial
理由：面向普通用户的技巧汇总，无工程价值
"""


def build_system_prompt(interest_profile: str = "") -> str:
    """
    构建 system prompt。
    interest_profile：从用户历史 like 行为中提取的兴趣关键词（Day 9 实现）
    """
    schema = ArticleScore.model_json_schema()
    interest_section = (
        f"\n\n【用户兴趣画像】\n{interest_profile}\n请根据用户偏好调整评分权重。"
        if interest_profile else ""
    )
    return f"""你是一个 AI/ML 技术内容筛选助手，面向的读者是 AI 工程师。

评分标准（1-10）：
- 8-10：前沿研究、重要框架更新、架构级工程实践
- 5-7：有参考价值的技术文章、行业动态
- 1-4：营销内容、重复信息、入门教程、无技术深度

【few-shot 参考】
{FEW_SHOT_EXAMPLES}{interest_section}

【输出要求】
严格按以下 JSON Schema 输出，不要有任何额外文本：
{json.dumps(schema, ensure_ascii=False, indent=2)}"""


def build_user_prompt(article: RawArticle) -> str:
    content_preview = article.content[:800] if article.content else "（正文未提取）"
    return f"""请对以下文章打分：

标题：{article.title}
来源：{article.source}
正文预览：
{content_preview}"""


async def score_article(
    article: RawArticle,
    interest_profile: str = "",
) -> ScoredArticle:
    """
    对单篇文章打分。
    返回 ScoredArticle，score_result 字段为 None 表示评分失败（有 fallback）。
    """
    scored = ScoredArticle(**article.model_dump())  # 拷贝基础字段

    try:
        raw_response = await chat_complete(
            messages=[
                {"role": "system", "content": build_system_prompt(interest_profile)},
                {"role": "user", "content": build_user_prompt(article)},
            ],
            temperature=0,  # 评分任务用 temperature=0，确保稳定
            response_format={"type": "json_object"},
        )

        # ← 关键：Pydantic 解析+校验，而不是 json.loads()
        score_result = ArticleScore.model_validate_json(raw_response)
        scored.score_result = score_result

        logger.info(
            f"Scored article: [{score_result.score}/10] {article.title[:50]}"
        )

    except Exception as e:
        # 评分失败不崩溃：给默认低分，标记为跳过
        logger.warning(f"Score failed for {article.url}: {e}")
        scored.score_result = ArticleScore(
            score=1,
            category=ContentCategory.OTHER,
            importance=ImportanceLevel.SKIP,
            reason=f"评分失败：{str(e)[:100]}",
            tags=[],
        )

    return scored


async def batch_score(
    articles: list[RawArticle],
    interest_profile: str = "",
    concurrency: int = 3,
) -> list[ScoredArticle]:
    """
    批量评分，控制并发（避免触发 API 限速）。
    用 asyncio.Semaphore 而不是 asyncio.gather 直接并发——后者没有速率控制。
    """
    sem = asyncio.Semaphore(concurrency)

    async def _score_with_sem(article: RawArticle) -> ScoredArticle:
        async with sem:
            return await score_article(article, interest_profile)

    tasks = [_score_with_sem(a) for a in articles]
    return await asyncio.gather(*tasks)