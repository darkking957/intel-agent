from __future__ import annotations
from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict, model_validator
from typing import Annotated

class ContentCategory(StrEnum):
    """文章分类枚举：用 StrEnum 而非普通 str，防止传入任意字符串"""
    LLM_RESEARCH = "llm_research"
    ENGINEERING_TOOL = "engineering_tool"
    INDUSTRY_NEWS = "industry_news"
    TUTORIAL = "tutorial"
    OTHER = "other"

class ImportanceLevel(StrEnum):
    """重要度标记"""
    MUST_READ = "must_read"      # 评分 8-10
    WORTH_READ = "worth_read"    # 评分 5-7
    SKIP = "skip"                # 评分 1-4

class RawArticle(BaseModel):
    """
    采集器输出的原始文章。
    """
    model_config = ConfigDict(frozen=False)

    url: str = Field(..., description="文章URL，全局唯一")
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(default="", description="正文内容，可能为空（全文提取失败时）")
    source: str = Field(..., description="来源标识，如 arxiv/hn/github_trending")
    published_at: datetime | None = None

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL 必须以 http/https 开头：{v}")
        return v
    
    @field_validator("content")
    @classmethod
    def truncate_content(cls, v: str) -> str:
        """防止超长正文占用过多 token，截断到 5000 字符"""
        return v[:5000] if len(v) > 5000 else v
    
class ArticleScore(BaseModel):
    """
    LLM 评分结果。
    这个模型同时作为：
    1. LLM 输出的解析目标（model_validate_json）
    2. JSON Schema 的来源（model_json_schema() → system prompt）
    """

    score: Annotated[int,Field(ge=1,le=10,description="相关度评分，1-10 整数")]
    category: ContentCategory
    importance: ImportanceLevel | None = Field(None, exclude=True)
    reason: str = Field(..., max_length=200, description="打分理由，≤200字")
    tags: list[str] = Field(default_factory=list, max_length=5, description="最多5个标签")

    @model_validator(mode="after")                     # ← 改用 model_validator
    def infer_importance(self) -> ArticleScore:
        """score 验证完毕后，若 importance 未传则自动推断"""
        if self.importance is None:                    # 只在没传的时候推断
            if self.score >= 8:
                self.importance = ImportanceLevel.MUST_READ
            elif self.score >= 5:
                self.importance = ImportanceLevel.WORTH_READ
            else:
                self.importance = ImportanceLevel.SKIP
        return self

    @classmethod
    def from_score(cls, score: int, category: str, reason: str, tags: list[str]) -> "ArticleScore":
        """工厂方法：importance 不传，由 model_validator 自动推断"""
        return cls(
            score=score,
            category=ContentCategory(category),
            reason=reason,
            tags=tags,
            # importance 故意不传
        )

class ScoredArticle(RawArticle):
    """打过分的文章：继承 RawArticle，添加评分字段"""
    score_result: ArticleScore | None = None

    @property
    def should_process(self) -> bool:
        return (
            self.score_result is not None and
            self.score_result.importance != ImportanceLevel.SKIP
        )