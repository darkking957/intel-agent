"""fetch_rss 工具：抓取 RSS Feed，返回文章列表"""
import feedparser
import asyncio
import httpx
import logging
from pydantic import BaseModel, Field, HttpUrl
from intel_agent.tools.registry import ToolDefinition, registry

logger = logging.getLogger(__name__)

class FetchRSSArgs(BaseModel):
    """fetch_rss 工具的参数定义"""
    url: str = Field(description="RSS Feed URL")
    limit: int = Field(default=20, ge=1, le=50, description="最多返回多少篇文章")

async def fetch_rss_impl(url: str, limit: int=20) -> list[dict]:
    """
    feedparser 是同步库，用 run_in_executor 避免阻塞事件循环
    """

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True,) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error for {url}: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            logger.warning(f"Request failed for {url}: {e}")
            return []

    feed = feedparser.parse(resp.text)
    
    if feed.bozo and not feed.entries:
        logger.warning(f"RSS parse failed for {url}: {feed.bozo_exception}")
        return []
    
    articles = []

    for entry in feed.entries[:limit]:
        articles.append(
            {
                "title": getattr(entry, "title", ""),
                "url": getattr(entry, "link", ""),
                "summary": getattr(entry, "summary", "")[:500],
                "published": str(getattr(entry, "published", "")),
                "source": url,
            }
        )
    logger.info(f"Fetched {len(articles)} articles from {url}")
    return articles

registry.register(ToolDefinition(
    name="fetch_rss",
    description="抓取 RSS Feed，返回最新文章列表（标题、URL、摘要）",
    args_model=FetchRSSArgs,
    func=fetch_rss_impl,
))