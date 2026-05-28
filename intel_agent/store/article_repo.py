import logging
from datetime import datetime
import asyncpg
from intel_agent.pipeline.models import RawArticle, ScoredArticle
from intel_agent.store.db import get_pool

logger = logging.getLogger(__name__)

async def is_url_exists(url: str) -> bool:

    pool = await get_pool()
    row = await pool.fetchrow("select 1 from articles where url = $1", url)
    return row is not None

async def insert_raw_article(article: RawArticle) -> str | None:
    "插入操作，存在就raise"
    pool = await get_pool()
    try:
        row = await pool.fetchrow(
        """
        INSERT INTO articles (url, title, content, source, published_at)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (url) DO NOTHING
        RETURNING id
        """,
        article.url,
        article.title,
        article.content,
        article.source,
        article.published_at,
        )
        if row:
            logger.info(f"Inserted article: {article.url}")
            return str(row["id"])
        else:
            logger.debug(f"Skipped duplicate: {article.url}")
            return None
    except Exception as e:
        logger.error(f"Insert failed for {article.url}: {e}")
        raise

async def update_score(url: str, scored: ScoredArticle) -> None:
    """更新文章的评分结果"""
    if not scored.score_result:
        return
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE articles
        SET score = $1,
            importance = $2,
            category = $3,
            tags = $4,
            reason = $5,
            processed_at = NOW()
        WHERE url = $6
        """,
        scored.score_result.score,
        scored.score_result.importance.value,
        scored.score_result.category.value,
        scored.score_result.tags,
        scored.score_result.reason,
        url,
    )

def _vec_str(embedding: list[float]) -> str:
    """asyncpg 没有内置 pgvector codec，需要序列化成字符串再由 PG 做 ::vector 转型。"""
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def update_embedding(url: str, embedding: list[float]) -> None:
    """更新文章的向量（Embedding）"""
    pool = await get_pool()
    await pool.execute(
        "UPDATE articles SET embedding = $1::vector WHERE url = $2",
        _vec_str(embedding),
        url,
    )

async def batch_insert(articles: list[RawArticle]) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            INSERT INTO articles (url, title, content, source, published_at)
            SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::timestamptz[])
            ON CONFLICT (url) DO NOTHING
            """,
            [a.url     for a in articles],
            [a.title   for a in articles],
            [a.content for a in articles],
            [a.source  for a in articles],
            [a.published_at for a in articles],
        )
    inserted = int(result.split(" ")[-1])  # execute() 才会返回 "INSERT 0 N"
    logger.info(f"Batch insert: {inserted}/{len(articles)} articles inserted")
    return inserted


async def semantic_search(
    query_embedding: list[float],
    limit: int = 10,
    min_score: int = 5,
) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, url, title, score, summary_short,
               1 - (embedding <=> $1::vector) AS similarity
        FROM articles
        WHERE embedding IS NOT NULL
          AND score >= $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
        """,
        _vec_str(query_embedding),
        min_score,
        limit,
    )
    return [dict(r) for r in rows]

    