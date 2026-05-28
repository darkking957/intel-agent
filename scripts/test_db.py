"""
Integration test: DB upsert idempotency + semantic_search.

Verifies:
1. Inserting the same URL twice leaves exactly one row in articles.
2. semantic_search() returns results when embedding and score are set.

Run:
    uv run python scripts/test_db.py
"""
import asyncio
import logging
import numpy as np
from datetime import datetime, timezone

from intel_agent.store.db import get_pool, close_pool
from intel_agent.store.article_repo import insert_raw_article, semantic_search
from intel_agent.pipeline.models import RawArticle

logging.basicConfig(level=logging.WARNING)

TEST_URL = "https://test.example.com/article/db-test-001"
EMBED_DIM = 1536  # bge-m3 native output dimension


def _fake_embedding(seed: int = 42) -> list[float]:
    """Random unit vector — avoids loading the local model in tests."""
    rng = np.random.default_rng(seed)
    vec = rng.random(EMBED_DIM).astype(np.float64)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


async def _cleanup(pool) -> None:
    await pool.execute("DELETE FROM articles WHERE url = $1", TEST_URL)


async def test_upsert_idempotency() -> None:
    """Duplicate URL insert must produce exactly one row."""
    print("=== 测试重复插入幂等性 ===")
    pool = await get_pool()
    await _cleanup(pool)

    article = RawArticle(
        url=TEST_URL,
        title="DB Integration Test Article",
        content="Content used only for automated testing.",
        source="test",
        published_at=datetime.now(timezone.utc),
    )

    id1 = await insert_raw_article(article)
    assert id1 is not None, "首次插入应返回 id"
    print(f"  首次插入 id: {id1}")

    id2 = await insert_raw_article(article)
    assert id2 is None, f"重复插入应返回 None，实际: {id2}"
    print(f"  重复插入返回: {id2}  (期望 None)")

    count = await pool.fetchval(
        "SELECT COUNT(*) FROM articles WHERE url = $1", TEST_URL
    )
    assert count == 1, f"期望 1 条记录，实际 {count} 条"
    print(f"  数据库记录数: {count}  (期望 1)")
    print("[PASS] 重复插入幂等性通过\n")


async def test_semantic_search() -> None:
    """semantic_search() must return the test article after embedding + score are set."""
    print("=== 测试语义搜索 ===")
    pool = await get_pool()

    embedding = _fake_embedding()
    # Build the pgvector literal string; PostgreSQL casts it via ::vector
    vec_literal = "[" + ",".join(f"{v:.10f}" for v in embedding) + "]"

    await pool.execute(
        """
        UPDATE articles
        SET embedding = $1::vector,
            score     = 7,
            importance = 'worth_read',
            category   = 'other',
            tags       = ARRAY['test'],
            reason     = 'automated test',
            processed_at = NOW()
        WHERE url = $2
        """,
        vec_literal,
        TEST_URL,
    )
    print(f"  已写入 embedding（{len(embedding)} 维）及 score=7")

    results = await semantic_search(embedding, limit=5, min_score=5)
    assert len(results) > 0, "semantic_search 应至少返回 1 条结果"

    urls = [r["url"] for r in results]
    assert TEST_URL in urls, f"结果中未找到测试文章，实际 URLs: {urls}"

    top = results[0]
    print(f"  返回 {len(results)} 条结果")
    print(f"  首条: url={top['url']}  similarity={top['similarity']:.4f}")
    print("[PASS] 语义搜索通过\n")


async def main() -> None:
    try:
        await test_upsert_idempotency()
        await test_semantic_search()
        print("[ALL PASS] 全部数据库验收测试通过")
    finally:
        pool = await get_pool()
        await _cleanup(pool)
        print("  测试数据已清理")
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
