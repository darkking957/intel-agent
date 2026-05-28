import asyncio
import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

LOCAL_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1536  # 与原 OpenAI 版本保持一致，无需迁移


@lru_cache(maxsize=1)
def _get_local_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(LOCAL_MODEL_NAME)


async def get_embedding(text: str) -> list[float]:
    text = text[:6000]
    loop = asyncio.get_event_loop()
    model = _get_local_model()

    raw: np.ndarray = await loop.run_in_executor(
        None,
        lambda: model.encode(text, normalize_embeddings=False, convert_to_numpy=True),
    )

    # Matryoshka 截断到 1536 维 + 重新归一化
    vec = raw[:1536]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()  # 维度 = 1536