# -*- coding: utf-8 -*-
"""BGE Embedding 封装（sentence-transformers）。

- 模型：BAAI/bge-base-zh-v1.5（默认，768 维）
- 查询侧按官方建议加指令前缀"为这个句子生成表示以用于检索相关文章："
- 懒加载 + 全局单例，避免重复载入
"""
from __future__ import annotations

import logging
from typing import Optional

from .config import BGE_QUERY_INSTRUCTION, DEVICE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model = None


def get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("加载 Embedding 模型: %s (device=%s)", EMBEDDING_MODEL, DEVICE)
        _model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        dim_fn = getattr(_model, "get_embedding_dimension", None) or \
            getattr(_model, "get_sentence_embedding_dimension", lambda: "?")
        logger.info("Embedding 模型加载完成, dim=%s", dim_fn())
    return _model


def embed_texts(texts: list[str], is_query: bool = False, batch_size: int = 32) -> list[list[float]]:
    """批量编码。is_query=True 时加 BGE 指令前缀。"""
    model = get_embedder()
    if is_query:
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    return embed_texts([text], is_query=True)[0]
