# -*- coding: utf-8 -*-
"""BGE Reranker（CrossEncoder 交叉编码器重排）。

默认 BAAI/bge-reranker-base；模型缺失/下载失败时优雅降级为原序返回。
"""
from __future__ import annotations

import logging
from typing import Optional

from .config import RERANK_MODEL, USE_RERANKER
from .models import RetrievalResult

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        logger.info("加载 Reranker 模型: %s", RERANK_MODEL)
        _model = CrossEncoder(RERANK_MODEL, max_length=512)
        logger.info("Reranker 加载完成")
    return _model


def rerank(query: str, results: list[RetrievalResult],
           top_k: Optional[int] = None) -> list[RetrievalResult]:
    """对检索结果按 query 相关性重排。失败时保持原序。"""
    if not USE_RERANKER or not results:
        return results[:top_k] if top_k else results
    try:
        model = _load_model()
    except Exception as e:  # noqa: BLE001
        logger.warning("Reranker 不可用，保持原序: %s", e)
        return results[:top_k] if top_k else results

    pairs = [(query, r.chunk.text[:512] if r.chunk else r.chunk_id) for r in results]
    scores = model.predict(pairs, show_progress_bar=False)
    scored = sorted(zip(results, scores), key=lambda kv: -float(kv[1]))
    out = []
    for r, s in scored:
        r.score = float(s)
        r.stage = "rerank"
        out.append(r)
    return out[:top_k] if top_k else out
