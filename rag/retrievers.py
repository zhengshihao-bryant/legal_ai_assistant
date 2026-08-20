# -*- coding: utf-8 -*-
"""Retrievers：Dense / BM25 / Hybrid(RRF Fusion) / Rerank 编排。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import jieba

from .config import RRF_K, TOP_K_BM25, TOP_K_DENSE, TOP_K_HYBRID
from .models import Chunk, RetrievalResult
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# jieba 初始化（加载词库，保证分词稳定）
jieba.setLogLevel(logging.WARNING)


@dataclass
class IndexData:
    """一次索引的内存态：children + 映射 + 词表。"""

    children: list[Chunk]
    child_by_id: dict[str, Chunk]
    parent_by_id: dict[str, Chunk]
    bm25: Optional["BM25Okapi"] = None          # rank_bm25 对象
    tokenized: list[list[str]] = field(default_factory=list)   # 每条 child 的分词


class DenseRetriever:
    def __init__(self, store: VectorStore, index: IndexData):
        self.store = store
        self.index = index

    def retrieve(self, query_vec: list[float], top_k: int = TOP_K_DENSE) -> list[RetrievalResult]:
        hits = self.store.search(query_vec, top_k=top_k)
        out = []
        for cid, score in hits:
            chunk = self.index.child_by_id.get(cid)
            if chunk is None:
                continue
            out.append(RetrievalResult(chunk_id=cid, score=score, chunk=chunk, stage="dense"))
        return out


class BM25Retriever:
    def __init__(self, index: IndexData):
        self.index = index
        if index.bm25 is None:
            from rank_bm25 import BM25Okapi
            if not index.tokenized:
                logger.info("构建 BM25 语料（jieba 分词 %d 条）...", len(index.children))
                index.tokenized = [self.tokenize(c.text) for c in index.children]
            self.index.bm25 = BM25Okapi(index.tokenized)
        self.bm25 = self.index.bm25

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t for t in jieba.lcut(text) if t.strip() and t not in " \t，。；：、（）()《》“”‘’？！!?.,;:"]

    def retrieve(self, query: str, top_k: int = TOP_K_BM25) -> list[RetrievalResult]:
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        out = []
        for i in order:
            cid = self.index.children[i].chunk_id
            out.append(RetrievalResult(
                chunk_id=cid, score=float(scores[i]),
                chunk=self.index.children[i], stage="bm25",
            ))
        return out


def rrf_merge(lists: list[list[RetrievalResult]], k: int = RRF_K) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion：对多个排序列表做倒数排名融合。"""
    scores: dict[str, float] = {}
    rank_map: dict[str, RetrievalResult] = {}
    for lst in lists:
        for rank, r in enumerate(lst, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
            if r.chunk_id not in rank_map:
                rank_map[r.chunk_id] = r
    merged = sorted(scores.items(), key=lambda kv: -kv[1])
    return [RetrievalResult(chunk_id=cid, score=sc, chunk=rank_map[cid].chunk, stage="hybrid")
            for cid, sc in merged]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever):
        self.dense = dense
        self.bm25 = bm25

    def retrieve(self, query: str, query_vec: list[float],
                 top_k: int = TOP_K_HYBRID) -> list[RetrievalResult]:
        dense_hits = self.dense.retrieve(query_vec, top_k=top_k * 2)
        bm25_hits = self.bm25.retrieve(query, top_k=top_k * 2)
        return rrf_merge([dense_hits, bm25_hits])[:top_k]
