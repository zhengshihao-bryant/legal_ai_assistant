# -*- coding: utf-8 -*-
"""Pipeline 编排：build_index / query / evaluate 全链路。

索引产物持久化：
- data/index/legal_vectors.npy + meta.json：向量（本地后端）
- data/chunks/children.json + parents.json：chunk 数据
- data/chunks/index_manifest.json：构建信息
第二次运行（非 rebuild）直接加载缓存，无需重新解析/向量化。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from . import config as cfg
from .chunking import build_chunks
from .context import build_context
from .doc_tree import build_tree, collapse_to_sections
from .embedding import embed_query, embed_texts
from .evaluation import evaluate_generation, evaluate_retrieval, write_reports
from .generation import generate_answer
from .loader import scan_documents
from .models import Answer, Chunk
from .parser import parse_file
from .query import understand_query
from .reranker import rerank
from .retrievers import (BM25Retriever, DenseRetriever, HybridRetriever,
                         IndexData)
from .vector_store import build_vector_store

logger = logging.getLogger(__name__)

CHILDREN_CACHE = cfg.CHUNKS_DIR / "children.json"
PARENTS_CACHE = cfg.CHUNKS_DIR / "parents.json"


class Pipeline:
    def __init__(self, vector_backend: str = "local"):
        self.vector_backend = vector_backend
        self.store = build_vector_store(vector_backend)
        self.index: Optional[IndexData] = None
        self.dense: Optional[DenseRetriever] = None
        self.bm25: Optional[BM25Retriever] = None
        self.hybrid: Optional[HybridRetriever] = None
        self._docs: list = []

    # ================================================================ 缓存
    def _persist_chunks(self, children: list[Chunk], parents: dict[str, Chunk]) -> None:
        CHILDREN_CACHE.write_text(json.dumps(
            [c.__dict__ for c in children], ensure_ascii=False), encoding="utf-8")
        PARENTS_CACHE.write_text(json.dumps(
            [p.__dict__ for p in parents.values()], ensure_ascii=False), encoding="utf-8")

    def _load_chunks(self) -> Optional[tuple[list[Chunk], dict[str, Chunk]]]:
        if not (CHILDREN_CACHE.exists() and PARENTS_CACHE.exists()):
            return None
        try:
            children = [Chunk(**c) for c in json.loads(CHILDREN_CACHE.read_text(encoding="utf-8"))]
            parents = {p["chunk_id"]: Chunk(**p) for p in
                       json.loads(PARENTS_CACHE.read_text(encoding="utf-8"))}
            logger.info("加载 chunk 缓存: %d children / %d parents",
                        len(children), len(parents))
            return children, parents
        except Exception as e:  # noqa: BLE001
            logger.warning("chunk 缓存损坏，将重建: %s", e)
            return None

    def _clear_artifacts(self) -> None:
        for f in list(cfg.INDEX_DIR.glob("legal_vectors*")):
            f.unlink(missing_ok=True)
        for f in (CHILDREN_CACHE, PARENTS_CACHE, cfg.CHUNKS_DIR / "index_manifest.json"):
            f.unlink(missing_ok=True)

    def _set_index(self, children: list[Chunk], parents: dict[str, Chunk]) -> None:
        child_by_id = {c.chunk_id: c for c in children}
        self.index = IndexData(children=children, child_by_id=child_by_id,
                               parent_by_id=parents)
        self.dense = DenseRetriever(self.store, self.index)
        self.bm25 = BM25Retriever(self.index)
        self.hybrid = HybridRetriever(self.dense, self.bm25)

    # ================================================================ 索引
    def build_index(self, rebuild: bool = False) -> dict:
        t0 = time.time()

        if not rebuild:
            cached = self._load_chunks()
            if cached and len(self.store) > 0:
                children, parents = cached
                self._set_index(children, parents)
                self._docs = self._docs or scan_documents()   # 评估需要 doc_id 清单
                logger.info("复用已构建索引（%d chunks, %.1fs）", len(children), time.time() - t0)
                return {"mode": "cached", "children": len(children), "parents": len(parents)}
            logger.info("未找到缓存索引，执行完整构建")

        self._clear_artifacts()
        self.store = build_vector_store(self.vector_backend)

        docs = scan_documents()
        self._docs = docs
        all_children: list[Chunk] = []
        parents: dict[str, Chunk] = {}
        stats = {"docs": len(docs), "parents": 0, "children": 0,
                 "parsed_ok": 0, "parsed_fail": []}

        for doc in docs:
            parsed_cache = cfg.PARSED_DIR / f"{doc.doc_id.replace('/', '__')}.txt"
            try:
                if parsed_cache.exists() and parsed_cache.stat().st_size > 0:
                    text = parsed_cache.read_text(encoding="utf-8", errors="replace")
                    from .parser import _blocks_from_lines
                    blocks = _blocks_from_lines(text.splitlines())
                else:
                    text, blocks = parse_file(Path(doc.path))
                    parsed_cache.write_text(text, encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                logger.warning("解析失败 %s: %s", doc.path, e)
                stats["parsed_fail"].append(doc.path)
                continue
            doc.plain_text = text
            doc.tree = build_tree(doc, blocks)
            sections = collapse_to_sections(doc.tree)
            children, ps = build_chunks(doc, sections)
            all_children.extend(children)
            for p in ps:
                parents[p.chunk_id] = p
            stats["parsed_ok"] += 1

        stats["parents"] = len(parents)
        stats["children"] = len(all_children)

        # 向量化（仅 child）
        texts = [c.text for c in all_children]
        logger.info("向量化 %d 个 child chunk ...", len(texts))
        vecs = embed_texts(texts)
        payloads = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id,
                     "parent_id": c.parent_id, "title": c.title, "source": c.source}
                    for c in all_children]
        self.store.add([c.chunk_id for c in all_children], vecs, payloads)

        self._persist_chunks(all_children, parents)
        self._set_index(all_children, parents)

        manifest = {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_model": cfg.EMBEDDING_MODEL,
            "vector_backend": self.vector_backend,
            "stats": stats,
        }
        (cfg.CHUNKS_DIR / "index_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("索引构建完成: %s（%.1fs）", stats, time.time() - t0)
        return stats

    # ================================================================ 检索
    def retrieve(self, question: str, top_k: int = cfg.RERANK_TOP_K,
                 use_rerank: bool = True, allow_llm_rewrite: bool = False,
                 return_stages: bool = False):
        """返回 (plan, final_results[, stages])。

        stages: dict{stage: [chunk_id, ...]}，供分层评估对比
        （dense / bm25 / hybrid / rerank）。
        """
        if self.hybrid is None:
            raise RuntimeError("请先 build_index()")

        plan = understand_query(question, allow_llm=allow_llm_rewrite)
        q = plan.effective_query
        q_vec = embed_query(q)
        hybrid = self.hybrid.retrieve(q, q_vec)
        if use_rerank:
            final = rerank(q, hybrid, top_k=top_k)
        else:
            final = hybrid[:top_k]
        if return_stages:
            dense = self.dense.retrieve(q_vec, top_k=cfg.TOP_K_HYBRID)
            bm25 = self.bm25.retrieve(q, top_k=cfg.TOP_K_HYBRID)
            stages = {
                "dense": [r.chunk_id for r in dense],
                "bm25": [r.chunk_id for r in bm25],
                "hybrid": [r.chunk_id for r in hybrid],
                "rerank": [r.chunk_id for r in final],
            }
            return plan, final, stages
        return plan, final

    def answer(self, question: str, top_k: int = cfg.RERANK_TOP_K,
               use_rerank: bool = True, allow_llm_rewrite: bool = False) -> tuple:
        """完整问答：retrieve -> context -> generate。返回 (plan, ctx, answer)。"""
        if self.index is None:
            raise RuntimeError("请先 build_index()")
        plan, final = self.retrieve(question, top_k=top_k, use_rerank=use_rerank,
                                    allow_llm_rewrite=allow_llm_rewrite)
        ctx = build_context(final, self.index.parent_by_id)
        ans = generate_answer(question, ctx)
        return plan, ctx, ans

    # ================================================================ 评估
    def evaluate(self, qa_path: Path | None = None, tag: str = "default") -> dict:
        qa_path = qa_path or cfg.EVAL_DIR / "qa.json"
        data = json.loads(qa_path.read_text(encoding="utf-8"))
        questions = data["questions"]
        self._docs = self._docs or scan_documents()
        loaded_ids = [d.doc_id for d in self._docs]

        def doc_of(chunk_id: str) -> str:
            c = self.index.child_by_id.get(chunk_id)
            return c.doc_id if c else chunk_id

        # 每个问题只跑一次全链路，缓存各阶段结果供分层评估复用
        stage_cache: dict[str, dict] = {}

        def stage_results(q: str) -> dict:
            if q not in stage_cache:
                _, _, stages = self.retrieve(q, use_rerank=True, return_stages=True)
                stage_cache[q] = stages
            return stage_cache[q]

        def make_retrieve_fn(stage: str):
            def fn(q: str, k: int) -> list[str]:
                return stage_results(q)[stage][:k]
            return fn

        retrieval_by_stage: dict[str, dict] = {}
        for stage in ("dense", "bm25", "hybrid", "rerank"):
            retrieval_by_stage[stage] = evaluate_retrieval(
                questions, make_retrieve_fn(stage), doc_of, loaded_ids)

        def gen_fn(qa: dict) -> Answer:
            _, _, ans = self.answer(qa["question"], use_rerank=True)
            return ans

        gen_res = evaluate_generation(questions, gen_fn)
        ret_res = {
            "by_stage": {s: r["metrics"] for s, r in retrieval_by_stage.items()},
            "per_question": retrieval_by_stage["rerank"]["per_question"],
        }
        md_path = write_reports(ret_res, gen_res, questions, tag=tag)
        return {
            "retrieval": ret_res["by_stage"],
            "generation": gen_res["metrics"],
            "error_cases": gen_res["error_cases"],
            "report_path": str(md_path),
        }
