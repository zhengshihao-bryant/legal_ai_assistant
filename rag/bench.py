# -*- coding: utf-8 -*-
"""bench：统一基准 —— 单进程一次加载模型，共享检索结果，一次性产出全部报告。

产出（data/evaluation/reports/）：
- report_v1.{json,md}            主评估（四阶段检索 + 本地生成）
- ablation_rerank_v1.*           ① Rerank Ablation
- ablation_query_rewrite_v1.*    ② Query Rewrite（原始/规则/LLM，LLM 走 DeepSeek）
- qa_class_v1.*                  ③ QA 分类 + Error Analysis
- data_quality_v1.*              ④ OCR / Chunking 数据质量
- generation_compare_v1.*        （新）本地抽取式 vs DeepSeek LLM 生成对比

相比逐个跑实验：模型只加载一次、每问只检索一次（三组改写共享 dense/bm25 语料、
rerank 只对原始查询做一次），预计耗时从 ~30min 降到 ~15min。

检查点：LLM 改写/生成结果落盘到 reports/bench_checkpoint_{tag}.json，
中断后续跑自动复用（不重复调用 LLM API）；改代码后请删该文件或加 --no-checkpoint。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from . import config as cfg
from .ablation import (run_query_rewrite_ablation, run_rerank_ablation,
                       write_ablation_report)
from .context import build_context
from .data_quality import run_data_quality, write_data_quality_report
from .embedding import embed_query
from .evaluation import (evaluate_generation, evaluate_retrieval,
                         gold_doc_ids, write_reports)
from .generation import LocalGenerator, OpenAIGenerator, generate_answer
from .models import Answer
from .pipeline import Pipeline
from .qa_analysis import run_qa_class_analysis, write_qa_class_report
from .query import QueryPlan, understand_query
from .reranker import rerank

logger = logging.getLogger(__name__)
REPORT_DIR = cfg.REPORT_DIR
RERANK_TOP = 20


# ---------------------------------------------------------------- 检查点
def _answer_to_dict(ans: Answer) -> dict:
    return {"answer": ans.answer, "citations": ans.citations,
            "groundedness": ans.groundedness, "unsupported": ans.unsupported,
            "generator": ans.meta.get("generator", "")}


def _answer_from_dict(d: Optional[dict]) -> Optional[Answer]:
    if d is None:
        return None
    return Answer(answer=d.get("answer", ""), citations=d.get("citations", []),
                  groundedness=d.get("groundedness", 0.0),
                  unsupported=d.get("unsupported", []),
                  meta={"generator": d.get("generator", "")})


def _load_checkpoint(path: Path) -> dict:
    if path.exists():
        try:
            ckpt = json.loads(path.read_text(encoding="utf-8"))
            logger.info("加载 bench 检查点: %d 条（跳过已完成的 LLM 调用）", len(ckpt))
            return ckpt
        except Exception as e:  # noqa: BLE001
            logger.warning("检查点读取失败，重新构建: %s", e)
    return {}


def build_bench_cache(p: Pipeline, questions: list[dict],
                      use_llm_rewrite: bool, use_llm_generation: bool,
                      checkpoint_path: Optional[Path] = None) -> dict:
    """每问一次全链路：三组改写查询的 hybrid + 原始查询的 rerank + 本地/LLM 生成。

    LLM 相关（改写查询 + 生成答案）写入检查点，可断点续跑。
    """
    ckpt = _load_checkpoint(checkpoint_path) if checkpoint_path else {}
    cache: dict = {}
    n_rewrite_used = 0
    for i, qa in enumerate(questions, 1):
        q = qa["question"]

        if q in ckpt:
            ent = ckpt[q]
            plan_rule = QueryPlan(original=q)
            plan_rule.expanded = ent["rule_q"]
            plan_llm = QueryPlan(original=q)
            if ent.get("llm_used"):
                plan_llm.rewritten = ent["llm_q"]
                plan_llm.used_llm = True
            else:
                plan_llm.expanded = ent["llm_q"]
            ans_local = _answer_from_dict(ent.get("local"))
            ans_llm = _answer_from_dict(ent.get("llm"))
        else:
            plan_rule = understand_query(q, allow_llm=False)
            plan_llm = understand_query(q, allow_llm=use_llm_rewrite)
            ans_local, ans_llm = None, None
            ent = {"rule_q": plan_rule.effective_query,
                   "llm_q": plan_llm.effective_query,
                   "llm_used": plan_llm.used_llm}
            ckpt[q] = ent

        if plan_llm.used_llm:
            n_rewrite_used += 1
        qr, qr2, qr3 = q, plan_rule.effective_query, plan_llm.effective_query

        # ---- 检索（每次重算，廉价）
        t0 = time.perf_counter()
        v1 = embed_query(qr)
        hybrid_raw = p.hybrid.retrieve(qr, v1)
        t1 = time.perf_counter()
        dense_raw = p.dense.retrieve(v1, top_k=cfg.TOP_K_HYBRID)
        bm25_raw = p.bm25.retrieve(qr, top_k=cfg.TOP_K_HYBRID)
        v2 = embed_query(qr2)
        hybrid_rule = p.hybrid.retrieve(qr2, v2)
        t2 = time.perf_counter()
        v3 = embed_query(qr3)
        hybrid_llm = p.hybrid.retrieve(qr3, v3)
        t3 = time.perf_counter()
        reranked = rerank(qr, hybrid_raw, top_k=RERANK_TOP)
        t4 = time.perf_counter()
        ctx = build_context(reranked, p.index.parent_by_id)

        # ---- 生成（仅未入检查点时调用 LLM）
        if ans_local is None:
            ans_local = generate_answer(q, ctx, generator=LocalGenerator())
            ans_llm = None
            if use_llm_generation:
                try:
                    ans_llm = generate_answer(q, ctx, generator=OpenAIGenerator())
                except Exception as e:  # noqa: BLE001
                    logger.warning("LLM 生成失败(%s)，该条记为本地兜底: %s", qa["id"], e)
                    ans_llm = ans_local
            ckpt[q].update({"local": _answer_to_dict(ans_local),
                            "llm": _answer_to_dict(ans_llm) if ans_llm is not None else None})
            if checkpoint_path:
                checkpoint_path.write_text(json.dumps(ckpt, ensure_ascii=False),
                                           encoding="utf-8")

        cache[q] = {
            "raw": {"query": qr, "hybrid": hybrid_raw, "dense": dense_raw,
                    "bm25": bm25_raw, "rerank": reranked,
                    "hybrid_ms": (t1 - t0) * 1000, "rerank_ms": (t4 - t3) * 1000},
            "rule": {"query": qr2, "hybrid": hybrid_rule, "plan": plan_rule,
                     "hybrid_ms": (t2 - t1) * 1000},
            "llm": {"query": qr3, "hybrid": hybrid_llm, "plan": plan_llm,
                    "hybrid_ms": (t3 - t2) * 1000},
            "ctx": ctx, "ans_local": ans_local, "ans_llm": ans_llm,
        }
        if i % 20 == 0:
            logger.info("bench 缓存进度: %d/%d (llm_rewrite=%d)", i, len(questions), n_rewrite_used)
    logger.info("bench 缓存完成，LLM 改写命中 %d/%d", n_rewrite_used, len(questions))
    return cache


def _write_generation_compare(local_metrics: dict, llm_metrics: Optional[dict],
                              tag: str) -> Path:
    lines = [f"# Generation 对比：本地抽取式 vs DeepSeek LLM（{tag}）", "",
             "| 指标 | 本地抽取式 | DeepSeek LLM |", "| --- | --- | --- |"]
    for k in ("em", "f1", "rouge1", "rouge2", "rougeL", "citation_rate", "avg_groundedness"):
        lv = local_metrics.get(k, "-")
        mv = llm_metrics.get(k, "未运行") if llm_metrics else "未运行"
        lines.append(f"| {k} | {lv} | {mv} |")
    lines.append("")
    lines.append("*由 bench 统一基准生成。*")
    md = REPORT_DIR / f"generation_compare_{tag}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / f"generation_compare_{tag}.json").write_text(
        json.dumps({"local": local_metrics, "llm": llm_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    logger.info("生成对比报告: generation_compare_%s.{json,md}", tag)
    return md


def run_bench(p: Pipeline, questions: list[dict], tag: str = "v1",
              use_llm_rewrite: bool = True, use_llm_generation: bool = True,
              use_checkpoint: bool = True) -> dict:
    t_start = time.time()
    loaded_ids = [d.doc_id for d in p._docs]
    checkpoint_path = REPORT_DIR / f"bench_checkpoint_{tag}.json" if use_checkpoint else None
    cache = build_bench_cache(p, questions, use_llm_rewrite, use_llm_generation,
                              checkpoint_path=checkpoint_path)

    def doc_of(chunk_id: str) -> str:
        c = p.index.child_by_id.get(chunk_id)
        return c.doc_id if c else chunk_id

    # ---- 1) 主评估（四阶段，生成用本地，保证与历史基线可比）
    stage_res: dict[str, dict] = {}
    for stage, key in (("dense", "dense"), ("bm25", "bm25"),
                       ("hybrid", "hybrid"), ("rerank", "rerank")):
        def fn(q: str, k: int, _key: str = key) -> list[str]:
            return [r.chunk_id for r in cache[q]["raw"][_key]][:k]
        stage_res[stage] = evaluate_retrieval(questions, fn, doc_of, loaded_ids)
    gen_local = evaluate_generation(
        questions, lambda qa: cache[qa["question"]]["ans_local"])
    write_reports({"by_stage": {s: stage_res[s]["metrics"] for s in stage_res},
                   "per_question": stage_res["rerank"]["per_question"]},
                  gen_local, questions, tag=tag)

    # ---- 2) ① Rerank Ablation
    r_rerank = run_rerank_ablation(p, questions, cache=cache)
    write_ablation_report(r_rerank, "rerank", tag)

    # ---- 3) ② Query Rewrite Ablation（LLM 组走 DeepSeek）
    r_qr = run_query_rewrite_ablation(p, questions, allow_llm=use_llm_rewrite, cache=cache)
    write_ablation_report(r_qr, "query_rewrite", tag)

    # ---- 4) ③ QA 分类
    r_qa = run_qa_class_analysis(p, questions, cache=cache)
    write_qa_class_report(r_qa, tag)

    # ---- 5) ④ 数据质量
    r_dq = run_data_quality(p, questions)
    write_data_quality_report(r_dq, tag)

    # ---- 6) 生成对比（本地 vs LLM）
    gen_llm_metrics = None
    if use_llm_generation:
        gen_llm = evaluate_generation(
            questions, lambda qa: cache[qa["question"]]["ans_llm"])
        gen_llm_metrics = gen_llm["metrics"]
    _write_generation_compare(gen_local["metrics"], gen_llm_metrics, tag)

    return {
        "runtime_s": round(time.time() - t_start, 1),
        "llm_rewrite_used": sum(1 for c in cache.values() if c["llm"]["plan"].used_llm),
        "llm_generation_used": sum(1 for c in cache.values()
                                   if c["ans_llm"] is not None and
                                   c["ans_llm"].meta.get("generator") == "OpenAIGenerator"),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "report_v1": str(REPORT_DIR / f"report_{tag}.md"),
        "reports": [f"ablation_rerank_{tag}.md", f"ablation_query_rewrite_{tag}.md",
                    f"qa_class_{tag}.md", f"data_quality_{tag}.md",
                    f"generation_compare_{tag}.md"],
    }
