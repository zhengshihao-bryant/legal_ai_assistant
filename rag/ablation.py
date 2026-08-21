# -*- coding: utf-8 -*-
"""Ablation Harness（消融实验框架）。

V1.5 ① Rerank Ablation：
- 配置：Hybrid（无 Rerank） vs Hybrid + Rerank，各取 Top-K ∈ {5, 10, 20}
- 指标：chunk-level Recall@K / MRR（伪 gold：gold 文档内与参考答案词汇重叠的子块）
         doc-level  Recall@K / MRR / NDCG@10（复用 qa.json related_docs）
- 附加：延迟分解（Hybrid / Rerank / 总）、按文档类型（法规/制度/合同/案例）拆分 Recall@10

后续实验（② Query Rewrite / ③ QA 分类 / ④ 数据质量）复用本框架的指标与报告结构。
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import jieba

from . import config as cfg
from .evaluation import _ndcg, gold_doc_ids, tokenize_zh
from .embedding import embed_query
from .pipeline import Pipeline
from .reranker import rerank

logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.WARNING)

REPORT_DIR = cfg.REPORT_DIR
DOC_KIND_LABEL = {"laws": "法规", "policies": "制度", "contracts": "合同", "cases": "案例"}


# ================================================================ chunk 级伪 gold
def build_chunk_gold(questions: list[dict], index, loaded_ids: list[str],
                     min_overlap: int = 2) -> dict[str, set[str]]:
    """qa_id -> 相关 chunk_id 集合。

    相关判定：chunk 来自 gold 文档，且与参考答案共享 >= min_overlap 个内容词。
    若某题没有满足条件的 chunk，回退为该 gold 文档的全部子块（避免假阴性）。
    """
    gold: dict[str, set[str]] = {}
    for qa in questions:
        qid = qa["id"]
        gold_docs = set(gold_doc_ids(qa, loaded_ids))
        ans_toks = set(tokenize_zh(qa.get("answer", "")))
        relevant: set[str] = set()
        for c in index.children:
            if c.doc_id not in gold_docs:
                continue
            c_toks = set(tokenize_zh(c.text))
            if len(ans_toks & c_toks) >= min_overlap:
                relevant.add(c.chunk_id)
        if not relevant:
            relevant = {c.chunk_id for c in index.children if c.doc_id in gold_docs}
        gold[qid] = relevant
    return gold


# ================================================================ 指标计算
def _chunk_metrics(ranks: list[int], gold_count: int, top_ks: tuple[int, ...]) -> dict:
    """ranks: 相关 chunk 在结果中的 1-based 位置。"""
    recall = {k: 1.0 if any(r <= k for r in ranks) else 0.0 for k in top_ks}
    mrr = 1.0 / min(ranks) if ranks else 0.0
    ndcg = {k: _ndcg(ranks, gold_count, k) for k in top_ks}
    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def _doc_metrics(doc_order: list[str], gold_docs: set[str], top_ks: tuple[int, ...]) -> dict:
    ranks = [i + 1 for i, d in enumerate(doc_order) if d in gold_docs]
    recall = {k: 1.0 if any(r <= k for r in ranks) else 0.0 for k in top_ks}
    mrr = 1.0 / min(ranks) if ranks else 0.0
    ndcg = {k: _ndcg(ranks, len(gold_docs), k) for k in top_ks}
    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


# ================================================================ Rerank Ablation
def run_rerank_ablation(p: Pipeline, questions: list[dict],
                        top_ks: tuple[int, ...] = (5, 10, 20),
                        min_overlap: int = 2,
                        cache: Optional[dict] = None) -> dict:
    """Hybrid vs Hybrid+Rerank × Top-K 对比，chunk/doc 双层指标 + 延迟 + 分域。

    cache: bench 共享缓存（question -> 含 raw.hybrid / raw.rerank / latency），
           提供时跳过检索只算指标。
    """
    loaded_ids = [d.doc_id for d in p._docs]
    chunk_gold = build_chunk_gold(questions, p.index, loaded_ids, min_overlap)
    doc_gold = {qa["id"]: set(gold_doc_ids(qa, loaded_ids)) for qa in questions}

    # config -> {指标聚合}
    agg: dict[str, dict] = {}
    lat_hybrid: list[float] = []
    lat_rerank: list[float] = []
    kind_recall10: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def bucket(config: str, qa: dict, chunk_ranks: list[int], doc_order: list[str],
               doc_gold_set: set[str]):
        cm = _chunk_metrics(chunk_ranks, len(chunk_gold[qa["id"]]), top_ks)
        dm = _doc_metrics(doc_order, doc_gold_set, top_ks)
        agg[config]["chunk_recall"].append(cm["recall"])
        agg[config]["chunk_mrr"].append(cm["mrr"])
        agg[config]["doc_recall"].append(dm["recall"])
        agg[config]["doc_mrr"].append(dm["mrr"])
        agg[config]["doc_ndcg"].append(dm["ndcg"])
        # 分域
        qa_kinds = {_kind_of_doc(d) for d in doc_gold_set}
        for k in qa_kinds or {"misc"}:
            kind_recall10[k][config].append(dm["recall"][10])

    for config in ("hybrid", "rerank"):
        agg[config] = {"chunk_recall": [], "chunk_mrr": [],
                       "doc_recall": [], "doc_mrr": [], "doc_ndcg": []}

    max_k = max(top_ks)
    for qa in questions:
        q = qa["question"]
        if cache and q in cache:
            hybrid = cache[q]["raw"]["hybrid"]
            reranked = cache[q]["raw"]["rerank"]
            lat_hybrid.append(cache[q]["raw"].get("hybrid_ms", 0))
            lat_rerank.append(cache[q]["raw"].get("rerank_ms", 0))
        else:
            t0 = time.perf_counter()
            q_vec = embed_query(q)
            hybrid = p.hybrid.retrieve(q, q_vec)
            t1 = time.perf_counter()
            lat_hybrid.append((t1 - t0) * 1000)
            reranked = rerank(q, hybrid, top_k=max_k)
            lat_rerank.append((time.perf_counter() - t1) * 1000)

        # Hybrid（无 Rerank）
        chunk_ranks_h = [i + 1 for i, r in enumerate(hybrid[:max_k])
                         if r.chunk_id in chunk_gold[qa["id"]]]
        doc_order_h: list[str] = []
        for r in hybrid[:max_k]:
            d = r.chunk.doc_id if r.chunk else r.chunk_id
            if d not in doc_order_h:
                doc_order_h.append(d)
        bucket("hybrid", qa, chunk_ranks_h, doc_order_h, doc_gold[qa["id"]])

        # Hybrid + Rerank（非缓存路径已在上面算好 reranked）
        chunk_ranks_r = [i + 1 for i, r in enumerate(reranked)
                         if r.chunk_id in chunk_gold[qa["id"]]]
        doc_order_r: list[str] = []
        for r in reranked:
            d = r.chunk.doc_id if r.chunk else r.chunk_id
            if d not in doc_order_r:
                doc_order_r.append(d)
        bucket("rerank", qa, chunk_ranks_r, doc_order_r, doc_gold[qa["id"]])

    # 汇总
    summary: dict = {"top_ks": list(top_ks), "min_overlap": min_overlap,
                     "configs": {}, "latency_ms": {
                         "hybrid_mean": round(sum(lat_hybrid) / len(lat_hybrid), 1),
                         "rerank_mean": round(sum(lat_rerank) / len(lat_rerank), 1),
                     }, "by_kind": {}}
    for config, d in agg.items():
        n = len(d["doc_recall"])
        summary["configs"][config] = {
            "chunk_recall": {str(k): round(sum(v[k] for v in d["chunk_recall"]) / n, 4)
                             for k in top_ks},
            "chunk_mrr": round(sum(d["chunk_mrr"]) / n, 4),
            "doc_recall": {str(k): round(sum(v[k] for v in d["doc_recall"]) / n, 4)
                           for k in top_ks},
            "doc_mrr": round(sum(d["doc_mrr"]) / n, 4),
            "doc_ndcg": {str(k): round(sum(v[k] for v in d["doc_ndcg"]) / n, 4)
                         for k in top_ks},
        }
    for kind, vals_by_config in kind_recall10.items():
        row = {config: round(sum(v) / len(v), 4) for config, v in vals_by_config.items()}
        summary["by_kind"][DOC_KIND_LABEL.get(kind, kind)] = row
    return summary


def _kind_of_doc(doc_id: str) -> str:
    d = doc_id.replace("\\", "/")
    for prefix in ("data/raw/", "raw/"):
        if d.startswith(prefix):
            d = d[len(prefix):]
            break
    return d.split("/", 1)[0]


# ================================================================ Query Rewrite Ablation
def run_query_rewrite_ablation(p: Pipeline, questions: list[dict],
                               top_ks: tuple[int, ...] = (5, 10, 20),
                               sample_examples: int = 8,
                               allow_llm: bool = True,
                               cache: Optional[dict] = None) -> dict:
    """无改写 / 规则扩展 / LLM 改写 三组，下游统一走 Hybrid 检索。

    LLM 组依赖 OPENAI_API_KEY（兼容 DeepSeek：LEGAL_LLM_BASE_URL / LEGAL_LLM_MODEL）。
    无 Key 时 LLM 组自动回退为规则扩展（组内 used_llm 计数会标明）。
    cache: bench 共享缓存（question -> raw/rule/llm.hybrid + plans）。
    """
    from .query import understand_query

    loaded_ids = [d.doc_id for d in p._docs]
    chunk_gold = build_chunk_gold(questions, p.index, loaded_ids)
    doc_gold = {qa["id"]: set(gold_doc_ids(qa, loaded_ids)) for qa in questions}

    groups = ("raw", "rule", "llm")
    agg: dict[str, dict] = {}
    kind_recall10: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lat: dict[str, list[float]] = defaultdict(list)
    examples: list[dict] = []
    llm_used = 0

    for g in groups:
        agg[g] = {"chunk_recall": [], "chunk_mrr": [], "doc_recall": [],
                  "doc_mrr": [], "doc_ndcg": []}

    def bucket(group: str, qa: dict, chunk_ranks: list[int], doc_order: list[str],
               doc_gold_set: set[str]):
        cm = _chunk_metrics(chunk_ranks, len(chunk_gold[qa["id"]]), top_ks)
        dm = _doc_metrics(doc_order, doc_gold_set, top_ks)
        agg[group]["chunk_recall"].append(cm["recall"])
        agg[group]["chunk_mrr"].append(cm["mrr"])
        agg[group]["doc_recall"].append(dm["recall"])
        agg[group]["doc_mrr"].append(dm["mrr"])
        agg[group]["doc_ndcg"].append(dm["ndcg"])
        for k in {_kind_of_doc(d) for d in doc_gold_set} or {"misc"}:
            kind_recall10[k][group].append(dm["recall"][10])

    for qa in questions:
        q = qa["question"]
        if cache and q in cache:
            c = cache[q]
            hybrid_raw = c["raw"]["hybrid"]
            hybrid_rule = c["rule"]["hybrid"]
            hybrid_llm = c["llm"]["hybrid"]
            plan_rule, plan_llm = c["rule"]["plan"], c["llm"]["plan"]
            lat["raw"].append(c["raw"].get("hybrid_ms", 0))
            lat["rule"].append(c["rule"].get("hybrid_ms", 0))
            lat["llm"].append(c["llm"].get("hybrid_ms", 0))
        else:
            # raw：不做任何改写
            t0 = time.perf_counter()
            hybrid_raw = p.hybrid.retrieve(q, embed_query(q))
            lat["raw"].append((time.perf_counter() - t0) * 1000)

            # rule：规则扩展
            plan_rule = understand_query(q, allow_llm=False)
            q_rule = plan_rule.effective_query
            t1 = time.perf_counter()
            hybrid_rule = p.hybrid.retrieve(q_rule, embed_query(q_rule))
            lat["rule"].append((time.perf_counter() - t1) * 1000)

            # llm：LLM 改写（无 Key 回退规则）
            plan_llm = understand_query(q, allow_llm=allow_llm)
            q_llm = plan_llm.effective_query
            t2 = time.perf_counter()
            hybrid_llm = p.hybrid.retrieve(q_llm, embed_query(q_llm))
            lat["llm"].append((time.perf_counter() - t2) * 1000)
        llm_used += 1 if plan_llm.used_llm else 0

        for group, hybrid in (("raw", hybrid_raw), ("rule", hybrid_rule),
                              ("llm", hybrid_llm)):
            chunk_ranks = [i + 1 for i, r in enumerate(hybrid[:max(top_ks)])
                           if r.chunk_id in chunk_gold[qa["id"]]]
            doc_order: list[str] = []
            for r in hybrid[:max(top_ks)]:
                d = r.chunk.doc_id if r.chunk else r.chunk_id
                if d not in doc_order:
                    doc_order.append(d)
            bucket(group, qa, chunk_ranks, doc_order, doc_gold[qa["id"]])

        if len(examples) < sample_examples:
            if cache and q in cache:
                q_rule = cache[q]["rule"]["query"]
                q_llm = cache[q]["llm"]["query"]
            examples.append({
                "question": q,
                "raw": q,
                "rule": q_rule,
                "llm": q_llm,
                "llm_used": plan_llm.used_llm,
            })

    summary: dict = {"top_ks": list(top_ks), "groups": {}, "by_kind": {},
                     "llm_used_queries": llm_used, "examples": examples,
                     "latency_ms": {g: round(sum(v) / len(v), 1) for g, v in lat.items()}}
    for g in groups:
        n = len(agg[g]["doc_recall"])
        summary["groups"][g] = {
            "chunk_recall": {str(k): round(sum(v[k] for v in agg[g]["chunk_recall"]) / n, 4)
                             for k in top_ks},
            "chunk_mrr": round(sum(agg[g]["chunk_mrr"]) / n, 4),
            "doc_recall": {str(k): round(sum(v[k] for v in agg[g]["doc_recall"]) / n, 4)
                           for k in top_ks},
            "doc_mrr": round(sum(agg[g]["doc_mrr"]) / n, 4),
            "doc_ndcg": {str(k): round(sum(v[k] for v in agg[g]["doc_ndcg"]) / n, 4)
                         for k in top_ks},
        }
    for kind, vals in kind_recall10.items():
        summary["by_kind"][DOC_KIND_LABEL.get(kind, kind)] = {
            g: round(sum(v) / len(v), 4) for g, v in vals.items()}
    return summary


# ================================================================ 报告输出
def write_ablation_report(result: dict, name: str, tag: str = "v1") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"ablation_{name}_{tag}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = result.get("configs") or result.get("groups") or {}
    lines = [f"# Ablation: {name}（{tag}）", "", f"- Top-K 档位：{result['top_ks']}", ""]
    lines += ["## 总表", "",
              "| 配置 | chunk R@5 | chunk R@10 | chunk R@20 | chunk MRR | doc R@5 | doc R@10 | doc R@20 | doc MRR | NDCG@10 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for config, m in rows.items():
        cr, dr = m["chunk_recall"], m["doc_recall"]
        lines.append(
            f"| {config} | {cr['5']} | {cr['10']} | {cr['20']} | {m['chunk_mrr']} | "
            f"{dr['5']} | {dr['10']} | {dr['20']} | {m['doc_mrr']} | {m['doc_ndcg']['10']} |")
    lines += ["", "## 延迟分解（ms/query）", ""]
    lat_labels = {"hybrid_mean": "Hybrid（embedding+dense+bm25+rrf）",
                  "rerank_mean": "Rerank（CrossEncoder）",
                  "raw": "原始查询", "rule": "规则扩展", "llm": "LLM 改写"}
    for k, v in (result.get("latency_ms") or {}).items():
        lines.append(f"- {lat_labels.get(k, k)}：{v}")
    lines.append("")

    by_kind = result.get("by_kind", {})
    if by_kind:
        col_labels = {"hybrid": "Hybrid", "rerank": "Rerank",
                      "raw": "原始", "rule": "规则", "llm": "LLM"}
        cols = list(next(iter(by_kind.values())).keys())
        lines += ["## 按文档类型 Recall@10（doc 级）", "",
                  "| 类型 | " + " | ".join(col_labels.get(c, c) for c in cols) + " |",
                  "| --- | " + " | ".join(["---"] * len(cols)) + " |"]
        for kind in ("法规", "制度", "合同", "案例"):
            row = by_kind.get(kind, {})
            lines.append(f"| {kind} | " + " | ".join(str(row.get(c, "-")) for c in cols) + " |")
        lines.append("")

    examples = result.get("examples")
    if examples:
        lines += ["## 改写示例（Query Rewrite）", "",
                  "| 问题 | 原始 | 规则 | LLM |", "| --- | --- | --- | --- |"]
        for ex in examples:
            lines.append(f"| {ex['question']} | {ex['raw']} | {ex['rule']} | {ex['llm']} |")
        lines.append("")

    n_llm = result.get("llm_used_queries")
    if n_llm is not None:
        total = len(result.get("examples", []))
        if n_llm == 0:
            lines += ["> ⚠️ **LLM 组未使用真实 LLM**：未检测到 OPENAI_API_KEY，结果与规则组相同。",
                      "> 配置后重跑：`$env:OPENAI_API_KEY=...; $env:LEGAL_LLM_BASE_URL=https://api.deepseek.com;",
                      "> $env:LEGAL_LLM_MODEL=deepseek-chat; python -m rag.cli ablate --exp query-rewrite --tag v1`", ""]
        else:
            lines.append(f"> LLM 组实际改写 {n_llm}/{total} 条查询。{''}")

    lines.append("")
    lines.append("*由 Ablation Harness 生成，完整明细见同名 JSON。*")
    md_path = REPORT_DIR / f"ablation_{name}_{tag}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Ablation 报告: %s / %s", json_path.name, md_path.name)
    return md_path
