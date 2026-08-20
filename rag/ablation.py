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
                        min_overlap: int = 2) -> dict:
    """Hybrid vs Hybrid+Rerank × Top-K 对比，chunk/doc 双层指标 + 延迟 + 分域。"""
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
        t0 = time.perf_counter()
        q_vec = embed_query(q)
        hybrid = p.hybrid.retrieve(q, q_vec)
        t1 = time.perf_counter()
        lat_hybrid.append((t1 - t0) * 1000)

        # Hybrid（无 Rerank）
        chunk_ranks_h = [i + 1 for i, r in enumerate(hybrid[:max_k])
                         if r.chunk_id in chunk_gold[qa["id"]]]
        doc_order_h: list[str] = []
        for r in hybrid[:max_k]:
            d = r.chunk.doc_id if r.chunk else r.chunk_id
            if d not in doc_order_h:
                doc_order_h.append(d)
        bucket("hybrid", qa, chunk_ranks_h, doc_order_h, doc_gold[qa["id"]])

        # Hybrid + Rerank
        reranked = rerank(q, hybrid, top_k=max_k)
        t2 = time.perf_counter()
        lat_rerank.append((t2 - t1) * 1000)
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


# ================================================================ 报告输出
def write_ablation_report(result: dict, name: str, tag: str = "v1") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"ablation_{name}_{tag}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Ablation: {name}（{tag}）", "", f"- Top-K 档位：{result['top_ks']}", ""]
    lines += ["## 总表", "",
              "| 配置 | chunk R@5 | chunk R@10 | chunk R@20 | chunk MRR | doc R@5 | doc R@10 | doc R@20 | doc MRR | NDCG@10 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for config, m in result["configs"].items():
        cr, dr = m["chunk_recall"], m["doc_recall"]
        lines.append(
            f"| {config} | {cr['5']} | {cr['10']} | {cr['20']} | {m['chunk_mrr']} | "
            f"{dr['5']} | {dr['10']} | {dr['20']} | {m['doc_mrr']} | {m['doc_ndcg']['10']} |")
    lines += ["", "## 延迟分解（ms/query）", "",
              f"- Hybrid（embedding + dense + bm25 + rrf）：{result['latency_ms']['hybrid_mean']}",
              f"- Rerank（CrossEncoder 12→20 对）：{result['latency_ms']['rerank_mean']}", ""]
    lines += ["## 按文档类型 Recall@10（doc 级）", "", "| 类型 | Hybrid | Rerank |", "| --- | --- | --- |"]
    kinds = result["by_kind"]
    for kind in ("法规", "制度", "合同", "案例"):
        row = kinds.get(kind, {})
        lines.append(f"| {kind} | {row.get('hybrid', '-')} | {row.get('rerank', '-')} |")
    lines.append("")
    lines.append("*由 Ablation Harness 生成，完整明细见同名 JSON。*")
    md_path = REPORT_DIR / f"ablation_{name}_{tag}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Ablation 报告: %s / %s", json_path.name, md_path.name)
    return md_path
