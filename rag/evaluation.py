# -*- coding: utf-8 -*-
"""Retrieval / Generation 分层评估。

数据：data/evaluation/qa.json（100 条），每条含 related_docs（文档级 gold）。

- Retrieval 层：Recall@k / MRR / NDCG@k（文档级命中）
- Generation 层：EM / F1 / ROUGE-1 / ROUGE-2 / ROUGE-L / 引用率 / 接地性
- 输出：JSON 明细 + Markdown 报告 + 错误案例分析
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import jieba

from .config import EVAL_DIR, EVAL_RERANK_TOP_K, EVAL_TOP_KS, REPORT_DIR

logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.WARNING)

_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])")


# ---------------------------------------------------------------- 文档级 gold
def gold_doc_ids(qa: dict, loaded_ids: list[str]) -> list[str]:
    """把 related_docs 路径归一化为已加载 doc_id（处理 md/docx 双格式）。"""
    loaded = {_stem_of(d): d for d in loaded_ids}
    gold = []
    for path in qa.get("related_docs", []):
        stem = _stem_of(path)
        if stem in loaded:
            gold.append(loaded[stem])
        else:
            gold.append(path)  # 保底：原路径
    return gold


def _stem_of(path: str) -> str:
    """路径归一化为 (kind/文档名) 无扩展形式，兼容 data/raw/ 前缀与 md/docx 双格式。"""
    p = path.replace("\\", "/")
    for suf in (".docx", ".md", ".pdf", ".txt"):
        if p.endswith(suf):
            p = p[: -len(suf)]
            break
    p = p.lower().strip("/")
    for prefix in ("data/raw/", "raw/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return p


# ---------------------------------------------------------------- 检索评估
@dataclass
class RetrievalMetrics:
    recall_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "recall_at": {str(k): round(v, 4) for k, v in self.recall_at.items()},
            "mrr": round(self.mrr, 4),
            "ndcg_at": {str(k): round(v, 4) for k, v in self.ndcg_at.items()},
        }


def _ndcg(ranks: list[int], gold_count: int, k: int) -> float:
    """ranks: 命中文档在结果中的位置（1-based）。文档级二值相关。"""
    if not ranks:
        return 0.0
    dcg = sum(1.0 / _log2(r + 1) for r in ranks if r <= k)
    ideal = sum(1.0 / _log2(i + 1) for i in range(1, min(gold_count, k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def _log2(x: int) -> float:
    import math
    return math.log2(x + 1) if x > 0 else 1.0


def evaluate_retrieval(questions: list[dict],
                       retrieve_fn: Callable[[str, int], list[str]],
                       doc_of: Callable[[str], str],
                       loaded_ids: list[str],
                       top_ks: tuple[int, ...] = EVAL_TOP_KS) -> dict:
    """retrieve_fn(query, k) -> 按序返回 chunk_id 列表；doc_of(chunk_id) -> doc_id。"""
    per_question: list[dict] = []
    agg_recall = {k: [] for k in top_ks}
    first_hits: list[float] = []
    ndcgs = {k: [] for k in top_ks}
    doc_hit_cache: dict[str, set[str]] = {}

    for qa in questions:
        q = qa["question"]
        gold = gold_doc_ids(qa, loaded_ids)
        gold_set = set(gold)
        top_k = max(top_ks)
        chunk_ids = retrieve_fn(q, top_k)
        doc_order: list[str] = []
        for cid in chunk_ids:
            d = doc_of(cid)
            if d not in doc_order:
                doc_order.append(d)
        # 命中的 gold 文档位置（1-based）
        ranks = [i + 1 for i, d in enumerate(doc_order) if d in gold_set]
        mrr = 1.0 / ranks[0] if ranks else 0.0
        first_hits.append(mrr)
        row: dict = {"id": qa["id"], "question": q, "gold_docs": gold,
                     "hit_ranks": ranks, "mrr": mrr}
        for k in top_ks:
            hit = any(r <= k for r in ranks)
            agg_recall[k].append(1.0 if hit else 0.0)
            ndcgs[k].append(_ndcg(ranks, len(gold_set), k))
            row[f"recall@{k}"] = hit
        per_question.append(row)

    metrics = RetrievalMetrics(
        recall_at={k: sum(v) / len(v) for k, v in agg_recall.items()},
        mrr=sum(first_hits) / len(first_hits),
        ndcg_at={k: sum(v) / len(v) for k, v in ndcgs.items()},
    )
    return {"metrics": metrics.to_dict(), "per_question": per_question}


# ---------------------------------------------------------------- 生成评估
def tokenize_zh(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip() and t not in " \t，。；：、（）()《》“”‘’？！!?.,;:[]"]


def _f1(pred: list[str], gold: list[str]) -> float:
    from collections import Counter
    if not pred or not gold:
        return 0.0
    pc, gc = Counter(pred), Counter(gold)
    tp = sum((pc & gc).values())
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _rouge_l(pred: list[str], gold: list[str]) -> float:
    """ROUGE-L：基于 LCS 的 F1。"""
    m, n = len(pred), len(gold)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gold[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def evaluate_generation(questions: list[dict],
                        generate_fn: Callable[[dict], object]) -> dict:
    """generate_fn(qa) -> Answer（含 answer/citations/groundedness/unsupported）。"""
    per_question: list[dict] = []
    agg: dict[str, list[float]] = defaultdict(list)
    citation_rates: list[float] = []
    groundedness_vals: list[float] = []
    errors: list[dict] = []

    for qa in questions:
        ans = generate_fn(qa)
        gold_toks = tokenize_zh(qa.get("answer", ""))
        pred_toks = tokenize_zh(ans.answer)
        em = 1.0 if pred_toks == gold_toks else 0.0
        f1 = _f1(pred_toks, gold_toks)
        # ROUGE-1/2（F1）
        p1, g1 = set(pred_toks), set(gold_toks)
        r1 = 2 * len(p1 & g1) / (len(p1) + len(g1)) if (p1 and g1) else 0.0
        p2, g2 = _ngrams(pred_toks, 2), _ngrams(gold_toks, 2)
        r2 = 2 * len(p2 & g2) / (len(p2) + len(g2)) if (p2 and g2) else 0.0
        rl = _rouge_l(pred_toks, gold_toks)
        has_citation = 1.0 if ans.citations else 0.0

        agg["em"].append(em)
        agg["f1"].append(f1)
        agg["rouge1"].append(r1)
        agg["rouge2"].append(r2)
        agg["rougeL"].append(rl)
        citation_rates.append(has_citation)
        groundedness_vals.append(ans.groundedness)

        row = {
            "id": qa["id"], "question": qa["question"],
            "gold_answer": qa.get("answer", ""),
            "pred_answer": ans.answer,
            "em": em, "f1": round(f1, 4), "rouge1": round(r1, 4),
            "rouge2": round(r2, 4), "rougeL": round(rl, 4),
            "citations": ans.citations,
            "groundedness": ans.groundedness,
            "unsupported": ans.unsupported,
        }
        per_question.append(row)
        if f1 < 0.25 or not has_citation or ans.groundedness < 0.5:
            errors.append(row)

    summary = {
        "em": round(sum(agg["em"]) / len(agg["em"]), 4),
        "f1": round(sum(agg["f1"]) / len(agg["f1"]), 4),
        "rouge1": round(sum(agg["rouge1"]) / len(agg["rouge1"]), 4),
        "rouge2": round(sum(agg["rouge2"]) / len(agg["rouge2"]), 4),
        "rougeL": round(sum(agg["rougeL"]) / len(agg["rougeL"]), 4),
        "citation_rate": round(sum(citation_rates) / len(citation_rates), 4),
        "avg_groundedness": round(sum(groundedness_vals) / len(groundedness_vals), 4),
    }
    return {"metrics": summary, "per_question": per_question, "error_cases": errors[:15]}


# ---------------------------------------------------------------- 报告输出
def write_reports(retrieval_result: dict, generation_result: dict,
                  questions: list[dict], out_dir: Path = REPORT_DIR,
                  tag: str = "default") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "tag": tag,
        "num_questions": len(questions),
        "retrieval": retrieval_result.get("by_stage", {}),
        "generation": generation_result["metrics"],
        "error_cases": generation_result["error_cases"],
        "per_question": {
            "retrieval": retrieval_result.get("per_question", []),
            "generation": generation_result["per_question"],
        },
    }
    json_path = out_dir / f"report_{tag}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = _render_markdown(report)
    md_path = out_dir / f"report_{tag}.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("评估报告已写入: %s / %s", json_path.name, md_path.name)
    return md_path


def _render_markdown(report: dict) -> str:
    r_all = report["retrieval"]          # {stage: metrics} 或 {"by_stage": {...}}
    g = report["generation"]
    by_stage = r_all.get("by_stage", {}) if isinstance(r_all, dict) and "by_stage" in r_all else (r_all or {})
    lines = [
        f"# RAG 评估报告（{report['tag']}）",
        "",
        f"- 样本数：{report['num_questions']}",
        "",
        "## Retrieval 层（四阶段对比）",
        "",
        "| 阶段 | Recall@5 | Recall@10 | Recall@20 | MRR | NDCG@10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    stage_order = ("dense", "bm25", "hybrid", "rerank")
    stage_label = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid(RRF)", "rerank": "Rerank"}
    for s in stage_order:
        m = by_stage.get(s)
        if not m:
            continue
        lines.append(
            f"| {stage_label.get(s, s)} | {m['recall_at'].get('5', 0)} | "
            f"{m['recall_at'].get('10', 0)} | {m['recall_at'].get('20', 0)} | "
            f"{m.get('mrr', 0)} | {m['ndcg_at'].get('10', 0)} |"
        )
    lines += ["", "## Generation 层", "", "| 指标 | 值 |", "| --- | --- |"]
    for k, v in g.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 错误案例分析", ""]
    if report.get("error_cases"):
        for ec in report["error_cases"][:10]:
            lines.append(f"### {ec['id']} · {ec['question']}")
            lines.append("")
            lines.append(f"- F1: {ec['f1']} · 引用: {bool(ec['citations'])} · 接地性: {ec['groundedness']}")
            lines.append(f"- 预测: {ec['pred_answer'][:120]}")
            lines.append(f"- 参考: {ec['gold_answer'][:120]}")
            if ec.get("unsupported"):
                lines.append(f"- 无支撑句: {ec['unsupported'][:3]}")
            lines.append("")
    else:
        lines.append("无显著错误案例。")
    lines.append("")
    lines.append(f"*生成时间：{__import__('datetime').datetime.now().isoformat(timespec='seconds')}*")
    return "\n".join(lines)
