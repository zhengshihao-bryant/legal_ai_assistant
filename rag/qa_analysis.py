# -*- coding: utf-8 -*-
"""③ QA 分类 + Error Analysis。

- 按文档类型（法规/制度/合同/案例）拆分 Retrieval / Generation 指标
- 错误案例分桶：
    A. 检索失败：gold 文档未进 top-10
    B. 证据命中但生成差：recall@10 命中但 F1 < 阈值（本地抽取式上限）
    C. 引用缺失：生成答案无引用
    D. 接地性风险：存在无支撑句
- 每个失败桶给出代表案例（问题 / gold 文档 / 实际检索到 / 预测 vs 参考）
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from . import config as cfg
from .evaluation import gold_doc_ids, tokenize_zh
from .pipeline import Pipeline

logger = logging.getLogger(__name__)

REPORT_DIR = cfg.REPORT_DIR
KIND_LABEL = {"laws": "法规", "policies": "制度", "contracts": "合同", "cases": "案例"}
F1_THRESHOLD = 0.25


def _kind_of(doc_id: str) -> str:
    d = doc_id.replace("\\", "/")
    for pre in ("data/raw/", "raw/"):
        if d.startswith(pre):
            d = d[len(pre):]
            break
    return d.split("/", 1)[0]


def run_qa_class_analysis(p: Pipeline, questions: list[dict],
                          f1_threshold: float = F1_THRESHOLD) -> dict:
    loaded_ids = [d.doc_id for d in p._docs]

    # 每题的检索细节（hybrid + rerank，top-10）
    qa_rows: list[dict] = []
    for qa in questions:
        q = qa["question"]
        _, final = p.retrieve(q, top_k=10, use_rerank=True)
        doc_order: list[str] = []
        for r in final:
            d = r.chunk.doc_id if r.chunk else r.chunk_id
            if d not in doc_order:
                doc_order.append(d)
        gold = set(gold_doc_ids(qa, loaded_ids))
        hit_ranks = [i + 1 for i, d in enumerate(doc_order) if d in gold]
        qa_rows.append({
            "id": qa["id"], "type": qa.get("type", ""), "question": q,
            "gold_docs": sorted(gold), "retrieved_docs": doc_order[:10],
            "hit": any(r <= 10 for r in hit_ranks), "hit_ranks": hit_ranks,
        })

    # 生成层（本地抽取式兜底）
    gen_rows: dict[str, dict] = {}
    for qa in questions:
        _, _, ans = p.answer(qa["question"], use_rerank=True)
        pred = tokenize_zh(ans.answer)
        gold = tokenize_zh(qa.get("answer", ""))
        f1 = _f1(pred, gold)
        gen_rows[qa["id"]] = {
            "f1": round(f1, 4),
            "citations": ans.citations,
            "groundedness": ans.groundedness,
            "unsupported": ans.unsupported,
            "pred": ans.answer[:150],
            "gold": qa.get("answer", "")[:150],
        }

    # 按类型聚合
    domain_metrics: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for row in qa_rows:
        kinds = {KIND_LABEL.get(_kind_of(d), _kind_of(d)) for d in row["gold_docs"]}
        for k in kinds or {"其他"}:
            domain_metrics[k]["recall10"].append(1.0 if row["hit"] else 0.0)
            domain_metrics[k]["f1"].append(gen_rows[row["id"]]["f1"])
            domain_metrics[k]["citation"].append(1.0 if gen_rows[row["id"]]["citations"] else 0.0)
            domain_metrics[k]["grounded"].append(gen_rows[row["id"]]["groundedness"])

    domain_summary = {}
    for k, v in domain_metrics.items():
        n = len(v["recall10"])
        domain_summary[k] = {
            "n": n,
            "recall10": round(sum(v["recall10"]) / n, 4),
            "f1": round(sum(v["f1"]) / n, 4),
            "citation_rate": round(sum(v["citation"]) / n, 4),
            "groundedness": round(sum(v["grounded"]) / n, 4),
        }

    # 错误分桶
    buckets = {"A_检索失败": [], "B_证据命中生成差": [], "C_引用缺失": [], "D_接地性风险": []}
    for row in qa_rows:
        g = gen_rows[row["id"]]
        if not row["hit"]:
            buckets["A_检索失败"].append(row["id"])
        elif g["f1"] < f1_threshold:
            buckets["B_证据命中生成差"].append(row["id"])
        if not g["citations"]:
            buckets["C_引用缺失"].append(row["id"])
        if g["unsupported"]:
            buckets["D_接地性风险"].append(row["id"])

    bucket_summary = {k: len(v) for k, v in buckets.items()}

    # 代表案例（每桶取 3 条：问题 + gold + 检索到 + 预测/参考）
    cases: dict[str, list[dict]] = {}
    for bname, ids in buckets.items():
        picked = []
        for qid in ids[:3]:
            row = next(r for r in qa_rows if r["id"] == qid)
            g = gen_rows[qid]
            picked.append({
                "id": qid, "type": row["type"], "question": row["question"],
                "gold_docs": row["gold_docs"], "retrieved_top3": row["retrieved_docs"][:3],
                "hit_ranks": row["hit_ranks"],
                "pred_answer": g["pred"], "gold_answer": g["gold"],
                "f1": g["f1"], "groundedness": g["groundedness"],
            })
        cases[bname] = picked

    return {
        "n_questions": len(questions),
        "f1_threshold": f1_threshold,
        "domain_metrics": domain_summary,
        "bucket_summary": bucket_summary,
        "cases": cases,
    }


def _f1(pred: list[str], gold: list[str]) -> float:
    from collections import Counter
    if not pred or not gold:
        return 0.0
    pc, gc = Counter(pred), Counter(gold)
    tp = sum((pc & gc).values())
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def write_qa_class_report(result: dict, tag: str = "v1") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"qa_class_{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# QA 分类 + Error Analysis（{tag}）", "", f"- 样本数：{result['n_questions']}", "",
             "## 按文档类型拆分指标", "",
             "| 类型 | 样本 | Recall@10 | F1 | 引用率 | 接地性 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k in ("法规", "制度", "合同", "案例"):
        m = result["domain_metrics"].get(k, {})
        lines.append(f"| {k} | {m.get('n', 0)} | {m.get('recall10', '-')} | {m.get('f1', '-')} "
                     f"| {m.get('citation_rate', '-')} | {m.get('groundedness', '-')} |")
    lines += ["", "## 错误分桶", "", "| 桶 | 数量 | 说明 |", "| --- | --- | --- |"]
    bs = result["bucket_summary"]
    lines += [
        f"| A. 检索失败 | {bs.get('A_检索失败', 0)} | gold 文档未进 top-10（根因待查：分词/OCR/语料缺口） |",
        f"| B. 证据命中但生成差 | {bs.get('B_证据命中生成差', 0)} | 召回正确但 F1<{result['f1_threshold']}（本地抽取式上限） |",
        f"| C. 引用缺失 | {bs.get('C_引用缺失', 0)} | 生成答案无引用标记 |",
        f"| D. 接地性风险 | {bs.get('D_接地性风险', 0)} | 存在无支撑句 |",
    ]
    lines += ["", "## 代表案例", ""]
    for bname, picked in result["cases"].items():
        if not picked:
            continue
        lines += [f"### 桶 {bname}", ""]
        for c in picked:
            lines.append(f"- **{c['id']} · {c['question']}**（{c['type']}，F1={c['f1']}）")
            lines.append(f"  - gold: {c['gold_docs']}")
            lines.append(f"  - 检索到: {c['retrieved_top3']}（命中位次 {c['hit_ranks']}）")
            lines.append(f"  - 预测: {c['pred_answer']}")
            lines.append(f"  - 参考: {c['gold_answer']}")
        lines.append("")
    lines.append("*由 qa_analysis 生成，完整明细见同名 JSON。*")
    md = REPORT_DIR / f"qa_class_{tag}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("QA 分类报告: qa_class_%s.json / qa_class_%s.md", tag, tag)
    return md
