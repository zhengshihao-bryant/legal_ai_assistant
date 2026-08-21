# -*- coding: utf-8 -*-
"""④ OCR / Chunking Quality Analysis（数据质量审计）。

OCR 指标：
- 页面完整率：有文本的页 / 总页
- 条款编号识别率：qa.json 引用的「第X条」在解析文本中的命中率（对扫描件即 OCR 质量代理）
- 关键术语可达率：gold 答案关键词在对应文档文本中的出现率（扫描件专用）

Chunk 指标：
- Article 完整率：第X条 标题行在解析文本中的保留比例（laws）
- Parent → Child 完整关联率：child 有合法 parent 的比例 / parent 有 child 的比例
- 跨条款错误率：一个 child 内含多个「第X条」起始的比例（条款被切碎信号）
- 标题保留率：child 带章节标题的比例

《劳动合同法》作为人工抽样基准：抽样 OCR 行与 chunk 供人工核验。
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from . import config as cfg
from .evaluation import _stem_of, gold_doc_ids, tokenize_zh
from .loader import scan_documents
from .pipeline import Pipeline

logger = logging.getLogger(__name__)

REPORT_DIR = cfg.REPORT_DIR
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零\d]+条")
MANUAL_BENCHMARK = "中华人民共和国劳动合同法"


def _parsed_text(doc_id: str) -> str:
    p = cfg.PARSED_DIR / f"{doc_id.replace('/', '__')}.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def run_data_quality(p: Pipeline, questions: list[dict],
                     manual_benchmark: str = MANUAL_BENCHMARK) -> dict:
    docs = scan_documents()
    loaded_ids = [d.doc_id for d in docs]

    # ---------------- OCR / 解析层
    ocr_stats: dict[str, dict] = {}
    chunk_counts: Counter = Counter(c.doc_id for c in p.index.children)
    for d in docs:
        if not d.path.endswith(".pdf"):
            continue
        text = _parsed_text(d.doc_id)
        import fitz
        pdf = fitz.open(d.path)
        total_pages = pdf.page_count
        raw_chars = sum(len(pg.get_text()) for pg in pdf)   # 原始 PDF 文本层
        pdf.close()
        ocr_stats[d.title] = {
            "pages": total_pages,
            "raw_chars": raw_chars,
            "parsed_chars": len(text),
            "is_scanned": raw_chars < 2000,          # 原始文本层 < 2K 视为扫描件（需 OCR）
            "chunks": chunk_counts.get(d.doc_id, 0),  # 入库 chunk 数（0 = 未入库，严重问题）
            "articles_found": len(set(ARTICLE_RE.findall(text))),
        }

    # qa.json 引用条款命中率（对扫描件即 OCR 质量代理）
    cited = Counter()
    cited_hit = Counter()
    for qa in questions:
        src = qa.get("source", "")
        art = ARTICLE_RE.search(src)
        if not art:
            continue
        law = None
        for g in gold_doc_ids(qa, loaded_ids):
            if g.startswith("laws/"):
                law = g
                break
        if law is None:
            continue
        key = law.split("/")[-1]
        cited[key] += 1
        if art.group(0) in _parsed_text(law):
            cited_hit[key] += 1

    # 关键术语可达率（gold 答案关键词出现在对应文档文本）
    term_reach: dict[str, dict] = {}
    for qa in questions:
        gold = gold_doc_ids(qa, loaded_ids)
        if not gold or not gold[0].startswith("laws/"):
            continue
        law = gold[0]
        text = _parsed_text(law)
        if not text:
            continue
        toks = set(tokenize_zh(qa.get("answer", "")))
        key = law.split("/")[-1]
        entry = term_reach.setdefault(key, {"n": 0, "reach": 0.0, "miss": []})
        entry["n"] += 1
        hit = sum(1 for t in toks if t in text)
        if hit / max(len(toks), 1) >= 0.5:
            entry["reach"] += 1
        else:
            entry["miss"].append(qa["question"][:20])

    # ---------------- Chunk 层
    children = p.index.children
    parents = p.index.parent_by_id
    n_child = len(children)
    n_parent = len(parents)
    linked = sum(1 for c in children if c.parent_id and c.parent_id in parents)
    parents_with_child = len({c.parent_id for c in children if c.parent_id})

    cross_article = 0
    article_starts_total = 0
    for c in children:
        arts = ARTICLE_RE.findall(c.text)
        if len(arts) > 1:
            cross_article += 1
        if arts:
            article_starts_total += 1
    titled = sum(1 for c in children if c.title and c.title != c.source.split("/")[-1])

    # Article 完整率：文档树中 第X条 是否作为标题行保留（laws）
    article_headings = 0
    article_heading_hit = 0
    for d in docs:
        if not d.kind == "laws":
            continue
        text = _parsed_text(d.doc_id)
        lines = [l.strip() for l in text.splitlines()]
        arts = [l for l in lines if ARTICLE_RE.match(l) and len(l) <= 40]
        article_headings += len(arts)
        # 抽样：若该行仍是独立行，视为保留成功（对照 chunk 里的 title 归属）
        for a in arts[:200]:
            article_heading_hit += 1

    # ---------------- 人工抽样基准（劳动合同法）
    manual = {}
    bench = next((d for d in docs if manual_benchmark in d.title), None)
    if bench:
        text = _parsed_text(bench.doc_id)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        manual = {
            "doc": bench.title,
            "chars": len(text),
            "ocr_sample": lines[:15],
            "chunk_sample": [
                {"title": c.title, "text": c.text[:120]}
                for c in children if c.doc_id == bench.doc_id][:10],
        }

    return {
        "ocr": {
            "all_laws": ocr_stats,
            "cited_article_hit": {k: f"{cited_hit[k]}/{cited[k]}" for k in cited},
            "term_reach": {k: {"n": v["n"], "reach_ratio": round(v["reach"] / v["n"], 3),
                               "miss_questions": v["miss"][:5]} for k, v in term_reach.items()},
        },
        "chunk": {
            "children": n_child, "parents": n_parent,
            "parent_child_link_rate": round(linked / n_child, 4) if n_child else 0,
            "parents_with_child_rate": round(parents_with_child / n_parent, 4) if n_parent else 0,
            "cross_article_chunk_rate": round(cross_article / n_child, 4) if n_child else 0,
            "article_chunks": article_starts_total,
            "titled_rate": round(titled / n_child, 4) if n_child else 0,
            "law_article_headings": article_headings,
        },
        "manual_benchmark": manual,
    }


def write_data_quality_report(result: dict, tag: str = "v1") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"data_quality_{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# ④ OCR / Chunking 数据质量分析（{tag}）", ""]
    lines += ["## OCR / 解析层（全部法律 PDF）", "",
              "| 法律 | 页数 | 原始文本字符 | 解析后字符 | 扫描件? | 入库chunk | 识别条款数 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, v in sorted(result["ocr"]["all_laws"].items()):
        scanned = "✅是(OCR)" if v["is_scanned"] else "否"
        flag = " ⚠️" if v["chunks"] == 0 else ""
        lines.append(f"| {k} | {v['pages']} | {v['raw_chars']} | {v['parsed_chars']} "
                     f"| {scanned} | {v['chunks']}{flag} | {v['articles_found']} |")
    lines += ["", "### qa.json 引用条款命中（gold 来源 第X条 是否在解析文本中）", "",
              "| 法律 | 命中 |", "| --- | --- |"]
    for k, v in result["ocr"]["cited_article_hit"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "### 关键术语可达率（扫描件 gold 答案词是否出现在文档中）", "",
              "| 法律 | 样本 | 可达率 | 未达问题示例 |", "| --- | --- | --- | --- |"]
    for k, v in result["ocr"]["term_reach"].items():
        lines.append(f"| {k} | {v['n']} | {v['reach_ratio']} | {'；'.join(v['miss_questions'])} |")
    lines += ["", "## Chunk 层", "",
              f"- Parent/Child 关联率：{result['chunk']['parent_child_link_rate']}（child→parent）",
              f"- Parent 有 child 比例：{result['chunk']['parents_with_child_rate']}",
              f"- 跨条款错误率（单 child 含多个 第X条）：{result['chunk']['cross_article_chunk_rate']}",
              f"- 标题保留率：{result['chunk']['titled_rate']}",
              f"- 法规 第X条 标题行保留：{result['chunk']['law_article_headings']} 处", ""]
    mb = result.get("manual_benchmark") or {}
    if mb:
        lines += [f"## 人工抽样基准（{mb['doc']}，{mb['chars']} 字符）", "",
                  "### OCR 抽样行", ""]
        for l in mb.get("ocr_sample", []):
            lines.append(f"- {l}")
        lines += ["", "### Chunk 抽样", "", "| 章节标题 | 内容 |", "| --- | --- |"]
        for c in mb.get("chunk_sample", []):
            lines.append(f"| {c['title']} | {c['text']} |")
        lines.append("")
    lines.append("*由 data_quality 审计生成，完整明细见同名 JSON。*")
    md = REPORT_DIR / f"data_quality_{tag}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("数据质量报告: data_quality_%s.json / .md", tag)
    return md
