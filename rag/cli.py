# -*- coding: utf-8 -*-
"""命令行入口：python -m rag.cli <command>

命令：
  index     构建索引（解析 -> Document Tree -> Chunk -> Embedding -> 入库）
  query     交互式问答
  evaluate  用 data/evaluation/qa.json 跑 Retrieval/Generation 评估
  all       index + evaluate
  stats     数据统计
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _pipeline(args) -> "Pipeline":
    from .pipeline import Pipeline
    return Pipeline(vector_backend=args.vector_backend or "local")


def cmd_index(args) -> None:
    p = _pipeline(args)
    stats = p.build_index(rebuild=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_query(args) -> None:
    p = _pipeline(args)
    p.build_index()
    if args.question:
        questions = [args.question]
    else:
        print("输入问题（输入 q 退出）：")
        questions = iter(sys.stdin)
    for q in questions:
        q = q.strip()
        if not q or q.lower() == "q":
            break
        _, ctx, ans = p.answer(q, use_rerank=not args.no_rerank,
                               allow_llm_rewrite=args.llm_rewrite)
        print("\n=== 问题 ===")
        print(q)
        print("=== 回答 ===")
        print(ans.answer)
        print("=== 引用 ===")
        for c in ans.citations:
            print(f"  - {c}")
        if ans.unsupported:
            print(f"=== 无支撑句子({ans.groundedness}) ===")
            for s in ans.unsupported:
                print(f"  ! {s}")
        print()


def cmd_evaluate(args) -> None:
    p = _pipeline(args)
    p.build_index()
    result = p.evaluate(tag=args.tag)
    print(json.dumps({"retrieval": result["retrieval"],
                      "generation": result["generation"],
                      "report": result["report_path"]},
                     ensure_ascii=False, indent=2))


def cmd_all(args) -> None:
    cmd_index(args)
    cmd_evaluate(args)


def cmd_ablate(args) -> None:
    """消融实验：--exp rerank | query-rewrite。"""
    import json as _json
    from . import config as _cfg
    from .ablation import (run_query_rewrite_ablation, run_rerank_ablation,
                           write_ablation_report)

    p = _pipeline(args)
    p.build_index()
    data = _json.loads((_cfg.EVAL_DIR / "qa.json").read_text(encoding="utf-8"))
    questions = data["questions"]

    if args.exp == "rerank":
        result = run_rerank_ablation(p, questions)
        path = write_ablation_report(result, "rerank", args.tag)
        print(_json.dumps({"configs": result["configs"], "by_kind": result["by_kind"],
                           "latency_ms": result["latency_ms"], "report": str(path)},
                          ensure_ascii=False, indent=2))
    elif args.exp == "query-rewrite":
        result = run_query_rewrite_ablation(p, questions, allow_llm=not args.no_llm)
        path = write_ablation_report(result, "query_rewrite", args.tag)
        print(_json.dumps({"groups": result["groups"], "by_kind": result["by_kind"],
                           "llm_used_queries": result["llm_used_queries"],
                           "latency_ms": result["latency_ms"], "report": str(path)},
                          ensure_ascii=False, indent=2))
    else:
        raise SystemExit(f"未知实验: {args.exp}")


def cmd_qa_class(args) -> None:
    """③ QA 分类 + Error Analysis。"""
    import json as _json
    from . import config as _cfg
    from .qa_analysis import run_qa_class_analysis, write_qa_class_report

    p = _pipeline(args)
    p.build_index()
    data = _json.loads((_cfg.EVAL_DIR / "qa.json").read_text(encoding="utf-8"))
    result = run_qa_class_analysis(p, data["questions"])
    path = write_qa_class_report(result, args.tag)
    print(_json.dumps({"domain_metrics": result["domain_metrics"],
                       "bucket_summary": result["bucket_summary"],
                       "report": str(path)}, ensure_ascii=False, indent=2))


def cmd_stats(args) -> None:
    from .loader import discover_files
    files = discover_files()
    from collections import Counter
    kinds = Counter()
    for f in files:
        parts = f.relative_to(__import__("rag.config", fromlist=["RAW_DIR"]).RAW_DIR).parts
        kinds[parts[0]] += 1
    print("文档统计：")
    for k, v in sorted(kinds.items()):
        print(f"  {k}: {v}")
    print(f"  合计: {len(files)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Legal Knowledge RAG Core")
    parser.add_argument("--vector-backend", choices=["local", "milvus"], default="local",
                        help="向量库后端（默认 local，Milvus 需服务）")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="构建索引")
    sub.add_parser("stats", help="数据统计")

    qp = sub.add_parser("query", help="问答")
    qp.add_argument("question", nargs="?", default=None)
    qp.add_argument("--no-rerank", action="store_true")
    qp.add_argument("--llm-rewrite", action="store_true")

    ep = sub.add_parser("evaluate", help="跑评估")
    ep.add_argument("--tag", default="default")

    ap = sub.add_parser("ablate", help="消融实验")
    ap.add_argument("--exp", choices=["rerank", "query-rewrite"], default="rerank")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--no-llm", action="store_true",
                    help="query-rewrite 时禁用 LLM 改写（无 Key 时自动回退规则，无需本参数）")

    qp = sub.add_parser("qa-class", help="③ QA 分类 + Error Analysis")
    qp.add_argument("--tag", default="v1")

    sub.add_parser("all", help="index + evaluate")

    args = parser.parse_args()
    {"index": cmd_index, "query": cmd_query, "evaluate": cmd_evaluate,
     "all": cmd_all, "stats": cmd_stats, "ablate": cmd_ablate,
     "qa-class": cmd_qa_class}[args.command](args)


if __name__ == "__main__":
    main()
