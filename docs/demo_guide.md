# Demo Guide — 法律知识助手复现指南

本文档给出从零复现本项目全部结果的命令序列。假设已按根 README 安装依赖。

## 0. 环境

```bash
pip install -r requirements.txt
# 可选：OCR 兜底（扫描件 PDF 需要）
pip install rapidocr-onnxruntime
# 可选：接 DeepSeek LLM（② LLM 改写 / LLM 生成对比）
#   $env:OPENAI_API_KEY="..."            # PowerShell
#   $env:LEGAL_LLM_BASE_URL="https://api.deepseek.com"
#   $env:LEGAL_LLM_MODEL="deepseek-chat"
# 网络不稳且模型已缓存时：
#   $env:LEGAL_OFFLINE="1"
```

模型（bge-base-zh-v1.5 / bge-reranker-base）首次运行自动下载到 `data/models/`。

## 1. 构建索引（解析 → OCR → Document Tree → Chunk → Embedding → 入库）

```bash
python -m rag.cli index
```

- 56+ 篇文档（19 部法律 PDF + 制度/合同/案例），5 部扫描件自动 RapidOCR
- 产物：`data/parsed/`（解析缓存）、`data/chunks/`（Parent/Child 缓存 + manifest）、`data/index/`（向量）
- 二次运行自动复用缓存（约 1 秒），`--vector-backend milvus` 可切 Milvus

## 2. 问答

```bash
python -m rag.cli query "试用期最长多久?"
python -m rag.cli query          # 交互模式
python -m rag.cli query "加班工资如何计算?" --llm-rewrite   # 开启 LLM 查询改写
```

## 3. 主评估（100 条 QA，四阶段检索 + 生成）

```bash
python -m rag.cli evaluate --tag v1
```

产物：`data/evaluation/reports/report_v1.{json,md}`

## 4. 消融实验（统一基准，单进程一次出全部报告）

```bash
python -m rag.cli bench --tag v1
```

一次产出（共享检索结果，约 15 分钟）：
- `report_v1.*` 主评估
- `ablation_rerank_v1.*` ① Rerank Ablation（Hybrid vs +Rerank × Top-K，chunk/doc 双层）
- `ablation_query_rewrite_v1.*` ② Query Rewrite（原始/规则/LLM 三组，LLM 走 DeepSeek）
- `qa_class_v1.*` ③ QA 分类 + Error Analysis
- `data_quality_v1.*` ④ OCR / Chunking 数据质量
- `generation_compare_v1.*` 本地抽取式 vs DeepSeek 生成对比

单独跑某个实验（每次新进程，约 8 分钟）：
```bash
python -m rag.cli ablate --exp rerank --tag v1
python -m rag.cli ablate --exp query-rewrite --tag v1
python -m rag.cli qa-class --tag v1
python -m rag.cli data-quality --tag v1
```

## 5. 想验证修复/改动对指标的影响

改代码后只需重跑 `bench` 一次，对比新报告与旧报告的数值（版本用 `--tag` 区分）：

```bash
python -m rag.cli bench --tag fix_lexical_gap
```

## 常见问题

- **扫描件 PDF 无文本**：确认已 `pip install rapidocr-onnxruntime`（纯 pip，无需系统程序）
- **HF 下载慢/网络抖动**：`$env:LEGAL_OFFLINE="1"`（模型已缓存时）；或 `HF_ENDPOINT=https://hf-mirror.com`
- **想换模型**：`$env:LEGAL_EMBEDDING=...` / `$env:LEGAL_RERANK_MODEL=...`
- **评估集**：`data/evaluation/qa.json`（100 条，字段含 question/answer/source/related_docs）

*文档最后更新：2026-08-21*
