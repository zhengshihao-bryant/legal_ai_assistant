# 法律知识助手 (Legal Knowledge Assistant)

基于 **RAG（检索增强生成）** 架构的中文法律知识问答系统。面向法律条文、企业制度、合同模板、案例等专业文档，提供多路检索与带引用溯源的 AI 问答能力。

---

## 项目背景

企业法务与合规场景中，法律问题往往是「口语问法 ↔ 法条术语」的跨表述检索：用户问「加班工资怎么算」，条文写的是「延长工作时间」。关键词检索无法弥合这种 lexical gap，通用大模型又无法保证答案锚定现行条文。本项目以 RAG 为核心，解决**可检索、可溯源、可评估**的法律知识问答问题。

## 项目目标

- **功能目标**：构建一条完整、可复现的 RAG 纵向能力链——从原始文档到可评估的问答系统
- **范围边界**：聚焦检索与生成核心；Web 服务、前端、多租户等应用层能力不在本仓库范围（见 Roadmap）

## 核心功能

- 📚 **多源文档知识库**：19 部官方法律 PDF（含 5 部扫描件 OCR）+ 28 份企业制度 + 7 份合同模板 + 4 份模拟案例
- 🔍 **多路检索**：Dense（BGE 向量）+ Sparse（BM25）+ Hybrid（RRF 融合）+ BGE Reranker 精排
- 💬 **RAG 问答**：检索证据组装上下文，LLM（或本地兜底）生成引用式回答
- 📎 **引用溯源**：逐句 Groundedness 校验，无支撑陈述自动标记
- 🧪 **评估体系**：100 条 QA 分层评估（Retrieval/Generation），四阶段对比，一键复现

## 技术栈

| 层级 | 选型 | 说明 |
| --- | --- | --- |
| 语言 | Python 3.10+ | |
| 文档解析 | PyMuPDF / python-docx / **RapidOCR** | 扫描件 PDF 自动 OCR 兜底 |
| 切块 | 自研 Document Tree + Parent/Child | 666 parents / 2115 children |
| 向量化 | BGE（bge-base-zh-v1.5，768 维） | sentence-transformers，本地推理 |
| 稠密检索 | numpy 向量库（可切 **Milvus**） | 接口已抽象，`--vector-backend milvus` 切换 |
| 稀疏检索 | BM25（jieba + rank-bm25） | |
| 重排 | BGE Reranker（CrossEncoder） | |
| LLM | OpenAI 兼容接口（DeepSeek 验证） | 可选；未配置时本地抽取式兜底 |
| 评估 | 自建分层评估 + Ablation Harness | `rag/evaluation.py` + `rag/bench.py` |

## 系统架构

```
数据层      data/raw：19 部法律 PDF（5 部扫描件）＋ 28 制度 ＋ 7 合同 ＋ 4 案例 ＋ 100 条 QA
              │
解析与索引层   PDF/DOCX 解析 ＋ RapidOCR 兜底 → Document Tree（标题层级）
              → Parent/Child Chunk → BGE Embedding（768 维）
              │
检索层        Dense（numpy/Milvus）＋ BM25（jieba） → Hybrid（RRF） → Reranker 精排
              │
生成层        Query Rewrite（规则/LLM） → Context 组装（[n] 引用） → LLM/本地生成
              → Citation / Groundedness 逐句校验
              │
评估层        100 条 QA · 四阶段对比 · 分域指标 · bench 一键复现
```

代码组织：`rag/` 包内每个环节一个模块（`parser` → `doc_tree` → `chunking` → `embedding` → `retrievers` → `reranker` → `query` → `context` → `generation` → `citation` → `evaluation`），`pipeline.py` 编排，`cli.py` 命令行入口。

## 快速 Demo

```bash
pip install -r requirements.txt          # 首次安装依赖
python -m rag.cli index                   # 构建索引（解析/OCR → Chunk → Embedding → 入库）
python -m rag.cli query "试用期最长多久?"  # 问答（交互模式不带参数）
python -m rag.cli bench --tag v1          # 一键复现全部评估报告（约 15-22 分钟）
```

- 模型（bge-base-zh-v1.5 / bge-reranker-base）首次运行自动下载至 `data/models/`；网络不稳时 `LEGAL_OFFLINE=1` 跳过联网检查
- 未配置 LLM Key 时自动使用**本地抽取式生成兜底**（离线可复现）；配置后走 LLM 生成（OpenAI 兼容，DeepSeek 示例）：

  ```powershell
  $env:OPENAI_API_KEY="..."; $env:LEGAL_LLM_BASE_URL="https://api.deepseek.com"; $env:LEGAL_LLM_MODEL="deepseek-chat"
  ```

- 完整命令与复现步骤见 [docs/demo_guide.md](docs/demo_guide.md)

## 项目结构

```
legal-ai-assistant/
├── rag/                     # 核心代码（一个环节一个模块）
│   ├── parser.py            # PDF/DOCX 解析 + RapidOCR 扫描件兜底
│   ├── doc_tree.py          # Document Tree（标题层级，无标题整篇回退）
│   ├── chunking.py          # Parent/Child Chunk
│   ├── embedding.py         # BGE Embedding
│   ├── vector_store.py      # 向量库抽象：本地 numpy / Milvus
│   ├── retrievers.py        # Dense / BM25 / Hybrid(RRF)
│   ├── reranker.py          # BGE Reranker
│   ├── query.py             # Query Understanding / Rewrite
│   ├── context.py           # Context Builder（[n] 引用）
│   ├── generation.py        # LLM 结构化生成 / 本地抽取式兜底
│   ├── citation.py          # Citation / Groundedness 校验
│   ├── evaluation.py        # Retrieval / Generation 分层评估
│   ├── ablation.py          # 消融实验（Rerank / Query Rewrite）
│   ├── data_quality.py      # OCR / Chunking 数据质量审计
│   ├── bench.py             # 统一基准（一键复现全部报告）
│   ├── pipeline.py          # 编排：index / query / evaluate
│   └── cli.py               # python -m rag.cli
├── data/
│   ├── raw/                 # 语料：laws(19 PDF) / policies(28) / contracts(7) / cases(4)
│   ├── evaluation/          # qa.json(100 条) + reports/（评估报告）
│   └── (parsed|chunks|index|models 为管线生成物，gitignore 排除)
├── docs/
│   ├── whitepaper.md        # 设计决策论证（每个「为什么」都有实验证据）
│   └── demo_guide.md        # 复现指南
├── tools/download_laws.py   # 官方法律下载工具（幂等）
├── requirements.txt
├── LICENSE                  # MIT（含数据许可说明）
└── README.md
```

## 评估结果

100 条 QA，文档级检索（修复后基线）：

| 阶段 | Recall@5 | Recall@10 | Recall@20 | MRR | NDCG@10 |
| --- | --- | --- | --- | --- | --- |
| Dense（BGE） | 0.88 | 0.91 | 0.91 | 0.676 | 0.793 |
| BM25（jieba） | 0.72 | 0.85 | 0.85 | 0.537 | 0.680 |
| **Hybrid（RRF）** | **0.85** | **0.86** | **0.86** | **0.652** | **0.756** |
| + Rerank | 0.84 | 0.86 | 0.86 | 0.646 | 0.752 |

关键结论：

- **Hybrid+RRF 综合最优**；Dense 语义检索显著优于纯关键词（符合法律口语↔术语场景）
- **数据质量是最大杠杆**：审计发现 5 部扫描件法律因解析失败**从未入库**，修复后法规域 Recall@10 **0.433 → 0.933**
- **LLM 改写**把整体 Recall@10 提升至 **0.93**（案例域 0.45→0.65）；**LLM 生成** F1 0.172 → **0.299**（+73%），引用率 1.0
- 生成层接地性校验拦截无支撑陈述（本地 0.977 / LLM 0.936）

完整报告在 `data/evaluation/reports/`（Rerank/Query Rewrite/QA 分类/数据质量/生成对比），设计与取舍论证见 [docs/whitepaper.md](docs/whitepaper.md)。

## 设计取舍

每个决策都有消融实验支撑（详见白皮书）：

1. **为什么 Dense + BM25 双路 + RRF**：两路召回集互补（语义 vs 字面），分数空间不可直接比较，RRF 用排序位置融合，无参稳健
2. **Rerank 的取舍**：不扩大候选集（Recall@20 与 Hybrid 相同），价值在精排；CPU 延迟 +40 倍，生产需 GPU 或候选截断
3. **Query Rewrite 规则 vs LLM**：规则补召回但稀释 top-5 精度（MRR 反降）；LLM 改写综合最优——常规查询规则够用，术语稀疏/案例域用 LLM
4. **Groundedness 校验是防线**：LLM 生成质量更高但接地性下降（0.936 vs 本地 0.977），模型越「会说话」越需要逐句校验
5. **数据质量优先**：OCR 文档必须验证「解析→结构→入库」全链路（零 chunk 检查），否则整类文档静默丢失

## Roadmap

- ✅ **V1**（2026-08）：RAG 纵向能力链全链路实现 + 100 条 QA 评估
- ✅ **V1.5**（2026-08）：五项实验（Rerank / Query Rewrite / QA 分类 / 数据质量 / 生成对比）+ 真实 bug 修复闭环 + 白皮书
- 📌 **后续方向**：真实裁判文书类评估集扩充；案例域结构化解析（争议焦点/裁判观点拆 parent）；OCR 感知的条文级结构化
- ⏸ **暂缓（不在本仓库范围）**：Web 服务 / 前端 / SSE / 多轮对话等应用层；Graph RAG / 知识库版本管理 / 权限多租户等 V2 能力

## 许可证

代码基于 **MIT** 许可（[LICENSE](LICENSE)）。数据分两类：法律法规原文为官方公版文本；制度/合同/案例为演示用模拟内容，**不构成法律意见，不得用于真实业务场景**。模型权重遵守其各自开源许可。

---

*最后更新：2026-08-21*
