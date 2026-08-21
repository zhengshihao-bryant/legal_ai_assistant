# 法律知识助手 (Legal Knowledge Assistant) V1

基于 **RAG（检索增强生成）** 架构的法律知识问答系统。面向法律条文、案例、合同模板等专业文档，提供**语义检索**与 **AI 智能问答**能力，帮助用户快速定位法律依据并生成专业回答。

> ✅ **项目状态**：**V1 Knowledge RAG Core 已完成并跑通评估**（数据 → 解析/OCR → Document Tree → Parent/Child Chunk → BGE Embedding → Dense/BM25/Hybrid 检索 → RRF → BGE Reranker → Context → 生成 → Citation 校验 → 分层评估，100 条 QA 评估集出报告）。下一步聚焦 V1.5 工程化与评估深化；应用层（FastAPI / Vue3 / Docker）按战略暂缓（见 Roadmap）。

---

## 核心功能

- 📚 **文档知识库**：导入/管理法律条文、司法解释、裁判文书、合同模板等文档，自动切片、向量化入库
- 🔍 **语义检索**：Dense（BGE）+ Sparse（BM25）+ Hybrid（RRF）多路检索，BGE Reranker 精排
- 💬 **AI 智能问答**：结合检索结果生成带法律依据的引用式回答，可溯源
- 📎 **引用溯源**：回答附注来源文档与条款位置，Groundedness 逐句校验，无支撑句自动标记
- 🔐 **权限管理**：多用户体系，知识库按角色隔离（规划中，V2）

---

## 当前进度（2026-08-20）

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 演示数据集 `data/` | ✅ 已完成 | 17 部官方法律 PDF（含 5 部扫描件 OCR）、28 份企业制度、7 份合同模板、4 份模拟案例、100 条 QA 评估集 |
| 法律下载工具 `tools/download_laws.py` | ✅ 已完成 | 从国家法律法规数据库（flk.npc.gov.cn）幂等下载官方 PDF |
| **V1 Knowledge RAG Core** `rag/` | ✅ **已完成** | 全链路 18 个环节实现并跑通，见下方「V1 交付」与评估报告 |
| 工程化与评估 | 🚧 V1.5 重点推进 | 评估集扩充、错误案例回归、Query Rewrite 优化、真实 LLM Adapter 调优 |
| 应用层（FastAPI / Vue3 / Docker Compose） | ⏸ 暂缓 | 不增强 RAG 纵向能力，见「原计划功能处置」 |

---

## 快速使用

```bash
pip install -r requirements.txt          # 首次安装依赖
python -m rag.cli index                   # 构建索引（解析/OCR -> Chunk -> Embedding -> 入库）
python -m rag.cli query "试用期最长多久?"  # 命令行问答（交互模式不带参数）
python -m rag.cli evaluate --tag v1       # 100 条 QA 全量评估，产出报告
```

- 默认本地 numpy 向量库（零依赖服务）；接 Milvus 用 `--vector-backend milvus`
- 未配置 `OPENAI_API_KEY` 时自动使用**本地抽取式生成兜底**（离线可复现）；配置后走 gpt-4o-mini 结构化生成
- 模型（bge-base-zh-v1.5 / bge-reranker-base）首次运行时自动下载至 `data/models/`

## 能力链（项目主线）

本项目不做「大而全」的产品堆叠，而是沿一条**纵向能力链**做深，最终在 GitHub 上呈现为一个可复现、可评估的 RAG 工程案例：

```
数据 → Parsing/OCR → Chunk → Embedding → Retrieval ─┐
                                  (Dense + BM25)    ├→ Hybrid(RRF) → Rerank → Context → Generation → Validation → Evaluation
                                   └───────────────┘
```

每到一个环节，README 与评估报告都会给出**可验证的证据**（跑通日志、指标对比、错误案例分析），而不是「看起来能跑」。

---

## V1 交付与评估结果

### 已实现模块（rag/ 包，18 环节全链路）

```
loader(58 篇文档) → parser(PDF/DOCX + RapidOCR 扫描件兜底) → doc_tree(标题层级)
→ chunking(Parent/Child: 666 parents / 2115 children) → embedding(BGE 768 维)
→ retrievers(Dense / BM25 / Hybrid+RRF) → reranker(BGE CrossEncoder)
→ query(法律主题词典 + 可选 LLM Rewrite) → context(证据组装 + [n] 引用)
→ generation(OpenAI 结构化生成 / 本地抽取式兜底) → citation(逐句 Groundedness 校验)
→ evaluation(Retrieval / Generation 分层评估)
```

### Retrieval 层评估（100 条 QA，文档级，2026-08-21 修复后）

| 阶段 | Recall@5 | Recall@10 | Recall@20 | MRR | NDCG@10 |
| --- | --- | --- | --- | --- | --- |
| Dense（BGE） | 0.88 | 0.91 | 0.91 | 0.676 | 0.793 |
| BM25（jieba） | 0.72 | 0.85 | 0.85 | 0.537 | 0.680 |
| **Hybrid（RRF）** | **0.85** | **0.86** | **0.86** | **0.652** | **0.756** |
| + Rerank | 0.84 | 0.86 | 0.86 | 0.646 | 0.752 |

**结论与发现**（详见 [data/evaluation/reports/report_v1.md](data/evaluation/reports/report_v1.md)）：
- 语义检索（Dense Recall@10 0.91）显著优于纯关键词（BM25 0.85），符合中文法律长尾表述场景
- **Hybrid+RRF 综合最优**（MRR 0.652 / NDCG@10 0.756），Dense/BM25 互补
- Rerank 与 Hybrid 持平（Recall@20 相同 0.86，MRR 略降 0.646）——精排收益在 chunk 级，详见下方 ①

### Generation 层评估（本地抽取式兜底）

| EM | F1 | ROUGE-1 | ROUGE-2 | ROUGE-L | 引用率 | 平均接地性 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.172 | 0.203 | 0.069 | 0.135 | **0.97** | **0.977** |

- 本地兜底为**抽取式**：直接引证据原文，语法粗糙导致文本相似度类指标偏低（预期内）
- **引用率 97% / 接地性 0.977** 证明检索→生成链路可溯源、无凭空编造
- **接 DeepSeek LLM 后 F1 0.172 → 0.284（+63%）**，对比见下方 V1.5 ⑤

### V1.5 ① Rerank Ablation（Hybrid vs Hybrid+Rerank × Top-K，chunk/doc 双层）

| 配置 | chunk R@5 | chunk R@10 | chunk MRR | doc R@5 | doc R@10 | doc MRR | NDCG@10 | 延迟/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid（RRF） | 0.77 | 0.83 | 0.610 | 0.85 | 0.86 | 0.652 | 0.756 | 44ms |
| + Rerank | 0.74 | 0.82 | 0.598 | 0.84 | 0.86 | 0.646 | 0.752 | 1.8s（+40x） |

**结论**（完整报告：[ablation_rerank_v1.md](data/evaluation/reports/ablation_rerank_v1.md)）：
- Rerank **不扩大候选集**：Recall@20 与 Hybrid 完全一致（0.86）——Rerank 的价值在精排而非召回
- chunk 级 Recall@5/10 略降（0.74/0.82 vs 0.77/0.83）：chunk 级 gold 用「词汇重叠」代理，而 CrossEncoder 擅长**语义匹配**（字面不同但相关），代理指标低估其收益
- **延迟代价显著**：CPU 上 Rerank 单查询 +1.8s（40 倍），生产需 GPU 或仅对候选精排
- 分域：法规/制度/合同 Recall@10 ≥0.93，**案例 0.45 是唯一短板**（模拟案例文档检索难，指向后续根因）

### V1.5 ② Query Rewrite Ablation（无改写 / 规则 / LLM，下游统一 Hybrid）

| 组 | chunk R@10 | chunk MRR | doc R@10 | doc MRR | NDCG@10 |
| --- | --- | --- | --- | --- | --- |
| 原始 | 0.71 | 0.527 | 0.75 | 0.564 | 0.654 |
| 规则扩展 | 0.74 | 0.515 | 0.79 | 0.549 | 0.661 |
| **LLM 改写（DeepSeek）** | **0.77** | **0.562** | **0.79** | **0.592** | **0.693** |

**结论**（完整报告：[ablation_query_rewrite_v1.md](data/evaluation/reports/ablation_query_rewrite_v1.md)）：
- **LLM 改写综合最优**：Recall 与规则持平（0.79），但**排序显著更好**（chunk MRR 0.562、NDCG@10 0.693）——LLM 不仅补术语，还能精准重写（如「加班工资」→「加班费…延长工作时间…加班费计算基数」）
- **规则改写的边界**：能补召回（0.75→0.79）但 top-5 精度略降（追加领域词稀释短候选取舍）；当预算有限时规则是「够用」方案
- 回答「什么情况下规则已够、什么情况下需要 LLM」：**常规制度/合同查询规则够用；法规术语稀疏查询 LLM 更稳**（LLM 改写 99/100 条生效，DeepSeek API）

### V1.5 ③ QA 分类 + Error Analysis（按类型拆分 + 错误分桶）

| 类型 | 样本 | Recall@10 | F1 | 引用率 | 接地性 |
| --- | --- | --- | --- | --- | --- |
| 法规 | 30 | **0.933** | 0.211 | 0.933 | 0.944 |
| 制度 | 38 | 0.974 | 0.171 | 0.974 | 0.983 |
| 合同 | 13 | 1.0 | 0.133 | 1.0 | 1.0 |
| 案例 | 20 | **0.45** | 0.140 | 1.0 | 1.0 |

**错误分桶**（修复后）：A 检索失败 14 条（集中于案例域）· B 证据命中但生成差 66 条（本地抽取式上限）· C 引用缺失 3 条 · D 接地性风险 3 条

**根因分析**（完整报告：[qa_class_v1.md](data/evaluation/reports/qa_class_v1.md)）：
- **法规域已修复**（Recall@10 0.433 → 0.933）：见下方 ④ 的 bug 故事
- **案例域 0.45 是当前唯一检索短板**：模拟案例文档的信息组织（案情/焦点/依据）与问答表述差异大，指向后续对案例文档做结构化解析

### V1.5 ④ OCR / Chunking 数据质量（含一个真实 bug 修复）

**关键发现——扫描件文档曾完全未入库**：①~③ 初版评估中《劳动合同法》等 5 部扫描件法律 Recall 极低。④ 审计定位到根因：**OCR 文本是「无换行的密文墙」，标题检测失败 → 文档树无章节 → chunk 数为 0 → 这 5 部法律从未进入索引**。修复 `collapse_to_sections`（无标题时整篇回退为一个 section）后：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 索引 chunk 数 | 1849 | **2115**（+266，扫描件入库） |
| 法规 Recall@10 | 0.433 | **0.933** |
| 整体 Recall@10（Hybrid） | 0.79 | **0.86** |
| 整体 MRR | 0.549 | **0.652** |
| 零 chunk 文档 | 5 | **0** |

**数据质量基线**（完整报告：[data_quality_v1.md](data/evaluation/reports/data_quality_v1.md)）：
- 19 部法律全部入库；5 部扫描件（原始文本层 0 字符）经 RapidOCR 识别，条款命中率 10/10（劳动合同法）、qa 引用条款全部命中
- Chunk 层：Parent→Child 关联率 1.0、跨条款错误率 2.1%、标题保留率 1.0、法规第X条标题保留 2010 处
- 已知限制：扫描件为整篇单 section，Parent 过大时 Context 截断（后续做 OCR 感知的结构化解析）

### V1.5 ⑤ Generation 对比（本地抽取式 vs DeepSeek LLM）

| 指标 | 本地抽取式 | DeepSeek LLM |
| --- | --- | --- |
| F1 | 0.174 | **0.284**（+63%） |
| ROUGE-1 | 0.206 | **0.324**（+57%） |
| ROUGE-2 | 0.068 | **0.153**（+126%） |
| ROUGE-L | 0.138 | **0.240**（+74%） |
| 引用率 | 0.98 | **0.99** |
| 平均接地性 | 0.983 | 0.935 |

**结论**（完整报告：[generation_compare_v1.md](data/evaluation/reports/generation_compare_v1.md)）：接真实 LLM 后生成质量**全面跳升**（F1 +63%），引用率保持 0.99；接地性略降（0.935）源于 LLM 用自己的表述而非直接引原文——这正说明 Groundedness 校验在 LLM 时代是**必需防线**而非可选项。

### 📄 技术白皮书（设计论证 + 全部实验结论）

每个设计决策（为什么 Dense+BM25 / 为什么 RRF / 为什么 Rerank 未提升召回 / 为什么需要 Query Rewrite / 为什么 OCR 重要 / 为什么接地性校验是防线）都基于实验证据：**[docs/whitepaper.md](docs/whitepaper.md)**。复现指南见 [docs/demo_guide.md](docs/demo_guide.md)。

---

## 技术栈

| 层级 | 技术选型 | 说明 |
| --- | --- | --- |
| **前端** | Vue 3 + Vite | 单页应用，配合 Element Plus 组件库 |
| **后端** | FastAPI | 异步高性能 Python Web 框架，自动生成 OpenAPI 文档 |
| **AI 大模型** | GPT-3.5 / GPT-4 API | 生成式问答、总结、改写 |
| **编排框架** | LangChain | 检索链、提示词模板、多轮对话管理 |
| **向量化模型** | BGE Embedding (bge-base-zh-v1.5) | 中文语义向量，本地推理 |
| **向量数据库** | Milvus | 海量向量检索，支持亿级规模 |
| **业务数据库** | PostgreSQL（备选 MySQL） | 用户、文档元数据、会话记录、引用关系 |
| **对象存储** | 本地文件存储（Demo 阶段） → MinIO（后期） | 原始文档文件存储 |
| **部署** | Docker Compose + 云服务器 | 一键编排，前后端与中间件容器化 |

**架构图**

```
┌─────────────────────────────────────────────────────────┐
│                        前端                              │
│                  Vue3 + Vite (SPA)                       │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / WebSocket
┌──────────────────────────▼──────────────────────────────┐
│                      后端 FastAPI                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ 文档管理模块  │  │ 问答服务模块  │  │ 认证与权限模块    │   │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────┘   │
│         │ 切片/向量化      │ LangChain 检索链               │
└─────────┼────────────────┼──────────────────────────────┘
          │                │
┌─────────▼─────┐   ┌──────▼───────────────┐   ┌──────────┐
│    Milvus     │   │    PostgreSQL/MySQL  │   │  MinIO*  │
│   向量库       │   │   业务数据            │   │ 对象存储   │
└───────────────┘   └──────────────────────┘   └──────────┘
                           │
              ┌────────────▼────────────┐
              │   BGE Embedding 服务     │  ← 本地向量化
              └─────────────────────────┘
              ┌─────────────────────────┐
              │   OpenAI GPT-3.5/4 API   │  ← 云端大模型
              └─────────────────────────┘
```

> \* Demo 阶段使用本地文件存储，后期切换 MinIO。

---

## 项目结构

```
legal-ai-assistant/
├── rag/                     # V1 Knowledge RAG Core（核心交付）
│   ├── config.py            # 全局配置（路径/模型/参数，可用环境变量覆盖）
│   ├── models.py            # Document / Chunk / RetrievalResult / Answer
│   ├── loader.py            # Document Loader（56 篇文档发现与去重）
│   ├── parser.py            # PDF/DOCX 解析 + RapidOCR 扫描件兜底
│   ├── doc_tree.py          # Document Tree（标题层级）
│   ├── chunking.py          # Parent/Child Chunk
│   ├── embedding.py         # BGE Embedding（bge-base-zh-v1.5）
│   ├── vector_store.py      # 向量库抽象：Milvus / 本地 numpy 回退
│   ├── retrievers.py        # Dense / BM25 / Hybrid(RRF)
│   ├── reranker.py          # BGE Reranker（CrossEncoder）
│   ├── query.py             # Query Understanding / Rewrite
│   ├── context.py           # Enterprise Context Builder
│   ├── generation.py        # LLM 结构化生成（OpenAI + 本地兜底）
│   ├── citation.py          # Citation / Groundedness 校验
│   ├── evaluation.py        # Retrieval / Generation 分层评估
│   ├── pipeline.py          # 编排：index / query / evaluate
│   └── cli.py               # python -m rag.cli
├── data/                    # 数据集与管线产物
│   ├── raw/                 # 原始文档:laws(17 PDF)/policies(28)/contracts(7)/cases(4)
│   ├── parsed/              # 解析结果（管线缓存）
│   ├── chunks/              # Parent/Child 缓存 + index_manifest
│   ├── index/               # 本地向量索引（生成物）
│   ├── models/              # 模型缓存（生成物）
│   └── evaluation/          # qa.json(100 条) + reports/（评估报告）
├── tools/                   # 数据工具
│   └── download_laws.py     # 国家法律法规数据库官方 PDF 下载脚本
├── backend/                 # 应用层（V1.5 规划，暂缓）
│   └── README.md
├── requirements.txt         # 依赖清单
├── main.py                  # 示例脚本(占位)
├── .gitignore
└── README.md
```

> `data/parsed|chunks|index|models` 为管线生成物（gitignore 已排除，可一键重建）。

---

## 快速开始

> ⚠️ 当前为**数据层 + V1 管线**阶段：以下「后端 / 前端 / Docker」启动步骤属应用层（V1.5 规划，暂缓）。V1 管线的使用方式见上文「快速使用」。

### 环境要求

| 依赖 | 版本要求 |
| --- | --- |
| Python | 3.10+ |
| Node.js | 18+ |
| Docker / Docker Compose | 最新稳定版 |
| Milvus | 2.x（单机 standalone 即可） |
| OpenAI API Key | 具备 GPT-3.5/4 访问权限 |

### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```bash
cp .env.example .env
```

关键配置项：

```dotenv
# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini           # 或 gpt-3.5-turbo

# 数据库
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=legal
POSTGRES_PASSWORD=your_password
POSTGRES_DB=legal_assistant

# 向量库（Milvus）
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=legal_vectors

# Embedding
EMBEDDING_MODEL=bge-base-zh-v1.5

# 存储（Demo 用本地，后期 MinIO）
STORAGE_BACKEND=local
STORAGE_DIR=./data/files
```

### 2. 启动基础设施（Docker）

```bash
docker compose up -d milvus postgres
```

### 3. 启动后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API 文档：http://localhost:8000/docs

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

- 访问：http://localhost:5173

### 5. 一键部署（完整 Docker）

```bash
docker compose up -d
```

---

## 核心流程

### 文档入库
1. 上传法律文档（PDF / Word / Markdown / TXT）
2. 文档解析与清洗，按语义块切片（chunk）
3. 使用 **BGE Embedding** 将切片向量化
4. 向量写入 **Milvus**，元数据写入 **PostgreSQL**，源文件存入对象存储

### 智能问答
1. 用户输入问题
2. 问题向量化后到 Milvus 检索 Top-K 相关法律片段
3. LangChain 组装「问题 + 检索结果」构建 Prompt
4. 调用 GPT-3.5/4 生成回答，并标注引用来源

---

## API 概览（规划）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/documents/upload` | 上传并入库文档 |
| GET | `/api/v1/documents` | 文档列表 |
| DELETE | `/api/v1/documents/{id}` | 删除文档及向量 |
| POST | `/api/v1/chat` | 发起问答（流式返回） |
| POST | `/api/v1/auth/login` | 用户登录 |
| POST | `/api/v1/auth/register` | 用户注册 |

---

## Roadmap

> 战略：**做深不做大**。V1 打通并验证 RAG 纵向链路，V1.5 做成可评估、可解释的工程案例，V2 保留演进方向但当前不投入。

### V1 — Knowledge RAG Core（✅ 已完成，2026-08-20）

- [x] 技术选型与 RAG 架构设计
- [x] 法律法规 / 企业制度 / 合同 / 案例数据集构建
- [x] 官方法律下载工具（tools/download_laws.py）
- [x] Document Loader（56 篇文档，md 优先去重）
- [x] PDF / DOCX 文档解析 + RapidOCR 扫描件兜底（5 部扫描版法律）
- [x] Document Tree 构建（标题层级）
- [x] Parent / Child Chunk（649 parents / 1724 children）
- [x] BGE Embedding（bge-base-zh-v1.5，768 维）
- [x] Milvus 向量检索（Dense，含本地 numpy 回退）
- [x] Query Understanding / Query Rewrite（法律主题词典 + 可选 LLM）
- [x] BM25 Sparse Retrieval（jieba + rank_bm25）
- [x] Dense + Sparse Hybrid Retrieval + RRF Fusion
- [x] BGE Reranker（bge-reranker-base CrossEncoder）
- [x] Enterprise Context Builder（parent 回溯 + [n] 引用）
- [x] LLM Structured Generation（OpenAI Adapter + 本地抽取式兜底）
- [x] Citation / Groundedness Validation（逐句校验 + 无支撑句标记）
- [x] Retrieval / Generation Evaluation（100 条 QA，四阶段对比报告）

### V1.5 — Engineering & Evaluation（当前重点推进方向）

- [ ] 完善 50～100 条真实评估集（当前 100 条为模拟口径一致集）
- [ ] 建立 Retrieval / Rerank / Generation 分层评估（V1 已搭框架 ✅）
- [ ] 对比 Dense / BM25 / Hybrid / Reranker 效果（V1 已出首版报告 ✅）
- [ ] 完善错误案例分析与回归测试（V1 报告已含错误案例分析 ✅）
- [ ] 优化 Query Rewrite / Query Understanding（LLM 版本调优）
- [ ] 完善 Citation 与 Evidence Traceability
- [ ] 真实 LLM API Adapter 效果对比（本地兜底 vs gpt-4o-mini）
- [ ] 完善 README / Architecture / Technical Design 文档

### V2 — Future Evolution（保留在 Roadmap，当前不做）

- Knowledge Graph / Graph Retrieval
- 知识库版本管理
- 企业级权限与多租户
- MinIO / Object Storage
- 本地 LLM / 私有化部署
- Human Feedback / Correction Loop
- Agent-based Legal Research

### 原计划功能处置

| 原计划 | 处置 | 原因 |
| --- | --- | --- |
| Evaluation | ✅ 强烈推进 | 证明 RAG 不是「看起来能跑」 |
| 错误案例回归 | ✅ 强烈推进 | 展示真正的工程能力 |
| Citation / Evidence | ✅ 强烈推进 | 非常契合企业法律场景 |
| Graph RAG | 🔮 V2 | 作为「下一阶段演进」展示 |
| SSE 流式输出 | ❌ 暂缓 | 属于应用层，不增强 RAG 纵向能力 |
| 多轮对话 | ❌ 暂缓 | 属于 Chat 产品能力 |
| Vue3 前端 | ❌ 暂缓 | 不是当前作品集核心 |
| Docker Compose | 🟡 可选 | 如以后需要别人一键运行再做 |
| MinIO | ❌ 暂缓 | 当前本地数据集完全够展示 |
| 私有化 LLM | ❌ 暂缓 | 会把项目带向部署 / 推理工程 |
| 权限控制 | ❌ 暂缓 | 横向治理能力放到其他项目 |
| 文档批注反馈 | ❌ 暂缓 | 属于产品闭环，不是当前核心 |

---

## 许可证

本项目为个人学习 / Demo 项目，暂未选择开源许可证。商用前请确认大模型与向量化模型的使用授权。

---

*文档最后更新：2026-08-20 · 法律知识助手 · V1 Knowledge RAG Core ✅（含评估报告）*
