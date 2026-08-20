# 法律知识助手 (Legal Knowledge Assistant) V1

基于 **RAG（检索增强生成）** 架构的法律知识问答系统。面向法律条文、案例、合同模板等专业文档，提供**语义检索**与 **AI 智能问答**能力，帮助用户快速定位法律依据并生成专业回答。

> ✅ **项目状态**：演示数据集层已完成。当前聚焦把数据资产做成一条**可验证、可解释、可评估**的 RAG 纵向管线（Roadmap V1），应用层（FastAPI / Vue3 / Docker）按战略暂缓（见「原计划功能处置」）。本文档的架构描述为目标形态。

---

## 核心功能

- 📚 **文档知识库**：导入/管理法律条文、司法解释、裁判文书、合同模板等文档，自动切片、向量化入库
- 🔍 **语义检索**：基于向量相似度检索，摆脱关键词匹配的局限，支持自然语言查询
- 💬 **AI 智能问答**：结合检索结果与大模型生成带法律依据的引用式回答，可溯源
- 📎 **引用溯源**：回答附注来源文档与条款位置，支持人工核验
- 🔐 **权限管理**：多用户体系，知识库按角色隔离（规划中）

---

## 当前进度（2026-08-08）

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 演示数据集 `data/` | ✅ 已完成 | 17 部官方法律 PDF、28 份企业制度、7 份合同模板、4 份模拟案例、100 条 QA 评估集（详见 [data/README.md](data/README.md)） |
| 法律下载工具 `tools/download_laws.py` | ✅ 已完成 | 从国家法律法规数据库（flk.npc.gov.cn）幂等下载官方 PDF，可断点重跑 |
| RAG 管线（解析 → 检索 → 生成 → 评估） | 🚧 V1 开发中 | 能力链各环节待实现，见下方 Roadmap V1 |
| 工程化与评估 | ⏳ V1.5 重点推进 | 分层评估、错误案例回归、Citation/Evidence |
| 应用层（FastAPI / Vue3 / Docker Compose） | ⏸ 暂缓 | 不增强 RAG 纵向能力，见「原计划功能处置」 |

---

## 能力链（项目主线）

本项目不做「大而全」的产品堆叠，而是沿一条**纵向能力链**做深，最终在 GitHub 上呈现为一个可复现、可评估的 RAG 工程案例：

```
数据 → Parsing/OCR → Chunk → Embedding → Retrieval ─┐
                                  (Dense + BM25)    ├→ Hybrid(RRF) → Rerank → Context → Generation → Validation → Evaluation
                                   └───────────────┘
```

每到一个环节，README 与评估报告都会给出**可验证的证据**（跑通日志、指标对比、错误案例分析），而不是「看起来能跑」。

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
├── data/                    # 演示数据集(RAG 数据管线)
│   ├── raw/                 # 原始文档:laws(17 部官方 PDF)/policies(28)/contracts(7)/cases(4)
│   ├── parsed/              # 解析结果(管线输出,当前为空)
│   ├── chunks/              # 切块结果(管线输出,当前为空)
│   └── evaluation/qa.json   # 问答评估集(100 条)
├── tools/                   # 数据工具
│   └── download_laws.py     # 国家法律法规数据库官方 PDF 下载脚本
├── backend/                 # FastAPI 后端(占位,开发中)
│   └── README.md
├── main.py                  # 示例脚本(占位)
├── .gitignore
└── README.md
```

> 应用层（backend/frontend/docker-compose）规划结构见下方「架构」与 Roadmap，开发中。

---

## 快速开始

> ⚠️ 当前仓库处于**数据层阶段**，以下「后端 / 前端 / Docker」启动步骤为目标形态，待 V1 管线落地（见 Roadmap）后生效。

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

### V1 — Core RAG Pipeline（纵向核心链路）

- [x] 技术选型与 RAG 架构设计
- [x] 多源法律文档数据集构建（17 部法律 + 28 制度 + 7 合同 + 4 案例 + 100 条评估集）
- [x] 官方法律下载工具（tools/download_laws.py）
- [ ] PDF / DOCX 文档解析与 OCR
- [ ] 文档清洗、结构化与层级 Chunk
- [ ] BGE Embedding（bge-base-zh-v1.5）
- [ ] Milvus 向量检索（Dense）
- [ ] BM25 Sparse Retrieval
- [ ] Hybrid Retrieval + RRF Fusion
- [ ] BGE Reranker
- [ ] Context Builder
- [ ] LLM Generation
- [ ] Citation Validation / Groundedness
- [ ] RAG Evaluation

### V1.5 — Engineering & Evaluation（当前重点推进方向）

- [ ] 完善 50～100 条真实评估集
- [ ] 建立 Retrieval / Rerank / Generation 分层评估
- [ ] 对比 Dense / BM25 / Hybrid / Reranker 效果
- [ ] 完善错误案例分析与回归测试
- [ ] 优化 Query Rewrite / Query Understanding
- [ ] 完善 Citation 与 Evidence Traceability
- [ ] 增加真实 LLM API Adapter
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

*文档最后更新：2026-08-08 · 法律知识助手 · 数据层 ✅ / V1 RAG 管线开发中*
