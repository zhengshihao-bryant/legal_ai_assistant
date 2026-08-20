# 法律知识助手 (Legal Knowledge Assistant) V1

基于 **RAG（检索增强生成）** 架构的法律知识问答系统。面向法律条文、案例、合同模板等专业文档，提供**语义检索**与 **AI 智能问答**能力，帮助用户快速定位法律依据并生成专业回答。

> ✅ **项目状态**：演示数据集层已完成（17 部官方法律 PDF + 企业制度/合同/案例 + 100 条 QA 评估集 + 官方法律下载工具）；应用层（FastAPI 后端 / Vue3 前端）开发中。本文档的架构描述为目标形态。

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
| 后端 FastAPI `backend/` | 🚧 开发中 | 目录已就位，待实现文档入库 / 检索 / 问答 |
| 前端 Vue3 `frontend/` | ⏳ 未开始 | 规划中 |

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

### V1（本期）
- [x] 技术选型与架构设计
- [x] 演示数据集构建（17 部法律 PDF + 制度/合同/案例 + 100 条评估集）
- [x] 官方法律下载工具（tools/download_laws.py）
- [ ] 后端基础框架（FastAPI + 配置 + 认证）
- [ ] 文档上传、解析、切片
- [ ] Milvus 向量化与检索
- [ ] LangChain + GPT 问答链路
- [ ] Vue3 前端（文档管理页 + 问答页）
- [ ] Docker Compose 一键部署

### V1.5
- [ ] 流式输出（SSE）
- [ ] 多轮对话上下文
- [ ] 引用溯源高亮

### V2（后期）
- [ ] 本地对象存储切换 **MinIO**
- [ ] 私有化部署大模型（如本地 Llama / 通义千问）
- [ ] 知识库版本管理与权限控制
- [ ] 文档批注与人工纠错反馈闭环

---

## 许可证

本项目为个人学习 / Demo 项目，暂未选择开源许可证。商用前请确认大模型与向量化模型的使用授权。

---

*文档最后更新：2026-08-08 · 法律知识助手 V1 规划*
