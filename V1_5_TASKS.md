# V1.5 工程化与评估 — 任务清单与进度追踪

> 战略：**做深不做大**。V1 已打通纵向链路；V1.5 把它做成可验证、可解释、可评估的 RAG 工程案例。
> 本文档记录每个任务的**状态 / 产物 / 下一步 / 阻塞**，作为开发与面试准备的进度基准。

---

## ✅ 已完成

### V1 Knowledge RAG Core（commit `f600a80`）
18 环节全链路实现 + 100 条 QA 评估，详见根 [README.md](README.md)。

### 数据修正（commit `ff68487` 内）
- 补下载《专利法》《著作权法》→ 语料库 **19 部法律**（原 17 部，R024/R026 曾无法作答）
- 主评估刷新（commit `c03673b`）：Hybrid Recall@10 **0.77 → 0.79**，MRR 0.549

### ① Rerank Ablation（commit `ff68487`）✅
- 产物：`data/evaluation/reports/ablation_rerank_v1.{json,md}`
- 框架：Ablation Harness（`rag/ablation.py` + `python -m rag.cli ablate --exp rerank`）
- 核心发现：
  - Rerank 改善 doc 级排序（MRR .570 vs .564、NDCG@10 .661 vs .654），**不扩大候选集**（Recall@20 相同 .76）
  - chunk 级 R@5/10 略降（词汇重叠代理低估语义匹配收益）
  - 延迟 +5x（442ms → 2.4s/query，CPU）
  - 分域短板：法规 Recall@10 = **0.50**（vs 合同 1.0 / 制度 .974）

### ② Query Rewrite Ablation — 离线两组（commit `c03673b`）✅
- 产物：`data/evaluation/reports/ablation_query_rewrite_v1.{json,md}`
- 框架：`python -m rag.cli ablate --exp query-rewrite`
- 核心发现：
  - **规则改写 Recall@10 0.75 → 0.79**，精准补强短板域：法规 +3.3pp、案例 +10pp、制度 →1.0
  - 代价：chunk R@5 0.66 → 0.61（追加领域词对短候选取舍有稀释）
  - LLM 组已接入 DeepSeek（OpenAI 兼容适配），**真实 LLM 对比待跑**（见下）

### ③ QA 分类 + Error Analysis（commit `6ef9c05`）✅
- 产物：`data/evaluation/reports/qa_class_v1.{json,md}`
- 框架：`rag/qa_analysis.py` + `python -m rag.cli qa-class`
- 核心发现（根因）：
  1. 法规 Recall@10 = 0.433 / 案例 = 0.45（短板域）
  2. **OCR 扫描件**（劳动合同法/招标投标法等）弱化向量表达 → 制度 FAQ 抢占检索结果
  3. **lexical gap 实证**：口语「加班」vs 法条「延长工作时间」——② 词典需补充映射
  4. 本地兜底生成器词重叠为 0 时直接放弃（3 条「找不到」）→ 应降级返回最相关证据句

---

## 🔄 进行中 / 未完成

### ②-LLM：Query Rewrite 的 LLM 变体（DeepSeek）
- 状态：代码就绪（`LEGAL_LLM_BASE_URL / LEGAL_LLM_MODEL` 已支持），**未运行**
- 阻塞：需要用户的 DeepSeek API Key（Key 只在用户终端环境变量里，不透传给我）
- 下一步（用户在自己的终端执行）：
  ```powershell
  cd C:\Users\25806\Desktop\legal-ai-assistant
  $env:OPENAI_API_KEY = "你的DeepSeekKey"
  $env:LEGAL_LLM_BASE_URL = "https://api.deepseek.com"
  $env:LEGAL_LLM_MODEL = "deepseek-chat"
  python -m rag.cli ablate --exp query-rewrite --tag v1
  ```
- 产出：覆盖 `ablation_query_rewrite_v1.md`，得到真实三组对比（回答「什么情况下规则已够、什么情况下需要 LLM」）

### ④ OCR / Chunking 数据质量审计
- 状态：**代码已写完（`rag/data_quality.py` + `python -m rag.cli data-quality`），尚未运行、未出报告、未提交**
- 待运行（模型已缓存，`LEGAL_OFFLINE=1` 可跳过联网）：
  ```powershell
  $env:LEGAL_OFFLINE = "1"
  python -m rag.cli data-quality --tag v1
  ```
- 产出：`data/evaluation/reports/data_quality_v1.{json,md}`
- 指标设计：
  - OCR：页面完整率 / qa.json 引用条款命中率（扫描件代理）/ 关键术语可达率
  - Chunk：Parent→Child 关联率 / 跨条款错误率 / 标题保留率 / 法规第X条标题行保留
  - 《劳动合同法》人工抽样基准（OCR 行 + chunk 抽样）
- 目的：回答「RAG 的问题是检索算法，还是数据根本没进来」

### ⑤ Technical Whitepaper + Demo Guide
- 状态：**未开始**
- 产出建议：
  - `docs/whitepaper.md`：以「为什么」为主线（为什么 Dense+BM25 / 为什么 RRF / 为什么 Rerank 未提升最终 Recall / 为什么需要 Query Rewrite / 为什么 OCR 重要）——把 ①②③④ 的实验结论写成设计论证，而非说明书
  - `docs/demo_guide.md`：一键复现指南（index → query → evaluate → ablate → qa-class → data-quality）

---

## 📌 未提交 / 未推送事项

| 事项 | 状态 |
| --- | --- |
| `rag/data_quality.py` + `rag/cli.py`（④ 代码） | 未提交（本文档提交时一并处理） |
| 本地 → GitHub 推送 | 本地领先 origin **5 个提交**（V1 实现、① ② ③、README 更新），用户终端执行 `git push origin main` |

---

## 📅 建议执行顺序（暂停点续跑）

1. （用户）跑 ②-LLM DeepSeek 变体 → 2 分钟 + 100 次 API 调用
2. （我）跑 ④ 数据质量审计 → 出报告、提交
3. （我）根据 ④ 结论 + ③ 根因，修复两个已知问题：
   - ② 词典补充 lexical gap 映射（加班→延长工作时间 等）
   - 本地兜底生成器「词重叠为 0 时返回最相关证据句」
4. （我）重跑 ②③ 验证修复效果（指标 → 根因 → 修改 → 再验证闭环）
5. （我）⑤ 白皮书 + Demo Guide
6. （用户）`git push origin main` 收尾

*文档最后更新：2026-08-21*
