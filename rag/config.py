# -*- coding: utf-8 -*-
"""全局配置：路径、模型、检索/生成参数。

所有参数可用环境变量覆盖（前缀 LEGAL_），便于评估实验切换。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LEGAL_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "evaluation"
REPORT_DIR = EVAL_DIR / "reports"
MODELS_DIR = DATA_DIR / "models"

# 模型缓存重定向到项目内（沙箱/无 home 写权限环境友好）
# 若 huggingface.co 直连慢，可自行 export HF_ENDPOINT=https://hf-mirror.com
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hf_cache" / "hub"))

for _d in (PARSED_DIR, CHUNKS_DIR, INDEX_DIR, REPORT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 文档加载
# 每个文件 stem 只加载一次；同 stem 存在 .md 时优先（更干净），否则 .docx
DOC_EXT_PREFERENCE = (".md", ".docx", ".txt", ".pdf")

# ---------------------------------------------------------------- 解析 / OCR
OCR_FALLBACK = os.environ.get("LEGAL_OCR", "1") == "1"  # PDF 无文本时尝试 OCR
MIN_PDF_CHARS = int(os.environ.get("LEGAL_MIN_PDF_CHARS", "20"))  # 低于该字符数视为扫描件

# ---------------------------------------------------------------- Chunk
CHILD_CHUNK_SIZE = int(os.environ.get("LEGAL_CHILD_SIZE", "220"))    # 子块字符数
CHILD_CHUNK_OVERLAP = int(os.environ.get("LEGAL_CHILD_OVERLAP", "40"))
MIN_CHILD_LEN = int(os.environ.get("LEGAL_MIN_CHILD", "40"))         # 过短子块并入上下文

# ---------------------------------------------------------------- 模型
EMBEDDING_MODEL = os.environ.get("LEGAL_EMBEDDING", "BAAI/bge-base-zh-v1.5")
RERANK_MODEL = os.environ.get("LEGAL_RERANK_MODEL", "BAAI/bge-reranker-base")
EMBEDDING_DIM = int(os.environ.get("LEGAL_EMBEDDING_DIM", "768"))
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DEVICE = os.environ.get("LEGAL_DEVICE", "cpu")

# ---------------------------------------------------------------- 检索
TOP_K_DENSE = int(os.environ.get("LEGAL_TOP_K_DENSE", "30"))
TOP_K_BM25 = int(os.environ.get("LEGAL_TOP_K_BM25", "30"))
TOP_K_HYBRID = int(os.environ.get("LEGAL_TOP_K_HYBRID", "20"))   # RRF 合并后截断
RRF_K = int(os.environ.get("LEGAL_RRF_K", "60"))
RERANK_TOP_K = int(os.environ.get("LEGAL_RERANK_TOP_K", "12"))   # Reranker 打分后保留
USE_RERANKER = os.environ.get("LEGAL_USE_RERANKER", "1") == "1"

# ---------------------------------------------------------------- Context
MAX_CONTEXT_CHARS = int(os.environ.get("LEGAL_MAX_CONTEXT", "4000"))

# ---------------------------------------------------------------- LLM
OPENAI_MODEL = os.environ.get("LEGAL_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.environ.get("LEGAL_OPENAI_TEMPERATURE", "0.2"))
# OpenAI 兼容接口适配（DeepSeek 等）：
#   LEGAL_LLM_BASE_URL=https://api.deepseek.com  LEGAL_LLM_MODEL=deepseek-chat
LLM_BASE_URL = os.environ.get("LEGAL_LLM_BASE_URL") or None
LLM_MODEL = os.environ.get("LEGAL_LLM_MODEL", OPENAI_MODEL)
USE_LOCAL_GENERATOR_FALLBACK = True   # 无 API Key 时用本地抽取式生成兜底

# ---------------------------------------------------------------- 评估
EVAL_TOP_KS = (5, 10, 20)             # Recall@k 报告档位
EVAL_RERANK_TOP_K = int(os.environ.get("LEGAL_EVAL_RERANK_TOP_K", "10"))

# ---------------------------------------------------------------- 随机
RANDOM_SEED = 42
