# -*- coding: utf-8 -*-
"""Legal Knowledge Assistant - V1 Knowledge RAG Core.

数据层 -> 解析/OCR -> Document Tree -> Parent/Child Chunk -> Embedding
-> Dense/BM25/Hybrid 检索 -> RRF -> Reranker -> Context -> 生成 -> Citation 校验 -> 评估
"""

__version__ = "1.0.0"
