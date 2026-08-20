# -*- coding: utf-8 -*-
"""向量库抽象：Milvus（生产）与本地 numpy 回退（Demo/离线评估）。

接口统一为：
- add(ids, vectors, payloads)
- search(vector, top_k) -> list[(id, score)]
- __len__

本地回退将向量持久化到 data/index/（.npy + .json），可重复加载。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import INDEX_DIR

logger = logging.getLogger(__name__)


class VectorStore:
    def add(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:  # pragma: no cover
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError


class LocalVectorStore(VectorStore):
    """numpy 余弦检索 + 本地持久化。"""

    def __init__(self, index_name: str = "legal", index_dir: Path | None = None):
        self.index_dir = Path(index_dir or INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.index_dir / f"{index_name}_vectors.npy"
        self.meta_path = self.index_dir / f"{index_name}_meta.json"
        self._vecs: np.ndarray | None = None
        self._ids: list[str] = []
        self._payloads: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------ 持久化
    def _load(self) -> None:
        if self.vec_path.exists() and self.meta_path.exists():
            try:
                self._vecs = np.load(self.vec_path)
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self._ids = meta["ids"]
                self._payloads = {p["chunk_id"]: p for p in meta["payloads"]}
                logger.info("本地向量库加载: %d 条", len(self._ids))
            except Exception as e:  # noqa: BLE001
                logger.warning("本地向量库加载失败（重建）: %s", e)
                self._vecs, self._ids, self._payloads = None, [], {}

    def _save(self) -> None:
        np.save(self.vec_path, self._vecs)
        meta = {"ids": self._ids, "payloads": [self._payloads[i] for i in self._ids]}
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ 接口
    def add(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms == 0, 1, norms)
        self._vecs = arr if self._vecs is None else np.vstack([self._vecs, arr])
        self._ids.extend(ids)
        for cid, p in zip(ids, payloads):
            self._payloads[cid] = p
        self._save()

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        if self._vecs is None or len(self._vecs) == 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._vecs @ q          # 归一化后 = 余弦
        top = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in top]

    def payload(self, cid: str) -> Optional[dict]:
        return self._payloads.get(cid)

    def __len__(self) -> int:
        return len(self._ids)


class MilvusVectorStore(VectorStore):
    """Milvus 2.x（pymilvus）。需先启动 Milvus 服务（docker compose up milvus）。"""

    def __init__(self, collection: str = "legal_vectors", uri: str = "http://localhost:19530",
                 dim: int = 768):
        from pymilvus import MilvusClient  # 延迟导入
        self.client = MilvusClient(uri=uri)
        self.collection = collection
        if not self.client.has_collection(collection):
            self.client.create_collection(collection_name=collection, dimension=dim, metric_type="COSINE")
        self.client.load_collection(collection)

    def add(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        rows = [
            {"id": cid, "vector": vec, **payload}
            for cid, vec, payload in zip(ids, vectors, payloads)
        ]
        for i in range(0, len(rows), 500):
            self.client.insert(self.collection, rows[i:i + 500])

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        res = self.client.search(
            collection_name=self.collection, data=[vector], limit=top_k, output_fields=["chunk_id"],
        )
        hits = res[0] if res else []
        return [(h["entity"].get("chunk_id", h["id"]), float(h["distance"])) for h in hits]

    def __len__(self) -> int:
        return self.client.get_collection_stats(self.collection).get("row_count", 0)


def build_vector_store(backend: str = "local", **kwargs) -> VectorStore:
    if backend == "milvus":
        logger.info("向量库后端: Milvus")
        return MilvusVectorStore(**kwargs)
    logger.info("向量库后端: 本地 numpy（Milvus 可通过 --vector-backend milvus 切换）")
    return LocalVectorStore(**kwargs)
