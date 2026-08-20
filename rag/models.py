# -*- coding: utf-8 -*-
"""核心数据类。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TreeNode:
    """Document Tree 节点：标题层级结构。"""

    title: str
    level: int = 0                # 0=文档, 1=章/一级标题, 2=条/二级标题 ...
    text: str = ""
    children: list["TreeNode"] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def subtree_text(self) -> str:
        """自身文本 + 所有后代文本（用于 Parent Chunk）。"""
        parts = [self.text]
        for c in self.children:
            t = c.subtree_text()
            if t:
                parts.append(t)
        return "\n".join(p for p in parts if p)


@dataclass
class Document:
    """一篇已解析文档。"""

    doc_id: str                   # 稳定 id：相对 raw 的路径
    path: str                     # 源文件绝对/相对路径
    kind: str                     # laws / policies / contracts / cases
    title: str
    tree: Optional[TreeNode] = None
    plain_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Parent/Child Chunk。child 用于检索，parent 用于喂给生成。"""

    chunk_id: str
    doc_id: str
    text: str
    role: str = "child"           # child | parent
    parent_id: Optional[str] = None
    title: str = ""               # 所在章节标题（用于引用）
    source: str = ""              # 文件路径
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """引用串：来源文件名 + 章节标题。"""
        name = self.source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if self.title and self.title != name:
            return f"{name} · {self.title}"
        return name


@dataclass
class RetrievalResult:
    """单条检索结果。"""

    chunk_id: str
    score: float
    chunk: Optional[Chunk] = None
    stage: str = "final"          # dense | bm25 | hybrid | rerank


@dataclass
class Answer:
    """生成结果（结构化输出）。"""

    answer: str
    citations: list[str] = field(default_factory=list)     # 引用串列表
    groundedness: float = 0.0     # 0~1 接地性/可溯源得分
    unsupported: list[str] = field(default_factory=list)   # 找不到证据支撑的句子
    meta: dict[str, Any] = field(default_factory=dict)
