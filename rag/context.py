# -*- coding: utf-8 -*-
"""Enterprise Context Builder：把检索命中的 child 回溯到 parent 并组装上下文。

- 按 Rerank 分数排序，去重（同一 parent 只保留一次，合并其 child 命中）
- 拼接 parent 全文，控制总长度 MAX_CONTEXT_CHARS
- 输出带 [n] 引用标记的上下文，供生成与 Citation 校验共用
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import MAX_CONTEXT_CHARS
from .models import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ContextUnit:
    """一条上下文证据。"""

    parent: Chunk
    matched_children: list[Chunk] = field(default_factory=list)
    score: float = 0.0
    ref_index: int = 0            # [n] 引用编号


@dataclass
class BuiltContext:
    units: list[ContextUnit] = field(default_factory=list)
    text: str = ""
    refs: dict[int, str] = field(default_factory=dict)   # ref_index -> citation 串

    @property
    def citations(self) -> list[str]:
        return [self.refs[i] for i in sorted(self.refs)]


def build_context(results: list[RetrievalResult],
                  parent_by_id: dict[str, Chunk],
                  max_chars: int = MAX_CONTEXT_CHARS) -> BuiltContext:
    # 按 parent 分组，保留最高分
    by_parent: dict[str, ContextUnit] = {}
    for r in results:
        if r.chunk is None:
            continue
        parent = parent_by_id.get(r.chunk.parent_id or "")
        if parent is None:
            parent = r.chunk
        unit = by_parent.get(parent.chunk_id)
        if unit is None:
            unit = ContextUnit(parent=parent, score=r.score)
            by_parent[parent.chunk_id] = unit
        unit.matched_children.append(r.chunk)
        unit.score = max(unit.score, r.score)

    units = sorted(by_parent.values(), key=lambda u: -u.score)

    ctx = BuiltContext()
    budget = max_chars
    for unit in units:
        text = unit.parent.text or unit.parent.title
        if len(text) > budget and ctx.units:
            break
        if len(text) > budget:
            text = text[:budget]
        unit.ref_index = len(ctx.units) + 1
        ctx.refs[unit.ref_index] = unit.parent.citation
        ctx.units.append(unit)
        budget -= len(text)

    parts = []
    for i, unit in enumerate(ctx.units, start=1):
        parts.append(f"[{i}] {unit.parent.text or unit.parent.title}")
    ctx.text = "\n\n".join(parts)
    return ctx
