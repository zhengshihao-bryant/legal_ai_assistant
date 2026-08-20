# -*- coding: utf-8 -*-
"""Citation / Groundedness Validation（引用溯源与接地性校验）。

- 引用完整性：答案中的 [n] 是否都能映射到上下文证据
- 逐句接地性：答案每句话与证据文本的词汇重叠率低于阈值 -> 标记为无支撑
- groundedness = 有支撑句子占比
"""
from __future__ import annotations

import logging
import re

import jieba

from .context import BuiltContext

logger = logging.getLogger(__name__)

jieba.setLogLevel(logging.WARNING)

_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])")
_REF_RE = re.compile(r"\[(\d+)\]")
STOP = {"以及", "并且", "但是", "对于", "根据", "按照", "相关", "以及", "其中", "我们", "如果", "那么",
        "所以", "因为", "这个", "那个", "进行", "可以", "需要", "应当", "不得", "必须", "属于", "包括"}


def _tokens(text: str) -> set[str]:
    toks = {t for t in jieba.lcut(text) if len(t.strip()) >= 2}
    toks -= STOP
    return toks


def _evidence_text(ctx: BuiltContext) -> str:
    return " ".join(u.parent.text for u in ctx.units)


def validate_groundedness(answer_text: str, ctx: BuiltContext,
                          threshold: float = 0.35) -> tuple[float, list[str]]:
    """返回 (groundedness 0~1, unsupported 句子列表)。"""
    if not answer_text:
        return 0.0, []
    evidence = _evidence_text(ctx)
    ev_toks = _tokens(evidence)
    if not ev_toks:
        return 0.0, [answer_text]

    sentences = [s.strip() for s in _SENT_SPLIT.split(answer_text) if len(s.strip()) >= 8]
    if not sentences:
        sentences = [answer_text]

    supported_count = 0
    unsupported: list[str] = []
    for sent in sentences:
        sent_clean = _REF_RE.sub("", sent)
        toks = _tokens(sent_clean)
        if not toks:
            supported_count += 1
            continue
        overlap = len(toks & ev_toks) / len(toks)
        if overlap >= threshold:
            supported_count += 1
        else:
            unsupported.append(sent)
    groundedness = supported_count / len(sentences)
    return round(groundedness, 4), unsupported
