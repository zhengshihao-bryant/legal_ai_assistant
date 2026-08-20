# -*- coding: utf-8 -*-
"""Parent / Child Chunk。

- Parent：章节级（Document Tree 节点），供生成阶段拼 Context，携带完整引用
- Child：段落级小切片（可重叠），供向量/BM25 检索，命中后回溯 Parent
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable

from .config import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, MIN_CHILD_LEN
from .models import Chunk, Document, TreeNode

logger = logging.getLogger(__name__)

# 软换行（法律条文里的换行），chunk 时按标点断句优先
_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _stable_hash(*parts: str) -> str:
    h = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return h


def split_text_into_children(text: str, size: int = CHILD_CHUNK_SIZE,
                             overlap: int = CHILD_CHUNK_OVERLAP) -> list[str]:
    """按句优先、定长回退的滑动窗口切片。"""
    text = _normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > size:  # 超长句按硬窗口
            for i in range(0, len(sent), size - overlap):
                pieces = sent[i:i + size]
                if len(pieces) >= MIN_CHILD_LEN:
                    chunks.append(pieces)
            buf = ""
            continue
        if len(buf) + len(sent) <= size:
            buf = (buf + sent).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)

    # 末尾重叠：与前块重叠 overlap 字符，缓解边界信息丢失
    merged: list[str] = []
    for c in chunks:
        if merged and overlap > 0 and len(c) > overlap:
            prev_tail = merged[-1][-overlap:]
            if not c.startswith(prev_tail):
                c = prev_tail + c
        merged.append(c)
    return merged


def build_chunks(doc: Document, sections: Iterable[TreeNode]) -> tuple[list[Chunk], list[Chunk]]:
    """由章节树生成 parent/child chunk 对。

    返回 (children, parents)。每个 parent 至少保留 1 个 child（即使为空章节）。
    """
    children: list[Chunk] = []
    parents: list[Chunk] = []

    for sec in sections:
        sec_text = _normalize(sec.subtree_text())
        sec_title = sec.title if sec.level >= 1 else doc.title
        parent_id = _stable_hash(doc.doc_id, sec_title)
        parent = Chunk(
            chunk_id=parent_id,
            doc_id=doc.doc_id,
            text=sec_text,
            role="parent",
            title=sec_title,
            source=doc.path,
            meta={"level": sec.level, "doc_kind": doc.kind},
        )
        parents.append(parent)

        if not sec_text or len(sec_text) < MIN_CHILD_LEN:
            # 空章节：仍建一个 child 指向 parent，保证可检索到该章节
            children.append(Chunk(
                chunk_id=_stable_hash(parent_id, "only"),
                doc_id=doc.doc_id,
                text=sec_text or sec_title,
                role="child",
                parent_id=parent_id,
                title=sec_title,
                source=doc.path,
                meta={"doc_kind": doc.kind},
            ))
            continue

        for i, piece in enumerate(split_text_into_children(sec_text)):
            child = Chunk(
                chunk_id=_stable_hash(parent_id, str(i)),
                doc_id=doc.doc_id,
                text=piece,
                role="child",
                parent_id=parent_id,
                title=sec_title,
                source=doc.path,
                meta={"doc_kind": doc.kind, "seq": i},
            )
            children.append(child)

    logger.debug("文档 %s: %d parents, %d children", doc.doc_id, len(parents), len(children))
    return children, parents
