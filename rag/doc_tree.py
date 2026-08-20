# -*- coding: utf-8 -*-
"""Document Tree 构建：由解析块（标题层级）构造章节树。

结构：文档 -> 章/一级标题 -> 条/二级标题 -> 正文段落
正文块挂到最近的标题节点下；无标题的正文归入"前言"节点。
"""
from __future__ import annotations

import logging
from typing import Optional

from .models import Document, TreeNode
from .parser import ParsedBlock

logger = logging.getLogger(__name__)

MAX_LEVEL = 3


def build_tree(doc: Document, blocks: list[ParsedBlock]) -> TreeNode:
    root = TreeNode(title=doc.title, level=0, text="", meta={"doc_id": doc.doc_id})
    stack: list[TreeNode] = [root]

    def current() -> TreeNode:
        return stack[-1]

    for b in blocks:
        if b.is_heading:
            lvl = min(max(b.level, 1), MAX_LEVEL)
            node = TreeNode(title=b.text, level=lvl, text="")
            # 找到合适父级：level 递增则挂在当前下，否则回退
            while len(stack) > 1 and current().level >= lvl:
                stack.pop()
            current().children.append(node)
            stack.append(node)
        else:
            node = current()
            node.text = (node.text + "\n" + b.text).strip()
    return root


def collapse_to_sections(tree: TreeNode) -> list[TreeNode]:
    """把树拍平为"章节级"节点列表（含子树文本），用于 Parent Chunk。"""
    sections: list[TreeNode] = []
    for node in tree.children:
        sections.append(node)
    return sections
