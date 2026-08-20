# -*- coding: utf-8 -*-
"""Document Loader：扫描 data/raw，按扩展名加载文档（md 优先于 docx 去重）。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from .config import RAW_DIR, DOC_EXT_PREFERENCE
from .models import Document

logger = logging.getLogger(__name__)

# 顶层分类目录 -> kind
KIND_BY_DIR = {
    "laws": "laws",
    "policies": "policies",
    "contracts": "contracts",
    "cases": "cases",
}


def _kind_of(path: Path) -> str:
    # 相对 raw 的第一层目录决定 kind
    try:
        rel = path.relative_to(RAW_DIR)
    except ValueError:
        return "misc"
    parts = rel.parts
    if parts and parts[0] in KIND_BY_DIR:
        return KIND_BY_DIR[parts[0]]
    return "misc"


def discover_files(root: Path | None = None) -> list[Path]:
    """发现并去重文档文件：同目录下同 stem 多扩展名只取优先级最高的一个。"""
    root = root or RAW_DIR
    files: dict[tuple[str, str], Path] = {}   # (dir, stem) -> path
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in DOC_EXT_PREFERENCE:
            continue
        # 跳过 pipeline 输出目录
        if any(seg.startswith(".") for seg in p.relative_to(RAW_DIR).parts):
            continue
        key = (str(p.parent), p.stem)
        cur = files.get(key)
        if cur is None or DOC_EXT_PREFERENCE.index(p.suffix.lower()) < DOC_EXT_PREFERENCE.index(cur.suffix.lower()):
            files[key] = p
    return sorted(files.values())


def doc_id_for(path: Path) -> str:
    rel = path.relative_to(RAW_DIR).as_posix()
    return rel


def load_plain(path: Path) -> str:
    """按扩展名读取纯文本（不做结构化解析，结构化在 parser 层）。"""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    # pdf/docx 在 parser 层处理
    return ""


def scan_documents(root: Path | None = None) -> list[Document]:
    """扫描并构造 Document（仅元数据 + md/txt 文本；pdf/docx 由 parser 填充）。"""
    docs: list[Document] = []
    for p in discover_files(root):
        kind = _kind_of(p)
        doc = Document(
            doc_id=doc_id_for(p),
            path=p.as_posix(),
            kind=kind,
            title=p.stem,
            plain_text=load_plain(p),
            meta={"ext": p.suffix.lower(), "size": p.stat().st_size},
        )
        docs.append(doc)
    logger.info("扫描到 %d 篇文档（raw 目录）", len(docs))
    return docs
