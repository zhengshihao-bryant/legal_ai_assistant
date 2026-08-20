# -*- coding: utf-8 -*-
"""PDF / DOCX 文档解析 + OCR 兜底。

输出：明文文本 + 轻量标题层级线索（供 Document Tree 使用）。
标题识别策略：
- MD：Markdown 标题（# 层级）天然结构化
- DOCX：Word 标题样式 / 加粗段落
- PDF：正则（第X章/第X条/第X节） + 字体大小启发式
- OCR：PDF 无文本层（扫描件）时尝试 pytesseract；未安装则跳过并告警
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import MIN_PDF_CHARS, OCR_FALLBACK

logger = logging.getLogger(__name__)

CHAPTER_RE = re.compile(r"^\s*(第[一二三四五六七八九十百零\d]+[章篇部]|附\s*则|目\s*录)\s*$")
SECTION_RE = re.compile(r"^\s*(第[一二三四五六七八九十百零\d]+[节条])\s*(.*)$")
ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百零\d]+条)\s*(.*)$")
ORDINAL_HEADING_RE = re.compile(r"^([一二三四五六七八九十]+|[0-9]+)[、.．]\s*\S")
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
DOCX_HEADING_STYLES = ("heading", "标题", "title", "标题 1", "标题 2", "标题 3")


@dataclass
class ParsedBlock:
    """解析出的块：标题或正文。"""

    text: str
    level: int = 0            # 0=正文, 1=章/大标题, 2=条/小节 ...
    is_heading: bool = False


def parse_file(path: Path, force_ocr: bool = False) -> tuple[str, list[ParsedBlock]]:
    """解析文件，返回 (全文, 块列表)。"""
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _parse_markdown(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix == ".pdf":
        return _parse_pdf(path, force_ocr=force_ocr)
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        return text, _blocks_from_lines(text.splitlines())
    raise ValueError(f"不支持的扩展名: {suffix}")


# ---------------------------------------------------------------- MD
def _parse_markdown(path: Path) -> tuple[str, list[ParsedBlock]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[ParsedBlock] = []
    for line in text.splitlines():
        m = MD_HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            blocks.append(ParsedBlock(m.group(2).strip(), level=level, is_heading=True))
        else:
            s = line.strip()
            if s:
                blocks.append(ParsedBlock(s, level=0, is_heading=False))
    return text, blocks


# ---------------------------------------------------------------- DOCX
def _parse_docx(path: Path) -> tuple[str, list[ParsedBlock]]:
    import docx  # python-docx

    d = docx.Document(str(path))
    blocks: list[ParsedBlock] = []
    for para in d.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        txt = para.text.strip()
        if not txt:
            continue
        if any(h in style for h in DOCX_HEADING_STYLES):
            lvl = 1
            for i, h in enumerate(("标题 1", "heading 1", "标题 2", "heading 2", "标题 3", "heading 3")):
                if h in style:
                    lvl = i // 2 + 1
                    break
            blocks.append(ParsedBlock(txt, level=lvl, is_heading=True))
        elif para.runs and all(r.bold for r in para.runs if r.text.strip()):
            blocks.append(ParsedBlock(txt, level=1, is_heading=True))
        else:
            blocks.append(ParsedBlock(txt, level=0, is_heading=False))
    text = "\n".join(b.text for b in blocks)
    return text, blocks


# ---------------------------------------------------------------- PDF
def _parse_pdf(path: Path, force_ocr: bool = False) -> tuple[str, list[ParsedBlock]]:
    import fitz  # PyMuPDF

    pdf = fitz.open(str(path))
    blocks: list[ParsedBlock] = []
    total_chars = 0
    for page in pdf:
        # 带字体大小的文本块 -> 标题启发式
        for blk in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                max_size = max(s["size"] for s in spans)
                total_chars += len(text)
                level, is_heading = _heading_by_text_and_size(text, max_size)
                blocks.append(ParsedBlock(text, level=level, is_heading=is_heading))
    pdf.close()
    text = "\n".join(b.text for b in blocks)
    if total_chars < MIN_PDF_CHARS and OCR_FALLBACK:
        text = _ocr_pdf(path)
        blocks = _blocks_from_lines(text.splitlines())
    return text, blocks


def _heading_by_text_and_size(text: str, font_size: float) -> tuple[int, bool]:
    if ARTICLE_RE.match(text):
        return 2, True
    if CHAPTER_RE.match(text):
        return 1, True
    if SECTION_RE.match(text):
        return 1, True
    if ORDINAL_HEADING_RE.match(text) and len(text) <= 40:
        return 1, True
    if font_size >= 14 and len(text) <= 40:
        return 1, True
    return 0, False


# ---------------------------------------------------------------- OCR
def _ocr_pdf(path: Path) -> str:
    """扫描件 OCR：优先 RapidOCR（ONNX，pip 可装），其次 pytesseract。"""
    import fitz

    pdf = fitz.open(str(path))
    pages: list[str] = []
    try:
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np

        engine = RapidOCR()
        for page in pdf:
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA -> RGB
                img = img[:, :, :3]
            result, _ = engine(img)
            if result:
                pages.append("".join(line[1] for line in result))
        logger.info("RapidOCR 兜底完成: %s", path.name)
    except ImportError:
        try:
            import pytesseract
            from PIL import Image

            for page in pdf:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(pytesseract.image_to_string(img, lang="chi_sim+eng"))
            logger.info("pytesseract OCR 兜底完成: %s", path.name)
        except ImportError:
            logger.warning("PDF 无文本层且未安装 RapidOCR/pytesseract，跳过 OCR：%s", path.name)
        except Exception as e:  # noqa: BLE001
            logger.warning("pytesseract OCR 失败: %s: %s", path.name, e)
    except Exception as e:  # noqa: BLE001
        logger.warning("OCR 失败: %s: %s", path.name, e)
    pdf.close()
    return "\n".join(pages)


# ---------------------------------------------------------------- 工具
def _blocks_from_lines(lines: list[str]) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        lvl, is_h = _heading_by_text_and_size(s, 0)
        blocks.append(ParsedBlock(s, level=lvl, is_heading=is_h))
    return blocks


def merge_blocks_to_text(blocks: list[ParsedBlock]) -> str:
    return "\n".join(b.text for b in blocks)
