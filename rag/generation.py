# -*- coding: utf-8 -*-
"""LLM 生成：结构化输出 + 引用标记。

- OpenAIGenerator：gpt-4o-mini，要求 JSON 输出 {answer, citations:[引用串]}
- LocalGenerator（无 API Key 时的兜底）：抽取式——从上下文证据中选句拼装，
  保证无 Key 也能跑通全链路、产出可评估结果
- generate_answer()：统一入口，输出 Answer（含接地性校验）
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .citation import validate_groundedness
from .context import BuiltContext
from .models import Answer

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])")
_REF_RE = re.compile(r"\[(\d+)\]")


class OpenAIGenerator:
    """真实 LLM API Adapter（结构化生成）。"""

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        from .config import OPENAI_MODEL, OPENAI_TEMPERATURE
        self.model = model or OPENAI_MODEL
        self.temperature = temperature if temperature != 0.2 else OPENAI_TEMPERATURE

    def generate(self, question: str, ctx: BuiltContext) -> tuple[str, list[str]]:
        from openai import OpenAI

        refs_desc = "\n".join(f"[{i}] {ctx.refs[i]}" for i in sorted(ctx.refs))
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        prompt = (
            "你是企业法律知识助手。基于【参考资料】回答用户问题，要求：\n"
            "1) 只使用参考资料中的信息，不要编造；\n"
            "2) 每个结论后标注引用编号，格式 [n]，对应【参考资料】编号；\n"
            "3) 若资料不足以回答，明确说明并指出缺失部分；\n"
            "4) 回答简洁、分点，输出 JSON：{\"answer\": \"...\", \"citations\": [\"引用串1\", ...]}\n\n"
            f"【问题】\n{question}\n\n"
            f"【引用列表】\n{refs_desc}\n\n"
            f"【参考资料】\n{ctx.text}\n\n"
            "输出 JSON："
        )
        resp = client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        return answer, citations


class LocalGenerator:
    """本地抽取式兜底：不依赖 API，保证链路可跑。

    策略：命中引用单元里与问题共现词最多的句子 -> 按证据顺序拼装，附 [n] 引用。
    输出前清洗 Markdown/流程图噪声，保证引用句与 [n] 一一对应。
    """

    def __init__(self, top_sentences: int = 4, max_sent_len: int = 160):
        self.top_sentences = top_sentences
        self.max_sent_len = max_sent_len

    @staticmethod
    def _tokens(text: str) -> set[str]:
        import jieba
        toks = set(t for t in jieba.lcut(text) if len(t.strip()) >= 2)
        toks.discard("以及")
        return toks

    @staticmethod
    def _is_noise(sent: str) -> bool:
        """过滤 Markdown / 流程图 / 表格噪声行。"""
        s = sent.strip()
        if not s or len(s) < 12:
            return True
        if s.startswith(("#", "```", "-", "*", "|", ">", "①", "②", "③", "→", "●", "▪", "表格")):
            return True
        if "```" in s or "→" in s or "├" in s or "└" in s or "│" in s:
            return True
        if s.startswith(("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、")) and len(s) <= 30:
            return True
        return False

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按换行与标点切句，去掉噪声。"""
        out = []
        for line in text.splitlines():
            for s in _SENT_SPLIT.split(line):
                s = s.strip()
                if s and not LocalGenerator._is_noise(s):
                    out.append(s)
        return out

    def generate(self, question: str, ctx: BuiltContext) -> tuple[str, list[str]]:
        q_toks = self._tokens(question)
        scored_sents: list[tuple[float, str, int]] = []
        for unit in ctx.units:
            for sent in self._split_sentences(unit.parent.text):
                toks = self._tokens(sent)
                overlap = len(q_toks & toks) if q_toks else 0
                if overlap > 0:
                    scored_sents.append((overlap, sent, unit.ref_index))
        # 同分时优先短句（信息密度高、噪声少）
        scored_sents.sort(key=lambda x: (-x[0], len(x[1])))
        chosen = scored_sents[:self.top_sentences]

        if not chosen:
            answer = f"未能在当前知识库中找到与「{question}」直接相关的条文。建议补充检索关键词或咨询专业人士。"
            return answer, []

        # 按引用顺序输出，同引用合并
        by_ref: dict[int, list[str]] = {}
        for _, s, ref in chosen:
            by_ref.setdefault(ref, []).append(s)
        answer_parts = []
        citations = []
        for ref in sorted(by_ref):
            text = "".join(by_ref[ref])[:self.max_sent_len * self.top_sentences]
            answer_parts.append(f"{text}[{ref}]")
            citations.append(ctx.refs[ref])
        return "".join(answer_parts), citations


def generate_answer(question: str, ctx: BuiltContext,
                    generator: Optional[object] = None) -> Answer:
    """统一生成入口：LLM 优先，本地兜底，输出带接地性校验的结构化 Answer。"""
    if generator is None:
        if os.environ.get("OPENAI_API_KEY"):
            generator = OpenAIGenerator()
        else:
            logger.info("未检测到 OPENAI_API_KEY，使用本地抽取式生成兜底")
            generator = LocalGenerator()

    try:
        answer_text, citations = generator.generate(question, ctx)
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM 生成失败，回退本地生成: %s", e)
        generator = LocalGenerator()
        answer_text, citations = generator.generate(question, ctx)

    # 引用归一化：只保留上下文里真实存在的引用
    valid_refs = {i for i in ctx.refs}
    answer_text = _REF_RE.sub(
        lambda m: f"[{m.group(1)}]" if int(m.group(1)) in valid_refs else "",
        answer_text,
    )
    used_refs = [ctx.refs[i] for i in sorted(valid_refs)
                 if f"[{i}]" in answer_text]
    if not used_refs and citations:
        # 某些 LLM 引用串不带编号：用上下文引用兜底
        used_refs = [c for c in citations if c in ctx.citations] or ctx.citations[:1]

    groundedness, unsupported = validate_groundedness(answer_text, ctx)
    return Answer(
        answer=answer_text,
        citations=used_refs,
        groundedness=groundedness,
        unsupported=unsupported,
        meta={"generator": type(generator).__name__},
    )
