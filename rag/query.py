# -*- coding: utf-8 -*-
"""Query Understanding / Query Rewrite。

两层：
1. 规则层（默认，离线可用）：法律领域主题词典 -> 查询扩展 + 意图分类
2. LLM 层（可选，OPENAI_API_KEY 存在时启用）：用 gpt-4o-mini 重写/扩展查询

输出 QueryPlan{rewritten, expanded, intent, target_domains}
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 法律主题词典：主题 -> (意图, 相关关键词/近义扩展)
TOPIC_LEXICON: dict[str, tuple[str, tuple[str, ...]]] = {
    "劳动用工": ("labor", ("劳动法", "劳动合同法", "试用期", "加班", "年假", "离职", "经济补偿",
                           "工伤", "社保", "劳动争议", "工资", "考勤", "入职", "奖惩")),
    "公司治理": ("corporate", ("公司章程", "股权", "印章", "法人授权", "股东会", "董事会")),
    "合同管理": ("contract", ("合同", "采购合同", "服务合同", "技术开发", "违约", "保密协议",
                             "租赁合同", "销售合同", "验收", "价款")),
    "知识产权": ("ip", ("商标", "著作权", "软件著作权", "商业秘密", "专利", "技术成果")),
    "信息安全": ("security", ("数据安全", "个人信息", "网络安全", "密码", "脱敏", "账号权限",
                             "数据泄露", "用户信息")),
    "财务报销": ("finance", ("报销", "发票", "差旅", "付款审批", "费用标准")),
    "采购管理": ("procurement", ("供应商", "采购审批", "招标投标", "采购合同")),
}

# 与上述主题绑定的法规/制度关键词（用于检索期 boost 或过滤）
DOMAIN_DOC_KEYWORDS = {
    "labor": ("劳动合同法", "劳动法", "劳动争议", "社会保险法", "就业促进法"),
    "corporate": ("公司法", "企业破产法", "印章", "授权"),
    "contract": ("民法典", "电子签名法", "合同法"),
    "ip": ("商标法", "反不正当竞争法", "著作权"),
    "security": ("数据安全法", "网络安全法", "个人信息保护法"),
    "procurement": ("招标投标法"),
}


@dataclass
class QueryPlan:
    original: str
    rewritten: str = ""                       # LLM 重写结果（无 LLM 时=original）
    expanded: str = ""                        # 规则扩展后的查询
    intents: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    used_llm: bool = False

    @property
    def effective_query(self) -> str:
        """用于检索的最终查询：LLM 重写优先，否则规则扩展。"""
        if self.rewritten and self.rewritten != self.original:
            return self.rewritten
        return self.expanded or self.original


def detect_intents(query: str) -> list[str]:
    intents = []
    for topic, (intent, kws) in TOPIC_LEXICON.items():
        if any(kw in query for kw in kws):
            intents.append(intent)
    return intents or ["general"]


def expand_query(query: str) -> str:
    """规则扩展：命中主题后追加领域关键词，缓解术语稀疏。"""
    intents = detect_intents(query)
    extra = []
    for it in intents:
        extra.extend(DOMAIN_DOC_KEYWORDS.get(it, ()))
    extra = [e for e in extra if e not in query][:3]
    if not extra:
        return query
    return f"{query}（{ ' '.join(extra)}）"


def _llm_rewrite(query: str, api_key: str | None, model: str, base_url: str | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content":
             "你是法律检索查询改写器。把用户的问句改写为适合法律知识库检索的查询："
             "1) 保留法律专有名词与条号；2) 补充同义/上位术语；3) 输出仅一行查询文本，不要解释。"},
            {"role": "user", "content": query},
        ],
    )
    return (resp.choices[0].message.content or query).strip()


def understand_query(query: str, allow_llm: bool = True) -> QueryPlan:
    plan = QueryPlan(original=query)
    plan.intents = detect_intents(query)
    plan.domains = [it for it in plan.intents if it != "general"]
    plan.expanded = expand_query(query)

    api_key = os.environ.get("OPENAI_API_KEY")
    if allow_llm and api_key:
        from .config import LLM_BASE_URL, LLM_MODEL
        try:
            plan.rewritten = _llm_rewrite(query, api_key, LLM_MODEL, LLM_BASE_URL)
            plan.used_llm = True
            logger.info("LLM 查询重写: %s -> %s", query, plan.rewritten)
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 查询重写失败，回退规则扩展: %s", e)
    return plan
