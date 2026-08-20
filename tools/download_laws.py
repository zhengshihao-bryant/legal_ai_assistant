# -*- coding: utf-8 -*-
"""
从国家法律法规数据库(flk.npc.gov.cn)下载法律法规官方 PDF。

管线: 搜索(searchType=1 全文检索) -> 精确标题匹配 -> 取 bbbs -> 取 data.url -> 下载保存
版本选择: 在"施行日期 <= 今天"的现行版本中取最新;同日期优先 sxx==3(现行有效)。
特性:
  - 幂等: 目标文件已存在且体积正常(>50KB)则跳过(--force 可强制重下)
  - PDF 失败时降级尝试 DOCX 并如实记录
  - 生成 data/raw/laws/download_report.md 报告
用法:
  python tools/download_laws.py
  python tools/download_laws.py --name 中华人民共和国电子签名法
  python tools/download_laws.py --force --name 中华人民共和国民法典
"""
import argparse
import datetime
import os
import re
import time

import requests
import urllib3

urllib3.disable_warnings()

BASE = "https://flk.npc.gov.cn"
SEARCH_URL = BASE + "/law-search/search/list"
DOWNLOAD_URL = BASE + "/law-search/download/pc"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAWS_DIR = os.path.join(ROOT, "data", "raw", "laws")
REPORT = os.path.join(LAWS_DIR, "download_report.md")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
H = {"User-Agent": UA, "Referer": "https://flk.npc.gov.cn/", "Content-Type": "application/json"}
H_DL = {"User-Agent": UA, "Referer": "https://flk.npc.gov.cn/"}

LAWS = [
    ("中华人民共和国劳动合同法", "劳动合同法"),
    ("中华人民共和国劳动法", "劳动法"),
    ("中华人民共和国社会保险法", "社会保险法"),
    ("中华人民共和国就业促进法", "就业促进法"),
    ("中华人民共和国劳动争议调解仲裁法", "劳动争议调解仲裁法"),
    ("最高人民法院关于审理劳动争议案件适用法律问题的解释（一）", "劳动争议"),
    ("最高人民法院关于审理劳动争议案件适用法律问题的解释（二）", "劳动争议"),
    ("中华人民共和国民法典", "民法典"),
    ("中华人民共和国电子签名法", "电子签名法"),
    ("中华人民共和国招标投标法", "招标投标法"),
    ("中华人民共和国公司法", "公司法"),
    ("中华人民共和国企业破产法", "企业破产法"),
    ("中华人民共和国反不正当竞争法", "反不正当竞争法"),
    ("中华人民共和国专利法", "专利法"),
    ("中华人民共和国著作权法", "著作权法"),
    ("中华人民共和国商标法", "商标法"),
    ("中华人民共和国个人信息保护法", "个人信息保护法"),
    ("中华人民共和国数据安全法", "数据安全法"),
    ("中华人民共和国网络安全法", "网络安全法"),
]

MIN_SIZE = 50 * 1024
TODAY = datetime.date.today().isoformat()


def norm(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"[<>\s（）()\[\]【】]", "", s)


def eff_date(row):
    return row.get("sxrq") or row.get("gbrq") or ""


def find_row(term, target):
    """返回 (bbbs, meta)。取标题精确匹配、施行日期<=今天的现行最新版本。"""
    payload = {"searchContent": term, "searchType": 1, "searchRange": 1, "pageNum": 1, "pageSize": 50}
    r = requests.post(SEARCH_URL, json=payload, headers=H, verify=False, timeout=30)
    r.raise_for_status()
    rows = r.json().get("rows") or []
    target_n = norm(target)
    cands = [x for x in rows if norm(x.get("title")) == target_n]
    if not cands:
        return None, None
    pool = [x for x in cands if eff_date(x) and eff_date(x) <= TODAY] or cands
    pool.sort(key=lambda x: (x.get("sxx") == 3, eff_date(x)), reverse=True)
    row = pool[0]
    meta = {"bbbs": row.get("bbbs"), "gbrq": row.get("gbrq"), "sxrq": row.get("sxrq"),
            "zdjg": row.get("zdjgName"), "sxx": row.get("sxx"), "flxz": row.get("flxz")}
    return row.get("bbbs"), meta


def fetch_download_url(bbbs, fmt):
    r = requests.get(DOWNLOAD_URL, params={"format": fmt, "bbbs": bbbs},
                     headers=H_DL, verify=False, timeout=30)
    r.raise_for_status()
    j = r.json()
    if str(j.get("code")) != "200":
        return None
    return (j.get("data") or {}).get("url")


def fetch_bytes(url, fmt):
    r = requests.get(url, headers=H_DL, verify=False, timeout=300, allow_redirects=True)
    if r.status_code != 200 or len(r.content) < MIN_SIZE:
        return None
    magic = b"%PDF" if fmt == "pdf" else b"PK\x03\x04"
    if r.content[:4] != magic:
        return None
    return r.content


def already_have(name):
    for ext in ("pdf", "docx"):
        p = os.path.join(LAWS_DIR, name + "." + ext)
        if os.path.exists(p) and os.path.getsize(p) > MIN_SIZE:
            return p
    return None


def download_law(name, term, force=False):
    if not force:
        have = already_have(name)
        if have:
            return ("跳过(已存在)", have, os.path.getsize(have), None)

    bbbs, meta = None, None
    for attempt in range(3):
        try:
            bbbs, meta = find_row(term, name)
            break
        except Exception as e:
            time.sleep(2)
    if not bbbs:
        return ("失败(未检索到)", None, 0, {"error": "search failed"})

    for fmt in ("pdf", "docx"):
        try:
            url = fetch_download_url(bbbs, fmt)
            if not url:
                continue
            data = fetch_bytes(url, fmt)
            if not data:
                continue
            path = os.path.join(LAWS_DIR, name + "." + fmt)
            with open(path, "wb") as f:
                f.write(data)
            # 成功下载 PDF 后,清理同名其他格式残留
            for other in ("pdf", "docx"):
                if other != fmt:
                    op = os.path.join(LAWS_DIR, name + "." + other)
                    if os.path.exists(op):
                        os.remove(op)
            return ("成功(%s)" % fmt, path, len(data), meta)
        except Exception as e:
            meta["error_" + fmt] = repr(e)
            time.sleep(1)
    return ("失败(下载异常)", None, 0, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="只处理某一部(用完整文件名)")
    ap.add_argument("--limit", type=int, help="只处理前 N 部")
    ap.add_argument("--force", action="store_true", help="忽略已存在,强制重下")
    args = ap.parse_args()

    os.makedirs(LAWS_DIR, exist_ok=True)
    todo = LAWS
    if args.name:
        todo = [x for x in LAWS if x[0] == args.name]
    if args.limit:
        todo = todo[: args.limit]

    results = []
    for i, (name, term) in enumerate(todo, 1):
        print("[%d/%d] %s ..." % (i, len(todo), name), flush=True)
        status, path, size, meta = download_law(name, term, force=args.force)
        results.append((name, status, path, size, meta))
        print("   -> %s (%d bytes)" % (status, size), flush=True)

    lines = [
        "# 法律法规下载报告", "",
        "- 生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "- 来源: 国家法律法规数据库 https://flk.npc.gov.cn(官方)",
        "- 说明: 仅收录官方原文;下载失败或未检索到的不生成替代文本。版本选择规则:施行日期<=当天的现行最新版。",
        "", "| 序号 | 法律 | 状态 | 文件 | 大小(KB) | 公布日期 | 施行日期 | bbbs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, (name, status, path, size, meta) in enumerate(results, 1):
        fn = os.path.basename(path) if path else "-"
        gbrq = (meta or {}).get("gbrq") or "-"
        sxrq = (meta or {}).get("sxrq") or "-"
        bbbs = (meta or {}).get("bbbs") or "-"
        lines.append("| %d | %s | %s | %s | %d | %s | %s | %s |" % (
            i, name, status, fn, size // 1024, gbrq, sxrq, bbbs))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n报告已写入:", REPORT)
    ok = sum(1 for r in results if r[1].startswith("成功") or r[1].startswith("跳过"))
    print("汇总: 成功/跳过 %d / %d" % (ok, len(results)))


if __name__ == "__main__":
    main()
