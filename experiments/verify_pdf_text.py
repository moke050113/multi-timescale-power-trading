"""PDF 成品文本抽取复核: 报告 md 通过审计不等于导出的 PDF 也对。

导出的 PDF 是要提交的工件, 里面若残留作废口径的数字 (旧版曾发生过), 就是虚假结论。
这里把 PDF 抽成文本再检三件事:
  (1) 作废数字模式一个都不许出现 (含迭代 7 的成本绝对值与百分比改进);
  (2) **由当前 md 现场抽出的**关键单元格与结论句确实在排版稿里 (缺列/缺行通常是导出截断);
  (3) md 里的占位符没有被带进 PDF (说明导出用的是没填完的稿子)。

第 (2) 项刻意不写死数字: 上一版在这里硬编码了消融表的单元格, 重训之后那些数全变,
检查器就退化成"提醒我去改脚本"而不是"挡住错稿"。现在它跟着 md 走, md 由
`verify_report_numbers.py` 绑定到 `results/`, 三者串成一条链。

用法:
    python experiments/verify_pdf_text.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import fitz  # PyMuPDF

VOID = [
    r"\+38\.3", r"38\.3%",              # 迭代 7 的"降本"主结论 (重复计量 3.25 倍)
    r"22[12],\d{3}", r"199,167", r"186,569", r"168,565",   # 旧口径年成本绝对值
    r"降本\s*2[014]\.\d%", r"全面优于", r"完全达成",
]
# 与具体数字无关、只验小节是否排版出来
ANCHORS = [
    "5.9 消融", "三阶段交替训练", "唯一通过判读门槛的部件",
    "600 个评估世界", "适用范围与局限", "第 6 章 结论与展望",
    "grad_steps", "phase_history", "action_oob", "source_lock",
]


def squash(text: str) -> str:
    """排版抽取会在单元格间断行, 压缩空白后再做子串判断。"""
    return re.sub(r"\s+", " ", text.replace("\u2212", "-"))


def clean(cell: str) -> str:
    """去掉 markdown 强调、行内代码与公式定界符, 只留 PDF 里会出现的字符。"""
    return squash(cell.replace("**", "").replace("`", "").replace("$", ""))


def md_tokens(md: str) -> list:
    """从当前报告正文抽出必须在 PDF 里出现的数字型串 (随 md 走, 不写死)。"""
    out = []
    # 表 5.9-1 每一行: 变体成本 + 配对 Δ + 更贵世界数
    for line in md.splitlines():
        if re.match(r"^\| (未来暴露成本塑形|现货奖励放大|三阶段交替训练) \|", line):
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            out += [clean(c) for c in (cells[2], cells[3], cells[4]) if clean(c)]
    # 表 5.4-1 首行 (场景 A 基线成本) —— 主表若被截断, 这里先炸
    m = re.search(r"^\| A_.*?\|([\d,\.]+)", md, re.M)
    if m:
        out.append(clean(m.group(1)))
    # 消融的逐角色优化器步数那句 (正文写死在 md 里, 由审计器绑定到 JSON); 先压平再找, 正文会折行
    m = re.search(r"完整方法 中长期 \d+ 次 / 柔性 \d+ 次[，,] *关掉交替 \d+ 次 / \d+ 次", squash(md))
    if m:
        out.append(clean(m.group(0)))
    return [t for t in dict.fromkeys(out) if len(t) >= 4]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="report/课程设计报告.pdf")
    ap.add_argument("--md", default="report/课程设计报告.md")
    args = ap.parse_args()
    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF 不存在: {args.pdf} (先跑 experiments/md_to_pdf.py)")
    md = open(args.md, encoding="utf-8").read()
    doc = fitz.open(args.pdf)
    body = squash("".join(p.get_text() for p in doc))
    print(f"{args.pdf}: {doc.page_count} 页, 抽取 {len(body)} 字符")

    bad = 0
    for pat in VOID:
        hits = re.findall(pat, body)
        if hits:
            bad += 1
            print(f"  [XX] 作废数字残留 {pat!r} x{len(hits)}")
    for token in ANCHORS + md_tokens(md):
        if squash(token) not in body:
            bad += 1
            print(f"  [XX] 正文应有但 PDF 里没有: {token!r}")
    left = sorted(set(re.findall(r"<[A-Z0-9_]{2,}>", body)))
    if left:
        bad += 1
        print(f"  [XX] PDF 里还有未填占位符: {left}")
    if len(body) < 8000:
        bad += 1
        print(f"  [XX] 抽取文本过短 ({len(body)}), 导出可能被截断")
    if doc.page_count < 20:
        bad += 1
        print(f"  [XX] 页数 {doc.page_count} 偏少, 图表可能没进去")
    doc.close()

    total = len(VOID) + len(ANCHORS) + 3
    print(f"检查 {total} 项 (另有 {len(md_tokens(md))} 项由当前正文抽出), 不符 {bad} 项")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
