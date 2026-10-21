"""PDF 成品文本抽取复核: 报告 md 通过审计不等于导出的 PDF 也对。

导出的 PDF 是要提交的工件, 里面若残留作废口径的数字 (旧版曾发生过), 就是虚假结论。
所以这里把 PDF 抽成文本再检两件事:
  (1) 作废数字模式一个都不许出现 (含迭代 7 的成本绝对值与百分比改进);
  (2) 5.9 消融小节的关键数字与表格表头确实排版出来 (缺列/缺行通常是导出截断)。

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
# 正文与 PDF 都必须出现的关键串 (排版时空格/换行会变, 故压缩后再比)
PRESENT = [
    "5.9 消融",
    "314,374", "+1.70%", "30/30", "0.871 (0.890)", "0.418 (0.091)",
    "0.973 / 0.967 掉到 0.875 / 0.848",
    "强度", "三阶段交替训练", "唯一通过判读门槛的部件",
    "600 个评估世界", "−0.09%",
]


def squash(text: str) -> str:
    """排版抽取会在单元格间断行, 压缩空白后再做子串判断。"""
    return re.sub(r"\s+", " ", text.replace("\u2212", "-"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="report/课程设计报告.pdf")
    args = ap.parse_args()
    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF 不存在: {args.pdf} (先跑 experiments/md_to_pdf.py)")
    doc = fitz.open(args.pdf)
    body = squash("".join(p.get_text() for p in doc))
    print(f"{args.pdf}: {doc.page_count} 页, 抽取 {len(body)} 字符")

    bad = 0
    for pat in VOID:
        hits = re.findall(pat, body)
        if hits:
            bad += 1
            print(f"  [XX] 作废数字残留 {pat!r} x{len(hits)}")
    for token in PRESENT:
        if squash(token) not in body:
            bad += 1
            print(f"  [XX] 正文应有但 PDF 里没有: {token!r}")
    if len(body) < 8000:
        bad += 1
        print(f"  [XX] 抽取文本过短 ({len(body)}), 导出可能被截断")
    if doc.page_count < 20:
        bad += 1
        print(f"  [XX] 页数 {doc.page_count} 偏少, 图表可能没进去")
    doc.close()

    print(f"检查 {len(VOID) + len(PRESENT) + 2} 项, 不符 {bad} 项")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
