"""审计器的反向用例: 故意把文档改错, 确认 `verify_report_numbers.py` 会拒绝。

审计"0 项不符"只有在这个检查器**能**抓到错数时才有意义, 所以每条变异都是一次反向用例:
把正文里某个数改掉 (或删掉一行), 检查器必须返回非零并指出是哪一项。
任何一条被漏掉, 就说明对应的检查是空转的 —— 那正是本项目在旧口径上吃过亏的地方。

变异同时打到报告与 README (后者复述了 5.9 的消融数字, 跨文档一致性也要能被抓住)。

用法:
    python experiments/selfcheck_verifier.py
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.verify_report_numbers as V                                  # noqa: E402

FILES = {"报告": "report/课程设计报告.md", "README": "README.md"}


def drop_line(text: str, pat: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith(pat))


# (标签, 打到哪份文档, 变异) —— 变异必须真的动到字符, 否则这条用例是空转的
MUTATIONS = [
    ("5.3 跨场景均值 +0.69% → +0.79%", "报告", lambda t: t.replace("+0.69%", "+0.79%")),
    ("验收 600 个评估世界 → 599", "报告", lambda t: t.replace("600 个评估世界", "599 个评估世界")),
    ("5.4.1 α 饱和例外 67% → 70%", "报告", lambda t: t.replace("（67%）", "（70%）")),
    ("5.7 E 剖面 0.966–0.997 → 0.960–0.997", "报告",
     lambda t: t.replace("0.966–0.997", "0.960–0.997")),
    ("5.4.2 加粗的 **30/30** → **29/30**", "报告", lambda t: t.replace("**30/30**", "**29/30**")),
    ("5.9 参照成本 309,116 → 309,117", "报告", lambda t: t.replace("309,116", "309,117")),
    ("表 5.9-1 变体成本 314,374 → 314,274", "报告", lambda t: t.replace("314,374", "314,274")),
    ("表 5.9-1 配对差 +1.70% → +1.07%", "报告", lambda t: t.replace("**+1.70%**", "**+1.07%**")),
    ("表 5.9-1 逐种子符号 −0.27 → +0.27", "报告",
     lambda t: t.replace("\u22120.27 / +0.16 / +0.31", "+0.27 / +0.16 / +0.31")),
    ("表 5.9-1 删掉交替训练整行", "报告", lambda t: drop_line(t, "| 三阶段交替训练 |")),
    ("5.9 结论二 γ 0.848 → 0.849", "报告", lambda t: t.replace("0.875 / 0.848", "0.875 / 0.849")),
    ("1.3 塑形代价区间 1.70% → 1.90%", "报告", lambda t: t.replace("贵 1.40%~1.70%", "贵 1.40%~1.90%")),
    ("摘要 塑形代价 1.40% → 1.50%", "报告", lambda t: t.replace("1.70% 与 1.40%", "1.70% 与 1.50%")),
    ("README 复述的 Δ 与重算脱钩 +0.26% → +0.96%", "README",
     lambda t: t.replace("+0.26% (24/30)", "+0.96% (24/30)")),
    ("README 消融表补写一个没跑过的组合", "README",
     lambda t: t.replace("| 三阶段交替训练 | — |", "| 三阶段交替训练 | +0.05% (22/30) |")),
]


def run(texts: dict, paths: dict) -> tuple:
    """把 (可能变异的) 两份文档写到临时文件, 用审计器的真实入口跑一遍。"""
    for key, path in paths.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(texts[key])
    sys.argv = ["selfcheck", "--md", paths["报告"], "--readme", paths["README"]]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = V.main()
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return rc, (lines[0] if lines else "(无输出)")


def main() -> int:
    tmp = tempfile.gettempdir()
    paths = {"报告": os.path.join(tmp, "_selfcheck_报告.md"),
             "README": os.path.join(tmp, "_selfcheck_README.md")}
    base = {k: open(v, encoding="utf-8").read() for k, v in FILES.items()}
    rc, first = run(base, paths)
    print(f"基准 (未变异): rc={rc}  {first[:90]}")
    if rc != 0:
        print("文档本身就没通过审计, 先修正文再跑自检")
        return 1

    missed = []
    for label, key, mut in MUTATIONS:
        texts = dict(base)
        if mut(texts[key]) == texts[key]:
            print(f"  [空转] {label}: 变异没改到任何字符, 字符串写错了")
            missed.append(label)
            continue
        texts[key] = mut(texts[key])
        rc, first = run(texts, paths)
        hit = bool(re.search(r"\[XX\]|Traceback|Error", first))
        print(f"  [{'抓到' if rc == 1 and hit else '漏掉':<4}] {label:<40} rc={rc}  {first[:66]}")
        if rc != 1 or not hit:
            missed.append(label)
    print(f"\n{len(MUTATIONS)} 条变异, 漏掉 {len(missed)} 条"
          + (f": {missed}" if missed else " (检查器对每条错数都会失败)"))
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
