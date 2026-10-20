"""报告正文数字自检: 把 5.3~5.7 引用的数字从 results/four_arm_benchmark.json 重算一遍。

存在的理由: 本文的性能结论全部落在正文叙述里 (配对改进、覆盖率剖面、成交均价)。
人工填数一旦与 JSON 脱钩就成了虚假结论, 所以每条主张都同时检查两件事:
  (1) 报告 md 里确实写着这个数;
  (2) 从 JSON 重算出来的数与它相符 (容差按正文给出的小数位取)。
另外表 5.4-1 ~ 5.4-4 必须与 `report_tables.py` 当场生成的表格逐字符相同。

用法:
    python experiments/verify_report_numbers.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.plot_four_arm_results import index_runs, load          # noqa: E402
from experiments.report_tables import layer_share, render, seed_pairwise  # noqa: E402

LEARNERS = ("sac_rule", "dual_sac")


class Checker:
    def __init__(self, md: str):
        # 正文用排版减号 "−", 计算结果用 ASCII "-"; 统一到后者
        self.md = md.replace("\u2212", "-")
        self.bad = 0
        self.n = 0

    def text(self, label: str, token: str) -> None:
        self.n += 1
        if token.replace("\u2212", "-") not in self.md:
            self.bad += 1
            print(f"  [XX] {label}: 报告里找不到 {token!r}")

    def either(self, label: str, tokens) -> None:
        """同一主张在正文里有加粗/非加粗两种写法。"""
        self.n += 1
        if not any(t.replace("\u2212", "-") in self.md for t in tokens):
            self.bad += 1
            print(f"  [XX] {label}: 报告里这几种写法都没有 {[t[:36] for t in tokens]}")

    def value(self, label: str, computed: float, written: float,
              token: str, tol: float) -> None:
        self.n += 1
        present = token.replace("\u2212", "-") in self.md
        close = abs(computed - written) <= tol
        if not (present and close):
            self.bad += 1
            print(f"  [XX] {label}: 重算={computed:.6g} 正文={written:.6g} "
                  f"{'超差' if not close else ''}{'正文缺失' if not present else ''} ({token!r})")

    def finish(self) -> int:
        print(f"检查 {self.n} 项主张, 不符 {self.bad} 项")
        return 1 if self.bad else 0


def rng(values) -> tuple:
    a = np.asarray(values, dtype=float)
    return float(a.min()), float(a.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/four_arm_benchmark.json")
    ap.add_argument("--md", default="report/课程设计报告.md")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    md = open(args.md, encoding="utf-8").read()
    data = load(args.json)
    by_key, scenarios, arms = index_runs(data)
    base = data["meta"]["baseline_arm"]
    strong = "rolling_rule"
    S = {s[:1]: s for s in scenarios}
    c = Checker(md)

    # ---------- 1. 四张表与生成器逐字符一致 ----------
    blocks, cur = [], []
    for line in render(data, strong=strong).splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            if len(cur) >= 3:
                blocks.append("\n".join(cur))
            cur = []
    if len(cur) >= 3:
        blocks.append("\n".join(cur))
    if len(blocks) != 4:
        print(f"  [XX] 生成器给出 {len(blocks)} 张表, 报告应为 4 张")
        return 1
    for i, t in enumerate(blocks, 1):
        c.text(f"表 5.4-{i} 逐字符等于生成器输出", t)

    # ---------- 2. 验收不变量 ----------
    worlds = [w for r in data["runs"] for w in r["worlds"]]
    chk = [w["checks"] for w in worlds]
    c.value("评估世界数", float(len(worlds)), 600, "600 个评估世界", 0)
    c.value("组合数", float(data["meta"]["n_runs"]), 60, "60 个组合", 0)
    c.value("failures 条数", float(len(data["failures"])), 0, "`failures` 为空", 0)
    lo, hi = rng([wch["energy_residual_mwh"] for wch in chk])
    c.value("电量守恒残差上限", hi / 1e-13, 8, r"8\times10^{-13}", 0.5)
    c.value("分项之和与市场成本的最大残差",
            max(abs(wch["cost_recomputed_residual"]) for wch in chk), 0,
            "600 个世界全为 0", 1e-9)
    c.value("合同层负电量/负成本总数",
            float(sum(wch["contract_qty_negative"] + wch["contract_cost_negative"]
                      for wch in chk)), 0, "无虚假退款", 0)
    c.value("37 步闭环世界数", float(sum(wch["steps_37"] for wch in chk)), 600,
            "37 步闭环", 0)
    c.value("设备完成任务世界数", float(sum(wch["device_task_met"] for wch in chk)), 600,
            "设备日任务按时完成", 0)
    c.value("总成本 − 市场成本 上限", max(abs(w["cost"] - w["market_cost"]) for w in worlds),
            160, "最多 160 元", 0.5)

    def costs(scn, arm):
        return float(np.mean([r["mean_cost"] for r in by_key[(scn, arm)]]))

    def delta(scn, arm, ref):
        ps = seed_pairwise(by_key, scn, arm, ref)
        return float(np.mean([p[1] for p in ps])), ps

    # ---------- 3. 5.3 非学习基线 ----------
    for scn in scenarios:
        s = scn[:1]
        fx, rl = costs(scn, base), costs(scn, strong)
        c.value(f"{s} fixed_rule 年成本", fx, round(fx), f"| {s} ", 0.49)
        c.text(f"{s} fixed_rule 成本数字", f"{fx:,.0f}")
        c.text(f"{s} rolling_rule 成本数字", f"{rl:,.0f}")
        d, _ = delta(scn, strong, base)
        c.value(f"{s} 滚动规则改进", d, round(d, 1), f"{d:+.1f}%", 0.05)

    # ---------- 4. 跨场景均值 ----------
    for arm, ref, written, token in [
        (strong, base, 0.69, "+0.69%"), ("dual_sac", base, 0.61, "+0.61%"),
        ("dual_sac", strong, -0.09, "\u22120.09%"), ("sac_rule", base, 0.25, "+0.25%"),
        ("sac_rule", strong, -0.45, "\u22120.45%"),
    ]:
        v = float(np.mean([delta(scn, arm, ref)[0] for scn in scenarios]))
        c.value(f"5 场景均值 {arm} vs {ref}", v, written, token, 0.005)

    # ---------- 5. 5.4.1 年度饱和 ----------
    for scn in scenarios:
        s = scn[:1]
        al = [w["alphas"][0] for a in LEARNERS for r in by_key[(scn, a)] for w in r["worlds"]]
        v = float(np.mean(al))
        c.text(f"5.4.1 {s} 学习臂 α 列", f"| {s} | ")
        c.value(f"5.4.1 {s} 学习臂 α 合并均值", v, round(v, 3), f"| {v:.3f} |", 5e-4)
    frac = {f"{scn[:1]}/{a}": np.mean([w["alphas"][0] > 0.885
              for r in by_key[(scn, a)] for w in r["worlds"]])
            for scn in scenarios for a in LEARNERS}
    exceptions = {k: v for k, v in frac.items() if v < 1.0}
    c.value("α>0.885 满额组合数", float(len(frac) - len(exceptions)), 9,
            "唯一例外是 B 场景的 `dual_sac`（67%）", 0)
    if exceptions != {"B/dual_sac": 2 / 3}:
        print(f"  [XX] α 饱和例外与正文不符: {exceptions}")
        c.bad += 1
    c.n += 1

    # ---------- 6. 5.4.2 dual 相对 sac ----------
    ds = []
    for scn in scenarios:
        s = scn[:1]
        d, ps = delta(scn, "dual_sac", "sac_rule")
        ds.append(d)
        w = sum(p[2] for p in ps)
        n = sum(p[3] for p in ps)
        row = f"| {s} | {d:+.2f}% |"
        c.value(f"5.4.2 {s} dual−sac", d, round(d, 2), row, 0.005)
        c.either(f"5.4.2 {s} 更省世界数", [f"{row} {w}/{n} |", f"{row} **{w}/{n}** |"])
        c.text(f"5.4.2 {s} 逐种子列", " / ".join(f"{p:+.2f}%" for _, p, _, _ in ps))
    c.value("5.4.2 五场景均值", float(np.mean(ds)), 0.365, "**+0.365%**", 0.0005)

    # ---------- 7. 5.7 剖面 / 均价 / 强度 ----------
    def prof(s, arm):
        return np.mean([w["gammas"] for r in by_key[(S[s], arm)] for w in r["worlds"]], axis=0)

    e_d, e_s, e_r = (prof("E", a) for a in ("dual_sac", "sac_rule", strong))
    b_d, b_s, b_r = (prof("B", a) for a in ("dual_sac", "sac_rule", strong))
    b_mid, b_all = np.concatenate([b_d[6:9], b_s[6:9]]), np.concatenate([b_d, b_s])
    for label, vals, written_lo, written_hi, token in [
        ("E dual 剖面", e_d, 0.966, 0.997, "0.966–0.997"),
        ("E sac 除首月", e_s[1:], 0.982, 0.999, "0.982–0.999"),
        ("E rolling 除首月", e_r[1:], 0.996, 0.998, "0.996–0.998"),
        ("B 学习臂 7–9 月", b_mid, 0.893, 0.899, "0.893–0.899"),
        ("B 学习臂剖面", b_all, 0.893, 0.962, "0.893–0.962"),
        ("B rolling 除首月", b_r[1:], 0.881, 0.893, "0.881–0.893"),
    ]:
        lo, hi = rng(vals)
        c.value(f"5.7 {label} 下界", lo, written_lo, token, 5e-4)
        c.value(f"5.7 {label} 上界", hi, written_hi, token, 5e-4)
    for label, v, written, token in [
        ("E sac 首月", e_s[0], 0.942, "除首月 0.942"),
        ("E rolling 首月", e_r[0], 0.968, "除首月 0.968"),
        ("B rolling 首月", b_r[0], 0.983, "除首月 0.983"),
        ("B dual 最低月", b_d.min(), 0.893, "第 9 月\n   0.893"),
        ("B sac 最低月", b_s.min(), 0.894, "第 12 月 0.894"),
    ]:
        c.value(f"5.7 {label}", float(v), written, token, 5e-4)
    argmins = [int(np.argmin(prof(s, a))) + 1 for s in "ABCDE" for a in LEARNERS]
    c.value("5.7 最低月落在 Q4 的剖面数", float(sum(m >= 10 for m in argmins)), 1,
            "最低月落在第 4 季度的只有 1 个", 0)
    others = [m for m in argmins if m < 10]
    c.text("5.7 其余剖面最低月集合", "1、5、8、9 月")
    if sorted(set(others)) != [1, 5, 8, 9]:
        print(f"  [XX] 5.7 其余剖面最低月集合实际为 {sorted(set(others))}")
        c.bad += 1
    c.n += 1

    def spot_price(s, arm):
        sc = [w["cost_breakdown"]["spot"] for r in by_key[(S[s], arm)] for w in r["worlds"]]
        en = [w["energy_breakdown"]["spot"] for r in by_key[(S[s], arm)] for w in r["worlds"]]
        return float(np.mean(sc) / np.mean(en)), float(np.mean(en))

    c.value("5.7 E fixed 现货成交均价", spot_price("E", base)[0], 493.3, "493.3 元/MWh", 0.05)
    c.value("5.7 B fixed 现货成交均价", spot_price("B", base)[0], 294.9, "294.9 元/MWh", 0.05)
    all_lower = all(spot_price(s, "dual_sac")[0]
                    < min(spot_price(s, base)[0], spot_price(s, strong)[0]) for s in "ABCDE")
    c.value("dual 现货均价 5 场景低于两条规则臂", float(all_lower), 1.0,
            "全部低于两条规则臂", 0)
    e_q = [round(spot_price("E", a)[1]) for a in (base, strong, "sac_rule", "dual_sac")]
    c.text("5.7 E 四臂现货电量", " / ".join(str(q) for q in e_q) + " MWh")
    for arm, token in [("dual_sac", "0.615–0.808"), ("rolling_rule", "0.018–0.347"),
                       ("sac_rule", "0.040–0.290")]:
        v = [float(np.mean([r["mean_intensity_abs"] for r in by_key[(scn, arm)]]))
             for scn in scenarios]
        lo, hi = rng(v)
        c.value(f"5.7 {arm} 强度下界", lo, float(token.split("–")[0]), token, 5e-4)
        c.value(f"5.7 {arm} 强度上界", hi, float(token.split("–")[1]), token, 5e-4)

    # 5.4-3 的分层占比来自 JSON (5.6 的 92%~99%/94.7% 来自 measure_reward_scale.py, 不同采样)
    share = [layer_share(by_key, scn, arm)[0] for scn in scenarios for arm in arms]
    lo, hi = rng(share)
    if args.verbose:
        print(f"  表 5.4-3 口径的中长期占比: {lo:.1f}% ~ {hi:.1f}%, "
              f"均值 {float(np.mean(share)):.1f}%")
    c.text("5.4-3 按世界均值口径标注", "按世界均值")
    return c.finish()


if __name__ == "__main__":
    raise SystemExit(main())
