"""报告正文数字自检: 把 5.3~5.7 与 5.9 引用的数字从 results/ 重算一遍。

存在的理由: 本文的性能结论全部落在正文叙述里 (配对改进、覆盖率剖面、成交均价、消融 Δ)。
人工填数一旦与 JSON 脱钩就成了虚假结论, 所以每条主张都同时检查两件事:
  (1) 报告 md 里确实写着这个数;
  (2) 从 JSON 重算出来的数与它相符 (容差按正文给出的小数位取)。
另外表 5.4-1 ~ 5.4-4 必须与 `report_tables.py` 当场生成的表格逐字符相同,
表 5.9-1 必须与 `summarize_ablation.py` 当场重算的配对结果逐格相符。

用法:
    python experiments/verify_report_numbers.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.plot_four_arm_results import index_runs, load          # noqa: E402
from experiments.report_tables import layer_share, render, seed_pairwise  # noqa: E402
from experiments.summarize_ablation import load_base, summarize          # noqa: E402
from src.utils.config import INDUSTRIAL_LOAD_PARAMS, SIM_PARAMS, TOTAL_DEMAND  # noqa: E402
from src.utils.reproducibility import source_lock                        # noqa: E402

LEARNERS = ("sac_rule", "dual_sac")
THRESHOLD = 0.4            # 5.1 节的运行间噪声门槛 (%), 小于它的差值不得用作结论
ABL_ROWS = {"未来暴露成本塑形": "no_exposure_shaping",
            "现货奖励放大": "no_spot_reward_scale",
            "三阶段交替训练": "no_alternating"}
# 每个变体应当"只关掉自己那一项", 其余两项仍按完整方法开启
ABL_EXPECT = {"no_exposure_shaping": {"exposure_shaping": False, "scale_spot_reward": True,
                                      "alternating": True},
              "no_spot_reward_scale": {"exposure_shaping": True, "scale_spot_reward": False,
                                       "alternating": True},
              "no_alternating": {"exposure_shaping": True, "scale_spot_reward": True,
                                 "alternating": False}}


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

    def plain(self, label: str, ok: bool, detail: str = "") -> None:
        """只验数据自身的一致性, 不要求正文出现某个字符串。"""
        self.n += 1
        if not ok:
            self.bad += 1
            print(f"  [XX] {label}{': ' + detail if detail else ''}")

    def finish(self) -> int:
        print(f"检查 {self.n} 项主张, 不符 {self.bad} 项")
        return 1 if self.bad else 0


def rng(values) -> tuple:
    a = np.asarray(values, dtype=float)
    return float(a.min()), float(a.max())


def penalty_units(worlds) -> dict:
    """
    只凭 JSON 复算惩罚的单位链 (审计 N01/N07):
        代表日转移电量 × (days_per_month × demand_scale) = 月口径转移电量 × 单价 = 惩罚
    任何一环按"单个代表日"计, 这里就会露馅 —— 不必重跑训练即可否定作废口径。
    """
    K = int(SIM_PARAMS["days_per_month"])
    tc = float(INDUSTRIAL_LOAD_PARAMS["transfer_cost"])
    need = ("demand_scale", "transferred_mwh_day", "transferred_mwh_total")
    absent = sorted({k for w in worlds for k in need if k not in w})
    if absent:
        # 字段缺失 = 这份 JSON 出自改口径之前的驱动; 让每条单位检查都判不过, 而不是抛 KeyError
        return {"absent": absent, "scales": set(), "month_factor": 0.0,
                "worst_scale": float("inf"), "worst_unit": float("inf"), "n_flex": 0,
                "ratio": float("inf"), "max_penalty": 0.0, "max_share": 0.0}
    scales = {round(float(w["demand_scale"]), 12) for w in worlds}
    ds = float(next(iter(scales))) if len(scales) == 1 else float("nan")
    pen = np.array([w["flexible_penalty"] for w in worlds], dtype=float)
    mwh = np.array([w["transferred_mwh_total"] for w in worlds], dtype=float)
    day = np.array([w["transferred_mwh_day"] for w in worlds], dtype=float)
    cost = np.array([w["cost"] for w in worlds], dtype=float)
    flex = mwh > 0
    return {
        "absent": [], "scales": scales, "month_factor": K * ds,
        "worst_scale": float(np.max(np.abs(mwh - day * K * ds))),
        "worst_unit": float(np.max(np.abs(pen - tc * mwh))),
        "n_flex": int(flex.sum()),
        "ratio": float(np.median(pen[flex] / (tc * day[flex]))) if flex.any() else 0.0,
        "max_penalty": float(pen.max()), "max_share": float((100.0 * pen / cost).max()),
    }


def nums(cell: str) -> list:
    """取一个 md 单元格里的带符号数字 (排版减号先统一, 千位逗号去掉)。"""
    s = cell.replace("\u2212", "-").replace(",", "")
    return [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", s)]


# 正文引用的代码行号: 改代码就会让"某某机制见 file:NN-MM"变成假证据 (审计 N07)。
# 除了检查行号还在文件范围内, 还要求那段里出现关键字 —— 只查范围挡不住"行号有效但指错了地方"。
CITATION_KEYS = {
    "src/market/environment.py:354-359": "compute_cvar",
    "src/market/environment.py:441-443": "pub_monthly",
    "src/utils/config.py:15": "annual",
    "experiments/run_four_arm_benchmark.py:378-383": "--no-alternating",
}
CITE_RE = re.compile(r"(src/[a-z_/]+\.py|experiments/[a-z_/]+\.py):(\d+)(?:-(\d+))?")


def check_doc_only(c: "Checker", md: str) -> None:
    """与 JSON 无关、只靠当前仓库就能判真伪的正文主张。"""
    for path, start, end in CITE_RE.findall(md):
        start, end = int(start), int(end or start)
        c.n += 1
        if not os.path.exists(path):
            c.bad += 1
            print(f"  [XX] 引用 {path}:{start}-{end}: 文件不存在")
            continue
        lines = open(path, encoding="utf-8").read().splitlines()
        key = CITATION_KEYS.get(f"{path}:{start}-{end}", "")
        body = "\n".join(lines[start - 1:end])
        if not (0 < start <= end <= len(lines)) or not body.strip():
            c.bad += 1
            print(f"  [XX] 引用 {path}:{start}-{end}: 行号越界或内容为空 (文件共 {len(lines)} 行)")
        elif key and key not in body:
            c.bad += 1
            print(f"  [XX] 引用 {path}:{start}-{end}: 那段里找不到 {key!r}, 引用已随代码漂移")
    c.plain("每条带行号的引用都被上面逐条检查过", len(CITE_RE.findall(md)) >= len(CITATION_KEYS),
            f"正文只找到 {len(CITE_RE.findall(md))} 处行号引用")

    n_cases = sum(len(re.findall(r"^def test_", open(p, encoding="utf-8").read(), re.M))
                  for p in sorted(glob.glob("experiments/test_*.py")))
    c.value("5.2 验收用例总数 = 各 test 文件里的 def test_ 之和", float(n_cases), n_cases,
            f"{n_cases} 个用例", 0)

    left = sorted(set(re.findall(r"<[A-Z0-9_]{2,}>", md)))
    c.plain("正文没有未填的占位符", not left, f"还剩 {left}")


def md_row(md: str, part: str, arm: str) -> list:
    """表 5.9-1 里 (剔除部件, 臂) 那一行的单元格; 没有该行返回空列表。"""
    pre = f"| {part} | `{arm}` |"
    for line in md.replace("\u2212", "-").splitlines():
        if line.startswith(pre):
            return [x.strip() for x in line.strip().strip("|").split("|")]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/four_arm_benchmark.json")
    ap.add_argument("--md", default="report/课程设计报告.md")
    ap.add_argument("--parts", default="results/parts/ablation_*.json")
    ap.add_argument("--summary", default="results/ablation_summary.json")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    md = open(args.md, encoding="utf-8").read()
    data = load(args.json)
    by_key, scenarios, arms = index_runs(data)
    base = data["meta"]["baseline_arm"]
    strong = "rolling_rule"
    S = {s[:1]: s for s in scenarios}
    c = Checker(md)

    # ---------- 0. 源锁: 发布出来的数字必须能追溯到产生它的那一份代码 (审计 N05/N07) ----------
    cur_lock = source_lock()

    def check_lock(meta, who: str) -> dict:
        lk = meta.get("source_lock")
        c.n += 1
        if lk is None:
            c.bad += 1
            print(f"  [XX] {who}: meta 里没有 source_lock —— "
                  f"无法确认数字由哪一份代码产生, 需按当前代码重跑")
            return {}
        bad = []
        for f, want in cur_lock["files"].items():
            got = lk.get("files", {}).get(f)
            if got != want:
                bad.append(f"{f}: {got} != {want}")
        for k in ("numpy", "torch"):
            if lk.get(k) != cur_lock[k]:
                bad.append(f"{k}: {lk.get(k)} != {cur_lock[k]}")
        c.plain(f"{who} 的源锁与当前代码树一致", not bad, "; ".join(bad[:3]))
        return lk

    json_lock = check_lock(data["meta"], args.json)

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

    # ---------- 2b. 数据鉴真: 惩罚单位与源锁 (审计 N01/N05/N07) ----------
    u = penalty_units(worlds)
    c.plain("落盘记录带惩罚单位链所需字段", not u["absent"],
            f"缺 {u['absent']} —— 这份 JSON 出自改口径之前的驱动, 必须按当前代码重跑")
    c.plain("全部世界共用同一个 demand_scale", len(u["scales"]) == 1, str(sorted(u["scales"])))
    c.plain("月电量 = 代表日电量 × days_per_month × demand_scale", u["worst_scale"] < 1e-6,
            f"最大残差 {u['worst_scale']:.3g} MWh")
    c.plain("惩罚 = 转移成本单价 × 月口径转移电量", u["worst_unit"] < 1e-6,
            f"最大残差 {u['worst_unit']:.3g} 元")
    c.plain("数据里确有发生转移的世界 (否则单位检查是空转)", u["n_flex"] > 0,
            f"{u['n_flex']} 个世界")
    c.plain("惩罚按整月电量而非单个代表日计", abs(u["ratio"] - u["month_factor"]) < 1e-6,
            f"实测倍数 {u['ratio']:.4f}, 应为 {u['month_factor']:.4f}")
    mx_pen, mx_share = u["max_penalty"], u["max_share"]
    c.value("全年柔性转移惩罚上限", mx_pen, round(mx_pen), f"{mx_pen:,.0f} 元", 0.5)
    c.value("柔性惩罚占年成本上限", mx_share, round(mx_share, 2), f"不超过 {mx_share:.2f}%", 0.005)
    fx = [w for r in data["runs"] if r["arm"] == base for w in r["worlds"]]
    net = np.array([w["energy_breakdown"]["total"] for w in worlds], dtype=float) - TOTAL_DEMAND
    c.value("保障调度臂年成交电量对 Q 的最大残差", float(np.max(np.abs(
        np.array([w["energy_breakdown"]["total"] for w in fx], dtype=float) - TOTAL_DEMAND))),
        0.0, "年成交电量都闭合于 Q", 1e-6)
    # 柔性动作只能通过 EV 往返损耗抬高网侧年电量, 绝不可能低于 Q (工业任务电量守恒)
    c.plain("全部世界的年成交电量不低于 Q", float(net.min()) > -1e-9, f"最小 {net.min():+.4f} MWh")
    ex = float(net.max()) / TOTAL_DEMAND * 100.0
    c.plain("柔性挪动的年电量在 1% 以内 (只可能是 EV 往返损耗)", 0 <= ex < 1.0,
            f"最多 +{ex:.3f}%")
    mismatch = [i for i, w in enumerate(worlds)
                if bool(w["device_task_met"]) != bool(w["checks"]["device_task_met"])]
    c.plain("逐世界设备结论与 checks 里的同一字段不矛盾", not mismatch,
            f"{len(mismatch)} 个世界两处记录不一致")
    ew = int(data["meta"]["eval_worlds"])
    bad_ids = [f"{r['scenario'][:1]}/{r['arm']}/seed{r['seed']}" for r in data["runs"]
               if len({w["world"] for w in r["worlds"]}) != len(r["worlds"])
               or not {w["world"] for w in r["worlds"]} <= set(range(ew))]
    c.plain(f"每个组合的世界编号互不重复且都在 0~{ew - 1}", not bad_ids, str(bad_ids[:4]))

    # 动作可达集 (审计 N02): 正文声称"四臂动作都不越界", 数据里必须有这个计数且为 0
    have_oob = [wch for wch in chk if "action_oob" in wch]
    c.plain("逐世界验收里带 action_oob 计数", len(have_oob) == len(chk),
            f"{len(have_oob)}/{len(chk)} 个世界有该字段 —— 缺字段的 JSON 出自旧驱动, 需重跑")
    n_oob = sum(int(wch["action_oob"]) for wch in have_oob)
    c.plain("全部世界的越界动作数为 0", len(have_oob) == len(chk) and n_oob == 0,
            f"{n_oob} 次越界 —— 规则臂或 SAC 输出超出了 ACTION_BOUNDS 声明的可达集")
    c.text("4.1 动作可达集写法", "checks.action_oob")
    c.text("3.2 惩罚月度尺度写法", "days\\_per\\_month")

    # 训练侧留痕 (审计 N05/N09): 每个 SAC 臂必须记下逐角色优化器步数与阶段日程
    sac_runs = [r for r in data["runs"] if r.get("trained")]
    gs = [r.get("grad_steps") or {} for r in sac_runs]
    c.plain("每个已训练组合都带 grad_steps", len(sac_runs) == 30 and all(g for g in gs),
            f"{len(sac_runs)} 个已训练组合, 其中有步数的 {sum(1 for g in gs if g)} 个")
    c.plain("中长期网络确实被更新过", bool(gs) and all(int(g.get("midlong", 0)) > 0 for g in gs),
            "存在 midlong=0 的训练组合")
    c.plain("分层臂不更新柔性网络",
            all(int(r.get("grad_steps", {}).get("spot", 0)) == 0
                for r in sac_runs if r["arm"] == "sac_rule"))
    ph = {r["arm"]: [q["phase"] for q in r.get("phase_history") or []] for r in sac_runs}
    c.plain("dual_sac 走三阶段、sac_rule 只训中长期",
            bool(ph) and all(v == ["P1-midlong", "P2-spot", "P3-joint"] for k, v in ph.items()
                             if k == "dual_sac")
            and all(v == ["P-midlong"] for k, v in ph.items() if k == "sac_rule"),
            str({k: v for k, v in ph.items()}))
    ab = data["meta"].get("ablation")
    c.plain("主基准 meta.ablation 三项全开",
            ab == {"exposure_shaping": True, "scale_spot_reward": True, "alternating": True},
            f"实测 {ab}")

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

    # ---------- 8. 5.9 消融 (数据源: results/parts/ablation_*.json + 主基准场景 A 分片) ----------
    base_meta, base_a = load_base(args.json, S["A"])
    parts = sorted(glob.glob(args.parts))
    if len(parts) != len(ABL_ROWS):
        print(f"  [XX] 消融分片 {len(parts)} 个, 报告写了 {len(ABL_ROWS)} 个部件: {parts}")
        return 1
    re_abl, worlds_of, n_worlds, n_fail = {}, {}, 0, 0
    abl_data = {}
    for p in parts:
        v = os.path.basename(p)[len("ablation_"):-len(".json")]
        d = load(p)
        abl_data[v] = d
        part_lock = check_lock(d["meta"], p)
        c.n += 1
        if json_lock.get("files") and part_lock.get("files") != json_lock["files"]:
            c.bad += 1
            print(f"  [XX] {p}: 消融分片与主基准不是同一份代码, "
                  f"完整方法与变体的配对差无法解释")
        c.n += 1
        if d["meta"].get("ablation") != ABL_EXPECT[v]:
            c.bad += 1
            print(f'  [XX] {v}: meta.ablation 不是 "只关这一项" {d["meta"].get("ablation")}')
        re_abl[v] = {a["arm"]: a for a in summarize(v, p, base_meta, base_a)["arms"]}
        pu = penalty_units([w for r in d["runs"] for w in r["worlds"]])
        c.plain(f"消融 {v}: 惩罚单位链也按月电量 (与完整方法同口径才可比)",
                pu["worst_scale"] < 1e-6 and pu["worst_unit"] < 1e-6
                and abs(pu["ratio"] - pu["month_factor"]) < 1e-6 and pu["scales"] == u["scales"],
                f"残差 {pu['worst_scale']:.3g}/{pu['worst_unit']:.3g}, "
                f"倍数 {pu['ratio']:.4f} vs {pu['month_factor']:.4f}, "
                f"demand_scale {sorted(pu['scales'])} vs {sorted(u['scales'])}")
        worlds_of[v] = sum(len(r["worlds"]) for r in d["runs"])
        n_worlds += worlds_of[v]
        n_fail += len(d["failures"])
        pc = [w["checks"] for r in d["runs"] for w in r["worlds"] if "action_oob" in w["checks"]]
        c.plain(f"消融 {v}: 变体逐世界也带越界动作计数", len(pc) == worlds_of[v],
                f"{len(pc)}/{worlds_of[v]} 个世界有该字段")
        c.plain(f"消融 {v}: 变体的动作同样不越界 (可达集与完整方法一致才可比)",
                len(pc) == worlds_of[v] and sum(int(x["action_oob"]) for x in pc) == 0,
                f"{sum(int(x['action_oob']) for x in pc)} 次越界")

    def seeds_of(a):
        return [s["delta_pct"] for s in sorted(a["seeds"], key=lambda z: z["seed"])]

    def worse(a):
        return tuple(int(x) for x in a["worse_worlds"].split("/"))

    # 落盘的汇总表必须与当场重算一致 (防止 parts 重跑后 summary 变陈旧)
    summ = json.load(open(args.summary, encoding="utf-8"))
    FIELDS = ("cost", "cost_ref", "delta_pct", "alpha", "alpha_ref",
              "gamma", "gamma_ref", "intensity", "intensity_ref")
    for res in summ["results"]:
        for a in res["arms"]:
            ref = re_abl[res["variant"]][a["arm"]]
            c.n += 1
            bad = [f for f in FIELDS if abs(a[f] - ref[f]) > 1e-9]
            if bad:
                c.bad += 1
                print(f"  [XX] {args.summary} 的 {res['variant']}/{a['arm']} 字段与重算不符: {bad}")

    # 表 5.9-1 逐格重算
    for part, v in ABL_ROWS.items():
        for arm in LEARNERS:
            a, cells = re_abl[v].get(arm), md_row(md, part, arm)
            if a is None or not cells:
                c.n += 1
                if (a is not None) != bool(cells):
                    c.bad += 1
                    print(f"  [XX] 表 5.9-1 {part}/{arm}: 重算与正文一方缺失 "
                          f"(重算={a is not None}, 正文行={bool(cells)})")
                continue
            c.n += 1
            if len(cells) != 9:
                c.bad += 1
                print(f"  [XX] 表 5.9-1 行列数不为 9: {part}/{arm} -> {cells}")
                continue
            wn, nn = worse(a)
            c.value(f"5.9-1 {part}/{arm} 变体成本", a["cost"], nums(cells[2])[0], cells[2], 0.5)
            c.value(f"5.9-1 {part}/{arm} 配对差", a["delta_pct"], nums(cells[3])[0], cells[3], 0.005)
            c.value(f"5.9-1 {part}/{arm} 更贵世界数", float(wn), nums(cells[4])[0], cells[4], 0)
            c.value(f"5.9-1 {part}/{arm} 配对世界总数", float(nn), nums(cells[4])[1], cells[4], 0)
            for i, (comp, wri) in enumerate(zip(seeds_of(a), nums(cells[5]))):
                c.value(f"5.9-1 {part}/{arm} 第 {i + 1} 个种子 Δ", comp, wri, cells[5], 0.005)
            for field, cell in (("alpha", cells[6]), ("gamma", cells[7]), ("intensity", cells[8])):
                c.value(f"5.9-1 {part}/{arm} {field}", a[field], nums(cell)[0], cell, 5e-4)
                c.value(f"5.9-1 {part}/{arm} {field} (完整)",
                        a[field + "_ref"], nums(cell)[1], cell, 5e-4)

    # 正文与摘要/1.3/4.4/6.1 里的消融数字: 由重算值拼出字符串, 要求原文照此写着
    sh = [re_abl["no_exposure_shaping"][a] for a in LEARNERS]
    sp = [re_abl["no_spot_reward_scale"][a] for a in LEARNERS]
    al = re_abl["no_alternating"]["dual_sac"]
    for arm in LEARNERS:
        ref_cost = float(np.mean([r["mean_cost"] for (a, _), r in base_a.items() if a == arm]))
        c.text(f"5.9 参照成本 {arm}", f"`{arm}` {ref_cost:,.0f} 元")
    c.text("5.9 评估世界总数", f"{n_worlds} 个评估世界")
    c.text("5.9 各变体世界数", "（" + "+".join(str(worlds_of[v]) for v in ABL_ROWS.values()) + "）")
    c.text("5.9 验收失败条数", f"验收失败 **{n_fail} 条**")
    c.text("5.9 冻结预算", f"{base_meta['episodes_per_sac_arm']} 回合、"
                          f"{base_meta['eval_worlds']} 个配对世界")
    c.text("5.9 冻结种子", "/".join(str(s) for s in base_meta["seeds"]))
    c.text("5.9 冻结阶段划分", "+".join(f"{p * 100:.0f}%" for p in base_meta["phase_split"]))
    c.text("5.9 塑形代价 (摘要/结论一/6.1)", f"{sh[0]['delta_pct']:.2f}% 与 {sh[1]['delta_pct']:.2f}%")
    c.text("5.9 塑形代价区间 (1.3)",
           f"贵 {min(a['delta_pct'] for a in sh):.2f}%~{max(a['delta_pct'] for a in sh):.2f}%")
    c.text("5.9 结论三 另两项 Δ",
           f"+{sp[0]['delta_pct']:.2f}%/+{sp[1]['delta_pct']:.2f}% 与 +{al['delta_pct']:.2f}%")
    c.text("4.4 另两项 Δ",
           f"+{sp[0]['delta_pct']:.2f}%/+{sp[1]['delta_pct']:.2f}%（奖励放大）与 "
           f"+{al['delta_pct']:.2f}%（交替训练）")
    for sep in (" / ", "/"):      # 5.9 结论二用宽排, 6.1 用紧排
        c.text(f"结论二 γ 掉档 (sep={sep!r})",
               f"{sh[0]['gamma_ref']:.3f}{sep}{sh[1]['gamma_ref']:.3f} 掉到 "
               f"{sh[0]['gamma']:.3f}{sep}{sh[1]['gamma']:.3f}")
    c.text("结论二 α 掉档",
           f"{sh[0]['alpha_ref']:.3f} / {sh[1]['alpha_ref']:.3f} 掉到 "
           f"{sh[0]['alpha']:.3f} / {sh[1]['alpha']:.3f}")
    c.text("结论二 强度旁证 (sac_rule)", f"{sh[0]['intensity_ref']:.3f} 升到 {sh[0]['intensity']:.3f}")
    n_seed = len(seeds_of(sh[0]))
    c.text("结论一 种子同号", f"个配对世界更贵、{n_seed}/{n_seed} 种子同号")
    c.text("结论三 符号不一致的例子",
           "`dual_sac` 关奖励放大为 " + "/".join(f"{d:+.2f}" for d in seeds_of(sp[1])))

    # README 复述了 5.9 的 Δ 与 γ 掉档: 跨文档必须与重算一致, 不允许各说一套
    if os.path.exists(args.readme):
        cr = Checker(open(args.readme, encoding="utf-8").read())
        cr.text("README 塑形 γ 掉档",
                f"{sh[0]['gamma_ref']:.3f}/{sh[1]['gamma_ref']:.3f} 掉到 "
                f"{sh[0]['gamma']:.3f}/{sh[1]['gamma']:.3f}")
        for part, v in ABL_ROWS.items():
            line = next((l for l in cr.md.splitlines() if l.startswith(f"| {part} |")), "")
            cells = [x.strip() for x in line.strip().strip("|").split("|")] if line else []
            cr.n += 1
            if len(cells) != 4:
                cr.bad += 1
                print(f"  [XX] README 消融表 {part} 行缺失或列数不为 4: {cells}")
                continue
            for arm, col in (("sac_rule", 1), ("dual_sac", 2)):
                a = re_abl[v].get(arm)
                if a is None:
                    cr.n += 1
                    if cells[col] != "—":
                        cr.bad += 1
                        print(f"  [XX] README {part}/`{arm}` 没跑, 表里却写着 {cells[col]!r}")
                    continue
                wr = nums(cells[col])
                cr.value(f"README {part}/{arm} Δ", a["delta_pct"], wr[0], cells[col], 0.005)
                cr.value(f"README {part}/{arm} 更贵世界数", float(worse(a)[0]), wr[1],
                         cells[col], 0)
        c.n += cr.n
        c.bad += cr.bad
    else:
        # README 复述了同一批数字, 它缺失时"跨文档一致"就只是一句空话 (审计 N07)
        c.plain(f"{args.readme} 存在 (跨文档一致性的前提)", False,
                "README 不在预期路径上, 无法检查跨文档复述是否脱钩")
    ok1 = all(a["delta_pct"] > THRESHOLD and all(d > 0 for d in seeds_of(a))
              and worse(a)[0] == worse(a)[1] for a in sh)
    c.value("5.9 结论一: 塑形两臂 Δ>门槛、逐种子同号、世界全更贵", float(ok1), 1.0,
            "唯一通过判读门槛的部件", 0)
    ok3 = all(a["delta_pct"] < THRESHOLD and len({d > 0 for d in seeds_of(a)}) == 2
              and worse(a)[0] < worse(a)[1] for a in sp + [al])
    c.value("5.9 结论三: 另两项 Δ<门槛且逐种子符号不一致", float(ok3), 1.0,
            "逐种子符号不一致", 0)

    # 局限 4 的三条证据: 驱动脚本无 --no-* / 开关默认取反 / 主基准 JSON 自己写着三部件全开
    sh_txt = open("experiments/run_all_scenarios.sh", encoding="utf-8").read()
    c.value("5.9 局限4: 主基准调用不含任何 --no-* 开关", float("--no-" not in sh_txt), 1.0,
            "不含任何 `--no-*` 开关", 0)
    drv = open("experiments/run_four_arm_benchmark.py", encoding="utf-8").read()
    c.value("5.9 局限4: 驱动把三项消融开关默认置为开启",
            float(len(re.findall(r"not args\.no_(?:exposure_shaping|spot_reward_scale|alternating)",
                                 drv))), 3, "三项开关默认置为开启", 0)
    c.value("5.9 局限4: 主基准 JSON 自带三部件全开的开关记录",
            float(base_meta.get("ablation") == {"exposure_shaping": True,
                                                "scale_spot_reward": True,
                                                "alternating": True}), 1.0,
            "三部件全开", 0)

    # 交替训练确实改变了训练日程: 不再靠训练日志, 直接读分片里的 phase_history 与 grad_steps
    al_part = abl_data.get("no_alternating", {})
    al_ph = [[q["phase"] for q in r.get("phase_history") or []] for r in al_part.get("runs", [])
             if r.get("trained")]
    c.plain("no_alternating 的每个 SAC 组合只有一步 [P-joint] 阶段",
            bool(al_ph) and all(x == ["P-joint"] for x in al_ph),
            f"实测日程集合 {sorted({tuple(x) for x in al_ph})}")
    full_ph = [[q["phase"] for q in r.get("phase_history") or []] for r in data["runs"]
               if r.get("trained") and r["arm"] == "dual_sac"]
    c.plain("主基准 dual_sac 走 P1/P2/P3 三阶段",
            bool(full_ph) and all(x == ["P1-midlong", "P2-spot", "P3-joint"] for x in full_ph),
            f"{len(full_ph)} 个组合, 首个 {full_ph[:1]}")
    g4 = [al.get(k) for k in ("grad_midlong", "grad_spot", "grad_midlong_ref", "grad_spot_ref")]
    fresh = all(isinstance(x, int) and x > 0 for x in g4)
    c.plain("5.9 交替变体的逐角色优化器步数可从 JSON 复算", fresh,
            f"实测 {g4} —— 全为 0/缺失说明消融分片出自旧驱动, 需按当前代码重跑")
    if fresh:
        c.text("5.9 交替变体逐角色步数",
               f"完整方法 中长期 {g4[2]:.0f} 次 / 柔性 {g4[3]:.0f} 次，"
               f"关掉交替 {g4[0]:.0f} 次 / {g4[1]:.0f} 次")
        c.plain("关掉交替后逐角色步数确实变了 (不是只改名字的零结果)",
                g4[0] != g4[2] or g4[1] != g4[3],
                f"完整 {g4[2]}/{g4[3]}, 变体 {g4[0]}/{g4[1]}")
    check_doc_only(c, md)
    if args.verbose:
        print(f"  5.9 重算 Δ: 塑形 {sh[0]['delta_pct']:.3f}%/{sh[1]['delta_pct']:.3f}%, "
              f"奖励放大 {sp[0]['delta_pct']:.3f}%/{sp[1]['delta_pct']:.3f}%, "
              f"交替 {al['delta_pct']:.3f}%; 世界 {n_worlds}, 失败 {n_fail}")
    return c.finish()


if __name__ == "__main__":
    raise SystemExit(main())
