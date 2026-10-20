"""
四臂基准结果可视化 (报告第 5 章图表入口)

用法:
    python experiments/plot_four_arm_results.py --in results/four_arm_benchmark.json
只画结果里存在的臂; 会计验收未通过的回合已在基准侧被排除, 本脚本不再重复判定。
输出 PNG 到 results/ (不入库)。
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ARM_LABEL = {
    "fixed_rule": "固定比例+保障调度",
    "rolling_rule": "滚动规则+预测调度",
    "sac_rule": "上层SAC+规则下层",
    "dual_sac": "双SAC",
}
MONTHS = list(range(1, 13))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_runs(data):
    """(scenario, arm) -> 该组合下所有种子的 run 列表。"""
    out = {}
    for r in data["runs"]:
        out.setdefault((r["scenario"], r["arm"]), []).append(r)
    scenarios = sorted({r["scenario"] for r in data["runs"]})
    arms = [a for a in data["meta"]["arms"] if any(r["arm"] == a for r in data["runs"])]
    return out, scenarios, arms


def world_cost_table(by_key, scen, arm):
    """按世界聚合多种子: world -> 该臂在此世界的平均成本。"""
    acc = {}
    for r in by_key.get((scen, arm), []):
        for w in r["worlds"]:
            acc.setdefault(w["world"], []).append(w["cost"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _paired_diffs(by_key, scen, arm, ref):
    """同种子同世界的配对改进 % 列表 (正 = arm 比 ref 省)。"""
    diffs = []
    for r in by_key.get((scen, ref), []):
        c = next((x for x in by_key.get((scen, arm), []) if x["seed"] == r["seed"]), None)
        if c is None:
            continue
        cand = {w["world"]: w["cost"] for w in c["worlds"]}
        pair = [(w["cost"], cand[w["world"]]) for w in r["worlds"] if w["world"] in cand]
        if not pair:
            continue
        bc = np.mean([p[0] for p in pair])
        cc = np.mean([p[1] for p in pair])
        diffs.append((bc - cc) / bc * 100)
    return diffs


def _bar_group(ax, x, by_key, scenarios, arms, ref, offset, width, hatch=None):
    for i, arm in enumerate(arms):
        means, errs = [], []
        for scen in scenarios:
            d = _paired_diffs(by_key, scen, arm, ref)
            means.append(np.mean(d) if d else 0.0)
            errs.append(np.std(d) / max(np.sqrt(len(d)), 1) if d else 0.0)
        ax.bar(x + offset + i * width, means, width * 0.9, yerr=errs, capsize=3,
               hatch=hatch, label=ARM_LABEL.get(arm, arm))


def plot_improvement(by_key, scenarios, arms, baseline, outdir):
    others = [a for a in arms if a != baseline]
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(others), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    _bar_group(ax, x, by_key, scenarios, others, baseline, -0.4 + width / 2, width)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s[:1] for s in scenarios])
    ax.set_xlabel("价格场景")
    ax.set_ylabel(f"配对改进 % (相对 {ARM_LABEL.get(baseline, baseline)})")
    ax.set_title("四臂配对世界基准: 同世界同设备任务下的成本改进")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "four_arm_improvement.png"), dpi=150)
    plt.close(fig)


def plot_vs_strong_rule(by_key, scenarios, arms, strong, outdir):
    """学习臂相对最强规则臂的增量 —— 赢弱基线不算证据, 这张图才是门槛。"""
    rule_arms = {"fixed_rule", "rolling_rule", strong}
    learners = [a for a in arms if a not in rule_arms]
    if not learners:
        return False
    x = np.arange(len(scenarios))
    width = 0.8 / len(learners)
    fig, ax = plt.subplots(figsize=(9, 5))
    _bar_group(ax, x, by_key, scenarios, learners, strong, -0.4 + width / 2, width)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s[:1] for s in scenarios])
    ax.set_xlabel("价格场景")
    ax.set_ylabel(f"配对改进 % (相对 {ARM_LABEL[strong]})")
    ax.set_title("学习臂 vs 非学习规则臂: 只有本图为正才说明学习带来增量")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "four_arm_vs_strong_rule.png"), dpi=150)
    plt.close(fig)
    return True


def plot_paired_worlds(by_key, scenarios, arms, baseline, outdir):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.4 * len(scenarios), 4.2),
                             squeeze=False)
    others = [a for a in arms if a != baseline]
    for j, scen in enumerate(scenarios):
        ax = axes[0][j]
        base = world_cost_table(by_key, scen, baseline)
        if not base:
            ax.set_visible(False)
            continue
        wx = np.array([base[w] for w in sorted(base)]) / 1e4
        all_pts = list(wx)
        for arm in others:
            cand = world_cost_table(by_key, scen, arm)
            if not cand:
                continue
            wy = np.array([cand.get(w, np.nan) for w in sorted(base)]) / 1e4
            ax.scatter(wx, wy, s=26, alpha=0.85, label=ARM_LABEL.get(arm, arm))
            all_pts += list(wy[~np.isnan(wy)])
        lo, hi = min(all_pts) * 0.995, max(all_pts) * 1.005
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("基线成本 (万元)")
        ax.set_title(scen[:1])
        if j == 0:
            ax.set_ylabel("对照臂成本 (万元)")
            ax.legend(fontsize=7)
    fig.suptitle("逐世界配对 (点落在对角线下方 = 同一世界更省)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "four_arm_paired_worlds.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(by_key, scenarios, arms, outdir):
    layers = [("annual", "年度"), ("monthly", "月度"),
              ("intraday", "月内"), ("spot", "现货")]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    fig, axes = plt.subplots(1, max(len(arms), 1),
                             figsize=(4.0 * max(len(arms), 1), 4.2), squeeze=False)
    for i, arm in enumerate(arms):
        ax = axes[0][i]
        x = np.arange(len(scenarios))
        bottom = np.zeros(len(scenarios))
        for (key, cname), color in zip(layers, colors):
            vals = []
            for scen in scenarios:
                bd = [w["cost_breakdown"][key]
                      for r in by_key.get((scen, arm), []) for w in r["worlds"]]
                vals.append(np.mean(bd) / 1e4 if bd else 0.0)
            ax.bar(x, vals, 0.62, bottom=bottom, color=color, label=cname)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([s[:1] for s in scenarios])
        ax.set_title(ARM_LABEL.get(arm, arm), fontsize=9)
        if i == 0:
            ax.set_ylabel("年均成本 (万元)")
            ax.set_xlabel("价格场景")
            ax.legend(fontsize=7)
    fig.suptitle("成本的分层构成: 中长期三层 vs 现货层")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "four_arm_cost_breakdown.png"), dpi=150)
    plt.close(fig)


def plot_gamma_profile(by_key, scenarios, arms, outdir):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.1 * len(scenarios), 4.2),
                             squeeze=False)
    for j, scen in enumerate(scenarios):
        ax = axes[0][j]
        for arm in arms:
            runs = by_key.get((scen, arm), [])
            series = [g for r in runs for g in r["worlds"] if g.get("gammas")]
            if not series:
                continue
            gm = np.mean(np.array([g["gammas"] for g in series]), axis=0)
            ax.plot(MONTHS[:len(gm)], gm, marker="o", ms=3, label=ARM_LABEL.get(arm, arm))
        ax.set_xlabel("交割月")
        ax.set_ylim(0.5, 1.02)
        ax.set_title(scen[:1])
        if j == 0:
            ax.set_ylabel("月内累计覆盖率 γ_m")
            ax.legend(fontsize=7)
    fig.suptitle("逐交割月覆盖率 (跨月下降合法: 各月独立签约, 不退款)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "four_arm_gamma_profile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="四臂基准结果可视化")
    parser.add_argument("--in", dest="infile", type=str,
                        default="results/four_arm_benchmark.json")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--strong", type=str, default="rolling_rule",
                        help="最强非学习臂, 用于'学习是否带来增量'一图; none=不出该图")
    args = parser.parse_args()

    data = load(args.infile)
    baseline = data["meta"].get("baseline_arm", "fixed_rule")
    by_key, scenarios, arms = index_runs(data)
    strong = args.strong if args.strong != "none" and args.strong in arms else None
    os.makedirs(args.outdir, exist_ok=True)
    print(f"场景 {len(scenarios)} 个, 臂 {arms}, 基线 {baseline}, 强参照 {strong}")
    plot_improvement(by_key, scenarios, arms, baseline, args.outdir)
    has_vs_strong = bool(strong) and plot_vs_strong_rule(
        by_key, scenarios, arms, strong, args.outdir)
    plot_paired_worlds(by_key, scenarios, arms, baseline, args.outdir)
    plot_breakdown(by_key, scenarios, arms, args.outdir)
    plot_gamma_profile(by_key, scenarios, arms, args.outdir)
    names = ["four_arm_improvement"] + (["four_arm_vs_strong_rule"] if has_vs_strong else []) + \
        ["four_arm_paired_worlds", "four_arm_cost_breakdown", "four_arm_gamma_profile"]
    print("输出:", ", ".join(f"{n}.png" for n in names))
