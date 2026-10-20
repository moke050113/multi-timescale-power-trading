"""
B 方案 (双 DRL 智能体) 结果可视化 + 与分层架构对比

读取:
    results/dual_agent_v2_summary.json      — 双智能体
    results/hierarchical_comparison.json    — 分层架构 (若存在)
输出:
    results/dual_v2_three_way_comparison.png — 三方成本对比 + 改进标注
    results/dual_v2_training_phases.png      — 三阶段训练成本曲线
    results/dual_v2_reward_scale.png         — 两个智能体奖励量级演化
"""
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SCEN_NAME = {
    "A": "A 年度低价后期上涨",
    "B": "B 年度高价后期下降",
    "C": "C 各阶段价格接近",
    "D": "D 现货高波动",
    "E": "E 月度低价现货极端",
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def three_way_comparison(dual, hier, out="results/dual_v2_three_way_comparison.png"):
    tags = list(dual["results"].keys())
    fixed = [dual["results"][t]["fixed_baseline"] for t in tags]
    dual_c = [dual["results"][t]["final_cost"] for t in tags]
    if hier and "scenarios" in hier:
        # hierarchical 用 scenario key "A"/"B"...
        hier_c = []
        for t in tags:
            match = [k for k in hier["scenarios"] if k.startswith(t)]
            hier_c.append(hier["scenarios"][match[0]]["hierarchical"]["cost_mean"] if match else np.nan)
    else:
        hier_c = [np.nan] * len(tags)

    x = np.arange(len(tags))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11.5, 6))
    has_hier = not np.any(np.isnan(hier_c))
    bars = []
    if has_hier:
        bars.append(ax.bar(x - w, fixed, w, label="固定比例", color="#95A5A6"))
        bars.append(ax.bar(x, hier_c, w, label="分层架构 (1 SAC+规则)", color="#F39C12"))
        bars.append(ax.bar(x + w, dual_c, w, label="双 DRL 智能体 (B方案)", color="#2E86DE"))
    else:
        bars.append(ax.bar(x - w/2, fixed, w, label="固定比例", color="#95A5A6"))
        bars.append(ax.bar(x + w/2, dual_c, w, label="双 DRL 智能体 (B方案)", color="#2E86DE"))

    for t, d in zip(tags, dual["results"].values()):
        i = tags.index(t)
        ax.text(i, d["fixed_baseline"] * 1.01, f"{d['improvement_vs_fixed_pct']:+.1f}%",
                ha="center", fontsize=9, fontweight="bold", color="#1B4F72")

    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_NAME[t] for t in tags], fontsize=9)
    ax.set_ylabel("年总成本 (元)")
    dual_avg = np.mean([d["improvement_vs_fixed_pct"] for d in dual["results"].values()])
    ax.set_title(f"三方策略对比 | 双 DRL 平均改进 {dual_avg:+.1f}% (柔性负荷场景: {dual['load_scenario']})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(fixed) * 1.15)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"生成: {out}")


def phase_curves(dual, out="results/dual_v2_training_phases.png"):
    """三阶段训练成本曲线"""
    results = dual["results"]
    tags = list(results.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 4), sharey=True)
    if len(tags) == 1:
        axes = [axes]
    color_map = {"P1-midlong": "#E67E22", "P2-spot": "#27AE60", "P3-joint": "#8E44AD"}

    for ax, tag in zip(axes, tags):
        hist = results[tag].get("history_sample")
        if not hist:
            ax.set_title(f"{tag} (无曲线)")
            continue
        eps = [h["episode"] for h in hist]
        costs = [h["cost"] for h in hist]
        phases = [h["phase"] for h in hist]
        colors = [color_map[p] for p in phases]
        ax.scatter(eps, costs, c=colors, s=12, alpha=0.7)
        ax.axhline(results[tag]["fixed_baseline"], color="#95A5A6", ls=":", lw=1, label="固定基准")
        ax.set_title(SCEN_NAME.get(tag, tag), fontsize=10)
        ax.set_xlabel("Episode")
        ax.grid(linestyle="--", alpha=0.4)

    axes[0].set_ylabel("成本 (元)")
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["P1-midlong"], markersize=8, label="P1 训练中长期"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["P2-spot"], markersize=8, label="P2 训练现货"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["P3-joint"], markersize=8, label="P3 联合"),
        plt.Line2D([0], [0], color="#95A5A6", ls=":", label="固定基准"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=7)
    plt.suptitle("三阶段交替训练成本演化 (B 方案双 DRL 智能体)")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"生成: {out}")


def reward_scale(dual, out="results/dual_v2_reward_scale.png"):
    """两个智能体的奖励信号量级演化 (证明信用分配已解决)"""
    results = dual["results"]
    if "A" not in results:
        return
    hist = results["A"].get("history_sample")
    if not hist:
        return
    eps = [h["episode"] for h in hist]
    r_mid = np.abs([h["reward_midlong"] for h in hist])
    r_spot = np.abs([h["reward_spot"] for h in hist])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(eps, r_mid, color="#E67E22", label="中长期智能体 |R|")
    ax1.plot(eps, r_spot, color="#27AE60", label="现货智能体 |R|")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("奖励绝对值 (episode 累计)")
    ax1.set_title("两个智能体奖励信号量级 (场景 A)")
    ax1.legend(); ax1.grid(linestyle="--", alpha=0.4)

    # 比值
    ratio = r_spot / np.maximum(r_mid, 1e-6)
    ax2.plot(eps, ratio, color="#8E44AD")
    ax2.axhline(1.0, color="k", ls=":", lw=1)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("|R_spot| / |R_mid|")
    ax2.set_title("奖励量级比 (旧共享方案≈0.02, 目标=同量级)")
    ax2.grid(linestyle="--", alpha=0.4)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"生成: {out}")


if __name__ == "__main__":
    dual = load("results/dual_agent_v2_summary.json")
    hier = load("results/hierarchical_comparison.json")
    hist_a = load("results/dual_v2_history_A.json")  # 带 history_sample
    if not dual:
        raise SystemExit("缺少 results/dual_agent_v2_summary.json")
    three_way_comparison(dual, hier)
    # 阶段曲线 + 奖励量级图需要 history, 从单独跑的场景文件读取
    if hist_a:
        phase_curves(hist_a)
        reward_scale(hist_a)
    print("全部图表生成完毕")
