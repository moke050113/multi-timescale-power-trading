"""
生成实验可视化图表
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_results():
    results = {}
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, f), encoding="utf-8") as fp:
                r = json.load(fp)
            results[r["scenario"]] = r
    return results


def plot_training_curves(results):
    """图1: 各场景训练成本曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        costs = r["episode_costs"]
        window = 50
        smoothed = np.convolve(costs, np.ones(window)/window, mode="valid")

        ax.plot(range(len(costs)), costs, alpha=0.12, color=colors[idx])
        ax.plot(range(window-1, len(costs)), smoothed, color=colors[idx], linewidth=2, label="DRL (滑动均值)")
        ax.axhline(y=r["fixed_baseline_cost"], color="red", linestyle="--", linewidth=1.5, label="固定比例基准")

        if r["eval_episodes"]:
            ax.plot(r["eval_episodes"], r["eval_costs"], "o", color="black", markersize=4, label="评估点")

        short_name = name.split("_")[0]
        ax.set_title(f"{short_name} (改进 {r['improvement_pct']:+.1f}%)", fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("总购电成本 (元)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    fig.suptitle("多场景DRL训练成本曲线", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("training_curves.png saved")


def plot_strategy_comparison(results):
    """图2: DRL vs 固定比例 成本对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    scenarios = list(results.keys())
    short_names = [s.split("_")[0] for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.35

    fixed_costs = [results[s]["fixed_baseline_cost"] for s in scenarios]
    drl_costs = [results[s]["final_drl_cost"] for s in scenarios]

    ax.bar(x - width/2, fixed_costs, width, label="固定比例策略", color="#FF7043", alpha=0.85)
    ax.bar(x + width/2, drl_costs, width, label="DRL策略", color="#42A5F5", alpha=0.85)

    ax.set_xlabel("价格场景")
    ax.set_ylabel("总购电成本 (元)")
    ax.set_title("DRL策略 vs 固定比例策略 — 成本对比")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for i, (fc, dc) in enumerate(zip(fixed_costs, drl_costs)):
        imp = (fc - dc) / fc * 100
        color = "green" if imp > 0 else "red"
        ax.text(i + width/2, dc + max(fixed_costs)*0.01, f"{imp:+.1f}%",
                ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "strategy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("strategy_comparison.png saved")


def plot_cost_breakdown(results):
    """图3: 固定比例策略成本结构分解"""
    fig, axes = plt.subplots(1, len(results), figsize=(16, 5), sharey=True)
    labels = ["年度", "月度", "月内", "现货"]
    bar_colors = ["#1565C0", "#43A047", "#FB8C00", "#E53935"]

    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        bd = r["fixed_breakdown"]
        values = [bd["annual"], bd["monthly"], bd["intraday"], bd["spot"]]
        ax.barh(labels, values, color=bar_colors, alpha=0.85)
        ax.set_title(name.split("_")[0], fontsize=11)
        ax.set_xlabel("成本 (元)")
        for i, v in enumerate(values):
            ax.text(v + max(values)*0.02, i, f"{v:.0f}", va="center", fontsize=8)

    fig.suptitle("固定比例策略 — 各市场成本结构", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cost_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("cost_breakdown.png saved")


def plot_improvement_summary(results):
    """图4: 改进效果汇总"""
    fig, ax = plt.subplots(figsize=(10, 6))
    scenarios = list(results.keys())
    short_names = [s.split("_")[0] for s in scenarios]
    improvements = [results[s]["improvement_pct"] for s in scenarios]

    bar_colors = ["#4CAF50" if imp > 0 else "#F44336" for imp in improvements]
    bars = ax.bar(short_names, improvements, color=bar_colors, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("成本改进百分比 (%)")
    ax.set_title("DRL相对固定比例策略的成本改进\n(正值=DRL更优, 负值=固定比例更优)")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{imp:+.1f}%", ha="center", va="bottom" if imp >= 0 else "top",
                fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "improvement_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("improvement_summary.png saved")


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} scenarios")
    plot_training_curves(results)
    plot_strategy_comparison(results)
    plot_cost_breakdown(results)
    plot_improvement_summary(results)
    print("All plots generated!")
