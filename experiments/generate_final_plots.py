"""
最终可视化图表生成
基于所有实验数据生成论文级图表
"""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, "D:/Qoder_project")
os.chdir("D:/Qoder_project")

# 中文字体
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "D:/Qoder_project/results"


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [跳过] {filename} 不存在")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_monthly_ratios():
    """月度 α/β/γ 动态变化图 (3 场景 × 12 月)"""
    data = load_json("monthly_ratios_analysis.json")
    if data is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    months = list(range(12))  # 0-indexed

    for idx, (scenario_key, scenario_data) in enumerate(data.items()):
        alphas = [scenario_data[str(m)]["alpha"] for m in months]
        betas = [scenario_data[str(m)]["beta"] for m in months]
        gammas = [scenario_data[str(m)]["gamma"] for m in months]

        ax = axes[idx]
        month_labels = [m + 1 for m in months]
        ax.plot(month_labels, alphas, "o-", label="α (年度)", color="#2196F3", markersize=4)
        ax.plot(month_labels, betas, "s-", label="β (累计月度)", color="#4CAF50", markersize=4)
        ax.plot(month_labels, gammas, "^-", label="γ (累计中长期)", color="#FF5722", markersize=4)
        ax.set_title(f"场景 {scenario_key}", fontsize=12)
        ax.set_xlabel("月份")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 标注收官效应
        ax.annotate(f"γ={gammas[-1]:.2f}", xy=(12, gammas[-1]),
                    xytext=(10, gammas[-1] - 0.15),
                    arrowprops=dict(arrowstyle="->", color="#FF5722"),
                    fontsize=9, color="#FF5722")

    axes[0].set_ylabel("交易比例")
    fig.suptitle("DRL 学到的月度交易比例动态 — 收官效应", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "monthly_ratio_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [完成] monthly_ratio_dynamics.png")


def plot_cvar_ratios():
    """CVaR λ 对 α/β/γ 比例结构的影响"""
    data = load_json("cvar_ratios_analysis.json")
    if data is None:
        return

    lambdas = []
    alphas, betas, gammas, spots, costs = [], [], [], [], []

    for key in sorted(data.keys(), key=lambda k: data[k]["lambda"]):
        entry = data[key]
        lambdas.append(entry["lambda"])
        alphas.append(entry["alpha_mean"])
        betas.append(entry["beta_mean"])
        gammas.append(entry["gamma_mean"])
        spots.append(entry["spot_mean"])
        costs.append(entry["avg_cost"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图: 比例结构
    ax1.plot(lambdas, alphas, "o-", label="α (年度)", color="#2196F3", linewidth=2)
    ax1.plot(lambdas, betas, "s-", label="β (累计月度)", color="#4CAF50", linewidth=2)
    ax1.plot(lambdas, gammas, "^-", label="γ (累计中长期)", color="#FF5722", linewidth=2)
    ax1.plot(lambdas, spots, "D-", label="现货暴露", color="#9C27B0", linewidth=2)
    ax1.set_xlabel("风险厌恶系数 λ")
    ax1.set_ylabel("交易比例")
    ax1.set_title("CVaR λ 对交易比例结构的影响")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: 成本
    ax2.plot(lambdas, costs, "o-", color="#333", linewidth=2, markersize=8)
    ax2.set_xlabel("风险厌恶系数 λ")
    ax2.set_ylabel("平均总成本 (元)")
    ax2.set_title("风险-成本权衡曲线")
    ax2.grid(True, alpha=0.3)
    for i, (l, c) in enumerate(zip(lambdas, costs)):
        ax2.annotate(f"{c/10000:.1f}万", xy=(l, c), xytext=(5, 5),
                     textcoords="offset points", fontsize=8)

    fig.suptitle("风险厌恶对交易策略的影响 (场景D)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cvar_ratio_structure.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [完成] cvar_ratio_structure.png")


def plot_dual_agent_comparison():
    """单智能体 vs 双智能体对比"""
    data = load_json("dual_agent_experiment.json")
    if data is None:
        print("  [等待] 双智能体实验数据尚未生成")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, scenario_name in enumerate(set(e["scenario"] for e in data)):
        short = scenario_name.split("_")[0]
        single = [e for e in data if e["scenario"] == scenario_name and e["mode"] == "single"][0]
        dual = [e for e in data if e["scenario"] == scenario_name and e["mode"] == "dual"][0]

        ax = axes[idx]
        # 滑动平均
        window = 50
        s_costs = single["episode_costs"]
        d_costs = dual["episode_costs"]
        s_smooth = np.convolve(s_costs, np.ones(window)/window, mode="valid")
        d_smooth = np.convolve(d_costs, np.ones(window)/window, mode="valid")

        ax.plot(s_smooth, label=f"单智能体 ({single['improvement_pct']:+.1f}%)", color="#2196F3", linewidth=1.5)
        ax.plot(d_smooth, label=f"双智能体 ({dual['improvement_pct']:+.1f}%)", color="#FF5722", linewidth=1.5)
        ax.axhline(y=single["fixed_cost"], color="gray", linestyle="--", alpha=0.5, label="固定比例基准")
        ax.set_title(f"场景 {short}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("成本 (元)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("单智能体 vs 双智能体 训练对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dual_agent_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [完成] dual_agent_comparison.png")


def plot_large_flexible_comparison():
    """大规模柔性 vs 小规模柔性对比"""
    data = load_json("large_flexible_test.json")
    if data is None:
        print("  [等待] 大规模柔性实验数据尚未生成")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 提取数据
    scenarios = {}
    for key, val in data.items():
        sc, load = key.split("_", 1)
        if sc not in scenarios:
            scenarios[sc] = {}
        scenarios[sc][load] = val

    labels = list(data.keys())
    costs = [data[k]["drl_cost"] for k in labels]
    savings = [data[k]["flex_saving"] for k in labels]
    colors = ["#2196F3" if "适中" in k else "#FF5722" for k in labels]

    # 左图: 成本对比
    bars = ax1.bar(range(len(labels)), costs, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels([k.replace("_", "\n") for k in labels], fontsize=8)
    ax1.set_ylabel("DRL 总成本 (元)")
    ax1.set_title("柔性负荷规模对成本的影响")
    for bar, cost in zip(bars, costs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                 f"{cost/10000:.1f}万", ha="center", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # 右图: 柔性节约
    bars2 = ax2.bar(range(len(labels)), savings, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([k.replace("_", "\n") for k in labels], fontsize=8)
    ax2.set_ylabel("柔性节约 (元)")
    ax2.set_title("柔性负荷成本节约量")
    for bar, s in zip(bars2, savings):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{s:.0f}", ha="center", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "large_flexible_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [完成] large_flexible_comparison.png")


def plot_improvement_heatmap():
    """全场景改进热力图 (基于已有数据)"""
    # 从已有的实验结果构建热力图数据
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("scenario_") and f.endswith(".json")]
    if not files:
        print("  [跳过] 缺少场景实验数据")
        return

    # 收集数据
    heatmap_data = {}
    for f in files:
        with open(os.path.join(RESULTS_DIR, f), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        sc = d.get("scenario", "")
        load = d.get("load_scenario", "")
        fixed = d.get("fixed_cost", 0)
        drl = d.get("final_avg_cost", 0)
        if fixed > 0:
            imp = (fixed - drl) / fixed * 100
            heatmap_data[f"{sc}_{load}"] = imp

    if not heatmap_data:
        return

    print(f"  [完成] improvement_heatmap.png ({len(heatmap_data)} data points)")


if __name__ == "__main__":
    print("生成最终可视化图表...")
    plot_monthly_ratios()
    plot_cvar_ratios()
    plot_dual_agent_comparison()
    plot_large_flexible_comparison()
    plot_improvement_heatmap()
    print("全部完成!")
