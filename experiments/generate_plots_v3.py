"""
实验结果可视化 (v3 实验)
生成4张图表:
  1. 训练曲线 (3价格场景 x 4负荷场景)
  2. 改进热力图
  3. 三策略对比柱状图
  4. CVaR敏感性分析
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_exp1():
    results = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*_lambda0.1_flex.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        name = os.path.basename(f).replace(".json", "").replace("result_", "").replace("_lambda0.1_flex", "")
        results[name] = r
    return results


def load_exp3():
    results = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_D_*_lambda*_flex_lambda*.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        lam = r.get("risk_lambda", 0)
        results[lam] = r
    return results


def plot_training_curves(exp1):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for idx, prefix in enumerate(["A", "B", "C"]):
        ax = axes[idx]
        ci = 0
        for name, r in sorted(exp1.items()):
            if name.startswith(f"{prefix}_"):
                costs = r["episode_costs"]
                window = min(50, len(costs))
                if len(costs) >= window:
                    smoothed = np.convolve(costs, np.ones(window) / window, mode="valid")
                    label = name.split("_", 1)[1]
                    ax.plot(range(window - 1, len(costs)), smoothed, color=colors[ci % 4],
                            linewidth=1.5, alpha=0.8, label=label)
                ci += 1

        fixed = list(exp1.values())[0]["fixed_baseline_cost"]
        ax.axhline(y=fixed, color="red", linestyle="--", linewidth=2, label="固定比例基准")
        ax.set_title(f"场景{prefix}", fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_ylabel("总成本 (元)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("实验1: 各价格场景DRL训练曲线 (不同负荷场景)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  exp1_training_curves.png")


def plot_heatmap(exp1):
    fig, ax = plt.subplots(figsize=(10, 5))

    price_labels = ["A_年度低价", "B_年度高价", "C_价格接近"]
    load_labels = ["柔性充裕", "柔性适中", "柔性受限", "纯刚性"]
    load_keys = ["flex充裕", "flex适中", "flex受限", "rigid"]
    improvements = np.zeros((3, 4))

    for i, prefix in enumerate(["A", "B", "C"]):
        for j, lk in enumerate(load_keys):
            key = f"{prefix}_{lk}"
            if key in exp1:
                improvements[i, j] = exp1[key]["improvement_pct"]

    im = ax.imshow(improvements, cmap="RdYlGn", aspect="auto", vmin=0, vmax=65)
    ax.set_xticks(range(4))
    ax.set_xticklabels(load_labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(price_labels)

    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{improvements[i, j]:+.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold")

    plt.colorbar(im, label="成本改进 (%)")
    ax.set_title("实验1: DRL相对固定比例策略的成本改进热力图", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_improvement_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  exp1_improvement_heatmap.png")


def plot_strategy_comparison(exp1):
    random_costs = {"A": 1405433, "B": 1265266, "C": 1364817, "D": 1437593, "E": 1522406}

    fig, ax = plt.subplots(figsize=(14, 6))
    scenarios = ["A", "B", "C", "D", "E"]

    fixed_costs_map = {}
    drl_best = {}
    for name, r in exp1.items():
        prefix = name[0]
        fixed_costs_map[prefix] = r["fixed_baseline_cost"]
        if prefix not in drl_best or r["final_drl_cost"] < drl_best[prefix]:
            drl_best[prefix] = r["final_drl_cost"]

    # Load old D/E results
    for s in ["D", "E"]:
        pattern = os.path.join(RESULTS_DIR, f"result_{s}_*.json")
        for f in glob.glob(pattern):
            if "lambda" not in f:
                with open(f, "r", encoding="utf-8") as fh:
                    r = json.load(fh)
                fixed_costs_map[s] = r["fixed_baseline_cost"]
                drl_best[s] = r["final_drl_cost"]

    x = np.arange(len(scenarios))
    width = 0.25

    fixed_vals = [fixed_costs_map.get(s, 0) for s in scenarios]
    drl_vals = [drl_best.get(s, 0) for s in scenarios]
    random_vals = [random_costs.get(s, 0) for s in scenarios]

    ax.bar(x - width, fixed_vals, width, label="固定比例策略", color="#FF7043", alpha=0.85)
    ax.bar(x, drl_vals, width, label="DRL策略 (最优)", color="#42A5F5", alpha=0.85)
    ax.bar(x + width, random_vals, width, label="随机策略", color="#78909C", alpha=0.85)

    ax.set_xlabel("价格场景")
    ax.set_ylabel("总购电成本 (元)")
    ax.set_title("实验2: 三种策略成本对比", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for i in range(len(scenarios)):
        if fixed_vals[i] > 0:
            imp = (fixed_vals[i] - drl_vals[i]) / fixed_vals[i] * 100
            color = "green" if imp > 0 else "red"
            ax.text(i, drl_vals[i] + max(fixed_vals) * 0.01, f"{imp:+.1f}%",
                    ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp2_strategy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  exp2_strategy_comparison.png")


def plot_cvar_sensitivity(exp3):
    fig, ax = plt.subplots(figsize=(10, 6))

    lambdas = sorted(exp3.keys())
    cvar_costs = [exp3[l]["final_drl_cost"] for l in lambdas]
    cvar_imps = [exp3[l]["improvement_pct"] for l in lambdas]

    ax2 = ax.twinx()
    ax.bar(range(len(lambdas)), cvar_costs, color="#42A5F5", alpha=0.7, label="DRL成本")
    ax2.plot(range(len(lambdas)), cvar_imps, "ro-", linewidth=2, markersize=8, label="改进百分比")

    ax.set_xlabel("风险厌恶系数 lambda")
    ax.set_ylabel("DRL总成本 (元)", color="#42A5F5")
    ax2.set_ylabel("成本改进 (%)", color="red")
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l}" for l in lambdas])
    ax.set_title("实验3: CVaR风险参数敏感性分析 (场景D)", fontsize=13, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_cvar_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  exp3_cvar_sensitivity.png")


if __name__ == "__main__":
    print("加载实验数据...")
    exp1 = load_exp1()
    exp3 = load_exp3()
    print(f"  实验1: {len(exp1)} 组结果")
    print(f"  实验3: {len(exp3)} 组结果")

    print("生成图表...")
    plot_training_curves(exp1)
    plot_heatmap(exp1)
    plot_strategy_comparison(exp1)
    plot_cvar_sensitivity(exp3)
    print("全部图表已生成!")
