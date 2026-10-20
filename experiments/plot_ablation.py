"""
消融实验结果可视化

输入: results/ablation_dual_v2.json
输出: results/dual_v2_ablation.png — 4 变体 × 2 场景改进分组柱状图
"""
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

VARIANT_LABEL = {
    "Full": "完整 B 方案",
    "NoScale": "去现货奖励放大",
    "NoShaping": "去暴露成本塑形",
    "NoAlternating": "去三阶段交替",
}
VARIANT_COLOR = {
    "Full": "#2E86DE",
    "NoScale": "#E74C3C",
    "NoShaping": "#F39C12",
    "NoAlternating": "#8E44AD",
}


def main(path="results/ablation_dual_v2.json", out="results/dual_v2_ablation.png"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]          # {scenario_tag: {variant: {...}}}
    variants = data["variants"]         # ["Full", "NoScale", ...]
    scen_tags = list(results.keys())    # ["A", "D"]

    fig, axes = plt.subplots(1, len(scen_tags), figsize=(6 * len(scen_tags), 5), sharey=True)
    if len(scen_tags) == 1:
        axes = [axes]

    for ax, st in zip(axes, scen_tags):
        vals = [results[st][v]["improvement_vs_fixed_pct"] for v in variants]
        colors = [VARIANT_COLOR[v] for v in variants]
        bars = ax.bar(range(len(variants)), vals, color=colors, edgecolor="white")
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{vals[i]:+.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([VARIANT_LABEL[v] for v in variants], rotation=18, ha="right", fontsize=9)
        ax.set_title(f"场景 {st}")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.axhline(0, color="k", lw=0.8)

    axes[0].set_ylabel("成本改进 (vs 固定比例, %)")
    plt.suptitle(f"B 方案消融实验 | 每变体 {data['episodes']} episodes | 负荷场景: 柔性适中")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"生成: {out}")

    # 控制台表
    print("\n变体              " + "".join(f"{t:>10}" for t in scen_tags))
    for v in variants:
        print(f"{VARIANT_LABEL[v]:<16}" + "".join(f"{results[t][v]['improvement_vs_fixed_pct']:>9.1f}%" for t in scen_tags))


if __name__ == "__main__":
    main()
