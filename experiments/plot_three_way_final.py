"""
最终主方案三方法对比图 (固定比例 / 分层架构 / 双 DRL 简化方案)

读取:
    results/dual_v2_simplified_summary.json   — 双 DRL 简化方案 (最终主方案)
    results/hierarchical_comparison.json       — 分层架构基线 (若存在)
输出:
    results/three_way_final.png                — 成本对比 + 改进幅度对比
"""
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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


def main():
    dual = load("results/dual_v2_simplified_summary.json")
    hier = load("results/hierarchical_comparison.json")
    if not dual:
        raise SystemExit("缺少 results/dual_v2_simplified_summary.json")

    tags = list(dual["results"].keys())
    fixed = [dual["results"][t]["fixed_baseline"] for t in tags]
    dual_c = [dual["results"][t]["final_cost"] for t in tags]
    dual_imp = [dual["results"][t]["improvement_vs_fixed_pct"] for t in tags]

    if hier and "scenarios" in hier:
        hier_c, hier_imp = [], []
        for t in tags:
            e = hier["scenarios"].get(t)
            hier_c.append(e["hierarchical"]["cost_mean"] if e else np.nan)
            hier_imp.append(e["improvement_vs_fixed_pct"] if e else np.nan)
    else:
        hier_c = [np.nan] * len(tags)
        hier_imp = [np.nan] * len(tags)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(tags))
    w = 0.27
    has_hier = not np.any(np.isnan(hier_c))

    # 左图: 年总成本
    if has_hier:
        ax1.bar(x - w, fixed, w, label="固定比例", color="#95A5A6")
        ax1.bar(x, hier_c, w, label="分层架构 (1 SAC+规则)", color="#F39C12")
        ax1.bar(x + w, dual_c, w, label="双 DRL 智能体 (最终方案)", color="#2E86DE")
    else:
        ax1.bar(x - w / 2, fixed, w, label="固定比例", color="#95A5A6")
        ax1.bar(x + w / 2, dual_c, w, label="双 DRL 智能体 (最终方案)", color="#2E86DE")
    ax1.set_xticks(x)
    ax1.set_xticklabels([SCEN_NAME[t] for t in tags], fontsize=8, rotation=12, ha="right")
    ax1.set_ylabel("年总成本 (元)")
    ax1.set_title("三方策略年总成本对比")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_ylim(0, max(fixed) * 1.12)

    # 右图: 改进幅度 (%)
    if has_hier:
        ax2.bar(x - w / 2, hier_imp, w, label="分层架构", color="#F39C12")
        ax2.bar(x + w / 2, dual_imp, w, label="双 DRL (最终)", color="#2E86DE")
    else:
        ax2.bar(x, dual_imp, 0.5, label="双 DRL (最终)", color="#2E86DE")
    for i, v in enumerate(dual_imp):
        ax2.text(i + w / 2, v + 0.5, f"{v:+.1f}%", ha="center", fontsize=8,
                 fontweight="bold", color="#1B4F72")
    if has_hier:
        for i, v in enumerate(hier_imp):
            ax2.text(i - w / 2, v + 0.5, f"{v:+.1f}%", ha="center", fontsize=8, color="#B9770E")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([SCEN_NAME[t] for t in tags], fontsize=8, rotation=12, ha="right")
    ax2.set_ylabel("相比固定比例的改进 (%)")
    d_avg = np.nanmean(dual_imp)
    h_avg = np.nanmean(hier_imp)
    ax2.set_title(f"改进幅度对比 | 双 DRL 均 {d_avg:+.1f}% vs 分层均 {h_avg:+.1f}%")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    os.makedirs("results", exist_ok=True)
    out = "results/three_way_final.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"生成: {out}")
    print(f"双 DRL 平均改进: {d_avg:+.1f}%  分层平均: {h_avg:+.1f}%")


if __name__ == "__main__":
    main()
