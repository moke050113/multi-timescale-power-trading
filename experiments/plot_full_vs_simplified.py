"""完整 B 方案 vs 简化方案 对比图 (5 场景)"""
import json, numpy as np, matplotlib.pyplot as plt, matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

f = json.load(open("results/dual_agent_v2_summary.json", encoding="utf-8"))
s = json.load(open("results/dual_v2_simplified_summary.json", encoding="utf-8"))
tags = ["A", "B", "C", "D", "E"]
fi = [f["results"][t]["improvement_vs_fixed_pct"] for t in tags]
si = [s["results"][t]["improvement_vs_fixed_pct"] for t in tags]

x = np.arange(len(tags)); w = 0.36
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - w/2, fi, w, label="完整 B 方案 (放大+塑形+交替)", color="#8FB4E8")
ax.bar(x + w/2, si, w, label="简化方案 (仅塑形+联合训练)", color="#1F618D")
for i in range(len(tags)):
    ax.text(i - w/2, fi[i]+0.5, f"{fi[i]:.1f}", ha="center", fontsize=9)
    ax.text(i + w/2, si[i]+0.5, f"{si[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(tags)
ax.set_xlabel("价格场景"); ax.set_ylabel("成本改进 (vs 固定比例, %)")
ax.set_title(f"完整 vs 简化 B 方案 | 均值 {np.mean(fi):.1f}% → {np.mean(si):.1f}% (+{np.mean(si)-np.mean(fi):.1f}pp)\n简化方案 5/5 场景全面更优")
ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_ylim(0, 48)
plt.tight_layout(); plt.savefig("results/dual_v2_full_vs_simplified.png", dpi=150); plt.close()
print("生成: results/dual_v2_full_vs_simplified.png")
