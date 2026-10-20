"""
B 方案消融实验 (Ablation Study)

验证三个关键设计各自的贡献。变体:
    1. Full          — 完整 B 方案 (放大 + 塑形 + 交替)
    2. NoScale       — 去掉现货奖励放大 (不除以 1-γ)
    3. NoShaping     — 去掉中长期未来暴露成本塑形
    4. NoAlternating — 去掉三阶段交替, 从一开始联合训练

在场景 A (常规) 和 D (现货高波动) 上各跑, 对比最终成本与改进幅度。
输出: results/ablation_dual_v2.json
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train_dual_agent_v2 import train_dual_agent

SCENARIOS = ["A_年度低价后期上涨", "D_现货高波动"]
EPISODES = 1000  # 每变体每场景

VARIANTS = {
    "Full":          dict(scale_spot_reward=True,  exposure_shaping=True,  alternating=True),
    "NoScale":       dict(scale_spot_reward=False, exposure_shaping=True,  alternating=True),
    "NoShaping":     dict(scale_spot_reward=True,  exposure_shaping=False, alternating=True),
    "NoAlternating": dict(scale_spot_reward=True,  exposure_shaping=True,  alternating=False),
}

# 三阶段切分 (alternating 用); NoAlternating 会忽略阶段直接用 total
p1 = EPISODES // 4
p2 = EPISODES // 4
p3 = EPISODES - p1 - p2


def main():
    results = {}
    for sc in SCENARIOS:
        tag = sc.split("_")[0]
        results[tag] = {}
        for vname, cfg in VARIANTS.items():
            print(f"\n{'='*60}\n[消融] 场景 {tag} | 变体 {vname} | {cfg}\n{'='*60}")
            r = train_dual_agent(
                scenario=sc,
                load_scenario="柔性适中",
                phase1_episodes=p1,
                phase2_episodes=p2,
                phase3_episodes=p3,
                print_interval=200,
                eval_interval=200,
                save_prefix=None,
                **cfg,
            )
            results[tag][vname] = {
                "final_cost": r["final_cost"],
                "fixed_baseline": r["fixed_baseline"],
                "improvement_vs_fixed_pct": r["improvement_vs_fixed_pct"],
                "final_flex_saving": r["final_flex_saving"],
            }

    out = "results/ablation_dual_v2.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"episodes": EPISODES, "variants": list(VARIANTS.keys()), "results": results},
                  f, ensure_ascii=False, indent=2)

    # 汇总表
    print("\n" + "=" * 72)
    print("消融实验结果 (改进% vs 固定比例)")
    print("=" * 72)
    header = f"{'变体':<16}" + "".join(f"{t:>12}" for t in results.keys())
    print(header)
    print("-" * 72)
    for vname in VARIANTS:
        row = f"{vname:<16}"
        for t in results:
            row += f"{results[t][vname]['improvement_vs_fixed_pct']:>11.1f}%"
        print(row)
    print(f"\n结果已写入: {out}")


if __name__ == "__main__":
    main()
