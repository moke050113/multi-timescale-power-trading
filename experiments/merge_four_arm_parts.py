"""
合并逐场景落盘的四臂结果 (run_four_arm_benchmark.py --out results/parts/four_arm_<场景>.json)

逐场景跑是为了让长运行中途被打断时不丢已完成的部分; 这里把分片拼回一份总结果,
并重新按 (场景, 臂) 跨种子汇总 —— 配对改进只在"同种子同世界"内计算, 所以合并必须是
同一冻结预算下的分片, 否则拒绝合并 (meta 不一致直接报错)。

用法:
    python experiments/merge_four_arm_parts.py                       # 默认 glob results/parts/four_arm_*.json
    python experiments/merge_four_arm_parts.py --out results/four_arm_benchmark.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_four_arm_benchmark import ARMS, SCENARIO_KEYS, print_table, summarize

# 合并前必须逐位一致的 meta 字段 (预算/随机性/基线口径)
FROZEN_KEYS = ("arms", "seeds", "load_scenario", "episodes_per_sac_arm",
               "eval_worlds", "phase_split", "baseline_arm", "smoke")


def load_parts(paths):
    parts = [json.load(open(p, encoding="utf-8")) for p in paths]
    ref = parts[0]["meta"]
    for p, path in zip(parts[1:], paths[1:]):
        diff = [k for k in FROZEN_KEYS if p["meta"].get(k) != ref.get(k)]
        if diff:
            raise ValueError(f"{path} 与 {paths[0]} 的冻结预算不一致: {diff}")
        # 源锁不同 = 分片由不同代码产生, 拼在一起的数字无法追溯到一次运行
        a, b = ref.get("source_lock"), p["meta"].get("source_lock")
        if a is not None or b is not None:
            if a is None or b is None:
                raise ValueError(f"{path} 与 {paths[0]} 一份有源锁一份没有, 拒绝合并")
            same = [k for k in ("files", "numpy", "torch") if a.get(k) != b.get(k)]
            if same:
                raise ValueError(f"{path} 与 {paths[0]} 的源锁 {same} 不一致 "
                                 f"(commit {a.get('git_commit')} vs {b.get('git_commit')}), "
                                 f"分片不是同一次代码状态跑出来的")
    return parts


def check_coverage(parts, arms, expect_scenarios):
    """报告缺失的 (场景, 种子, 臂) 组合; 分片跑一半时这里会显式提醒。

    分片文件只在某个场景的 3 种子 × 4 臂全部跑完后才写出, 所以"已完成的场景内部齐不齐"
    与"5 个场景有没有缺席"是两件事, 都检查。
    """
    seen = {(r["scenario"], r["seed"], r["arm"]) for p in parts for r in p["runs"]}
    have = sorted({r["scenario"] for p in parts for r in p["runs"]})
    seeds = sorted(parts[0]["meta"]["seeds"])
    missing = [(s, sd, a) for s in have for sd in seeds for a in arms
               if (s, sd, a) not in seen]
    absent = [s for s in expect_scenarios if s not in have]
    return missing, absent, len(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=str, default="results/parts/four_arm_*.json")
    parser.add_argument("--out", type=str, default="results/four_arm_benchmark.json")
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--expect-scenarios", type=str, default="all",
                        help="预期场景数 (all=5 个场景; 逗号分隔的场景首字母; none=不检查缺席场景)")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.parts))
    if not paths:
        raise SystemExit(f"未找到分片: {args.parts}")
    parts = load_parts(paths)
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    if args.expect_scenarios == "all":
        expect = SCENARIO_KEYS
    elif args.expect_scenarios == "none":
        expect = []
    else:
        wanted = {s.strip().upper() for s in args.expect_scenarios.split(",")}
        expect = [k for k in SCENARIO_KEYS if k[0] in wanted]

    runs = [r for p in parts for r in p["runs"]]
    failures = [f for p in parts for f in p.get("failures", [])]
    missing, absent, n_done = check_coverage(parts, arms, expect)

    table = summarize(runs)
    print_table(table, arms)

    meta = dict(parts[0]["meta"])
    meta["merged_from"] = paths
    meta["scenarios"] = sorted({r["scenario"] for r in runs})
    meta["n_runs"] = len(runs)
    meta["incomplete_cells"] = missing
    meta["absent_scenarios"] = absent
    meta["expect_scenarios"] = expect
    payload = {
        "meta": meta,
        "summary": [{"scenario": s, "arm": a, "seeds": e["seed_count"],
                     "mean_cost": float(np.mean(e["costs"])),
                     "improvement_vs_baseline_pct": float(np.mean(e["improvements"])),
                     "per_seed_wins": e["wins"]}
                    for (s, a), e in sorted(table.items())],
        "runs": runs,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n合并 {len(paths)} 个分片, {n_done} 个 (场景,种子,臂) 组合 → {args.out}")
    print(f"验收失败记录 {len(failures)} 条")
    if absent:
        print(f"⚠️ 缺席场景 {len(absent)} 个 (分片尚未写出): "
              f"{', '.join(s[:2] for s in absent)}")
    if missing:
        print(f"⚠️ 尚有 {len(missing)} 个组合未跑完, 当前汇总不完整, 不可用于结论:")
        for s, sd, a in missing[:12]:
            print(f"   {s} seed{sd} {a}")
        if len(missing) > 12:
            print(f"   ... 其余 {len(missing) - 12} 条见 meta.incomplete_cells")
    if not missing and not absent:
        print("✅ 预期场景与组合全部齐备")


if __name__ == "__main__":
    main()
