"""跨进程重复运行的训练侧噪声度量 (二次审计 N05)。

主基准与重复运行同代码、同 seed、同配置、同配对世界, 唯一差别是进程启动时的 RNG 轨迹。
两者逐世界成本之差因此就是"冻结配置不等于冻结结果"的量级, 也就是判读门槛里
"小于该量级的差值不得写成改进"的证据来源。此前这组数只能从不入库的训练日志里读,
本脚本把它的计算落到仓库内: 输入主基准分片与重复运行分片 (学习臂一份、规则臂一份, 因为
规则臂不需要重训, 单独用 --no-train 跑一遍很快), 输出 results/repeat_noise.json。

规则臂是确定性的, 顺便用它验证"同 seed 同世界"这一前提本身成立 (差值应为 0)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEARNERS = ("sac_rule", "dual_sac")
RULES = ("fixed_rule", "rolling_rule")


def runs_by_arm_seed(payload: dict) -> dict:
    return {(r["arm"], r["seed"]): r for r in payload["runs"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/parts/four_arm_A.json")
    ap.add_argument("--rep", nargs="+", default=["results/parts/repeat_four_arm_A.json",
                                                "results/parts/repeat_rule_A.json"],
                    help="重复运行分片, 可给多份 (学习臂与规则臂分别跑时), 合起来必须覆盖 --ref 的比较对象")
    ap.add_argument("--out", default="results/repeat_noise.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    reps = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.rep]
    pk, rep_locks = {}, {}
    for p, payload in zip(args.rep, reps):
        for r in payload["runs"]:
            key = (r["arm"], r["seed"])
            if key in pk:
                raise SystemExit(f"{key} 同时出现在 {pk[key][1]} 与 {p}, --rep 分片不得重叠")
            pk[key] = (r, p)
        rep_locks[p] = payload["meta"].get("source_lock")
    rep = reps[0]
    for p, payload in zip(args.rep, reps):
        for k in ("scenarios", "seeds", "episodes_per_sac_arm", "eval_worlds", "load_scenario"):
            if k in payload["meta"] and payload["meta"][k] != reps[0]["meta"][k]:
                raise SystemExit(f"{p} 的 meta.{k}={payload['meta'][k]} 与 {args.rep[0]} 不一致")
    pk = {k: v for k, (v, _p) in pk.items()}
    rk = runs_by_arm_seed(ref)

    learner_rows, rule_gaps, paired = [], [], []
    for key in sorted(rk, key=lambda k: (LEARNERS.index(k[0]) if k[0] in LEARNERS else 9, k[1])):
        arm, seed = key
        if arm not in LEARNERS or key not in pk:
            continue
        a, b = rk[key], pk[key]
        if len(a["worlds"]) != len(b["worlds"]):
            raise SystemExit(f"{arm}/seed{seed} 的世界数不同: {len(a['worlds'])} vs {len(b['worlds'])}")
        delta = (float(b["mean_cost"]) - float(a["mean_cost"])) / float(a["mean_cost"]) * 100
        learner_rows.append({"arm": arm, "seed": seed,
                             "ref_cost": float(a["mean_cost"]), "rep_cost": float(b["mean_cost"]),
                             "delta_pct": delta,
                             "ref_gamma": float(a["mean_gamma"]), "rep_gamma": float(b["mean_gamma"])})
        for wa, wb in zip(a["worlds"], b["worlds"]):
            if wa["world"] != wb["world"]:
                raise SystemExit(f"{arm}/seed{seed} 的世界编号不匹配: {wa['world']} vs {wb['world']}")
            paired.append({"arm": arm, "seed": seed, "world": wa["world"],
                           "ref_cost": float(wa["cost"]), "rep_cost": float(wb["cost"]),
                           "delta_pct": (float(wb["cost"]) - float(wa["cost"]))
                                        / float(wa["cost"]) * 100})
    for key in sorted(rk, key=lambda k: (k[0], k[1])):
        arm, seed = key
        if arm not in RULES or key not in pk:
            continue
        a, b = rk[key], pk[key]
        gaps = []
        for wa, wb in zip(a["worlds"], b["worlds"]):
            if wa["world"] != wb["world"]:
                raise SystemExit(f"{arm}/seed{seed} 的世界编号不匹配: {wa['world']} vs {wb['world']}")
            gaps.append(abs(float(wb["cost"]) - float(wa["cost"])))
        rule_gaps.append({"arm": arm, "seed": seed, "max_abs_cost_gap": max(gaps),
                          "worlds": len(gaps)})

    run_abs = [abs(r["delta_pct"]) for r in learner_rows]
    world_abs = [abs(p["delta_pct"]) for p in paired]
    if not rule_gaps:
        raise SystemExit("--rep 里没有规则臂分片: 需要一份 --no-train 的重复运行才能验证"
                         "“同 seed 同世界”这一前提 (参见 README 的第 9 步)")
    out = {
        "meta": {
            "ref": args.ref, "rep": args.rep,
            "ref_source_lock": ref["meta"].get("source_lock"),
            "rep_source_lock": {p: lk for p, lk in rep_locks.items()},
            "scenario": rep["meta"]["scenarios"], "seeds": rep["meta"]["seeds"],
            "episodes_per_sac_arm": rep["meta"]["episodes_per_sac_arm"],
            "eval_worlds": rep["meta"]["eval_worlds"],
            "learner_paired_worlds": len(paired),
            "rule_paired_worlds": sum(r["worlds"] for r in rule_gaps),
            "note": "同一份代码、同 seed、同配对世界, 只换一次进程启动的两次独立训练的逐世界成本差",
        },
        "learner_runs": learner_rows,
        "paired_worlds": paired,
        "rule_arms": rule_gaps,
        "run_level_abs_pct": {"min": min(run_abs), "max": max(run_abs)},
        "world_level_abs_pct": {"min": min(world_abs), "max": max(world_abs), "n": len(paired)},
        "rule_arms_max_gap_yuan": max(r["max_abs_cost_gap"] for r in rule_gaps),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"学习臂重复运行 {len(learner_rows)} 组 / {len(paired)} 个配对世界")
    for r in learner_rows:
        print(f"  {r['arm']:<9} seed{r['seed']}: {r['ref_cost']:,.0f} -> {r['rep_cost']:,.0f} "
              f"({r['delta_pct']:+.3f}%), γ {r['ref_gamma']:.3f} -> {r['rep_gamma']:.3f}")
    print(f"运行级 |Δ| = {out['run_level_abs_pct']['min']:.2f}%~{out['run_level_abs_pct']['max']:.2f}%")
    print(f"世界级 |Δ| = {out['world_level_abs_pct']['min']:.2f}%~{out['world_level_abs_pct']['max']:.2f}%"
          f" (n={len(paired)})")
    print(f"规则臂最大逐世界成本差 = {out['rule_arms_max_gap_yuan']:.6f} 元 "
          f"(应为 0, 否则“同 seed 同世界”前提不成立; 共 {out['meta']['rule_paired_worlds']} 个世界)")
    print(f"已写 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
