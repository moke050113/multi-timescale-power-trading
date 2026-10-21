"""消融结果汇总: 把"关闭某个设计部件"的变体与完整方法在**同种子、同世界**上配对比较。

基准是主四臂基准里的场景分片 (完整方法), 变体来自 `run_ablation.sh` 的
`results/parts/ablation_<变体>.json`。两侧都是同一入口、同一冻结预算跑出来的,
所以逐世界配对是合法的; 脚本会先校验预算字段一致, 不一致直接拒绝而不是硬凑。

输出: 终端表格 + markdown (可直接贴进报告) + JSON。
正号 = 变体比完整方法**更贵** = 被关掉的部件在完整方法里做了正贡献。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

BUDGET_KEYS = ("episodes_per_sac_arm", "eval_worlds", "phase_split", "load_scenario")


def load_base(path: str, scenario_prefix: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in d["runs"]:
        if r["scenario"].startswith(scenario_prefix):
            out[(r["arm"], r["seed"])] = r
    return d["meta"], out


def check_budget(base_meta: dict, var_meta: dict, name: str) -> None:
    for k in BUDGET_KEYS:
        if k in base_meta and k in var_meta and base_meta[k] != var_meta[k]:
            raise SystemExit(f"{name}: 预算字段 {k} 与基准不一致 "
                             f"({var_meta[k]} vs {base_meta[k]}), 拒绝配对")


def summarize(variant: str, path: str, base_meta: dict, base: dict) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    check_budget(base_meta, d["meta"], variant)
    if d.get("failures"):
        print(f"  !! {variant}: 有 {len(d['failures'])} 条验收失败记录, 先修会计再谈消融")
    rows = []
    for r in d["runs"]:
        ref = base.get((r["arm"], r["seed"]))
        if ref is None:
            print(f"  !! {variant}: 基准缺少 ({r['arm']}, seed={r['seed']})")
            continue
        ref_by_world = {w["world"]: w for w in ref["worlds"]}
        pairs = [(w, ref_by_world[w["world"]]) for w in r["worlds"]
                 if w["world"] in ref_by_world]
        if not pairs:
            continue
        delta = np.array([(w["cost"] - b["cost"]) / b["cost"] * 100.0 for w, b in pairs])
        rows.append({
            "arm": r["arm"], "seed": r["seed"],
            "cost": float(r["mean_cost"]), "cost_ref": float(ref["mean_cost"]),
            "delta_pct": float(delta.mean()),
            "worse_worlds": int((delta > 0).sum()), "n_worlds": len(delta),
            "alpha": float(np.mean([w["alphas"][0] for w in r["worlds"]])),
            "alpha_ref": float(np.mean([w["alphas"][0] for w in ref["worlds"]])),
            "gamma": float(r["mean_gamma"]), "gamma_ref": float(ref["mean_gamma"]),
            "intensity": float(r["mean_intensity_abs"]),
            "intensity_ref": float(ref["mean_intensity_abs"]),
        })
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    arms = []
    for arm, rs in sorted(by_arm.items()):
        arms.append({
            "arm": arm,
            "cost": float(np.mean([x["cost"] for x in rs])),
            "cost_ref": float(np.mean([x["cost_ref"] for x in rs])),
            "delta_pct": float(np.mean([x["delta_pct"] for x in rs])),
            "per_seed_delta": [round(x["delta_pct"], 3) for x in sorted(rs, key=lambda z: z["seed"])],
            "worse_worlds": f"{sum(x['worse_worlds'] for x in rs)}/{sum(x['n_worlds'] for x in rs)}",
            "alpha": float(np.mean([x["alpha"] for x in rs])),
            "alpha_ref": float(np.mean([x["alpha_ref"] for x in rs])),
            "gamma": float(np.mean([x["gamma"] for x in rs])),
            "gamma_ref": float(np.mean([x["gamma_ref"] for x in rs])),
            "intensity": float(np.mean([x["intensity"] for x in rs])),
            "intensity_ref": float(np.mean([x["intensity_ref"] for x in rs])),
            "seeds": rs,
        })
    return {"variant": variant, "file": os.path.basename(path),
            "ablation": d["meta"].get("ablation", {}), "arms": arms}


def emit_markdown(results: list) -> str:
    lines = ["| 消融变体 | 臂 | 成本 (元) | 相对完整方法 | 更贵世界数 | α (完整) | γ (完整) | \\|强度\\| (完整) |",
             "|---|---|---|---|---|---|---|---|"]
    for res in results:
        for a in res["arms"]:
            lines.append(
                f"| {res['variant']} | {a['arm']} | {a['cost']:,.0f} | {a['delta_pct']:+.2f}% "
                f"| {a['worse_worlds']} | {a['alpha']:.3f} ({a['alpha_ref']:.3f}) "
                f"| {a['gamma']:.3f} | {a['intensity']:.3f} ({a['intensity_ref']:.3f}) |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/four_arm_benchmark.json")
    ap.add_argument("--parts", default="results/parts/ablation_*.json")
    ap.add_argument("--scenario", default="A", help="场景首字母 (消融只跑了一个场景)")
    ap.add_argument("--out", default="results/ablation_summary.json")
    args = ap.parse_args()

    base_meta, base = load_base(args.base, args.scenario)
    files = sorted(glob.glob(args.parts))
    if not files:
        raise SystemExit(f"没有匹配的消融分片: {args.parts}")
    print(f"基准: {args.base} 场景 {args.scenario} ({len(base)} 个 (臂,种子) 组合)")
    results = []
    for p in files:
        name = os.path.basename(p)[len("ablation_"):-len(".json")]
        print(f"\n=== {name} ({p})")
        res = summarize(name, p, base_meta, base)
        for a in res["arms"]:
            print(f"  {a['arm']:<10} 成本={a['cost']:>10,.0f} (完整 {a['cost_ref']:>10,.0f}) "
                  f"Δ={a['delta_pct']:+.2f}% 逐种子={a['per_seed_delta']} "
                  f"更贵世界={a['worse_worlds']} α={a['alpha']:.3f}(完整{a['alpha_ref']:.3f}) "
                  f"γ={a['gamma']:.3f} |强度|={a['intensity']:.3f}(完整{a['intensity_ref']:.3f})")
        results.append(res)

    md = emit_markdown(results)
    print("\n" + md)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"base": os.path.basename(args.base), "scenario": args.scenario,
                   "budget_keys": list(BUDGET_KEYS), "results": results,
                   "sign": "delta_pct>0 = 关闭该部件后更贵 = 该部件在完整方法里有正贡献"},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
