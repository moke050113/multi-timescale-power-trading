"""
把合并后的四臂结果渲染成报告 5.4 节的三张 Markdown 表。

配对改进只在"同种子同世界"内计算; 任一世界会计验收未通过时该组合的成本不进入表格,
而是单独列在末尾, 避免把不可信数字写进报告。

用法:
    python experiments/report_tables.py --in results/four_arm_benchmark.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.plot_four_arm_results import ARM_LABEL, index_runs, load  # noqa: E402


def seed_pairwise(by_key, scen, arm, baseline):
    """逐种子配对改进 % 与更省的世界数; 返回 [(seed, pct, wins, total)]."""
    out = []
    for b in sorted(by_key.get((scen, baseline), []), key=lambda r: r["seed"]):
        c = next((r for r in by_key.get((scen, arm), []) if r["seed"] == b["seed"]), None)
        if c is None:
            continue
        cand = {w["world"]: w["cost"] for w in c["worlds"]}
        pair = [(w["cost"], cand[w["world"]]) for w in b["worlds"] if w["world"] in cand]
        if not pair:
            continue
        bc = np.mean([p[0] for p in pair])
        cc = np.mean([p[1] for p in pair])
        wins = sum(1 for p in pair if p[1] < p[0])
        out.append((b["seed"], (bc - cc) / bc * 100, wins, len(pair)))
    return out


def layer_share(by_key, scen, arm):
    """(中长期三层占比, 现货层占比) —— 按世界平均。"""
    mid, spot = [], []
    for r in by_key.get((scen, arm), []):
        for w in r["worlds"]:
            b, total = w["cost_breakdown"], w["cost_breakdown"]["total"]
            mid.append((b["annual"] + b["monthly"] + b["intraday"]) / total)
            spot.append(b["spot"] / total)
    if not mid:
        return None
    return float(np.mean(mid)) * 100, float(np.mean(spot)) * 100


def render(data, strong=None):
    baseline = data["meta"].get("baseline_arm", "fixed_rule")
    by_key, scenarios, arms = index_runs(data)
    arms = [a for a in arms if a != baseline]
    # 强参照臂: DRL 必须同时压过规则臂, 只赢弱基线不构成证据
    strong = strong if strong and strong != baseline and strong in arms else None
    seeds = sorted(data["meta"]["seeds"])
    lines = []

    head = f"### 表 5.4-1 年均成本与配对改进 (基线 = {ARM_LABEL.get(baseline, baseline)})\n"
    if strong and any(a != strong for a in arms):
        head += (f"\n*改进列同时给出相对弱基线与相对 **{ARM_LABEL[strong]}** 的差值; "
                 "只有后者为正才说明学习真的胜过规则。*\n")
    lines.append(head)
    cols = [ARM_LABEL[a] for a in arms]
    lines.append("| 价格场景 | 基线成本 (元) | " + " | ".join(
        f"{c} 成本 (元) / vs基线"
        + (" / vs强规则" if strong and a != strong else "") for c, a in zip(cols, arms)) + " |")
    lines.append("|---|" + "---|" * (len(arms) + 1))
    for scen in scenarios:
        base_runs = by_key.get((scen, baseline), [])
        costs = [r["mean_cost"] for r in base_runs]
        cells = [f"{np.mean(costs):,.0f} ±{np.std(costs):,.0f}"]
        for arm in arms:
            runs = by_key.get((scen, arm), [])
            if not runs:
                cells.append("—")
                continue
            cell = (f"{np.mean([r['mean_cost'] for r in runs]):,.0f} "
                    f"±{np.std([r['mean_cost'] for r in runs]):,.0f}")
            ps = seed_pairwise(by_key, scen, arm, baseline)
            cell += f" / {np.mean([p[1] for p in ps]):+.2f}%" if ps else " / —"
            if strong and arm != strong:
                qs = seed_pairwise(by_key, scen, arm, strong)
                cell += f" / {np.mean([p[1] for p in qs]):+.2f}%" if qs else " / —"
            cells.append(cell)
        lines.append(f"| {scen} | " + " | ".join(cells) + " |")

    lines.append("\n### 表 5.4-2 逐种子稳健性 (配对改进 % · 更省世界数)\n")
    lines.append("| 价格场景 | " + " | ".join(
        ARM_LABEL[a] + ("" if a == strong else " (vs 基线" + (", vs 强规则)" if strong else ")"))
        for a in arms) + " |")
    lines.append("|---|" + "---|" * len(arms))
    for scen in scenarios:
        cells = []
        for arm in arms:
            ps = seed_pairwise(by_key, scen, arm, baseline)
            qs = seed_pairwise(by_key, scen, arm, strong) if strong and arm != strong else []
            if not ps:
                cells.append("—")
                continue
            rows = []
            for i, (s, p, w, t) in enumerate(ps):
                row = f"seed{s}: {p:+.2f}% ({w}/{t})"
                if qs and i < len(qs):
                    row += f" / {qs[i][1]:+.2f}% ({qs[i][2]}/{qs[i][3]})"
                rows.append(row)
            cells.append("<br>".join(rows))
        lines.append(f"| {scen} | " + " | ".join(cells) + " |")

    lines.append("\n### 表 5.4-3 成本分层占比 (中长期三层 / 现货, 按世界均值)\n")
    lines.append("| 价格场景 | " + " | ".join(
        f"{ARM_LABEL[a]} 中长期 / 现货" for a in [baseline] + arms) + " |")
    lines.append("|---|" + "---|" * (len(arms) + 1))
    for scen in scenarios:
        cells = []
        for arm in [baseline] + arms:
            share = layer_share(by_key, scen, arm)
            cells.append(f"{share[0]:.1f}% / {share[1]:.1f}%" if share else "—")
        lines.append(f"| {scen} | " + " | ".join(cells) + " |")

    total_worlds = data["meta"]["eval_worlds"]
    lines.append("\n### 表 5.4-4 决策口径与验收 "
                 f"({len(data['meta']['seeds'])} 种子 × {total_worlds} 世界合并)\n")
    lines.append("| 价格场景 | 臂 | 通过世界 | γ 均值 | \\|强度\\| 均值 |")
    lines.append("|---|---|---|---|---|")
    for scen in scenarios:
        for arm in [baseline] + arms:
            runs = by_key.get((scen, arm), [])
            if not runs:
                continue
            passed = sum(r["n_worlds_passed"] for r in runs)
            n_total = len(runs) * total_worlds
            flag = "✅" if passed == n_total else "⚠️"
            lines.append(
                f"| {scen} | {ARM_LABEL[arm]} | {flag} {passed}/{n_total} | "
                f"{np.mean([r['mean_gamma'] for r in runs]):.3f} | "
                f"{np.mean([r['mean_intensity_abs'] for r in runs]):.3f} |")

    bad = [r for r in data["runs"] if r["n_worlds_passed"] != total_worlds]
    if bad:
        lines.append("\n**验收未通过组合 (数字不得进入结论):**")
        for r in bad:
            lines.append(f"- {r['scenario']} / {ARM_LABEL[r['arm']]} / seed{r['seed']}: "
                         f"{r['n_worlds_passed']}/{total_worlds}")
    lines.append(f"\n合并自 {len(data['meta'].get('merged_from', []))} 个分片, "
                 f"种子 {seeds}, 每臂 {data['meta']['episodes_per_sac_arm']} 回合, "
                 f"评估世界 {total_worlds} 个, 负荷场景 {data['meta']['load_scenario']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="渲染报告 5.4 节表格")
    parser.add_argument("--in", dest="infile", type=str,
                        default="results/four_arm_benchmark.json")
    parser.add_argument("--strong", type=str, default="rolling_rule",
                        help="强参照臂 (与基线一起给出配对改进); none 表示只看基线")
    args = parser.parse_args()
    print(render(load(args.infile), strong=None if args.strong == "none" else args.strong))
