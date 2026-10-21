"""报告正文数字自检: 把 5.3~5.7 与 5.9 引用的数字从 results/ 重算一遍。

存在的理由: 本文的性能结论全部落在正文叙述里 (配对改进、覆盖率剖面、成交均价、消融 Δ)。
人工填数一旦与 JSON 脱钩就成了虚假结论, 所以每条主张都同时检查两件事:
  (1) 报告 md 里确实写着这个数;
  (2) 从 JSON 重算出来的数与它相符 (容差按正文给出的小数位取)。
另外表 5.4-1 ~ 5.4-4 必须与 `report_tables.py` 当场生成的表格逐字符相同,
表 5.9-1 必须与 `summarize_ablation.py` 当场重算的配对结果逐格相符。

用法:
    python experiments/verify_report_numbers.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.plot_four_arm_results import index_runs, load          # noqa: E402
from experiments.report_tables import layer_share, render, seed_pairwise  # noqa: E402
from experiments.summarize_ablation import load_base, summarize          # noqa: E402
from src.utils.config import INDUSTRIAL_LOAD_PARAMS, SIM_PARAMS, TOTAL_DEMAND  # noqa: E402
from src.utils.reproducibility import source_lock                        # noqa: E402

LEARNERS = ("sac_rule", "dual_sac")
THRESHOLD = 0.4            # 5.1 节的判读门槛 (%): 经验性审慎规则, 当前代码实测的运行间噪声为 0
ABL_ROWS = {"未来暴露成本塑形": "no_exposure_shaping",
            "现货奖励放大": "no_spot_reward_scale",
            "三阶段交替训练": "no_alternating"}
# 每个变体应当"只关掉自己那一项", 其余两项仍按完整方法开启
ABL_EXPECT = {"no_exposure_shaping": {"exposure_shaping": False, "scale_spot_reward": True,
                                      "alternating": True},
              "no_spot_reward_scale": {"exposure_shaping": True, "scale_spot_reward": False,
                                       "alternating": True},
              "no_alternating": {"exposure_shaping": True, "scale_spot_reward": True,
                                 "alternating": False}}


class Checker:
    def __init__(self, md: str):
        # 正文用排版减号 "−", 计算结果用 ASCII "-"; 统一到后者
        self.md = md.replace("\u2212", "-")
        self.bad = 0
        self.n = 0

    def text(self, label: str, token: str) -> None:
        self.n += 1
        if token.replace("\u2212", "-") not in self.md:
            self.bad += 1
            print(f"  [XX] {label}: 报告里找不到 {token!r}")

    def either(self, label: str, tokens) -> None:
        """同一主张在正文里有加粗/非加粗两种写法。"""
        self.n += 1
        if not any(t.replace("\u2212", "-") in self.md for t in tokens):
            self.bad += 1
            print(f"  [XX] {label}: 报告里这几种写法都没有 {[t[:36] for t in tokens]}")

    def value(self, label: str, computed: float, written: float,
              token: str, tol: float) -> None:
        self.n += 1
        present = token.replace("\u2212", "-") in self.md
        close = abs(computed - written) <= tol
        if not (present and close):
            self.bad += 1
            print(f"  [XX] {label}: 重算={computed:.6g} 正文={written:.6g} "
                  f"{'超差' if not close else ''}{'正文缺失' if not present else ''} ({token!r})")

    def plain(self, label: str, ok: bool, detail: str = "") -> None:
        """只验数据自身的一致性, 不要求正文出现某个字符串。"""
        self.n += 1
        if not ok:
            self.bad += 1
            print(f"  [XX] {label}{': ' + detail if detail else ''}")

    def finish(self) -> int:
        print(f"检查 {self.n} 项主张, 不符 {self.bad} 项")
        return 1 if self.bad else 0


def rng(values) -> tuple:
    a = np.asarray(values, dtype=float)
    return float(a.min()), float(a.max())


def penalty_units(worlds) -> dict:
    """
    只凭 JSON 复算惩罚的单位链 (审计 N01/N07):
        代表日转移电量 × (days_per_month × demand_scale) = 月口径转移电量 × 单价 = 惩罚
    任何一环按"单个代表日"计, 这里就会露馅 —— 不必重跑训练即可否定作废口径。
    """
    K = int(SIM_PARAMS["days_per_month"])
    tc = float(INDUSTRIAL_LOAD_PARAMS["transfer_cost"])
    need = ("demand_scale", "transferred_mwh_day", "transferred_mwh_total")
    absent = sorted({k for w in worlds for k in need if k not in w})
    if absent:
        # 字段缺失 = 这份 JSON 出自改口径之前的驱动; 让每条单位检查都判不过, 而不是抛 KeyError
        return {"absent": absent, "scales": set(), "month_factor": 0.0,
                "worst_scale": float("inf"), "worst_unit": float("inf"), "n_flex": 0,
                "ratio": float("inf"), "max_penalty": 0.0, "max_share": 0.0}
    scales = {round(float(w["demand_scale"]), 12) for w in worlds}
    ds = float(next(iter(scales))) if len(scales) == 1 else float("nan")
    pen = np.array([w["flexible_penalty"] for w in worlds], dtype=float)
    mwh = np.array([w["transferred_mwh_total"] for w in worlds], dtype=float)
    day = np.array([w["transferred_mwh_day"] for w in worlds], dtype=float)
    cost = np.array([w["cost"] for w in worlds], dtype=float)
    flex = mwh > 0
    return {
        "absent": [], "scales": scales, "month_factor": K * ds,
        "worst_scale": float(np.max(np.abs(mwh - day * K * ds))),
        "worst_unit": float(np.max(np.abs(pen - tc * mwh))),
        "n_flex": int(flex.sum()),
        "ratio": float(np.median(pen[flex] / (tc * day[flex]))) if flex.any() else 0.0,
        "max_penalty": float(pen.max()), "max_share": float((100.0 * pen / cost).max()),
    }


def nums(cell: str) -> list:
    """取一个 md 单元格里的带符号数字 (排版减号先统一, 千位逗号去掉)。"""
    s = cell.replace("\u2212", "-").replace(",", "")
    return [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", s)]


# 正文引用的代码行号: 改代码就会让"某某机制见 file:NN-MM"变成假证据 (审计 N07)。
# 除了检查行号还在文件范围内, 还要求那段里出现关键字 —— 只查范围挡不住"行号有效但指错了地方"。
CITATION_KEYS = {
    "src/market/environment.py:354-359": "compute_cvar",
    "src/market/environment.py:335-340": "scale_spot_reward",
    "src/market/environment.py:441-443": "pub_monthly",
    "src/market/settlement.py:365-375": "frac",
    "src/utils/config.py:15-15": "annual",
    "src/utils/config.py:123-123": "risk_lambda",
    "src/utils/config.py:124-124": "cvar_eta",
    "experiments/run_four_arm_benchmark.py:378-383": "--no-alternating",
    "experiments/run_four_arm_benchmark.py:102-104": "torch.manual_seed",
}
CITE_RE = re.compile(r"(src/[a-z_/]+\.py|experiments/[a-z_/]+\.py):(\d+)(?:-(\d+))?")


def check_doc_only(c: "Checker", md: str, readme_path: str, completion_path: str = "") -> None:
    """与 JSON 无关、只靠当前仓库就能判真伪的正文主张。"""
    for path, start, end in CITE_RE.findall(md):
        start, end = int(start), int(end or start)
        c.n += 1
        if not os.path.exists(path):
            c.bad += 1
            print(f"  [XX] 引用 {path}:{start}-{end}: 文件不存在")
            continue
        lines = open(path, encoding="utf-8").read().splitlines()
        key = CITATION_KEYS.get(f"{path}:{start}-{end}", "")
        body = "\n".join(lines[start - 1:end])
        if not (0 < start <= end <= len(lines)) or not body.strip():
            c.bad += 1
            print(f"  [XX] 引用 {path}:{start}-{end}: 行号越界或内容为空 (文件共 {len(lines)} 行)")
        elif not key:
            c.bad += 1
            print(f"  [XX] 引用 {path}:{start}-{end}: 未在 CITATION_KEYS 登记 —— "
                  f"行号有效但没人核对它指向哪段 (新增/改动引用请同时登记关键字)")
        elif key not in body:
            c.bad += 1
            print(f"  [XX] 引用 {path}:{start}-{end}: 那段里找不到 {key!r}, 引用已随代码漂移")
    c.plain("每条带行号的引用都被上面逐条检查过", len(CITE_RE.findall(md)) >= len(CITATION_KEYS),
            f"正文只找到 {len(CITE_RE.findall(md))} 处行号引用")

    n_cases = sum(len(re.findall(r"^def test_", open(p, encoding="utf-8").read(), re.M))
                  for p in sorted(glob.glob("experiments/test_*.py")))
    token = f"{n_cases} 个验收用例"
    c.value("5.2 验收用例总数 = 各 test 文件里的 def test_ 之和", float(n_cases), n_cases, token, 0)
    # 文档里任何"N 个验收用例 / N 用例"的写法都必须等于实测值, 否则是第二套没同步的旧计数
    stray = {int(m) for txt in (md, open(readme_path, encoding="utf-8").read()
                                if os.path.exists(readme_path) else "")
             for m in re.findall(r"(\d+)\s*个?验收用例", txt)} - {n_cases}
    c.plain("报告与 README 里出现的验收用例计数只有实测值", not stray, f"还出现过的计数: {sorted(stray)}")
    # README 复述同一个用例数; 它不在或数字脱钩都算不符 (审计 N07: 缺文档不能"通过")
    readme_txt = (open(readme_path, encoding="utf-8").read()
                  if os.path.exists(readme_path) else "")
    c.n += 1
    if token not in readme_txt:
        c.bad += 1
        print(f"  [XX] README 的用例数: 应写着 {token!r}, 实际找不到 (或 README 不在 {readme_path})")

    left = sorted(set(re.findall(r"<[A-Z0-9_]{2,}>", md)))
    c.plain("正文没有未填的占位符", not left, f"还剩 {left}")

    # 反向用例的条数只能由自检脚本自己给出: 文档里手写一个数, 迟早和脚本脱钩 (审计 N07)
    try:
        import experiments.selfcheck_verifier as S
    except Exception as exc:                                  # 导入失败也要报告, 不能默默跳过
        mut_token = ""
        c.plain("反向用例条数可读 (selfcheck_verifier 能被导入)", False, f"{type(exc).__name__}: {exc}")
    else:
        n_mut = len(S.MUTATIONS) + len(S.DATA_MUTATIONS) + len(S.REPEAT_MUTATIONS)
        mut_token = f"{n_mut} 处变异"
        c.text(f"附录 反向用例条数 = 脚本里的 {len(S.MUTATIONS)}+{len(S.DATA_MUTATIONS)}"
               f"+{len(S.REPEAT_MUTATIONS)}", mut_token)
        c.n += 1
        if mut_token not in readme_txt:
            c.bad += 1
            print(f"  [XX] README 的反向用例条数: 应写着 {mut_token!r}, 实际找不到")

    # 完成度评估复述同一个用例数与变异条数, 它脱钩时同样是虚假主张
    if completion_path and os.path.exists(completion_path):
        ct = open(completion_path, encoding="utf-8").read()
        for tk, what in ((token, "验收用例数"), (mut_token, "反向用例条数")):
            if not tk:
                continue
            c.n += 1
            if tk not in ct:
                c.bad += 1
                print(f"  [XX] 完成度评估的{what}: 应写着 {tk!r}, 实际找不到")


def md_row(md: str, part: str, arm: str) -> list:
    """表 5.9-1 里 (剔除部件, 臂) 那一行的单元格; 没有该行返回空列表。"""
    pre = f"| {part} | `{arm}` |"
    for line in md.replace("\u2212", "-").splitlines():
        if line.startswith(pre):
            return [x.strip() for x in line.strip().strip("|").split("|")]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/four_arm_benchmark.json")
    ap.add_argument("--md", default="report/课程设计报告.md")
    ap.add_argument("--parts", default="results/parts/ablation_*.json")
    ap.add_argument("--summary", default="results/ablation_summary.json")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--completion", default="项目完成度评估报告.md")
    ap.add_argument("--repeat-learner", dest="repeat_learner",
                    default="results/parts/repeat_four_arm_A.json")
    ap.add_argument("--repeat-rule", dest="repeat_rule",
                    default="results/parts/repeat_rule_A.json")
    ap.add_argument("--repeat-summary", dest="repeat_summary", default="results/repeat_noise.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    md = open(args.md, encoding="utf-8").read()
    data = load(args.json)
    by_key, scenarios, arms = index_runs(data)
    base = data["meta"]["baseline_arm"]
    strong = "rolling_rule"
    S = {s[:1]: s for s in scenarios}
    c = Checker(md)

    # ---------- 0. 源锁: 发布出来的数字必须能追溯到产生它的那一份代码 (审计 N05/N07) ----------
    cur_lock = source_lock()

    def check_lock(meta, who: str) -> dict:
        lk = meta.get("source_lock")
        c.n += 1
        if lk is None:
            c.bad += 1
            print(f"  [XX] {who}: meta 里没有 source_lock —— "
                  f"无法确认数字由哪一份代码产生, 需按当前代码重跑")
            return {}
        bad = []
        for f, want in cur_lock["files"].items():
            got = lk.get("files", {}).get(f)
            if got != want:
                bad.append(f"{f}: {got} != {want}")
        for k in ("numpy", "torch"):
            if lk.get(k) != cur_lock[k]:
                bad.append(f"{k}: {lk.get(k)} != {cur_lock[k]}")
        c.plain(f"{who} 的源锁与当前代码树一致", not bad, "; ".join(bad[:3]))
        return lk

    json_lock = check_lock(data["meta"], args.json)

    # ---------- 1. 四张表与生成器逐字符一致 ----------
    blocks, cur = [], []
    for line in render(data, strong=strong).splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            if len(cur) >= 3:
                blocks.append("\n".join(cur))
            cur = []
    if len(cur) >= 3:
        blocks.append("\n".join(cur))
    if len(blocks) != 4:
        print(f"  [XX] 生成器给出 {len(blocks)} 张表, 报告应为 4 张")
        return 1
    for i, t in enumerate(blocks, 1):
        c.text(f"表 5.4-{i} 逐字符等于生成器输出", t)

    # ---------- 2. 验收不变量 ----------
    worlds = [w for r in data["runs"] for w in r["worlds"]]
    chk = [w["checks"] for w in worlds]
    c.value("评估世界数", float(len(worlds)), 600, "600 个评估世界", 0)
    c.value("组合数", float(data["meta"]["n_runs"]), 60, "60 个组合", 0)
    c.value("failures 条数", float(len(data["failures"])), 0, "`failures` 为空", 0)
    lo, hi = rng([wch["energy_residual_mwh"] for wch in chk])
    c.text("电量守恒残差上限 (正文按 1e-12 写)", f"{hi / 1e-12:.2f}\\times10^{{-12}}")
    c.value("分项之和与市场成本的最大残差",
            max(abs(wch["cost_recomputed_residual"]) for wch in chk), 0,
            "600 个世界全为 0", 1e-9)
    c.value("合同层负电量/负成本总数",
            float(sum(wch["contract_qty_negative"] + wch["contract_cost_negative"]
                      for wch in chk)), 0, "无虚假退款", 0)
    c.value("37 步闭环世界数", float(sum(wch["steps_37"] for wch in chk)), 600,
            "37 步闭环", 0)
    c.value("设备完成任务世界数", float(sum(wch["device_task_met"] for wch in chk)), 600,
            "设备日任务按时完成", 0)

    # ---------- 2b. 数据鉴真: 惩罚单位与源锁 (审计 N01/N05/N07) ----------
    u = penalty_units(worlds)
    c.plain("落盘记录带惩罚单位链所需字段", not u["absent"],
            f"缺 {u['absent']} —— 这份 JSON 出自改口径之前的驱动, 必须按当前代码重跑")
    c.plain("全部世界共用同一个 demand_scale", len(u["scales"]) == 1, str(sorted(u["scales"])))
    c.plain("月电量 = 代表日电量 × days_per_month × demand_scale", u["worst_scale"] < 1e-6,
            f"最大残差 {u['worst_scale']:.3g} MWh")
    c.plain("惩罚 = 转移成本单价 × 月口径转移电量", u["worst_unit"] < 1e-6,
            f"最大残差 {u['worst_unit']:.3g} 元")
    c.plain("数据里确有发生转移的世界 (否则单位检查是空转)", u["n_flex"] > 0,
            f"{u['n_flex']} 个世界")
    c.plain("惩罚按整月电量而非单个代表日计", abs(u["ratio"] - u["month_factor"]) < 1e-6,
            f"实测倍数 {u['ratio']:.4f}, 应为 {u['month_factor']:.4f}")
    mx_pen, mx_share = u["max_penalty"], u["max_share"]
    c.value("全年柔性转移惩罚上限", mx_pen, round(mx_pen), f"{mx_pen:,.0f} 元", 0.5)
    c.value("柔性惩罚占年成本上限", mx_share, round(mx_share, 2), f"不超过 {mx_share:.2f}%", 0.005)
    fx = [w for r in data["runs"] if r["arm"] == base for w in r["worlds"]]
    net = np.array([w["energy_breakdown"]["total"] for w in worlds], dtype=float) - TOTAL_DEMAND
    c.value("保障调度臂年成交电量对 Q 的最大残差", float(np.max(np.abs(
        np.array([w["energy_breakdown"]["total"] for w in fx], dtype=float) - TOTAL_DEMAND))),
        0.0, "年成交电量都闭合于", 1e-6)
    # 柔性动作只能通过 EV 往返损耗抬高网侧年电量, 绝不可能低于 Q (工业任务电量守恒)
    c.plain("全部世界的年成交电量不低于 Q", float(net.min()) > -1e-9, f"最小 {net.min():+.4f} MWh")
    ex = float(net.max()) / TOTAL_DEMAND * 100.0
    c.plain("柔性挪动的年电量在 1% 以内 (只可能是 EV 往返损耗)", 0 <= ex < 1.0,
            f"最多 +{ex:.3f}%")
    mismatch = [i for i, w in enumerate(worlds)
                if bool(w["device_task_met"]) != bool(w["checks"]["device_task_met"])]
    c.plain("逐世界设备结论与 checks 里的同一字段不矛盾", not mismatch,
            f"{len(mismatch)} 个世界两处记录不一致")
    ew = int(data["meta"]["eval_worlds"])
    bad_ids = [f"{r['scenario'][:1]}/{r['arm']}/seed{r['seed']}" for r in data["runs"]
               if len({w["world"] for w in r["worlds"]}) != len(r["worlds"])
               or not {w["world"] for w in r["worlds"]} <= set(range(ew))]
    c.plain(f"每个组合的世界编号互不重复且都在 0~{ew - 1}", not bad_ids, str(bad_ids[:4]))

    # 动作可达集 (审计 N02): 正文声称"四臂动作都不越界", 数据里必须有这个计数且为 0
    have_oob = [wch for wch in chk if "action_oob" in wch]
    c.plain("逐世界验收里带 action_oob 计数", len(have_oob) == len(chk),
            f"{len(have_oob)}/{len(chk)} 个世界有该字段 —— 缺字段的 JSON 出自旧驱动, 需重跑")
    n_oob = sum(int(wch["action_oob"]) for wch in have_oob)
    c.plain("全部世界的越界动作数为 0", len(have_oob) == len(chk) and n_oob == 0,
            f"{n_oob} 次越界 —— 规则臂或 SAC 输出超出了 ACTION_BOUNDS 声明的可达集")
    c.text("4.1 动作可达集写法", "checks.action_oob")
    c.text("3.2 惩罚月度尺度写法", "days\\_per\\_month")

    # 训练侧留痕 (审计 N05/N09): 每个 SAC 臂必须记下逐角色优化器步数与阶段日程
    sac_runs = [r for r in data["runs"] if r.get("trained")]
    gs = [r.get("grad_steps") or {} for r in sac_runs]
    c.plain("每个已训练组合都带 grad_steps", len(sac_runs) == 30 and all(g for g in gs),
            f"{len(sac_runs)} 个已训练组合, 其中有步数的 {sum(1 for g in gs if g)} 个")
    c.plain("中长期网络确实被更新过", bool(gs) and all(int(g.get("midlong", 0)) > 0 for g in gs),
            "存在 midlong=0 的训练组合")
    c.plain("分层臂不更新柔性网络",
            all(int(r.get("grad_steps", {}).get("spot", 0)) == 0
                for r in sac_runs if r["arm"] == "sac_rule"))
    ph = {r["arm"]: [q["phase"] for q in r.get("phase_history") or []] for r in sac_runs}
    c.plain("dual_sac 走三阶段、sac_rule 只训中长期",
            bool(ph) and all(v == ["P1-midlong", "P2-spot", "P3-joint"] for k, v in ph.items()
                             if k == "dual_sac")
            and all(v == ["P-midlong"] for k, v in ph.items() if k == "sac_rule"),
            str({k: v for k, v in ph.items()}))
    ab = data["meta"].get("ablation")
    c.plain("主基准 meta.ablation 三项全开",
            ab == {"exposure_shaping": True, "scale_spot_reward": True, "alternating": True},
            f"实测 {ab}")

    # ---------- 2c. 3.4 遗留调度接口: 报告声称"数值不经过它", 这里静态核对 (审计 F05/N08) ----------
    chain = ("experiments/run_four_arm_benchmark.py", "src/market/environment.py",
             "src/market/settlement.py", "src/load/flexible_executor.py",
             "src/agent/rule_policies.py", "src/agent/multi_agent.py")
    src_txt = {p: (Path(p).read_text(encoding="utf-8") if Path(p).exists() else "") for p in chain}
    hits = {p: [t for t in ("optimize_transfer", "get_combined_profile", "FlexibleDispatcher")
                if t in txt] for p, txt in src_txt.items()}
    c.plain("四臂链路的 6 个文件都不引用遗留调度接口",
            all(not v for v in hits.values()), str({k: v for k, v in hits.items() if v}))
    ls = Path("src/load/load_scenarios.py").read_text(encoding="utf-8")
    c.plain("遗留 optimize_transfer 确实只在场景构建里被调用一次",
            ls.count("optimize_transfer(") == 1, f"实测 {ls.count('optimize_transfer(')} 处")
    c.text("3.4 遗留接口已删除的表述", "已在迭代 8.7 随作废产物一并删除")
    c.text("3.4 遗留接口调用方表述", "哑价格调用过一次")

    # ---------- 2d. 5.6 奖励量级比与成本分层: 当场重跑 10 条 rollout, 不依赖脚本的历史输出 ----------
    from experiments.measure_reward_scale import _roll
    from src.agent.rule_policies import fixed_ratio_action, rolling_rule_action
    from src.utils.config import PRICE_SCENARIOS

    rr, ss = [], []
    for scn in PRICE_SCENARIOS:
        for rule in (fixed_ratio_action, rolling_rule_action):
            m, s, mid_share, _ = _roll(scn, rule)
            rr.append(s / m)
            ss.append(mid_share)
    lo_r, hi_r, mean_r = min(rr), max(rr), float(np.mean(rr))
    lo_s, hi_s, mean_s = min(ss), max(ss), float(np.mean(ss))
    c.text("5.6 奖励量级比区间", f"{lo_r:.1f}\\sim{hi_r:.1f}")
    c.text("5.6 奖励量级比均值", f"均值 {mean_r:.1f}")
    c.text("5.6 中长期成本占比区间", f"{lo_s * 100:.0f}\\%\\sim{hi_s * 100:.0f}\\%")
    c.text("5.6 中长期成本占比均值", f"均值 {mean_s * 100:.1f}%")

    # ---------- 2e. 0.4% 判读门槛的仓库内证据: 跨进程重复运行, 差值当场由分片重算 (审计 N05/N07) ----------
    rep_runs, have_rep = {}, True
    for who, p in (("学习臂", args.repeat_learner), ("规则臂", args.repeat_rule)):
        c.n += 1
        if not os.path.exists(p):
            c.bad += 1
            print(f"  [XX] {who}的重复运行分片不存在: {p} —— 门槛的证据要随仓库发布, 不能只留结论")
            have_rep = False
            continue
        d = load(p)
        check_lock(d["meta"], f"{p} (重复运行)")
        for r in d["runs"]:
            c.n += 1
            if (r["arm"], r["seed"]) in rep_runs:
                c.bad += 1
                print(f"  [XX] {r['arm']}/seed{r['seed']} 同时出现在两份重复运行分片里, 配对对象不唯一")
            rep_runs[(r["arm"], r["seed"])] = r
    if have_rep:
        _, base_a_ref = load_base(args.json, S["A"])
        learner_gaps, rule_gaps, mis = [], [], []
        for (arm, seed), rep in sorted(rep_runs.items()):
            ref = base_a_ref.get((arm, seed))
            if ref is None:
                mis.append(f"{arm}/seed{seed} 不在主基准场景 A 里")
                continue
            if len(ref["worlds"]) != len(rep["worlds"]):
                mis.append(f"{arm}/seed{seed} 世界数 {len(rep['worlds'])} != {len(ref['worlds'])}")
                continue
            for wa, wb in zip(ref["worlds"], rep["worlds"]):
                if wa["world"] != wb["world"]:
                    mis.append(f"{arm}/seed{seed} 世界编号 {wa['world']} vs {wb['world']}")
                    break
                g = abs(float(wb["cost"]) - float(wa["cost"]))
                (learner_gaps if arm in LEARNERS else rule_gaps).append(g)
        gaps = learner_gaps + rule_gaps
        c.plain("重复运行的每个组合都在主基准场景 A 里且逐世界编号对齐", not mis, "; ".join(mis[:3]))
        c.value("重复运行覆盖的 (臂, 种子) 组合数", float(len(rep_runs)), 12, "12 个组合", 0)
        c.value("重复运行的学习臂配对世界数", float(len(learner_gaps)), 60, "60 个配对世界", 0)
        c.value("重复运行的规则臂配对世界数", float(len(rule_gaps)), 60, "60 个配对世界", 0)
        c.value("重复运行的配对世界总数", float(len(gaps)), 120, "120 个配对世界", 0)
        c.plain("重复运行: 学习臂逐世界成本差为 0 (三套随机流已锁定, 同配置跨进程可复现)",
                bool(learner_gaps) and max(learner_gaps) == 0.0,
                f"最大 {max(learner_gaps):.6g} 元" if learner_gaps else "没有学习臂配对世界")
        c.plain("重复运行: 规则臂逐世界成本差为 0 (“同 seed 同世界”前提本身成立)",
                bool(rule_gaps) and max(rule_gaps) == 0.0,
                f"最大 {max(rule_gaps):.6g} 元" if rule_gaps else "没有规则臂配对世界")
        rp = args.repeat_summary
        c.n += 1
        if not os.path.exists(rp):
            c.bad += 1
            print(f"  [XX] {rp} 缺失: 门槛证据的汇总要入库, 否则正文的 0 只差值没人复核")
        else:
            s = load(rp)
            c.value(f"{os.path.basename(rp)} 的学习臂配对世界数与重算相符",
                    float(s["meta"]["learner_paired_worlds"]), float(len(learner_gaps)),
                    "120 个配对世界", 0)
            c.value(f"{os.path.basename(rp)} 的规则臂最大差 = 当场重算值",
                    float(s["rule_arms_max_gap_yuan"]), float(max(rule_gaps)) if rule_gaps else 0.0,
                    "120 个配对世界", 0)
            c.plain(f"{os.path.basename(rp)} 记录的重复运行噪声上界为 0",
                    float(s["run_level_abs_pct"]["max"]) == 0.0,
                    f"运行级 |Δ| 最大 {s['run_level_abs_pct']['max']:.6g}%")
        for label, path in (("报告", args.md), ("README", args.readme), ("完成度评估", args.completion)):
            txt = md if path == args.md else (open(path, encoding="utf-8").read()
                                              if os.path.exists(path) else "")
            c.n += 1
            if "120 个配对世界" not in txt:
                c.bad += 1
                print(f"  [XX] {path}: 没写出重复运行的证据规模 (120 个配对世界), 门槛来源仍无处可查")
            c.n += 1
            if "噪声门槛" in txt:
                c.bad += 1
                print(f"  [XX] {path}: 仍把 0.4% 写成“噪声门槛” —— 实测噪声已是 0, 该门槛是审慎规则")

    def costs(scn, arm):
        return float(np.mean([r["mean_cost"] for r in by_key[(scn, arm)]]))

    def delta(scn, arm, ref):
        ps = seed_pairwise(by_key, scn, arm, ref)
        return float(np.mean([p[1] for p in ps])), ps

    # ---------- 3. 5.3 非学习基线 ----------
    for scn in scenarios:
        s = scn[:1]
        fx, rl = costs(scn, base), costs(scn, strong)
        c.value(f"{s} fixed_rule 年成本", fx, round(fx), f"| {s} ", 0.49)
        c.text(f"{s} fixed_rule 成本数字", f"{fx:,.0f}")
        c.text(f"{s} rolling_rule 成本数字", f"{rl:,.0f}")
        d, _ = delta(scn, strong, base)
        c.value(f"{s} 滚动规则改进", d, round(d, 1), f"{d:+.1f}%", 0.05)

    # ---------- 4. 跨场景均值 ----------
    # 这些串一律由数据拼出: 写死一个"上次算出来的数"就等于把审计器变成另一份要手工同步的正文。
    means = {}
    for arm, ref in [(strong, base), ("dual_sac", base), ("dual_sac", strong),
                     ("sac_rule", base), ("sac_rule", strong)]:
        v = float(np.mean([delta(scn, arm, ref)[0] for scn in scenarios]))
        means[(arm, ref)] = v
        c.text(f"5 场景均值 {arm} vs {ref}", f"{v:+.2f}%")
    rb = [delta(scn, strong, base)[0] for scn in scenarios]
    rb_lo, rb_hi = rng(rb)
    c.text("5.3 规则臂差距区间 (正文 4 处引用同一区间)", f"{rb_lo:+.1f}%~{rb_hi:+.1f}%")

    # ---------- 5. 5.4.1 年度饱和与"年度层贵贱"机制 ----------
    def layer_price(scn, layer):
        ws = [w for r in by_key[(scn, base)] for w in r["worlds"]]
        return (float(np.mean([w["cost_breakdown"][layer] for w in ws]))
                / float(np.mean([w["energy_breakdown"][layer] for w in ws])))

    for scn in scenarios:
        s = scn[:1]
        al = [w["alphas"][0] for a in LEARNERS for r in by_key[(scn, a)] for w in r["worlds"]]
        v = float(np.mean(al))
        c.text(f"5.4.1 {s} 学习臂 α 列", f"| {s} | ")
        c.value(f"5.4.1 {s} 学习臂 α 合并均值", v, round(v, 3), f"| {v:.3f} |", 5e-4)
        pa, pm = layer_price(scn, "annual"), layer_price(scn, "monthly")
        c.text(f"5.4.1 {s} 年度/月度层均价列", f"| {pa:.1f} | {pm:.1f} | {pa - pm:+.1f} |")
        c.text(f"5.4.1 {s} 两学习臂相对滚动规则",
               " / ".join(f"{delta(scn, a, strong)[0]:+.2f}%" for a in LEARNERS))
    sign_ok = [(layer_price(scn, "annual") < layer_price(scn, "monthly"))
               == (delta(scn, a, strong)[0] > 0)
               for scn in scenarios for a in LEARNERS]
    c.value("5.4.1 改进符号与年度层贵贱符号一致", float(all(sign_ok)), 1.0,
            "符号完全一致", 0)
    # 年度超配的定量估算: 规则臂年度锚定 α=0.800 (两条都是), 学习臂合并均值见 α 列
    expl = {}
    for s in ("E", "B"):
        scn = [x for x in scenarios if x.startswith(s)][0]
        a_sac = float(np.mean([w["alphas"][0] for r in by_key[(scn, "sac_rule")] for w in r["worlds"]]))
        dp = layer_price(scn, "annual") - layer_price(scn, "monthly")
        est = (a_sac - 0.80) * TOTAL_DEMAND * dp
        obs = costs(scn, "sac_rule") - costs(scn, strong)
        c.text(f"5.4.1 {s} 超配电量", f"{(a_sac - 0.80) * TOTAL_DEMAND:.1f} MWh")
        c.text(f"5.4.1 {s} 估算超配成本", f"{est:,.0f} 元")
        c.text(f"5.4.1 {s} 观测贵出的金额", f"贵 {obs:,.0f} 元")
        c.value(f"5.4.1 {s} 可解释比例", est / obs * 100, round(est / obs * 100),
                f"{est / obs * 100:.0f}%", 0.5)
        expl[s] = est / obs * 100
    frac = {f"{scn[:1]}/{a}": np.mean([w["alphas"][0] > 0.885
              for r in by_key[(scn, a)] for w in r["worlds"]])
            for scn in scenarios for a in LEARNERS}
    exceptions = {k: v for k, v in frac.items() if v < 1.0}
    by_s: dict = {}
    for k, v in frac.items():
        by_s.setdefault(k.split("/")[0], []).append(float(v))
    half = len(next(iter(by_s.values())))
    mixed = [s for s, v in by_s.items() if any(x < 1.0 for x in v) and any(x >= 1.0 for x in v)]
    uneven = [s for s, v in by_s.items() if len(set(v)) > 1]
    c.plain("α 未饱和总是整条场景两臂一起且比例相同 (正文写两条学习臂各 N% 才成立)",
            not mixed and not uneven, f"半饱和场景 {mixed}, 两臂比例不一 {uneven}")
    c.text("5.4.1 α>0.885 满额组合数", f"{len(frac) - len(exceptions)} 个的全部 30 个世界满足")
    for s, v in sorted(by_s.items()):
        if all(x >= 1.0 for x in v):
            continue
        c.text(f"5.4.1 {s} 场景两臂 α 未饱和比例 ({half} 个组合)",
               f"{s} 场景的两条学习臂（各 {min(v) * 100:.0f}%）")

    # ---------- 6. 5.4.2 dual 相对 sac ----------
    ds = []
    for scn in scenarios:
        s = scn[:1]
        d, ps = delta(scn, "dual_sac", "sac_rule")
        ds.append(d)
        w = sum(p[2] for p in ps)
        n = sum(p[3] for p in ps)
        row = f"| {s} | {d:+.2f}% |"
        c.value(f"5.4.2 {s} dual−sac", d, round(d, 2), row, 0.005)
        c.either(f"5.4.2 {s} 更省世界数", [f"{row} {w}/{n} |", f"{row} **{w}/{n}** |"])
        c.text(f"5.4.2 {s} 逐种子列", " / ".join(f"{p:+.2f}%" for _, p, _, _ in ps))
    mean_ds = float(np.mean(ds))
    c.text("5.4.2 五场景均值", f"**{mean_ds:+.3f}%**")
    c.text("5.4.2 五场景均值 (6.1 紧排)", f"{mean_ds:+.2f}%")
    a_gap = {scn[:1]: float(np.mean([w["alphas"][0] for r in by_key[(scn, "dual_sac")] for w in r["worlds"]])
                            - np.mean([w["alphas"][0] for r in by_key[(scn, "sac_rule")] for w in r["worlds"]]))
             for scn in scenarios}
    c.text("5.4.2 两臂年度覆盖率差上界", f"≤{max(abs(v) for v in a_gap.values()):.3f}")
    b_key = [s for s in scenarios if s.startswith("B")][0]
    gap_b = costs(b_key, "sac_rule") - costs(b_key, "dual_sac")
    est_b = abs(a_gap["B"]) * TOTAL_DEMAND * abs(layer_price(b_key, "annual")
                                                 - layer_price(b_key, "monthly"))
    c.text("5.4.2 B 场景年度层可解释的差额", f"只值约 {est_b:,.0f} 元")
    c.text("5.4.2 B 场景两臂成本差", f"{gap_b:,.0f} 元")
    c.text("5.4.2 B 场景年度层解释占比", f"{est_b / gap_b * 100:.1f}% 能由年度层解释")

    # ---------- 7. 5.7 剖面 / 均价 / 强度 ----------
    def prof(s, arm):
        return np.mean([w["gammas"] for r in by_key[(S[s], arm)] for w in r["worlds"]], axis=0)

    DASH = "–"
    e_d, e_s, e_r = (prof("E", a) for a in ("dual_sac", "sac_rule", strong))
    b_d, b_s, b_r = (prof("B", a) for a in ("dual_sac", "sac_rule", strong))
    b_mid, b_all = np.concatenate([b_d[6:9], b_s[6:9]]), np.concatenate([b_d, b_s])
    c.text("5.7 E dual 剖面区间", f"{e_d.min():.3f}{DASH}{e_d.max():.3f}")
    for label, p in (("E sac_rule", e_s), ("E 滚动规则", e_r), ("B 滚动规则", b_r)):
        c.text(f"5.7 {label} 首月+其余月区间",
               f"除首月 {p[0]:.3f} 外落在 {p[1:].min():.3f}{DASH}{p[1:].max():.3f}")
    lo, hi = rng(b_mid)
    c.text("5.7 B 学习臂 7–9 月区间", f"{lo:.3f}{DASH}{hi:.3f}")
    lo, hi = rng(b_all)
    c.text("5.7 B 学习臂全区间", f"{lo:.3f}{DASH}{hi:.3f}")
    for arm, p in (("dual_sac", b_d), ("sac_rule", b_s)):
        c.text(f"5.7 B {arm} 最低月", f"`{arm}` 为 {p.min():.3f}（第 {int(np.argmin(p)) + 1} 月）")
    argmins = [int(np.argmin(prof(s, a))) + 1 for s in "ABCDE" for a in LEARNERS]
    q4 = sum(m >= 10 for m in argmins)
    c.value("5.7 最低月落在 Q4 的剖面数", float(q4), float(q4), f"最低月落在第 4 季度的有 {q4} 个", 0)
    others = sorted({m for m in argmins if m < 10})
    c.text("5.7 其余剖面最低月集合", "、".join(str(m) for m in others) + " 月")
    q4s = [m for m in argmins if m >= 10]
    c.text("5.7 Q4 最低月的月序分布",
           "、".join(f"第 {m} 月 {q4s.count(m)} 个" for m in sorted(set(q4s))))
    lo, hi = rng([float(np.ptp(prof(s, a))) for s in "ABCDE" for a in LEARNERS])
    c.text("5.7 学习臂剖面全距区间", f"全距只有 {lo:.3f}{DASH}{hi:.3f}")
    c.plain("B sac_rule 最低月与第 8 月同值 (正文写'相同')",
            abs(float(b_s[7]) - float(b_s.min())) < 5e-4,
            f"8 月 {b_s[7]:.3f} vs 最低 {b_s.min():.3f}")

    def spot_price(s, arm):
        sc = [w["cost_breakdown"]["spot"] for r in by_key[(S[s], arm)] for w in r["worlds"]]
        en = [w["energy_breakdown"]["spot"] for r in by_key[(S[s], arm)] for w in r["worlds"]]
        return float(np.mean(sc) / np.mean(en)), float(np.mean(en))

    for s in "EB":
        v = spot_price(s, base)[0]
        c.value(f"5.7 {s} fixed 现货成交均价", v, round(v, 1), f"{v:.1f} 元/MWh", 0.05)
    all_lower = all(spot_price(s, "dual_sac")[0]
                    < min(spot_price(s, base)[0], spot_price(s, strong)[0]) for s in "ABCDE")
    c.value("dual 现货均价 5 场景低于两条规则臂", float(all_lower), 1.0,
            "全部低于两条规则臂", 0)
    d_ex = [spot_price("D", a)[0] for a in ("dual_sac", strong, base)]
    c.text("5.7 D 现货均价例子", f"{d_ex[0]:.1f} vs {d_ex[1]:.1f} / {d_ex[2]:.1f} 元/MWh")
    for s in "ED":
        q = [round(spot_price(s, a)[1]) for a in (base, strong, "sac_rule", "dual_sac")]
        c.text(f"5.7 {s} 四臂现货电量", " / ".join(str(x) for x in q) + " MWh")
    ints = {arm: [float(np.mean([r["mean_intensity_abs"] for r in by_key[(scn, arm)]]))
                  for scn in scenarios]
            for arm in ("dual_sac", strong, "sac_rule")}
    for arm in ints:
        lo, hi = rng(ints[arm])
        c.text(f"5.7 {arm} 强度区间", f"{lo:.3f}{DASH}{hi:.3f}")
    lead = [i for i in range(len(scenarios))
            if ints["dual_sac"][i] > max(ints[strong][i], ints["sac_rule"][i])]
    lag = [i for i in range(len(scenarios)) if i not in lead]
    c.text("5.7 dual 强度高于两条规则臂的场景",
           "、".join(scenarios[i][:1] for i in lead) + f" {len(lead)} 个场景高于两条规则臂")
    c.plain("未领先的场景与滚动规则强度之差 < 0.01 (正文才敢写基本持平)",
            all(abs(ints["dual_sac"][i] - ints[strong][i]) < 0.01 for i in lag),
            str([scenarios[i][:1] for i in lag]))
    for i in lag:
        c.text(f"5.7 dual 强度持平场景 {scenarios[i][:1]}",
               f"{scenarios[i][:1]} 场景 {ints['dual_sac'][i]:.3f} vs 滚动规则 {ints[strong][i]:.3f}")

    # 5.4-3 的分层占比来自 JSON (5.6 的 92%~99%/94.7% 来自 measure_reward_scale.py, 不同采样)
    share = [layer_share(by_key, scn, arm)[0] for scn in scenarios for arm in arms]
    lo, hi = rng(share)
    if args.verbose:
        print(f"  表 5.4-3 口径的中长期占比: {lo:.1f}% ~ {hi:.1f}%, "
              f"均值 {float(np.mean(share)):.1f}%")
    c.text("5.4-3 按世界均值口径标注", "按世界均值")

    # ---------- 8. 5.9 消融 (数据源: results/parts/ablation_*.json + 主基准场景 A 分片) ----------
    base_meta, base_a = load_base(args.json, S["A"])
    parts = sorted(glob.glob(args.parts))
    if len(parts) != len(ABL_ROWS):
        print(f"  [XX] 消融分片 {len(parts)} 个, 报告写了 {len(ABL_ROWS)} 个部件: {parts}")
        return 1
    re_abl, worlds_of, n_worlds, n_fail = {}, {}, 0, 0
    abl_data = {}
    for p in parts:
        v = os.path.basename(p)[len("ablation_"):-len(".json")]
        d = load(p)
        abl_data[v] = d
        part_lock = check_lock(d["meta"], p)
        c.n += 1
        if json_lock.get("files") and part_lock.get("files") != json_lock["files"]:
            c.bad += 1
            print(f"  [XX] {p}: 消融分片与主基准不是同一份代码, "
                  f"完整方法与变体的配对差无法解释")
        c.n += 1
        if d["meta"].get("ablation") != ABL_EXPECT[v]:
            c.bad += 1
            print(f'  [XX] {v}: meta.ablation 不是 "只关这一项" {d["meta"].get("ablation")}')
        re_abl[v] = {a["arm"]: a for a in summarize(v, p, base_meta, base_a)["arms"]}
        pu = penalty_units([w for r in d["runs"] for w in r["worlds"]])
        c.plain(f"消融 {v}: 惩罚单位链也按月电量 (与完整方法同口径才可比)",
                pu["worst_scale"] < 1e-6 and pu["worst_unit"] < 1e-6
                and abs(pu["ratio"] - pu["month_factor"]) < 1e-6 and pu["scales"] == u["scales"],
                f"残差 {pu['worst_scale']:.3g}/{pu['worst_unit']:.3g}, "
                f"倍数 {pu['ratio']:.4f} vs {pu['month_factor']:.4f}, "
                f"demand_scale {sorted(pu['scales'])} vs {sorted(u['scales'])}")
        worlds_of[v] = sum(len(r["worlds"]) for r in d["runs"])
        n_worlds += worlds_of[v]
        n_fail += len(d["failures"])
        pc = [w["checks"] for r in d["runs"] for w in r["worlds"] if "action_oob" in w["checks"]]
        c.plain(f"消融 {v}: 变体逐世界也带越界动作计数", len(pc) == worlds_of[v],
                f"{len(pc)}/{worlds_of[v]} 个世界有该字段")
        c.plain(f"消融 {v}: 变体的动作同样不越界 (可达集与完整方法一致才可比)",
                len(pc) == worlds_of[v] and sum(int(x["action_oob"]) for x in pc) == 0,
                f"{sum(int(x['action_oob']) for x in pc)} 次越界")

    def seeds_of(a):
        return [s["delta_pct"] for s in sorted(a["seeds"], key=lambda z: z["seed"])]

    def worse(a):
        return tuple(int(x) for x in a["worse_worlds"].split("/"))

    # 落盘的汇总表必须与当场重算一致 (防止 parts 重跑后 summary 变陈旧)
    summ = json.load(open(args.summary, encoding="utf-8"))
    FIELDS = ("cost", "cost_ref", "delta_pct", "alpha", "alpha_ref",
              "gamma", "gamma_ref", "intensity", "intensity_ref")
    for res in summ["results"]:
        for a in res["arms"]:
            ref = re_abl[res["variant"]][a["arm"]]
            c.n += 1
            bad = [f for f in FIELDS if abs(a[f] - ref[f]) > 1e-9]
            if bad:
                c.bad += 1
                print(f"  [XX] {args.summary} 的 {res['variant']}/{a['arm']} 字段与重算不符: {bad}")

    # 表 5.9-1 逐格重算
    for part, v in ABL_ROWS.items():
        for arm in LEARNERS:
            a, cells = re_abl[v].get(arm), md_row(md, part, arm)
            if a is None or not cells:
                c.n += 1
                if (a is not None) != bool(cells):
                    c.bad += 1
                    print(f"  [XX] 表 5.9-1 {part}/{arm}: 重算与正文一方缺失 "
                          f"(重算={a is not None}, 正文行={bool(cells)})")
                continue
            c.n += 1
            if len(cells) != 9:
                c.bad += 1
                print(f"  [XX] 表 5.9-1 行列数不为 9: {part}/{arm} -> {cells}")
                continue
            wn, nn = worse(a)
            c.value(f"5.9-1 {part}/{arm} 变体成本", a["cost"], nums(cells[2])[0], cells[2], 0.5)
            c.value(f"5.9-1 {part}/{arm} 配对差", a["delta_pct"], nums(cells[3])[0], cells[3], 0.005)
            c.value(f"5.9-1 {part}/{arm} 更贵世界数", float(wn), nums(cells[4])[0], cells[4], 0)
            c.value(f"5.9-1 {part}/{arm} 配对世界总数", float(nn), nums(cells[4])[1], cells[4], 0)
            for i, (comp, wri) in enumerate(zip(seeds_of(a), nums(cells[5]))):
                c.value(f"5.9-1 {part}/{arm} 第 {i + 1} 个种子 Δ", comp, wri, cells[5], 0.005)
            for field, cell in (("alpha", cells[6]), ("gamma", cells[7]), ("intensity", cells[8])):
                c.value(f"5.9-1 {part}/{arm} {field}", a[field], nums(cell)[0], cell, 5e-4)
                c.value(f"5.9-1 {part}/{arm} {field} (完整)",
                        a[field + "_ref"], nums(cell)[1], cell, 5e-4)

    # 正文与摘要/1.3/4.4/6.1 里的消融数字: 由重算值拼出字符串, 要求原文照此写着
    sh = [re_abl["no_exposure_shaping"][a] for a in LEARNERS]
    sp = [re_abl["no_spot_reward_scale"][a] for a in LEARNERS]
    al = re_abl["no_alternating"]["dual_sac"]
    for arm in LEARNERS:
        ref_cost = float(np.mean([r["mean_cost"] for (a, _), r in base_a.items() if a == arm]))
        c.text(f"5.9 参照成本 {arm}", f"`{arm}` {ref_cost:,.0f} 元")
    c.text("5.9 评估世界总数", f"{n_worlds} 个评估世界")
    c.text("5.9 各变体世界数", "（" + "+".join(str(worlds_of[v]) for v in ABL_ROWS.values()) + "）")
    c.text("5.9 验收失败条数", f"验收失败 **{n_fail} 条**")
    c.text("5.9 冻结预算", f"{base_meta['episodes_per_sac_arm']} 回合、"
                          f"{base_meta['eval_worlds']} 个配对世界")
    c.text("5.9 冻结种子", "/".join(str(s) for s in base_meta["seeds"]))
    c.text("5.9 冻结阶段划分", "+".join(f"{p * 100:.0f}%" for p in base_meta["phase_split"]))
    c.text("5.9 塑形代价 (摘要/结论一/6.1)", f"{sh[0]['delta_pct']:.2f}% 与 {sh[1]['delta_pct']:.2f}%")
    c.text("5.9 塑形代价区间 (1.3)",
           f"贵 {min(a['delta_pct'] for a in sh):.2f}%~{max(a['delta_pct'] for a in sh):.2f}%")
    c.text("5.9 结论三 另两项 Δ",
           f"{sp[0]['delta_pct']:+.2f}%/{sp[1]['delta_pct']:+.2f}% 与 {al['delta_pct']:+.2f}%")
    c.text("4.4 另两项 Δ",
           f"{sp[0]['delta_pct']:+.2f}%/{sp[1]['delta_pct']:+.2f}%（奖励放大）与 "
           f"{al['delta_pct']:+.2f}%（交替训练）")
    for sep in (" / ", "/"):      # 5.9 结论二用宽排, 6.1 用紧排
        c.text(f"结论二 γ 掉档 (sep={sep!r})",
               f"{sh[0]['gamma_ref']:.3f}{sep}{sh[1]['gamma_ref']:.3f} 掉到 "
               f"{sh[0]['gamma']:.3f}{sep}{sh[1]['gamma']:.3f}")
    c.text("结论二 α 掉档",
           f"{sh[0]['alpha_ref']:.3f} / {sh[1]['alpha_ref']:.3f} 掉到 "
           f"{sh[0]['alpha']:.3f} / {sh[1]['alpha']:.3f}")
    c.text("结论二 强度旁证 (sac_rule)", f"{sh[0]['intensity_ref']:.3f} 升到 {sh[0]['intensity']:.3f}")
    n_seed = len(seeds_of(sh[0]))
    c.text("结论一 种子同号", f"个配对世界更贵、{n_seed}/{n_seed} 种子同号")
    c.text("结论三 符号不一致的例子",
           "`dual_sac` 关奖励放大为 " + "/".join(f"{d:+.2f}" for d in seeds_of(sp[1])))

    # README 复述了 5.9 的 Δ 与 γ 掉档: 跨文档必须与重算一致, 不允许各说一套
    if os.path.exists(args.readme):
        cr = Checker(open(args.readme, encoding="utf-8").read())
        cr.text("README 塑形 γ 掉档",
                f"{sh[0]['gamma_ref']:.3f}/{sh[1]['gamma_ref']:.3f} 掉到 "
                f"{sh[0]['gamma']:.3f}/{sh[1]['gamma']:.3f}")
        cr.text("README 奖励量级比", f"{lo_r:.1f}~{hi_r:.1f}（均值 {mean_r:.1f}）")
        cr.text("README 中长期成本占比", f"占 {lo_s * 100:.0f}%~{hi_s * 100:.0f}%、均值 {mean_s * 100:.1f}%")
        for part, v in ABL_ROWS.items():
            line = next((l for l in cr.md.splitlines() if l.startswith(f"| {part} |")), "")
            cells = [x.strip() for x in line.strip().strip("|").split("|")] if line else []
            cr.n += 1
            if len(cells) != 4:
                cr.bad += 1
                print(f"  [XX] README 消融表 {part} 行缺失或列数不为 4: {cells}")
                continue
            for arm, col in (("sac_rule", 1), ("dual_sac", 2)):
                a = re_abl[v].get(arm)
                if a is None:
                    cr.n += 1
                    if cells[col] != "—":
                        cr.bad += 1
                        print(f"  [XX] README {part}/`{arm}` 没跑, 表里却写着 {cells[col]!r}")
                    continue
                wr = nums(cells[col])
                cr.value(f"README {part}/{arm} Δ", a["delta_pct"], wr[0], cells[col], 0.005)
                cr.value(f"README {part}/{arm} 更贵世界数", float(worse(a)[0]), wr[1],
                         cells[col], 0)
        c.n += cr.n
        c.bad += cr.bad
    else:
        # README 复述了同一批数字, 它缺失时"跨文档一致"就只是一句空话 (审计 N07)
        c.plain(f"{args.readme} 存在 (跨文档一致性的前提)", False,
                "README 不在预期路径上, 无法检查跨文档复述是否脱钩")
    ok1 = all(a["delta_pct"] > THRESHOLD and all(d > 0 for d in seeds_of(a))
              and worse(a)[0] == worse(a)[1] for a in sh)
    c.value("5.9 结论一: 塑形两臂 Δ>门槛、逐种子同号、世界全更贵", float(ok1), 1.0,
            "唯一通过判读门槛的部件", 0)
    ok3 = all(a["delta_pct"] < THRESHOLD and len({d > 0 for d in seeds_of(a)}) == 2
              and worse(a)[0] < worse(a)[1] for a in [sp[1], al])
    c.value("5.9 结论三: 双SAC 的另两项 Δ<门槛且逐种子符号不一致", float(ok3), 1.0,
            "逐种子符号不一致", 0)

    # 分层臂的现货层是规则, --no-spot-reward-scale 只改现货步奖励的分母
    # (src/market/environment.py:335-340), 因此它对 sac_rule 是构造性零结果, 不是"实测无差异"。
    # 这一点只能由数据自己承认: 逐世界成本必须与完整方法逐位相同。
    def wc(d, arm):
        return {(r["seed"], w["world"]): w["cost"]
                for r in d["runs"] if r["arm"] == arm for w in r["worlds"]}

    ref_sac = {(r["seed"], w["world"]): w["cost"]
               for (a, _), r in base_a.items() if a == "sac_rule" for w in r["worlds"]}
    var_sac = wc(abl_data["no_spot_reward_scale"], "sac_rule")
    diff_n = sum(1 for k in var_sac if ref_sac.get(k) != var_sac[k])
    c.plain("关掉现货放大后分层臂逐世界成本与完整方法逐位相同 (构造性零结果)",
            ref_sac.keys() == var_sac.keys() and not diff_n, f"{diff_n} 个世界不同")
    c.text("5.9 构造性零结果表述", "`sac_rule` 的逐世界成本与完整方法逐位相同")

    # 局限 4 的三条证据: 驱动脚本无 --no-* / 开关默认取反 / 主基准 JSON 自己写着三部件全开
    sh_txt = open("experiments/run_all_scenarios.sh", encoding="utf-8").read()
    c.value("5.9 局限4: 主基准调用不含任何 --no-* 开关", float("--no-" not in sh_txt), 1.0,
            "不含任何 `--no-*` 开关", 0)
    drv = open("experiments/run_four_arm_benchmark.py", encoding="utf-8").read()
    c.value("5.9 局限4: 驱动把三项消融开关默认置为开启",
            float(len(re.findall(r"not args\.no_(?:exposure_shaping|spot_reward_scale|alternating)",
                                 drv))), 3, "三项开关默认置为开启", 0)
    c.value("5.9 局限4: 主基准 JSON 自带三部件全开的开关记录",
            float(base_meta.get("ablation") == {"exposure_shaping": True,
                                                "scale_spot_reward": True,
                                                "alternating": True}), 1.0,
            "三部件全开", 0)

    # 交替训练确实改变了训练日程: 不再靠训练日志, 直接读分片里的 phase_history 与 grad_steps
    al_part = abl_data.get("no_alternating", {})
    al_ph = [[q["phase"] for q in r.get("phase_history") or []] for r in al_part.get("runs", [])
             if r.get("trained")]
    c.plain("no_alternating 的每个 SAC 组合只有一步 [P-joint] 阶段",
            bool(al_ph) and all(x == ["P-joint"] for x in al_ph),
            f"实测日程集合 {sorted({tuple(x) for x in al_ph})}")
    full_ph = [[q["phase"] for q in r.get("phase_history") or []] for r in data["runs"]
               if r.get("trained") and r["arm"] == "dual_sac"]
    c.plain("主基准 dual_sac 走 P1/P2/P3 三阶段",
            bool(full_ph) and all(x == ["P1-midlong", "P2-spot", "P3-joint"] for x in full_ph),
            f"{len(full_ph)} 个组合, 首个 {full_ph[:1]}")
    g4 = [al.get(k) for k in ("grad_midlong", "grad_spot", "grad_midlong_ref", "grad_spot_ref")]
    fresh = all(isinstance(x, int) and x > 0 for x in g4)
    c.plain("5.9 交替变体的逐角色优化器步数可从 JSON 复算", fresh,
            f"实测 {g4} —— 全为 0/缺失说明消融分片出自旧驱动, 需按当前代码重跑")
    if fresh:
        c.text("5.9 交替变体逐角色步数",
               f"完整方法 中长期 {g4[2]:.0f} 次 / 柔性 {g4[3]:.0f} 次，"
               f"关掉交替 {g4[0]:.0f} 次 / {g4[1]:.0f} 次")
        c.plain("关掉交替后逐角色步数确实变了 (不是只改名字的零结果)",
                g4[0] != g4[2] or g4[1] != g4[3],
                f"完整 {g4[2]}/{g4[3]}, 变体 {g4[0]}/{g4[1]}")
    check_doc_only(c, md, args.readme, args.completion)
    # 完成度评估报告复述了同一批数字: 三份文档必须同一口径, 不允许它单独漂移
    if os.path.exists(args.completion):
        cw = Checker(open(args.completion, encoding="utf-8").read())
        lo_ex, hi_ex = sorted(expl.values())
        for label, token in [
            ("完成度 双 DRL 相对滚动规则均值", f"{means[('dual_sac', strong)]:+.2f}%"),
            ("完成度 分层臂相对滚动规则均值", f"{means[('sac_rule', strong)]:+.2f}%"),
            ("完成度 双 DRL 相对固定比例均值", f"{means[('dual_sac', base)]:+.2f}%"),
            ("完成度 规则臂差距区间", f"{rb_lo:+.1f}%~{rb_hi:+.1f}%"),
            ("完成度 第二个智能体增量", f"{mean_ds:+.2f}%"),
            ("完成度 塑形两臂代价",
             f"{sh[0]['delta_pct']:.2f}% 与 {sh[1]['delta_pct']:.2f}%"),
            ("完成度 γ 掉档", f"{sh[0]['gamma_ref']:.3f}/{sh[1]['gamma_ref']:.3f} 掉到 "
                             f"{sh[0]['gamma']:.3f}/{sh[1]['gamma']:.3f}"),
            ("完成度 另两项 Δ",
             f"{sp[0]['delta_pct']:+.2f}%/{sp[1]['delta_pct']:+.2f}% 与 {al['delta_pct']:+.2f}%"),
            ("完成度 年度超配可解释区间", f"{lo_ex:.0f}%~{hi_ex:.0f}%"),
            ("完成度 奖励量级比均值", f"均值 {mean_r:.1f}"),
            ("完成度 中长期成本占比", f"占 {lo_s * 100:.0f}%~{hi_s * 100:.0f}%、均值 {mean_s * 100:.1f}%"),
        ]:
            cw.text(label, token)
        cw.text("完成度 评估世界数", f"{len(worlds)} 个评估世界")
        cw.text("完成度 消融评估世界数", f"{n_worlds} 个评估世界")
        c.n += cw.n
        c.bad += cw.bad
    else:
        print(f"  [!] 未找到 {args.completion}, 完成度文档的一致性检查本轮跳过")
    # "这次审计共检查多少项主张"只能由审计器自己说了算 (审计 N07): 报告/README/完成度评估各复述一次
    docs = [p for p in (args.md, args.readme, args.completion) if os.path.exists(p)]
    token = f"{c.n + len(docs)} 项主张"
    c.n += len(docs)
    for path in docs:
        txt = open(path, encoding="utf-8").read().replace("\u2212", "-")
        if token not in txt:
            c.bad += 1
            print(f"  [XX] {path} 的检查项数: 应写着 {token!r}, 实际找不到")
    if args.verbose:
        print(f"  5.9 重算 Δ: 塑形 {sh[0]['delta_pct']:.3f}%/{sh[1]['delta_pct']:.3f}%, "
              f"奖励放大 {sp[0]['delta_pct']:.3f}%/{sp[1]['delta_pct']:.3f}%, "
              f"交替 {al['delta_pct']:.3f}%; 世界 {n_worlds}, 失败 {n_fail}")
    return c.finish()


if __name__ == "__main__":
    raise SystemExit(main())
