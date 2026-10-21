"""审计器的反向用例: 故意把文档或数据改错, 确认 `verify_report_numbers.py` 会拒绝。

审计"0 项不符"只有在这个检查器**能**抓到错数时才有意义, 所以每条变异都是一次反向用例:
把正文里某个数改掉 (或删掉一行), 检查器必须返回非零并指出是哪一项;
把落盘 JSON 里的惩罚口径/设备结论/世界编号/源锁改掉, 同样必须被拒。
任何一条被漏掉, 就说明对应的检查是空转的 —— 那正是本项目在旧口径上吃过亏的地方。

变异同时打到报告、README (后者复述了 5.9 的消融数字) 与 results/four_arm_benchmark.json。

用法:
    python experiments/selfcheck_verifier.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.verify_report_numbers as V                                  # noqa: E402
from src.utils.config import INDUSTRIAL_LOAD_PARAMS                            # noqa: E402

FILES = {"报告": "report/课程设计报告.md", "README": "README.md"}
DATA = "results/four_arm_benchmark.json"


def drop_line(text: str, pat: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith(pat))


def worlds(payload: dict) -> list:
    return [w for r in payload["runs"] for w in r["worlds"]]


# (标签, 打到哪份文档, 变异) —— 变异必须真的动到字符, 否则这条用例是空转的
MUTATIONS = [
    ("5.3 跨场景均值 +0.69% → +0.79%", "报告", lambda t: t.replace("+0.69%", "+0.79%")),
    ("验收 600 个评估世界 → 599", "报告", lambda t: t.replace("600 个评估世界", "599 个评估世界")),
    ("5.4.1 α 饱和例外 67% → 70%", "报告", lambda t: t.replace("（67%）", "（70%）")),
    ("5.7 E 剖面 0.966–0.997 → 0.960–0.997", "报告",
     lambda t: t.replace("0.966–0.997", "0.960–0.997")),
    ("5.4.2 加粗的 **30/30** → **29/30**", "报告", lambda t: t.replace("**30/30**", "**29/30**")),
    ("5.9 参照成本 309,116 → 309,117", "报告", lambda t: t.replace("309,116", "309,117")),
    ("表 5.9-1 变体成本 314,374 → 314,274", "报告", lambda t: t.replace("314,374", "314,274")),
    ("表 5.9-1 配对差 +1.70% → +1.07%", "报告", lambda t: t.replace("**+1.70%**", "**+1.07%**")),
    ("表 5.9-1 逐种子符号 −0.27 → +0.27", "报告",
     lambda t: t.replace("\u22120.27 / +0.16 / +0.31", "+0.27 / +0.16 / +0.31")),
    ("表 5.9-1 删掉交替训练整行", "报告", lambda t: drop_line(t, "| 三阶段交替训练 |")),
    ("5.9 结论二 γ 0.848 → 0.849", "报告", lambda t: t.replace("0.875 / 0.848", "0.875 / 0.849")),
    ("1.3 塑形代价区间 1.70% → 1.90%", "报告", lambda t: t.replace("贵 1.40%~1.70%", "贵 1.40%~1.90%")),
    ("摘要 塑形代价 1.40% → 1.50%", "报告", lambda t: t.replace("1.70% 与 1.40%", "1.70% 与 1.50%")),
    ("README 复述的 Δ 与重算脱钩 +0.26% → +0.96%", "README",
     lambda t: t.replace("+0.26% (24/30)", "+0.96% (24/30)")),
    ("README 消融表补写一个没跑过的组合", "README",
     lambda t: t.replace("| 三阶段交替训练 | — |", "| 三阶段交替训练 | +0.05% (22/30) |")),
    ("代码行号引用漂到另一段 (行号仍有效, 但那段没有 CVaR)", "报告",
     lambda t: t.replace("src/market/environment.py:354-359", "src/market/environment.py:120-125")),
    ("5.2 验收用例数 58 → 57", "报告", lambda t: t.replace("58 个验收用例", "57 个验收用例")),
    ("README 的用例数与 test 文件脱钩 58 → 60", "README",
     lambda t: t.replace("58 个验收用例", "60 个验收用例")),
]


# 数据侧变异 (审计 N07): 落盘 JSON 被改动时, 鉴真检查必须拒绝, 而不是只信文件里的数
def _penalty_back_to_day_scale(payload):
    """迭代 8.11 前的错法: 惩罚按单个代表日计, 电量却按整月结算。"""
    for w in worlds(payload):
        w["flexible_penalty"] = w["transferred_mwh_day"] * INDUSTRIAL_LOAD_PARAMS["transfer_cost"]
    return payload


def _energy_back_to_day_scale(payload):
    """转移电量按代表日落档: 单价与惩罚仍自洽, 只有尺度因子检查能抓到。"""
    for w in worlds(payload):
        w["transferred_mwh_total"] = w["transferred_mwh_day"]
    return payload


def _flip_device_flag(payload):
    w = worlds(payload)[123 % len(worlds(payload))]
    w["device_task_met"] = not w["checks"]["device_task_met"]
    return payload


def _duplicate_world(payload):
    r = payload["runs"][0]
    r["worlds"].append(copy.deepcopy(r["worlds"][0]))
    return payload


def _break_energy_closure(payload):
    w = worlds(payload)[3]
    w["energy_breakdown"]["total"] *= 1.001
    return payload


def _forge_lock_hash(payload):
    payload["meta"]["source_lock"]["files"]["src/market/settlement.py"] = "deadbeefdeadbeef0000"
    return payload


def _drop_lock(payload):
    payload["meta"].pop("source_lock", None)
    return payload


def _let_actions_go_out_of_bounds(payload):
    """审计 N02 声称"四臂动作都不越界": 把一次越界藏进 checks, 必须被拒。"""
    worlds(payload)[7]["checks"]["action_oob"] = 1
    return payload


def _forge_phase_history(payload):
    """把某个 dual_sac 组合的训练日程改成全程联合 —— 与"三阶段"记录矛盾。"""
    r = next(x for x in payload["runs"] if x["arm"] == "dual_sac" and x.get("trained"))
    r["phase_history"] = [{"phase": "P-joint", "episodes": 2000,
                           "freeze_midlong": False, "freeze_spot": False}]
    return payload


def _drop_grad_steps(payload):
    """删掉逐角色优化器步数: 5.9 关于"交替确实改变了更新次数"的主张就没有数据来源。"""
    r = next(x for x in payload["runs"] if x["arm"] == "dual_sac" and x.get("trained"))
    r["grad_steps"] = {}
    return r


def _switch_says_ablated(payload):
    """主基准的 meta.ablation 里出现关闭项: 参照就不再是"三部件全开"。"""
    payload["meta"]["ablation"]["alternating"] = False
    return payload


DATA_MUTATIONS = [
    ("惩罚退回单个代表日口径 (作废算法的特征)", _penalty_back_to_day_scale),
    ("转移电量按代表日落档 (单价自洽但尺度错)", _energy_back_to_day_scale),
    ("某世界的设备结论与 checks 里的同一字段相反", _flip_device_flag),
    ("同一评估世界在一个组合里被记两次", _duplicate_world),
    ("某世界年成交电量偏 0.1%, 不再闭合于 Q", _break_energy_closure),
    ("源锁里 settlement.py 的哈希被改", _forge_lock_hash),
    ("整段源锁被删掉, 数字无法追溯代码版本", _drop_lock),
    ("把一次动作越界藏进 checks (可达集主张)", _let_actions_go_out_of_bounds),
    ("dual_sac 的训练日程被改写成全程联合", _forge_phase_history),
    ("删掉逐角色优化器步数记录", _drop_grad_steps),
    ("主基准 meta.ablation 出现关闭项", _switch_says_ablated),
]


def run(texts: dict, paths: dict, payload: dict = None) -> tuple:
    """把 (可能变异的) 文档与数据写到临时文件, 用审计器的真实入口跑一遍。"""
    for key, path in paths.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(texts[key])
    sys.argv = ["selfcheck", "--md", paths["报告"], "--readme", paths["README"]]
    if payload is not None:
        with open(paths["数据"], "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        sys.argv += ["--json", paths["数据"]]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = V.main()
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return rc, (lines[0] if lines else "(无输出)")


def _judge(label: str, rc: int, first: str, missed: list) -> None:
    hit = bool(re.search(r"\[XX\]|Traceback|Error", first))
    ok = rc == 1 and hit
    print(f"  [{'抓到' if ok else '漏掉':<4}] {label:<40} rc={rc}  {first[:66]}")
    if not ok:
        missed.append(label)


def main() -> int:
    tmp = tempfile.gettempdir()
    paths = {"报告": os.path.join(tmp, "_selfcheck_报告.md"),
             "README": os.path.join(tmp, "_selfcheck_README.md"),
             "数据": os.path.join(tmp, "_selfcheck_data.json")}
    base = {k: open(v, encoding="utf-8").read() for k, v in FILES.items()}
    base_data = json.load(open(DATA, encoding="utf-8"))
    frozen = json.dumps(base_data, sort_keys=True, ensure_ascii=False)

    rc, first = run(base, paths, base_data)
    print(f"基准 (未变异): rc={rc}  {first[:90]}")
    if rc != 0:
        print("文档/数据本身就没通过审计, 先修正再跑自检")
        return 1

    missed = []
    for label, key, mut in MUTATIONS:
        texts = dict(base)
        if mut(texts[key]) == texts[key]:
            print(f"  [空转] {label}: 变异没改到任何字符, 字符串写错了")
            missed.append(label)
            continue
        texts[key] = mut(texts[key])
        _judge(label, *run(texts, paths, base_data), missed=missed)

    for label, mut in DATA_MUTATIONS:
        payload = mut(copy.deepcopy(base_data))
        if json.dumps(payload, sort_keys=True, ensure_ascii=False) == frozen:
            print(f"  [空转] {label}: 变异没改到任何字段")
            missed.append(label)
            continue
        _judge(label, *run(base, paths, payload), missed=missed)

    n = len(MUTATIONS) + len(DATA_MUTATIONS)
    print(f"\n{n} 条变异 ({len(MUTATIONS)} 条打文档, {len(DATA_MUTATIONS)} 条打数据), "
          f"漏掉 {len(missed)} 条"
          + (f": {missed}" if missed else " (检查器对每条错数都会失败)"))
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
