"""
四臂配对世界基准 (审计建议 5: 重建可信实验)

四个策略臂, 同一套环境、同一批配对世界、同一组设备任务:
    fixed_rule   固定比例 + 保障调度      (不学习, 不看价格)
    rolling_rule 滚动规则 + 预测调度      (不学习, 只用已发布信息的比价规则)
    sac_rule     上层 SAC + 规则下层调度  (只训中长期 SAC)
    dual_sac     双 SAC (中长期 + 柔性调度)

冻结的预算与随机性 (meta 里全部记录):
    --seeds      训练随机种子与市场价格世界组的编号 (同一 seed 下四臂共用同一批世界)
    --episodes   SAC 臂的训练回合数 (sac_rule 与 dual_sac 相同)
    --eval-worlds 评价用配对世界数, 四臂逐同一批世界比较

每个世界都记录: 成本、成本/电量分解、α/β/γ 与调度强度序列、设备校验、
以及会计验收结论 (电量守恒 / 无虚假退款 / 合同量非负 / 费用可复算)。
凡不满足的写入 failures 列表, 不进入结果汇总。

用法:
    python experiments/run_four_arm_benchmark.py --smoke          # 快速自检
    python experiments/run_four_arm_benchmark.py                  # 5 场景 x 4 臂 x 3 种子
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.multi_agent import MultiScaleCoordinator
from src.agent.rule_policies import (
    STAGE_SPOT, RulePolicy, fixed_ratio_action, forecast_intensity, rolling_rule_action,
)
from src.market.environment import TradingEnvironment
from src.utils.config import PRICE_SCENARIOS
from src.utils.logger import setup_logger

logger = setup_logger("four_arm")

SCENARIO_KEYS = list(PRICE_SCENARIOS.keys())
ARMS = ("fixed_rule", "rolling_rule", "sac_rule", "dual_sac")
BASELINE = "fixed_rule"

# SAC 臂三阶段交替训练的比例 (phase1 只训中长期 / phase2 只训调度 / phase3 联合)
PHASE_SPLIT = (0.25, 0.25, 0.50)
NUM_STEPS_PER_EPISODE = 37


# ============================================================
# 策略臂封装
# ============================================================
class SacArm:
    """
    SAC 臂。spot_from_rule=True 时现货强度改由预测调度规则给出 (分层臂),
    中长期仍由 SAC 输出; 此时柔性调度 SAC 全程冻结, 不写经验。
    """

    def __init__(self, coord: MultiScaleCoordinator, spot_from_rule: bool = False):
        self.coord = coord
        self.spot_from_rule = spot_from_rule

    def act(self, obs, stage, deterministic=False):
        action = self.coord.select_action(obs, stage, deterministic=deterministic)
        if self.spot_from_rule and stage == STAGE_SPOT:
            action = np.array([0.0, forecast_intensity(obs)], dtype=np.float32)
        return action

    def begin_episode(self):
        self.coord.begin_episode()

    def store(self, obs, action, info, next_obs, done, stage):
        self.coord.store_transition(
            obs, action,
            {"reward_midlong": info["reward_midlong"], "reward_spot": info["reward_spot"]},
            next_obs, done, stage=stage,
        )

    def update(self, stage):
        return self.coord.update(stage=stage)

    def set_phase(self, freeze_midlong: bool, freeze_spot: bool):
        self.coord.set_training_phase(freeze_midlong=freeze_midlong, freeze_spot=freeze_spot)


def make_arm(arm: str, state_dim: int, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if arm == "fixed_rule":
        return RulePolicy(fixed_ratio_action)
    if arm == "rolling_rule":
        return RulePolicy(rolling_rule_action)
    coord = MultiScaleCoordinator(
        state_dim=state_dim, use_dual_agent=True,
        midlong_action_scale=5.0, spot_action_scale=1.0,
    )
    if arm == "sac_rule":
        return SacArm(coord, spot_from_rule=True)
    if arm == "dual_sac":
        return SacArm(coord, spot_from_rule=False)
    raise ValueError(f"未知策略臂: {arm}")


# ============================================================
# 回合执行
# ============================================================
def rollout(env, arm, world=None, deterministic=True, learn=False):
    """
    跑一个回合。learn=True 时写入经验并更新 (训练); 否则纯评价。
    返回逐世界记录 (含会计验收结论)。
    """
    obs, _ = env.reset(options=None if world is None else {"world": world})
    arm.begin_episode()
    alphas, betas, gammas, intensities = [], [], [], []
    reward_mid, reward_spot = 0.0, 0.0

    while True:
        stage = env.current_stage
        action = arm.act(obs, stage, deterministic=deterministic)
        next_obs, _, terminated, truncated, info = env.step(action)
        if learn:
            arm.store(obs, action, info, next_obs, terminated, stage)
            arm.update(stage)
        if info["stage"] == "annual":
            alphas.append(info["alpha"])
        elif info["stage"] == "monthly":
            betas.append(info["beta"])
        elif info["stage"] == "intraday":
            gammas.append(info["gamma"])
        else:
            intensities.append(info["intensity"])
        reward_mid += info["reward_midlong"]
        reward_spot += info["reward_spot"]
        obs = next_obs
        if terminated or truncated:
            break

    checks = _accounting_checks(env, info)
    return {
        "world": None if env.current_world is None else int(env.current_world),
        "cost": float(info["total_cost"]),
        "market_cost": float(info["market_cost"]),
        "flexible_penalty": float(info["flexible_penalty"]),
        "flexible_saving": float(info.get("flexible_cost_saving", 0.0)),
        "cost_breakdown": {k: float(v) for k, v in info["cost_breakdown"].items()},
        "energy_breakdown": {k: float(v) for k, v in env.settlement.get_energy_breakdown().items()},
        "alphas": [float(v) for v in alphas],
        "betas": [float(v) for v in betas],
        "gammas": [float(v) for v in gammas],
        "intensities": [float(v) for v in intensities],
        "device_task_met": bool(info["device_task_met"]),
        "steps": int(info["step"]),
        "reward_midlong_sum": float(reward_mid),
        "reward_spot_sum": float(reward_spot),
        "checks": checks,
    }


def _accounting_checks(env, info) -> dict:
    """审计最低门槛的逐回合复核 (不满足即记入 failures)。"""
    se = env.settlement
    ledger = se.get_ledger()
    actual = (sum(c["day_energy_mwh"] for c in info["device_checks"])
              * env.days_per_month * env.demand_scale)
    contract_total = sum(slot["contract"] for slot in se.get_monthly_closure().values())
    energy = se.get_energy_breakdown()
    contract_costs = [r for r in ledger if r["stage"] != "spot"]
    return {
        "steps_37": int(info["step"]) == NUM_STEPS_PER_EPISODE,
        # 电量守恒: 台账成交量 = 实际网侧净需求
        "energy_residual_mwh": float(abs(sum(r["quantity"] for r in ledger) - actual)),
        "spot_matches_residual": float(abs(energy["spot"] - (actual - contract_total))) < 1e-6,
        # 无虚假退款: 合同层电量与成本都不得为负 (只有现货富余可回售)
        "contract_qty_negative": int(sum(1 for r in contract_costs if r["quantity"] < -1e-9)),
        "contract_cost_negative": int(sum(1 for r in contract_costs if r["cost"] < -1e-9)),
        # 费用可复算: 分项之和 = 总成本
        "cost_recomputed_residual": float(abs(
            sum(info["cost_breakdown"][k] for k in ("annual", "monthly", "intraday", "spot"))
            - info["market_cost"])),
        "device_task_met": bool(info["device_task_met"]),
        "finite_cost": bool(np.isfinite(info["total_cost"])),
    }


def check_failed(checks: dict) -> list:
    bad = []
    if not checks["steps_37"]:
        bad.append("steps_37")
    if checks["energy_residual_mwh"] > 1e-6:
        bad.append("energy_conservation")
    if not checks["spot_matches_residual"]:
        bad.append("spot_residual")
    if checks["contract_qty_negative"] or checks["contract_cost_negative"]:
        bad.append("false_refund")
    if checks["cost_recomputed_residual"] > 1e-6:
        bad.append("cost_not_recomputable")
    if not checks["device_task_met"]:
        bad.append("device_task_unmet")
    if not checks["finite_cost"]:
        bad.append("nonfinite_cost")
    return bad


# ============================================================
# 训练
# ============================================================
def train_sac_arm(env, arm, episodes, log_interval=200, dual=True):
    """SAC 臂按冻结相位训练; 返回相位记录 (供 meta 复现)。"""
    history = []
    if dual:
        bounds = [int(episodes * PHASE_SPLIT[0]), int(episodes * (PHASE_SPLIT[0] + PHASE_SPLIT[1]))]
        phases = [("P1-midlong", 0, bounds[0], False, True),
                  ("P2-spot", bounds[0], bounds[1], True, False),
                  ("P3-joint", bounds[1], episodes, False, False)]
    else:
        phases = [("P-midlong", 0, episodes, False, True)]   # 下层是规则, 调度 SAC 全程冻结

    done_eps = 0
    for name, start, end, fm, fs in phases:
        arm.set_phase(freeze_midlong=fm, freeze_spot=fs)
        for _ in range(end - start):
            done_eps += 1
            rec = rollout(env, arm, world=None, deterministic=False, learn=True)
            if done_eps % log_interval == 0:
                logger.info(f"  [{name}] ep={done_eps}/{episodes} cost={rec['cost']:.0f}")
        history.append({"phase": name, "episodes": end - start,
                        "freeze_midlong": fm, "freeze_spot": fs})
    return history


# ============================================================
# 单臂单场景单种子
# ============================================================
def run_one(scenario, arm_name, seed, episodes, eval_worlds, load_scenario,
            model_dir, do_train=True):
    env = TradingEnvironment(scenario_name=scenario, load_scenario=load_scenario,
                             use_rule_dispatcher=False, seed=seed)
    obs, _ = env.reset()
    arm = make_arm(arm_name, state_dim=env.observation_space.shape[0], seed=seed)

    trainable = arm_name in ("sac_rule", "dual_sac")
    phase_history = []
    if trainable and do_train:
        t0 = time.time()
        phase_history = train_sac_arm(env, arm, episodes, dual=(arm_name == "dual_sac"))
        logger.info(f"  训练耗时 {time.time()-t0:.0f}s | {scenario[:2]}/{arm_name}/seed{seed}")
        os.makedirs(model_dir, exist_ok=True)
        arm.coord.save(os.path.join(model_dir, f"{scenario[:1]}_{arm_name}_seed{seed}"))

    worlds = []
    failures = []
    for w in range(eval_worlds):
        rec = rollout(env, arm, world=w, deterministic=True, learn=False)
        bad = check_failed(rec["checks"])
        if bad:
            failures.append({"scenario": scenario, "arm": arm_name, "seed": seed,
                             "world": w, "failed_checks": bad, "checks": rec["checks"]})
            continue
        worlds.append(rec)

    costs = [r["cost"] for r in worlds]
    return {
        "scenario": scenario, "arm": arm_name, "seed": seed,
        "trained": bool(trainable and do_train),
        "episodes": episodes if trainable and do_train else 0,
        "phase_history": phase_history,
        "n_worlds_passed": len(worlds),
        "mean_cost": float(np.mean(costs)) if costs else None,
        "std_cost": float(np.std(costs)) if costs else None,
        "worst_cost": float(np.max(costs)) if costs else None,
        "mean_flexible_saving": float(np.mean([r["flexible_saving"] for r in worlds])) if worlds else None,
        "mean_gamma": float(np.mean([np.mean(r["gammas"]) for r in worlds])) if worlds else None,
        "mean_intensity_abs": float(np.mean([np.mean(np.abs(r["intensities"])) for r in worlds])) if worlds else None,
        "worlds": worlds,
    }, failures


# ============================================================
# 汇总
# ============================================================
def summarize(runs):
    """按 (场景, 臂) 跨种子平均, 再对基线臂算配对改进 (同种子同世界)。"""
    by_key = {}
    for r in runs:
        by_key.setdefault((r["scenario"], r["seed"]), {})[r["arm"]] = r

    table = {}
    for (scenario, seed), arms in by_key.items():
        base = arms.get(BASELINE)
        if base is None or not base["worlds"]:
            continue
        base_by_world = {w["world"]: w["cost"] for w in base["worlds"]}
        for arm_name, r in arms.items():
            pairs = [(w["cost"], base_by_world[w["world"]]) for w in r["worlds"]
                     if w["world"] in base_by_world]
            if not pairs:
                continue
            imp = [(b - d) / b * 100 for d, b in pairs]
            wins = sum(1 for d, b in pairs if d < b - 1e-9)
            entry = table.setdefault((scenario, arm_name), {"improvements": [], "wins": [],
                                                            "costs": [], "seed_count": 0})
            entry["improvements"].append(float(np.mean(imp)))
            entry["wins"].append(f"{wins}/{len(pairs)}")
            entry["costs"].append(r["mean_cost"])
            entry["seed_count"] += 1
    return table


def print_table(table, arms):
    print("\n" + "=" * 78)
    print(f"{'场景':<26}{'策略臂':<14}{'平均成本':>12}{'对基线改进':>12}{'逐世界胜出':>14}")
    print("-" * 78)
    for scenario in SCENARIO_KEYS:
        for arm_name in arms:
            e = table.get((scenario, arm_name))
            if e is None:
                continue
            cost = np.mean(e["costs"])
            imp = np.mean(e["improvements"])
            wins = "; ".join(e["wins"])
            print(f"{scenario:<26}{arm_name:<14}{cost:>12,.0f}{imp:>+11.1f}%{wins:>14}")
    print("=" * 78)


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="四臂配对世界基准")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="逗号分隔的场景首字母 (如 A,C); 默认全部 5 个")
    parser.add_argument("--arms", type=str, default=",".join(ARMS))
    parser.add_argument("--seeds", type=str, default="42,142,242")
    parser.add_argument("--load", type=str, default="柔性适中")
    parser.add_argument("--episodes", type=int, default=2000, help="SAC 臂训练回合数")
    parser.add_argument("--eval-worlds", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="冒烟: 2 场景 x 1 种子 x 200 回合")
    parser.add_argument("--no-train", action="store_true", help="只跑两个规则臂")
    parser.add_argument("--out", type=str, default="results/four_arm_benchmark.json")
    parser.add_argument("--model-subdir", type=str, default="four_arm")
    args = parser.parse_args()

    if args.smoke:
        scenarios = [k for k in SCENARIO_KEYS if k.startswith("A")]
        seeds, episodes, eval_worlds = [42], 200, 4
    elif args.scenarios:
        wanted = [s.strip().upper() for s in args.scenarios.split(",")]
        scenarios = [k for k in SCENARIO_KEYS if k[0] in wanted]
        seeds = [int(s) for s in args.seeds.split(",")]
        episodes, eval_worlds = args.episodes, args.eval_worlds
    else:
        scenarios, seeds = SCENARIO_KEYS, [int(s) for s in args.seeds.split(",")]
        episodes, eval_worlds = args.episodes, args.eval_worlds

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    if args.no_train:
        arms = [a for a in arms if a in ("fixed_rule", "rolling_rule")]
    model_dir = os.path.join("results", "models", args.model_subdir)

    logger.info(f"四臂基准 | 场景 {len(scenarios)} | 臂 {arms} | 种子 {seeds} "
                f"| episodes={episodes} | eval_worlds={eval_worlds}")

    runs, failures = [], []
    for scenario in scenarios:
        for seed in seeds:
            for arm_name in arms:
                run, fails = run_one(scenario, arm_name, seed, episodes, eval_worlds,
                                     args.load, model_dir, do_train=not args.no_train)
                runs.append(run)
                failures.extend(fails)
                if run["mean_cost"] is None:
                    logger.warning(f"[{scenario[:2]}|{arm_name}|seed{seed}] 无世界通过验收 "
                                   f"({len(fails)} 条失败记录)")
                else:
                    logger.info(f"[{scenario[:2]}|{arm_name}|seed{seed}] "
                                f"cost={run['mean_cost']:,.0f} ±{run['std_cost']:,.0f} "
                                f"| 通过世界 {run['n_worlds_passed']}/{eval_worlds} "
                                f"| γ均值={run['mean_gamma']:.3f} "
                                f"| |强度|={run['mean_intensity_abs']:.2f}")

    table = summarize(runs)
    print_table(table, arms)

    payload = {
        "meta": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenarios": scenarios, "arms": arms, "seeds": seeds,
            "load_scenario": args.load, "episodes_per_sac_arm": episodes,
            "eval_worlds": eval_worlds, "phase_split": PHASE_SPLIT,
            "baseline_arm": BASELINE, "smoke": bool(args.smoke),
            "note": "改进% 为同种子、同世界、同设备任务下相对 fixed_rule 臂的配对差异; "
                    "会计验收未通过的回合不进入汇总, 记录在 failures",
        },
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
    logger.info(f"结果已写入 {args.out} | 验收失败记录 {len(failures)} 条")
    if failures:
        logger.warning("存在会计/设备验收失败的回合, 结果不可用于结论")


if __name__ == "__main__":
    main()
