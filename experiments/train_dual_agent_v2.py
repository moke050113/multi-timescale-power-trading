"""
双 DRL 智能体三阶段交替训练 (B 方案)

Phase 1 (0 ~ N1):     训练 midlong_agent, 冻结 spot_agent (spot 随机输出)
Phase 2 (N1 ~ N1+N2): 冻结 midlong_agent (使用 phase1 学到的策略), 训练 spot_agent
Phase 3 (... ~ N):    两个智能体同时训练 (联合微调)

超参:
    phase1_episodes = 500
    phase2_episodes = 500
    phase3_episodes = 1000
    总计            = 2000

用法:
    python experiments/train_dual_agent_v2.py --scenario A --episodes 500   # 冒烟
    python experiments/train_dual_agent_v2.py --all                          # 5 场景完整
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import PRICE_SCENARIOS, DRL_PARAMS
from src.utils.logger import setup_logger

logger = setup_logger("train_dual_v2")

# 评价用固定世界编号: 所有策略 (含固定比例基准) 都在同一批价格实现上评价
NUM_EVAL_WORLDS = 5
NUM_FINAL_WORLDS = 10


# ============================================================
# 训练原语
# ============================================================
def _run_episode(env, coordinator, deterministic=False, store_and_update=True, world=None):
    """执行一个 episode, 返回 (cost, reward_m_sum, reward_s_sum, flex_saving)"""
    state, _ = env.reset(options=None if world is None else {"world": world})
    coordinator.begin_episode()
    total_reward = 0.0
    r_m_sum = 0.0
    r_s_sum = 0.0

    while True:
        stage_before = env.current_stage
        action = coordinator.select_action(state, stage_before, deterministic=deterministic)
        next_state, _, terminated, truncated, info = env.step(action)

        if store_and_update:
            rewards = {
                "reward_midlong": info["reward_midlong"],
                "reward_spot": info["reward_spot"],
            }
            coordinator.store_transition(
                state, action, rewards, next_state, terminated, stage=stage_before
            )
            coordinator.update(stage=stage_before)

        total_reward += info["reward_global"]
        r_m_sum += info["reward_midlong"]
        r_s_sum += info["reward_spot"]
        state = next_state
        if terminated or truncated:
            break

    return {
        "cost": info["total_cost"],
        "market_cost": info["market_cost"],
        "flex_saving": info.get("flexible_cost_saving", 0.0),
        "flex_penalty": info.get("flexible_penalty", 0.0),
        "reward_total": total_reward,
        "r_midlong_sum": r_m_sum,
        "r_spot_sum": r_s_sum,
        "positions": info["positions"],
        "energy": env.settlement.get_energy_breakdown(),
        "world": info["world"],
    }


def _fixed_costs_per_world(env, num_worlds: int = NUM_EVAL_WORLDS):
    """固定比例基准在各评价世界上的成本 (与 DRL 评价使用同一批价格实现)"""
    return [float(env.get_fixed_ratio_cost(world=w)["total_cost"]) for w in range(num_worlds)]


def _evaluate(env, coordinator, num_eval: int = NUM_EVAL_WORLDS):
    """确定性策略评估: 在世界 0..num_eval-1 上逐一对比, 返回逐世界成本与柔性节约"""
    costs = []
    savings = []
    for w in range(num_eval):
        r = _run_episode(env, coordinator, deterministic=True, store_and_update=False, world=w)
        costs.append(r["cost"])
        savings.append(r["flex_saving"])
    return costs, savings


# ============================================================
# 三阶段主训练
# ============================================================
def train_dual_agent(
    scenario: str = "A_年度低价后期上涨",
    load_scenario: str = "柔性适中",
    phase1_episodes: int = 500,
    phase2_episodes: int = 500,
    phase3_episodes: int = 1000,
    print_interval: int = 100,
    eval_interval: int = 200,
    seed: int = 42,
    save_prefix: str = None,
    scale_spot_reward: bool = True,
    exposure_shaping: bool = True,
    alternating: bool = True,
):
    total_episodes = phase1_episodes + phase2_episodes + phase3_episodes
    logger.info(f"[双智能体 B方案] {scenario} | {load_scenario} | total={total_episodes}")

    env = TradingEnvironment(
        scenario_name=scenario,
        load_scenario=load_scenario,
        use_rule_dispatcher=False,
        scale_spot_reward=scale_spot_reward,
        exposure_shaping=exposure_shaping,
        seed=seed,
    )
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]

    coordinator = MultiScaleCoordinator(
        state_dim=state_dim,
        use_dual_agent=True,
        midlong_action_scale=5.0,
        spot_action_scale=1.0,
    )

    fixed_mid = _fixed_costs_per_world(env, NUM_EVAL_WORLDS)
    fixed_final = _fixed_costs_per_world(env, NUM_FINAL_WORLDS)
    fixed_cost_mid = float(np.mean(fixed_mid))
    fixed_cost = float(np.mean(fixed_final))
    logger.info(
        f"固定比例基准: 中期({NUM_EVAL_WORLDS}世界)={fixed_cost_mid:.0f} "
        f"| 最终({NUM_FINAL_WORLDS}世界)={fixed_cost:.0f}"
    )

    history = []
    ep_counter = 0

    def _phase_loop(phase_name, n_eps, freeze_midlong, freeze_spot):
        nonlocal ep_counter
        coordinator.set_training_phase(
            freeze_midlong=freeze_midlong, freeze_spot=freeze_spot
        )
        logger.info(
            f"[{phase_name}] episodes {ep_counter+1} ~ {ep_counter+n_eps} "
            f"| freeze_midlong={freeze_midlong}, freeze_spot={freeze_spot}"
        )
        for _ in range(n_eps):
            ep_counter += 1
            r = _run_episode(env, coordinator)
            history.append({
                "episode": ep_counter,
                "phase": phase_name,
                "cost": r["cost"],
                "flex_saving": r["flex_saving"],
                "reward_midlong": r["r_midlong_sum"],
                "reward_spot": r["r_spot_sum"],
            })

            if ep_counter % print_interval == 0:
                recent = history[-print_interval:]
                avg_cost = np.mean([h["cost"] for h in recent])
                avg_rm = np.mean([h["reward_midlong"] for h in recent])
                avg_rs = np.mean([h["reward_spot"] for h in recent])
                logger.info(
                    f"Ep {ep_counter:5d} [{phase_name}] | cost={avg_cost:10.0f} "
                    f"| R_mid={avg_rm:8.2f} | R_spot={avg_rs:8.2f}"
                )

            if ep_counter % eval_interval == 0:
                ec_list, es_list = _evaluate(env, coordinator)
                ec = float(np.mean(ec_list))
                imp = (fixed_cost_mid - ec) / fixed_cost_mid * 100
                buf = coordinator.get_agent_sizes()
                logger.info(
                    f"  >> Eval Ep {ep_counter}: cost={ec:.0f} | imp={imp:+.1f}% "
                    f"| flex={np.mean(es_list):.0f} | buffers={buf}"
                )

    if alternating:
        _phase_loop("P1-midlong", phase1_episodes, freeze_midlong=False, freeze_spot=True)
        _phase_loop("P2-spot", phase2_episodes, freeze_midlong=True, freeze_spot=False)
        _phase_loop("P3-joint", phase3_episodes, freeze_midlong=False, freeze_spot=False)
    else:
        # 消融: 无交替训练, 从一开始两智能体同时联合训练
        _phase_loop("Joint-only", total_episodes, freeze_midlong=False, freeze_spot=False)

    # 最终评估: DRL 与固定比例基准使用同一批世界 (配对评价)
    final_costs, final_savings = _evaluate(env, coordinator, num_eval=NUM_FINAL_WORLDS)
    final_cost = float(np.mean(final_costs))
    final_flex = float(np.mean(final_savings))
    improvement = (fixed_cost - final_cost) / fixed_cost * 100
    per_world = [
        {"world": w, "fixed": fixed_final[w], "drl": final_costs[w],
         "improvement_pct": (fixed_final[w] - final_costs[w]) / fixed_final[w] * 100}
        for w in range(NUM_FINAL_WORLDS)
    ]
    pair_win = sum(1 for r in per_world if r["drl"] < r["fixed"])
    logger.info(
        f"[FINAL] {scenario}: DRL={final_cost:.0f} | Fixed={fixed_cost:.0f} "
        f"| 改进={improvement:+.1f}% | 逐世界胜出={pair_win}/{NUM_FINAL_WORLDS} "
        f"| 柔性节约={final_flex:.0f}"
    )

    # 保存模型
    if save_prefix:
        os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
        coordinator.save(save_prefix)
        logger.info(f"模型已保存: {save_prefix}_midlong.pt, {save_prefix}_spot.pt")

    return {
        "scenario": scenario,
        "load_scenario": load_scenario,
        "seed": seed,
        "final_cost": final_cost,
        "final_flex_saving": final_flex,
        "fixed_baseline": fixed_cost,
        "improvement_vs_fixed_pct": improvement,
        "num_final_worlds": NUM_FINAL_WORLDS,
        "paired_win_count": pair_win,
        "per_world": per_world,
        "history": history,
    }


# ============================================================
# CLI
# ============================================================
SCENARIO_KEYS = list(PRICE_SCENARIOS.keys())


def _short(name):
    return name.split("_")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None,
                        help="场景前缀 A/B/C/D/E 或完整名称")
    parser.add_argument("--all", action="store_true", help="跑全部 5 个场景")
    parser.add_argument("--load", type=str, default="柔性适中")
    parser.add_argument("--episodes", type=int, default=None,
                        help="若指定, 三阶段按 25%/25%/50% 切分")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试: 3 场景 300 episodes")
    parser.add_argument("--out", type=str, default="results/dual_agent_v2_summary.json",
                        help="结果 JSON 输出路径")
    parser.add_argument("--no-scale", action="store_true", help="消融/简化: 去掉现货奖励放大")
    parser.add_argument("--no-shaping", action="store_true", help="消融: 去掉暴露成本塑形")
    parser.add_argument("--no-alternating", action="store_true", help="消融/简化: 从头联合训练")
    parser.add_argument("--model-subdir", type=str, default="dual_v2", help="模型保存子目录")
    args = parser.parse_args()

    if args.smoke:
        p1, p2, p3 = 60, 60, 180
    elif args.episodes:
        p1 = args.episodes // 4
        p2 = args.episodes // 4
        p3 = args.episodes - p1 - p2
    else:
        p1, p2, p3 = 500, 500, 1000

    if args.all or args.smoke:
        scenarios = SCENARIO_KEYS
    elif args.scenario:
        # 支持短标签
        match = [k for k in SCENARIO_KEYS if k.startswith(args.scenario)]
        if not match:
            raise SystemExit(f"未知场景: {args.scenario}. 可选: {SCENARIO_KEYS}")
        scenarios = [match[0]]
    else:
        scenarios = [SCENARIO_KEYS[0]]

    all_results = {}
    os.makedirs("results", exist_ok=True)

    for sc in scenarios:
        tag = _short(sc)
        prefix = os.path.join("results", "models", args.model_subdir, f"{tag}_dual")
        r = train_dual_agent(
            scenario=sc,
            load_scenario=args.load,
            phase1_episodes=p1,
            phase2_episodes=p2,
            phase3_episodes=p3,
            save_prefix=prefix,
            scale_spot_reward=not args.no_scale,
            exposure_shaping=not args.no_shaping,
            alternating=not args.no_alternating,
        )
        # 保留精简 history 供训练曲线绘图 (每 10 ep 采样一次)
        hist_sample = r.pop("history")
        r["history_sample"] = [
            {"episode": h["episode"], "phase": h["phase"],
             "cost": h["cost"], "flex_saving": h["flex_saving"],
             "reward_midlong": h["reward_midlong"], "reward_spot": h["reward_spot"]}
            for i, h in enumerate(hist_sample) if i % 10 == 0 or i == len(hist_sample) - 1
        ]
        all_results[tag] = r

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase1": p1, "phase2": p2, "phase3": p3,
            "load_scenario": args.load,
            "config": {
                "scale_spot_reward": not args.no_scale,
                "exposure_shaping": not args.no_shaping,
                "alternating": not args.no_alternating,
            },
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)

    # 汇总表
    print("\n" + "=" * 72)
    print(f"{'场景':<4} | {'固定比例':>12} | {'双DRL':>12} | {'改进':>8} | {'柔性节约':>8}")
    print("-" * 72)
    for tag, r in all_results.items():
        print(
            f"{tag:<4} | {r['fixed_baseline']:>12,.0f} | {r['final_cost']:>12,.0f} "
            f"| {r['improvement_vs_fixed_pct']:>+7.1f}% | {r['final_flex_saving']:>8,.0f}"
        )
    print(f"\n结果已写入: {out_path}")
