"""
场景 D/E 负荷场景扫描实验
单智能体, 5种负荷场景 (含柔性大规模)
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, "D:/Qoder_project")
os.chdir("D:/Qoder_project")

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import PRICE_SCENARIOS
from src.utils.logger import setup_logger

logger = setup_logger("de_scan")


def run_experiment(scenario_name, load_scenario, max_episodes=500, seed=42):
    env = TradingEnvironment(
        scenario_name=scenario_name, load_scenario=load_scenario, seed=seed
    )
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]

    coord = MultiScaleCoordinator(
        state_dim=state_dim, single_action_dim=2, use_dual_agent=False
    )
    fixed = env.get_fixed_ratio_cost()

    costs = []
    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        while True:
            stage_before = env.current_stage
            action = coord.select_action(state, stage_before)
            next_state, reward, terminated, truncated, info = env.step(action)
            coord.store_transition(
                state, action, reward, next_state, terminated, stage=stage_before
            )
            coord.update()
            state = next_state
            if terminated:
                break
        costs.append(info["total_cost"])

    final_avg = np.mean(costs[-100:])
    improvement = (fixed["total_cost"] - final_avg) / fixed["total_cost"] * 100
    flex_saving = info["flexible_cost_saving"]
    flex_penalty = info["total_flexible_penalty"]

    return {
        "scenario": scenario_name,
        "load_scenario": load_scenario,
        "fixed_cost": fixed["total_cost"],
        "drl_cost": float(final_avg),
        "improvement_pct": float(improvement),
        "flex_saving": float(flex_saving),
        "flex_penalty": float(flex_penalty),
    }


def main():
    scenarios = ["D_现货高波动", "E_月度低价现货极端风险"]
    load_scenarios = ["柔性充裕", "柔性适中", "柔性受限", "纯刚性", "柔性大规模"]
    episodes = 500

    all_results = []
    for sc in scenarios:
        for ls in load_scenarios:
            short_sc = sc.split("_")[0]
            logger.info(f"训练 {short_sc} + {ls} ({episodes} episodes)...")
            result = run_experiment(sc, ls, max_episodes=episodes)
            all_results.append(result)

            os.makedirs("results", exist_ok=True)
            with open("results/de_load_scan.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            logger.info(
                f"  DRL={result['drl_cost']:.0f} | "
                f"imp={result['improvement_pct']:+.1f}% | "
                f"saving={result['flex_saving']:.1f}"
            )

    # 打印汇总表
    logger.info("=" * 80)
    for sc in scenarios:
        short = sc.split("_")[0]
        logger.info(f"场景 {short}:")
        for r in all_results:
            if r["scenario"] == sc:
                ls = r["load_scenario"]
                logger.info(
                    f"  {ls:8s} | fixed={r['fixed_cost']:>10.0f} | "
                    f"DRL={r['drl_cost']:>10.0f} | imp={r['improvement_pct']:+.1f}% | "
                    f"saving={r['flex_saving']:>8.1f}"
                )
    logger.info("=" * 80)
    logger.info("实验完成!")


if __name__ == "__main__":
    main()
