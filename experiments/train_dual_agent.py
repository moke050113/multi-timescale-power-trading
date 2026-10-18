"""
双智能体分阶段奖励实验
对比单智能体 vs 双智能体在 A/B/C 场景上的表现
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

logger = setup_logger("dual_agent_exp")


def train_agent(scenario_name, use_dual_agent, max_episodes=1000, seed=42):
    """训练单个智能体并返回结果"""
    env = TradingEnvironment(scenario_name=scenario_name, seed=seed)
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]

    coordinator = MultiScaleCoordinator(
        state_dim=state_dim, action_dim=2, use_dual_agent=use_dual_agent
    )

    fixed = env.get_fixed_ratio_cost()
    mode = "dual" if use_dual_agent else "single"
    logger.info(f"开始训练 | {scenario_name} | {mode} | episodes={max_episodes}")

    episode_costs = []

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        while True:
            stage_before = env.current_stage
            action = coordinator.select_action(state, stage_before)
            next_state, reward, terminated, truncated, info = env.step(action)

            if use_dual_agent:
                # 双智能体: 共享全局奖励, 但动作空间和经验池独立
                # 中长期智能体只看到 ratio 部分的动作, 现货智能体只看到 flexible 部分
                if stage_before in [0, 1, 2]:
                    coordinator.store_transition(
                        state, action, reward, next_state, terminated,
                        stage=stage_before
                    )
                    coordinator.update(stage=stage_before)
                else:
                    coordinator.store_transition(
                        state, action, reward, next_state, terminated,
                        stage=stage_before
                    )
                    coordinator.update(stage=stage_before)
            else:
                coordinator.store_transition(
                    state, action, reward, next_state, terminated,
                    stage=stage_before
                )
                coordinator.update()

            state = next_state
            if terminated:
                break

        episode_costs.append(info["total_cost"])

        if ep % 200 == 0:
            avg = np.mean(episode_costs[-100:])
            imp = (fixed["total_cost"] - avg) / fixed["total_cost"] * 100
            logger.info(f"  Ep {ep:5d} | avg_cost={avg:10.0f} | improvement={imp:+.1f}%")

    # 评估最后 100 episodes
    final_avg = np.mean(episode_costs[-100:])
    improvement = (fixed["total_cost"] - final_avg) / fixed["total_cost"] * 100

    result = {
        "scenario": scenario_name,
        "mode": mode,
        "fixed_cost": fixed["total_cost"],
        "final_avg_cost": float(final_avg),
        "improvement_pct": float(improvement),
        "episode_costs": [float(c) for c in episode_costs],
    }

    logger.info(f"完成 | {mode} | final_avg={final_avg:.0f} | improvement={improvement:+.1f}%")
    return result


def main():
    scenarios = ["A_年度低价后期上涨", "B_年度高价后期下降", "C_各阶段价格接近"]
    episodes = 1000
    all_results = []

    for sc in scenarios:
        for use_dual in [False, True]:
            result = train_agent(sc, use_dual, max_episodes=episodes)
            all_results.append(result)

            # 保存中间结果
            os.makedirs("results", exist_ok=True)
            with open("results/dual_agent_experiment.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印对比表
    logger.info("=" * 70)
    logger.info("单智能体 vs 双智能体 对比")
    logger.info("=" * 70)
    for sc in scenarios:
        single = [r for r in all_results if r["scenario"] == sc and r["mode"] == "single"][0]
        dual = [r for r in all_results if r["scenario"] == sc and r["mode"] == "dual"][0]
        short = sc.split("_")[0]
        logger.info(
            f"场景 {short} | 固定={single['fixed_cost']:>10.0f} | "
            f"单智能体={single['final_avg_cost']:>10.0f} ({single['improvement_pct']:+.1f}%) | "
            f"双智能体={dual['final_avg_cost']:>10.0f} ({dual['improvement_pct']:+.1f}%)"
        )
    logger.info("=" * 70)
    logger.info("实验完成, 结果已保存到 results/dual_agent_experiment.json")


if __name__ == "__main__":
    main()
