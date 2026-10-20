"""
分层架构训练脚本 (v3 — 兼容 B 方案改造后的环境)

上层 SAC (1D: ratio_raw) + 下层规则柔性强度 (环境内 _rule_intensity + FlexibleExecutor 执行)
用于与双智能体 B 方案做对比实验。

关键: TradingEnvironment 需实例化时 use_rule_dispatcher=True
      (现货阶段不再采用 DRL 的 flexible_action, 改用峰谷比规则给柔性强度)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import DRL_PARAMS
from src.utils.logger import setup_logger

logger = setup_logger("train")


def train(
    scenario: str = "A_年度低价后期上涨",
    load_scenario: str = "柔性适中",
    max_episodes: int = DRL_PARAMS["max_episode"],
    print_interval: int = 100,
    eval_interval: int = 500,
):
    logger.info(f"[分层模式] 价格场景: {scenario} | 负荷场景: {load_scenario}")

    env = TradingEnvironment(
        scenario_name=scenario,
        load_scenario=load_scenario,
        use_rule_dispatcher=True,     # 启用规则调度器 (分层)
        seed=42,
    )

    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]

    # 单智能体模式: 输出 2D 动作, 但只有 action[0] 有意义 (action[1] 被 env 忽略)
    coordinator = MultiScaleCoordinator(
        state_dim=state_dim,
        use_dual_agent=False,
        single_action_dim=2,
        single_action_scale=5.0,
    )

    fixed_result = env.get_fixed_ratio_cost()
    logger.info(f"固定比例基准: {fixed_result['total_cost']:.0f}")

    episode_rewards = []
    episode_costs = []
    episode_flex_savings = []
    eval_results = []

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        while True:
            stage_before = env.current_stage
            action = coordinator.select_action(state, stage_before)
            next_state, reward, terminated, truncated, info = env.step(action)

            # 单智能体: 奖励走 reward_global, 用 stage_before 路由
            coordinator.store_transition(
                state, action, info["reward_global"], next_state, terminated,
                stage=stage_before,
            )
            coordinator.update()

            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_costs.append(info["total_cost"])
        episode_flex_savings.append(info.get("flexible_cost_saving", 0))

        if ep % print_interval == 0:
            avg_cost = np.mean(episode_costs[-print_interval:])
            avg_flex = np.mean(episode_flex_savings[-print_interval:])
            logger.info(
                f"Ep {ep:5d} | 成本 {avg_cost:10.0f} | 柔性节约 {avg_flex:8.0f}"
            )

        if ep % eval_interval == 0:
            eval_cost, eval_flex = evaluate(env, coordinator)
            eval_results.append(eval_cost)
            improvement = (fixed_result["total_cost"] - eval_cost) / fixed_result["total_cost"] * 100
            logger.info(
                f"  >> 评估: DRL={eval_cost:.0f} | 改进={improvement:+.1f}% | 柔性={eval_flex:.0f}"
            )

    model_dir = os.path.join("results", "models")
    os.makedirs(model_dir, exist_ok=True)
    sc_tag = scenario.split("_")[0]
    ls_tag = load_scenario.replace(" ", "")
    coordinator.save(os.path.join(model_dir, f"{sc_tag}_{ls_tag}_hierarchical"))

    return {
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "episode_flex_savings": episode_flex_savings,
        "eval_results": eval_results,
        "fixed_baseline": fixed_result["total_cost"],
    }


def evaluate(env: TradingEnvironment, coordinator: MultiScaleCoordinator, num_eval: int = 5):
    costs = []
    flex_savings = []
    for _ in range(num_eval):
        state, _ = env.reset()
        while True:
            stage = env.current_stage
            action = coordinator.select_action(state, stage, deterministic=True)
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        costs.append(info["total_cost"])
        flex_savings.append(info.get("flexible_cost_saving", 0))
    return np.mean(costs), np.mean(flex_savings)


if __name__ == "__main__":
    train(
        scenario="A_年度低价后期上涨",
        load_scenario="柔性适中",
        max_episodes=2000,
    )
