"""
训练入口脚本 (v2 — 分层架构)
上层 SAC 智能体: 决策 α/β/γ 仓位比例 (1维动作)
下层规则调度器: 现货阶段自动调度柔性负荷 (环境内部执行)
"""
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import DRL_PARAMS, SIM_PARAMS
from src.utils.logger import setup_logger

logger = setup_logger("train")


def train(
    scenario: str = "A_年度低价后期上涨",
    load_scenario: str = "柔性适中",
    max_episodes: int = DRL_PARAMS["max_episode"],
    print_interval: int = 100,
    eval_interval: int = 500,
):
    """
    训练主循环 (分层架构: 上层 SAC + 下层规则)

    Args:
        scenario: 价格场景名称
        load_scenario: 负荷场景名称
        max_episodes: 最大训练回合数
        print_interval: 每隔多少回合打印一次
        eval_interval: 每隔多少回合评估一次
    """
    logger.info(f"开始训练 | 价格场景: {scenario} | 负荷场景: {load_scenario}")

    # 创建环境 (分层架构: 1维动作空间)
    env = TradingEnvironment(
        scenario_name=scenario,
        load_scenario=load_scenario,
        seed=42,
    )

    # 获取状态/动作维度
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logger.info(f"状态维度: {state_dim}, 动作维度: {action_dim} (分层: 仅仓位比例)")

    # 创建单智能体 (上层 SAC)
    coordinator = MultiScaleCoordinator(
        state_dim=state_dim,
        action_dim=action_dim,
        use_dual_agent=False,
    )

    # 固定比例基准
    fixed_result = env.get_fixed_ratio_cost()
    logger.info(f"固定比例策略基准成本: {fixed_result['total_cost']:.0f} 元")

    # 训练记录
    episode_rewards = []
    episode_costs = []
    episode_flex_savings = []
    eval_results = []

    # ---- 训练循环 ----
    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        step_count = 0

        while True:
            # 选择动作 (1维: ratio_raw)
            action = coordinator.select_action(state, env.current_stage)

            # 执行
            next_state, reward, terminated, truncated, info = env.step(action)

            # 存储经验并更新 (单智能体)
            coordinator.store_transition(
                state, action, reward, next_state, terminated,
                stage=env.current_stage
            )
            coordinator.update()

            total_reward += reward
            state = next_state
            step_count += 1

            if terminated or truncated:
                break

        # 记录
        episode_rewards.append(total_reward)
        episode_costs.append(info["total_cost"])
        episode_flex_savings.append(info.get("flexible_cost_saving", 0))

        # 打印
        if ep % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_cost = np.mean(episode_costs[-print_interval:])
            avg_flex = np.mean(episode_flex_savings[-print_interval:])
            logger.info(
                f"Episode {ep:5d} | "
                f"平均奖励: {avg_reward:10.1f} | "
                f"平均成本: {avg_cost:10.0f} | "
                f"柔性节约: {avg_flex:8.0f} | "
                f"步数: {step_count:3d}"
            )

        # 评估
        if ep % eval_interval == 0:
            eval_cost, eval_flex = evaluate(env, coordinator)
            eval_results.append(eval_cost)
            improvement = (fixed_result["total_cost"] - eval_cost) / fixed_result["total_cost"] * 100
            logger.info(
                f"  >> 评估 (Ep {ep}): DRL成本={eval_cost:.0f} | "
                f"固定成本={fixed_result['total_cost']:.0f} | "
                f"改进: {improvement:+.1f}% | "
                f"柔性节约: {eval_flex:.0f}"
            )

    # 保存模型
    model_dir = os.path.join("results", "models")
    os.makedirs(model_dir, exist_ok=True)
    sc_tag = scenario.split("_")[0]
    ls_tag = load_scenario.replace(" ", "")
    coordinator.save(os.path.join(model_dir, f"{sc_tag}_{ls_tag}_hierarchical"))
    logger.info(f"训练完成, 模型已保存到 {model_dir}/")

    return {
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "episode_flex_savings": episode_flex_savings,
        "eval_results": eval_results,
        "fixed_baseline": fixed_result["total_cost"],
    }


def evaluate(env: TradingEnvironment, coordinator: MultiScaleCoordinator,
             num_eval: int = 5):
    """评估当前策略"""
    costs = []
    flex_savings = []
    for _ in range(num_eval):
        state, _ = env.reset()
        while True:
            action = coordinator.select_action(state, env.current_stage, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        costs.append(info["total_cost"])
        flex_savings.append(info.get("flexible_cost_saving", 0))
    return np.mean(costs), np.mean(flex_savings)


if __name__ == "__main__":
    results = train(
        scenario="A_年度低价后期上涨",
        load_scenario="柔性适中",
        max_episodes=2000,
        print_interval=100,
        eval_interval=500,
    )
