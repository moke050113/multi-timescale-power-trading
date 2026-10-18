"""
分层架构对比实验
比较: 分层(SAC+规则) vs 固定比例 vs 纯规则调度

在 5 种价格场景上训练, 记录成本和柔性节约
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('D:/Qoder_project')

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import DRL_PARAMS

# 5 种价格场景
PRICE_SCENARIOS = [
    "A_年度低价后期上涨",
    "B_年度高价后期下降",
    "C_各阶段价格接近",
    "D_现货高波动",
    "E_月度低价现货极端风险",
]

LOAD_SCENARIO = "柔性适中"
EPISODES = 1000
EVAL_INTERVAL = 200
NUM_EVAL = 5


def evaluate_deterministic(env, coordinator, num_eval=NUM_EVAL):
    """确定性评估"""
    costs = []
    flex_savings = []
    flex_penalties = []
    for _ in range(num_eval):
        state, _ = env.reset()
        while True:
            action = coordinator.select_action(state, env.current_stage, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        costs.append(info["total_cost"])
        flex_savings.append(info.get("flexible_cost_saving", 0))
        flex_penalties.append(info.get("total_flexible_penalty", 0))
    return {
        "cost_mean": np.mean(costs),
        "cost_std": np.std(costs),
        "flex_saving_mean": np.mean(flex_savings),
        "flex_penalty_mean": np.mean(flex_penalties),
    }


def evaluate_fixed_ratio(env, num_eval=NUM_EVAL):
    """固定比例策略 (无 DRL, 无柔性)"""
    costs = []
    for _ in range(num_eval):
        result = env.get_fixed_ratio_cost()
        costs.append(result["total_cost"])
    return {
        "cost_mean": np.mean(costs),
        "cost_std": np.std(costs),
        "flex_saving_mean": 0,
        "flex_penalty_mean": 0,
    }


def evaluate_pure_rule(env, num_eval=NUM_EVAL):
    """纯规则调度: 固定比例 + 规则柔性调度 (无 DRL)"""
    costs = []
    flex_savings = []
    flex_penalties = []
    for _ in range(num_eval):
        state, _ = env.reset()
        while True:
            # 固定比例动作 (sigmoid(0) = 0.5, 但需要看具体映射)
            # 实际上用 FIXED_RATIOS 对应的 raw 值
            from src.utils.config import FIXED_RATIOS, MARKET_BOUNDS
            from src.market.environment import sigmoid

            if env.current_stage == env.STAGE_ANNUAL:
                # 反推 ratio_raw: alpha = sigmoid(ratio_raw) * max
                # ratio_raw = logit(alpha / max)
                target = FIXED_RATIOS["alpha"] / MARKET_BOUNDS["annual"]["max"]
                raw = np.log(target / (1 - target + 1e-8) + 1e-8)
                action = np.array([raw], dtype=np.float32)
            elif env.current_stage == env.STAGE_MONTHLY:
                lo = env.alpha
                hi = MARKET_BOUNDS["monthly"]["max"]
                target = (FIXED_RATIOS["beta"] - lo) / (hi - lo + 1e-8)
                target = np.clip(target, 0.01, 0.99)
                raw = np.log(target / (1 - target + 1e-8) + 1e-8)
                action = np.array([raw], dtype=np.float32)
            elif env.current_stage == env.STAGE_INTRADAY:
                lo = env.beta
                hi = MARKET_BOUNDS["intraday"]["max"]
                target = (FIXED_RATIOS["gamma"] - lo) / (hi - lo + 1e-8)
                target = np.clip(target, 0.01, 0.99)
                raw = np.log(target / (1 - target + 1e-8) + 1e-8)
                action = np.array([raw], dtype=np.float32)
            else:
                # 现货阶段: 随便给个动作 (规则调度器会自动处理)
                action = np.array([0.0], dtype=np.float32)

            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        costs.append(info["total_cost"])
        flex_savings.append(info.get("flexible_cost_saving", 0))
        flex_penalties.append(info.get("total_flexible_penalty", 0))
    return {
        "cost_mean": np.mean(costs),
        "cost_std": np.std(costs),
        "flex_saving_mean": np.mean(flex_savings),
        "flex_penalty_mean": np.mean(flex_penalties),
    }


def train_hierarchical(scenario, load_scenario=LOAD_SCENARIO, episodes=EPISODES):
    """训练分层架构并返回评估结果"""
    env = TradingEnvironment(
        scenario_name=scenario,
        load_scenario=load_scenario,
        seed=42,
    )

    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    coordinator = MultiScaleCoordinator(
        state_dim=state_dim,
        action_dim=action_dim,
        use_dual_agent=False,
    )

    # 训练
    episode_costs = []
    eval_curve = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        while True:
            action = coordinator.select_action(state, env.current_stage)
            next_state, reward, terminated, truncated, info = env.step(action)
            coordinator.store_transition(state, action, reward, next_state, terminated, stage=env.current_stage)
            coordinator.update()
            state = next_state
            if terminated or truncated:
                break

        episode_costs.append(info["total_cost"])

        if ep % EVAL_INTERVAL == 0:
            result = evaluate_deterministic(env, coordinator)
            eval_curve.append({
                "episode": ep,
                "cost": result["cost_mean"],
                "flex_saving": result["flex_saving_mean"],
            })
            print(f"  [{scenario[:1]}] Ep {ep:4d} | 成本: {result['cost_mean']:>10,.0f} | "
                  f"柔性节约: {result['flex_saving_mean']:>6,.0f}")

    # 最终评估
    final_eval = evaluate_deterministic(env, coordinator)

    # 保存模型
    model_dir = os.path.join("results", "models", "hierarchical")
    os.makedirs(model_dir, exist_ok=True)
    sc_tag = scenario.split("_")[0]
    coordinator.save(os.path.join(model_dir, f"{sc_tag}_hierarchical"))

    return {
        "scenario": scenario,
        "final_eval": final_eval,
        "eval_curve": eval_curve,
        "training_costs": episode_costs,
    }


def run_all_experiments():
    """运行全部对比实验"""
    results = {}

    for scenario in PRICE_SCENARIOS:
        sc_tag = scenario.split("_")[0]
        print(f"\n{'='*60}")
        print(f"场景 {scenario}")
        print(f"{'='*60}")

        # 1. 固定比例基准
        env = TradingEnvironment(scenario_name=scenario, load_scenario=LOAD_SCENARIO, seed=42)
        fixed = evaluate_fixed_ratio(env)
        print(f"  固定比例:     成本={fixed['cost_mean']:>10,.0f}")

        # 2. 纯规则调度 (固定比例 + 规则柔性)
        pure_rule = evaluate_pure_rule(env)
        print(f"  纯规则调度:   成本={pure_rule['cost_mean']:>10,.0f} | "
              f"柔性节约={pure_rule['flex_saving_mean']:>6,.0f}")

        # 3. 分层架构 (SAC + 规则)
        print(f"  训练分层架构 ({EPISODES} episodes)...")
        hierarchical = train_hierarchical(scenario)

        h_eval = hierarchical["final_eval"]
        print(f"  分层架构:     成本={h_eval['cost_mean']:>10,.0f} | "
              f"柔性节约={h_eval['flex_saving_mean']:>6,.0f}")

        # 计算改进
        h_impr_vs_fixed = (fixed["cost_mean"] - h_eval["cost_mean"]) / fixed["cost_mean"] * 100
        h_impr_vs_rule = (pure_rule["cost_mean"] - h_eval["cost_mean"]) / pure_rule["cost_mean"] * 100

        results[sc_tag] = {
            "scenario": scenario,
            "fixed_ratio": fixed,
            "pure_rule": pure_rule,
            "hierarchical": h_eval,
            "eval_curve": hierarchical["eval_curve"],
            "improvement_vs_fixed_pct": h_impr_vs_fixed,
            "improvement_vs_rule_pct": h_impr_vs_rule,
        }

        print(f"  >> vs 固定比例: {h_impr_vs_fixed:+.1f}%")
        print(f"  >> vs 纯规则:   {h_impr_vs_rule:+.1f}%")

    # 保存结果
    output = {
        "load_scenario": LOAD_SCENARIO,
        "episodes": EPISODES,
        "scenarios": results,
    }

    output_path = os.path.join("results", "hierarchical_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存到 {output_path}")

    # 汇总表
    print(f"\n{'='*70}")
    print(f"{'汇总':^70}")
    print(f"{'='*70}")
    print(f"{'场景':>4} | {'固定比例':>12} | {'纯规则':>12} | {'分层SAC':>12} | {'vs固定':>8} | {'vs规则':>8}")
    print(f"{'-'*4} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*8} | {'-'*8}")

    for sc_tag, r in results.items():
        print(f"{sc_tag:>4} | {r['fixed_ratio']['cost_mean']:>12,.0f} | "
              f"{r['pure_rule']['cost_mean']:>12,.0f} | "
              f"{r['hierarchical']['cost_mean']:>12,.0f} | "
              f"{r['improvement_vs_fixed_pct']:>+7.1f}% | "
              f"{r['improvement_vs_rule_pct']:>+7.1f}%")

    return output


if __name__ == "__main__":
    run_all_experiments()
