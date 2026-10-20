"""
完整实验脚本 v2
包含:
  1. 5种价格场景 × 4种负荷场景 对比
  2. 随机策略基线
  3. CVaR 风险参数敏感性
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import DRL_PARAMS, PRICE_SCENARIOS
from src.utils.logger import setup_logger

logger = setup_logger("experiment")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_training(scenario_name, load_scenario="柔性适中", max_episodes=2000,
                 risk_lambda=0.1, seed=42, include_flexible=True):
    """单次训练"""
    env = TradingEnvironment(
        scenario_name=scenario_name,
        load_scenario=load_scenario,
        include_flexible=include_flexible,
        risk_lambda=risk_lambda,
        seed=seed,
    )
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]

    coordinator = MultiScaleCoordinator(
        state_dim=state_dim, single_action_dim=2, use_dual_agent=False
    )

    fixed_result = env.get_fixed_ratio_cost()
    fixed_cost = fixed_result["total_cost"]

    episode_costs = []
    eval_costs = []
    eval_episodes = []
    eval_interval = 200

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            action = coordinator.select_action(state, env.current_stage)
            ns, r, term, trunc, info = env.step(action)
            coordinator.store_transition(state, action, r, ns, term, env.current_stage)
            coordinator.update()
            total_reward += r
            state = ns
            if term or trunc:
                break

        episode_costs.append(info["total_cost"])

        if ep % eval_interval == 0:
            ec = evaluate(env, coordinator, num_eval=3)
            eval_costs.append(ec)
            eval_episodes.append(ep)

    final_cost = evaluate(env, coordinator, num_eval=10)
    improvement = (fixed_cost - final_cost) / fixed_cost * 100

    return {
        "scenario": scenario_name,
        "load_scenario": load_scenario,
        "risk_lambda": risk_lambda,
        "include_flexible": include_flexible,
        "fixed_baseline_cost": fixed_cost,
        "final_drl_cost": final_cost,
        "improvement_pct": improvement,
        "episode_costs": [float(c) for c in episode_costs],
        "eval_episodes": eval_episodes,
        "eval_costs": [float(c) for c in eval_costs],
        "flexible_saving": float(env.flexible_cost_saving),
        "flexible_penalty": float(env.total_flexible_penalty),
    }


def run_random_baseline(scenario_name, load_scenario="柔性适中", num_episodes=100, seed=42):
    """随机策略基线"""
    env = TradingEnvironment(
        scenario_name=scenario_name,
        load_scenario=load_scenario,
        seed=seed,
    )
    costs = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        while True:
            action = env.action_space.sample()
            state, r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        costs.append(info["total_cost"])
    return np.mean(costs)


def evaluate(env, coordinator, num_eval=5):
    costs = []
    for _ in range(num_eval):
        state, _ = env.reset()
        while True:
            action = coordinator.select_action(state, env.current_stage, deterministic=True)
            state, r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        costs.append(info["total_cost"])
    return np.mean(costs)


def save_result(result, tag=""):
    name = result["scenario"].split("_")[0]
    ls = result.get("load_scenario", "").replace("柔性", "flex").replace("纯刚性", "rigid")
    rl = result.get("risk_lambda", 0.1)
    flex = "flex" if result.get("include_flexible", True) else "no_flex"
    filename = f"result_{name}_{ls}_lambda{rl}_{flex}{tag}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return filepath


# ============================================================
# 实验 1: 负荷场景对比
# ============================================================

def experiment_load_scenarios():
    logger.info("=" * 60)
    logger.info("实验1: 负荷场景对比 (3价格 × 4负荷)")
    logger.info("=" * 60)

    load_scenarios = ["柔性充裕", "柔性适中", "柔性受限", "纯刚性"]
    price_scenarios = list(PRICE_SCENARIOS.keys())[:3]

    results = {}
    for ps in price_scenarios:
        for ls in load_scenarios:
            logger.info(f"  {ps.split('_')[0]} + {ls} ...")
            r = run_training(ps, load_scenario=ls, max_episodes=1000, seed=42)
            results[f"{ps.split('_')[0]}_{ls}"] = r
            save_result(r)
            logger.info(f"    DRL={r['final_drl_cost']:.0f} | 固定={r['fixed_baseline_cost']:.0f} | 改进={r['improvement_pct']:+.1f}%")

    return results


# ============================================================
# 实验 2: 随机策略基线
# ============================================================

def experiment_random_baseline():
    logger.info("=" * 60)
    logger.info("实验2: 随机策略基线")
    logger.info("=" * 60)

    price_scenarios = list(PRICE_SCENARIOS.keys())
    results = {}

    for ps in price_scenarios:
        random_cost = run_random_baseline(ps, num_episodes=100)
        logger.info(f"  {ps.split('_')[0]}: 随机策略成本 = {random_cost:.0f}")
        results[ps.split("_")[0]] = {"random_cost": random_cost}

    return results


# ============================================================
# 实验 3: CVaR 风险参数敏感性
# ============================================================

def experiment_cvar_sensitivity():
    logger.info("=" * 60)
    logger.info("实验3: CVaR 风险参数敏感性")
    logger.info("=" * 60)

    lambdas = [0.0, 0.05, 0.1, 0.2, 0.5]
    ps = "D_现货高波动"
    results = {}

    for lam in lambdas:
        logger.info(f"  λ={lam} ...")
        r = run_training(ps, max_episodes=1000, risk_lambda=lam, seed=42)
        results[f"lambda_{lam}"] = r
        save_result(r, tag=f"_lambda{lam}")
        logger.info(f"    DRL={r['final_drl_cost']:.0f} | 改进={r['improvement_pct']:+.1f}%")

    return results


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 实验 1: 负荷场景对比
    load_results = experiment_load_scenarios()

    # 实验 2: 随机基线
    random_results = experiment_random_baseline()

    # 实验 3: CVaR 敏感性
    cvar_results = experiment_cvar_sensitivity()

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("全部实验完成")
    logger.info("=" * 60)

    logger.info("\n实验1 - 负荷场景对比:")
    for k, r in load_results.items():
        logger.info(f"  {k:25s} | DRL: {r['final_drl_cost']:10.0f} | 改进: {r['improvement_pct']:+.1f}%")

    logger.info("\n实验2 - 随机基线:")
    for k, r in random_results.items():
        logger.info(f"  {k:6s} | 随机成本: {r['random_cost']:10.0f}")

    logger.info("\n实验3 - CVaR敏感性:")
    for k, r in cvar_results.items():
        logger.info(f"  {k:12s} | DRL: {r['final_drl_cost']:10.0f} | 改进: {r['improvement_pct']:+.1f}%")
