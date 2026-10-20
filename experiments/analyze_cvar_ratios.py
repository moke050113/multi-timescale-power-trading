"""
CVaR实验补充: 记录不同lambda下DRL学到的alpha/beta/gamma
回答研究设想问题5: "风险厌恶程度变化时, alpha/beta/gamma的变化规律是什么?"
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import PRICE_SCENARIOS
from src.utils.logger import setup_logger

logger = setup_logger("cvar_ratios")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def train_and_record_ratios(scenario_name, risk_lambda, max_episodes=1000, seed=42):
    """训练并记录最后200轮的比例"""
    env = TradingEnvironment(
        scenario_name=scenario_name,
        include_flexible=True,
        risk_lambda=risk_lambda,
        seed=seed,
    )
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    coordinator = MultiScaleCoordinator(state_dim=state_dim, single_action_dim=2, use_dual_agent=False)

    all_ratios = []
    episode_costs = []

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_ratios = {}
        while True:
            action = coordinator.select_action(state, env.current_stage)
            ns, r, term, trunc, info = env.step(action)
            coordinator.store_transition(state, action, r, ns, term, env.current_stage)
            coordinator.update()
            stage_name = info.get("stage", "")
            if stage_name == "annual":
                ep_ratios["alpha"] = info["positions"]["alpha"]
            elif stage_name == "monthly":
                ep_ratios["beta"] = info["positions"]["beta"]
            elif stage_name == "intraday":
                ep_ratios["gamma"] = info["positions"]["gamma"]
            state = ns
            if term or trunc:
                break

        episode_costs.append(info["total_cost"])
        if ep > max_episodes - 200:
            all_ratios.append(ep_ratios)

        if ep % 200 == 0:
            logger.info(f"    Ep {ep}: cost={info['total_cost']:.0f}, "
                       f"a={ep_ratios.get('alpha',0):.3f} b={ep_ratios.get('beta',0):.3f} g={ep_ratios.get('gamma',0):.3f}")

    alphas = [r.get("alpha", 0) for r in all_ratios]
    betas = [r.get("beta", 0) for r in all_ratios]
    gammas = [r.get("gamma", 0) for r in all_ratios]

    return {
        "lambda": risk_lambda,
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "beta_mean": float(np.mean(betas)),
        "beta_std": float(np.std(betas)),
        "gamma_mean": float(np.mean(gammas)),
        "gamma_std": float(np.std(gammas)),
        "spot_mean": float(1.0 - np.mean(gammas)),
        "avg_cost": float(np.mean(episode_costs[-200:])),
    }


if __name__ == "__main__":
    scenario = "D_现货高波动"
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.5]

    logger.info("=" * 70)
    logger.info(f"CVaR比例分析: 场景{scenario}, lambda={lambdas}")
    logger.info("=" * 70)

    results = {}
    for lam in lambdas:
        logger.info(f"  lambda={lam} ...")
        r = train_and_record_ratios(scenario, risk_lambda=lam, max_episodes=1000, seed=42)
        results[f"lambda_{lam}"] = r
        logger.info(f"    alpha={r['alpha_mean']:.3f}+/-{r['alpha_std']:.3f}  "
                   f"beta={r['beta_mean']:.3f}+/-{r['beta_std']:.3f}  "
                   f"gamma={r['gamma_mean']:.3f}+/-{r['gamma_std']:.3f}  "
                   f"spot={r['spot_mean']:.3f}")

    filepath = os.path.join(RESULTS_DIR, "cvar_ratios_analysis.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"  结果已保存: {filepath}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("风险厌恶对交易比例的影响:")
    logger.info("=" * 70)
    logger.info(f"{'lambda':<10} {'alpha':>12} {'beta':>12} {'gamma':>12} {'现货暴露':>12} {'均成本':>12}")
    logger.info("-" * 70)
    for lam in lambdas:
        r = results[f"lambda_{lam}"]
        logger.info(f"{lam:<10.2f} {r['alpha_mean']:>6.3f}+/-{r['alpha_std']:.3f} "
                   f"{r['beta_mean']:>6.3f}+/-{r['beta_std']:.3f} "
                   f"{r['gamma_mean']:>6.3f}+/-{r['gamma_std']:.3f} "
                   f"{r['spot_mean']:>6.3f}  {r['avg_cost']:>10.0f}")
