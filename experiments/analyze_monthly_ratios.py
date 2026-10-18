"""
月度差异化比例分析
训练DRL后, 按月汇总alpha/beta/gamma, 分析不同月份的比例差异
回答研究设想问题: "不同月份下的最优交易比例 alpha_m, beta_m, gamma_m"
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import PRICE_SCENARIOS
from src.utils.logger import setup_logger

logger = setup_logger("monthly_ratios")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def train_and_record_monthly(scenario_name, max_episodes=1000, seed=42):
    """训练并记录最后200轮每月的alpha/beta/gamma"""
    env = TradingEnvironment(scenario_name=scenario_name, include_flexible=True, seed=seed)
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    coordinator = MultiScaleCoordinator(state_dim=state_dim, action_dim=2, use_dual_agent=False)

    monthly_ratios = {m: {"alphas": [], "betas": [], "gammas": []} for m in range(12)}

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_month_data = {m: {} for m in range(12)}

        while True:
            action = coordinator.select_action(state, env.current_stage)
            ns, r, term, trunc, info = env.step(action)
            coordinator.store_transition(state, action, r, ns, term, env.current_stage)
            coordinator.update()

            month = info.get("month", 0)
            stage_name = info.get("stage", "")
            if stage_name == "annual":
                ep_month_data[0]["alpha"] = info["positions"]["alpha"]
            elif stage_name == "monthly" and month < 12:
                ep_month_data[month]["beta"] = info["positions"]["beta"]
            elif stage_name == "intraday" and month < 12:
                ep_month_data[month]["gamma"] = info["positions"]["gamma"]

            state = ns
            if term or trunc:
                break

        if ep > max_episodes - 200:
            for m in range(12):
                if "alpha" in ep_month_data[m]:
                    monthly_ratios[m]["alphas"].append(ep_month_data[m]["alpha"])
                if "beta" in ep_month_data[m]:
                    monthly_ratios[m]["betas"].append(ep_month_data[m]["beta"])
                if "gamma" in ep_month_data[m]:
                    monthly_ratios[m]["gammas"].append(ep_month_data[m]["gamma"])

        if ep % 200 == 0:
            logger.info(f"    Ep {ep}: cost={info['total_cost']:.0f}")

    result = {}
    for m in range(12):
        d = monthly_ratios[m]
        result[m] = {
            "alpha": float(np.mean(d["alphas"])) if d["alphas"] else 0,
            "alpha_std": float(np.std(d["alphas"])) if d["alphas"] else 0,
            "beta": float(np.mean(d["betas"])) if d["betas"] else 0,
            "beta_std": float(np.std(d["betas"])) if d["betas"] else 0,
            "gamma": float(np.mean(d["gammas"])) if d["gammas"] else 0,
            "gamma_std": float(np.std(d["gammas"])) if d["gammas"] else 0,
        }
    return result


if __name__ == "__main__":
    scenarios = list(PRICE_SCENARIOS.keys())[:3]

    logger.info("=" * 70)
    logger.info("月度差异化比例分析 (3场景, 各1000 episodes)")
    logger.info("=" * 70)

    all_results = {}
    for scenario_name in scenarios:
        short = scenario_name.split("_")[0]
        logger.info(f"  场景{short} ...")
        result = train_and_record_monthly(scenario_name, max_episodes=1000, seed=42)
        all_results[short] = result

        logger.info(f"  {'月份':<6} {'alpha':>12} {'beta':>12} {'gamma':>12} {'spot':>8}")
        logger.info(f"  " + "-" * 55)
        for m in range(12):
            r = result[m]
            spot = 1.0 - r["gamma"]
            logger.info(f"  {m+1:<6} {r['alpha']:>6.3f}+/-{r['alpha_std']:.3f} "
                       f"{r['beta']:>6.3f}+/-{r['beta_std']:.3f} "
                       f"{r['gamma']:>6.3f}+/-{r['gamma_std']:.3f} "
                       f"{spot:>6.3f}")
        logger.info("")

    filepath = os.path.join(RESULTS_DIR, "monthly_ratios_analysis.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {filepath}")
