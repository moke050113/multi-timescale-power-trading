"""
补充实验: 静态优化策略 + DRL比例分析
1. 静态一次性优化: 遍历所有alpha/beta/gamma组合, 找最优固定比例
2. DRL比例分析: 训练后观察智能体选择的alpha/beta/gamma
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import PRICE_SCENARIOS, MARKET_BOUNDS
from src.utils.logger import setup_logger

logger = setup_logger("supplement")


def static_optimization(scenario_name, seed=42):
    """
    静态一次性优化: 网格搜索最优固定 (alpha, beta, gamma) 组合
    对比: 固定比例 (0.80, 0.90, 0.95) vs 最优静态比例
    """
    env = TradingEnvironment(scenario_name=scenario_name, include_flexible=False, seed=seed)

    best_cost = float("inf")
    best_combo = None
    results = []

    # 网格搜索
    alphas = np.arange(0.5, 0.91, 0.05)
    betas = np.arange(0.7, 0.96, 0.05)
    gammas = np.arange(0.85, 1.01, 0.05)

    total = len(alphas) * len(betas) * len(gammas)
    count = 0

    for a in alphas:
        for b in betas:
            if b < a:
                continue
            for g in gammas:
                if g < b:
                    continue

                env.reset(seed=seed)
                cost = 0.0
                cost += env.settlement.settle_annual(a, env.prices["annual_price"])
                env.alpha = env.settlement.alpha

                for m in range(env.num_months):
                    cost += env.settlement.settle_monthly(
                        b, env.prices["monthly_prices"][m], m
                    )
                    env.beta = env.settlement.beta
                    cost += env.settlement.settle_intraday(
                        g, env.prices["intraday_prices"][m], m
                    )
                    env.gamma = env.settlement.gamma
                    cost += env.settlement.settle_spot(env.prices["spot_prices"][m], m)

                total_cost = env.settlement.get_total_cost()
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_combo = (a, b, g)

                results.append({"alpha": a, "beta": b, "gamma": g, "cost": total_cost})
                count += 1

    return {
        "best_combo": best_combo,
        "best_cost": best_cost,
        "total_combos_evaluated": count,
        "all_results": results,
    }


def analyze_drl_ratios(scenario_name, load_scenario="柔性适中",
                       max_episodes=500, num_eval=5, seed=42):
    """
    训练DRL并分析学到的alpha/beta/gamma
    """
    env = TradingEnvironment(
        scenario_name=scenario_name,
        load_scenario=load_scenario,
        seed=seed,
    )
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]

    coordinator = MultiScaleCoordinator(state_dim=state_dim, action_dim=2, use_dual_agent=False)

    # 训练
    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        while True:
            action = coordinator.select_action(state, env.current_stage)
            ns, r, term, trunc, info = env.step(action)
            coordinator.store_transition(state, action, r, ns, term, env.current_stage)
            coordinator.update()
            state = ns
            if term or trunc:
                break

    # 训练, 并记录最后几个episode的比例
    all_ratios = []
    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_ratios = {}
        while True:
            action = coordinator.select_action(state, env.current_stage)
            ns, r, term, trunc, info = env.step(action)
            coordinator.store_transition(state, action, r, ns, term, env.current_stage)
            coordinator.update()
            # 记录比例
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
        # 保存最后200个episode的比例
        if ep > max_episodes - 200:
            all_ratios.append(ep_ratios)
        # 每200轮打印进度
        if ep % 200 == 0:
            logger.info(f"    Ep {ep}: cost={info['total_cost']:.0f}, "
                       f"alpha={ep_ratios.get('alpha', 0):.3f}, "
                       f"beta={ep_ratios.get('beta', 0):.3f}, "
                       f"gamma={ep_ratios.get('gamma', 0):.3f}")

    # 汇总
    alphas = [r.get("alpha", 0) for r in all_ratios]
    betas = [r.get("beta", 0) for r in all_ratios]
    gammas = [r.get("gamma", 0) for r in all_ratios]
    spots = [1.0 - r.get("gamma", 0) for r in all_ratios]

    return {
        "alpha_mean": np.mean(alphas), "alpha_std": np.std(alphas),
        "beta_mean": np.mean(betas), "beta_std": np.std(betas),
        "gamma_mean": np.mean(gammas), "gamma_std": np.std(gammas),
        "spot_mean": np.mean(spots), "spot_std": np.std(spots),
        "final_cost": info["total_cost"],
    }


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    # === 实验4: 静态优化 vs 固定比例 vs DRL ===
    logger.info("=" * 70)
    logger.info("实验4: 静态一次性优化 (网格搜索最优固定比例)")
    logger.info("=" * 70)

    static_results = {}
    for scenario_name in list(PRICE_SCENARIOS.keys())[:3]:
        short = scenario_name.split("_")[0]
        logger.info(f"  {short} ...")
        result = static_optimization(scenario_name, seed=42)
        static_results[short] = result
        a, b, g = result["best_combo"]
        logger.info(f"    最优: alpha={a:.2f}, beta={b:.2f}, gamma={g:.2f}")
        logger.info(f"    成本: {result['best_cost']:.0f} (共搜索 {result['total_combos_evaluated']} 种组合)")

    # 保存结果
    filepath = os.path.join(RESULTS_DIR, "static_optimization.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({k: {"best_combo": v["best_combo"], "best_cost": v["best_cost"],
                       "total_combos": v["total_combos_evaluated"]}
                  for k, v in static_results.items()}, f, ensure_ascii=False, indent=2)
    logger.info(f"  结果已保存: {filepath}")

    # === 实验5: DRL 学到的比例分析 ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("实验5: DRL 学到的 alpha/beta/gamma 比例")
    logger.info("=" * 70)

    ratio_results = {}
    for scenario_name in list(PRICE_SCENARIOS.keys())[:3]:
        short = scenario_name.split("_")[0]
        logger.info(f"  {short} ...")
        result = analyze_drl_ratios(scenario_name, max_episodes=1000, num_eval=10, seed=42)
        ratio_results[short] = result
        logger.info(f"    alpha={result['alpha_mean']:.3f}+/-{result['alpha_std']:.3f}")
        logger.info(f"    beta ={result['beta_mean']:.3f}+/-{result['beta_std']:.3f}")
        logger.info(f"    gamma={result['gamma_mean']:.3f}+/-{result['gamma_std']:.3f}")
        logger.info(f"    spot ={result['spot_mean']:.3f}+/-{result['spot_std']:.3f}")

    filepath = os.path.join(RESULTS_DIR, "drl_ratios_analysis.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ratio_results, f, ensure_ascii=False, indent=2)
    logger.info(f"  结果已保存: {filepath}")

    # === 汇总对比 ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("策略对比汇总")
    logger.info("=" * 70)
    logger.info(f"{'场景':<6} {'固定比例':>10} {'静态最优':>10} {'DRL':>10} {'DRL改进vs固定':>14}")
    logger.info("-" * 55)

    for short in static_results:
        # 找到完整场景名
        full_name = None
        for sn in PRICE_SCENARIOS:
            if sn.startswith(f"{short}_"):
                full_name = sn
                break

        env = TradingEnvironment(scenario_name=full_name, include_flexible=False, seed=42)
        fixed = env.get_fixed_ratio_cost()
        fixed_cost = fixed["total_cost"]

        static_cost = static_results[short]["best_cost"]
        drl_cost = ratio_results[short]["final_cost"]
        imp = (fixed_cost - drl_cost) / fixed_cost * 100

        logger.info(f"{short:<6} {fixed_cost:>10.0f} {static_cost:>10.0f} {drl_cost:>10.0f} {imp:>+13.1f}%")
