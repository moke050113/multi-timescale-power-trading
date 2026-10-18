"""
任务书对照分析脚本
1. 分析 DRL 学到的 alpha/beta/gamma 比例
2. 对比固定比例 vs DRL 动态比例的差异
3. 生成任务书完整性检查报告
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator
from src.utils.config import PRICE_SCENARIOS, FIXED_RATIOS
from src.utils.logger import setup_logger

logger = setup_logger("analysis")


def analyze_learned_ratios(scenario_name, load_scenario="flex moderate", num_eval=10, seed=42):
    """
    运行评估并记录 DRL 选择的 alpha/beta/gamma
    """
    env = TradingEnvironment(
        scenario_name=scenario_name,
        load_scenario=load_scenario,
        seed=seed,
    )

    # 创建未训练的智能体 (随机策略) 作为对照
    obs, _ = env.reset()
    state_dim = env.observation_space.shape[0]

    # 注意: 这里用的是未训练的智能体, 只能看随机策略的比例
    # 要分析训练后的比例, 需要加载训练好的模型
    # 当前先分析固定比例和随机策略的比例分布

    fixed_ratios = []
    random_ratios = []

    # 固定比例
    fixed_result = env.get_fixed_ratio_cost()
    fixed_pos = fixed_result["positions"]

    # 随机策略比例 (多次运行取均值)
    for _ in range(num_eval):
        state, _ = env.reset()
        while True:
            action = env.action_space.sample()
            state, r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        pos = info["positions"]
        random_ratios.append({
            "alpha": pos["alpha"],
            "beta": pos["beta"],
            "gamma": pos["gamma"],
            "spot": pos["spot_exposure"],
        })

    random_mean = {k: np.mean([r[k] for r in random_ratios]) for k in ["alpha", "beta", "gamma", "spot"]}
    random_std = {k: np.std([r[k] for r in random_ratios]) for k in ["alpha", "beta", "gamma", "spot"]}

    return {
        "fixed": fixed_pos,
        "random_mean": random_mean,
        "random_std": random_std,
        "fixed_cost": fixed_result["total_cost"],
    }


def gap_analysis():
    """
    对照任务书和研究设想, 检查项目完整性
    """
    gaps = {
        "completed": [],
        "partial": [],
        "missing": [],
    }

    # === 任务书要求 ===
    checks = [
        ("multi_timescale_framework", "多时间尺度滚动交易框架 (alpha/beta/gamma)", "completed",
         "已实现年度-月度-月内-现货四级市场, alpha/beta/gamma比例约束"),
        ("price_scenarios", "5种价格场景 (A~E)", "completed",
         "已实现5种典型价格场景, 覆盖年度低价/高价/价格接近/现货波动/极端风险"),
        ("ev_load_model", "电动汽车负荷模型", "completed",
         "已实现EV集群模型: 充放电功率/SOC约束/到达离开时间"),
        ("industrial_load", "工业可转移负荷模型", "completed",
         "已实现: 可转移比例/转移窗口/转移惩罚/最低负荷率"),
        ("load_scenarios", "4种负荷场景", "completed",
         "柔性充裕/柔性适中/柔性受限/纯刚性"),
        ("sac_agent", "SAC深度强化学习智能体", "completed",
         "已实现完整SAC: twin Q/Gaussian policy/auto temperature"),
        ("fixed_baseline", "固定比例策略基准", "completed",
         "alpha=0.80, beta=0.90, gamma=0.95"),
        ("random_baseline", "随机策略基准", "completed",
         "5场景各100 episodes随机策略成本"),
        ("cvar_risk", "CVaR风险度量", "completed",
         "已实现CVaR计算和lambda参数敏感性分析"),
        ("training_curves", "训练曲线可视化", "completed",
         "成本曲线/评估点/滑动均值"),
        ("strategy_comparison", "策略对比图表", "completed",
         "DRL vs 固定 vs 随机, 热力图, 改进汇总"),
        ("flexible_load_integration", "柔性负荷接入交易环境", "completed",
         "v3环境: EV+工业模型直接驱动柔性动作, 物理约束"),
        ("cost_risk_tradeoff", "成本-风险权衡分析", "completed",
         "CVaR lambda=0~0.5敏感性, 场景D"),

        ("multi_agent_coordination", "多智能体协调 (每尺度独立智能体)", "partial",
         "已实现双智能体框架但降级为单智能体模式, 信用分配问题未解决"),
        ("learned_ratios_analysis", "DRL学到的alpha/beta/gamma分析", "partial",
         "有比例约束机制, 但未系统分析训练后智能体选择的比例规律"),
        ("static_optimization", "静态一次性优化策略对比", "missing",
         "任务书要求: 固定/静态/滚动/动态比例/考虑风险 5种策略对比"),
        ("spot_exposure_constraint", "现货暴露约束 delta", "missing",
         "研究设想提到 1-gamma <= delta, 当前未实现显式约束"),
        ("sensitivity_market_bounds", "中长期比例约束敏感性分析", "missing",
         "研究设想要求: 不同最低覆盖率/不同现货上限对比例的影响"),
    ]

    for key, name, status, detail in checks:
        gaps[status].append({"key": key, "name": name, "detail": detail})

    return gaps


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("任务书对照分析")
    logger.info("=" * 70)

    # 1. 完整性检查
    gaps = gap_analysis()

    logger.info("")
    logger.info(f"已完成 ({len(gaps['completed'])} 项):")
    for item in gaps["completed"]:
        logger.info(f"  [OK] {item['name']}")
        logger.info(f"       {item['detail']}")

    logger.info("")
    logger.info(f"部分完成 ({len(gaps['partial'])} 项):")
    for item in gaps["partial"]:
        logger.info(f"  [~~] {item['name']}")
        logger.info(f"       {item['detail']}")

    logger.info("")
    logger.info(f"缺失 ({len(gaps['missing'])} 项):")
    for item in gaps["missing"]:
        logger.info(f"  [!!] {item['name']}")
        logger.info(f"       {item['detail']}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("建议优先补充:")
    logger.info("  1. 静态一次性优化策略 (计算上简单, 补全5种策略对比)")
    logger.info("  2. DRL学到的alpha/beta/gamma比例规律分析")
    logger.info("  3. 中长期比例约束敏感性分析")
    logger.info("=" * 70)
