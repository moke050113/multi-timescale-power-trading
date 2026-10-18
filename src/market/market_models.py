"""
价格场景生成器
根据研究设想中的5种典型场景生成各阶段市场价格
"""
import numpy as np
from typing import Dict, Optional
from ..utils.config import PRICE_SCENARIOS, SIM_PARAMS


class PriceScenarioGenerator:
    """
    生成年度-月度-月内-现货四级市场的价格序列

    场景说明:
        A: 年度低价、后期价格上涨  → 模型应倾向提高 α
        B: 年度高价、后期价格下降  → 模型应倾向降低 α
        C: 各阶段价格水平接近     → 风险约束主导比例配置
        D: 现货市场高波动         → 模型应提高 γ (中长期覆盖率)
        E: 月度低价、现货极端风险  → 模型在低价机会与高价风险间权衡
    """

    def __init__(self, scenario_name: str, seed: Optional[int] = None):
        if scenario_name not in PRICE_SCENARIOS:
            raise ValueError(f"未知场景: {scenario_name}, 可选: {list(PRICE_SCENARIOS.keys())}")

        self.scenario_name = scenario_name
        self.params = PRICE_SCENARIOS[scenario_name]
        self.rng = np.random.default_rng(seed)

        self.num_months = SIM_PARAMS["num_months"]
        self.hours_per_day = SIM_PARAMS["hours_per_day"]
        self.days_per_month = SIM_PARAMS["days_per_month"]

    def generate(self) -> Dict[str, np.ndarray]:
        """
        生成完整的价格序列

        Returns:
            dict: {
                "annual_price":   float,           # 年度合同价格 (元/MWh)
                "monthly_prices": np.ndarray[12],   # 各月月度合同价格
                "intraday_prices": np.ndarray[12],  # 各月月内交易价格 (取月均)
                "spot_prices":    np.ndarray[12, 24], # 各月各时段现货价格
            }
        """
        p = self.params

        # 年度价格 — 固定
        annual_price = float(p["annual"])

        # 月度价格 — 在基准附近加入小幅随机波动
        monthly_base = p["monthly"]
        monthly_prices = np.array([
            monthly_base + self.rng.normal(0, monthly_base * 0.03)
            for _ in range(self.num_months)
        ])

        # 月内价格 — 在月度基础上再加波动
        intraday_base = p["intraday"]
        intraday_prices = np.array([
            intraday_base + self.rng.normal(0, intraday_base * 0.04)
            for _ in range(self.num_months)
        ])

        # 现货价格 — 各月各时段 (月×24h)，对数正态分布模拟尖峰
        spot_mean = p["spot_mean"]
        spot_std = p["spot_std"]
        spot_prices = np.zeros((self.num_months, self.hours_per_day))
        for m in range(self.num_months):
            for h in range(self.hours_per_day):
                # 高峰时段 (9-12, 18-21) 价格上浮
                hour_factor = 1.0
                if 9 <= h <= 12 or 18 <= h <= 21:
                    hour_factor = 1.2 + self.rng.uniform(0, 0.15)
                elif 0 <= h <= 6:
                    hour_factor = 0.7 + self.rng.uniform(-0.1, 0.05)

                price = max(0, spot_mean * hour_factor + self.rng.normal(0, spot_std))
                spot_prices[m, h] = price

        return {
            "annual_price": annual_price,
            "monthly_prices": monthly_prices,
            "intraday_prices": intraday_prices,
            "spot_prices": spot_prices,
        }

    def get_scenario_info(self) -> str:
        """返回场景描述信息"""
        p = self.params
        return (
            f"场景 [{self.scenario_name}]\n"
            f"  年度: {p['annual']}  月度: {p['monthly']}  "
            f"月内: {p['intraday']}  现货均值: {p['spot_mean']}±{p['spot_std']}"
        )

    @staticmethod
    def list_scenarios():
        """列出所有可用场景"""
        return list(PRICE_SCENARIOS.keys())
