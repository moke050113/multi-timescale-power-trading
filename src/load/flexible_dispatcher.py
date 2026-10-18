"""
规则型柔性负荷调度器
基于现货价格曲线执行削峰填谷策略，无需 DRL 训练

策略:
  - 高价时段 (top 25%): EV 放电 + 工业减载
  - 低价时段 (bottom 25%): EV 充电 + 工业增载
  - 能量守恒: 工业转移量 = 减少量，EV 充电满足日需求
  - 受物理功率上下限约束
"""
import numpy as np
from typing import Tuple, Optional


class FlexibleDispatcher:
    """
    规则型柔性负荷调度器

    在现货阶段自动根据价格曲线调度 EV 和工业负荷，
    替代 DRL 智能体的 flexible_action 输出。
    """

    def __init__(self, price_percentile_high: float = 0.75,
                 price_percentile_low: float = 0.25):
        """
        Args:
            price_percentile_high: 高价时段阈值 (默认 top 25%)
            price_percentile_low: 低价时段阈值 (默认 bottom 25%)
        """
        self.pct_high = price_percentile_high
        self.pct_low = price_percentile_low

    def dispatch(
        self,
        spot_prices: np.ndarray,
        ev_model,
        ind_model,
        include_flexible: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        基于现货价格曲线执行柔性负荷调度

        Args:
            spot_prices: 24 时段现货价格 (元/MWh), shape=(24,)
            ev_model: EVLoadModel 实例
            ind_model: IndustrialLoadModel 实例
            include_flexible: 是否启用柔性调度

        Returns:
            (shift, penalty):
                shift: 各时段功率调整量 (MWh), shape=(24,)
                       正值=增加用电, 负值=减少用电
                penalty: 转移惩罚成本 (元)
        """
        hours = len(spot_prices)

        if not include_flexible:
            return np.zeros(hours), 0.0

        shift = np.zeros(hours)
        penalty = 0.0

        # ---- 1. EV 集群调度 ----
        ev_charge_upper, ev_discharge_upper = ev_model.get_cluster_profile()

        price_high_threshold = np.percentile(spot_prices, self.pct_high * 100)
        price_low_threshold = np.percentile(spot_prices, self.pct_low * 100)

        for h in range(hours):
            if spot_prices[h] >= price_high_threshold and ev_discharge_upper[h] > 0:
                # 高价时段: 放电 (减少用电)
                discharge = ev_discharge_upper[h] * 0.5  # 用 50% 放电能力
                shift[h] -= discharge / 1000.0  # kW → MWh

            elif spot_prices[h] <= price_low_threshold and ev_charge_upper[h] > 0:
                # 低价时段: 充电 (增加用电)
                charge = ev_charge_upper[h] * 0.5  # 用 50% 充电能力
                shift[h] += charge / 1000.0

        # ---- 2. 工业可转移负荷调度 ----
        ind_lower, ind_upper = ind_model.get_adjustment_bounds(spot_prices)

        # 高价时段减少工业负荷
        high_price_hours = np.where(spot_prices >= price_high_threshold)[0]
        total_reduced = 0.0

        for h in high_price_hours:
            reducible = abs(ind_lower[h]) * 0.5  # 用 50% 可减空间
            if reducible > 0:
                shift[h] -= reducible / 1000.0  # kW → MWh
                total_reduced += reducible
                penalty += reducible / 1000.0 * ind_model.transfer_cost

        # 低价时段增加工业负荷 (能量守恒)
        low_price_hours = np.where(spot_prices <= price_low_threshold)[0]
        remaining_to_add = total_reduced

        for h in low_price_hours:
            if remaining_to_add <= 0:
                break
            addable = min(remaining_to_add, ind_upper[h] * 0.5)
            if addable > 0:
                shift[h] += addable / 1000.0
                remaining_to_add -= addable
                penalty += addable / 1000.0 * ind_model.transfer_cost

        return shift, penalty

    def get_summary(self, spot_prices: np.ndarray, ev_model, ind_model) -> str:
        """返回调度策略摘要"""
        shift, penalty = self.dispatch(spot_prices, ev_model, ind_model)
        total_shift = np.abs(shift).sum()
        peak_reduction = -shift[spot_prices >= np.percentile(spot_prices, 75)].sum()
        valley_increase = shift[spot_prices <= np.percentile(spot_prices, 25)].sum()

        return (
            f"规则调度摘要:\n"
            f"  总调整量: {total_shift:.3f} MWh\n"
            f"  高峰削减: {peak_reduction:.3f} MWh\n"
            f"  低谷增加: {valley_increase:.3f} MWh\n"
            f"  转移惩罚: {penalty:.1f} 元"
        )
