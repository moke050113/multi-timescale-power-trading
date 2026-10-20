"""
交易结算引擎

会计口径 (v1.1, 修复双重计量):
    Q        = 全年总需求电量
    d        = Q / num_months = 每月需求电量
    α        = 年度合同覆盖率 (全年一次成交, 覆盖 α·Q)
    β_m      = 第 m 月的月度合同覆盖率 (0 ≤ α ≤ β_m ≤ 1)
    γ_m      = 第 m 月的中长期累计覆盖率 (0 ≤ α ≤ β_m ≤ γ_m ≤ 1)

    各阶段**新增**成交电量:
        Q_year     = α · Q                    (全年, 只成交一次)
        Q_month,m  = (β_m − α) · d            (本月月度新增)
        Q_day,m    = (γ_m − β_m) · d          (本月月内新增)
        Q_spot,m   = (1 − γ_m) · d            (本月现货敞口)

    恒等式:  α·Q + Σ_m[(β_m−α)·d + (γ_m−β_m)·d + (1−γ_m)·d] = Q

因此平价市场下总成本恰为 Q × price, 不会出现重复计量。
已成交合同不可撤销: 月度/月内增量由覆盖率单调性保证非负;
β_m、γ_m 是"本月"覆盖率, 跨月下降属于不同交割月, 不推翻已签合同。
"""
import numpy as np
from typing import Dict

from ..utils.config import TOTAL_DEMAND, SIM_PARAMS


class SettlementEngine:
    """结算引擎：根据交易比例和市场价格计算各阶段购电成本与电量。"""

    def __init__(
        self,
        total_demand: float = TOTAL_DEMAND,
        num_months: int = SIM_PARAMS["num_months"],
    ):
        self.total_demand = float(total_demand)
        self.num_months = int(num_months)
        self.monthly_demand = self.total_demand / self.num_months

        self._clear_state()

    def _clear_state(self):
        # 仓位: alpha 全年有效; beta/gamma 为"当前交割月"覆盖率
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.annual_settled = False

        # 成本 / 电量跟踪
        self.cost_annual = 0.0
        self.cost_monthly = 0.0
        self.cost_intraday = 0.0
        self.cost_spot = 0.0
        self.energy_annual = 0.0
        self.energy_monthly = 0.0
        self.energy_intraday = 0.0
        self.energy_spot = 0.0

        self.history = []

    def reset(self):
        """重置所有仓位、成本与电量。"""
        self._clear_state()

    def settle_annual(self, alpha: float, annual_price: float) -> float:
        """年度交易结算: 成交 α·Q, 全年只允许一次。"""
        assert 0 <= alpha <= 1, f"α 必须在 [0,1] 范围内, 当前: {alpha}"
        assert not self.annual_settled, "年度合同已成交, 同一回合不得重复计量年度电量"

        self.alpha = float(alpha)
        q_year = self.alpha * self.total_demand
        cost = float(annual_price) * q_year

        self.cost_annual = cost
        self.energy_annual = q_year
        self.annual_settled = True

        self._record("annual", alpha=self.alpha, price=annual_price,
                     quantity=q_year, cost=cost)
        return cost

    def settle_monthly(self, beta: float, monthly_price: float, month_idx: int) -> float:
        """月度交易结算: 本月新增 (β_m − α)·d。"""
        self._check_month(month_idx)
        assert self.alpha <= beta <= 1, f"需满足 α≤β≤1, 当前: α={self.alpha}, β={beta}"

        delta_beta = float(beta) - self.alpha
        q_month = delta_beta * self.monthly_demand
        cost = float(monthly_price) * q_month

        # 本月覆盖率状态: 月内决策从 β_m 起步, 保证月内不回溯
        self.beta = float(beta)
        self.gamma = self.beta
        self.cost_monthly += cost
        self.energy_monthly += q_month

        self._record("monthly", beta=self.beta, delta_beta=delta_beta,
                     price=monthly_price, quantity=q_month, cost=cost,
                     month=int(month_idx))
        return cost

    def settle_intraday(self, gamma: float, intraday_price: float, month_idx: int) -> float:
        """月内交易结算: 本月新增 (γ_m − β_m)·d。"""
        self._check_month(month_idx)
        assert self.beta <= gamma <= 1, f"需满足 β≤γ≤1, 当前: β={self.beta}, γ={gamma}"

        delta_gamma = float(gamma) - self.beta
        q_day = delta_gamma * self.monthly_demand
        cost = float(intraday_price) * q_day

        self.gamma = float(gamma)
        self.cost_intraday += cost
        self.energy_intraday += q_day

        self._record("intraday", gamma=self.gamma, delta_gamma=delta_gamma,
                     price=intraday_price, quantity=q_day, cost=cost,
                     month=int(month_idx))
        return cost

    def settle_spot(self, spot_prices: np.ndarray, month_idx: int,
                    flexible_shift: np.ndarray = None) -> float:
        """
        现货结算: 本月敞口电量 (1 − γ_m)·d 按时段分配。

        Args:
            spot_prices: 当月各时段现货价格 (24,)
            month_idx: 月份索引
            flexible_shift: 各时段负荷调整量 (MWh), 正值=增载, 负值=减载。
                当前视为纯时序转移 (总电量不变), 只改变敞口电量在各时段的分布。
        """
        self._check_month(month_idx)
        spot_ratio = 1.0 - self.gamma
        q_spot_total = spot_ratio * self.monthly_demand

        hours = self.hours_per_day()
        base_load = np.full(hours, q_spot_total / hours)

        if flexible_shift is not None:
            adjusted_load = np.maximum(base_load + np.asarray(flexible_shift, dtype=float), 0.0)
        else:
            adjusted_load = base_load

        total_load = adjusted_load.sum()
        if total_load > 0:
            hourly_qty = adjusted_load / total_load * q_spot_total
        else:
            hourly_qty = base_load.copy() if q_spot_total > 0 else adjusted_load

        prices = np.asarray(spot_prices, dtype=float)
        cost = float(np.sum(prices * hourly_qty))

        self.cost_spot += cost
        self.energy_spot += float(hourly_qty.sum())

        self._record("spot", spot_ratio=spot_ratio, prices=prices.copy(),
                     quantities=hourly_qty, cost=cost,
                     quantity=float(hourly_qty.sum()), month=int(month_idx))
        return cost

    def _check_month(self, month_idx: int):
        assert self.annual_settled, "必须先完成年度交易结算"
        assert 0 <= month_idx < self.num_months, (
            f"month_idx 越界: {month_idx}, 有效范围 0~{self.num_months - 1}"
        )

    @staticmethod
    def hours_per_day():
        return 24

    def get_total_cost(self) -> float:
        """获取总购电成本。"""
        return self.cost_annual + self.cost_monthly + self.cost_intraday + self.cost_spot

    def get_cost_breakdown(self) -> Dict[str, float]:
        """获取各市场成本明细。"""
        return {
            "annual": self.cost_annual,
            "monthly": self.cost_monthly,
            "intraday": self.cost_intraday,
            "spot": self.cost_spot,
            "total": self.get_total_cost(),
        }

    def get_energy_breakdown(self) -> Dict[str, float]:
        """获取各市场成交电量明细 (MWh), 用于会计闭合检查。"""
        return {
            "annual": self.energy_annual,
            "monthly": self.energy_monthly,
            "intraday": self.energy_intraday,
            "spot": self.energy_spot,
            "total": (self.energy_annual + self.energy_monthly
                      + self.energy_intraday + self.energy_spot),
        }

    def get_positions(self) -> Dict[str, float]:
        """获取当前仓位状态 (β/γ 为当前交割月覆盖率)。"""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "spot_exposure": 1.0 - self.gamma,
            "monthly_demand": self.monthly_demand,
        }

    def _record(self, stage: str, **kwargs):
        """记录交易历史。"""
        record = {"stage": stage, **kwargs}
        self.history.append(record)

    def compute_cvar(self, spot_costs: list, eta: float = 0.95) -> float:
        """
        计算条件风险价值 (CVaR)。

        Args:
            spot_costs: 各月现货成本列表
            eta: 置信水平 (默认 0.95)

        Returns:
            CVaR 值
        """
        if len(spot_costs) < 2:
            return 0.0
        costs = np.array(spot_costs)
        var = np.percentile(costs, eta * 100)
        cvar = costs[costs >= var].mean() if np.any(costs >= var) else var
        return float(cvar)
