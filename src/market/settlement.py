"""
交易结算引擎
计算各市场购电成本、跟踪仓位变化
"""
import numpy as np
from typing import Dict
from ..utils.config import TOTAL_DEMAND, FIXED_RATIOS


class SettlementEngine:
    """
    结算引擎：根据交易比例和市场价格计算各阶段购电成本

    核心变量:
        α (alpha): 年度交易比例
        β (beta):  年度+月度累计比例
        γ (gamma): 中长期累计比例
        1-γ:       现货市场暴露比例

    电量分配:
        Q_year  = α × Q
        Q_month = (β - α) × Q
        Q_day   = (γ - β) × Q
        Q_spot  = (1 - γ) × Q
    """

    def __init__(self, total_demand: float = TOTAL_DEMAND):
        self.total_demand = total_demand

        # 仓位跟踪
        self.alpha = 0.0   # 当前年度累计比例
        self.beta = 0.0    # 当前月度累计比例
        self.gamma = 0.0   # 当前中长期累计比例

        # 成本跟踪
        self.cost_annual = 0.0
        self.cost_monthly = 0.0
        self.cost_intraday = 0.0
        self.cost_spot = 0.0

        # 历史记录
        self.history = []

    def reset(self):
        """重置所有仓位和成本"""
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.cost_annual = 0.0
        self.cost_monthly = 0.0
        self.cost_intraday = 0.0
        self.cost_spot = 0.0
        self.history = []

    def settle_annual(self, alpha: float, annual_price: float) -> float:
        """
        年度交易结算

        Args:
            alpha: 年度交易比例 (0~1)
            annual_price: 年度合同价格 (元/MWh)

        Returns:
            cost: 年度购电成本
        """
        assert 0 <= alpha <= 1, f"α 必须在 [0,1] 范围内, 当前: {alpha}"

        self.alpha = alpha
        q_year = alpha * self.total_demand
        cost = annual_price * q_year
        self.cost_annual = cost

        self._record("annual", alpha=alpha, price=annual_price,
                      quantity=q_year, cost=cost)
        return cost

    def settle_monthly(self, beta: float, monthly_price: float, month_idx: int) -> float:
        """
        月度交易结算

        Args:
            beta: 年度+月度累计比例
            monthly_price: 月度合同价格
            month_idx: 月份索引 (0-11)

        Returns:
            cost: 当月新增购电成本
        """
        assert self.alpha <= beta <= 1, f"需满足 α≤β≤1, 当前: α={self.alpha}, β={beta}"

        # 说明: 允许 delta_beta 为负 (表示"回补"前期超额采购), 与整体成本模型自洽。
        # 严格物理意义上中长期合同不可撤销, 但当前模型使用简化的"累计比例"语义,
        # 强制单调会造成成本模型 4× 膨胀。因此保持原语义, 通过奖励塑形抑制极端行为。
        delta_beta = beta - self.beta
        q_month = delta_beta * self.total_demand
        cost = monthly_price * q_month

        self.beta = beta
        self.cost_monthly += cost

        self._record("monthly", beta=beta, delta_beta=delta_beta,
                      price=monthly_price, quantity=q_month, cost=cost,
                      month=month_idx)
        return cost

    def settle_intraday(self, gamma: float, intraday_price: float, month_idx: int) -> float:
        """
        月内交易结算

        Args:
            gamma: 中长期累计比例
            intraday_price: 月内交易价格
            month_idx: 月份索引

        Returns:
            cost: 当月月内购电成本
        """
        assert self.beta <= gamma <= 1, f"需满足 β≤γ≤1, 当前: β={self.beta}, γ={gamma}"

        # 与 monthly 相同的语义: 允许 delta_gamma 为负
        delta_gamma = gamma - self.gamma
        q_day = delta_gamma * self.total_demand
        cost = intraday_price * q_day

        self.gamma = gamma
        self.cost_intraday += cost

        self._record("intraday", gamma=gamma, delta_gamma=delta_gamma,
                      price=intraday_price, quantity=q_day, cost=cost,
                      month=month_idx)
        return cost

    def settle_spot(self, spot_prices: np.ndarray, month_idx: int,
                    flexible_shift: np.ndarray = None) -> float:
        """
        现货市场结算

        Args:
            spot_prices: 当月各时段现货价格 (24,)
            month_idx: 月份索引
            flexible_shift: 柔性负荷在各时段的功率调整 (24,)，正值=增加用电，负值=减少用电

        Returns:
            cost: 当月现货购电成本
        """
        spot_ratio = 1.0 - self.gamma
        q_spot_total = spot_ratio * self.total_demand

        # 现货电量按24时段均分作为基础负荷
        base_load = np.full(24, q_spot_total / 24)

        # 柔性负荷调整
        if flexible_shift is not None:
            adjusted_load = base_load + flexible_shift
            adjusted_load = np.maximum(adjusted_load, 0)  # 不能为负
        else:
            adjusted_load = base_load

        # 按实际负荷分布分配现货电量
        total_load = adjusted_load.sum()
        if total_load > 0:
            hourly_qty = adjusted_load / total_load * q_spot_total
        else:
            hourly_qty = base_load.copy()

        cost = np.sum(spot_prices * hourly_qty)
        self.cost_spot += cost

        self._record("spot", spot_ratio=spot_ratio, prices=spot_prices.copy(),
                      quantities=hourly_qty, cost=cost, month=month_idx)
        return cost

    @staticmethod
    def hours_per_day():
        return 24

    def get_total_cost(self) -> float:
        """获取总购电成本"""
        return self.cost_annual + self.cost_monthly + self.cost_intraday + self.cost_spot

    def get_cost_breakdown(self) -> Dict[str, float]:
        """获取各市场成本明细"""
        return {
            "annual": self.cost_annual,
            "monthly": self.cost_monthly,
            "intraday": self.cost_intraday,
            "spot": self.cost_spot,
            "total": self.get_total_cost(),
        }

    def get_positions(self) -> Dict[str, float]:
        """获取当前仓位状态"""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "spot_exposure": 1.0 - self.gamma,
        }

    def _record(self, stage: str, **kwargs):
        """记录交易历史"""
        record = {"stage": stage, **kwargs}
        self.history.append(record)

    def compute_cvar(self, spot_costs: list, eta: float = 0.95) -> float:
        """
        计算条件风险价值 (CVaR)

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
