"""
多时间尺度电力交易 Gym 环境 (v3.2 — 分层架构)
上层: SAC 智能体决策 α/β/γ 仓位比例 (动作空间 1 维)
下层: 规则型柔性负荷调度器 (FlexibleDispatcher, 现货阶段自动执行)

改进 (v3.2):
  - 动作空间从 2 维缩减为 1 维 (仅 ratio_raw)
  - 现货阶段由 FlexibleDispatcher 自动调度 EV + 工业负荷
  - 上层智能体专注于中长期合同仓位决策
  - 保留 ev_cap/ind_cap 作为状态信息供上层感知柔性容量
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple

from ..utils.config import (
    TOTAL_DEMAND, MARKET_BOUNDS, SIM_PARAMS, DRL_PARAMS
)
from ..load.ev_model import EVLoadModel
from ..load.industrial_load import IndustrialLoadModel
from ..load.load_scenarios import LoadScenarioBuilder
from ..load.flexible_dispatcher import FlexibleDispatcher
from .market_models import PriceScenarioGenerator
from .settlement import SettlementEngine


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class TradingEnvironment(gym.Env):
    """
    多时间尺度电力交易环境 v3.2 (分层架构)

    === 分层设计 ===
        上层 (SAC): 决策 α/β/γ 仓位比例, 动作空间 1 维
        下层 (规则): 现货阶段由 FlexibleDispatcher 自动调度柔性负荷

    === 动作空间 (1维) ===
        [ratio_raw] → sigmoid → 仓位比例 (用于 annual/monthly/intraday)
        现货阶段不使用智能体动作, 自动调用规则调度器

    === 状态空间 (13维) ===
        [α, β, γ,                                    # 仓位 (3)
         annual_p, monthly_p, intraday_p,             # 价格 (3)
         spot_mean, spot_std,                         # 现货统计 (2)
         month_sin, month_cos,                        # 时间编码 (2)
         remaining,                                   # 剩余比例 (1)
         ev_capacity, ind_capacity]                   # EV/工业可调容量 (2)
    """

    metadata = {"render_modes": []}

    STAGE_ANNUAL = 0
    STAGE_MONTHLY = 1
    STAGE_INTRADAY = 2
    STAGE_SPOT = 3

    def __init__(
        self,
        scenario_name: str = "A_年度低价后期上涨",
        load_scenario: str = "柔性适中",
        total_demand: float = TOTAL_DEMAND,
        include_flexible: bool = True,
        risk_lambda: float = DRL_PARAMS["risk_lambda"],
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.scenario_name = scenario_name
        self.load_scenario = load_scenario
        self.total_demand = total_demand
        self.include_flexible = include_flexible
        self.risk_lambda = risk_lambda

        self.num_months = SIM_PARAMS["num_months"]
        self.hours_per_day = SIM_PARAMS["hours_per_day"]

        # 子模块
        self.price_gen = PriceScenarioGenerator(scenario_name, seed=seed)
        self.settlement = SettlementEngine(total_demand)

        # 柔性负荷模型
        if include_flexible:
            self.load_builder = LoadScenarioBuilder(load_scenario)
            self.ev_model = self.load_builder.ev_model
            self.ind_model = self.load_builder.industrial_model
        else:
            self.load_builder = None
            self.ev_model = None
            self.ind_model = None

        # 下层规则调度器 (分层架构)
        self.dispatcher = FlexibleDispatcher()

        # 状态变量
        self.prices = None
        self.current_step = 0
        self.current_month = 0
        self.current_stage = self.STAGE_ANNUAL
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.monthly_spot_costs = []
        self.episode_cost = 0.0
        self.flexible_cost_saving = 0.0  # 柔性负荷节约的成本
        self.total_flexible_penalty = 0.0  # 柔性负荷转移惩罚

        # 动作空间: [ratio_raw] (1维, 仅仓位比例)
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(1,), dtype=np.float32
        )

        # 状态空间: 13维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.price_gen.rng = np.random.default_rng(seed)

        self.prices = self.price_gen.generate()
        self.settlement.reset()

        self.current_step = 0
        self.current_month = 0
        self.current_stage = self.STAGE_ANNUAL
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.monthly_spot_costs = []
        self.episode_cost = 0.0
        self.flexible_cost_saving = 0.0
        self.total_flexible_penalty = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        ratio_raw = sigmoid(float(action[0]))

        cost = 0.0
        flexible_penalty = 0.0
        info = {}

        if self.current_stage == self.STAGE_ANNUAL:
            new_alpha = ratio_raw * MARKET_BOUNDS["annual"]["max"]
            cost = self.settlement.settle_annual(new_alpha, self.prices["annual_price"])
            self.alpha = self.settlement.alpha
            self.current_stage = self.STAGE_MONTHLY
            self.current_step += 1
            info["stage"] = "annual"
            info["alpha"] = self.alpha

        elif self.current_stage == self.STAGE_MONTHLY:
            lo = self.alpha
            hi = MARKET_BOUNDS["monthly"]["max"]
            new_beta = lo + ratio_raw * (hi - lo)
            cost = self.settlement.settle_monthly(
                new_beta, self.prices["monthly_prices"][self.current_month], self.current_month
            )
            self.beta = self.settlement.beta
            self.current_stage = self.STAGE_INTRADAY
            self.current_step += 1
            info["stage"] = "monthly"
            info["beta"] = self.beta

        elif self.current_stage == self.STAGE_INTRADAY:
            lo = self.beta
            hi = MARKET_BOUNDS["intraday"]["max"]
            new_gamma = lo + ratio_raw * (hi - lo)
            cost = self.settlement.settle_intraday(
                new_gamma, self.prices["intraday_prices"][self.current_month], self.current_month
            )
            self.gamma = self.settlement.gamma
            self.current_stage = self.STAGE_SPOT
            self.current_step += 1
            info["stage"] = "intraday"
            info["gamma"] = self.gamma

        elif self.current_stage == self.STAGE_SPOT:
            spot_prices = self.prices["spot_prices"][self.current_month]

            # === 下层规则调度器自动执行柔性负荷调节 ===
            flexible_shift, flexible_penalty = self.dispatcher.dispatch(
                spot_prices, self.ev_model, self.ind_model, self.include_flexible
            )

            cost = self.settlement.settle_spot(
                spot_prices, self.current_month, flexible_shift
            )
            self.monthly_spot_costs.append(cost)
            self.total_flexible_penalty += flexible_penalty

            # 计算柔性负荷带来的成本节约
            # 对比: 无柔性调节时的现货成本
            no_flex_cost = np.sum(spot_prices * (self.total_demand * (1 - self.gamma) / 24))
            saving = max(0, no_flex_cost - cost - flexible_penalty)
            self.flexible_cost_saving += saving

            self.current_month += 1
            if self.current_month >= self.num_months:
                pass
            else:
                self.current_stage = self.STAGE_MONTHLY
            self.current_step += 1
            info["stage"] = "spot"
            info["spot_cost"] = cost
            info["flexible_penalty"] = flexible_penalty
            info["flexible_saving"] = saving

        self.episode_cost += cost

        # 归一化奖励 (单智能体用)
        reward = -(cost + flexible_penalty) / self.total_demand

        if self.risk_lambda > 0 and len(self.monthly_spot_costs) >= 3:
            cvar = self.settlement.compute_cvar(self.monthly_spot_costs, DRL_PARAMS["cvar_eta"])
            reward -= self.risk_lambda * cvar / self.total_demand

        # 分阶段奖励 (双智能体用)
        # 注意: 此时 self.current_stage 已被更新为下一阶段, 需用 info["stage"]
        executed_stage = info.get("stage")
        if executed_stage == "spot":
            # 现货阶段刚执行完 → 现货智能体奖励
            reward_spot = -(cost + flexible_penalty) / self.total_demand
            reward_midlong = 0.0
        else:
            # 中长期阶段 (annual/monthly/intraday) → 中长期智能体奖励
            reward_midlong = -cost / self.total_demand
            reward_spot = 0.0

        terminated = self.current_month >= self.num_months
        truncated = False

        # 合并基础信息和步骤信息
        final_info = self._get_info()
        final_info.update(info)
        final_info["reward_midlong"] = reward_midlong
        final_info["reward_spot"] = reward_spot

        return self._get_obs(), reward, terminated, truncated, final_info

    def _apply_real_flexible_load(self, flexible_action: float, spot_prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        基于真实物理模型的柔性负荷调节

        Args:
            flexible_action: [-1, 1] 调节系数
            spot_prices: 24时段现货价格

        Returns:
            (shift, penalty): 功率调整量 (MWh), 转移惩罚成本 (元)
        """
        if not self.include_flexible:
            return np.zeros(self.hours_per_day), 0.0

        shift = np.zeros(self.hours_per_day)
        penalty = 0.0

        # 1. EV 集群调节
        ev_charge_upper, ev_discharge_upper = self.ev_model.get_cluster_profile()
        # flexible_action > 0: 充电 (增加用电), < 0: 放电 (减少用电)
        if flexible_action > 0:
            # 充电: 在低价时段充电
            ev_shift = ev_charge_upper * flexible_action * 0.5  # 最多用50%充电能力
        else:
            # 放电: 在高价时段放电
            ev_shift = -ev_discharge_upper * abs(flexible_action) * 0.5

        # 转换为 MWh (kW → MWh, 每时段1小时)
        ev_shift_mwh = ev_shift / 1000.0
        shift += ev_shift_mwh

        # 2. 工业可转移负荷调节
        ind_lower, ind_upper = self.ind_model.get_adjustment_bounds(spot_prices)
        # flexible_action > 0: 向低价时段转移 (减少高价时段负荷)
        if flexible_action > 0:
            # 减少高价时段负荷
            price_rank = np.argsort(-spot_prices)
            transfer_amount = abs(flexible_action) * self.ind_model.max_transferable * 0.3
            for idx in price_rank:
                reducible = min(transfer_amount, abs(ind_lower[idx]))
                if reducible > 0:
                    shift[idx] -= reducible / 1000.0  # kW → MWh
                    transfer_amount -= reducible
                    penalty += reducible / 1000.0 * self.ind_model.transfer_cost
                    if transfer_amount <= 0:
                        break
        elif flexible_action < 0:
            # 增加低价时段负荷
            price_rank = np.argsort(spot_prices)
            transfer_amount = abs(flexible_action) * self.ind_model.max_transferable * 0.3
            for idx in price_rank:
                addable = min(transfer_amount, ind_upper[idx])
                if addable > 0:
                    shift[idx] += addable / 1000.0
                    transfer_amount -= addable
                    penalty += addable / 1000.0 * self.ind_model.transfer_cost
                    if transfer_amount <= 0:
                        break

        return shift, penalty

    def _get_obs(self) -> np.ndarray:
        month = min(self.current_month, self.num_months - 1)
        annual_p = self.prices["annual_price"] if self.prices else 0
        monthly_p = self.prices["monthly_prices"][month] if self.prices else 0
        intraday_p = self.prices["intraday_prices"][month] if self.prices else 0
        spot_mean = self.prices["spot_prices"][month].mean() if self.prices else 0
        spot_std = self.prices["spot_prices"][month].std() if self.prices else 0

        month_norm = month / self.num_months
        month_sin = np.sin(2 * np.pi * month_norm)
        month_cos = np.cos(2 * np.pi * month_norm)
        remaining = 1.0 - self.gamma

        # EV 和工业负荷的可调容量
        if self.include_flexible:
            ev_cap = self.ev_model.get_daily_energy_demand()  # MWh
            ind_cap = self.ind_model.get_transfer_capacity()["max_transfer_energy_per_window"] / 1000.0  # MWh
        else:
            ev_cap = 0.0
            ind_cap = 0.0

        obs = np.array([
            self.alpha, self.beta, self.gamma,
            annual_p / 1000.0, monthly_p / 1000.0, intraday_p / 1000.0,
            spot_mean / 1000.0, spot_std / 1000.0,
            month_sin, month_cos,
            remaining,
            ev_cap / self.total_demand,
            ind_cap / self.total_demand,
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> Dict:
        return {
            "step": self.current_step,
            "month": self.current_month,
            "stage": self.current_stage,
            "positions": self.settlement.get_positions(),
            "cost_breakdown": self.settlement.get_cost_breakdown(),
            "total_cost": self.settlement.get_total_cost(),
            "episode_cost": self.episode_cost,
            "flexible_cost_saving": self.flexible_cost_saving,
            "total_flexible_penalty": self.total_flexible_penalty,
            "load_scenario": self.load_scenario,
        }

    def get_fixed_ratio_cost(self) -> Dict:
        """固定比例策略基准 (无柔性负荷)"""
        from ..utils.config import FIXED_RATIOS
        self.reset()
        total_cost = 0.0
        alpha = FIXED_RATIOS["alpha"]
        total_cost += self.settlement.settle_annual(alpha, self.prices["annual_price"])
        self.alpha = self.settlement.alpha

        for m in range(self.num_months):
            beta = FIXED_RATIOS["beta"]
            total_cost += self.settlement.settle_monthly(
                beta, self.prices["monthly_prices"][m], m
            )
            self.beta = self.settlement.beta

            gamma = FIXED_RATIOS["gamma"]
            total_cost += self.settlement.settle_intraday(
                gamma, self.prices["intraday_prices"][m], m
            )
            self.gamma = self.settlement.gamma

            spot_cost = self.settlement.settle_spot(self.prices["spot_prices"][m], m)
            total_cost += spot_cost

        return {
            "total_cost": total_cost,
            "breakdown": self.settlement.get_cost_breakdown(),
            "positions": self.settlement.get_positions(),
        }
