"""
多时间尺度电力交易 Gym 环境 (v3.3 — 双 DRL 智能体)

架构:
    中长期智能体 (SAC): 输出 ratio_raw → α/β/γ, 负责年度/月度/月内 3 个阶段
    现货智能体   (SAC): 输出 flexible_action ∈ [-1,1], 负责现货阶段柔性负荷调度

改进 (v3.3):
    - 动作空间从 1 维恢复为 2 维: [ratio_raw, flexible_action]
    - 现货阶段的柔性负荷由 DRL 智能体输出决定 (非规则调度器)
    - 分阶段独立奖励:
        reward_midlong = -contract_cost / total_demand
        reward_spot    = -(spot_cost + penalty) / (total_demand * max(spot_exposure, 0.05))
      现货奖励通过除以 spot_exposure 放大 5-20× 使其与中长期奖励同量级, 缓解信用分配问题
    - 保留 FlexibleDispatcher 作为对比基线 (纯规则)

状态空间 (13 维):
    [α, β, γ,                                    # 仓位 (3)
     annual_p, monthly_p, intraday_p,             # 价格 (3)
     spot_mean, spot_std,                         # 现货统计 (2)
     month_sin, month_cos,                        # 时间编码 (2)
     remaining,                                   # 现货暴露 (1)
     ev_capacity, ind_capacity]                   # 柔性可调容量 (2)
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
from .market_models import PriceScenarioGenerator
from .settlement import SettlementEngine


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class TradingEnvironment(gym.Env):
    """多时间尺度电力交易环境 v3.3 (双 DRL 智能体)"""

    metadata = {"render_modes": []}

    STAGE_ANNUAL = 0
    STAGE_MONTHLY = 1
    STAGE_INTRADAY = 2
    STAGE_SPOT = 3

    # 现货奖励归一化时下界保护: 避免 spot_exposure→0 时奖励爆炸
    MIN_SPOT_EXPOSURE = 0.05

    def __init__(
        self,
        scenario_name: str = "A_年度低价后期上涨",
        load_scenario: str = "柔性适中",
        total_demand: float = TOTAL_DEMAND,
        include_flexible: bool = True,
        risk_lambda: float = DRL_PARAMS["risk_lambda"],
        use_rule_dispatcher: bool = False,
        scale_spot_reward: bool = True,
        exposure_shaping: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.scenario_name = scenario_name
        self.load_scenario = load_scenario
        self.total_demand = total_demand
        self.include_flexible = include_flexible
        self.risk_lambda = risk_lambda
        # 消融/基线开关: True 时现货阶段用规则调度器代替 DRL 动作
        self.use_rule_dispatcher = use_rule_dispatcher
        # 消融开关: 现货奖励是否按暴露比例放大 (scale_spot_reward=False → 不除以 1-γ)
        self.scale_spot_reward = scale_spot_reward
        # 消融开关: 中长期 intraday 步是否加未来暴露成本塑形
        self.exposure_shaping = exposure_shaping

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

        # 可选规则调度器 (仅用于基线对比)
        if use_rule_dispatcher:
            from ..load.flexible_dispatcher import FlexibleDispatcher
            self.dispatcher = FlexibleDispatcher()
        else:
            self.dispatcher = None

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
        self.total_market_cost = 0.0        # 4 级市场累计成本
        self.total_flexible_penalty = 0.0   # 柔性转移惩罚累计
        self.flexible_cost_saving = 0.0     # 柔性负荷带来的成本节约 (相对无调节)

        # 动作空间: [ratio_raw (midlong), flexible_action (spot)]
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(2,), dtype=np.float32
        )

        # 状态空间: 13 维
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
        self.total_market_cost = 0.0
        self.total_flexible_penalty = 0.0
        self.flexible_cost_saving = 0.0

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------
    # 主 step 逻辑
    # ------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).flatten()
        ratio_raw = float(action[0])
        flexible_action = float(np.clip(action[1], -1.0, 1.0))

        cost = 0.0
        flexible_penalty = 0.0
        info = {}

        if self.current_stage == self.STAGE_ANNUAL:
            new_alpha = sigmoid(ratio_raw) * MARKET_BOUNDS["annual"]["max"]
            cost = self.settlement.settle_annual(new_alpha, self.prices["annual_price"])
            self.alpha = self.settlement.alpha
            self.current_stage = self.STAGE_MONTHLY
            self.current_step += 1
            info["stage"] = "annual"
            info["alpha"] = self.alpha

        elif self.current_stage == self.STAGE_MONTHLY:
            lo = self.alpha
            hi = MARKET_BOUNDS["monthly"]["max"]
            target_beta = lo + sigmoid(ratio_raw) * (hi - lo)
            # settlement 内部保证跨月不回溯 (delta_beta>=0)
            cost = self.settlement.settle_monthly(
                target_beta, self.prices["monthly_prices"][self.current_month], self.current_month
            )
            self.beta = self.settlement.beta
            self.current_stage = self.STAGE_INTRADAY
            self.current_step += 1
            info["stage"] = "monthly"
            info["beta"] = self.beta

        elif self.current_stage == self.STAGE_INTRADAY:
            lo = self.beta
            hi = MARKET_BOUNDS["intraday"]["max"]
            target_gamma = lo + sigmoid(ratio_raw) * (hi - lo)
            cost = self.settlement.settle_intraday(
                target_gamma, self.prices["intraday_prices"][self.current_month], self.current_month
            )
            self.gamma = self.settlement.gamma
            self.current_stage = self.STAGE_SPOT
            self.current_step += 1
            info["stage"] = "intraday"
            info["gamma"] = self.gamma

        elif self.current_stage == self.STAGE_SPOT:
            spot_prices = self.prices["spot_prices"][self.current_month]

            # 现货柔性调度: DRL 动作 or 规则基线
            if self.use_rule_dispatcher and self.dispatcher is not None:
                flexible_shift, flexible_penalty = self.dispatcher.dispatch(
                    spot_prices, self.ev_model, self.ind_model, self.include_flexible
                )
            else:
                flexible_shift, flexible_penalty = self._apply_real_flexible_load(
                    flexible_action, spot_prices
                )

            cost = self.settlement.settle_spot(
                spot_prices, self.current_month, flexible_shift
            )
            self.monthly_spot_costs.append(cost)
            self.total_flexible_penalty += flexible_penalty

            # 无柔性调节基准 (供分析)
            no_flex_cost = float(np.sum(spot_prices * (self.total_demand * (1 - self.gamma) / 24)))
            saving = max(0.0, no_flex_cost - cost - flexible_penalty)
            self.flexible_cost_saving += saving

            self.current_month += 1
            if self.current_month < self.num_months:
                self.current_stage = self.STAGE_MONTHLY
            self.current_step += 1
            info["stage"] = "spot"
            info["spot_cost"] = cost
            info["flexible_penalty"] = flexible_penalty
            info["flexible_saving"] = saving
            info["flexible_action"] = flexible_action

        # 累计成本
        self.episode_cost += cost
        self.total_market_cost += cost

        executed_stage = info.get("stage")

        # ---- 分阶段独立奖励 ----
        if executed_stage == "spot":
            spot_exposure = max(1.0 - self.gamma, self.MIN_SPOT_EXPOSURE)
            denom = self.total_demand * spot_exposure if self.scale_spot_reward else self.total_demand
            reward_spot = -(cost + flexible_penalty) / denom
            reward_midlong = 0.0
            reward_global = reward_spot
        else:
            reward_midlong = -cost / self.total_demand
            # 关键奖励塑形: 中长期智能体的 γ 决策会导致本月剩余电量流入现货市场,
            # 若不感知该成本, 智能体没有保持高覆盖率的动力 (credit assignment 反向问题)
            if executed_stage == "intraday" and self.exposure_shaping:
                # 使用场景的现货均价作为"未来成本预估", 避免完美预见具体月价格
                expected_spot = self.price_gen.params["spot_mean"]
                exposure_cost = (1.0 - self.gamma) * self.total_demand * expected_spot
                reward_midlong -= exposure_cost / self.total_demand
            reward_spot = 0.0
            reward_global = reward_midlong

        # CVaR 风险项: 只施加于中长期智能体 (它决定 γ → 暴露)
        if self.risk_lambda > 0 and len(self.monthly_spot_costs) >= 3 and executed_stage == "intraday":
            cvar = self.settlement.compute_cvar(self.monthly_spot_costs, DRL_PARAMS["cvar_eta"])
            risk_term = self.risk_lambda * cvar / self.total_demand
            reward_midlong -= risk_term
            reward_global -= risk_term

        terminated = self.current_month >= self.num_months
        truncated = False

        final_info = self._get_info()
        final_info.update(info)
        final_info["reward_midlong"] = reward_midlong
        final_info["reward_spot"] = reward_spot
        final_info["reward_global"] = reward_global

        return self._get_obs(), reward_global, terminated, truncated, final_info

    # ------------------------------------------------------------
    # 柔性负荷物理模型
    # ------------------------------------------------------------
    def _apply_real_flexible_load(self, flexible_action: float, spot_prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        基于 EV + 工业模型的柔性负荷调度

        Args:
            flexible_action: [-1, 1]
                > 0: 增加低价时段充电/增载 (把负荷移向低价时段)
                < 0: 减少高价时段用电/放电 (把负荷从高价时段移走)
            spot_prices: 24 时段现货价格

        Returns:
            (shift, penalty): 各时段功率调整量 (MWh), 转移惩罚成本 (元)
        """
        if not self.include_flexible:
            return np.zeros(self.hours_per_day), 0.0

        shift = np.zeros(self.hours_per_day)
        penalty = 0.0

        # ---- 1. EV 集群 ----
        ev_charge_upper, ev_discharge_upper = self.ev_model.get_cluster_profile()
        if flexible_action > 0:
            # 正动作: 充电 (增加负荷), 尽量落在低价时段
            charge = ev_charge_upper * flexible_action * 0.5
            shift += charge / 1000.0
        else:
            # 负动作: 放电 (减少负荷), 尽量落在高价时段
            discharge = ev_discharge_upper * abs(flexible_action) * 0.5
            shift -= discharge / 1000.0

        # ---- 2. 工业可转移负荷 ----
        ind_lower, ind_upper = self.ind_model.get_adjustment_bounds(spot_prices)
        transfer_amount = abs(flexible_action) * self.ind_model.max_transferable * 0.3

        if flexible_action > 0:
            # 增载: 从低价时段开始
            price_rank = np.argsort(spot_prices)
            for idx in price_rank:
                if transfer_amount <= 0:
                    break
                addable = min(transfer_amount, ind_upper[idx])
                if addable > 0:
                    shift[idx] += addable / 1000.0
                    transfer_amount -= addable
                    penalty += addable / 1000.0 * self.ind_model.transfer_cost
        elif flexible_action < 0:
            # 减载: 从高价时段开始
            price_rank = np.argsort(-spot_prices)
            for idx in price_rank:
                if transfer_amount <= 0:
                    break
                reducible = min(transfer_amount, abs(ind_lower[idx]))
                if reducible > 0:
                    shift[idx] -= reducible / 1000.0
                    transfer_amount -= reducible
                    penalty += reducible / 1000.0 * self.ind_model.transfer_cost

        return shift, penalty

    # ------------------------------------------------------------
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

        if self.include_flexible:
            ev_cap = self.ev_model.get_daily_energy_demand()
            ind_cap = self.ind_model.get_transfer_capacity()["max_transfer_energy_per_window"] / 1000.0
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
        # 统一成本口径: 包含柔性惩罚
        total_cost_including_penalty = (
            self.settlement.get_total_cost() + self.total_flexible_penalty
        )
        return {
            "step": self.current_step,
            "month": self.current_month,
            "stage": self.current_stage,
            "positions": self.settlement.get_positions(),
            "cost_breakdown": self.settlement.get_cost_breakdown(),
            "total_cost": total_cost_including_penalty,
            "market_cost": self.settlement.get_total_cost(),
            "flexible_penalty": self.total_flexible_penalty,
            "episode_cost": self.episode_cost,
            "flexible_cost_saving": self.flexible_cost_saving,
            "load_scenario": self.load_scenario,
        }

    def get_total_cost(self) -> float:
        """统一成本口径: 市场成本 + 柔性转移惩罚"""
        return self.settlement.get_total_cost() + self.total_flexible_penalty

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
