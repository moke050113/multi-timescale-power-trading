"""
多时间尺度电力交易 Gym 环境 (v3.5 — 设备执行闭环 + 偏差结算)

架构:
    中长期智能体 (SAC): 输出 ratio_raw → α/β_m/γ_m, 负责年度/月度/月内 3 个阶段
    柔性调度智能体 (SAC): 输出 flexible_action ∈ [-1,1] → 本月柔性强度, 负责现货阶段
        (它只决定"多大程度择优调度", 逐时段落点由执行层按已公布日前价与设备物理约束决定)

改进 (v3.5):
    - 现货阶段闭环: 合法观察 → 动作 → 设备实际执行 (EV SOC 积分 / 工业日任务守恒)
      → 得到真实网侧逐时段电量 → 与合同电量之差按现货结算 (缺口 1:1 买入, 过购折价回售)
    - 删除"调整后负荷重新归一化为固定现货总量"的做法; 需求形状由执行层给出
    - 物理日负荷量级一次性标定到合同电量 Q (标定系数只由中性剖面决定, 动作后不再缩放)

改进 (v3.4):
    - 结算按交割月记账: 年度 α·Q 只成交一次, 月度/月内按本月 R_m 的覆盖率增量结算,
      保证全年合同电量恒等于 Q (修复 3.25× 双重计量)
    - 奖励按各步决策电量基准归一化 (年度步 Q, 其余 R_m), 单位元/MWh

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
from ..load.flexible_executor import FlexibleExecutor
from .market_models import PriceScenarioGenerator
from .settlement import SettlementEngine


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class TradingEnvironment(gym.Env):
    """多时间尺度电力交易环境 v3.5 (双 SAC + 设备执行闭环)"""

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
        self.days_per_month = SIM_PARAMS["days_per_month"]

        # 世界种子: 同一 (world_seed, world 编号) 的价格实现可复现, 用于策略间配对评价
        self.world_seed = SIM_PARAMS["random_seed"] if seed is None else int(seed)
        self.current_world = None
        self.world_cursor = 0

        # 子模块
        self.price_gen = PriceScenarioGenerator(scenario_name, seed=seed)
        self.settlement = SettlementEngine(total_demand, num_months=self.num_months)

        # 柔性负荷模型: 无论是否启用柔性, 都构建设备模型 (设备任务是所有策略的公共硬条件);
        # include_flexible=False 时执行层把柔性强度强制为 0, 即"只保障、不择优"
        self.load_builder = LoadScenarioBuilder(load_scenario)
        self.ev_model = self.load_builder.ev_model
        self.ind_model = self.load_builder.industrial_model
        self.executor = FlexibleExecutor(
            self.ev_model, self.ind_model,
            include_flexible=include_flexible,
            hours=self.hours_per_day, days_per_month=self.days_per_month,
        )

        # 量级标定: 物理代表日负荷 (约 5 MWh/日) 与合同电量 Q (约 83 MWh/月) 不同尺度,
        # 用**中性剖面**一次性求出比例系数; 该系数与动作无关, 动作之后不再做任何重归一化。
        self.neutral_day_mwh = self.executor.neutral_profile()
        neutral_year_mwh = float(self.neutral_day_mwh.sum()) * self.days_per_month * self.num_months
        assert neutral_year_mwh > 0, "中性日净负荷为 0, 无法标定需求尺度"
        self.demand_scale = self.total_demand / neutral_year_mwh

        # 可选规则调度器 (仅用于基线对比): 只给柔性强度, 落点仍由执行层决定
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
        self.flexible_cost_saving = 0.0     # 柔性调度节约 (相对同月保障调度, 含惩罚)
        self.device_checks = []             # 逐月设备执行诊断 (达标/守恒)

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
            self.world_seed = int(seed)
            self.price_gen.rng = np.random.default_rng(seed)
            self.world_cursor = 0

        world = None if options is None else options.get("world")
        if world is None:
            self.prices = self.price_gen.generate()
            self.current_world = self.world_cursor
        else:
            self.prices = self.price_gen.generate(seed=[self.world_seed, int(world)])
            self.current_world = int(world)
        self.world_cursor += 1
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
        self.device_checks = []

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
        # 本次决策所属交割月的参考电量 (现货步会推进 current_month, 故需先取)
        basis_energy = self.settlement.reference_energy(self.current_month)

        if self.current_stage == self.STAGE_ANNUAL:
            new_alpha = sigmoid(ratio_raw) * MARKET_BOUNDS["annual"]["max"]
            cost = self.settlement.settle_annual(
                new_alpha, self.prices["annual_price"], signed_step=self.current_step
            )
            self.alpha = self.settlement.alpha
            self.current_stage = self.STAGE_MONTHLY
            self.current_step += 1
            info["stage"] = "annual"
            info["alpha"] = self.alpha

        elif self.current_stage == self.STAGE_MONTHLY:
            lo = self.alpha
            hi = MARKET_BOUNDS["monthly"]["max"]
            target_beta = lo + sigmoid(ratio_raw) * (hi - lo)
            # 本月月度新增电量 (β_m - α)·R_m >= 0: 年度合同不会被推翻
            cost = self.settlement.settle_monthly(
                target_beta, self.prices["monthly_prices"][self.current_month],
                self.current_month, signed_step=self.current_step
            )
            self.beta = self.settlement.beta
            self.gamma = self.settlement.gamma
            self.current_stage = self.STAGE_INTRADAY
            self.current_step += 1
            info["stage"] = "monthly"
            info["beta"] = self.beta

        elif self.current_stage == self.STAGE_INTRADAY:
            lo = self.beta
            hi = MARKET_BOUNDS["intraday"]["max"]
            target_gamma = lo + sigmoid(ratio_raw) * (hi - lo)
            cost = self.settlement.settle_intraday(
                target_gamma, self.prices["intraday_prices"][self.current_month],
                self.current_month, signed_step=self.current_step
            )
            self.gamma = self.settlement.gamma
            self.current_stage = self.STAGE_SPOT
            self.current_step += 1
            info["stage"] = "intraday"
            info["gamma"] = self.gamma

        elif self.current_stage == self.STAGE_SPOT:
            spot_prices = self.prices["spot_prices"][self.current_month]

            # 柔性强度: DRL 动作 or 规则基线 (只决定"择优程度", 落点由执行层按物理约束给出)
            if self.use_rule_dispatcher:
                intensity = self._rule_intensity(spot_prices)
            else:
                intensity = flexible_action

            executed = self.executor.execute_day(spot_prices, intensity)
            neutral = self.executor.execute_day(spot_prices, 0.0)
            month_demand_mwh = executed["net_load_mwh"] * self.days_per_month * self.demand_scale
            flexible_penalty = executed["penalty"]

            cost = self.settlement.settle_spot(
                spot_prices, self.current_month, month_demand_mwh,
                signed_step=self.current_step,
            )
            self.monthly_spot_costs.append(cost)
            self.total_flexible_penalty += flexible_penalty

            # 基准口径: 同一价格、同一合同、只做保障调度 (强度 0) 时的现货成本 + 惩罚
            neutral_demand = neutral["net_load_mwh"] * self.days_per_month * self.demand_scale
            no_flex_cost = self._spot_cost_of(spot_prices, self.gamma, self.current_month,
                                              neutral_demand)
            saving = max(0.0, (no_flex_cost + neutral["penalty"]) - (cost + flexible_penalty))
            self.flexible_cost_saving += saving
            self.device_checks.append(self._device_check(executed, self.current_month))

            self.current_month += 1
            if self.current_month < self.num_months:
                self.current_stage = self.STAGE_MONTHLY
            self.current_step += 1
            info["stage"] = "spot"
            info["spot_cost"] = cost
            info["flexible_penalty"] = flexible_penalty
            info["flexible_saving"] = saving
            info["flexible_action"] = flexible_action
            info["intensity"] = executed["intensity"]
            info["month_demand_mwh"] = executed["month_energy_mwh"] * self.demand_scale
            info["spot_deviation_mwh"] = self.settlement.last_spot_deviation
            info["device"] = self.device_checks[-1]

        # 累计成本
        self.episode_cost += cost
        self.total_market_cost += cost

        executed_stage = info.get("stage")

        # ---- 分阶段独立奖励 ----
        # 归一化基准 = 该步决策所覆盖的电量: 年度步用全年 Q, 月度/月内/现货步用月需求 d。
        # 这样每步奖励都是"元/MWh"量级的单位成本, 不会因为年度合同电量是月度 12 倍而淹没后续决策。
        d = basis_energy
        if executed_stage == "spot":
            spot_energy = max(abs(self.settlement.last_spot_deviation), self.MIN_SPOT_EXPOSURE * d)
            denom = spot_energy if self.scale_spot_reward else d
            reward_spot = -(cost + flexible_penalty) / denom
            reward_midlong = 0.0
            reward_global = reward_spot
        else:
            basis = self.total_demand if executed_stage == "annual" else d
            reward_midlong = -cost / basis
            # 关键奖励塑形: 中长期智能体的 γ 决策会导致本月剩余电量流入现货市场,
            # 若不感知该成本, 智能体没有保持高覆盖率的动力 (credit assignment 反向问题)
            if executed_stage == "intraday" and self.exposure_shaping:
                # 使用场景的现货均价作为"未来成本预估", 避免完美预见具体月价格
                expected_spot = self.price_gen.params["spot_mean"]
                reward_midlong -= (1.0 - self.gamma) * expected_spot
            reward_spot = 0.0
            reward_global = reward_midlong

        # CVaR 风险项: 只施加于中长期智能体 (它决定 γ → 暴露)
        if self.risk_lambda > 0 and len(self.monthly_spot_costs) >= 3 and executed_stage == "intraday":
            cvar = self.settlement.compute_cvar(self.monthly_spot_costs, DRL_PARAMS["cvar_eta"])
            risk_term = self.risk_lambda * cvar / d
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
    # 柔性执行 / 设备达标
    # ------------------------------------------------------------
    def _rule_intensity(self, spot_prices: np.ndarray) -> float:
        """
        规则基线的柔性强度 (只用已公布的当日前价, 不看未来实现值)。

        峰谷比每高出 1.0 一个单位折算 1/1.5 的择优强度, 上限 1.0; 规则基线不做 V2G。
        """
        spread = float(spot_prices.max() - spot_prices.min())
        mean = max(float(spot_prices.mean()), 1e-9)
        return float(np.clip((spread / mean - 1.0) / 1.5, 0.0, 1.0))

    def _spot_cost_of(self, spot_prices: np.ndarray, gamma: float, month_idx: int,
                      month_demand_mwh: np.ndarray) -> float:
        """按同一偏差结算规则手算某月现货成本 (不写台账), 用于基准比较。"""
        ref = self.settlement.reference_energy(month_idx)
        actual = float(month_demand_mwh.sum())
        if actual <= 1e-12:
            return 0.0
        deviation = month_demand_mwh * (1.0 - gamma * ref / actual)
        buy = np.maximum(deviation, 0.0)
        sell = np.minimum(deviation, 0.0)
        return float(np.sum(spot_prices * buy)
                     + (1.0 - self.settlement.sell_discount) * np.sum(spot_prices * sell))

    def _device_check(self, executed: Dict, month_idx: int) -> Dict:
        """本月设备执行硬条件 (所有策略共用同一物理任务): 离站达标 + 日任务守恒。"""
        ev, ind = executed["ev"], executed["ind"]
        target = (self.executor.target_soc if self.executor.ev_n > 0
                  else ev["soc_depart"])
        return {
            "month": int(month_idx),
            "intensity": executed["intensity"],
            "day_energy_mwh": executed["energy_mwh"],
            "ev_soc_depart": ev["soc_depart"],
            "ev_unmet_kwh": ev["unmet_kwh"],
            "ev_discharged_kwh": ev["discharged_kwh"],
            "ind_unmet_kwh": ind["unmet_kwh"],
            "ind_transferred_mwh": ind["transferred_mwh"],
            "task_met": bool(ev["soc_depart"] >= target - 1e-6
                             and ev["unmet_kwh"] < 1e-3
                             and ind["unmet_kwh"] < 1e-3),
        }

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
            "world": self.current_world,
            "stage": self.current_stage,
            "positions": self.settlement.get_positions(),
            "cost_breakdown": self.settlement.get_cost_breakdown(),
            "total_cost": total_cost_including_penalty,
            "market_cost": self.settlement.get_total_cost(),
            "flexible_penalty": self.total_flexible_penalty,
            "episode_cost": self.episode_cost,
            "flexible_cost_saving": self.flexible_cost_saving,
            "load_scenario": self.load_scenario,
            "device_checks": self.device_checks,
            "device_task_met": bool(all(c["task_met"] for c in self.device_checks)),
            "demand_scale": self.demand_scale,
        }

    def get_total_cost(self) -> float:
        """统一成本口径: 市场成本 + 柔性转移惩罚"""
        return self.settlement.get_total_cost() + self.total_flexible_penalty

    def get_fixed_ratio_cost(self, world: Optional[int] = None) -> Dict:
        """
        固定比例 + 保障调度 基准: 合同按 FIXED_RATIOS 签约, 柔性强度恒为 0 (只满足设备任务)。
        world 给定时在该价格实现上评价, 便于与 DRL 配对比较。
        """
        from ..utils.config import FIXED_RATIOS
        self.reset(options=None if world is None else {"world": world})
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

            neutral_demand = (self.neutral_day_mwh * self.days_per_month * self.demand_scale)
            total_cost += self.settlement.settle_spot(
                self.prices["spot_prices"][m], m, neutral_demand
            )

        return {
            "total_cost": total_cost,
            "breakdown": self.settlement.get_cost_breakdown(),
            "positions": self.settlement.get_positions(),
        }
