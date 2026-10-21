"""
交易结算引擎

会计口径 (v1.2, 逐月交割 + 合同台账):
    Q        = 全年总需求电量
    R_m      = 第 m 月的参考交付电量, Σ_m R_m = Q (默认均匀 Q/12; 可传入月度形状, 会被缩放至 Q)
    α        = 年度合同覆盖率 (全年一次签约, 按 12 个交割月分别记 R_m 的 α 份额)
    β_m      = 第 m 月的月度合同覆盖率 (α ≤ β_m ≤ 1)
    γ_m      = 第 m 月的中长期累计覆盖率 (α ≤ β_m ≤ γ_m ≤ 1)

    各阶段**新增**成交电量 (每批合同都是一条台账记录):
        Q^Y_m = α · R_m              年度合同, 交割月 m
        Q^M_m = (β_m - α) · R_m      月度合同, 交割月 m
        Q^I_m = (γ_m - β_m) · R_m    月内合同, 交割月 m
        Q^S_m = A_m - γ_m · R_m      现货/偏差净电量, 交割月 m

    A_m = 设备执行后的本月实际网侧净需求 (MWh)。不执行柔性调度时 A_m = R_m,
    此时恒等式 Q^Y_m + Q^M_m + Q^I_m + Q^S_m = R_m, Σ_m R_m = Q 闭合;
    执行柔性调度后 A_m 会因 EV 往返损耗 / V2G 放电偏离 R_m, 台账同时记录两者。

覆盖率单调性 (β_m ≥ α, γ_m ≥ β_m) 由断言保证, 而不是把负增量裁成 0:
已签合同不可撤销, 下月覆盖率可以降低属于不同交割月, 不冲销上月合同。

偏差处理规则 (明确写在结算里, 不隐含在数组缩放里):
    缺口 (Δ_h > 0) 按该时段现货价 1:1 买入 —— 现货就是残余市场, 不再叠加惩罚项;
    过购 (Δ_h < 0) 按该时段现货价 × (1 - sell_discount) 回售, 代表批零价差,
                   且绝不以退款形式冲销已签合同成本。
"""
import copy

import numpy as np
from typing import Dict, Optional, Sequence

from ..utils.config import SETTLEMENT_PARAMS, SIM_PARAMS, TOTAL_DEMAND


class SettlementEngine:
    """结算引擎：按交割月记账, 跟踪合同台账、成本与电量。"""

    def __init__(
        self,
        total_demand: float = TOTAL_DEMAND,
        num_months: int = SIM_PARAMS["num_months"],
        monthly_reference: Optional[Sequence[float]] = None,
        sell_discount: Optional[float] = None,
    ):
        self.total_demand = float(total_demand)
        self.num_months = int(num_months)
        self.sell_discount = (SETTLEMENT_PARAMS["sell_discount"]
                              if sell_discount is None else float(sell_discount))
        self.R = self._build_reference(monthly_reference)

        self._clear_state()

    def _build_reference(self, monthly_reference) -> np.ndarray:
        """参考交付电量 R_m, 强制 Σ R_m = Q (给定形状则等比缩放)。"""
        if monthly_reference is None:
            return np.full(self.num_months, self.total_demand / self.num_months)

        R = np.asarray(monthly_reference, dtype=float)
        assert R.shape == (self.num_months,), f"R_m 长度须为 {self.num_months}, 当前 {R.shape}"
        assert np.all(R >= 0) and R.sum() > 0, "R_m 必须非负且总量为正"
        return R / R.sum() * self.total_demand

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

        # 合同台账: 每批合同一条记录 (数量/价格/签约序号/交割月)
        self.contracts = []
        self._contract_seq = 0
        # 时序守卫: 交割月 → 已结算阶段; 已完成现货交割的月份数
        self._stage_done = {}
        self._delivered_before = 0
        # 逐月实际净需求 (设备执行结果), 由 settle_spot 写入
        self.actual_demand = np.zeros(self.num_months)
        self.last_spot_deviation = 0.0
        self.last_spot_energy_abs = 0.0

    def reset(self):
        """重置所有仓位、成本、电量与台账。"""
        self._clear_state()

    # ------------------------------------------------------------
    # 台账
    # ------------------------------------------------------------
    def reference_energy(self, month_idx: int) -> float:
        """第 month_idx 月的参考交付电量 R_m。"""
        return float(self.R[int(month_idx)])

    def _record(self, stage: str, deliver_month: Optional[int], signed_step: Optional[int],
                quantity: float, price: float, cost: float, **coverage):
        """写入一条合同台账 (同时充当交易历史)。"""
        self._contract_seq += 1
        price_val = (np.asarray(price, dtype=float).copy()
                     if isinstance(price, (np.ndarray, list, tuple)) else float(price))
        record = {
            "contract_id": self._contract_seq,
            "stage": stage,
            "signed_step": self._contract_seq - 1 if signed_step is None else int(signed_step),
            "deliver_month": None if deliver_month is None else int(deliver_month),
            "price": price_val,
            "quantity": float(quantity),
            "cost": float(cost),
            **coverage,
        }
        self.contracts.append(record)
        return record

    @property
    def history(self):
        """兼容旧调用: 交易历史即合同台账 (同样返回副本)。"""
        return self.get_ledger()

    def get_ledger(self) -> list:
        """返回合同台账的**深拷贝**: 外部拿到的账本改动不了引擎内部状态。"""
        return copy.deepcopy(self.contracts)

    def get_monthly_closure(self) -> Dict[int, Dict[str, float]]:
        """按交割月汇总电量, 用于验收 R_m 是否被完整覆盖、偏差是否等于实际净需求。"""
        out = {}
        for rec in self.contracts:
            m = rec["deliver_month"]
            if m is None:
                continue
            slot = out.setdefault(m, {"annual": 0.0, "monthly": 0.0, "intraday": 0.0,
                                      "spot": 0.0, "total": 0.0,
                                      "reference": self.reference_energy(m),
                                      "contract": 0.0, "actual_energy": self.reference_energy(m)})
            slot[rec["stage"]] += rec["quantity"]
            slot["total"] += rec["quantity"]
            if rec["stage"] != "spot":
                slot["contract"] += rec["quantity"]
            if rec["stage"] == "spot":
                slot["actual_energy"] = rec["actual_energy"]
        return out

    # ------------------------------------------------------------
    # 各阶段结算
    # ------------------------------------------------------------
    def settle_annual(self, alpha: float, annual_price: float,
                      signed_step: Optional[int] = None) -> float:
        """年度交易结算: 按 12 个交割月各成交 α·R_m, 全年只允许签约一次。"""
        assert 0 <= alpha <= 1, f"α 必须在 [0,1] 范围内, 当前: {alpha}"
        assert not self.annual_settled, "年度合同已签约, 同一回合不得重复计量年度电量"
        self._finite(annual_price, "年度合同价")

        self.alpha = float(alpha)
        total_cost = 0.0
        for m in range(self.num_months):
            q = self.alpha * self.reference_energy(m)
            c = float(annual_price) * q
            total_cost += c
            self.energy_annual += q
            self._record("annual", deliver_month=m, signed_step=signed_step,
                         quantity=q, price=annual_price, cost=c, alpha=self.alpha)

        self.cost_annual = total_cost
        self.annual_settled = True
        return total_cost

    def settle_monthly(self, beta: float, monthly_price: float, month_idx: int,
                       signed_step: Optional[int] = None) -> float:
        """月度交易结算: 本月新增 (β_m - α)·R_m。"""
        self._check_month(month_idx)
        self._open_stage("monthly", month_idx)
        self._finite(monthly_price, "月度合同价")
        assert self.alpha <= beta <= 1, f"需满足 α≤β≤1, 当前: α={self.alpha}, β={beta}"

        delta_beta = float(beta) - self.alpha
        q_month = delta_beta * self.reference_energy(month_idx)
        cost = float(monthly_price) * q_month

        # 本月覆盖率状态: 月内决策从 β_m 起步, 保证月内不回溯
        self.beta = float(beta)
        self.gamma = self.beta
        self.cost_monthly += cost
        self.energy_monthly += q_month

        self._record("monthly", deliver_month=month_idx, signed_step=signed_step,
                     quantity=q_month, price=monthly_price, cost=cost,
                     beta=self.beta, delta_beta=delta_beta)
        return cost

    def settle_intraday(self, gamma: float, intraday_price: float, month_idx: int,
                        signed_step: Optional[int] = None) -> float:
        """月内交易结算: 本月新增 (γ_m - β_m)·R_m。"""
        self._check_month(month_idx)
        self._open_stage("intraday", month_idx, requires=("monthly",))
        self._finite(intraday_price, "月内合同价")
        assert self.beta <= gamma <= 1, f"需满足 β≤γ≤1, 当前: β={self.beta}, γ={gamma}"

        delta_gamma = float(gamma) - self.beta
        q_day = delta_gamma * self.reference_energy(month_idx)
        cost = float(intraday_price) * q_day

        self.gamma = float(gamma)
        self.cost_intraday += cost
        self.energy_intraday += q_day

        self._record("intraday", deliver_month=month_idx, signed_step=signed_step,
                     quantity=q_day, price=intraday_price, cost=cost,
                     gamma=self.gamma, delta_gamma=delta_gamma)
        return cost

    def settle_spot(self, spot_prices: np.ndarray, month_idx: int,
                    demand_hourly: Optional[np.ndarray] = None,
                    signed_step: Optional[int] = None) -> float:
        """
        现货 / 偏差结算: 本月实际净需求与已签合同电量之差在现货市场出清。

        Args:
            spot_prices: 当月各时段现货价格 (24,), 元/MWh
            month_idx: 交割月索引
            demand_hourly: 设备执行后的本月逐时段网侧电量 (MWh, shape=(24,), 已含该月天数)。
                None → 按 R_m 均匀分布, 即"不执行柔性调度"的基准口径。
            signed_step: 签约时序 (台账字段)

        规则 (不做重归一化, 不隐含缩放):
            A_m   = Σ_h demand_h                       实际净需求
            ΔE_h  = demand_h · (1 - γ_m·R_m / A_m)     逐时段偏差电量
            ΔE_h > 0 → 按该时段现货价 1:1 买入
            ΔE_h < 0 → 按该时段现货价 × (1 - sell_discount) 回售 (负成本=收入)
        """
        self._check_month(month_idx)
        self._open_stage("spot", month_idx, requires=("monthly",))
        hours = self.hours_per_day()
        ref = self.reference_energy(month_idx)
        contract_energy = self.gamma * ref

        # 先把所有输入校验完再动账: 旧写法在价格长度断言之前就已写入 actual_demand,
        # 一次失败的调用会留下半个月的账 (部分提交)。
        prices = self._finite(spot_prices, "现货价格剖面")
        assert prices.shape == (hours,), f"现货价格长度须为 {hours}, 当前 {prices.shape}"
        if demand_hourly is None:
            demand = np.full(hours, ref / hours)
        else:
            demand = self._finite(demand_hourly, "逐时段网侧电量")
            assert demand.shape == (hours,), f"demand_hourly 长度须为 {hours}, 当前 {demand.shape}"
            assert np.all(demand >= -1e-9), f"网侧净需求不得为负: min={demand.min():.6f}"

        actual_energy = float(demand.sum())
        assert actual_energy > 1e-12, "本月实际净需求为 0, 无法结算偏差"

        self.actual_demand[month_idx] = actual_energy
        deviation = demand * (1.0 - contract_energy / actual_energy)

        buy_qty = np.maximum(deviation, 0.0)
        sell_qty = np.minimum(deviation, 0.0)
        cost = float(np.sum(prices * buy_qty)
                     + (1.0 - self.sell_discount) * np.sum(prices * sell_qty))
        spot_ratio = float(deviation.sum() / ref) if ref > 0 else 0.0

        self.cost_spot += cost
        self.energy_spot += float(deviation.sum())
        self.last_spot_deviation = float(deviation.sum())
        self.last_spot_energy_abs = float(np.abs(deviation).sum())

        self._record("spot", deliver_month=month_idx, signed_step=signed_step,
                     quantity=float(deviation.sum()), price=prices, cost=cost,
                     spot_ratio=spot_ratio, quantities=deviation,
                     reference_energy=ref, contract_energy=contract_energy,
                     actual_energy=actual_energy,
                     buy_energy=float(buy_qty.sum()),
                     sell_energy=float(-sell_qty.sum()),
                     sell_discount=self.sell_discount)
        # 本月交割完成: 市场游标前移, 之后任何回到本月的补签都会被 _check_month 拒绝
        self._delivered_before = int(month_idx) + 1
        return cost

    def _check_month(self, month_idx: int):
        assert self.annual_settled, "必须先完成年度交易签约"
        assert 0 <= month_idx < self.num_months, (
            f"month_idx 越界: {month_idx}, 有效范围 0~{self.num_months - 1}"
        )
        assert month_idx == self._delivered_before, (
            f"市场按交割月顺序开市: 当前只能交易交割月 {self._delivered_before}, "
            f"收到 {month_idx} (跳月或回退到已交割月都会被拒绝)"
        )

    def _open_stage(self, stage: str, month_idx: int, requires=()):
        """公开 API 的幂等 + 前置阶段守卫 (在任何状态写入之前调用)。"""
        done = self._stage_done.setdefault(int(month_idx), set())
        assert stage not in done, (
            f"交割月 {month_idx} 的 {stage} 已结算, 拒绝同月重复计量"
        )
        for req in requires:
            assert req in done, f"交割月 {month_idx} 结算 {stage} 前必须先完成 {req}"
        done.add(stage)

    @staticmethod
    def _finite(value, name: str):
        """价格/电量必须是有限实数: NaN/inf 会污染整本台账 (且无法手算核对)。"""
        arr = np.asarray(value, dtype=float)
        assert np.all(np.isfinite(arr)), f"{name} 含 NaN/inf: {value}"
        return arr

    @staticmethod
    def hours_per_day():
        return SIM_PARAMS["hours_per_day"]

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
            "annual_reference": self.total_demand,
        }

    def compute_cvar(self, spot_costs: list, eta: float = 0.95) -> float:
        """
        经验分布上的期望短缺 ES_η (尾部样本按分数权重计入)。

        Args:
            spot_costs: 各月现货成本列表 (元)
            eta: 置信水平 (默认 0.95)

        迭代 8.11 前的写法是"np.percentile 线性插值取 VaR, 再对 cost ≥ VaR 的样本求均值"。
        它在离散样本上偏小: [1,2,3,4] 与 η=0.625 给 3.5, 而尾部质量 0.375 对应
        1.5 个样本 (4 全计 + 3 计一半) → 精确 ES = (4 + 0.5×3)/1.5 = 11/3 ≈ 3.667。
        这里按 (1-η)·n 个样本、边界样本取分数权重, 结果可手算核对。

        注意样本单元: 传入的是**同一回合内 12 个相互相关的月成本**, 不是 12 个独立年度。
        """
        costs = np.sort(np.asarray(spot_costs, dtype=float))[::-1]   # 降序: 最贵(最差)在前
        n = int(costs.size)
        if n == 0:
            return 0.0
        if n < 2 or not (0.0 < eta < 1.0):
            return float(costs[0])
        k = (1.0 - float(eta)) * n          # 尾部样本数 (可为分数)
        full = int(np.floor(k))
        frac = k - full
        total = float(costs[:full].sum()) + frac * float(costs[full])
        return float(total / k)
