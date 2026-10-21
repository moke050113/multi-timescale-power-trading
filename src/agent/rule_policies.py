"""
规则型策略 (四臂对照基准里的两个非学习臂, 以及分层臂的下层)

设计约束:
    1. 只读**观察**里已发布的量 (见 environment._get_obs / 发布游标), 不看未来价格实现值;
    2. 动作仍是环境的 2 维 [ratio_raw, flexible_action], 与 SAC 臂走同一条 step 路径,
       因此四个臂的成本、电量台账、设备校验可以逐条对账;
    3. 规则必须是可手算的: 每个阶段一条公式, 参数集中在 RULE_PARAMS。

观察槽位 (13 维):
    0 α, 1 β, 2 γ, 3 年度价, 4 本月月度价, 5 本月月内价,
    6 现货水平预测, 7 现货波幅预测, 8/9 月份正余弦, 10 剩余暴露 1-γ,
    11 EV 可调容量/Q, 12 工业可转移容量/Q
"""
import numpy as np

from ..utils.config import (
    ACTION_BOUNDS, EV_PARAMS, FIXED_RATIOS, INDUSTRIAL_LOAD_PARAMS, MARKET_BOUNDS,
    SETTLEMENT_PARAMS,
)

STAGE_ANNUAL, STAGE_MONTHLY, STAGE_INTRADAY, STAGE_SPOT = 0, 1, 2, 3

# 放电商需要覆盖的物理与批零损耗 (相对值): 往返效率损失 + 富余回售折价
ROUND_TRIP_LOSS = 1.0 / (EV_PARAMS["charge_eff"] * EV_PARAMS["discharge_eff"]) - 1.0
V2G_GATE = ROUND_TRIP_LOSS + SETTLEMENT_PARAMS["sell_discount"]

RULE_PARAMS = {
    # 层间比价: (公开现货预测水平 - 本层合同价) / 预测水平 → 覆盖率偏移
    # rel > 0 表示合同比现货便宜, 应多签; rel < 0 表示合同偏贵, 留缺口给现货
    "hedge_gain": 1.5,
    # 调度收益相对"物理门槛"的灵敏度 (日内价差扣掉转移惩罚后 / 门槛)
    "dispatch_gain": 1.0,
    # 年度层: 签约时只有年度报价, 无层间信息可用 → 取保守锚定
    "annual_anchor": FIXED_RATIOS["alpha"],
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def raw_for_coverage(target: float, lo: float, hi: float,
                     cap: float = ACTION_BOUNDS["ratio_raw_limit"]) -> float:
    """把期望覆盖率反解成 ratio_raw: 环境用 target = lo + sigmoid(raw)·(hi − lo) 定价。

    cap 取 ACTION_BOUNDS["ratio_raw_limit"] —— 与 SAC 的 tanh×action_scale 同界,
    反解不到目标覆盖率时**如实停在可达边界上**, 不再用 cap=8 走到 SAC 无法输出的动作。
    """
    frac = (target - lo) / max(hi - lo, 1e-9)
    frac = float(np.clip(frac, 1e-4, 1.0 - 1e-4))
    return float(np.clip(np.log(frac / (1.0 - frac)), -cap, cap))


# ============================================================
# 臂 1: 固定比例 + 保障调度
# ============================================================
def fixed_ratio_action(stage: int, obs: np.ndarray) -> np.ndarray:
    """α/β/γ 恒取 FIXED_RATIOS, 柔性强度恒 0 (只满足设备任务, 不择优)。"""
    if stage == STAGE_ANNUAL:
        target, lo, hi = FIXED_RATIOS["alpha"], 0.0, MARKET_BOUNDS["annual"]["max"]
    elif stage == STAGE_MONTHLY:
        target, lo, hi = FIXED_RATIOS["beta"], float(obs[0]), MARKET_BOUNDS["monthly"]["max"]
    elif stage == STAGE_INTRADAY:
        target, lo, hi = FIXED_RATIOS["gamma"], float(obs[1]), MARKET_BOUNDS["intraday"]["max"]
    else:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([raw_for_coverage(target, lo, hi), 0.0], dtype=np.float32)


# ============================================================
# 臂 2: 滚动规则 + 预测调度
# ============================================================
def _hedge_fraction(rel: float, gain: float) -> float:
    return float(np.clip(0.5 + gain * rel, 0.02, 1.0))


def _relative_gap(contract_price: float, forecast_level: float) -> float:
    """合同价相对公开现货预测便宜多少 (>0 划算) ; 预测缺失时按 0 (不加偏移)。"""
    if forecast_level <= 1e-9:
        return 0.0
    return (forecast_level - contract_price) / forecast_level


def rolling_rule_action(stage: int, obs: np.ndarray,
                        params: dict = None) -> np.ndarray:
    """
    逐层滚动比价 + 按公开日前剖面的价差强度调度。

    年度: 无层间信息 → 保守锚定。
    月度/月内: 本层合同价 vs 公开现货预测水平 → 决定该层新增覆盖占"可签约空间"的比例。
    现货: 净价差 (日内波幅 − 工业转移惩罚) 相对物理门槛决定强度;
          波幅/水平 超过门槛 (往返损耗 + 回售折价) 才启用 V2G (负号),
          否则只做充电侧移峰填谷 (正号); 无可调度容量 → 0。
    """
    p = RULE_PARAMS if params is None else {**RULE_PARAMS, **params}
    # 观察里的价格槽位已除以 1000, 规则内部一律还原成 元/MWh, 便于逐条手算
    monthly_p = float(obs[4]) * 1000.0
    intraday_p = float(obs[5]) * 1000.0
    level = float(obs[6]) * 1000.0
    spread = float(obs[7]) * 1000.0

    if stage == STAGE_ANNUAL:
        raw = raw_for_coverage(p["annual_anchor"], 0.0, MARKET_BOUNDS["annual"]["max"])
        return np.array([raw, 0.0], dtype=np.float32)

    if stage == STAGE_MONTHLY:
        alpha = float(obs[0])
        rel = _relative_gap(monthly_p, level)
        space = MARKET_BOUNDS["monthly"]["max"] - alpha
        target = alpha + space * _hedge_fraction(rel, p["hedge_gain"])
        return np.array([raw_for_coverage(target, alpha, MARKET_BOUNDS["monthly"]["max"]),
                         0.0], dtype=np.float32)

    if stage == STAGE_INTRADAY:
        beta = float(obs[1])
        rel = _relative_gap(intraday_p, level)
        space = MARKET_BOUNDS["intraday"]["max"] - beta
        target = beta + space * _hedge_fraction(rel, p["hedge_gain"])
        return np.array([raw_for_coverage(target, beta, MARKET_BOUNDS["intraday"]["max"]),
                         0.0], dtype=np.float32)

    capability = float(obs[11]) + float(obs[12])
    if capability <= 1e-9 or level <= 1e-9:
        return np.array([0.0, 0.0], dtype=np.float32)
    net = spread - INDUSTRIAL_LOAD_PARAMS["transfer_cost"]
    strength = float(np.clip(p["dispatch_gain"] * net / (V2G_GATE * level), 0.0, 1.0))
    # 放电的变现能力受"本月还剩多少偏差空间"限制: 覆盖率 γ_m 已签定, 实际净负荷低于
    # 合同量的部分只能按现货价 ×(1-回售折价) 卖出, 经济性由正转负; 缺口越窄, 可放的电越少。
    # 充电侧移峰填谷不改变当月总量, 不受此约束 (故只在放电符号时收紧)。
    if strength > 0.0 and spread / level > V2G_GATE:
        strength = min(strength, float(obs[10]) / MARKET_BOUNDS["spot"]["max"])
        sign = -1.0
    else:
        sign = 1.0
    return np.array([0.0, sign * strength], dtype=np.float32)


# ============================================================
# 策略包装 (供训练/评价脚本按同一接口调用)
# ============================================================
class RulePolicy:
    """固定规则臂: 不可训练, act 即最终动作 (确定性, 与 SAC 臂同一调用接口)。"""

    def __init__(self, rule):
        self.rule = rule

    def act(self, obs, stage, deterministic=False):
        return self.rule(stage, np.asarray(obs, dtype=np.float64))

    def store(self, *args, **kwargs):
        pass

    def update(self, stage=None):
        return {}

    def begin_episode(self):
        pass


def forecast_intensity(obs: np.ndarray) -> float:
    """分层臂的下层: 只取预测调度规则的现货强度。"""
    return float(rolling_rule_action(STAGE_SPOT, np.asarray(obs, dtype=np.float64))[1])
