"""
环境端到端会计 + 设备执行闭环验收测试 (审计修复路线 第 3 步 / 任务 #38)

对应审计"最低验收门槛":
    1. 电量守恒    : 合同量闭合于 Q; 合同量 + 现货偏差 = 实际网侧净需求 (可复算)
    2. 设备按时完成 : 每月 EV 离站 SOC 达标、工业日任务守恒 (所有强度、所有负荷场景)
    3. 费用可手算    : 逐台账记录用 quantity×price (现货按小时, 过购按折价) 复算 == 系统总成本
    4. 无虚假退款    : 平价市场下任何柔性动作都赚不到钱 (往返损耗只会增加成本);
                       柔性只改变现货偏差, 绝不冲销已签合同成本
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.market.environment import TradingEnvironment
from src.load.load_scenarios import LoadScenarioBuilder
from src.utils.config import TOTAL_DEMAND

NUM_STEPS = 37  # 1 年度 + 12 × 3


def _rollout(env, intensities, ratio_raw=0.5, world=None):
    """按给定 (ratio_raw, intensity) 序列走一整回合, 返回终局 info 与结算引擎。"""
    obs, _ = env.reset(options=None if world is None else {"world": world})
    i = 0
    while True:
        k = intensities[i % len(intensities)]
        obs, r, term, trunc, info = env.step(np.array([ratio_raw, k], dtype=np.float32))
        i += 1
        if term or trunc:
            break
    assert i == NUM_STEPS, i
    return info, i


def _random_intensities(rng, n=24):
    return [float(rng.uniform(-1, 1)) for _ in range(n)]


def test_contract_closure_and_actual_demand_conservation():
    """逐月合同量 = γ_m·R_m; 合同量 + 现货偏差 = 实际网侧净需求 (可复算)。"""
    rng = np.random.default_rng(7)
    env = TradingEnvironment(include_flexible=True, seed=42)
    for ep in range(200):
        info, steps = _rollout(env, _random_intensities(rng))
        se = env.settlement
        assert steps == NUM_STEPS

        actual = sum(c["day_energy_mwh"] for c in info["device_checks"]) \
            * env.days_per_month * env.demand_scale
        ledger_total = sum(r["quantity"] for r in se.get_ledger())
        assert abs(ledger_total - actual) < 1e-6, (ep, ledger_total, actual)

        contract_total = 0.0
        for m, slot in se.get_monthly_closure().items():
            spot_rec = [r for r in se.get_ledger()
                        if r["stage"] == "spot" and r["deliver_month"] == m][0]
            assert abs(slot["contract"] - spot_rec["contract_energy"]) < 1e-9, (m, slot)
            assert slot["contract"] <= se.reference_energy(m) + 1e-9, (m, slot)
            assert abs(slot["total"] - slot["actual_energy"]) < 1e-6, (m, slot)
            contract_total += slot["contract"]
        energy = se.get_energy_breakdown()
        assert abs(energy["spot"] - (actual - contract_total)) < 1e-6, (ep, energy)
        assert contract_total <= TOTAL_DEMAND + 1e-6
        for rec in se.get_ledger():
            if rec["stage"] != "spot":
                assert rec["quantity"] >= -1e-9, rec
        assert np.isfinite(info["total_cost"])
    print(f"  [PASS] 200 回合随机柔性动作: 逐月合同 = gamma_m*R_m, 台账总量 = 实际净需求")
    print(f"         末回合: 合同 {contract_total:.2f} MWh, 实际净需求 {actual:.2f} MWh, "
          f"现货偏差 {energy['spot']:+.2f} MWh")


def test_neutral_dispatch_sums_to_annual_demand():
    """强度恒为 0 (保障调度) 时, 标定后的全年实际净需求恰为 Q, 台账电量完全闭合。"""
    env = TradingEnvironment(include_flexible=True, seed=42)
    info, _ = _rollout(env, [0.0])
    actual = sum(c["day_energy_mwh"] for c in info["device_checks"]) \
        * env.days_per_month * env.demand_scale
    assert abs(actual - TOTAL_DEMAND) < 1e-6, actual
    assert abs(env.settlement.get_energy_breakdown()["total"] - TOTAL_DEMAND) < 1e-6
    print(f"  [PASS] 保障调度全年净需求 = {actual:.6f} MWh = Q, 台账电量 = Q")


def test_device_tasks_completed_all_scenarios():
    """设备按时完成任务: 5 个负荷场景 × 正负强度扫描, 每月 task_met 全为真。"""
    for scenario in LoadScenarioBuilder.list_scenarios():
        env = TradingEnvironment(load_scenario=scenario, seed=11)
        for intensities in ([1.0, -1.0], [-1.0, 1.0, 0.0], [0.0]):
            info, _ = _rollout(env, intensities)
            checks = info["device_checks"]
            assert len(checks) == env.num_months, (scenario, len(checks))
            assert info["device_task_met"], (scenario, intensities, checks)
            for c in checks:
                assert c["ev_unmet_kwh"] < 1e-3, (scenario, c)
                assert c["ind_unmet_kwh"] < 1e-3, (scenario, c)
                assert c["ind_transferred_mwh"] >= 0.0
        print(f"    {scenario}: EV 离站 SOC {checks[0]['ev_soc_depart']:.3f}, "
              f"工业未完成任务 {max(c['ind_unmet_kwh'] for c in checks):.2e} kWh")
    print("  [PASS] 全部负荷场景 × 全部柔性强度: 设备任务逐月按时完成 (硬约束, 非奖励惩罚)")


def test_spot_cost_hand_computable_from_ledger():
    """费用可手算: 逐条台账用 数量×价格 复算 (现货按小时; 过购按 (1-折价)) == 系统成本。"""
    rng = np.random.default_rng(3)
    env = TradingEnvironment(seed=42)
    for ep in range(20):
        _rollout(env, _random_intensities(rng))
        hand = 0.0
        for rec in env.settlement.get_ledger():
            if isinstance(rec["price"], np.ndarray):
                gross = float(np.sum(rec["price"] * rec["quantities"]))
                hand += gross if rec["quantity"] >= 0 else gross * (1.0 - rec["sell_discount"])
            else:
                hand += rec["quantity"] * rec["price"]
        sys_cost = env.settlement.get_total_cost()
        assert abs(hand - sys_cost) < 1e-6, (ep, hand, sys_cost)
        for rec in env.settlement.get_ledger():
            if rec["stage"] == "spot":
                assert abs(rec["actual_energy"] - rec["contract_energy"]
                           - rec["quantity"]) < 1e-6, rec
    print("  [PASS] 20 回合: 台账手算成本 = 系统成本, 且 现货电量 = 实际需求 - 合同量")


def test_flat_price_flexibility_cannot_make_money():
    """无虚假退款: 平价市场中柔性调度最多不赚不赔 (往返损耗与转移惩罚只会推高成本)。"""
    price = 400.0
    env = TradingEnvironment(scenario_name="C_各阶段价格接近", seed=42)
    env.reset()
    template = dict(env.prices)

    def flat_generate():
        p = dict(template)
        p["annual_price"] = price
        p["monthly_prices"] = np.full(env.num_months, price)
        p["intraday_prices"] = np.full(env.num_months, price)
        p["spot_prices"] = np.full((env.num_months, 24), price)
        return p

    env.price_gen.generate = flat_generate
    neutral_cost = env.get_fixed_ratio_cost()["total_cost"]
    assert abs(neutral_cost - TOTAL_DEMAND * price) < 1e-6, neutral_cost

    for intensities in ([1.0], [-1.0], [-0.6, 0.6]):
        env.price_gen.generate = flat_generate
        info, _ = _rollout(env, intensities, ratio_raw=0.0)
        market = env.settlement.get_total_cost()
        total = market + env.total_flexible_penalty
        assert total >= TOTAL_DEMAND * price - 1e-6, (intensities, total)
        print(f"    强度 {intensities}: 市场成本 {market:,.0f} + 惩罚 "
              f"{env.total_flexible_penalty:,.0f} ≥ 平价基准 {TOTAL_DEMAND * price:,.0f}")
    print("  [PASS] 平价市场下柔性调度无法套出负成本 (不存在虚假退款/凭空收益)")


def test_flexibility_never_rewrites_signed_contracts():
    """同一签约序列、不同柔性强度: 各合同阶段成本逐条相同, 只有现货偏差变化。"""
    env = TradingEnvironment(seed=42)
    runs = {}
    for intensities in ([0.0], [1.0], [-1.0]):
        info, _ = _rollout(env, intensities, ratio_raw=0.3, world=0)
        contract = [(r["contract_id"], r["stage"], r["deliver_month"], r["quantity"], r["cost"])
                    for r in env.settlement.get_ledger() if r["stage"] != "spot"]
        spot = [(r["deliver_month"], r["quantity"], r["actual_energy"])
                for r in env.settlement.get_ledger() if r["stage"] == "spot"]
        runs[tuple(intensities)] = (contract, spot, env.settlement.get_total_cost())

    base_contract = runs[(0.0,)][0]
    for key, (contract, spot, total) in runs.items():
        assert contract == base_contract, key
        contract_q = sum(c[3] for c in contract)
        assert abs(sum(s[1] for s in spot)
                   - (sum(s[2] for s in spot) - contract_q)) < 1e-6
    diffs = {k: runs[k][2] for k in runs}
    print("    同签约序列下市场成本:", {k[0]: f"{v:,.0f}" for k, v in diffs.items()})
    assert len({f"{v:.1f}" for v in diffs.values()}) > 1, "柔性强度未改变成本, 闭环可能失效"
    print("  [PASS] 柔性只改变现货偏差电量, 48 条合同记录逐条不变 (不冲销已签合同)")


def test_month_residual_identity():
    """单月核对: 现货电量 = A_m - γ_m·R_m, 成本 = Σ_h price_h·Δ_h (缺口价/过购折价)。"""
    env = TradingEnvironment(seed=42)
    env.reset(options={"world": 1})
    for _ in range(3):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    m = env.current_month
    prices = env.prices["spot_prices"][m]
    intensity = -0.7
    executed = env.executor.execute_day(prices, intensity)
    demand = executed["net_load_mwh"] * env.days_per_month * env.demand_scale
    gamma_before = env.gamma
    ref = env.settlement.reference_energy(m)

    cost = env.settlement.settle_spot(prices, m, demand)
    a = float(demand.sum())
    dev = demand * (1.0 - gamma_before * ref / a)
    hand = float(np.sum(prices * dev)) if dev.sum() >= 0 else \
        float(np.sum(prices * dev) * (1.0 - env.settlement.sell_discount))
    assert abs(dev.sum() - (a - gamma_before * ref)) < 1e-9
    assert abs(cost - hand) < 1e-6, (cost, hand)
    print(f"  [PASS] 单月复算: A_m={a:.3f}, γ·R_m={gamma_before * ref:.3f}, "
          f"偏差={dev.sum():+.3f} MWh, 现货成本 {cost:,.2f} 元")


if __name__ == "__main__":
    tests = [
        ("合同闭合/实际需求守恒", test_contract_closure_and_actual_demand_conservation),
        ("保障调度电量闭合", test_neutral_dispatch_sums_to_annual_demand),
        ("设备逐月按时达标", test_device_tasks_completed_all_scenarios),
        ("台账费用可手算", test_spot_cost_hand_computable_from_ledger),
        ("平价不可套利", test_flat_price_flexibility_cannot_make_money),
        ("柔性不冲销合同", test_flexibility_never_rewrites_signed_contracts),
        ("单月偏差恒等式", test_month_residual_identity),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as exc:
            failed.append((name, exc))
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
    print(f"\n通过 {len(tests) - len(failed)}/{len(tests)}")
    if failed:
        sys.exit(1)
