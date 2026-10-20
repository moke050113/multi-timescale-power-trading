"""
结算会计验收测试 (审计修复路线 第 2 步)

验证目标:
    1. 平价市场下合同电量恒等于 Q, 成本恒等于 Q × price (消除 3.25× 双重计量)
    2. 任一覆盖率序列 (含跨月下降) 都不产生负合同电量、负合同成本
    3. 已成交合同不被推翻: β<α、γ<β、重复年度结算均报错
    4. 现货/偏差电量 = 设备执行后的实际需求 - 合同量; 过购按折价回售, 不冲销合同成本
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.market.settlement import SettlementEngine

PRICE = 400.0
Q = 1000.0
NUM_MONTHS = 12
TOL = 1e-9


def _run_plan(alpha, betas, gammas, demand=None):
    se = SettlementEngine(total_demand=Q, num_months=NUM_MONTHS)
    se.reset()
    se.settle_annual(alpha, PRICE)
    prices = np.full(24, PRICE)
    for m in range(NUM_MONTHS):
        se.settle_monthly(betas[m], PRICE, m)
        se.settle_intraday(gammas[m], PRICE, m)
        q = None if demand is None else demand(m)
        se.settle_spot(prices, m, q)
    return se


def _expect_error(fn, label):
    try:
        fn()
    except AssertionError:
        print(f"  [PASS] {label}: 按预期拒绝")
        return
    raise AssertionError(f"{label}: 期望 AssertionError, 实际未报错")


def test_flat_price_exact():
    """平价 400 / Q=1000 / α=0.8, β=0.9, γ=0.95 → 1000 MWh, 400000 元。"""
    se = _run_plan(0.80, [0.90] * NUM_MONTHS, [0.95] * NUM_MONTHS)
    energy = se.get_energy_breakdown()
    cost = se.get_cost_breakdown()

    print("平价场景电量:", {k: round(v, 6) for k, v in energy.items()})
    print("平价场景成本:", {k: round(v, 6) for k, v in cost.items()})

    assert abs(energy["annual"] - 800.0) < TOL, energy["annual"]
    assert abs(energy["monthly"] - 100.0) < TOL, energy["monthly"]
    assert abs(energy["intraday"] - 50.0) < TOL, energy["intraday"]
    assert abs(energy["spot"] - 50.0) < TOL, energy["spot"]
    assert abs(energy["total"] - Q) < 1e-6, energy["total"]
    assert abs(cost["total"] - Q * PRICE) < 1e-6, cost["total"]
    print("  [PASS] 平价总电量=1000 MWh, 总成本=400000 元 (无双重计量)")


def test_cross_month_decrease_and_random():
    """跨月覆盖率下降属于不同交割月, 但会计仍闭合、电量非负。"""
    rng = np.random.default_rng(0)
    for trial in range(200):
        alpha = float(rng.uniform(0.0, 0.9))
        betas = rng.uniform(alpha, 1.0, NUM_MONTHS)
        gammas = rng.uniform(betas, 1.0, NUM_MONTHS)
        se = _run_plan(alpha, betas, gammas)

        energy = se.get_energy_breakdown()
        assert abs(energy["total"] - Q) < 1e-6, (trial, energy)
        for rec in se.history:
            assert rec["quantity"] >= -TOL, rec
            assert rec["cost"] >= -TOL, rec
    print("  [PASS] 200 组随机覆盖率(含跨月下降): 总电量恒为 Q, 无负电量/负成本")


def test_no_overturn():
    """已成交仓位不可推翻: 违反覆盖率单调性或重复年度结算必须报错。"""
    def annual_twice():
        se = SettlementEngine(Q, NUM_MONTHS)
        se.settle_annual(0.8, PRICE)
        se.settle_annual(0.8, PRICE)

    def beta_below_alpha():
        se = SettlementEngine(Q, NUM_MONTHS)
        se.settle_annual(0.8, PRICE)
        se.settle_monthly(0.7, PRICE, 0)

    def gamma_below_beta():
        se = SettlementEngine(Q, NUM_MONTHS)
        se.settle_annual(0.8, PRICE)
        se.settle_monthly(0.9, PRICE, 0)
        se.settle_intraday(0.85, PRICE, 0)

    def spot_before_intraday_uses_beta():
        se = SettlementEngine(Q, NUM_MONTHS)
        se.settle_annual(0.8, PRICE)
        se.settle_monthly(0.9, PRICE, 0)
        se.settle_spot(np.full(24, PRICE), 0)
        # 月内之前, 现货敞口应停留在 1-β
        return se.get_positions()["spot_exposure"]

    _expect_error(annual_twice, "重复年度结算")
    _expect_error(beta_below_alpha, "β < α 推翻年度合同")
    _expect_error(gamma_below_beta, "γ < β 推翻月度合同")
    assert abs(spot_before_intraday_uses_beta() - 0.10) < TOL
    print("  [PASS] 月内成交前现货敞口按 β 计, 不会被隐藏的高覆盖率掩盖")


def test_monthly_energy_split():
    """每月各阶段电量之和等于该月需求 d。"""
    se = _run_plan(0.80, [0.90] * NUM_MONTHS, [0.95] * NUM_MONTHS)
    per_month = {}
    for rec in se.history:
        if rec["stage"] == "annual" or rec["deliver_month"] is None:
            continue
        per_month.setdefault(rec["deliver_month"], []).append((rec["stage"], rec["quantity"]))

    d = Q / NUM_MONTHS
    assert abs(se.reference_energy(0) - d) < TOL
    # 年度合同 α·Q 相当于每月交割 α·d, 故本月三阶段新增电量之和应为 (1-α)·d
    per_month_expected = (1.0 - 0.80) * d
    for m, parts in per_month.items():
        total = sum(q for _, q in parts)
        assert abs(total - per_month_expected) < 1e-9, (m, parts, total)
    assert len(per_month) == NUM_MONTHS
    print(f"  [PASS] 每月 d = {d:.4f} MWh 中, 月度+月内+现货 = (1-α)·d = "
          f"{per_month_expected:.4f} MWh, 年度合同补足 α·d")


def test_spot_residual_tracks_actual_demand():
    """现货/偏差结算: 电量 = A_m - γ_m·R_m; 平价下成本 = 偏差 × price (不再重归一化)。"""
    rng = np.random.default_rng(1)
    gamma = 0.95
    for _ in range(50):
        # 任意"执行后"逐时段净需求 (非负, 总量 ≥ R_m ⇒ 全部为缺口买入)
        shape = np.abs(rng.normal(1.0, 0.25, 24))
        demand = shape / shape.sum() * (Q / NUM_MONTHS) * rng.uniform(1.0, 1.25)
        se = SettlementEngine(Q, NUM_MONTHS)
        se.settle_annual(0.80, PRICE)
        for m in range(NUM_MONTHS):
            se.settle_monthly(0.90, PRICE, m)
            se.settle_intraday(gamma, PRICE, m)
            se.settle_spot(np.full(24, PRICE), m, demand)
        actual = float(se.actual_demand.sum())
        assert abs(se.energy_spot - (actual - gamma * Q)) < 1e-9, (se.energy_spot, actual)
        assert abs(se.get_total_cost() - PRICE * actual) < 1e-6, se.get_total_cost()
    print("  [PASS] 50 组随机执行结果: 现货电量 = 实际需求 - 合同量, 平价成本可手算")


def test_surplus_is_discounted_not_refunded():
    """过购 (A_m < γ·R_m) 按折价回售: 现货成本为负但幅度小于合同成本, 已签合同不被冲销。"""
    se = SettlementEngine(Q, NUM_MONTHS)
    se.settle_annual(0.80, PRICE)
    ref = se.reference_energy(0)
    se.settle_monthly(0.95, PRICE, 0)
    se.settle_intraday(0.95, PRICE, 0)
    demand = np.full(24, 0.5 * ref / 24)  # 实际需求只有 R_m 的一半, 合同必然过购
    spot = se.settle_spot(np.full(24, PRICE), 0, demand)
    contract = se.gamma * ref
    assert spot < 0, spot
    assert abs(spot + PRICE * (contract - float(demand.sum()))
               * (1 - se.sell_discount)) < 1e-9, spot
    assert abs(spot) < PRICE * contract, (spot, contract)  # 回售收入小于该月已付合同成本
    assert se.cost_annual > 0 and se.cost_monthly > 0
    closure = se.get_monthly_closure()[0]
    assert abs(closure["total"] - closure["actual_energy"]) < 1e-9, closure
    print(f"  [PASS] 过购回售 {spot:,.0f} 元 (折价 {se.sell_discount:.0%}), "
          f"合同成本 {se.cost_annual + se.cost_monthly:,.0f} 元保持不变, 无退款冲销")


def test_price_sensitivity():
    """非平价: 成本 = Σ 各阶段电量 × 各阶段价格 (手工复算)。"""
    se = SettlementEngine(Q, NUM_MONTHS)
    prices = {"annual": 300.0, "monthly": 350.0, "intraday": 380.0, "spot": 420.0}
    se.settle_annual(0.8, prices["annual"])
    for m in range(NUM_MONTHS):
        se.settle_monthly(0.9, prices["monthly"], m)
        se.settle_intraday(0.95, prices["intraday"], m)
        se.settle_spot(np.full(24, prices["spot"]), m)

    hand = (0.8 * Q * prices["annual"] + 0.1 * Q * prices["monthly"]
            + 0.05 * Q * prices["intraday"] + 0.05 * Q * prices["spot"])
    assert abs(se.get_total_cost() - hand) < 1e-6, (se.get_total_cost(), hand)
    print(f"  [PASS] 分阶段价格成本 = 手工复算 {hand:.2f} 元")


def test_ledger_and_monthly_closure():
    """合同台账: 每个交割月四类电量之和恰为 R_m, 费用可手算, 签约时序不倒退。"""
    raw_R = [60, 90, 85, 70, 75, 80, 95, 100, 90, 80, 75, 65]  # 非均匀月度形状
    se = SettlementEngine(total_demand=Q, num_months=NUM_MONTHS, monthly_reference=raw_R)
    scale = Q / sum(raw_R)
    assert abs(se.R.sum() - Q) < 1e-9, se.R.sum()
    assert abs(se.reference_energy(0) - raw_R[0] * scale) < 1e-9

    rng = np.random.default_rng(5)
    alpha = 0.6
    betas = rng.uniform(0.7, 0.95, NUM_MONTHS)
    gammas = np.minimum(betas + rng.uniform(0.0, 0.05, NUM_MONTHS), 1.0)

    step = 0
    se.settle_annual(alpha, 300.0, signed_step=step); step += 1
    for m in range(NUM_MONTHS):
        se.settle_monthly(float(betas[m]), 350.0, m, signed_step=step); step += 1
        se.settle_intraday(float(gammas[m]), 380.0, m, signed_step=step); step += 1
        se.settle_spot(np.full(24, 420.0), m, signed_step=step); step += 1

    closure = se.get_monthly_closure()
    assert len(closure) == NUM_MONTHS
    for m, slot in closure.items():
        assert abs(slot["total"] - slot["reference"]) < 1e-9, (m, slot)
        assert abs(slot["annual"] - alpha * slot["reference"]) < 1e-9, (m, slot)

    ledger = se.get_ledger()
    assert len(ledger) == NUM_MONTHS * 4, len(ledger)
    assert all(r["signed_step"] >= 0 for r in ledger)
    assert [r["signed_step"] for r in ledger] == sorted(r["signed_step"] for r in ledger), "签约时序倒退"
    assert all(r["quantity"] >= -TOL for r in ledger)

    hand = 0.0
    for rec in ledger:
        if isinstance(rec["price"], np.ndarray):
            hand += float(np.sum(rec["price"] * rec["quantities"]))
        else:
            hand += rec["quantity"] * rec["price"]
    assert abs(hand - se.get_total_cost()) < 1e-6, (hand, se.get_total_cost())
    print(f"  [PASS] 台账 {len(ledger)} 条: 逐交割月电量闭合于 R_m, "
          f"手算成本 {hand:,.2f} = 系统成本 {se.get_total_cost():,.2f}")


if __name__ == "__main__":
    tests = [
        ("平价会计闭合", test_flat_price_exact),
        ("随机覆盖率/跨月下降", test_cross_month_decrease_and_random),
        ("合同不可推翻", test_no_overturn),
        ("逐月电量分配", test_monthly_energy_split),
        ("合同台账/逐月交割", test_ledger_and_monthly_closure),
        ("现货偏差跟随实际需求", test_spot_residual_tracks_actual_demand),
        ("过购折价回售不退款", test_surplus_is_discounted_not_refunded),
        ("非平价成本复算", test_price_sensitivity),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as exc:
            failed.append((name, exc))
            print(f"  [FAIL] {exc}")
    print(f"\n通过 {len(tests) - len(failed)}/{len(tests)}")
    if failed:
        sys.exit(1)
