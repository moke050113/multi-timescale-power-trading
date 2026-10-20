"""
环境端到端会计验收测试 (审计修复路线 第 2 步)

验证 TradingEnvironment 在新的逐月结算下:
    1. 全年成交电量恒等于 Q (随机策略 200 回合, 含柔性动作)
    2. 无负电量、无 NaN/Inf 奖励、37 步正常终止
    3. 固定比例基准在平价市场下成本 = Q × price
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.market.environment import TradingEnvironment
from src.utils.config import TOTAL_DEMAND


def _rollout(env, rng, episodes=1):
    """随机策略 rollout, 返回每回合 (能量总量, 市场成本, 步数, 奖励绝对值和)。"""
    out = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_r, steps = 0.0, 0
        while True:
            action = np.array([rng.uniform(-5, 5), rng.uniform(-1, 1)], dtype=np.float32)
            obs, r, term, trunc, info = env.step(action)
            assert np.isfinite(r), r
            total_r += r
            steps += 1
            if term or trunc:
                break
        energy = env.settlement.get_energy_breakdown()
        out.append((energy, env.settlement.get_cost_breakdown(), total_r, steps))
    return out


def test_random_policy_energy_closure():
    rng = np.random.default_rng(7)
    env = TradingEnvironment(include_flexible=True, seed=42)
    recs = _rollout(env, rng, episodes=200)
    for energy, cost, total_r, steps in recs:
        assert abs(energy["total"] - TOTAL_DEMAND) < 1e-6, energy
        assert all(v >= -1e-9 for v in energy.values()), energy
        assert steps == 37, steps
    print(f"  [PASS] 200 回合随机策略: 成交电量恒为 {TOTAL_DEMAND:.1f} MWh, 37 步, 奖励有限")
    print(f"         成本区间 [{min(c['total'] for _, c, _, _ in recs):,.0f}, "
          f"{max(c['total'] for _, c, _, _ in recs):,.0f}] 元")


def test_flat_price_fixed_ratio():
    """把价格场景手工压成平价, 固定比例基准成本应等于 Q × price。"""
    env = TradingEnvironment(scenario_name="C_各阶段价格接近", include_flexible=False, seed=42)
    price = 400.0

    env.reset()
    template = dict(env.prices)

    def flat_generate():
        p = dict(template)
        p["annual_price"] = price
        p["monthly_prices"] = np.full(env.num_months, price)
        p["intraday_prices"] = np.full(env.num_months, price)
        p["spot_prices"] = np.full((env.num_months, 24), price)
        return p

    # get_fixed_ratio_cost 内部会 reset() 并重新生成价格, 因此需要替换价格生成器
    env.price_gen.generate = flat_generate
    res = env.get_fixed_ratio_cost()
    print(f"  平价固定比例成本: {res['total_cost']:,.2f} 元, 仓位 {res['positions']['alpha']:.2f}")
    assert abs(res["total_cost"] - TOTAL_DEMAND * price) < 1e-6, res["total_cost"]
    energy = env.settlement.get_energy_breakdown()
    assert abs(energy["total"] - TOTAL_DEMAND) < 1e-6, energy
    print(f"  [PASS] 平价固定比例 = Q × price = {TOTAL_DEMAND * price:,.0f} 元")


def test_no_flex_benchmark_uses_month_basis():
    """无柔性基准应基于本月敞口电量, 而不是全年 Q。"""
    env = TradingEnvironment(include_flexible=False, seed=42)
    env.reset()
    # 走一遍到某月现货步前, 手工比对
    for _ in range(3):
        env.step(np.array([2.0, 0.0], dtype=np.float32))
    d = env.settlement.reference_energy(env.current_month)
    spot_prices = env.prices["spot_prices"][env.current_month]
    month_spot_energy = d * (1.0 - env.gamma)
    expected = float(np.sum(spot_prices * month_spot_energy / 24))

    cost_before = env.settlement.cost_spot
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    actual = env.settlement.cost_spot - cost_before
    assert abs(actual - expected) < 1e-6, (actual, expected)
    print(f"  [PASS] 无柔性时现货成本 = 敞口电量({month_spot_energy:.2f} MWh) × 现货价格, "
          f"基准为月需求 d={d:.2f} MWh")


if __name__ == "__main__":
    tests = [
        ("随机策略能量闭合", test_random_policy_energy_closure),
        ("平价固定比例基准", test_flat_price_fixed_ratio),
        ("月基准现货电量", test_no_flex_benchmark_uses_month_basis),
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
