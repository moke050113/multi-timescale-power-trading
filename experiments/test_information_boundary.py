"""
信息边界验收测试 (审计修复路线 第 4 步 / 任务 #39)

对应审计"最低验收门槛"里的 **观察不含未来实现值**:
    1. 观察中每个价格槽位都能按"公布时点规则"手算 (未开市市场记 0, 未来实现值不可见)
    2. 篡改未来月份的价格实现后, 之前决策的观察与奖励必须逐位不变
    3. 本月日前剖面只在现货决策时点可见; 交割之后只能以"已交割历史统计"被引用
    4. 未来暴露成本塑形只读公开滚动预测, 不读生成器隐含分布参数
       (把 params 污染成 1e6 之后整回合成本/奖励不变; 并逐月手算该预测值)
    5. 同一 (世界种子, 世界编号) 的观察序列完全可复现 — 四种策略必须在同一年市场上比较
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.market.environment import TradingEnvironment

NUM_STEPS = 37  # 1 年度 + 12 x 3
ATOL = 1e-6     # 观察存成 float32, 留 1e-6 容差


def _expected_price_slots(prices, stage, month):
    """
    按公布规则手算观察的 5 个价格槽位: (年度, 月度, 月内, 现货水平, 现货波幅)。

    槽位语义 = "该市场最近一次已公布的报价" (0 = 尚未开市过)。
    月内报价在月度决策时点上显示的是上一个月的成交价, 属于公开历史而非未来实现值。
    """
    annual = float(prices["annual_price"])
    if stage == TradingEnvironment.STAGE_ANNUAL:
        return annual, 0.0, 0.0, annual, 0.0

    if stage == TradingEnvironment.STAGE_SPOT:
        day = np.asarray(prices["spot_prices"][month], dtype=float)
        level, spread = float(day.mean()), float(day.max() - day.min())
    elif month == 0:
        level, spread = annual, 0.0          # 尚无已交割月份: 退回公开年度合同价
    else:
        delivered = np.asarray([prices["spot_prices"][m] for m in range(month)], dtype=float)
        levels = delivered.mean(axis=1)
        spreads = delivered.max(axis=1) - delivered.min(axis=1)
        level, spread = float(levels.mean()), float(spreads.mean())

    monthly = float(prices["monthly_prices"][month])
    if stage >= TradingEnvironment.STAGE_INTRADAY:
        intraday = float(prices["intraday_prices"][month])
    elif month > 0:
        intraday = float(prices["intraday_prices"][month - 1])   # 上月成交价 = 公开历史
    else:
        intraday = 0.0
    return annual, monthly, intraday, level, spread


def _decision_trace(env, world=0, ratio_raw=0.5, intensity=0.2, mutate=None):
    """
    走一整回合, 记录"每个决策时点的观察 + 该步执行结果"。

    返回 (decisions, steps):
        decisions[i] = 第 i 个决策的观察向量与其发布快照
        steps[i]     = 第 i 次 step 执行的阶段/奖励/仓位/该步台账成本
    """
    obs, info = env.reset(options={"world": world})
    if mutate is not None:
        mutate(env)
        obs, info = env._get_obs(), env._get_info()

    decisions = [{
        "obs": np.array(obs, dtype=np.float64),
        "stage": info["obs_stage"],
        "month": info["obs_month"],
        "published": dict(info["published"]),
    }]
    steps = []

    for _ in range(NUM_STEPS):
        obs, reward, term, trunc, info = env.step(
            np.array([ratio_raw, intensity], dtype=np.float32))
        steps.append({
            "stage": info["stage"],
            "reward": float(reward),
            "reward_midlong": float(info["reward_midlong"]),
            "alpha": float(env.alpha),
            "beta": float(env.beta),
            "gamma": float(env.gamma),
            "cost": float(env.settlement.get_ledger()[-1]["cost"]),
            "deliver_month": int(env.settlement.get_ledger()[-1]["deliver_month"]),
        })
        decisions.append({
            "obs": np.array(obs, dtype=np.float64),
            "stage": info["obs_stage"],
            "month": info["obs_month"],
            "published": dict(info["published"]),
        })
        if term or trunc:
            break

    assert len(steps) == NUM_STEPS, len(steps)
    return decisions, steps


def test_every_price_slot_is_publishable():
    """每个决策的观察槽位 = 公布规则手算值; 未开市市场为 0, 未来实现值不出现。"""
    env = TradingEnvironment(seed=11)
    decisions, _ = _decision_trace(env, world=3)
    prices = env.prices

    for i, d in enumerate(decisions[:-1]):  # 末条是终局状态, 不再驱动决策
        annual, monthly, intraday, level, spread = _expected_price_slots(
            prices, d["stage"], d["month"])
        expected = np.array([annual, monthly, intraday, level, spread]) / 1000.0
        got = d["obs"][3:8]
        assert np.allclose(got, expected, atol=ATOL), (i, d["stage"], d["month"], got, expected)

    # 年度决策看不到任何月度/月内报价, 也看不到任何现货实现值
    first = decisions[0]
    assert first["stage"] == TradingEnvironment.STAGE_ANNUAL
    assert first["obs"][4] == 0.0 and first["obs"][5] == 0.0
    assert first["published"]["spot_day_published"] is False
    print("  [PASS] 37 个决策的观察价格槽位逐个手算吻合 (未开市记 0, 无未来实现值)")


def test_future_realizations_cannot_change_earlier_observations():
    """篡改未来月份的全部价格实现: 首月 4 个决策的观察与奖励逐位不变。"""
    ref = TradingEnvironment(seed=11)
    decisions_a, steps_a = _decision_trace(ref, world=2)
    months = ref.num_months
    untouched = copy.deepcopy(ref.prices)

    tampered = TradingEnvironment(seed=11)

    def _mutate(env):
        for m in range(1, months):
            env.prices["monthly_prices"][m] = 5.0 * untouched["monthly_prices"][m] + 900.0
            env.prices["intraday_prices"][m] = 10.0 * m
            env.prices["spot_prices"][m] = np.full_like(env.prices["spot_prices"][m],
                                                        500.0 + 40.0 * m)

    decisions_b, steps_b = _decision_trace(tampered, world=2, mutate=_mutate)
    assert not np.allclose(tampered.prices["spot_prices"][1], untouched["spot_prices"][1]), \
        "篡改未生效"

    for i in range(4):  # 年度 / 月0月度 / 月0月内 / 月0现货
        assert np.allclose(decisions_a[i]["obs"], decisions_b[i]["obs"], atol=0.0), \
            (i, decisions_a[i]["obs"], decisions_b[i]["obs"])
    for i in range(4):
        assert steps_a[i]["reward"] == steps_b[i]["reward"], (i, steps_a[i], steps_b[i])

    changed = [i for i in range(4, len(decisions_b))
               if not np.allclose(decisions_a[i]["obs"], decisions_b[i]["obs"], atol=0.0)]
    assert changed, "未来实现值变化后, 后续决策的观察理应随之改变"
    print(f"  [PASS] 未来 11 个月价格被完全篡改: 首月 4 个决策观察/奖励逐位相同, "
          f"其后 {len(changed)} 个决策才更新")


def test_day_ahead_snapshot_only_at_spot_decision():
    """发布游标本身: 日前剖面仅在现货决策可见, 历史条数 = 已交割月份数。"""
    env = TradingEnvironment(seed=11)
    decisions, _ = _decision_trace(env, world=1)

    for i, d in enumerate(decisions[:-1]):
        published = d["published"]
        assert published["spot_day_published"] == (d["stage"] == TradingEnvironment.STAGE_SPOT), \
            (i, d)
        assert published["delivered_months"] == d["month"], (i, d)

    last = decisions[-1]
    assert last["published"]["delivered_months"] == env.num_months, last
    assert last["published"]["monthly"] > 0.0
    print("  [PASS] 日前剖面仅在现货决策时点发布; delivered_months 随交割逐月 +1")


def test_poisoned_generator_params_do_not_change_anything():
    """生成器隐含分布参数 (spot_mean 等) 污染成 1e6: 整回合观察/奖励/成本不变。"""
    clean = TradingEnvironment(seed=5)
    decisions_a, steps_a = _decision_trace(clean, world=4)

    blinded = TradingEnvironment(seed=5)

    def _poison(env):
        # 只换实例属性, 不动全局 PRICE_SCENARIOS
        poisoned = dict(env.price_gen.params)
        poisoned.update({"annual": 1e6, "monthly": 1e6, "intraday": 1e6,
                         "spot_mean": 1e6, "spot_std": 1e6})
        env.price_gen.params = poisoned

    decisions_b, steps_b = _decision_trace(blinded, world=4, mutate=_poison)
    assert blinded.price_gen.params["spot_mean"] == 1e6, "确认真的在参数被污染的环境下运行"
    assert np.allclose(blinded.prices["spot_prices"], clean.prices["spot_prices"]), \
        "同一世界应当给出同一年市场"

    for i in range(len(decisions_a)):
        assert np.allclose(decisions_a[i]["obs"], decisions_b[i]["obs"], atol=0.0), i
    for i in range(NUM_STEPS):
        assert steps_a[i]["reward"] == steps_b[i]["reward"], (i, steps_a[i], steps_b[i])
        assert steps_a[i]["cost"] == steps_b[i]["cost"], (i, steps_a[i], steps_b[i])
    print("  [PASS] 隐参污染 (spot_mean=1e6) 后 37 步观察/奖励/成本完全一致")


def test_exposure_shaping_uses_public_rolling_forecast():
    """逐月复算暴露塑形项: (1-γ_m) x 已交割月份日前均价的滚动均值。"""
    env = TradingEnvironment(seed=13, risk_lambda=0.0, exposure_shaping=True)
    _, steps = _decision_trace(env, world=6)
    prices = env.prices
    months = env.num_months

    for m in range(months):
        st = steps[2 + 3 * m]          # 年度步占 0, 之后每月 3 步: 月度/月内/现货
        assert st["stage"] == "intraday", (m, st)
        if m == 0:
            forecast = float(prices["annual_price"])   # 无历史 -> 公开年度合同价
        else:
            delivered = np.asarray([prices["spot_prices"][k] for k in range(m)], dtype=float)
            forecast = float(delivered.mean(axis=1).mean())
        basis = env.settlement.reference_energy(m)
        expected = -st["cost"] / basis - (1.0 - st["gamma"]) * forecast
        assert abs(st["reward_midlong"] - expected) < 1e-6, \
            (m, st["reward_midlong"], expected, forecast)

    print(f"  [PASS] {months} 个月月内步的塑形项 = (1-gamma_m) x 公开滚动预测, 逐月复算吻合")


def test_paired_world_observations_reproducible():
    """同一 (世界种子, 世界编号) 重复采样: 观察序列逐位一致 (四策略配对评价的前提)。"""
    a = TradingEnvironment(seed=17)
    b = TradingEnvironment(seed=17)
    decisions_a, steps_a = _decision_trace(a, world=9)
    decisions_b, steps_b = _decision_trace(b, world=9)

    for i in range(len(decisions_a)):
        assert np.array_equal(decisions_a[i]["obs"], decisions_b[i]["obs"]), i
    for i in range(NUM_STEPS):
        assert steps_a[i]["cost"] == steps_b[i]["cost"], i

    other = TradingEnvironment(seed=17)
    decisions_c, _ = _decision_trace(other, world=10)
    assert not np.allclose(a.prices["spot_prices"], other.prices["spot_prices"]), \
        "换世界编号应得到另一年的市场"
    later_diff = [i for i in range(len(decisions_a))
                  if not np.array_equal(decisions_a[i]["obs"], decisions_c[i]["obs"])]
    assert later_diff, "不同世界应在若干决策上给出不同观察"
    print(f"  [PASS] 配对世界可复现: 同种子同编号观察逐位一致; 换编号后 "
          f"{len(later_diff)}/{len(decisions_a)} 个决策观察改变")


if __name__ == "__main__":
    tests = [
        test_every_price_slot_is_publishable,
        test_future_realizations_cannot_change_earlier_observations,
        test_day_ahead_snapshot_only_at_spot_decision,
        test_poisoned_generator_params_do_not_change_anything,
        test_exposure_shaping_uses_public_rolling_forecast,
        test_paired_world_observations_reproducible,
    ]
    print("=== 信息边界验收 (#39) ===")
    for fn in tests:
        print(f"--- {fn.__name__} ---")
        fn()
    print(f"\n通过 {len(tests)}/{len(tests)}")
