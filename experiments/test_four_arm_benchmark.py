"""
四臂基准框架验收 (任务 #40)

要证明的是"实验装置本身可信", 不是任何策略的经济结论:
    1. 固定比例臂的覆盖率与 FIXED_RATIOS 完全一致 (反解公式可手算)
    2. 固定比例臂的成本 = 环境内置基准 get_fixed_ratio_cost 在同一世界上的成本
    3. 规则臂只用已发布信息: 篡改未来价格实现, 首月 4 个决策的动作不变
    4. 现货调度强度 = 规则公式手算值 (含单位换算)
    5. 四个臂在同一批世界上电量守恒 / 无虚假退款 / 设备按时完成任务
    6. 分层臂只训中长期: 调度 SAC 不写经验, 下层强度确实来自规则
    7. 四个臂共用同一个动作可达集, 越界动作被环境计数并判为对照不成立 (审计 N02)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_four_arm_benchmark import (
    ARMS, _accounting_checks, check_failed, make_arm, rollout,
)
from src.agent.rule_policies import (
    STAGE_SPOT, V2G_GATE, forecast_intensity, raw_for_coverage, rolling_rule_action,
)
from src.market.environment import TradingEnvironment
from src.utils.config import (
    ACTION_BOUNDS, FIXED_RATIOS, INDUSTRIAL_LOAD_PARAMS, MARKET_BOUNDS, PRICE_SCENARIOS,
)

SCENARIOS = list(PRICE_SCENARIOS.keys())
SEED = 42
WORLDS = 4


def _env(scenario="A_年度低价后期上涨"):
    return TradingEnvironment(scenario_name=scenario, load_scenario="柔性适中",
                              use_rule_dispatcher=False, seed=SEED)


def test_fixed_ratio_arm_hits_configured_coverage():
    arm = make_arm("fixed_rule", state_dim=13, seed=SEED)
    env = _env()
    for w in range(WORLDS):
        rec = rollout(env, arm, world=w, deterministic=True)
        assert all(abs(v - FIXED_RATIOS["alpha"]) < 1e-6 for v in rec["alphas"]), rec["alphas"]
        assert all(abs(v - FIXED_RATIOS["beta"]) < 1e-6 for v in rec["betas"]), rec["betas"]
        assert all(abs(v - FIXED_RATIOS["gamma"]) < 1e-6 for v in rec["gammas"]), rec["gammas"]
        assert all(abs(v) < 1e-9 for v in rec["intensities"]), "保障调度臂强度必须为 0"
    print(f"  [PASS] 固定比例臂 {WORLDS} 个世界上 α/β/γ = "
          f"{FIXED_RATIOS['alpha']}/{FIXED_RATIOS['beta']}/{FIXED_RATIOS['gamma']} 逐位达标")


def test_fixed_ratio_arm_matches_builtin_benchmark():
    env = _env()
    arm = make_arm("fixed_rule", state_dim=13, seed=SEED)
    for w in range(WORLDS):
        mine = rollout(env, arm, world=w, deterministic=True)["cost"]
        builtin = env.get_fixed_ratio_cost(world=w)["total_cost"]
        assert abs(mine - builtin) / builtin < 1e-6, (w, mine, builtin)
    print(f"  [PASS] 规则臂走环境 step 路径的成本与内置基准逐世界一致 (<0.0001%)")


def _actions_by_obs(arm, env, mutate_future=None):
    """记录每个决策 (世界0) 的 (stage, 动作, 观察); mutate_future 在首步执行前调用一次。"""
    obs, _ = env.reset(options={"world": 0})
    arm.begin_episode()
    actions = []
    pending_mutation = mutate_future is not None
    while True:
        stage = env.current_stage
        if pending_mutation:
            mutate_future(env)
            pending_mutation = False
        action = arm.act(obs, stage, deterministic=True)
        actions.append((stage, np.array(action, dtype=float), np.array(obs, dtype=float)))
        next_obs, _, term, _, _ = env.step(action)
        obs = next_obs
        if term:
            break
    return actions


def test_rule_arms_use_only_published_information():
    """篡改未来 11 个月的月度/月内/现货实现价后, 首月 4 个决策的动作必须不变。"""
    def tamper(env):
        env.prices["monthly_prices"][1:] *= 3.0
        env.prices["intraday_prices"][1:] *= 3.0
        env.prices["spot_prices"][1:] *= 0.2

    for arm_name in ("fixed_rule", "rolling_rule"):
        base = _actions_by_obs(make_arm(arm_name, 13, SEED), _env())
        mut = _actions_by_obs(make_arm(arm_name, 13, SEED), _env(), tamper)
        assert all(np.allclose(a[1], b[1], atol=1e-6)
                   for a, b in zip(base[:4], mut[:4])), arm_name
        changed = sum(0 if np.allclose(a[1], b[1], atol=1e-6) else 1
                      for a, b in zip(base[4:], mut[4:]))
        if arm_name == "rolling_rule":
            # 规则用到了后续月份的公开报价, 因此只有这些决策才应改变
            assert changed > 0, arm_name
        else:
            assert changed == 0, "固定比例臂不看价格, 任何步都不应改变"
    print("  [PASS] 规则臂首月 4 个决策对未来价格实现不敏感; "
          "滚动规则只在后续月份公开报价出现后才改变")


def test_spot_dispatch_strength_is_hand_computable():
    """
    强度 = clip((波幅 − 转移惩罚) / (门槛 × 水平), 0, 1);
    波幅/水平 超过门槛时取负号 (V2G) 并按"本月剩余偏差空间"封顶,
    否则取正号 (只做充电侧移峰填谷)。
    """
    env = _env("D_现货高波动")
    obs, _ = env.reset(options={"world": 1})
    seen = []
    while True:
        stage = env.current_stage
        if stage == STAGE_SPOT:
            level = float(obs[6]) * 1000.0
            spread = float(obs[7]) * 1000.0
            remaining = float(obs[10])
            strength = np.clip((spread - INDUSTRIAL_LOAD_PARAMS["transfer_cost"])
                               / (V2G_GATE * level), 0.0, 1.0)
            if strength > 0.0 and spread / level > V2G_GATE:
                expected = -min(strength, remaining / MARKET_BOUNDS["spot"]["max"])
            else:
                expected = strength
            action = rolling_rule_action(stage, np.asarray(obs, dtype=np.float64))
            assert abs(float(action[1]) - expected) < 1e-6, (expected, action[1], level, spread)
            assert abs(forecast_intensity(np.asarray(obs, dtype=np.float64))
                       - float(action[1])) < 1e-9
            seen.append((level, spread, remaining, float(action[1])))
        next_obs, _, term, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        obs = next_obs
        if term:
            break
    assert len(seen) == 12
    assert all(abs(v[3]) > 1e-6 for v in seen), "高波动场景下规则应给出非零调度"
    assert all(v[3] < 0 for v in seen), "波幅超门槛的月份应选放电符号"
    print(f"  [PASS] 12 个月现货强度逐月手算吻合, 例如 水平={seen[0][0]:.0f} "
          f"波幅={seen[0][1]:.0f} 缺口={seen[0][2]:.3f} -> 强度={seen[0][3]:+.2f} "
          f"(门槛={V2G_GATE:.3f})")


def test_all_arms_pass_accounting_checks():
    """四臂 (SAC 臂用未初始化网络) 在同一批世界上都要过会计与设备验收。"""
    for scenario in SCENARIOS:
        env = _env(scenario)
        for arm_name in ARMS:
            arm = make_arm(arm_name, state_dim=13, seed=SEED)
            if arm_name == "sac_rule":
                arm.set_phase(freeze_midlong=False, freeze_spot=True)
            for w in range(WORLDS):
                rec = rollout(env, arm, world=w, deterministic=True)
                assert not check_failed(rec["checks"]), (scenario[:2], arm_name, w, rec["checks"])
    print(f"  [PASS] {len(SCENARIOS)} 场景 x {len(ARMS)} 臂 x {WORLDS} 世界: "
          f"电量守恒 / 无虚假退款 / 费用可复算 / 设备按时完成任务 全部通过")


def test_hierarchical_arm_trains_only_midlong():
    env = _env()
    arm = make_arm("sac_rule", state_dim=13, seed=SEED)
    arm.set_phase(freeze_midlong=False, freeze_spot=True)
    for _ in range(2):
        rollout(env, arm, world=None, deterministic=False, learn=True)
    assert len(arm.coord.spot_agent.replay_buffer) == 0, "分层臂不应训练调度 SAC"
    assert len(arm.coord.midlong_agent.replay_buffer) == 50, "2 回合 x 25 条中长期决策"

    # 下层动作确实来自规则: 评价时现货强度 = forecast_intensity(观察)
    obs, _ = env.reset(options={"world": 0})
    arm.begin_episode()
    checked = 0
    while True:
        stage = env.current_stage
        action = arm.act(obs, stage, deterministic=True)
        if stage == STAGE_SPOT:
            assert abs(float(action[1]) - forecast_intensity(obs)) < 1e-9
            assert float(action[0]) == 0.0, "分层臂现货步不应输出合同动作"
            checked += 1
        obs, _, term, _, _ = env.step(action)
        if term:
            break
    assert checked == 12
    print("  [PASS] 分层臂: 调度 SAC 零经验, 中长期 25 条/回合, 现货强度来自预测调度规则")


def test_arms_share_one_action_reachability():
    """
    对照成立的前提: 四个臂面对**同一个动作可达集**。

    修正前规则臂反解覆盖率用 cap=8, 而 SAC 的 tanh×scale 最多 ±5,
    规则臂因此能走到学习臂根本输出不了的动作 (审计 N02)。
    """
    env = _env()
    env.reset(options={"world": 0})
    hi = np.asarray(env.action_space.high, dtype=float)
    assert np.allclose(hi, [ACTION_BOUNDS["ratio_raw_limit"],
                            ACTION_BOUNDS["intensity_limit"]]), hi
    assert np.allclose(env.action_space.low, -hi)

    # 反解不到目标覆盖率时如实停在可达边界 (该边界 SAC 同样能输出)
    cap = ACTION_BOUNDS["ratio_raw_limit"]
    ann_hi = MARKET_BOUNDS["annual"]["max"]
    assert abs(raw_for_coverage(ann_hi, 0.0, ann_hi) - cap) < 1e-12
    assert abs(raw_for_coverage(0.0, 0.0, ann_hi) + cap) < 1e-12
    reachable = 1.0 / (1.0 + np.exp(-cap))
    assert reachable < 1.0 - 1e-6, "边界已能覆盖全部, 反向对照失去判别力"

    # SAC 臂尺度 = 盒子上界: tanh 输出恒在 ±scale 内, 不会被环境再裁一刀
    arm = make_arm("dual_sac", state_dim=13, seed=SEED)
    assert arm.coord.midlong_agent.action_scale == cap
    assert arm.coord.spot_agent.action_scale == hi[1]

    for arm_name in ARMS:
        a = make_arm(arm_name, state_dim=13, seed=SEED)
        if arm_name == "sac_rule":
            a.set_phase(freeze_midlong=False, freeze_spot=True)
        # 确定性动作逐条落在盒子内 (规则臂与学习臂同界)
        acts = [np.asarray(x[1]) for x in _actions_by_obs(a, _env())]
        assert all(np.all(np.abs(x) <= hi + 1e-6) for x in acts), arm_name
        for w in range(WORLDS):
            for det in (True, False):
                rec = rollout(env, a, world=w, deterministic=det)
                assert rec["checks"]["action_oob"] == 0, (arm_name, w, det, rec["checks"])
                assert not check_failed(rec["checks"])

    # 反向用例: 越界动作不静默接受 —— 裁剪后计入 action_oob, 结果与边界动作一致
    ref = TradingEnvironment(scenario_name="A_年度低价后期上涨", load_scenario="柔性适中",
                             use_rule_dispatcher=False, seed=SEED)
    ref.reset(options={"world": 0})
    _, _, _, _, clipped_info = ref.step(np.array([cap, hi[1]], dtype=np.float32))
    oob = TradingEnvironment(scenario_name="A_年度低价后期上涨", load_scenario="柔性适中",
                             use_rule_dispatcher=False, seed=SEED)
    oob.reset(options={"world": 0})
    _, _, _, _, oob_info = oob.step(np.array([8.0, 2.0], dtype=np.float32))
    assert oob.action_oob == 1 and ref.action_oob == 0
    assert abs(oob_info["alpha"] - clipped_info["alpha"]) < 1e-9, (oob_info["alpha"], clipped_info["alpha"])
    assert "action_out_of_bounds" in check_failed(
        {**_accounting_checks(oob, oob_info), "steps_37": True}), "越界未被验收拦下"
    print(f"  [PASS] 四臂共用可达集 ±{cap}/±{hi[1]}: {len(ARMS)} 臂 x {WORLDS} 世界 x 2 模式零越界; "
          f"cap=8 的越界动作被裁剪并判为对照不成立")


if __name__ == "__main__":
    tests = [
        test_fixed_ratio_arm_hits_configured_coverage,
        test_fixed_ratio_arm_matches_builtin_benchmark,
        test_rule_arms_use_only_published_information,
        test_spot_dispatch_strength_is_hand_computable,
        test_all_arms_pass_accounting_checks,
        test_arms_share_one_action_reachability,
        test_hierarchical_arm_trains_only_midlong,
    ]
    print("=== 四臂基准框架验收 (#40) ===")
    for fn in tests:
        print(f"--- {fn.__name__} ---")
        fn()
    print(f"\n通过 {len(tests)}/{len(tests)}")
