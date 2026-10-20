"""
双智能体决策时序 + 模型重载验收测试 (审计修复路线 第 5 步 / 任务 #36)

对应审计门槛里的 **"观察不含未来实现值"之外的两条**:
    - **各角色按自己的决策时序学习**: 经验的 next_state 必须是该角色下一次决策的观察,
      跨 Δt 个环境步时用 gamma**Δt 折扣; 回合结束时未闭合的决策按终止处理。
    - **模型重载一致**: 保存/加载后动作逐位一致, 且温度参数优化器仍然绑定到同一个
      log_alpha 张量 (旧实现重新赋值 log_alpha, 会让 alpha_optim 与参数脱钩,
      加载后续训时温度不再更新且不报错)。
    - 结构/超参指纹: 动作尺度或网络维度不符的检查点拒绝加载。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.agent.multi_agent import MultiScaleCoordinator
from src.agent.sac_agent import SACAgent
from src.market.environment import TradingEnvironment
from src.utils.config import DRL_PARAMS

MIDLONG_STEPS = 1 + 12 * 2   # 年度 1 + 12 x (月度 + 月内)
SPOT_STEPS = 12


def _episode_via_coordinator(env, coord, deterministic=True):
    """跑一整回合, 同时记录角色决策流 (协调器经验池随之写入)。"""
    state, _ = env.reset(options={"world": 0})
    coord.begin_episode()
    decisions = []        # (t, role, state, 该角色奖励, next_state, done)
    t = 0
    while True:
        t += 1
        stage = env.current_stage
        role = "spot" if stage == env.STAGE_SPOT else "midlong"
        action = coord.select_action(state, stage, deterministic=deterministic)
        next_state, _, term, _, info = env.step(action)
        rewards = {"reward_midlong": info["reward_midlong"],
                   "reward_spot": info["reward_spot"],
                   "reward_global": info["reward_global"]}
        coord.store_transition(state, action, rewards, next_state, term, stage=stage)
        decisions.append({
            "t": t, "role": role, "state": np.array(state, dtype=np.float64),
            "action": action, "next_state": np.array(next_state, dtype=np.float64),
            "reward_midlong": float(info["reward_midlong"]),
            "reward_spot": float(info["reward_spot"]),
            "done": bool(term),
        })
        state = next_state
        if term:
            break
    return decisions


def _expected_transitions(decisions, role, gamma):
    """
    按"角色自己的决策序列"手算期望经验: (state, reward, next_state, done, discount)。

    末条经验按终止处理, 其 TD 目标不含后继值, 因此 discount 记 None (不参与比对)。
    """
    idx = [i for i, d in enumerate(decisions) if d["role"] == role]
    out = []
    for pos, i in enumerate(idx):
        d = decisions[i]
        reward = d["reward_midlong"] if role == "midlong" else d["reward_spot"]
        if pos + 1 < len(idx):
            nxt = decisions[idx[pos + 1]]
            out.append({
                "state": d["state"], "reward": reward, "next_state": nxt["state"],
                "done": False, "discount": gamma ** (nxt["t"] - d["t"]),
            })
        else:
            out.append({
                "state": d["state"], "reward": reward,
                "next_state": decisions[-1]["next_state"], "done": True,
                "discount": None,
            })
    return out


def _buffer_entries(agent):
    return list(agent.replay_buffer.buffer)


def _assert_matches_expectation(entries, expected, role):
    assert len(entries) == len(expected), (role, len(entries), len(expected))
    for k, (buf, exp) in enumerate(zip(entries, expected)):
        state, action, reward, next_state, done, disc = buf
        assert np.allclose(np.asarray(state, float), exp["state"], atol=1e-6), (role, k)
        assert np.allclose(np.asarray(next_state, float), exp["next_state"], atol=1e-6), \
            (role, k, "next_state 必须是该角色下一次决策的观察")
        assert abs(float(reward) - exp["reward"]) < 1e-6, (role, k)
        assert bool(done) == exp["done"], (role, k)
        if exp["discount"] is not None:
            assert abs(float(disc) - exp["discount"]) < 1e-9, (role, k, disc, exp["discount"])
        assert np.asarray(action).shape == (1,), (role, k, np.asarray(action).shape)


def test_dual_transitions_follow_each_roles_own_decision_stream():
    """每个角色经验的后继状态 = 自己下一次决策的观察, 折扣 = gamma**(两次决策间环境步数)。"""
    env = TradingEnvironment(seed=31)
    coord = MultiScaleCoordinator(state_dim=13, use_dual_agent=True)
    decisions = _episode_via_coordinator(env, coord)

    assert len(decisions) == 37, len(decisions)
    gamma = DRL_PARAMS["gamma"]

    _assert_matches_expectation(_buffer_entries(coord.midlong_agent),
                                _expected_transitions(decisions, "midlong", gamma), "midlong")
    _assert_matches_expectation(_buffer_entries(coord.spot_agent),
                                _expected_transitions(decisions, "spot", gamma), "spot")
    assert len(coord.midlong_agent.replay_buffer) == MIDLONG_STEPS
    assert len(coord.spot_agent.replay_buffer) == SPOT_STEPS

    # 跨步折扣确实存在: 月内步 -> 下月月度步 跨 2 个环境步
    midlong_discs = [float(e[5]) for e in _buffer_entries(coord.midlong_agent)]
    assert any(abs(d - gamma ** 2) < 1e-9 for d in midlong_discs), midlong_discs
    assert any(abs(d - gamma) < 1e-9 for d in midlong_discs)
    print(f"  [PASS] 双角色 {MIDLONG_STEPS}+{SPOT_STEPS} 条经验的后继状态/奖励/折扣逐条手算吻合 "
          f"(含 gamma^2 跨步折扣)")


def test_naive_next_state_would_have_been_wrong():
    """
    反向确认旧口径的错位规模 (按角色各自的决策序列比对)。

    旧口径把"环境下一步观察"当作 next_state, 只有两类决策恰好碰对:
    年度→月度、月度→月内 (同类决策相邻), 以及终局那条 (其后已无观察变化)。
    真正错位的是跨步的那 11 处: 中长期每个月的 月内→下月月度, 现货的 现货→下月现货。
    """
    env = TradingEnvironment(seed=32)
    coord = MultiScaleCoordinator(state_dim=13, use_dual_agent=True)
    decisions = _episode_via_coordinator(env, coord)
    gamma = DRL_PARAMS["gamma"]

    def _naive_mismatch_steps(role):
        acting = [d for d in decisions if d["role"] == role]
        chained = _expected_transitions(decisions, role, gamma)
        return [d["t"] for d, exp in zip(acting, chained)
                if not np.allclose(d["next_state"], exp["next_state"], atol=1e-6)]

    midlong_steps = _naive_mismatch_steps("midlong")
    spot_steps = _naive_mismatch_steps("spot")
    # 12 个月的月内决策在 t=3,6,...,36; 最后一个月内决策已是终局, 不错位
    assert midlong_steps == [3 + 3 * m for m in range(11)], midlong_steps
    # 12 个月现货在 t=4,7,...,37; 终局现货 (t=37) 的后继已是终止状态, 不错位
    assert spot_steps == [4 + 3 * m for m in range(11)], spot_steps
    print(f"  [PASS] 旧口径错位: 中长期 {len(midlong_steps)}/{MIDLONG_STEPS} 条 "
          f"(跨月 t={midlong_steps[0]}..{midlong_steps[-1]}), "
          f"柔性调度 {len(spot_steps)}/{SPOT_STEPS} 条 (跨月 t={spot_steps[0]}..{spot_steps[-1]}) "
          f"— 已按各自决策序列修正")


def test_no_experience_leaks_across_episodes():
    """begin_episode 清空挂起决策: 两回合各自 25/12 条, 第二回合经验完全由自己的决策流决定。"""
    env = TradingEnvironment(seed=33)
    coord = MultiScaleCoordinator(state_dim=13, use_dual_agent=True)

    ep1 = _episode_via_coordinator(env, coord)
    n_mid_1 = len(coord.midlong_agent.replay_buffer)
    n_spot_1 = len(coord.spot_agent.replay_buffer)
    assert (n_mid_1, n_spot_1) == (MIDLONG_STEPS, SPOT_STEPS), (n_mid_1, n_spot_1)

    ep2 = _episode_via_coordinator(env, coord)
    gamma = DRL_PARAMS["gamma"]
    entries_mid = _buffer_entries(coord.midlong_agent)
    entries_spot = _buffer_entries(coord.spot_agent)
    assert len(entries_mid) == 2 * MIDLONG_STEPS and len(entries_spot) == 2 * SPOT_STEPS

    # 第二回合的经验必须与"只按第二回合决策流手算"的结果一致 (有跨回合残留就会不等)
    _assert_matches_expectation(entries_mid[MIDLONG_STEPS:],
                                _expected_transitions(ep2, "midlong", gamma), "midlong-ep2")
    _assert_matches_expectation(entries_spot[SPOT_STEPS:],
                                _expected_transitions(ep2, "spot", gamma), "spot-ep2")
    print(f"  [PASS] 两回合经验互不衔接: 每回合 {MIDLONG_STEPS}/{SPOT_STEPS} 条, 回合末闭合")


def test_frozen_role_stores_nothing():
    """冻结标志: 被冻结角色既不写经验也不留下挂起决策。"""
    env = TradingEnvironment(seed=34)
    coord = MultiScaleCoordinator(state_dim=13, use_dual_agent=True)

    coord.set_training_phase(freeze_midlong=False, freeze_spot=True)
    _episode_via_coordinator(env, coord)
    assert len(coord.spot_agent.replay_buffer) == 0
    assert len(coord.midlong_agent.replay_buffer) == MIDLONG_STEPS
    assert coord._pending["spot"] is None

    coord.begin_episode()
    coord.set_training_phase(freeze_midlong=True, freeze_spot=False)
    _episode_via_coordinator(env, coord)
    assert len(coord.midlong_agent.replay_buffer) == MIDLONG_STEPS, "冻结 midlong 后不应新增"
    assert len(coord.spot_agent.replay_buffer) == SPOT_STEPS
    print("  [PASS] 冻结角色零写入, 解冻后按自己的时序正常写入")


def _fill(agent, n, rng, state_dim=13):
    for _ in range(n):
        s = rng.normal(size=state_dim).astype(np.float32)
        a = rng.normal(size=1).astype(np.float32)
        agent.store_transition(s, a, float(rng.normal()), s.copy(), False)


def test_reload_reproduces_actions_and_keeps_temperature_trainable():
    """保存/加载: 动作逐位一致, 温度值一致, 且温度优化器仍绑定同一张量、续训会更新温度。"""
    torch.manual_seed(1234)
    rng = np.random.default_rng(4)
    batch = DRL_PARAMS["batch_size"]

    src = SACAgent(state_dim=13, action_dim=1, action_scale=1.0)
    _fill(src, batch + 8, rng)
    src.update()
    src.update()

    states = rng.normal(size=(64, 13)).astype(np.float32)
    actions_before = np.array([src.select_action(s, deterministic=True) for s in states])
    alpha_before = src.alpha

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sac.pt")
        src.save(path)

        dst = SACAgent(state_dim=13, action_dim=1, action_scale=1.0)
        dst.load(path)
        actions_after = np.array([dst.select_action(s, deterministic=True) for s in states])
        assert np.array_equal(actions_before, actions_after), "重载后确定性动作必须逐位一致"
        assert abs(dst.alpha - alpha_before) < 1e-12
        assert abs(float(dst.log_alpha.item()) - float(src.log_alpha.item())) < 1e-12

        # 关键回归点: alpha_optim 必须仍然绑定当前 log_alpha 张量
        params = list(dst.alpha_optim.param_groups[0]["params"])
        assert len(params) == 1 and params[0] is dst.log_alpha, \
            "温度优化器与 log_alpha 脱钩 (加载后温度将无法更新)"

        _fill(dst, batch + 8, rng)
        before = float(dst.log_alpha.item())
        for _ in range(20):
            dst.update()
        after = float(dst.log_alpha.item())
        assert after != before, "重载后温度应继续学习, 实际保持不变 (说明优化器脱钩)"
        assert abs(dst.alpha - np.exp(after)) < 1e-9
        print(f"  [PASS] 重载后动作逐位一致 (64 状态), 温度 {np.exp(before):.4f} -> "
              f"{np.exp(after):.4f} 仍可学习")


def test_reload_rejects_mismatched_hyperparameters():
    """动作尺度/网络维度不同的检查点必须拒绝加载 (避免评价时用错尺度的动作)。"""
    torch.manual_seed(4321)
    rng = np.random.default_rng(5)
    src = SACAgent(state_dim=13, action_dim=1, action_scale=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sac.pt")
        src.save(path)

        wrong_scale = SACAgent(state_dim=13, action_dim=1, action_scale=5.0)
        try:
            wrong_scale.load(path)
        except ValueError as exc:
            assert "action_scale" in str(exc), str(exc)
        else:
            raise AssertionError("动作尺度不符的检查点本应被拒绝")

        wrong_dim = SACAgent(state_dim=11, action_dim=1, action_scale=1.0)
        try:
            wrong_dim.load(path)
        except ValueError as exc:
            assert "state_dim" in str(exc), str(exc)
        else:
            raise AssertionError("状态维度不符的检查点本应被拒绝")
    print("  [PASS] action_scale / state_dim 不匹配的检查点被拒绝加载")


def test_coordinator_save_load_roundtrip_for_both_roles():
    """协调器整体保存/加载: 两个角色的动作都与重载前一致。"""
    torch.manual_seed(99)
    rng = np.random.default_rng(6)
    env = TradingEnvironment(seed=35)
    coord = MultiScaleCoordinator(state_dim=13, use_dual_agent=True)
    _episode_via_coordinator(env, coord)

    states = rng.normal(size=(32, 13)).astype(np.float32)
    before = {stage: np.array([coord.select_action(s, stage, deterministic=True)
                               for s in states]) for stage in (0, 3)}

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "dual")
        coord.save(prefix)
        reloaded = MultiScaleCoordinator(state_dim=13, use_dual_agent=True)
        reloaded.load(prefix)
        for stage, acts in before.items():
            after = np.array([reloaded.select_action(s, stage, deterministic=True)
                              for s in states])
            assert np.array_equal(acts, after), stage
    print("  [PASS] 协调器双角色重载: 中长期/柔性调度动作在 32 个状态上逐位一致")


if __name__ == "__main__":
    tests = [
        test_dual_transitions_follow_each_roles_own_decision_stream,
        test_naive_next_state_would_have_been_wrong,
        test_no_experience_leaks_across_episodes,
        test_frozen_role_stores_nothing,
        test_reload_reproduces_actions_and_keeps_temperature_trainable,
        test_reload_rejects_mismatched_hyperparameters,
        test_coordinator_save_load_roundtrip_for_both_roles,
    ]
    print("=== 决策时序与模型重载验收 (#36) ===")
    for fn in tests:
        print(f"--- {fn.__name__} ---")
        fn()
    print(f"\n通过 {len(tests)}/{len(tests)}")
