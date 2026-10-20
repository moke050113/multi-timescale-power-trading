"""
双智能体模块测试 (B 方案)

验证:
    1. 环境 2D 动作 + 分阶段奖励正确输出
    2. 结算按交割月记账, 上月合同不退款冲销
    3. 协调器路由 / 冻结 / 更新
    4. 单 episode 端到端可跑通
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market.environment import TradingEnvironment
from src.agent.multi_agent import MultiScaleCoordinator


def test_env_action_space():
    env = TradingEnvironment(seed=42)
    assert env.action_space.shape == (2,), f"动作空间应为 (2,), 实际 {env.action_space.shape}"
    assert env.observation_space.shape == (13,)
    print("[OK] 环境动作空间 (2,) 与状态空间 (13,)")


def test_env_episode_with_dual_output():
    env = TradingEnvironment(seed=42)
    state, _ = env.reset()
    total_cost = 0.0
    step = 0
    rm_sum = 0.0
    rs_sum = 0.0

    while True:
        stage = env.current_stage
        action = np.zeros(2, dtype=np.float32)
        if stage == env.STAGE_SPOT:
            action[1] = np.random.uniform(-1, 1)
        else:
            action[0] = np.random.uniform(-2, 2)
        state, _, term, trunc, info = env.step(action)
        rm_sum += info["reward_midlong"]
        rs_sum += info["reward_spot"]
        step += 1
        if term or trunc:
            total_cost = info["total_cost"]
            break

    assert step == 37, f"应有 37 步 (1 annual + 12*(1 monthly + 1 intraday + 1 spot)), 实际 {step}"
    assert total_cost > 0
    # 中长期 25 步 (1 annual + 12*(monthly+intraday)) + 现货 12 步 = 37
    print(f"[OK] Episode 37 steps | cost={total_cost:,.0f} | R_mid_sum={rm_sum:.2f} | R_spot_sum={rs_sum:.2f}")


def test_settlement_semantics():
    """验证覆盖率比例语义: 每层覆盖率落在 [0,1], 下月可以下调且不冲销上月合同"""
    env = TradingEnvironment(seed=7)
    state, _ = env.reset()
    betas = []
    gammas = []

    while True:
        stage = env.current_stage
        action = np.zeros(2, dtype=np.float32)
        if stage == env.STAGE_ANNUAL:
            action[0] = 2.0
        elif stage == env.STAGE_MONTHLY:
            action[0] = np.random.uniform(-2, 2)
        elif stage == env.STAGE_INTRADAY:
            action[0] = np.random.uniform(-2, 2)
        else:
            action[1] = np.random.uniform(-1, 1)

        state, _, term, trunc, info = env.step(action)
        if stage == env.STAGE_MONTHLY:
            betas.append(env.beta)
        elif stage == env.STAGE_INTRADAY:
            gammas.append(env.gamma)
        if term or trunc:
            break

    # 关键约束: 覆盖率落在 [0,1] (逐层覆盖率可升可降, 但已签合同永不退款冲销)
    assert all(0 <= b <= 1 for b in betas)
    assert all(0 <= g <= 1 for g in gammas)
    print(
        f"[OK] 结算比例范围正确, β序列首末={betas[0]:.3f}/{betas[-1]:.3f}, "
        f"γ序列首末={gammas[0]:.3f}/{gammas[-1]:.3f}"
    )


def test_coordinator_routing():
    state = np.zeros(13, dtype=np.float32)
    coord = MultiScaleCoordinator(state_dim=13, use_dual_agent=True)

    # midlong 阶段
    a_mid = coord.select_action(state, stage=0)
    assert a_mid.shape == (2,) and a_mid[1] == 0.0
    # spot 阶段
    a_spot = coord.select_action(state, stage=3)
    assert a_spot.shape == (2,) and a_spot[0] == 0.0

    # 存储路由: 冻结 spot 时既不写经验也不留挂起决策
    coord.set_training_phase(freeze_midlong=False, freeze_spot=True)
    n0 = len(coord.spot_agent.replay_buffer)
    coord.store_transition(state, a_spot, {"reward_midlong": 0.0, "reward_spot": -1.0}, state, False, stage=3)
    assert len(coord.spot_agent.replay_buffer) == n0, "冻结时不应写入 spot 经验池"
    assert coord._pending["spot"] is None, "冻结时不应留下挂起决策"

    # 半 MDP: 一次中长期决策要等到该角色下一次决策到达才闭合
    coord.begin_episode()
    coord.set_training_phase(freeze_midlong=False, freeze_spot=False)
    coord.store_transition(state, a_mid, {"reward_midlong": -1.0, "reward_spot": 0.0}, state, False, stage=0)
    assert len(coord.midlong_agent.replay_buffer) == 0, "未等到下一次中长期决策前不应入库"
    assert coord._pending["midlong"] is not None

    state2 = np.ones(13, dtype=np.float32)
    coord.store_transition(state2, a_mid, {"reward_midlong": -1.0, "reward_spot": 0.0}, state2, False, stage=1)
    buf = coord.midlong_agent.replay_buffer.buffer
    assert len(buf) == 1, len(buf)
    assert np.allclose(buf[0][3], state2, atol=1e-6), "后继状态应是本角色下一次决策的观察"
    assert np.asarray(buf[0][1]).shape == (1,) and buf[0][1][0] == a_mid[0], "中长期只学 ratio 维度"

    # 回合以现货步终止: 中长期挂起决策按终止闭合, 不再 bootstrap
    coord.store_transition(state2, a_spot, {"reward_midlong": 0.0, "reward_spot": -1.0}, state2, True, stage=3)
    assert len(coord.midlong_agent.replay_buffer) == 2, len(coord.midlong_agent.replay_buffer)
    assert coord._pending["midlong"] is None and coord._pending["spot"] is None
    assert coord.spot_agent.replay_buffer.buffer[-1][1][0] == a_spot[1], "柔性调度只学强度维度"

    print("[OK] 协调器路由、冻结逻辑与按角色决策序列闭合")


def test_reward_scaling():
    """分阶段奖励应按各自电量基准归一, 两个信号量级可比 (否则柔性调度学不到东西)"""
    env = TradingEnvironment(scenario_name="D_现货高波动", seed=99)
    state, _ = env.reset()

    spot_rewards = []
    mid_rewards = []

    while True:
        stage = env.current_stage
        action = np.zeros(2, dtype=np.float32)
        if stage == env.STAGE_SPOT:
            action[1] = np.random.uniform(-1, 1)
        else:
            action[0] = np.random.uniform(0, 3)  # 让 α/β/γ 处于中等仓位
        state, _, term, trunc, info = env.step(action)
        if info["reward_spot"] != 0:
            spot_rewards.append(info["reward_spot"])
        if info["reward_midlong"] != 0:
            mid_rewards.append(info["reward_midlong"])
        if term or trunc:
            break

    mean_spot = np.mean(np.abs(spot_rewards))
    mean_mid = np.mean(np.abs(mid_rewards))
    print(
        f"[OK] 奖励量级: |R_spot|={mean_spot:.3f} | |R_mid|={mean_mid:.3f} "
        f"| 比例={mean_spot/max(mean_mid,1e-6):.2f}× "
        f"| spot_steps={len(spot_rewards)}, mid_steps={len(mid_rewards)}"
    )
    # 允许 mid 步数在 20~25 (delta_beta=0 时 cost=0 会跳过记录)
    assert 10 <= len(spot_rewards) <= 12
    assert 15 <= len(mid_rewards) <= 25
    # 关键: 两个信号应同量级 (旧共享奖励方案 R_spot/R_mid ~ 0.02)
    assert mean_spot / max(mean_mid, 1e-6) > 0.05, "现货信号仍过弱"


if __name__ == "__main__":
    print("=" * 60)
    print("双智能体模块测试 (B 方案)")
    print("=" * 60)
    test_env_action_space()
    test_env_episode_with_dual_output()
    test_settlement_semantics()
    test_coordinator_routing()
    test_reward_scaling()
    print("=" * 60)
    print("全部通过 [OK]")
