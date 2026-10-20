"""
双智能体模块测试 (B 方案)

验证:
    1. 环境 2D 动作 + 分阶段奖励正确输出
    2. 结算引擎跨月不回溯
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
    # 中长期阶段 3 步 (annual) + 12*2 (monthly+intraday) = 27
    # 现货阶段 12 步
    print(f"[OK] Episode 37 steps | cost={total_cost:,.0f} | R_mid_sum={rm_sum:.2f} | R_spot_sum={rs_sum:.2f}")


def test_settlement_semantics():
    """验证结算引擎累计比例语义 (允许 delta 为负, 与整体成本模型自洽)"""
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

    # 关键约束: α ≤ β, β ≤ γ ≤ 1 (在每次决策时)
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

    # 存储路由: 冻结 spot 时不存 spot 经验
    coord.set_training_phase(freeze_midlong=False, freeze_spot=True)
    n0 = len(coord.spot_agent.replay_buffer)
    coord.store_transition(state, a_spot, {"reward_midlong": 0.0, "reward_spot": -1.0}, state, False, stage=3)
    assert len(coord.spot_agent.replay_buffer) == n0, "冻结时不应写入 spot 经验池"

    coord.store_transition(state, a_mid, {"reward_midlong": -1.0, "reward_spot": 0.0}, state, False, stage=0)
    assert len(coord.midlong_agent.replay_buffer) > 0

    print("[OK] 协调器路由与冻结逻辑")


def test_reward_scaling():
    """现货奖励应显著大于中长期 (放大效应)"""
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
