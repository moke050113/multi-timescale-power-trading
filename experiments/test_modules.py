"""
快速验证脚本
测试各模块是否正常工作
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_price_generator():
    """测试价格场景生成"""
    print("=" * 50)
    print("1. 测试价格场景生成器")
    print("=" * 50)
    from src.market.market_models import PriceScenarioGenerator

    for name in PriceScenarioGenerator.list_scenarios():
        gen = PriceScenarioGenerator(name, seed=42)
        prices = gen.generate()
        print(f"\n{gen.get_scenario_info()}")
        print(f"  年度价格: {prices['annual_price']:.0f}")
        print(f"  月度价格: {prices['monthly_prices'].round(1)}")
        print(f"  现货均价: {prices['spot_prices'].mean():.1f} ± {prices['spot_prices'].std():.1f}")
    print("\n[PASS] 价格场景生成器正常")


def test_settlement():
    """测试结算引擎"""
    print("\n" + "=" * 50)
    print("2. 测试结算引擎")
    print("=" * 50)
    from src.market.settlement import SettlementEngine

    se = SettlementEngine(total_demand=1000)

    cost_y = se.settle_annual(0.8, 300)
    cost_m = se.settle_monthly(0.9, 350, 0)
    cost_d = se.settle_intraday(0.95, 380, 0)
    spot_prices = np.full(24, 400.0)
    cost_s = se.settle_spot(spot_prices, 0)

    print(f"  年度成本: {cost_y:.0f} 元 (α={se.alpha})")
    print(f"  月度成本: {cost_m:.0f} 元 (β={se.beta})")
    print(f"  月内成本: {cost_d:.0f} 元 (γ={se.gamma})")
    print(f"  现货成本: {cost_s:.0f} 元 (暴露={1-se.gamma:.2f})")
    print(f"  总成本:   {se.get_total_cost():.0f} 元")
    print(f"  仓位: {se.get_positions()}")
    print("\n[PASS] 结算引擎正常")


def test_flexible_load():
    """测试柔性负荷模型"""
    print("\n" + "=" * 50)
    print("3. 测试柔性负荷模型")
    print("=" * 50)
    from src.load.load_scenarios import LoadScenarioBuilder

    for name in LoadScenarioBuilder.list_scenarios():
        builder = LoadScenarioBuilder(name)
        print(f"\n{builder.get_summary()}")
    print("\n[PASS] 柔性负荷模型正常")


def test_environment():
    """测试交易环境"""
    print("\n" + "=" * 50)
    print("4. 测试交易环境 (Gym)")
    print("=" * 50)
    from src.market.environment import TradingEnvironment

    env = TradingEnvironment(scenario_name="A_年度低价后期上涨", seed=42)
    obs, info = env.reset()
    print(f"  初始状态维度: {obs.shape}")
    print(f"  初始状态: {obs}")
    print(f"  动作空间: {env.action_space}")

    # 随机走几步
    total_reward = 0.0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    print(f"  随机策略: {steps} 步, 总奖励={total_reward:.1f}, 总成本={info['total_cost']:.0f}")

    # 固定比例基准
    fixed = env.get_fixed_ratio_cost()
    print(f"  固定比例基准成本: {fixed['total_cost']:.0f} 元")
    print(f"  成本明细: {fixed['breakdown']}")
    print("\n[PASS] 交易环境正常")


def test_agent():
    """测试 SAC 智能体"""
    print("\n" + "=" * 50)
    print("5. 测试 SAC 智能体")
    print("=" * 50)
    from src.agent.sac_agent import SACAgent

    agent = SACAgent(state_dim=13, action_dim=2)
    state = np.random.randn(13).astype(np.float32)

    action = agent.select_action(state)
    print(f"  状态: {state.shape} → 动作: {action}")

    # 存入一些经验并更新
    for _ in range(300):
        s = np.random.randn(13).astype(np.float32)
        a = np.random.randn(2).astype(np.float32) * 0.1
        r = -100.0
        ns = np.random.randn(13).astype(np.float32)
        agent.store_transition(s, a, r, ns, False)

    metrics = agent.update()
    print(f"  更新指标: {metrics}")
    print("\n[PASS] SAC 智能体正常")


def test_short_training():
    """快速训练测试"""
    print("\n" + "=" * 50)
    print("6. 快速训练测试 (50 episodes)")
    print("=" * 50)
    from src.market.environment import TradingEnvironment
    from src.agent.multi_agent import MultiScaleCoordinator

    env = TradingEnvironment(scenario_name="A_年度低价后期上涨", seed=42)
    obs, _ = env.reset()

    coordinator = MultiScaleCoordinator(
        state_dim=env.observation_space.shape[0],
        use_dual_agent=True,
    )

    for ep in range(50):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            stage_before = env.current_stage
            action = coordinator.select_action(state, stage_before)
            next_state, reward, terminated, truncated, info = env.step(action)
            # 按角色路由奖励: 各智能体只接收自己负责阶段的信号
            coordinator.store_transition(state, action, info, next_state, terminated, stage_before)
            coordinator.update(stage=stage_before)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d} | 奖励: {total_reward:10.1f} | 成本: {info['total_cost']:10.0f}")

    print("\n[PASS] 快速训练测试通过")


if __name__ == "__main__":
    test_price_generator()
    test_settlement()
    test_flexible_load()
    test_environment()
    test_agent()
    test_short_training()
    print("\n" + "=" * 50)
    print("全部测试通过!")
    print("=" * 50)
