"""
逐回合单位奖励量级测量 (报告 5.6 / 完成度评估 §4 引用该脚本的输出)

奖励已按各步决策的电量基准归一化 (年度步 Q, 其余 R_m; 现货步再除以实际暴露电量),
所以 |r_spot|/|r_mid| 反映的是"两个角色收到的学习信号是否同量级", 不是性能改进。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.rule_policies import fixed_ratio_action, rolling_rule_action
from src.market.environment import TradingEnvironment
from src.utils.config import PRICE_SCENARIOS


def _roll(scenario, rule, seed=42, world=0):
    env = TradingEnvironment(scenario_name=scenario, include_flexible=True, seed=seed)
    obs, _ = env.reset(options={"world": world})
    mid, spot = [], []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(rule(env.current_stage, obs))
        done = terminated or truncated
        if info.get("stage") == "spot":
            spot.append(abs(info["reward_spot"]))
        elif reward != 0.0:
            mid.append(abs(info["reward_midlong"]))
    return float(np.mean(mid)), float(np.mean(spot))


if __name__ == "__main__":
    ratios = []
    for scenario in PRICE_SCENARIOS:
        for name, rule in (("fixed_rule", fixed_ratio_action),
                           ("rolling_rule", rolling_rule_action)):
            m, s = _roll(scenario, rule)
            ratios.append(s / m)
            print(f"{scenario[:2]} {name:<12} |r_mid|={m:7.2f} |r_spot|={s:7.2f} "
                  f"ratio={s / m:5.2f}")
    print(f"区间 {min(ratios):.2f}~{max(ratios):.2f}, 均值 {np.mean(ratios):.2f}")
