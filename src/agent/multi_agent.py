"""
多时间尺度智能体协调器
实现中长期交易智能体与现货交易智能体的信息传递与决策衔接
"""
import numpy as np
from typing import Dict, List, Tuple
from .sac_agent import SACAgent
from ..utils.config import DRL_PARAMS, SIM_PARAMS


class MultiScaleCoordinator:
    """
    多时间尺度协调器

    架构:
        ┌─────────────────────────────────────────┐
        │         中长期仓位规划智能体              │
        │  (决定 α, β, γ — 粗粒度比例决策)        │
        └──────────────┬──────────────────────────┘
                       │ 仓位信息 ↓
        ┌──────────────▼──────────────────────────┐
        │         现货实时调整智能体                │
        │  (决定柔性负荷调度 — 细粒度实时调整)     │
        └─────────────────────────────────────────┘

    协调规则:
        1. 前期已成交仓位不可推翻
        2. 后续智能体在剩余空间内滚动优化
        3. 中长期智能体输出比例约束给现货智能体
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        use_dual_agent: bool = True,
    ):
        self.use_dual_agent = use_dual_agent
        self.state_dim = state_dim

        if use_dual_agent:
            # 中长期智能体: 负责比例决策, 动作尺度5.0 (经sigmoid后为[0,1])
            self.midlong_agent = SACAgent(
                state_dim=state_dim,
                action_dim=1,
                action_scale=5.0,
            )
            # 现货智能体: 负责柔性负荷调度, 动作尺度5.0
            self.spot_agent = SACAgent(
                state_dim=state_dim,
                action_dim=1,
                action_scale=5.0,
            )
        else:
            # 单智能体模式
            self.single_agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_scale=5.0,
            )

        # 仓位记录
        self.position_history = []

    def select_action(self, state: np.ndarray, stage: int,
                      deterministic: bool = False) -> np.ndarray:
        """
        根据当前阶段选择动作

        Args:
            state: 状态向量
            stage: 当前交易阶段 (0=年度, 1=月度, 2=月内, 3=现货)
            deterministic: 是否确定性策略

        Returns:
            action: [ratio_action, flexible_action]
        """
        if self.use_dual_agent:
            if stage in [0, 1, 2]:  # 年度/月度/月内
                ratio_action = self.midlong_agent.select_action(state, deterministic)
                return np.array([ratio_action[0], 0.0], dtype=np.float32)
            else:  # 现货
                flexible_action = self.spot_agent.select_action(state, deterministic)
                return np.array([0.0, flexible_action[0]], dtype=np.float32)
        else:
            return self.single_agent.select_action(state, deterministic)

    def store_transition(self, state, action, reward, next_state, done, stage: int):
        """存储经验到对应的智能体"""
        if self.use_dual_agent:
            if stage in [0, 1, 2]:
                self.midlong_agent.store_transition(
                    state, action[:1], reward, next_state, done
                )
            else:
                self.spot_agent.store_transition(
                    state, action[1:], reward, next_state, done
                )
        else:
            self.single_agent.store_transition(state, action, reward, next_state, done)

    def update(self, stage: int = None) -> Dict:
        """更新智能体参数"""
        if self.use_dual_agent:
            if stage is not None:
                if stage in [0, 1, 2]:
                    return self.midlong_agent.update()
                else:
                    return self.spot_agent.update()
            else:
                m = self.midlong_agent.update()
                s = self.spot_agent.update()
                return {**{f"midlong_{k}": v for k, v in m.items()},
                        **{f"spot_{k}": v for k, v in s.items()}}
        else:
            return self.single_agent.update()

    def record_position(self, positions: Dict, step: int):
        """记录仓位历史"""
        self.position_history.append({**positions, "step": step})

    def save(self, prefix: str):
        """保存所有智能体"""
        if self.use_dual_agent:
            self.midlong_agent.save(f"{prefix}_midlong.pt")
            self.spot_agent.save(f"{prefix}_spot.pt")
        else:
            self.single_agent.save(f"{prefix}_single.pt")

    def load(self, prefix: str):
        """加载所有智能体"""
        if self.use_dual_agent:
            self.midlong_agent.load(f"{prefix}_midlong.pt")
            self.spot_agent.load(f"{prefix}_spot.pt")
        else:
            self.single_agent.load(f"{prefix}_single.pt")
