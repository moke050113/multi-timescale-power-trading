"""
多时间尺度智能体协调器 (v2 — 双 DRL 智能体, B 方案)

架构:
    ┌──────────────────────────────────────────┐
    │  midlong_agent (SAC, action_scale=5.0)   │
    │  1D action: ratio_raw → α/β/γ             │
    │  奖励: -contract_cost / total_demand     │
    │  经验池: 独立 (annual+monthly+intraday)  │
    └──────────────────┬───────────────────────┘
                       │ γ 决定现货暴露 1-γ
    ┌──────────────────▼───────────────────────┐
    │  spot_agent (SAC, action_scale=1.0)      │
    │  1D action: flexible_action ∈ [-1, 1]    │
    │  奖励: -(spot_cost+penalty)              │
    │        / (total_demand × max(1-γ, 0.05)) │
    │  经验池: 独立 (spot 阶段)                 │
    └──────────────────────────────────────────┘

关键设计:
    1. 动作解耦: 每步只由对应阶段的智能体输出, 另一维填 0
    2. 独立经验池: 避免分布混合
    3. 冻结标志: 支持三阶段交替训练 (phase1 只训 midlong, phase2 只训 spot, phase3 联合)
    4. 奖励缩放: spot 奖励除以 spot_exposure 放大信号 (缓解信用分配)
"""
import numpy as np
from typing import Dict, Optional, Union

from .sac_agent import SACAgent


STAGE_MIDLONG = {"annual", "monthly", "intraday", 0, 1, 2}
STAGE_SPOT = {"spot", 3}


def _is_spot_stage(stage: Union[int, str]) -> bool:
    return stage in STAGE_SPOT


def _is_midlong_stage(stage: Union[int, str]) -> bool:
    return stage in STAGE_MIDLONG


class MultiScaleCoordinator:
    """双 SAC 智能体协调器 (支持冻结)"""

    def __init__(
        self,
        state_dim: int = 13,
        use_dual_agent: bool = True,
        midlong_action_scale: float = 5.0,
        spot_action_scale: float = 1.0,
        single_action_dim: int = 2,
        single_action_scale: float = 5.0,
    ):
        self.use_dual_agent = use_dual_agent
        self.state_dim = state_dim

        if use_dual_agent:
            self.midlong_agent = SACAgent(
                state_dim=state_dim,
                action_dim=1,
                action_scale=midlong_action_scale,
            )
            self.spot_agent = SACAgent(
                state_dim=state_dim,
                action_dim=1,
                action_scale=spot_action_scale,
            )
        else:
            self.single_agent = SACAgent(
                state_dim=state_dim,
                action_dim=single_action_dim,
                action_scale=single_action_scale,
            )

        # 冻结标志 (三阶段交替训练用)
        self.frozen_midlong: bool = False
        self.frozen_spot: bool = False

        self.position_history = []

    # ------------------------------------------------------------
    def set_training_phase(
        self,
        freeze_midlong: bool = False,
        freeze_spot: bool = False,
    ):
        """设置训练阶段: 通过冻结标志控制哪个智能体接受梯度更新"""
        self.frozen_midlong = freeze_midlong
        self.frozen_spot = freeze_spot

    # ------------------------------------------------------------
    def select_action(
        self,
        state: np.ndarray,
        stage: Union[int, str],
        deterministic: bool = False,
    ) -> np.ndarray:
        """根据当前阶段返回完整 2 维动作: [ratio_raw, flexible_action]"""
        if self.use_dual_agent:
            if _is_spot_stage(stage):
                flex = self.spot_agent.select_action(state, deterministic)
                return np.array([0.0, float(flex[0])], dtype=np.float32)
            else:
                ratio = self.midlong_agent.select_action(state, deterministic)
                return np.array([float(ratio[0]), 0.0], dtype=np.float32)
        else:
            return self.single_agent.select_action(state, deterministic)

    # ------------------------------------------------------------
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        rewards: Union[Dict[str, float], float],
        next_state: np.ndarray,
        done: bool,
        stage: Union[int, str],
    ):
        """
        将经验路由到对应智能体

        Args:
            rewards: dict 需含 'reward_midlong', 'reward_spot'; 或直接 float (单智能体)
        """
        if not self.use_dual_agent:
            r = rewards if isinstance(rewards, (int, float)) else rewards.get("reward_global", 0.0)
            self.single_agent.store_transition(state, action, r, next_state, done)
            return

        if _is_spot_stage(stage):
            if self.frozen_spot:
                return
            r = rewards.get("reward_spot", 0.0) if isinstance(rewards, dict) else float(rewards)
            self.spot_agent.store_transition(
                state, np.array([action[1]], dtype=np.float32), r, next_state, done
            )
        else:
            if self.frozen_midlong:
                return
            r = rewards.get("reward_midlong", 0.0) if isinstance(rewards, dict) else float(rewards)
            self.midlong_agent.store_transition(
                state, np.array([action[0]], dtype=np.float32), r, next_state, done
            )

    # ------------------------------------------------------------
    def update(self, stage: Optional[Union[int, str]] = None) -> Dict:
        """
        更新对应智能体 (仅当未冻结)

        Args:
            stage: 若指定则只更新该阶段所属的智能体; None 表示按当前冻结状态更新所有活跃智能体
        """
        if not self.use_dual_agent:
            return self.single_agent.update()

        metrics = {}

        if stage is None:
            # 遍历两个智能体
            if not self.frozen_midlong:
                m = self.midlong_agent.update()
                metrics.update({f"midlong_{k}": v for k, v in m.items()})
            if not self.frozen_spot:
                s = self.spot_agent.update()
                metrics.update({f"spot_{k}": v for k, v in s.items()})
            return metrics

        if _is_spot_stage(stage):
            if self.frozen_spot:
                return {}
            s = self.spot_agent.update()
            return {f"spot_{k}": v for k, v in s.items()}
        else:
            if self.frozen_midlong:
                return {}
            m = self.midlong_agent.update()
            return {f"midlong_{k}": v for k, v in m.items()}

    # ------------------------------------------------------------
    def record_position(self, positions: Dict, step: int):
        self.position_history.append({**positions, "step": step})

    def save(self, prefix: str):
        if self.use_dual_agent:
            self.midlong_agent.save(f"{prefix}_midlong.pt")
            self.spot_agent.save(f"{prefix}_spot.pt")
        else:
            self.single_agent.save(f"{prefix}_single.pt")

    def load(self, prefix: str):
        if self.use_dual_agent:
            self.midlong_agent.load(f"{prefix}_midlong.pt")
            self.spot_agent.load(f"{prefix}_spot.pt")
        else:
            self.single_agent.load(f"{prefix}_single.pt")

    def get_agent_sizes(self) -> Dict[str, int]:
        """返回各智能体经验池大小 (监控用)"""
        if self.use_dual_agent:
            return {
                "midlong_buffer": len(self.midlong_agent.replay_buffer),
                "spot_buffer": len(self.spot_agent.replay_buffer),
            }
        return {"single_buffer": len(self.single_agent.replay_buffer)}
