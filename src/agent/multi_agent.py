"""
多时间尺度智能体协调器 (v3 — 双 DRL 智能体, 半 MDP 决策衔接)

架构:
    ┌──────────────────────────────────────────┐
    │  midlong_agent (SAC, action_scale=5.0)   │
    │  1D action: ratio_raw → α/β/γ             │
    │  奖励: -contract_cost / total_demand     │
    │  经验池: 独立 (annual+monthly+intraday)  │
    └──────────────────┬───────────────────────┘
                       │ γ 决定现货暴露 1-γ
    ┌──────────────────▼───────────────────────┐
    │  spot_agent = 柔性调度 SAC (scale 1.0)   │
    │  1D action: flexible_action ∈ [-1, 1]    │
    │  只决定"多大程度择优调度", 不决定现货采购量 │
    │  奖励: -(spot_cost+penalty)/实际偏差电量  │
    │  经验池: 独立 (spot 阶段)                 │
    └──────────────────────────────────────────┘

关键设计:
    1. 动作解耦: 每步只由对应阶段的智能体输出, 另一维填 0
    2. 独立经验池: 避免分布混合
    3. 冻结标志: 支持三阶段交替训练 (phase1 只训 midlong, phase2 只训 spot, phase3 联合)
    4. **半 MDP 决策衔接**: 每个智能体的 next_state 必须是"自己下一次决策"的观察,
       不是环境下一步的观察。用时间戳记录该角色两次决策之间的环境步数 dt,
       经验里存 gamma**dt; 回合结束时仍未等到下一次决策的经验按终止处理 (不再 bootstrap)。
       (旧实现把环境下一步直接当 next_state: 月内步的后继是现货决策的观察,
        现货步的后继是下月月度决策的观察, 目标值与状态分布都错位。)
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


def _role_of(stage: Union[int, str]) -> str:
    return "spot" if _is_spot_stage(stage) else "midlong"


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

        # 半 MDP 决策衔接: 每个角色待闭合的上一次决策 (state/action/reward/时间戳)
        self._pending: Dict[str, Optional[Dict]] = {"midlong": None, "spot": None}
        self._t = 0

        self.position_history = []

    # ------------------------------------------------------------
    def begin_episode(self):
        """回合开始: 清空待闭合经验与步计数 (跨回合的经验必须相接, 不能沿用上一回合)。"""
        self._pending = {"midlong": None, "spot": None}
        self._t = 0

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
        按"角色自己的决策序列"写入经验 (双智能体路径)。

        调用时机: 环境执行完一步之后 (state = 该步决策的观察, next_state = 下一步观察)。
        本次决策属于角色 r; 观察到 r 的下一次决策时, 才把上一次决策的 next_state
        定为**本次的 state**, 折扣取 gamma**(两次决策间的环境步数)。
        回合结束 (done) 时, 本次决策直接按终止写入, 另一角色仍未闭合的决策同样按终止处理
        (即使最后一步属于被冻结角色 —— 否则该角色每回合少一条经验)。

        Args:
            rewards: dict 需含 'reward_midlong', 'reward_spot'; 或直接 float (单智能体)
        """
        if not self.use_dual_agent:
            r = rewards if isinstance(rewards, (int, float)) else rewards.get("reward_global", 0.0)
            self.single_agent.store_transition(state, action, r, next_state, done)
            return

        self._t += 1
        role = _role_of(stage)
        frozen = {"midlong": self.frozen_midlong, "spot": self.frozen_spot}
        agent = {"midlong": self.midlong_agent, "spot": self.spot_agent}
        key = {"midlong": "reward_midlong", "spot": "reward_spot"}

        if frozen[role]:
            # 本角色不学习, 但回合结束时另一角色挂起的决策仍必须闭合
            if done:
                self._flush_pending(next_state, frozen, agent)
            return
        own = float(rewards.get(key[role], 0.0)) if isinstance(rewards, dict) else float(rewards)
        # 每个角色的 SAC 只有 1 维动作: midlong 取 action[0], 柔性调度取 action[1]
        own_action = np.array([action[1] if role == "spot" else action[0]], dtype=np.float32)

        # 1) 闭合本角色上一次决策: 它的后继状态就是本次决策的观察
        pend = self._pending[role]
        if pend is not None:
            agent[role].store_transition(
                pend["state"], pend["action"], pend["reward"], state, False,
                discount=agent[role].gamma ** (self._t - pend["t"]),
            )
            self._pending[role] = None

        # 2) 本次决策: 终止则立即入库, 否则挂起等待本角色下一次决策
        if done:
            agent[role].store_transition(state, own_action, own, next_state, True)
        else:
            self._pending[role] = {"state": state, "action": own_action,
                                   "reward": own, "t": self._t}

        # 3) 回合结束: 没等到下一次决策的挂起经验按终止处理
        if done:
            self._flush_pending(next_state, frozen, agent)

    def _flush_pending(self, next_state, frozen, agent):
        """回合结束时把两个角色仍未闭合的挂起决策按终止写入 (不再 bootstrap)。"""
        for waiting_role in ("midlong", "spot"):
            waiting = self._pending[waiting_role]
            if waiting is not None and not frozen[waiting_role]:
                agent[waiting_role].store_transition(
                    waiting["state"], waiting["action"], waiting["reward"],
                    next_state, True,
                    discount=agent[waiting_role].gamma ** (self._t - waiting["t"] + 1),
                )
            self._pending[waiting_role] = None

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
