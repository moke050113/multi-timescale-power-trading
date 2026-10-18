"""
SAC (Soft Actor-Critic) 智能体
用于多时间尺度电力交易的深度强化学习算法
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Dict
from collections import deque
import random

from ..utils.config import DRL_PARAMS


# ============================================================
# 网络结构
# ============================================================

class QNetwork(nn.Module):
    """双 Q 网络 (Twin Critic)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_l1(x))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        q2 = F.relu(self.q2_l1(x))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return q1, q2


class GaussianPolicy(nn.Module):
    """高斯策略网络 (Actor)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作及其对数概率"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()  # 重参数化技巧
        action = torch.tanh(x_t)  # 映射到 [-1, 1]

        # 对数概率修正 (tanh 变换的 Jacobian)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


# ============================================================
# 经验回放池
# ============================================================

class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# SAC 智能体
# ============================================================

class SACAgent:
    """
    SAC (Soft Actor-Critic) 智能体

    特点:
        - 最大熵框架: 鼓励探索, 提升策略鲁棒性
        - 连续动作空间: 适合交易比例决策
        - 自动温度调节: 平衡探索与利用
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = DRL_PARAMS["hidden_dim"],
        lr_actor: float = DRL_PARAMS["lr_actor"],
        lr_critic: float = DRL_PARAMS["lr_critic"],
        lr_alpha: float = DRL_PARAMS["lr_alpha"],
        gamma: float = DRL_PARAMS["gamma"],
        tau: float = DRL_PARAMS["tau"],
        buffer_size: int = DRL_PARAMS["buffer_size"],
        batch_size: int = DRL_PARAMS["batch_size"],
        init_alpha: float = 0.2,
        action_scale: float = 5.0,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_scale = action_scale

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 网络
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)

        # 硬拷贝目标网络
        self._hard_update(self.critic_target, self.critic)

        # 优化器
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # 自动温度调节
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练记录
        self.training_log = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        选择动作 (输出经 action_scale 缩放)

        Args:
            state: 状态向量
            deterministic: 是否使用确定性策略 (评估时用 True)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean).cpu().numpy().flatten()
            else:
                action, _ = self.actor.sample(state_t)
                action = action.cpu().numpy().flatten()
        return action * self.action_scale

    def store_transition(self, state, action, reward, next_state, done):
        """存储一条经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """
        执行一步参数更新

        Returns:
            dict: 各项损失和指标
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # 采样
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ---- 更新 Critic ----
        with torch.no_grad():
            next_actions_raw, next_log_probs = self.actor.sample(next_states)
            next_actions = next_actions_raw * self.action_scale
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- 更新 Actor ----
        new_actions_raw, log_probs = self.actor.sample(states)
        new_actions = new_actions_raw * self.action_scale
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ---- 更新温度参数 ----
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()

        # ---- 软更新目标网络 ----
        self._soft_update(self.critic_target, self.critic)

        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
        }
        self.training_log.append(metrics)
        return metrics

    def save(self, path: str):
        """保存模型"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = torch.tensor(ckpt["log_alpha"], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def _hard_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)
