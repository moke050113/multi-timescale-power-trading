# 双DRL智能体系统设计文档（B方案）

**版本**: v1.0  
**日期**: 2026-10-19  
**状态**: 历史设计文档（结构已采纳，见下方说明）

> **阅读须知（2026-10-21，迭代 8.7）**：本文的**架构设计**（两个独立 SAC、独立经验池、
> 暴露成本塑形、按角色决策时序）仍是现行方案；但它写于结算重复计量修复之前，
> 因此文中的成本/改进数值与预期目标**已作废**，"现货智能体"亦已在迭代 8 更名为**柔性调度 SAC**。
> 文中提到的入口脚本（`train.py`、`train_dual_agent_v2.py`、`run_ablation.py`、
> `flexible_dispatcher.py`）已随作废产物一并删除，训练与评价统一使用
> `experiments/run_four_arm_benchmark.py`；消融改为在该入口上加开关实现。
> 现行实现与结论以 `report/课程设计报告.md`、`项目完成度评估报告.md` 为准。

---

## 1. 设计目标

### 1.1 任务书要求

> "针对不同交易时间尺度构建**相互衔接**的深度强化学习智能体，实现交易仓位与可调负荷的动态优化"

**核心要求**：
- ✅ 两套DRL智能体（中长期 + 现货）
- ✅ 相互衔接（信息交互与协调）
- ✅ 交易仓位优化（中长期智能体）
- ✅ 可调负荷优化（现货智能体）

### 1.2 设计原则

1. **满足任务书**：必须是两套DRL智能体，不能是规则调度器
2. **解决信用分配**：每个智能体有独立的奖励信号
3. **训练稳定**：采用交替训练策略
4. **可解释性**：每个智能体的动作有明确物理含义

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    TradingEnvironment                    │
│  - 4级市场：年度/月度/月内/现货                            │
│  - 状态空间：13维（仓位+价格+时间+柔性容量）                │
│  - 动作空间：2维（ratio_raw + flexible_action）           │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌────────▼────────
│  中长期智能体     │    │   现货智能体      │
│  (SAC Agent)    │    │   (SAC Agent)    │
│                 │    │                 │
│ 输入: 13维状态   │    │ 输入: 13维状态   │
│ 输出: ratio_raw │    │ 输出: flex_action│
│ 奖励: 合同成本   │    │ 奖励: 现货成本    │
│ 训练: 阶段1-3   │    │ 训练: 阶段4      │
└─────────────────┘    └─────────────────
```

### 2.2 智能体分工

| 智能体 | 负责阶段 | 动作 | 奖励 | 目标 |
|--------|---------|------|------|------|
| **中长期** | 年度/月度/月内 | `ratio_raw` → α/β/γ | `-合同成本/总需求` | 优化仓位比例 |
| **现货** | 现货 | `flexible_action` → EV+工业调度 | `-现货成本/(总需求×现货暴露)` | 优化柔性负荷 |

---

## 3. 环境改造

### 3.1 动作空间

**当前（分层架构）**：
```python
self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,))
```

**改造后（双智能体）**：
```python
self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
# action[0]: ratio_raw (中长期智能体)
# action[1]: flexible_action (现货智能体)
```

### 3.2 分阶段奖励

**关键改进**：每个智能体有独立的奖励信号

```python
def step(self, action):
    ratio_raw = sigmoid(action[0])
    flexible_action = np.clip(action[1], -1.0, 1.0)
    
    # ... 执行动作 ...
    
    # 分阶段奖励计算
    if self.current_stage == STAGE_SPOT:
        # 现货智能体奖励（放大信号）
        spot_exposure = 1 - self.gamma  # 例如 0.2
        reward_spot = -(spot_cost + penalty) / (self.total_demand * spot_exposure)
        reward_midlong = 0.0
    else:
        # 中长期智能体奖励
        reward_midlong = -contract_cost / self.total_demand
        reward_spot = 0.0
    
    return next_state, {
        "reward_midlong": reward_midlong,
        "reward_spot": reward_spot,
        "reward_global": reward_midlong + reward_spot  # 用于评估
    }, terminated, truncated, info
```

**为什么除以 spot_exposure**：
- 现货成本只占总成本的5-20%
- 除以现货暴露比例（0.05-0.2）可以放大5-20倍
- 使现货智能体的奖励信号与中长期智能体在同一量级

### 3.3 修复已知问题

#### 问题1：跨月结算bug

**当前代码**：
```python
# settlement.py
delta_beta = beta - self.beta  # 可能为负！
q_month = delta_beta * self.total_demand
cost = monthly_price * q_month
```

**修复后**：
```python
# 每月独立计算采购量
q_month = beta * self.total_demand  # 本月总采购量
cost = monthly_price * q_month
self.beta = beta  # 更新累计仓位
```

#### 问题2：费用口径不一致

**当前代码**：
```python
# reward 包含柔性惩罚
reward = -(cost + flexible_penalty) / total_demand

# 但 get_total_cost() 不包含
def get_total_cost(self):
    return self.settlement.get_total_cost()  # 只有市场成本
```

**修复后**：
```python
def get_total_cost(self):
    return (self.settlement.get_total_cost() + 
            self.total_flexible_penalty)  # 包含柔性惩罚
```

---

## 4. 智能体协调器

### 4.1 类设计

```python
class MultiScaleCoordinator:
    """
    双智能体协调器
    
    架构:
    - midlong_agent: 负责年度/月度/月内阶段
    - spot_agent: 负责现货阶段
    """
    
    def __init__(self, state_dim=13):
        # 中长期智能体
        self.midlong_agent = SACAgent(
            state_dim=state_dim,
            action_dim=1,  # ratio_raw
            action_scale=5.0,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=100000
        )
        
        # 现货智能体
        self.spot_agent = SACAgent(
            state_dim=state_dim,
            action_dim=1,  # flexible_action
            action_scale=1.0,  # 限制在[-1, 1]
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=100000
        )
        
        # 训练阶段标记
        self.frozen_midlong = False
        self.frozen_spot = False
    
    def select_action(self, state, stage, deterministic=False):
        """
        根据当前阶段选择动作
        
        Returns:
            action: np.ndarray([ratio_raw, flexible_action])
        """
        if stage in [STAGE_ANNUAL, STAGE_MONTHLY, STAGE_INTRADAY]:
            # 中长期阶段：使用中长期智能体
            ratio = self.midlong_agent.select_action(state, deterministic)
            return np.array([ratio[0], 0.0], dtype=np.float32)
        else:
            # 现货阶段：使用现货智能体
            flex = self.spot_agent.select_action(state, deterministic)
            return np.array([0.0, flex[0]], dtype=np.float32)
    
    def store_transition(self, state, action, rewards, next_state, done, stage):
        """存储经验到对应智能体"""
        if stage in [STAGE_ANNUAL, STAGE_MONTHLY, STAGE_INTRADAY]:
            if not self.frozen_midlong:
                self.midlong_agent.store_transition(
                    state, action[:1], rewards["reward_midlong"], 
                    next_state, done
                )
        else:
            if not self.frozen_spot:
                self.spot_agent.store_transition(
                    state, action[1:], rewards["reward_spot"],
                    next_state, done
                )
    
    def update(self, stage):
        """更新对应智能体"""
        if stage in [STAGE_ANNUAL, STAGE_MONTHLY, STAGE_INTRADAY]:
            if not self.frozen_midlong:
                return self.midlong_agent.update()
        else:
            if not self.frozen_spot:
                return self.spot_agent.update()
        return {}
```

### 4.2 关键设计决策

**决策1：动作解耦**
- 中长期智能体只输出 `ratio_raw`（action[0]）
- 现货智能体只输出 `flexible_action`（action[1]）
- 每个智能体只负责自己阶段的动作

**决策2：独立经验池**
- 两个智能体有独立的Replay Buffer
- 避免经验混合导致的分布偏移

**决策3：独立更新**
- 每个智能体只在自己的阶段更新
- 避免梯度冲突

---

## 5. 训练算法

### 5.1 三阶段交替训练

```python
def train_dual_agent(scenario, max_episodes=2000):
    env = TradingEnvironment(scenario_name=scenario)
    coordinator = MultiScaleCoordinator()
    
    # ========== 阶段1: 训练中长期智能体 (episodes 1-500) ==========
    print("Phase 1: Training mid-long term agent...")
    coordinator.frozen_spot = True  # 冻结现货智能体
    coordinator.frozen_midlong = False
    
    for ep in range(1, 501):
        train_episode(env, coordinator)
        
        if ep % 100 == 0:
            eval_cost = evaluate(env, coordinator)
            print(f"  Ep {ep}: cost={eval_cost:.0f}")
    
    # ========== 阶段2: 训练现货智能体 (episodes 501-1000) ==========
    print("Phase 2: Training spot agent...")
    coordinator.frozen_midlong = True  # 冻结中长期智能体
    coordinator.frozen_spot = False
    
    for ep in range(501, 1001):
        train_episode(env, coordinator)
        
        if ep % 100 == 0:
            eval_cost = evaluate(env, coordinator)
            print(f"  Ep {ep}: cost={eval_cost:.0f}")
    
    # ========== 阶段3: 联合训练 (episodes 1001-2000) ==========
    print("Phase 3: Joint training...")
    coordinator.frozen_midlong = False
    coordinator.frozen_spot = False
    
    for ep in range(1001, 2001):
        train_episode(env, coordinator)
        
        if ep % 100 == 0:
            eval_cost = evaluate(env, coordinator)
            print(f"  Ep {ep}: cost={eval_cost:.0f}")
    
    return coordinator
```

### 5.2 为什么交替训练？

**问题**：如果两个智能体同时训练
- 两个智能体的策略都在变化
- 环境对每个智能体来说都是非平稳的
- 难以收敛

**解决**：交替训练
- 阶段1：现货智能体固定（用随机策略），中长期智能体学习
- 阶段2：中长期智能体固定（用学到的策略），现货智能体学习
- 阶段3：两个智能体都放开，联合微调

**参考**：Codex项目（P5-Lite-S）也采用类似的三阶段训练

### 5.3 训练超参数

| 参数 | 中长期智能体 | 现货智能体 |
|------|------------|-----------|
| 学习率 | 3e-4 | 3e-4 |
| Batch size | 256 | 256 |
| Buffer size | 100,000 | 100,000 |
| Gamma (折扣因子) | 0.99 | 0.99 |
| Tau (软更新) | 0.005 | 0.005 |
| Alpha (系数) | 自动调节 | 自动调节 |
| Action scale | 5.0 | 1.0 |

---

## 6. 实验设计

### 6.1 实验1：单场景验证

**目标**：验证双智能体能正常训练

**设置**：
- 场景：A（年度低价后期上涨）
- Episodes：500
- 基线：固定比例策略

**预期结果**：
- 训练曲线收敛
- 成本低于固定比例基线
- 两个智能体的动作合理

### 6.2 实验2：5场景完整训练

**目标**：验证双智能体在不同场景下的表现

**设置**：
- 场景：A/B/C/D/E
- Episodes：2000（三阶段训练）
- 基线：固定比例 + 单智能体（分层架构）

**评估指标**：
- 平均成本改进（vs 固定比例）
- 与单智能体（分层架构）对比
- 两个智能体的动作分布

### 6.3 实验3：消融实验

**目标**：验证各个改进的有效性

**变体**：
1. **无交替训练**：两个智能体同时训练
2. **无奖励缩放**：现货奖励不除以 spot_exposure
3. **共享经验池**：两个智能体共用Replay Buffer

**预期**：完整方案 > 任何变体

---

## 7. 预期结果

### 7.1 乐观情况（概率40%）

| 智能体 | 效果 | 贡献 |
|--------|------|------|
| 中长期 | 学会动态调整α/β/γ | +20% |
| 现货 | 学会柔性调度 | +5% |
| **总计** | | **+25%** |

### 7.2 保守情况（概率50%）

| 智能体 | 效果 | 贡献 |
|--------|------|------|
| 中长期 | 学会动态调整α/β/γ | +20% |
| 现货 | 效果有限 | +1-2% |
| **总计** | | **+21-22%** |

### 7.3 最差情况（概率10%）

| 智能体 | 效果 | 贡献 |
|--------|------|------|
| 中长期 | 学会动态调整α/β/γ | +20% |
| 现货 | 学不到有效策略 | 0% |
| **总计** | | **+20%** |

**注意**：即使最差情况，也满足任务书"两套DRL智能体"要求

---

## 8. 风险与应对

### 8.1 风险矩阵

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 现货智能体仍学不到 | 中 | 高 | 进一步放大奖励，或简化动作空间 |
| 训练不稳定 | 低 | 中 | 降低学习率，增加batch size |
| 收敛太慢 | 中 | 中 | 增加预训练阶段，或使用行为克隆 |
| 时间不够 | 低 | 高 | 你有10周，足够完成 |

### 8.2 备用方案

**如果现货智能体确实学不到**：

方案A：简化动作空间
```python
# 从连续动作改为离散动作
action_space = spaces.Discrete(3)
# 0: 完全放电/减载
# 1: 不调节
# 2: 完全充电/增载
```

方案B：使用行为克隆
```python
# 先用规则调度器生成示范数据
# 然后用行为克隆预训练现货智能体
# 最后用DRL微调
```

方案C：混合架构
```python
# 现货智能体输出"是否使用规则"的决策
# 如果使用规则，调用规则调度器
# 如果不使用，自己决策
```

---

## 9. 实施时间线

### Week 1: 环境改造（3天）

| 天数 | 任务 | 交付物 |
|------|------|--------|
| Day 1 | 修改 `environment.py` | 2维动作空间 + 分阶段奖励 |
| Day 2 | 修复跨月结算bug | 修复后的 `settlement.py` |
| Day 3 | 统一费用口径 | 修复后的 `get_total_cost()` |

### Week 2: 智能体改造（3天）

| 天数 | 任务 | 交付物 |
|------|------|--------|
| Day 1 | 修改 `multi_agent.py` | 双智能体协调器 |
| Day 2 | 创建 `train_dual_agent_v2.py` | 交替训练脚本 |
| Day 3 | 模块测试 | `test_dual_agent.py` |

### Week 3: 实验验证（4天）

| 天数 | 任务 | 交付物 |
|------|------|--------|
| Day 1 | 单场景快速测试（200 episodes） | 训练曲线 |
| Day 2-3 | 5场景完整训练（2000 episodes） | 5个模型 + 结果 |
| Day 4 | 消融实验 + 结果分析 | 对比图表 |

### Week 4: 论文撰写（7天）

| 天数 | 任务 | 交付物 |
|------|------|--------|
| Day 1-2 | 方法论部分 | 架构图 + 算法描述 |
| Day 3-4 | 实验部分 | 结果表格 + 图表 |
| Day 5-6 | 讨论部分 | 与单智能体对比 + 局限性 |
| Day 7 | 修改润色 | 完整论文初稿 |

---

## 10. 与分层架构的对比

| 维度 | 分层架构（当前） | 双DRL智能体（B方案） |
|------|----------------|-------------------|
| **智能体数量** | 1个DRL + 1个规则 | 2个DRL |
| **任务书满足度** | ⚠️ 部分满足 | ✅ 完全满足 |
| **实现复杂度** | 低 | 中 |
| **训练稳定性** | 高 | 中 |
| **预期性能** | +24% | +20-25% |
| **研究价值** | 中（分层思想） | 高（多智能体协调） |
| **论文卖点** | "为什么用规则" | "如何解决信用分配" |

---

## 11. 代码修改清单

### 需要修改的文件

| 文件 | 改动内容 | 预计行数 |
|------|---------|---------|
| `src/market/environment.py` | 恢复2维动作 + 分阶段奖励 | +50行 |
| `src/market/settlement.py` | 修复跨月结算 | +10行 |
| `src/agent/multi_agent.py` | 双智能体协调器 | +80行 |
| `experiments/train.py` | 更新训练循环 | +30行 |

### 需要新建的文件

| 文件 | 内容 | 预计行数 |
|------|------|---------|
| `experiments/train_dual_agent_v2.py` | 交替训练脚本 | 200行 |
| `experiments/test_dual_agent.py` | 模块测试 | 100行 |

### 可以删除的文件

| 文件 | 原因 |
|------|------|
| `src/load/flexible_dispatcher.py` | 不再使用规则调度器 |

---

## 12. 验收标准

### 功能验收

- [ ] 环境能正常完成episode（37步）
- [ ] 两个智能体都能正常选择动作
- [ ] 分阶段奖励计算正确
- [ ] 交替训练流程正常
- [ ] 模型能正常保存和加载

### 性能验收

- [ ] 单场景训练收敛（成本曲线下降）
- [ ] 5场景平均改进 ≥ 20%（vs 固定比例）
- [ ] 现货智能体有正贡献（≥ 1%）

### 文档验收

- [ ] 代码注释完整
- [ ] 实验结果可视化
- [ ] 论文方法论部分完成

---

## 13. 下一步行动

**待你审核后**：

1. **开始Week 1**：修改环境代码
2. **每日进度同步**：每天汇报进展
3. **遇到问题随时讨论**：及时调整方案

**请审核以下问题**：

1. ✅ 架构设计是否合理？
2. ✅ 奖励缩放方案是否可行？
3. ✅ 交替训练策略是否合适？
4. ✅ 时间线是否可行？
5. ✅ 有其他建议吗？

---

**文档结束**

*等待你的审核意见...*
