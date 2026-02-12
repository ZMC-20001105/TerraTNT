# TerraTNT 预测模式详解

## 两种预测模式的区别

### 1. **pred(gt_goal)** - 使用真实目标预测

**定义**：
- 模型在预测时使用**真实的终点目标**（ground truth goal）
- 相当于告诉模型"你要去这个地方"，然后评估它能否规划出合理的路径

**训练阶段**：
- **Phase1 (fas1)**: `goal_mode='given'`
- **Phase2 (fas2)**: `goal_mode='given'`

**评估指标**：
- 衡量模型在**已知终点**情况下的轨迹生成能力
- 专注于路径规划质量，不考虑目标选择

**可视化命令**：
```bash
python scripts/visualize_terratnt_predictions.py \
  --checkpoint path/to/checkpoint.pth \
  --goal_mode given \
  --num_samples 10
```

**应用场景**：
- 任务明确的场景（如导航到指定地点）
- 评估纯轨迹生成能力
- Phase1/2 训练验证

---

### 2. **pred(top1/ckpt1)** - 模型自主选择目标

**定义**：
- 模型从候选目标中**自主选择**最可能的终点
- 通过目标分类器评估所有候选目标，选择概率最高的
- 然后生成到达该目标的轨迹

**训练阶段**：
- **Phase3 (fas3)**: `goal_mode='joint'`

**评估指标**：
- 衡量模型的**完整预测能力**：
  - 目标选择准确性（分类任务）
  - 轨迹生成质量（回归任务）

**可视化命令**：
```bash
python scripts/visualize_terratnt_predictions.py \
  --checkpoint path/to/checkpoint.pth \
  --goal_mode joint \
  --num_samples 10
```

**应用场景**：
- 目标不明确的场景（如预测目标可能去哪里）
- 完整的端到端预测
- Phase3 训练验证

---

## 可视化输出解读

### 示例输出
```
样本 1: top1 ADE=1748.3m FDE=1714.4m | gt ADE=1748.3m FDE=1714.4m
```

**字段含义**：
- **top1 ADE/FDE**: 模型自选目标（概率最高的候选）的预测误差
- **gt ADE/FDE**: 使用真实目标的预测误差

### 为什么两者可能相同？

当使用 `--goal_mode given` 时：
- 模型被强制使用真实目标
- top1 和 gt 的预测结果完全相同
- 主要用于评估轨迹生成质量

当使用 `--goal_mode joint` 时：
- top1: 模型自选的目标（可能不是真实目标）
- gt: 使用真实目标
- 两者通常不同，用于评估完整预测能力

---

## 训练阶段对应关系

### Phase1 (fas1) - 基础轨迹生成
```python
goal_mode = 'given'
# 模型输入：环境、历史、真实目标
# 模型输出：轨迹预测
# 损失函数：只有轨迹损失（loss_traj）
```

### Phase2 (fas2) - 增强轨迹生成
```python
goal_mode = 'given'
# 与 Phase1 相同，但使用更多数据或更长训练
```

### Phase3 (fas3) - 联合预测
```python
goal_mode = 'joint'
# 模型输入：环境、历史、候选目标集合
# 模型输出：
#   1. 目标概率分布（goal_logits）
#   2. 轨迹预测（基于选中的目标）
# 损失函数：
#   - 分类损失（loss_cls）：目标选择
#   - 轨迹损失（loss_traj, loss_ade, loss_fde）：路径生成
```

---

## 评估指标对比

### pred(gt_goal) 指标
- **优点**：
  - 纯粹评估轨迹生成质量
  - 排除目标选择的影响
  - 适合对比不同模型的路径规划能力

- **局限**：
  - 不反映真实应用场景
  - 无法评估目标选择能力

### pred(top1) 指标
- **优点**：
  - 反映完整的预测能力
  - 更接近真实应用
  - 同时评估目标选择和路径生成

- **局限**：
  - 如果目标选择错误，ADE/FDE 会很高
  - 难以区分是目标选择问题还是路径生成问题

---

## 最佳实践

### 训练策略
1. **Phase1/2**: 先训练 `goal_mode='given'`
   - 让模型学会基础的轨迹生成
   - 不考虑目标选择的复杂性

2. **Phase3**: 再训练 `goal_mode='joint'`
   - 在已有轨迹生成能力的基础上
   - 学习目标选择

### 评估策略
1. **开发阶段**: 主要看 `pred(gt_goal)`
   - 快速评估轨迹生成质量
   - 调试模型架构和超参数

2. **最终评估**: 同时看 `pred(gt_goal)` 和 `pred(top1)`
   - `pred(gt_goal)`: 评估轨迹生成上限
   - `pred(top1)`: 评估实际应用性能
   - 两者差距反映目标选择的准确性

---

## 当前训练状态

### 修复后的效果 (Epoch 4, batch_size=16)
- **pred(gt_goal) ADE**: 2886 m (改善 39.9%)
- **pred(gt_goal) FDE**: 3251 m (改善 59.9%)

### 正在进行的训练
- **配置**: batch_size=64, 30 epochs
- **预期**: 进一步改善并加速训练（约 3x 加速）
- **目标**: ADE < 2500m, FDE < 3000m
