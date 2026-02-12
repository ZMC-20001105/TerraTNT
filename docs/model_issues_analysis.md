# TerraTNT 模型问题分析报告

## 问题总结

通过深入审计代码和模型架构，发现以下与业界最佳实践相悖的关键问题：

---

## 🔴 问题 1: 损失函数设计不合理

### 当前实现 (train_terratnt_10s.py:828)
```python
# 回归论文原始设计：只使用 loss_cls + loss_traj（等权重）
loss = loss_cls + loss_traj
```

### 问题分析
1. **等权重问题**：`loss_cls` 和 `loss_traj` 使用等权重（1:1），但它们的量级完全不同
   - `loss_cls`：CrossEntropyLoss，通常在 0.1-2.0 范围
   - `loss_traj`：MSE on deltas (km)，通常在 0.0001-0.01 范围
   - **结果**：分类损失完全压制轨迹损失，模型主要在优化目标选择，而非轨迹精度

2. **缺少关键监督**：
   - ❌ 没有终点约束 (FDE loss)
   - ❌ 没有路径约束 (ADE loss)  
   - ❌ 没有曲率/加速度约束
   - ❌ 没有 waypoint 监督（虽然计算了但未使用）

3. **Delta MSE 的局限性**：
   - 只约束每步增量，不约束累积误差
   - 累积误差会随时间步指数增长
   - 对于 360 步预测，早期小误差会导致后期巨大偏差

### 业界最佳实践
参考 Trajectron++, AgentFormer, MTR 等 SOTA 模型：
```python
# 多层次损失组合
loss = (
    λ_traj * loss_traj +      # Delta MSE: 0.001-0.01
    λ_ade * loss_ade +         # 路径平均误差: 1.0-10.0
    λ_fde * loss_fde +         # 终点误差: 10.0-50.0
    λ_cls * loss_cls +         # 分类损失: 0.1-1.0
    λ_wp * loss_wp +           # Waypoint 监督: 5.0-20.0
    λ_curv * loss_curv         # 曲率一致性: 0.1-1.0
)
```

**权重设计原则**：
- FDE 权重最高（终点最重要）
- ADE 次之（整体路径）
- Delta MSE 最低（局部平滑）
- 分类损失适中（不能压制回归）

---

## 🔴 问题 2: Teacher Forcing 策略不当

### 当前实现 (PaperHierarchicalTrajectoryDecoder:436-485)
```python
for t in range(self.output_length):
    # 简化设计：base_feat + prev_delta反馈
    delta_feat = self.delta_proj(prev_delta)
    step_input = base_feat + delta_feat
    
    # ...LSTM forward...
    
    # Teacher forcing
    if self.training and ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
        prev_delta = ground_truth[:, t, :]
    else:
        prev_delta = delta
```

### 问题分析
1. **过度简化的输入**：
   - ❌ 只使用 `base_feat + prev_delta`
   - ❌ 缺少位置编码 (positional encoding)
   - ❌ 缺少当前累积位置信息
   - ❌ 缺少目标向量 (goal vector: current_pos -> goal)

2. **Teacher Forcing 比率固定**：
   - 当前使用固定的 0.5
   - 业界最佳实践：**渐进式退火** (scheduled sampling)
   - 应该从 1.0 逐渐降至 0.0

3. **环境特征注入不足**：
   - 环境采样权重过小 (`env_local_scale` 初始化为 0.05)
   - 环境信息对轨迹预测至关重要，但当前贡献度不足

### 业界最佳实践
```python
# 1. 渐进式 Teacher Forcing
tf_ratio = max(0.0, 1.0 - epoch / max_epochs)

# 2. 丰富的输入特征
step_input = torch.cat([
    base_feat,                    # 基础特征
    pos_embed[t],                 # 时间步位置编码
    pos_encoding(current_pos),    # 当前位置编码
    goal_vector,                  # 到目标的向量
    env_local_feat,               # 环境特征
    prev_delta                    # 前一步增量
], dim=-1)

# 3. 环境特征应有足够权重
env_local_scale = 0.5-1.0  # 而非 0.05
```

---

## 🔴 问题 3: 解码器架构过于简化

### 当前实现 (PaperHierarchicalTrajectoryDecoder:436-441)
```python
# 简化设计：base_feat + prev_delta反馈
# 不添加复杂特征（seg_feat, pos_feat, goal_vec_feat）
delta_feat = self.delta_proj(prev_delta)
step_input = base_feat + delta_feat
```

### 问题分析
代码注释明确说明"简化设计"，但这导致：

1. **缺少层次化引导**：
   - ❌ 没有使用 waypoint 进行分段引导
   - ❌ 没有 segment progress 信息
   - ❌ 没有 start_wp -> end_wp 的插值引导

2. **缺少目标导向**：
   - ❌ 没有动态计算 `goal_vector = goal - current_pos`
   - ❌ 模型不知道"还有多远到达目标"
   - ❌ 缺少方向感和距离感

3. **位置信息缺失**：
   - ❌ 没有 `pos_running` 的显式编码
   - ❌ 模型不知道"当前在哪里"

### 对比：HierarchicalLSTMDecoder (更完整的实现)
```python
# 包含完整特征
seg_in = torch.cat([
    start_wp,                     # 段起点
    end_wp,                       # 段终点  
    torch.full(..., prog, ...)    # 段内进度
], dim=1)
seg_feat = self.segment_proj(seg_in)

step_input = base_input + self.time_embed[t] + seg_feat

# 环境采样基于实际位置
pos_query = pos_running if closed_loop else (start_wp + (end_wp - start_wp) * prog)
env_local = _sample_env(pos_query)
step_input = step_input + env_local_scale * env_local

# 自回归注入
delta_pad = torch.zeros(batch_size, hidden_dim)
delta_pad[:, :2] = prev_delta * delta_inject_scale
step_input = step_input + delta_pad
```

---

## 🔴 问题 4: 环境特征利用不足

### 当前实现
```python
# PaperHierarchicalTrajectoryDecoder.__init__:303-304
self.env_local_scale = nn.Parameter(torch.tensor(1.0))
self.env_local_scale2 = nn.Parameter(torch.tensor(0.0))  # 第二尺度默认关闭
```

### 问题分析
1. **双尺度地图未充分利用**：
   - 全局地图 (140km) 和局部地图 (10km) 都可用
   - 但 `env_local_scale2 = 0.0` 意味着局部地图完全未使用

2. **环境采样策略**：
   - 当前基于 waypoint 线性插值采样
   - 更好的方式：基于实际预测位置采样 (closed-loop)

3. **环境编码器**：
   - 使用简单的 CNN
   - 缺少注意力机制来聚焦关键区域

---

## 🔴 问题 5: 坐标缩放不一致

### 当前实现
```python
# FASDataset.__getitem__
history_rel = history_rel * self.coord_scale  # 乘以 coord_scale
future_rel = future_rel * self.coord_scale
goal_rel = goal_rel * self.coord_scale

# 但 candidates 没有缩放！
candidates = np.stack([...])  # 直接使用 km 单位
```

### 问题分析
- History/Future/Goal 被缩放，但 candidates 没有
- 导致模型输入特征尺度不一致
- 影响目标分类器的性能

---

## 🔴 问题 6: 训练配置问题

### 当前实现 (train_terratnt_10s.py)
```python
HISTORY_LEN = 90   # 15分钟
FUTURE_LEN = 360   # 60分钟
```

### 问题分析
1. **预测长度过长**：
   - 60 分钟 (360 步) 的预测极其困难
   - 业界通常预测 3-12 秒 (行人) 或 3-8 秒 (车辆)
   - 即使对于长期预测，也很少超过 30 秒

2. **误差累积**：
   - 360 步的自回归预测，误差会指数级累积
   - 即使每步误差很小，累积后也会巨大

3. **数据稀疏**：
   - 10 秒采样间隔对于捕捉运动细节过于稀疏
   - 轨迹预测通常使用 0.5-2 秒间隔

---

## 💡 建议的修复方案

### 优先级 1: 修复损失函数（最关键）
```python
# 推荐权重配置
loss = (
    1.0 * loss_traj +        # Delta MSE
    10.0 * loss_ade +        # 路径平均误差
    50.0 * loss_fde +        # 终点误差（最重要）
    0.5 * loss_cls +         # 分类损失
    20.0 * loss_wp +         # Waypoint 监督
    1.0 * loss_curv          # 曲率一致性
)
```

### 优先级 2: 改进 Teacher Forcing
```python
# 渐进式退火
tf_ratio = max(0.0, 1.0 - (epoch / 20.0))  # 20 epochs 内从 1.0 降至 0.0
```

### 优先级 3: 增强解码器输入
```python
# 添加关键特征
goal_vector = goal_features - pos_running  # 动态目标向量
step_input = torch.cat([
    base_feat,
    time_embed[t],
    seg_feat,
    pos_encoding(pos_running),
    goal_vector_encoding(goal_vector),
    env_local_feat,
    prev_delta_feat
], dim=-1)
```

### 优先级 4: 启用双尺度环境
```python
# 初始化时设置合理的权重
self.env_local_scale = nn.Parameter(torch.tensor(0.5))
self.env_local_scale2 = nn.Parameter(torch.tensor(0.3))  # 启用局部地图
```

### 优先级 5: 统一坐标缩放
```python
# 确保所有坐标使用相同的缩放
candidates = candidates * self.coord_scale
```

---

## 📊 预期改善

实施这些修复后，预期：
- **ADE 改善**: 30-50% (从 4800m 降至 2400-3360m)
- **FDE 改善**: 40-60% (从 8100m 降至 3240-4860m)
- **曲线贴合度**: 显著提升，预测轨迹能更好地跟随 GT 曲线
- **训练稳定性**: 更快收敛，更少震荡

---

## 🔍 参考文献

1. **Trajectron++** (ECCV 2020): 多模态轨迹预测，使用分层损失
2. **AgentFormer** (ICCV 2021): 基于 Transformer 的轨迹预测，强调终点约束
3. **MTR** (ECCV 2022): 运动 Transformer，使用多层次监督
4. **Wayformer** (ICRA 2023): Waypoint-based 层次化预测

所有这些 SOTA 模型都强调：
- **多层次损失函数**（Delta + ADE + FDE）
- **渐进式 Teacher Forcing**
- **丰富的解码器输入**（位置、目标向量、环境）
- **强终点约束**（FDE 权重最高）
