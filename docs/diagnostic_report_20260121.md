# TerraTNT 模型诊断报告
**日期**: 2026-01-21  
**问题**: 可视化 ADE 30.5km vs 训练验证 ADE 5.9km（5.2倍差距）

## 一、核心问题总结

### 1.1 指标不一致现象
- **训练验证指标**: `val_ade=5858.96m`, `val_fde=10790.28m`
- **可视化指标**: `pred(gt_goal) ADE=30456.5m`, `FDE=71769.3m`
- **差距**: 5.2倍 ADE，6.6倍 FDE
- **模型表现**: 比恒速 baseline 差 3.4倍，比直线到 goal baseline 差 14.6倍

### 1.2 预测质量分析
- **不再是 100% 直线**: `straight_frac=0.140`（14% 直线）
- **但弯曲方向错误**: `mean_norm_dev=0.2342` >> GT 的 `0.0717`
- **说明**: 模型有弯曲，但弯曲方式与真实轨迹不符

---

## 二、代码审查发现的关键问题

### 2.1 坐标归一化不一致 ⚠️ **严重**

#### 问题描述
训练和可视化中的坐标归一化存在多处不一致：

**训练脚本 (`train_terratnt_10s.py`)**:
```python
# Line 191-197: 坐标缩放
history[:, 0:2] = history[:, 0:2] * self.coord_scale  # 默认 1.0
future_delta = torch.diff(future_rel, dim=0, prepend=torch.zeros(1, 2))
future = future_delta * self.coord_scale
goal = torch.as_tensor(goal_rel_km * self.coord_scale, dtype=torch.float32)
candidates = torch.as_tensor(cand_rel_km.astype(np.float32), dtype=torch.float32) * self.coord_scale
```

**模型解码器 (`terratnt.py`)**:
```python
# Line 309-314: _pos_to_grid 坐标归一化
def _pos_to_grid(self, pos_xy: torch.Tensor, coverage_km: Optional[float] = None) -> torch.Tensor:
    cov = float(self.env_coverage_km if coverage_km is None else coverage_km)
    half = max(1e-6, cov * 0.5)
    gx = (pos_xy[:, 0] / half).clamp(-1.0, 1.0)  # 除以 70km
    gy = (-pos_xy[:, 1] / half).clamp(-1.0, 1.0)  # 除以 70km
    return torch.stack([gx, gy], dim=1)
```

**问题分析**:
1. **输入坐标单位**: 数据中 `history_rel`, `future_rel`, `goal_rel` 都是 **km** 单位
2. **coord_scale 应用**: 训练时对所有坐标乘以 `coord_scale=1.0`（实际无效）
3. **解码器归一化**: `_pos_to_grid` 假设输入是 km，除以 `env_coverage_km/2=70km` 归一化到 `[-1,1]`
4. **目标归一化**: 解码器中 `goal_norm_denom=70km`，但实际输入已经是 km 单位

**潜在影响**:
- 如果 `coord_scale` 在训练和可视化中不一致，会导致坐标量级错误
- 目标归一化分母 `70km` 可能过大，导致 goal 信号过弱

---

### 2.2 环境特征归一化缺失 ⚠️ **严重**

#### 问题描述
环境地图的各通道没有进行归一化处理：

**数据生成 (`trajectory_generator_v2.py`)**:
```python
# DEM: 原始高程值（米）
# Slope: 原始坡度值（度）
# LULC: one-hot 编码（0 或 1）
# Road: 二值（0 或 1）
```

**CNN 编码器 (`terratnt.py`)**:
```python
# Line 24-26: 直接输入 CNN，没有归一化
nn.Conv2d(input_channels, 32, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
nn.BatchNorm2d(out_ch),  # BatchNorm 在卷积后
nn.ReLU(inplace=True),
```

**问题分析**:
1. **DEM 值范围**: 可能从 0 到数千米，量级远大于 LULC 的 0/1
2. **Slope 值范围**: 0-90 度，量级也远大于 LULC
3. **BatchNorm 位置**: 在卷积后，无法解决输入通道间量级差异
4. **业界最佳实践**: SAPI 论文中环境图像使用 0-255 归一化编码

**潜在影响**:
- DEM 和 Slope 通道会主导 CNN 的学习
- LULC 和 Road 等关键信息被淹没
- 环境特征提取效果差

---

### 2.3 特征融合方式问题 ⚠️ **中等**

#### 问题描述
解码器中的特征融合存在潜在问题：

**当前实现**:
```python
# Line 258-260: 基础输入拼接
base_input = torch.cat([history_features, env_global, scaled_goal_embed], dim=1)  # (batch, 320)
base_input = self.input_projection(base_input)  # (batch, hidden_dim)
```

**问题分析**:
1. **直接拼接**: 没有考虑各特征的量级差异
2. **缺少残差连接**: SAPI 论文中使用了 refiner 机制（短路连接）
3. **env_local 注入方式**: 通过加法注入，但缺少缩放控制

**业界最佳实践** (SAPI):
```python
# 环境特征先通过 CNN 编码
# 然后与历史轨迹拼接
# 通过 LSTM 处理
# 使用 refiner 机制回看原始历史轨迹
```

---

### 2.4 环境覆盖范围不一致 ⚠️ **严重**

#### 问题描述
环境地图的实际覆盖范围与配置不一致：

**数据生成**:
```python
# trajectory_generator_v2.py: extract_100km_env_map
coverage = 140000  # 硬编码 140km
# 但字段名为 env_map_100km
```

**训练配置**:
```python
env_coverage_km = 140.0  # 正确
```

**解码器配置**:
```python
self.env_coverage_km = float(kwargs.get('env_coverage_km', 140.0))  # 默认 140km
```

**问题分析**:
1. **字段名误导**: `env_map_100km` 实际是 140km
2. **如果配置错误**: 会导致环境采样位置错位
3. **已确认**: 当前训练使用 `env_coverage_km=140.0`，与数据一致

**验证需求**:
- 检查可视化脚本是否正确读取 checkpoint 中的 `env_coverage_km`

---

### 2.5 自回归模式不一致 ⚠️ **中等**

#### 问题描述
可视化脚本之前默认关闭自回归：

**之前的可视化脚本**:
```python
# 强制设置为 False
model.decoder.autoregressive = False
model.decoder.closed_loop_env_sampling = False
```

**训练默认**:
```python
self.autoregressive = True  # Line 761
```

**已修复**: 最新的可视化脚本已移除强制覆盖

---

## 三、与业界最佳实践的对比

### 3.1 SAPI 论文的关键设计

**环境编码**:
- 使用 **单通道图像** 编码环境信息
- 像素值有明确物理意义：255=可达区域，0=不可达
- **能量编码**: 高像素值 = 低能量 = 更倾向的状态

**特征提取**:
- **3D 卷积**: 处理时序环境图像序列
- **2D 卷积**: 提取空间特征
- **特征融合**: 环境编码 + 历史轨迹拼接 → LSTM

**Refiner 机制**:
```python
# 短路连接，回看原始历史轨迹
refined = W1 * raw_history + W2 * learned_features
```

### 3.2 TerraTNT 的差异

| 方面 | SAPI | TerraTNT | 问题 |
|------|------|----------|------|
| 环境编码 | 0-255 归一化 | 原始值（DEM 数千米） | ❌ 量级不一致 |
| 时序处理 | 3D Conv | 单帧 2D Conv | ⚠️ 缺少时序建模 |
| 特征融合 | Refiner 短路 | 直接拼接 | ⚠️ 缺少残差 |
| 坐标归一化 | 明确归一化 | 多处不一致 | ❌ 严重问题 |

---

## 四、推荐的修复方案

### 4.1 立即修复（高优先级）

#### 修复 1: 环境地图归一化
```python
# 在 FASDataset.__getitem__ 中添加
def normalize_env_map(env_map_np):
    """归一化环境地图各通道到 [0, 1]"""
    normalized = np.zeros_like(env_map_np)
    
    # DEM (channel 0): 归一化到 [0, 1]
    dem = env_map_np[0]
    if dem.max() > dem.min():
        normalized[0] = (dem - dem.min()) / (dem.max() - dem.min())
    
    # Slope (channel 1): 除以 90 度
    normalized[1] = np.clip(env_map_np[1] / 90.0, 0, 1)
    
    # Aspect sin/cos (channels 2-3): 已经在 [-1, 1]，映射到 [0, 1]
    normalized[2] = (env_map_np[2] + 1.0) / 2.0
    normalized[3] = (env_map_np[3] + 1.0) / 2.0
    
    # LULC one-hot (channels 4-13): 已经是 0/1
    normalized[4:14] = env_map_np[4:14]
    
    # Road (channel 15): 已经是 0/1
    normalized[15] = env_map_np[15]
    
    # History heatmap (channel 16): 归一化
    hist = env_map_np[16]
    if hist.max() > 0:
        normalized[16] = hist / hist.max()
    
    # Goal map (channel 17): 已经是 0/1
    normalized[17] = env_map_np[17]
    
    return normalized
```

#### 修复 2: 检查坐标缩放一致性
```python
# 在可视化脚本中验证
print(f"Checkpoint coord_scale: {ckpt.get('coord_scale', 1.0)}")
print(f"Dataset coord_scale: {val_dataset.coord_scale}")
assert abs(ckpt.get('coord_scale', 1.0) - val_dataset.coord_scale) < 1e-6
```

#### 修复 3: 验证 env_coverage_km 一致性
```python
# 在可视化脚本中验证
print(f"Checkpoint env_coverage_km: {ckpt.get('env_coverage_km', 140.0)}")
print(f"Model env_coverage_km: {model.decoder.env_coverage_km}")
assert abs(ckpt.get('env_coverage_km', 140.0) - model.decoder.env_coverage_km) < 1e-6
```

### 4.2 中期改进（中优先级）

#### 改进 1: 添加 Refiner 机制
```python
class Refiner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
    
    def forward(self, raw, learned):
        return self.w1 * raw + self.w2 * learned
```

#### 改进 2: 改进特征融合
```python
# 使用 LayerNorm 而不是直接拼接
self.history_norm = nn.LayerNorm(history_feature_dim)
self.env_norm = nn.LayerNorm(env_feature_dim)
self.goal_norm = nn.LayerNorm(64)

base_input = torch.cat([
    self.history_norm(history_features),
    self.env_norm(env_global),
    self.goal_norm(scaled_goal_embed)
], dim=1)
```

### 4.3 长期优化（低优先级）

#### 优化 1: 引入 3D 卷积处理时序环境
```python
class TemporalEnvironmentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(18, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2))
        self.conv2d = nn.Conv2d(32, 64, kernel_size=3, stride=2)
```

#### 优化 2: 改进目标编码
```python
# 使用相对距离和方向
goal_dist = torch.norm(goal_features, dim=1, keepdim=True)
goal_dir = goal_features / (goal_dist + 1e-6)
goal_embed = self.goal_encoder(torch.cat([goal_dist, goal_dir], dim=1))
```

---

## 五、下一步行动计划

### 5.1 立即执行
1. ✅ 创建诊断报告（本文档）
2. ⏳ 运行环境消融实验验证模型是否使用环境特征
3. ⏳ 检查训练日志中的 `coord_scale` 和 `env_coverage_km`
4. ⏳ 在可视化脚本中添加配置一致性检查

### 5.2 短期计划
1. 实现环境地图归一化
2. 重新训练模型验证效果
3. 对比归一化前后的指标

### 5.3 中期计划
1. 添加 Refiner 机制
2. 改进特征融合方式
3. 引入 LayerNorm

---

## 六、风险评估

### 6.1 高风险问题
- ❌ **环境地图归一化缺失**: 可能导致模型无法有效学习环境特征
- ❌ **坐标归一化不一致**: 可能导致训练和推理的量级错误

### 6.2 中风险问题
- ⚠️ **特征融合方式**: 可能导致某些特征被淹没
- ⚠️ **缺少残差连接**: 可能导致梯度消失

### 6.3 低风险问题
- ℹ️ **缺少时序建模**: 当前使用单帧环境图，可能损失时序信息

---

## 七、参考文献

1. **SAPI**: Surroundings-Aware Vehicle Trajectory Prediction at Intersections (arXiv:2306.01812)
2. **TNT**: Target-driveN Trajectory Prediction (arXiv:2008.08294)
3. **Batch Normalization**: Accelerating Deep Network Training by Reducing Internal Covariate Shift

---

**报告生成时间**: 2026-01-21 21:30  
**下次更新**: 完成环境消融实验后
