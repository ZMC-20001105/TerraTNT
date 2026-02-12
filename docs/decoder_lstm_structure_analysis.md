# 解码器LSTM结构深度分析

## 📐 LSTM架构

### 基本配置（代码第775-781行）

```python
self.lstm = nn.LSTM(
    input_size=hidden_dim,      # 256
    hidden_size=hidden_dim,     # 256
    num_layers=num_layers,      # 2 (双层LSTM)
    batch_first=True,
    dropout=0.2                 # 层间dropout
)
```

**关键参数**：
- **输入维度**: 256
- **隐藏维度**: 256
- **层数**: 2层
- **dropout**: 0.2（第1层到第2层之间）

### LSTM的形状和运转方式

#### 1. LSTM的内部结构

**每一层LSTM包含4个门**：
```
输入门 (i_t):  决定新信息有多少被接受
遗忘门 (f_t):  决定旧信息有多少被遗忘
输出门 (o_t):  决定隐藏状态有多少被输出
候选值 (g_t):  新的候选记忆内容
```

**公式**：
```
i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)
f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)
g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)
o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

#### 2. 双层LSTM的数据流

```
时间步 t:

输入 x_t (256维)
    ↓
┌─────────────────────┐
│   LSTM Layer 1      │
│  h1_t, c1_t (256)   │
└─────────────────────┘
    ↓ (dropout 0.2)
┌─────────────────────┐
│   LSTM Layer 2      │
│  h2_t, c2_t (256)   │
└─────────────────────┘
    ↓
输出 h2_t (256维)
```

**参数量计算**：
- 每层LSTM参数: 4 * (input_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
- Layer 1: 4 * (256 * 256 + 256 * 256 + 2 * 256) = 526,336
- Layer 2: 4 * (256 * 256 + 256 * 256 + 2 * 256) = 526,336
- **总计**: 1,052,672 参数

---

## 🔄 解码器每一步的输入构成

### Step 1: 基础输入构建（代码第858-862行）

```python
# 目标编码
goal_embed = self.goal_encoder(goal_features)  # (batch, 2) -> (batch, 64)

# 缩放目标向量
scaled_goal_embed = goal_embed * self.goal_vec_scale  # 当前值: 1.0

# 拼接基础输入
base_input = torch.cat([history_features, env_global, scaled_goal_embed], dim=1)
# 形状: (batch, 128 + 128 + 64) = (batch, 320)

# 投影到隐藏维度
base_input = self.input_projection(base_input)  # (batch, 320) -> (batch, 256)
```

**组成**：
- `history_features`: 128维（历史轨迹编码）
- `env_global`: 128维（全局环境特征）
- `scaled_goal_embed`: 64维（目标位置编码，经过缩放）

**关键点**：
- `goal_vec_scale = 1.0`（已从0.5增长到1.0）
- `base_input`在整个解码过程中**保持不变**
- 这是所有时间步的**共享基础**

---

### Step 2: 每个时间步的输入增强（代码第925-981行）

对于每个时间步 `t = 0, 1, ..., 359`：

#### 2.1 确定当前segment（waypoint段）

```python
# waypoint_stride = 18，共20个waypoint
# waypoint_indices = [17, 35, 53, ..., 341, 359]

seg_id = 当前t所属的waypoint段
start_wp = wp_nodes[:, seg_id, :]      # 段起点waypoint
end_wp = wp_nodes[:, seg_id + 1, :]    # 段终点waypoint
prog = (t - segment_start) / segment_length  # 段内进度 [0, 1]
```

#### 2.2 构建segment特征

```python
seg_in = torch.cat([
    start_wp,        # (batch, 2)
    end_wp,          # (batch, 2)
    torch.full((batch_size, 1), prog)  # (batch, 1)
], dim=1)  # (batch, 5)

seg_feat = self.segment_proj(seg_in)  # (batch, 5) -> (batch, 256)
```

#### 2.3 组装step_input（第一阶段）

```python
step_input = base_input + self.time_embed[t].unsqueeze(0) + seg_feat
# 形状: (batch, 256)
```

**此时包含**：
- `base_input`: 历史+全局环境+目标（不变）
- `time_embed[t]`: 时间步位置编码（可学习参数）
- `seg_feat`: 当前waypoint段特征

#### 2.4 添加环境token注意力

```python
if env_tokens is not None and self.use_pos_condition:
    step_input = step_input + self.env_attn(step_input, env_tokens)
    # env_attn: CrossAttention，从16个环境token中提取信息
```

#### 2.5 添加局部环境特征（关键！）

```python
if env_spatial is not None:
    # 确定环境采样位置
    if self.closed_loop_env_sampling:
        pos_query = pos_running  # 实际累积预测位置
    else:
        pos_query = start_wp + (end_wp - start_wp) * float(prog)  # waypoint线性插值
    
    # 从空间环境地图采样
    env_local = _sample_env(pos_query)  # (batch, 128) -> (batch, 256)
    
    # 添加到输入（权重很小！）
    step_input = step_input + self.env_local_scale_step * env_local
    # env_local_scale_step = 0.055
```

**关键问题**：
- 即使`closed_loop_env_sampling=True`，环境特征的权重只有**0.055**
- 这意味着环境信息对最终输入的贡献仅为**5.5%**

#### 2.6 添加自回归信息

```python
if self.autoregressive:
    delta_pad = torch.zeros(batch_size, self.hidden_dim, device=env_global.device)
    delta_pad[:, :2] = prev_delta * self.delta_inject_scale  # delta_inject_scale = 1.0
    step_input = step_input + delta_pad
```

**此时包含**：
- 前一步预测的位移`prev_delta`（前2维）
- 其余254维为0

---

### Step 3: LSTM前向传播（代码第962行）

```python
lstm_out, hidden = self.lstm(step_input.unsqueeze(1), hidden)
# 输入: (batch, 1, 256)
# 输出: lstm_out (batch, 1, 256), hidden (h, c) 各为 (2, batch, 256)
```

**LSTM内部处理**：
1. Layer 1接收`step_input`和上一步的`h1_{t-1}, c1_{t-1}`
2. Layer 1输出`h1_t`，经过dropout
3. Layer 2接收`h1_t`和上一步的`h2_{t-1}, c2_{t-1}`
4. Layer 2输出`h2_t`作为最终输出

---

### Step 4: 预测位移（代码第965行）

```python
delta = self.output_layer(lstm_out.squeeze(1))  # (batch, 256) -> (batch, 2)
predictions.append(delta)
```

---

### Step 5: 更新状态（代码第969-974行）

```python
# Teacher forcing
if self.training and ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
    prev_delta = ground_truth[:, t, :]  # 使用真实值
else:
    prev_delta = delta  # 使用预测值

# 更新累积位置
pos_running = pos_running + prev_delta
```

---

## 📊 完整的数据流图

```
时间步 t:

┌─────────────────────────────────────────────────────────────┐
│ 固定输入 (base_input, 256维)                                │
│   = history (128) + env_global (128) + goal*scale (64)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
                            + time_embed[t] (256维, 可学习)
                            ↓
                            + seg_feat (256维, 基于waypoint)
                            ↓
                            + env_attn(...) (256维, 环境token注意力)
                            ↓
                            + 0.055 * env_local (256维, 局部环境) ← 权重很小！
                            ↓
                            + delta_pad (前2维=prev_delta, 其余254维=0)
                            ↓
                      step_input (256维)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Layer 1                             │
│  输入: step_input (256) + h1_{t-1} (256) + c1_{t-1} (256)   │
│  输出: h1_t (256), c1_t (256)                               │
└─────────────────────────────────────────────────────────────┘
                            ↓ dropout(0.2)
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Layer 2                             │
│  输入: h1_t (256) + h2_{t-1} (256) + c2_{t-1} (256)         │
│  输出: h2_t (256), c2_t (256)                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    output_layer (256 -> 2)
                            ↓
                      delta_t (2维位移)
                            ↓
                  pos_running += delta_t
```

---

## 🔴 关键发现

### 1. 环境特征被严重抑制

```python
env_contribution = 0.055 * env_local
```

即使`env_local`包含了正确位置的环境信息（通过闭环采样），它对`step_input`的贡献仅为**5.5%**。

**对比其他组件**：
- `base_input`: 100%（基准）
- `time_embed[t]`: ~100%（可学习，初始化std=0.02）
- `seg_feat`: ~100%（MLP输出）
- `env_attn`: ~100%（注意力输出）
- `env_local`: **5.5%** ← 被严重抑制！
- `delta_pad`: 前2维100%，其余0

### 2. Waypoint的作用

Waypoint通过`seg_feat`影响每一步的输入：
```python
seg_feat = MLP([start_wp, end_wp, progress])
```

但是：
- Waypoint本身蜷缩在起点附近
- `start_wp ≈ (0, 0)`, `end_wp ≈ (0.5, 0.3)`
- 导致`seg_feat`几乎不提供有用的空间引导

### 3. LSTM的"记忆"问题

LSTM通过隐藏状态`(h, c)`在时间步之间传递信息：
- 如果早期步骤建立了"直线"的模式
- LSTM会倾向于延续这个模式（惯性）
- 需要足够强的外部信号（环境特征）来打破这个模式
- 但环境信号只有5.5%，太弱了！

### 4. 自回归的影响

```python
delta_pad[:, :2] = prev_delta
```

前一步的预测直接注入到当前步的输入：
- 如果前一步是直线方向，当前步倾向于继续直线
- 这是一个**正反馈循环**
- 除非有强环境约束来纠正方向

---

## 💡 为什么模型输出直线

### 综合分析

1. **基础输入不变**
   - `base_input`包含历史和目标，在所有时间步保持不变
   - 这给LSTM一个强烈的"从当前位置到目标"的信号
   - 最简单的路径就是直线

2. **Waypoint失效**
   - Waypoint蜷缩，无法提供有效的中间引导
   - `seg_feat`几乎不包含有用的空间信息

3. **环境信息太弱**
   - `env_local_scale_step = 0.055`
   - 即使闭环采样提供了正确位置的环境，权重太小
   - 无法对抗"直线偏好"

4. **LSTM的平滑倾向**
   - LSTM天然倾向于平滑、连续的输出
   - 在缺乏强约束的情况下，直线是最"安全"的选择
   - 最小化预测误差的波动

5. **自回归强化直线**
   - 一旦开始走直线，`prev_delta`会强化这个趋势
   - 形成正反馈循环

---

## 🎯 问题的本质

**模型实际上主要依赖**：
1. `base_input`（历史+目标）→ 指向直线
2. `time_embed`（时间编码）→ 中性
3. `seg_feat`（waypoint段）→ 失效（waypoint蜷缩）
4. `env_attn`（环境token）→ 全局信息，不足以引导弯曲
5. `env_local`（局部环境）→ **权重太小（5.5%）**
6. `prev_delta`（自回归）→ 强化当前趋势

**结果**：
- 主导信号指向直线
- 唯一能引导弯曲的信号（`env_local`）被严重抑制
- LSTM学习到的最优策略就是：**忽略环境，走直线**

---

## 📝 验证方法

要确认这个分析，我们需要：

1. **打印实际的数值**
   - `base_input`的范数
   - `env_local`的范数
   - `env_local_scale_step * env_local`的范数
   - 对比它们的相对大小

2. **环境消融实验**
   - 将环境地图置零，看ADE变化
   - 如果变化<5%，说明环境确实被忽略

3. **强制增大环境权重**
   - 设置`env_local_scale_step = 0.5`
   - 重新训练，看是否能学习弯曲

4. **可视化LSTM隐藏状态**
   - 查看`h_t`在时间步之间的变化
   - 分析是否存在"直线模式"的记忆

---

## 🔧 建议的修复方案

### 方案1: 增大环境权重（最直接）

```python
# 修改初始化
self.env_local_scale_step = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
# 或者训练时强制设置
--override_env_local_scale 0.5 --freeze_env_local_scale
```

### 方案2: 修改输入构建方式

```python
# 不要简单相加，而是拼接后投影
step_input = torch.cat([
    base_input,
    time_embed[t],
    seg_feat,
    env_local,  # 不缩放
    delta_pad
], dim=1)
step_input = projection_layer(step_input)  # 让模型学习权重
```

### 方案3: 添加环境约束loss

```python
# 惩罚预测位置与环境不一致
env_consistency_loss = compute_env_cost(pred_positions, env_map)
total_loss += lambda_env * env_consistency_loss
```

### 方案4: 修改LSTM架构

```python
# 使用attention-based decoder而非纯LSTM
# 或者在LSTM每一步注入更强的环境信息
```

---

**结论**：当前架构的核心问题是环境特征权重太小（5.5%），导致即使闭环采样提供了正确的环境信息，模型仍然主要依赖历史和目标来预测，自然倾向于直线。
