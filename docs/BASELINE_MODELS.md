# 对比基线模型说明

## 📊 实验对比方案

为了全面评估TerraTNT模型的性能，我们将与以下6个基线模型进行对比实验：

---

## 1. Constant Velocity (CV) 模型

**类型：** 简单基线

**原理：**
- 假设目标以当前速度和方向匀速直线运动
- 预测公式：`p(t) = p(0) + v(0) × t`

**优点：**
- 实现简单，计算快速
- 适合短时预测

**缺点：**
- 无法处理转弯和加减速
- 不考虑环境约束

**预期性能：**
- ADE: ~15-20m
- FDE: ~40-50m

---

## 2. Social-LSTM

**论文：** Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded Spaces", CVPR 2016

**原理：**
- 使用LSTM编码历史轨迹
- Social Pooling层建模目标间交互
- LSTM解码器生成未来轨迹

**特点：**
- 考虑多目标社交交互
- 适用于人群场景

**局限性：**
- 不考虑环境约束（地形、道路）
- 主要针对行人，不适合车辆

**实现：**
```python
class SocialLSTM(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(input_size=2, hidden_size=128)
        self.social_pooling = SocialPoolingLayer()
        self.decoder = nn.LSTM(input_size=128, hidden_size=128)
```

---

## 3. YNet

**论文：** Mangalam et al., "It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction", ECCV 2020

**原理：**
- 场景编码器：CNN提取环境特征
- 目标预测：预测可能的终点位置
- 轨迹生成：基于终点生成完整轨迹

**特点：**
- 考虑场景语义信息
- 端点驱动的预测方式
- 多模态预测

**与TerraTNT的区别：**
- YNet使用语义分割图，TerraTNT使用18通道环境地图
- YNet不考虑地形高程和坡度
- TerraTNT有层次化解码器

**实现：**
```python
class YNet(nn.Module):
    def __init__(self):
        self.scene_encoder = UNet()  # 场景编码
        self.goal_predictor = GoalPredictor()  # 终点预测
        self.trajectory_decoder = Decoder()  # 轨迹解码
```

---

## 4. PECNet

**论文：** Mangalam et al., "PECNet: Trajectory Prediction with Planning-based Endpoint Conditioned Network", CVPR 2020

**原理：**
- 端点规划模块：预测多个候选终点
- 轨迹生成模块：基于终点生成轨迹
- 非参数化轨迹优化

**特点：**
- 显式建模终点不确定性
- 考虑场景约束
- 多模态预测

**与TerraTNT的区别：**
- PECNet使用占据栅格，TerraTNT使用多通道环境特征
- PECNet不考虑地形坡度和车辆动力学
- TerraTNT有XGBoost速度预测

**实现：**
```python
class PECNet(nn.Module):
    def __init__(self):
        self.endpoint_predictor = EndpointPredictor()
        self.trajectory_generator = TrajectoryGenerator()
        self.refinement = NonParametricRefinement()
```

---

## 5. Trajectron++

**论文：** Salzmann et al., "Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data", ECCV 2020

**原理：**
- 动态模型：考虑车辆动力学约束
- 图神经网络：建模目标间交互
- 条件变分自编码器：多模态预测

**特点：**
- 考虑动力学可行性
- 支持异构数据（车辆、行人）
- 概率预测

**与TerraTNT的区别：**
- Trajectron++主要用于城市交通场景
- 不考虑越野地形约束
- TerraTNT专门针对地面目标和复杂地形

**实现：**
```python
class TrajectronPP(nn.Module):
    def __init__(self):
        self.dynamics_model = DynamicsModel()
        self.interaction_graph = GNN()
        self.cvae = ConditionalVAE()
```

---

## 6. AgentFormer

**论文：** Yuan et al., "AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting", ICCV 2021

**原理：**
- Transformer架构：自注意力机制
- 时空编码：同时建模时间和空间关系
- 多目标交互：agent-aware attention

**特点：**
- 最新的Transformer架构
- 强大的表示学习能力
- 适合多目标场景

**与TerraTNT的区别：**
- AgentFormer主要关注社交交互
- 不考虑地形环境约束
- TerraTNT融合了环境感知和目标驱动

**实现：**
```python
class AgentFormer(nn.Module):
    def __init__(self):
        self.temporal_encoder = TemporalTransformer()
        self.spatial_encoder = SpatialTransformer()
        self.decoder = TransformerDecoder()
```

---

## 📈 对比实验设计

### 评估指标

| 指标 | 说明 | 单位 |
|------|------|------|
| **ADE** | 平均位移误差 | 米 (m) |
| **FDE** | 最终位移误差 | 米 (m) |
| **Goal Accuracy** | 目标预测准确率 | % |
| **Miss Rate** | 失败率 (FDE > 2m) | % |
| **Inference Time** | 推理时间 | 毫秒 (ms) |

### 实验场景

1. **短时预测** (10分钟)
   - 评估模型的短期预测能力
   
2. **长时预测** (60分钟)
   - 评估模型的长期预测能力
   
3. **复杂地形**
   - 山地、森林、城市等不同地形
   
4. **不同战术意图**
   - Intent1: 距离优先
   - Intent2: 隐蔽优先
   - Intent3: 地形优先

### 消融实验

| 实验 | 说明 | 目的 |
|------|------|------|
| **TerraTNT (完整)** | 所有模块 | 基准性能 |
| **w/o Environment** | 移除环境编码器 | 验证环境感知的作用 |
| **w/o Goal** | 移除目标分类器 | 验证目标驱动的作用 |
| **w/o Hierarchical** | 单层解码器 | 验证层次化解码的作用 |
| **Simple Env** | 仅DEM+LULC | 验证18通道地图的必要性 |

---

## 🎯 预期实验结果

### 定量对比（60分钟预测）

| 模型 | ADE (m) | FDE (m) | Goal Acc (%) | Time (ms) |
|------|---------|---------|--------------|-----------|
| **Constant Velocity** | 45.2 | 128.5 | 12.3 | 0.1 |
| **Social-LSTM** | 38.7 | 95.3 | 28.5 | 15 |
| **YNet** | 25.4 | 62.8 | 45.7 | 35 |
| **PECNet** | 22.1 | 58.3 | 52.3 | 42 |
| **Trajectron++** | 20.8 | 54.7 | 56.8 | 68 |
| **AgentFormer** | 19.5 | 51.2 | 61.2 | 85 |
| **TerraTNT (Ours)** | **15.3** | **38.6** | **73.5** | 45 |

### 定性分析

**TerraTNT的优势：**
1. ✅ 显式建模地形约束（坡度、LULC）
2. ✅ 考虑车辆动力学（速度、加速度）
3. ✅ 目标驱动的预测方式
4. ✅ 多战术意图支持
5. ✅ 实时推理能力

**局限性：**
1. ⚠️ 需要详细的环境数据
2. ⚠️ 不考虑多目标交互
3. ⚠️ 计算开销较大

---

## 💻 实现计划

### 1. 基线模型实现

```python
models/
├── baselines/
│   ├── constant_velocity.py
│   ├── social_lstm.py
│   ├── ynet.py
│   ├── pecnet.py
│   ├── trajectron_pp.py
│   └── agentformer.py
└── terratnt.py  # 我们的模型
```

### 2. 评估框架

```python
evaluation/
├── metrics.py          # ADE, FDE, Goal Accuracy
├── evaluator.py        # 统一评估接口
├── visualizer.py       # 结果可视化
└── comparison.py       # 模型对比
```

### 3. 实验脚本

```python
experiments/
├── train_baselines.py  # 训练所有基线模型
├── evaluate_all.py     # 评估所有模型
├── ablation_study.py   # 消融实验
└── plot_results.py     # 绘制对比图表
```

---

## 📊 论文图表规划

### Figure 1: 定量对比柱状图
- ADE/FDE对比
- 不同预测时长的性能

### Figure 2: 定性可视化
- 预测轨迹对比
- 多模态预测展示

### Figure 3: 消融实验
- 各模块贡献分析

### Figure 4: 案例分析
- 成功案例
- 失败案例分析

---

## 🚀 实施时间线

1. **Week 1-2**: 实现基线模型
2. **Week 3**: 训练所有模型
3. **Week 4**: 评估和对比实验
4. **Week 5**: 消融实验
5. **Week 6**: 论文撰写和图表制作

---

## 📚 参考文献

1. Alahi et al., "Social LSTM", CVPR 2016
2. Mangalam et al., "YNet", ECCV 2020
3. Mangalam et al., "PECNet", CVPR 2020
4. Salzmann et al., "Trajectron++", ECCV 2020
5. Yuan et al., "AgentFormer", ICCV 2021
