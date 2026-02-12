# 主流轨迹预测模型架构分析与TerraTNT对比

## 1. 模型概览

### 1.1 Trajectron++ (2020)
**核心思想**: 多模态图结构化场景表示 + 动态图注意力

**架构组件**:
- **历史编码**: LSTM编码历史轨迹
- **交互建模**: 动态边缘图神经网络 (Dynamic Edge GNN)
- **场景表示**: 栅格化语义地图 (semantic map rasterization)
- **解码器**: CVAE (Conditional VAE) 生成多模态轨迹
- **目标处理**: 无显式目标分类器，依赖CVAE隐变量

**特征使用**:
- Agent历史轨迹 (位置、速度、加速度)
- 邻近agent交互 (相对位置、速度)
- 语义地图 (道路、人行道、车道线等)
- 地图通过CNN编码为空间特征图

**优势**:
- 强大的多agent交互建模
- 多模态预测能力
- 场景上下文感知

**劣势**:
- 计算复杂度高 (图神经网络)
- 需要完整的场景图信息
- 对地图质量依赖高

---

### 1.2 PECNet (2020)
**核心思想**: Predict Endpoint Conditioned Network - 先预测终点，再生成轨迹

**架构组件**:
- **历史编码**: MLP编码历史轨迹
- **终点预测**: VAE生成多个候选终点
- **轨迹解码**: 给定终点条件下的MLP解码器
- **目标处理**: 显式建模终点分布

**特征使用**:
- Agent历史轨迹 (相对位置)
- 目标终点 (绝对或相对坐标)
- **不使用环境地图** (纯轨迹数据驱动)
- 社交池化 (可选，用于多agent场景)

**优势**:
- 架构简单，训练快速
- 多模态终点采样
- 不依赖地图数据

**劣势**:
- **无环境约束** - 可能生成穿墙/越野轨迹
- 缺乏细粒度场景理解
- 对复杂环境适应性差

---

### 1.3 AgentFormer (2021)
**核心思想**: Transformer架构 + agent-aware注意力机制

**架构组件**:
- **历史编码**: Transformer encoder处理历史序列
- **交互建模**: Agent-aware attention (区分时间和社交维度)
- **场景表示**: 可选地图编码器 (CNN)
- **解码器**: Transformer decoder + 多模态采样
- **目标处理**: 隐式学习，无显式目标分类器

**特征使用**:
- Agent历史轨迹 (位置、速度)
- 多agent交互 (通过attention机制)
- 语义地图 (可选，通过CNN编码)
- 位置编码 (绝对和相对)

**优势**:
- Transformer强大的序列建模能力
- 灵活的注意力机制
- 可扩展到大规模场景

**劣势**:
- 训练数据需求大
- 计算成本高
- 对地图特征利用不充分

---

### 1.4 LaneGCN (2020)
**核心思想**: 车道图卷积网络 + 多尺度特征融合

**架构组件**:
- **历史编码**: 1D CNN处理轨迹序列
- **地图表示**: **车道图结构** (节点=车道段，边=连接关系)
- **交互建模**: Actor-to-Lane + Lane-to-Actor + Actor-to-Actor GCN
- **解码器**: 多头预测 (每个头对应一种可能的未来)
- **目标处理**: 隐式，通过车道图引导

**特征使用**:
- Agent历史轨迹
- **结构化车道图** (车道中心线、连接关系、车道属性)
- 交通规则 (转向限制、优先级)
- 多agent交互

**优势**:
- **显式利用道路拓扑结构**
- 强大的车道级推理能力
- 适合自动驾驶场景

**劣势**:
- 需要高质量矢量化地图 (HD map)
- 对越野/非结构化环境不适用
- 地图预处理复杂

---

### 1.5 HOME (2021)
**核心思想**: Hierarchical Object-to-Zone Graph + 多尺度环境建模

**架构组件**:
- **历史编码**: GRU编码轨迹
- **地图表示**: **分层图结构** (对象级 + 区域级)
- **交互建模**: 对象-区域-对象三层图卷积
- **解码器**: 目标条件化的GRU解码器
- **目标处理**: 显式目标区域预测

**特征使用**:
- Agent历史轨迹
- **分层语义地图** (道路、建筑、停车场等区域)
- 对象属性 (类型、大小、方向)
- 空间关系 (包含、邻接)

**优势**:
- 多尺度环境理解
- 显式建模空间层次关系
- 适合复杂城市场景

**劣势**:
- 需要详细的语义分割地图
- 图构建开销大
- 对地图质量敏感

---

## 2. TerraTNT架构分析

### 2.1 当前架构
**核心思想**: 目标驱动 + 环境约束的层次化轨迹生成

**架构组件**:
- **历史编码**: 双层LSTM (history_encoder)
- **环境编码**: 
  - 全局: PaperCNNEnvironmentEncoder (100km×100km, 18通道)
  - 局部: DualScaleEnvironmentEncoder (可选，10km×10km)
- **目标分类器**: MLP对候选终点评分
- **解码器**: PaperHierarchicalTrajectoryDecoder
  - Waypoint层: LSTM生成中间路点
  - Step层: LSTM生成逐步位移

**特征使用**:
- Agent历史轨迹 (相对位置，10分钟)
- **18通道环境栅格地图**:
  1. DEM (高程)
  2. Slope (坡度)
  3. Aspect sin/cos (坡向)
  4-13. LULC one-hot (土地利用类型)
  14. Tree cover (植被覆盖度)
  15. Road (道路掩码)
  16. History heatmap (历史轨迹热图)
  17. Candidate goal map (候选目标热图)
- 目标终点 (从道路采样的候选点)
- Waypoint引导 (中间路点)

### 2.2 当前问题诊断

**问题1: 预测轨迹过于直线化**
- 现象: straight_frac(pred)≈0.905 vs GT≈0.656
- 可能原因:
  1. **Goal vector注入过强**: `goal_vec_feat`直接加到`step_input`，无可学习缩放
  2. **局部环境特征利用不足**: `env_local_scale2`参数缺失
  3. **Waypoint stride过大**: stride=36意味着只有2个中间路点 (60步/36≈1.67)
  4. **Teacher forcing schedule**: 可能过早衰减导致模型依赖直线捷径

**问题2: ADE较高 (≈3545m)**
- 可能原因:
  1. 环境特征提取不充分 (CNN编码器容量)
  2. 历史轨迹编码维度较低
  3. 缺乏多模态预测能力 (只预测单条轨迹)
  4. 目标分类器准确率不足

**问题3: 环境约束未有效利用**
- 现象: Line-to-goal baseline ADE≈1918m < TerraTNT≈3545m
- 说明: 模型未能有效利用环境信息来改善预测

---

## 3. 与主流模型对比

| 特性 | TerraTNT | Trajectron++ | PECNet | AgentFormer | LaneGCN | HOME |
|------|----------|--------------|--------|-------------|---------|------|
| **地图表示** | 栅格 (18通道) | 栅格 (语义) | 无 | 栅格 (可选) | 矢量图 | 分层图 |
| **地图覆盖** | 100km×100km | 局部 (~100m) | N/A | 局部 | 车道级 | 区域级 |
| **历史编码** | LSTM | LSTM | MLP | Transformer | 1D CNN | GRU |
| **交互建模** | 无 | Dynamic GNN | 社交池化 | Attention | Multi-GCN | 分层图 |
| **目标处理** | 显式分类器 | 隐式 (CVAE) | 显式 (VAE) | 隐式 | 隐式 | 显式区域 |
| **多模态** | 单模态 | ✓ (CVAE) | ✓ (VAE) | ✓ (采样) | ✓ (多头) | ✓ (条件) |
| **环境约束** | 强 (地形) | 中 (语义) | 弱 (无) | 中 (可选) | 强 (车道) | 强 (区域) |
| **计算复杂度** | 中 | 高 (GNN) | 低 (MLP) | 高 (Transformer) | 高 (GCN) | 高 (图) |
| **适用场景** | 越野/军事 | 城市交通 | 行人预测 | 通用 | 自动驾驶 | 城市规划 |

---

## 4. TerraTNT的独特优势与劣势

### 4.1 独特优势
1. **大尺度环境建模**: 100km×100km覆盖范围，适合长时域预测
2. **地形感知**: 包含DEM、坡度、坡向等地形特征
3. **土地利用约束**: LULC one-hot编码提供环境类型信息
4. **目标驱动**: 显式建模终点选择过程
5. **层次化解码**: Waypoint + Step两层结构

### 4.2 主要劣势
1. **无交互建模**: 单agent预测，忽略多agent交互
2. **单模态输出**: 缺乏不确定性建模
3. **栅格地图限制**: 
   - 分辨率固定 (100km/128≈781m/pixel)
   - 无法表达拓扑关系
   - 旋转/缩放不变性差
4. **环境特征利用不充分**:
   - CNN编码器容量可能不足
   - 局部环境注入机制未充分训练
5. **Goal vector注入过强**: 导致直线化偏差

---

## 5. 改进方向建议

### 5.1 短期改进 (架构微调)
1. **Goal vector注入优化**:
   - 添加可学习缩放因子 `goal_vec_scale`
   - 使用MLP投影 `goal_vec_proj`
   - 实施schedule逐步增强goal引导

2. **Waypoint stride调整**:
   - 减小stride (36→12或18)，增加中间路点数量
   - 添加waypoint-level环境采样

3. **局部环境特征增强**:
   - 确保`env_local_scale2`参数被正确训练
   - 增加局部地图分辨率 (10km→20km)
   - 在每个step采样局部环境特征

4. **损失函数改进**:
   - 增加`curvature_consistency_loss`权重
   - 添加`trajectory_smoothness_loss`
   - 添加`environment_alignment_loss` (轨迹与道路对齐)

### 5.2 中期改进 (架构扩展)
1. **多模态预测**:
   - 引入VAE或Flow-based模型
   - 为每个候选目标生成多条轨迹
   - 添加轨迹多样性损失

2. **注意力机制**:
   - 在环境编码器中添加spatial attention
   - 在解码器中添加temporal attention
   - 学习动态关注重要环境区域

3. **双尺度融合优化**:
   - 改进全局-局部特征融合方式
   - 添加跨尺度注意力
   - 动态调整局部窗口大小

4. **历史编码增强**:
   - 增加历史编码器维度
   - 添加速度、加速度特征
   - 使用Transformer替代LSTM

### 5.3 长期改进 (架构重构)
1. **混合地图表示**:
   - 栅格地图 + 矢量道路图
   - 学习道路拓扑结构
   - 结合LaneGCN的车道图思想

2. **分层环境建模**:
   - 借鉴HOME的分层图结构
   - 区域级 + 对象级 + 像素级
   - 多尺度环境推理

3. **交互建模**:
   - 添加多agent场景支持
   - 引入图神经网络
   - 建模agent-environment交互

4. **端到端学习**:
   - 可微分地图渲染
   - 联合优化地图编码和轨迹生成
   - 元学习快速适应新环境

---

## 6. 实验验证计划

### 6.1 消融实验
- [ ] Goal vector scale ablation (0.0, 0.5, 1.0, 2.0)
- [ ] Waypoint stride ablation (12, 18, 24, 36)
- [ ] 环境特征消融 (zero_all, shuffle_all, road_only)
- [ ] 局部环境注入消融 (with/without local scale)

### 6.2 Baseline对比
- [ ] PECNet (纯轨迹驱动)
- [ ] Constant Velocity (简单基线)
- [ ] Line-to-goal (目标驱动基线)
- [ ] 人类驾驶员 (OORD数据集)

### 6.3 指标评估
- [ ] ADE / FDE (精度)
- [ ] Straightness fraction (真实性)
- [ ] Environment alignment (环境约束)
- [ ] Diversity (多模态，如果实现)

---

## 7. 参考文献

1. Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data (ECCV 2020)
2. PECNet: It Is Not the Journey but the Destination (ECCV 2020)
3. AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting (ICCV 2021)
4. LaneGCN: Learning Lane Graph Representations for Motion Forecasting (ECCV 2020)
5. HOME: Heatmap Output for Future Motion Estimation (ITSC 2021)

---

**最后更新**: 2026-01-20
**分析者**: Cascade AI Assistant
