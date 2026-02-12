# 工作总结 - 2026-01-20

## 执行概览

按照你的要求，我已完成了所有可以独立完成的工作，包括问题诊断、架构分析、代码优化和训练任务启动。

---

## 已完成工作

### 1. 可视化工具改进 ✅

**实现功能**:
- Sample selection: 支持按`--val_index`或`--select_file + --select_sample_idx`选择特定样本
- Manifest生成: 自动生成`manifest.jsonl`记录每个样本的identity和指标
- 样本身份嵌入: 图表标题和控制台输出包含`(file_path, sample_idx)`信息

**验证结果**:
- 功能测试通过
- Manifest文件格式正确
- 支持跨实验对比同一样本

**文件修改**:
- `scripts/visualize_terratnt_predictions.py`
- `scripts/train_terratnt_10s.py` (FASDataset返回sample identity)

---

### 2. 问题诊断与定量分析 ✅

**直线化偏差分析**:
- 全样本（n=50）: 预测直线化88% vs GT 56%
- 弯曲样本（n=20）: 预测直线化70% vs GT 0%
- **关键发现**: TerraTNT ADE≈2990m vs 直线基线≈3061m（几乎相同）

**环境特征利用评估**:
- 环境消融实验（zero_all）: 性能完全无变化（ADE差异<1%）
- **结论**: 环境特征未被有效利用

**根因诊断**:
1. 🔴 Goal vector注入过强（无可学习缩放）
2. 🟡 Waypoint stride过大（36步，中间路点太少）
3. 🔴 局部环境特征未训练（env_local_scale2缺失）
4. 🟡 训练策略需要优化

---

### 3. 主流模型架构对比分析 ✅

**完成文档**: `docs/trajectory_models_analysis.md`

**对比模型**:
- Trajectron++ (动态图神经网络)
- PECNet (终点条件化预测)
- AgentFormer (Transformer + agent-aware attention)
- LaneGCN (车道图卷积)
- HOME (分层对象-区域图)

**TerraTNT优势**:
- 大尺度环境建模（100km×100km）
- 地形感知（DEM、坡度、坡向）
- 目标驱动设计

**TerraTNT劣势**:
- 无交互建模
- 单模态输出
- Goal vector注入过强
- 环境特征利用不足

---

### 4. Goal Vector优化实现 ✅

**代码修改**: `models/terratnt.py`

**实现内容**:
```python
# 添加可学习缩放因子
self.goal_vec_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

# 在forward中应用缩放
scaled_goal_embed = goal_embed * self.goal_vec_scale
```

**优化策略**:
- 初始值: 0.5（降低目标引导强度）
- Schedule: 0.5 → 1.0 over 15 epochs
- 配合waypoint_stride减小（36 → 18）

**预期效果**:
- ADE降低20-30%
- 直线化比例降低到70%以下
- 环境特征开始被利用

---

### 5. 代码Bug修复 ✅

**问题**: `AttributeError: 'tuple' object has no attribute 'size'`

**原因**: decoder的`env_features`解包逻辑只支持2/3元组，但实际传入4元组

**修复**:
```python
# 添加对4元组的支持
elif len(env_features) == 4:
    env_global, env_tokens, env_spatial, env_spatial_local = env_features
```

---

### 6. 训练任务启动 ✅

**训练配置**:
- Region: bohemian_forest
- Phase: fas3
- Sample fraction: 0.2
- Epochs: 30
- Batch size: 16
- Learning rate: 0.0001
- **Goal vec scale**: 0.5 → 1.0 (15 epochs)
- **Waypoint stride**: 18 (vs 36 baseline)

**训练状态**:
- 进程ID: 38900 (主进程)
- 日志文件: `runs/train_goal_scale_opt_v4.log`
- 状态: 正在运行（数据加载阶段）
- 预计完成时间: 2-4小时

**输出目录**: `runs/terratnt_fas3_10s/terratnt_goal_scale_opt_20260120_214624/`

---

### 7. 实验文档生成 ✅

**生成文档**:
1. `docs/trajectory_models_analysis.md` - 主流模型架构对比
2. `docs/experiment_report_20260120.md` - 实验诊断与改进方案
3. `docs/experiment_progress_20260120.md` - 实验进展跟踪
4. `docs/work_summary_20260120.md` - 本文档

**训练脚本**:
- `scripts/train_goal_scale_optimized.sh` - 优化训练脚本

---

## 待完成工作（需等待训练完成）

### 8. 评估优化效果

**评估命令**（训练完成后执行）:
```bash
# 找到最新checkpoint
CKPT=$(ls -t runs/terratnt_fas3_10s/terratnt_goal_scale_opt_*/best_model.pth | head -1)

# 弯曲轨迹评估
conda run -n torch-sm120 python scripts/visualize_terratnt_predictions.py \
  --checkpoint ${CKPT} \
  --region bohemian_forest \
  --num_samples 50 \
  --only_curved \
  --curvature_threshold 0.08 \
  --output_dir ${CKPT%/*}/viz_curved

# 环境消融实验
conda run -n torch-sm120 python scripts/visualize_terratnt_predictions.py \
  --checkpoint ${CKPT} \
  --region bohemian_forest \
  --num_samples 50 \
  --env_ablation zero_all \
  --stats_only \
  --output_dir ${CKPT%/*}/viz_ablation
```

**对比指标**:
- ADE / FDE
- 直线化比例 (straight_frac)
- 弯曲度 (mean_norm_dev)
- 环境消融性能下降

---

## 改进效果预期

### 短期目标（本次优化）
- ✅ Goal vector可学习缩放
- ✅ Waypoint stride减小
- 📊 ADE目标: <2500m (vs baseline 2990m)
- 📊 直线化目标: <70% (vs baseline 70%)
- 📊 环境消融影响: >10% (vs baseline 0%)

### 中期改进方向
- 修复env_local_scale2参数
- 增加curvature_consistency_loss权重
- 实现多模态预测（VAE）
- 添加spatial attention机制

### 长期改进方向
- 混合地图表示（栅格+矢量）
- 分层环境建模
- 交互建模（多agent）
- 端到端学习

---

## 监控与下一步

### 检查训练进度
```bash
# 查看训练日志
tail -f runs/train_goal_scale_opt_v4.log

# 查看GPU使用
nvidia-smi

# 查看进程状态
ps aux | grep train_terratnt
```

### 训练完成后
1. 检查训练日志，确认收敛情况
2. 运行可视化评估新模型性能
3. 对比baseline和优化模型的指标
4. 生成对比报告和可视化图表
5. 根据结果决定下一步改进方向

---

## 技术要点总结

### 问题本质
TerraTNT的直线化偏差和环境特征利用失败，根本原因是**goal vector注入过强**，导致模型过度依赖"直线到目标"策略，忽略了环境约束和轨迹弯曲模式。

### 解决方案
通过添加可学习的`goal_vec_scale`参数，让模型自主学习最优的目标引导强度，平衡目标导向和环境感知。

### 创新点
- 首次在TerraTNT中引入可学习的goal vector缩放机制
- 使用schedule逐步增强目标引导，避免训练初期过度依赖
- 配合waypoint stride优化，提升轨迹细粒度建模能力

---

## 文件清单

### 修改的代码文件
- `models/terratnt.py` (添加goal_vec_scale + 修复bug)
- `scripts/visualize_terratnt_predictions.py` (sample selection + manifest)
- `scripts/train_terratnt_10s.py` (返回sample identity)

### 新增的文档文件
- `docs/trajectory_models_analysis.md`
- `docs/experiment_report_20260120.md`
- `docs/experiment_progress_20260120.md`
- `docs/work_summary_20260120.md`

### 新增的脚本文件
- `scripts/train_goal_scale_optimized.sh`

### 实验输出目录
- `viz_test/` - 可视化测试
- `viz_test_specific/` - 特定样本可视化
- `viz_curved_analysis/` - 弯曲轨迹分析
- `viz_ablation_zero/` - 环境消融实验
- `runs/terratnt_fas3_10s/terratnt_goal_scale_opt_20260120_214624/` - 当前训练

---

**工作完成时间**: 2026-01-20 21:50  
**训练状态**: 进行中  
**下一步**: 等待训练完成，评估优化效果
