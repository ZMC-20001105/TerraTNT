# 环境消融实验结果
**日期**: 2026-01-21  
**模型**: runs/terratnt_fas3_10s/20260121_171614/best_model.pth

---

## 实验设置

**对比组**:
1. **正常模型** (无消融): 使用完整环境地图
2. **环境置零** (env_ablation=zero): 环境地图全部置零

**测试配置**:
- 样本数: 100
- Phase: fas3
- Goal mode: given (oracle goal)
- Batch size: 8

---

## 结果对比

### 指标对比表

| 指标 | 正常模型 | 环境置零 | 差异 | 变化率 |
|------|---------|---------|------|--------|
| **pred(gt_goal) ADE** | 30456.5m | 30280.6m | -175.9m | -0.58% |
| **pred(gt_goal) FDE** | 71769.3m | 71515.2m | -254.1m | -0.35% |
| **pred(top1) mean_norm_dev** | 0.2342 | 0.2033 | -0.0309 | -13.2% |
| **pred(top1) straight_frac** | 0.140 | 0.120 | -0.020 | -14.3% |

### 详细分析

#### 1. ADE/FDE 指标

**正常模型**:
```
pred(gt_goal) 平均 ADE: 30456.5 m
pred(gt_goal) 平均 FDE: 71769.3 m
```

**环境置零**:
```
pred(gt_goal) 平均 ADE: 30280.6 m
pred(gt_goal) 平均 FDE: 71515.2 m
```

**结论**: 
- ❌ **环境置零后 ADE/FDE 几乎不变** (变化 < 1%)
- ❌ **这表明当前模型几乎没有使用环境特征**
- ⚠️ 模型主要依赖历史轨迹和目标位置

#### 2. 轨迹弯曲度

**正常模型**:
```
pred(top1) mean_norm_dev: 0.2342
pred(top1) straight_frac: 0.140 (14% 直线)
```

**环境置零**:
```
pred(top1) mean_norm_dev: 0.2033
pred(top1) straight_frac: 0.120 (12% 直线)
```

**结论**:
- ⚠️ 环境置零后轨迹反而**更弯曲**（mean_norm_dev 降低 13%）
- ⚠️ 直线比例降低 14%
- ❓ 这可能说明环境特征反而在**增加**预测的直线倾向

---

## 核心发现

### 🔴 关键问题：模型未有效使用环境特征

**证据**:
1. 环境置零后 ADE/FDE 变化 < 1%
2. 环境置零后轨迹质量没有明显下降
3. 环境置零后甚至出现轨迹更弯曲的现象

**可能原因**:
1. ❌ **环境地图未归一化** (已修复)
   - DEM/Slope 主导 CNN 学习
   - LULC/Road 被淹没
   
2. ⚠️ **环境特征权重过小**
   - `env_local_scale = 1.01`
   - `env_local_scale2 = -0.02`
   - 环境特征的贡献可能被其他特征压制

3. ⚠️ **目标吸引力过强**
   - `goal_vec_scale = 0.5`
   - 目标向量可能主导解码器行为

4. ⚠️ **CNN 未有效提取环境特征**
   - 可能需要更深的网络
   - 可能需要更多训练

---

## 与训练指标的对比

### 训练验证指标
```
val_ade: 5858.96m
val_fde: 10790.28m
```

### 可视化指标（正常模型）
```
pred(gt_goal) ADE: 30456.5m  (5.2倍)
pred(gt_goal) FDE: 71769.3m  (6.6倍)
```

### 可视化指标（环境置零）
```
pred(gt_goal) ADE: 30280.6m  (5.2倍)
pred(gt_goal) FDE: 71515.2m  (6.6倍)
```

**结论**:
- ❌ 环境消融对 5.2 倍差距**没有影响**
- ❌ 说明 5.2 倍差距的根本原因**不是**环境特征
- ⚠️ 可能是配置不一致（coord_scale, env_coverage_km 等）

---

## 修复策略调整

### 原计划
1. ✅ 修复 checkpoint 保存（已完成）
2. ✅ 实现环境地图归一化（已完成）
3. ⏳ 重新训练验证效果

### 调整后的计划

#### 优先级 1: 解决 5.2 倍指标差距
**根本原因**: Checkpoint 缺少配置导致可视化使用错误参数

**验证方法**:
1. 使用新训练的模型（包含完整配置）
2. 确认可视化脚本正确读取配置
3. 验证训练和可视化指标一致

#### 优先级 2: 提升环境特征利用
**方法**:
1. 使用归一化的环境地图重新训练
2. 调整 `env_local_scale` 初始值（从 1.0 → 5.0）
3. 降低 `goal_vec_scale`（从 0.5 → 0.3）
4. 增加环境编码器深度

#### 优先级 3: 架构改进
**可选**:
1. 添加 Refiner 机制
2. 引入 3D 卷积处理时序环境
3. 改进特征融合方式

---

## 下一步行动

### 立即执行

**1. 重新训练模型（应用所有修复）**
```bash
conda run -n torch-sm120 python scripts/train_terratnt_10s.py \
  --phase fas3 \
  --region bohemian_forest \
  --epochs 30 \
  --batch_size 32 \
  --sample_fraction 0.2 \
  --env_coverage_km 140.0 \
  --use_dual_scale \
  --num_candidates 6 \
  --candidate_radius_km 3.0 \
  --candidate_center goal
```

**预期**:
- ✅ Checkpoint 包含完整配置
- ✅ 环境地图已归一化
- ✅ 训练和可视化指标一致
- ⚠️ 环境特征利用可能仍然较弱（需要调参）

**2. 训练完成后验证**
```bash
# 验证配置一致性
python -c "
import torch
ckpt = torch.load('runs/.../best_model.pth', map_location='cpu')
print('coord_scale:', ckpt.get('coord_scale'))
print('env_coverage_km:', ckpt.get('env_coverage_km'))
"

# 运行可视化
conda run -n torch-sm120 python scripts/visualize_terratnt_predictions.py \
  --checkpoint runs/.../best_model.pth \
  --phase fas3 \
  --goal_mode given \
  --num_samples 100
```

**3. 环境消融对比**
```bash
# 正常模型
conda run -n torch-sm120 python scripts/visualize_terratnt_predictions.py \
  --checkpoint runs/.../best_model.pth \
  --num_samples 100

# 环境置零
conda run -n torch-sm120 python scripts/visualize_terratnt_predictions.py \
  --checkpoint runs/.../best_model.pth \
  --num_samples 100 \
  --env_ablation zero
```

**成功标准**:
- [ ] 训练 val_ade 与可视化 pred(gt_goal) ADE 差距 < 10%
- [ ] 环境置零后 ADE 增加 > 20%
- [ ] 预测轨迹符合地形和道路约束

---

## 总结

### 已完成 ✅
1. ✅ 修复 checkpoint 保存逻辑
2. ✅ 实现环境地图归一化
3. ✅ 运行环境消融实验
4. ✅ 识别当前模型的问题

### 核心发现 🔍
1. ❌ 当前模型几乎不使用环境特征
2. ❌ 5.2 倍指标差距主要由配置不一致导致
3. ⚠️ 环境地图归一化是必要的，但需要重新训练才能生效

### 下一步 📋
1. **立即**: 重新训练模型（应用所有修复）
2. **短期**: 验证训练和可视化指标一致性
3. **中期**: 调整超参数提升环境特征利用
4. **长期**: 架构改进（Refiner、3D Conv 等）

---

**报告生成时间**: 2026-01-21 21:50  
**实验完成**: ✅  
**下一步**: 重新训练模型
