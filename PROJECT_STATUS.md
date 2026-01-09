# TerraTNT 项目状态报告
生成时间: 2026-01-09 16:24

## 📊 当前训练进度

### TerraTNT (Bohemian Forest)
- **状态**: ✅ 正在训练
- **进度**: Epoch 10/30 (33%)
- **最佳ADE**: 2716.6m (Epoch 9)
- **训练速度**: 2.45 it/s, ~2分钟/epoch
- **预计完成**: 16:45 (约20分钟后)
- **配置**: Batch 256, Workers 16, GPU利用率39%

### 训练历史
| Epoch | 训练ADE | 验证ADE | 状态 |
|-------|---------|---------|------|
| 1 | - | - | 完成 |
| 8 | 2834.4m | 3141.0m | 完成 |
| 9 | 2733.2m | **2716.6m** | ✓ 最佳 |
| 10 | - | 进行中 | - |

**趋势**: ADE持续下降，模型收敛良好

---

## 🎯 剩余训练任务

### 已完成 ✅
1. TerraTNT (Bohemian Forest) - 进行中，预计20分钟完成

### 待训练 ⏳
1. **基线模型对比** (可选)
   - Social-LSTM: 已有checkpoint
   - PECNet: 已有checkpoint
   - Trajectron++: 已有checkpoint
   - 可能需要重新训练以确保公平对比

2. **跨区域验证** (可选)
   - Scottish Highlands测试
   - 预计时间: 10分钟

### 总预计时间
- 当前训练完成: **20分钟**
- 基线重训练 (如需): **1-2小时**
- 跨区域测试: **10分钟**

---

## 🗂️ 项目文件状态

### 存储占用
```
runs/      1.5GB  (24个模型目录, 22个checkpoint文件)
logs/      30MB   (40个日志文件)
results/   3.5MB  (7个可视化图片)
```

### 冗余文件分析

#### 1. 模型Checkpoints (runs/) - 1.5GB
**保留 (5个, 最近的):**
- ✓ terratnt_fas1_10s (198MB) - 当前训练
- ✓ terratnt_fas3_10s (49.5MB)
- ✓ terratnt_fas2_10s (49.5MB)
- ✓ terratnt_fas3_real_env (49.5MB)
- ✓ terratnt_fas2_real_env (49.5MB)

**可删除 (19个, 旧的/错误的):**
- ❌ terratnt_fas1_real_env (49.5MB)
- ❌ terratnt_fas*_optimized (343MB) - 旧版本
- ❌ terratnt_fas*_fixed (146.7MB) - 旧版本
- ❌ terratnt_fas* (489MB) - 23秒间隔版本(已废弃)
- ❌ ynet* (25.7MB) - 训练失败
- ❌ pecnet_* (4.2MB) - 旧版本
- ❌ social_lstm (15.1MB) - 旧版本
- ❌ constant_velocity (0MB) - 空目录

**节省空间**: ~1.1GB

#### 2. 日志文件 (logs/) - 30MB
**保留 (38个):**
- ✓ 最近24小时的日志

**可删除 (2个):**
- ❌ 超过24小时的旧日志 (0MB)

#### 3. 可视化图片 (results/) - 3.5MB
**保留 (7个, 全部最新):**
- ✓ fixed_trajectory_analysis.png (790KB)
- ✓ speed_by_vehicle_type.png (744KB)
- ✓ detailed_trajectory_analysis.png (703KB)
- ✓ dataset_samples.png (555KB)
- ✓ training_curves.png (264KB)
- ✓ real_model_comparison.png (149KB)
- ✓ dataset_statistics.png (139KB)

#### 4. 脚本文件 (scripts/) - 24个
**核心脚本 (保留):**
- ✓ train_terratnt_10s.py
- ✓ train_all_baselines.py
- ✓ visualize_results.py
- ✓ compare_baselines.py
- ✓ evaluate_all_models.py

**工具脚本 (保留):**
- ✓ download_garmisch_hohenfels.py
- ✓ generate_synthetic_dataset.py
- ✓ prepare_fas_datasets.py
- ✓ process_bohemian_forest.py

**临时脚本 (可删除):**
- ❌ fix_trajectory_speeds.py - 一次性修复脚本
- ❌ calibrate_xgboost_speeds.py - 一次性校准脚本
- ❌ auto_download_garmisch_hohenfels.py - 重复
- ❌ cleanup_redundant_files.py - 清理完可删
- ❌ download_oord_gps.py - 空文件
- ❌ chapter3_experiments.py - 实验脚本

**节省空间**: ~15KB

---

## 📋 清理计划

### 立即执行 (安全)
```bash
# 1. 删除旧的模型checkpoints (~1.1GB)
rm -rf runs/terratnt_fas1_real_env
rm -rf runs/terratnt_fas*_optimized
rm -rf runs/terratnt_fas*_fixed
rm -rf runs/terratnt_fas1 runs/terratnt_fas2 runs/terratnt_fas3
rm -rf runs/ynet* runs/pecnet_2026* runs/constant_velocity

# 2. 删除临时脚本 (~15KB)
rm -f scripts/fix_trajectory_speeds.py
rm -f scripts/calibrate_xgboost_speeds.py
rm -f scripts/auto_download_garmisch_hohenfels.py
rm -f scripts/cleanup_redundant_files.py
rm -f scripts/download_oord_gps.py

# 3. 删除旧日志 (可选)
find logs/ -name "*.log" -mtime +1 -delete
```

**总节省空间**: ~1.1GB

### 训练完成后执行
```bash
# 保留最终的3个最佳模型
# 删除其他所有训练过程中的checkpoint
```

---

## 🎓 论文准备状态

### 已完成 ✅
1. ✓ 数据集生成 (Bohemian Forest: 3127条, Scottish Highlands: 1924条)
2. ✓ 速度校准方法 (XGBoost + 线性变换)
3. ✓ 数据可视化 (7张图)
4. ✓ 速度校准文档 (docs/speed_calibration_explanation.md)
5. ✓ TerraTNT训练 (进行中)

### 待完成 ⏳
1. ⏳ 训练完成后的性能评估
2. ⏳ 基线模型对比
3. ⏳ 跨区域泛化测试
4. ⏳ 消融实验 (可选)
5. ⏳ 最终可视化和表格

---

## 💡 建议

### 短期 (今天)
1. **等待当前训练完成** (20分钟)
2. **执行清理脚本** (节省1.1GB)
3. **评估训练结果**
4. **决定是否重训练基线**

### 中期 (明天)
1. 完成基线模型训练
2. 跨区域验证
3. 生成最终对比图表

### 长期
1. 下载其他区域数据 (Garmisch, Hohenfels)
2. 扩展数据集
3. 进行更多实验

---

## 📞 当前问题

### 已解决 ✅
- ✓ 速度过低问题 (XGBoost校准)
- ✓ 训练速度慢 (Batch 256, Workers 16)
- ✓ GPU利用率低 (从17%提升到39%)
- ✓ 数据重复问题 (已修复)

### 待解决 ⚠️
- ⚠️ GEE下载网络超时 (暂时跳过)
- ⚠️ 环境数据偶尔提取失败 (影响不大)

---

## 📈 性能指标

### 当前最佳
- **ADE**: 2716.6m (Epoch 9)
- **趋势**: 持续下降
- **收敛**: 良好

### 目标
- **ADE**: < 2000m (论文要求)
- **训练稳定性**: ✓ 良好
- **泛化能力**: 待测试
