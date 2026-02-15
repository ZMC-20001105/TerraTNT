# 论文图表材料全面审计报告 (v3)

> 审计时间: 2026-02-15
> 审计范围: 论文全部图表 (第1-5章) — 逐图审查数据真实性、视觉质量、与论文描述的一致性
> 统一输出: `outputs/paper_final/`
> 统一脚本: `scripts/generate_all_paper_figures.py`

---

## 一、总览统计

| 类别 | 数量 |
|------|------|
| 论文引用的图总数 | ~52个(含子图) |
| 第1-2章引用图(来自文献) | 14个 — 无需自制 |
| 第3章需自制图 | ~12个 |
| 第4章需自制图 | ~26个(含子图) |
| **已完成且数据可靠** | **15个** |
| **已完成但需小幅修改** | **8个** |
| **数据虚假/硬编码，必须重做** | **5个** |
| **完全缺失，需要新制作** | **~24个** |

---

## 二、第3章图表状态 (轨迹生成)

### 已完成 (3个)

| 文件 | 对应论文 | 数据源 | 质量评价 |
|------|---------|--------|---------|
| `fig3_kfold_validation.pdf` | 表3.9对应图 | experiment_results.json | ✅ 数据真实，三panel(R²/RMSE/MAPE)清晰 |
| `fig3_loocv.pdf` | 图3.8 | experiment_results.json | ✅ 柱状图+平均线+人类基线，信息完整 |
| `fig3_feature_importance.pdf` | 图3.11(XGBoost版) | experiment_results.json | ⚠️ 特征名为英文，建议改中文 |

### 已有但需修改 (1个)

| 文件 | 问题 | 建议 |
|------|------|------|
| `fig3_shap_importance.png` | 英文标题"Feature Importance (SHAP)"；只有柱状图，缺少论文提到的SHAP汇总图(图3.12) | 改中文标题；补充SHAP beeswarm图 |

### 完全缺失 (8个) — 第3章最大缺口

| 论文图号 | 内容 | 难度 | 所需数据 |
|---------|------|------|---------|
| 图3.1(流程) | 轨迹生成方法整体流程图 | 中 | 手绘/绘图工具 |
| 图3.1(地图) | Bellmouth段轨迹采集区域 | 低 | OORD GPS数据+底图 |
| 图3.2(速度) | OORD数据集轨迹速度曲线 | 低 | OORD原始数据 |
| 图3.2(成本) | 成本地图生成流程(DEM/LULC/Slope/Cost四子图) | 中 | utm_grid环境数据 |
| 图3.3 | 分层A*规划示例(粗+精) | 中 | 规划结果数据 |
| 图3.4 | 轨迹生成结果(速度着色) | 低 | complete_dataset_10s轨迹 |
| 图3.5 | 不同车辆类型轨迹对比 | 低 | 同上 |
| 图3.6(意图) | 不同战术意图路径对比 | 低 | 同上 |
| 图3.6(速度) | 真实速度vs预测速度曲线 | 低 | 速度预测结果 |
| 图3.12 | SHAP汇总图(beeswarm) | 低 | 需重新运行SHAP |

**结论: 第3章图表严重不足，12个图中只完成了3个，缺失率75%。**

---

## 三、第4章图表状态 (预测模型)

### ✅ 已完成且数据可靠 (8个)

| 文件 | 论文图号 | 数据源 | 质量评价 |
|------|---------|--------|---------|
| `fig4_6_training_curves.pdf` | 图4.6 | 真实训练日志 | ✅ 双panel(ADE+Loss)，对数坐标合理 |
| `fig4_7_phase_comparison.pdf` | 图4.7柱状图 | phase_v2_with_faithful | ✅ 三阶段对比，数值标注清晰 |
| `fig4_8_ablation.pdf` | 图4.14 | ablation_results.json | ✅ 三panel消融，数值完整 |
| `fig4_10_phase_heatmap.pdf` | 图4.10 | phase_v2_with_faithful | ✅ 热力图颜色映射合理 |
| `fig4_9_waypoint_ablation.pdf` | 图4.13 | waypoint_ablation | ✅ ADE/FDE双线图趋势清晰 |
| `fig_example_1~6.pdf` | 图4.7c | 真实预测结果 | ✅ DEM背景+多模型轨迹对比 |

### ⚠️ 已完成但需修改 (8个)

| 文件 | 论文图号 | 问题 | 建议操作 |
|------|---------|------|---------|
| `fig_v6r_architecture.pdf` | 图4.1 | 全英文标注；显示了Waypoint/Spatial等论文未描述的组件 | 改中文；简化或在论文中补充描述 |
| `fig4_box_phase1.pdf` | 图4.7a | 模型标签为LSTM+Env+Goal/Seq2Seq+Attn(自建baseline)，非YNet/PECNet | 与论文正文统一 |
| `fig4_time_phase1.pdf` | 图4.7b | 同上 | 同上 |
| `fig4_box_phase2.pdf` | 图4.8a | 同上 | 同上 |
| `fig4_time_phase2.pdf` | 图4.8b | 同上 | 同上 |
| `fig4_box_phase3.pdf` | 图4.9a | 同上 | 同上 |
| `fig4_time_phase3.pdf` | 图4.9b | 同上 | 同上 |

**箱线图/时间趋势图说明**: `paper_final/`中的这些图使用了v5 per-sample数据，模型标签是诚实的（没有伪装成YNet/PECNet），但与论文正文声称对比的YNet/PECNet不一致。建议方案：
1. **方案A(推荐)**: 论文正文改为使用这些自建baseline名称，因为它们有per-sample数据可画箱线图
2. **方案B**: 重新用真正的YNet/PECNet生成per-sample数据（工作量大）

### 🔴 数据虚假，必须重做 (5个)

| 文件 | 论文图号 | 问题 | 操作 |
|------|---------|------|------|
| `fig4_9_candidate_sensitivity` (旧) | 图4.11 | 数据完全硬编码虚构 | 用K敏感性实验真实数据重画 |
| `fig4_25_region_bars` (旧) | 图4.25 | 硬编码；新版生成全0空白 | 完成多区域实验后重画 |
| `fig4_26_cross_matrix` (旧) | 图4.26 | 4×4矩阵16个数值全部硬编码 | 完成跨区域实验后重画 |
| `fig4_27_gen_loss` (旧) | 图4.27 | 硬编码 | 完成跨区域实验后重画 |
| `fig4_25_26_27_PLACEHOLDER` (新) | 图4.25-27 | 红字占位图 | 替换为真实数据 |

### ❌ 完全缺失 (约10个)

| 论文图号 | 内容 | 前置条件 |
|---------|------|---------|
| 图4.2 | 环境编码器网络结构图 | 手绘/绘图工具 |
| 图4.3 | 历史轨迹编码器结构图 | 同上 |
| 图4.4 | 目标分类器结构图 | 同上 |
| 图4.5 | 轨迹生成器结构图 | 同上 |
| 图4.7d | Phase 1按轨迹特征分层矩阵 | 需编写分析脚本 |
| 图4.8c | Phase 2轨迹可视化 | 需Phase 2预测结果 |
| 图4.9b | Phase 3不同候选范围对比 | 需运行候选范围实验 |
| 图4.12 | 观测长度敏感性 | 已有实验数据，需绘图 |
| 图4.19 | 四区域地形DEM对比 | 需从utm_grid数据绘制 |
| 图4.20 | 四区域土地覆盖对比 | 需从LULC数据绘制 |
| 图4.24 | 各区域训练曲线 | 需完成多区域训练 |

---

## 四、关键发现与建议

### 发现1: 第3章图表是最大缺口
第3章引用了约12个图，但只完成了3个(25%)。这些图大多可以从已有数据直接绘制（轨迹数据、环境数据、速度预测结果都已存在），工作量不大但数量多。

### 发现2: 图4.11(K敏感性)可以立即用真实数据重画
K敏感性实验已完成(K=6~200, ADE差异<1%)，只需编写绘图代码。但注意：实际结果显示K对ADE几乎无影响，与论文描述的趋势不同。

### 发现3: 图4.12(观测长度)可以立即用真实数据绘制
观测长度实验已完成(3~15min, ADE几乎无差异)。同样，实际结果与论文声称的"9分钟最优"不同。

### 发现4: 跨区域图表(图4.25-27)依赖未完成的多区域实验
需要先完成: Carpathians轨迹生成 → 4区域独立训练 → 跨区域评估。这是最大的阻塞项。

### 发现5: 架构图(图4.1)与论文描述不一致
当前架构图展示了V6R的完整架构(含Waypoint Predictor, Spatial Env Sampling, Goal Dropout等)，但论文只描述了简化版(四层CNN+双层LSTM+MLP分类器+LSTM解码器)。需要统一。

---

## 五、优先级排序

### P0 — 可立即执行 (已有数据，只需绘图)

1. **图4.11 K敏感性**: 用真实实验数据替换虚构图
2. **图4.12 观测长度敏感性**: 用真实实验数据绘制
3. **图3.4/3.5/3.6**: 从complete_dataset_10s轨迹数据绘制
4. **图3.2(成本地图)**: 从utm_grid环境数据绘制DEM/LULC/Slope/Cost四子图
5. **图3.6(速度)**: 从速度预测结果绘制真实vs预测曲线
6. **fig3_shap/fig3_feature_importance**: 改中文标题/特征名

### P1 — 需要少量额外工作

7. **图4.1架构图**: 改中文标注，与论文描述统一
8. **箱线图/时间趋势图**: 决定方案A或B后统一模型标签
9. **图3.1(流程图)**: 需要用绘图工具制作
10. **图3.3(A*规划)**: 需要从规划中间结果绘制

### P2 — 依赖未完成实验

11. **图4.19/4.20**: 四区域地形/LULC对比(需Carpathians数据)
12. **图4.24**: 各区域训练曲线(需完成多区域训练)
13. **图4.25/4.26/4.27**: 跨区域泛化(需完成全部跨区域实验)

### P3 — 可选/低优先级

14. **图4.2-4.5**: 子模块结构图(可用文字描述替代)
15. **图4.7d**: 特征分层矩阵(增强分析，非必须)
16. **图4.8c/4.9b**: Phase 2/3可视化子图

---

## 六、文件位置汇总

### 可靠的评估数据
- `outputs/evaluation/phase_v2_with_faithful/phase_v2_results.json` — Phase V2 (13模型×7Phase)
- `outputs/evaluation/ablation/ablation_results.json` — 消融实验
- `outputs/evaluation/control_variables/waypoint_ablation_results.json` — Waypoint消融
- `_trash/results/chapter3/experiment_results.json` — 第3章速度预测

### 可靠的图表 (outputs/paper_final/)
- `fig4_6_training_curves.pdf` — 训练曲线 ✅
- `fig4_7_phase_comparison.pdf` — Phase对比柱状图 ✅
- `fig4_8_ablation.pdf` — 消融实验 ✅
- `fig4_10_phase_heatmap.pdf` — 三阶段热力图 ✅
- `fig4_9_waypoint_ablation.pdf` — Waypoint消融 ✅
- `fig3_kfold_validation.pdf` — K-fold验证 ✅
- `fig3_loocv.pdf` — LOOCV ✅
- `fig_example_1~6.pdf` — 轨迹预测可视化 ✅

### 需要重做的图表
- `fig4_9_candidate_sensitivity` — 虚构数据 🔴
- `fig4_25/26/27` — 硬编码/占位 🔴

### 生成脚本
- `scripts/generate_all_paper_figures.py` — 统一脚本(推荐)
- `scripts/generate_paper_ch4_figures.py` — 旧脚本(有虚假数据问题，勿用)
