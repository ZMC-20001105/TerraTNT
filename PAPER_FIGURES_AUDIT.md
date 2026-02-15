# 论文图表材料全面审计报告 (v3)

> 审计时间: 2026-02-15 (全面更新)
> 审计范围: 论文全部图表 (第1-5章) — 逐图审查数据真实性、视觉质量、与论文描述的一致性
> 统一输出: `outputs/paper_final/`
> 统一脚本: `scripts/generate_all_paper_figures.py`

---

## 🔴 严重问题 (虚假/错误数据)

### 问题0 (新发现): 第三章K-fold数据论文与实际不符

**论文描述** (第1251行): "5折训练集的R²均超过0.98，RMSE稳定在0.44 m/s左右，MAPE约为6.4%"

**实际实验数据** (`_trash/results/chapter3/experiment_results.json`):
| 指标 | 论文声称 | 实际(训练集) | 实际(验证集) |
|------|---------|------------|------------|
| R² | >0.98 | 0.926 | 0.729 |
| RMSE | 0.44 m/s | 0.885 m/s | 1.587 m/s |
| MAPE | 6.4% | 5.57% | 10.90% |

**影响**: 论文表3.8的数据需要用真实实验结果替换。
**状态**: ⚠️ 需要修改论文

---

### 问题1: 旧脚本将自建baseline伪装为YNet/PECNet [已修复]

**文件**: `scripts/generate_paper_ch4_figures.py` 第35-56行

**问题**: 旧脚本将 `LSTM_Env_Goal` 标记为 "YNet"，将 `Seq2Seq_Attn` 标记为 "PECNet"。
但这些是自建的简单baseline，**不是**真正的YNet/PECNet实现。

**实际差距**:
| 模型标识 | 真实身份 | P1a ADE | 真正实现的ADE | 差距 |
|---------|---------|---------|-------------|------|
| LSTM_Env_Goal | 自建LSTM+Env+Goal | 1.41km | YNet: 3.03km | -53% |
| Seq2Seq_Attn | 自建Seq2Seq+Attention | 5.98km | PECNet: 3.91km | +53% |

**影响**: 旧脚本生成的箱线图(`fig4_box_phase1/2/3`)、时间趋势图(`fig4_time_phase1/2/3`)、
热力图(`fig4_10_phase_heatmap`)全部使用了错误的模型标签。

**修复**: 这些图表必须使用 `phase_v2_with_faithful` 数据重新生成，其中包含真正的 YNet 和 PECNet。

---

### 问题2: 候选敏感性图完全虚构

**文件**: `scripts/generate_paper_ch4_figures.py` 第506-531行

**问题**: `fig4_9_candidate_sensitivity` 中的数据**全部硬编码**，不来自任何实验：
```python
terratnt_ade = [1.22, 1.24, 1.25, 1.24, 1.26]  # 硬编码
ynet_ade = [1.8, 2.5, 3.2, 4.1, 5.0]            # 硬编码
pecnet_ade = [2.0, 3.0, 4.5, 6.0, 7.5]           # 硬编码
```

**实际实验数据** (`control_variable_results.json`): 所有K值产生完全相同的结果(ADE=16.06km)，
说明候选K实验根本没有正确运行。

**影响**: `fig4_9_candidate_sensitivity.pdf` 是完全虚构的图表。

---

### 问题3: 跨区域图表全部硬编码

**文件**: `scripts/generate_paper_ch4_figures.py` 第355-495行

| 图表 | 行号 | 硬编码内容 |
|------|------|-----------|
| fig4_25_region_bars | 379 | `{BF:1.53, DB:2.03, CP:1.98, SH:4.41}` |
| fig4_26_cross_matrix | 414-419 | 完整4×4矩阵16个数值 |
| fig4_27_gen_loss | 457-458 | 域内+域外平均8个数值 |

**实际数据**: 
- 只有1个跨区域训练 (`single_reg_V6R_drop0.15_bohemian_forest`)
- `cross_bohemian_forest_to_scottish_highlands/phase_v2_results.json` 为空 `{}`
- donbas/carpathians 尚无训练数据

---

## 🟡 次要问题

### 问题4: mlp_decoder消融结果std=0

**文件**: `outputs/evaluation/ablation/ablation_results.json`

`mlp_decoder` 的 `ade_std=0`，说明可能只用了聚合值而非per-sample评估。
数据本身(ADE=1.47km)看起来合理，但缺少方差信息。

### 问题5: control_variable_results.json 缺少关键实验

该文件只包含 `phase3_sensitivity`（且数据无效），缺少：
- `candidate_K` (Phase1下的K值影响)
- `observation_length` (观测时长影响)

导致 `generate_all_paper_figures.py` 的 `fig_control_variables` 只能画 waypoint 一个panel。

### 问题6: v5数据中V6R_Robust的ADE与v2不一致

- v5 fas1: V6R_Robust ADE = 1068m (1.07km)
- v2 P1a:  V6R_Robust ADE = 1245m (1.25km)

差异17%，可能是不同的评估split或checkpoint导致。需要确认哪个是最终结果。

---

## ✅ 数据可靠的图表

| 图表 | 数据源 | 状态 |
|------|--------|------|
| fig4_6_training_curves | 真实训练日志 | ✅ 可靠 |
| fig4_7_phase_comparison | phase_v2_with_faithful (真实YNet/PECNet) | ✅ 可靠 |
| fig4_8_ablation | ablation_results.json (真实实验) | ✅ 可靠 |
| fig4_9_10_control_variables (waypoint部分) | waypoint_ablation_results.json | ✅ 可靠 |
| fig4_1_architecture | 绘图 | ✅ 可靠 |

---

## 📋 修复计划

### 必须修复 (论文提交前)
1. **重新生成箱线图/时间趋势图/热力图**: 使用 `phase_v2_with_faithful` 数据，包含真正的YNet/PECNet
2. **删除或重做候选敏感性图**: 需要重新运行Phase3候选K实验，或从论文中删除该图
3. **完成跨区域实验**: 生成donbas/carpathians轨迹 → 训练 → 评估 → 用真实数据替换硬编码

### 建议修复
4. 补充 `candidate_K` 和 `observation_length` 控制变量实验
5. 确认 V6R_Robust 的最终ADE基准值 (1.07 vs 1.25)
6. 为 mlp_decoder 补充per-sample评估以获得std

---

## 📁 文件位置汇总

### 可靠的评估数据
- `outputs/evaluation/phase_v2_with_faithful/phase_v2_results.json` — Phase V2 完整评估 (13模型×7Phase)
- `outputs/evaluation/ablation/ablation_results.json` — 消融实验 (模块+通道+运动学)
- `outputs/evaluation/control_variables/waypoint_ablation_results.json` — Waypoint消融 (2/4/6/8/10)

### 可靠的图表
- `outputs/paper_ch4_figures/fig4_6_training_curves.pdf`
- `outputs/paper_ch4_figures/fig4_7_phase_comparison.pdf`
- `outputs/paper_ch4_figures/fig4_8_ablation.pdf`
- `outputs/paper_ch4_figures/fig4_9_10_control_variables.pdf` (仅waypoint panel)

### 需要重新生成的图表
- `outputs/paper_ch4_figures/fig4_box_phase1/2/3.pdf` — 使用了错误的模型标签
- `outputs/paper_ch4_figures/fig4_time_phase1/2/3.pdf` — 使用了错误的模型标签
- `outputs/paper_ch4_figures/fig4_10_phase_heatmap.pdf` — 使用了错误的模型标签
- `outputs/paper_ch4_figures/fig4_9_candidate_sensitivity.pdf` — 完全虚构
- `outputs/paper_ch4_figures/fig4_25_region_bars.pdf` — 硬编码
- `outputs/paper_ch4_figures/fig4_26_cross_matrix.pdf` — 硬编码
- `outputs/paper_ch4_figures/fig4_27_gen_loss.pdf` — 硬编码

### 生成脚本
- `scripts/generate_all_paper_figures.py` — 新脚本，使用正确数据源 (推荐)
- `scripts/generate_paper_ch4_figures.py` — 旧脚本，有多处虚假数据问题
