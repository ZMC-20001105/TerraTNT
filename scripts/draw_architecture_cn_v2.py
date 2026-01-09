#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TerraTNT 系统架构图（优化版）- 论文风格
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
import warnings

# 配置中文字体
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',
    'Noto Sans CJK TC', 
    'Droid Sans Fallback',
    'DejaVu Sans',
    'sans-serif'
]
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

# 清除字体缓存
try:
    fm._load_fontmanager(try_read_cache=False)
except:
    pass

# 创建图形 - 增大画布以容纳更大字体
fig, ax = plt.subplots(figsize=(18, 12), facecolor='white')
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# 标题 - 更大更醒目
ax.text(9, 11.2, 'TerraTNT 轨迹预测系统架构', 
        fontsize=32, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#333', linewidth=2))

# ============ 第1层：用户交互层 ============
ui_box = FancyBboxPatch((0.5, 8.8), 17, 1.8, 
                        boxstyle="round,pad=0.12", 
                        edgecolor='#1565c0', facecolor='#e3f2fd', linewidth=3)
ax.add_patch(ui_box)
ax.text(1.2, 10.2, '用户交互层', fontsize=18, fontweight='bold', color='#0d47a1')
ax.text(1.2, 9.8, 'User Interface Layer', fontsize=11, color='#1976d2', style='italic')

# UI 子模块 - 增大字号
ui_modules = [
    ('主控界面', 'Main Window'),
    ('参数配置', 'Configuration'),
    ('数据管理', 'Data Manager'),
    ('轨迹可视化', 'Visualization'),
    ('训练监控', 'Training Monitor')
]
for i, (name_cn, name_en) in enumerate(ui_modules):
    x = 1.5 + i * 3.2
    rect = FancyBboxPatch((x, 9), 2.9, 0.9, 
                          boxstyle="round,pad=0.06",
                          edgecolor='#1976d2', facecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.45, 9.6, name_cn, fontsize=13, ha='center', va='center', fontweight='bold')
    ax.text(x + 1.45, 9.2, name_en, fontsize=9, ha='center', va='center', 
            color='#666', style='italic')

# ============ 第2层：应用逻辑层 ============
app_box = FancyBboxPatch((0.5, 6.2), 17, 2.2, 
                        boxstyle="round,pad=0.12",
                        edgecolor='#2e7d32', facecolor='#e8f5e9', linewidth=3)
ax.add_patch(app_box)
ax.text(1.2, 8.1, '应用逻辑层', fontsize=18, fontweight='bold', color='#1b5e20')
ax.text(1.2, 7.7, 'Application Logic Layer', fontsize=11, color='#388e3c', style='italic')

# 应用层子模块 - 更详细的描述
app_modules = [
    ('轨迹生成引擎', 'Trajectory Generator', '分层A*路径规划\n多进程并行计算\n地形约束优化'),
    ('训练编排器', 'Training Orchestrator', 'FAS三阶段训练\n早停与学习率调度\n模型检查点管理'),
    ('评估分析器', 'Evaluation Analyzer', 'ADE/FDE/MR指标\n基线模型对比\n性能可视化')
]
for i, (name_cn, name_en, desc) in enumerate(app_modules):
    x = 1.5 + i * 5.4
    rect = FancyBboxPatch((x, 6.4), 5, 1.6,
                          boxstyle="round,pad=0.08",
                          edgecolor='#388e3c', facecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 2.5, 7.7, name_cn, fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(x + 2.5, 7.35, name_en, fontsize=10, ha='center', va='center', 
            color='#555', style='italic')
    ax.text(x + 2.5, 6.75, desc, fontsize=9.5, ha='center', va='center', 
            color='#666', linespacing=1.4)

# ============ 第3层：模型层 & 数据层 ============
# 模型层
model_box = FancyBboxPatch((0.5, 3.6), 8, 2.2,
                          boxstyle="round,pad=0.12",
                          edgecolor='#e65100', facecolor='#fff3e0', linewidth=3)
ax.add_patch(model_box)
ax.text(1.2, 5.5, '深度学习模型层', fontsize=18, fontweight='bold', color='#e65100')
ax.text(1.2, 5.1, 'Deep Learning Models', fontsize=11, color='#f57c00', style='italic')

# 模型子模块 - 更专业的术语
models = [
    ('TerraTNT', 'Core Model', 'CNN环境编码器\nLSTM历史编码器\n目标驱动解码器'),
    ('基线模型', 'Baselines', 'Trajectron++\nPECNet\nYNet'),
    ('速度预测', 'Speed Predictor', 'XGBoost回归\nOORD数据训练')
]
for i, (name_cn, name_en, desc) in enumerate(models):
    x = 1.5 + i * 2.3
    rect = FancyBboxPatch((x, 3.8), 2.1, 1.5,
                          boxstyle="round,pad=0.06",
                          edgecolor='#f57c00', facecolor='#ffe0b2', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.05, 5.05, name_cn, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(x + 1.05, 4.75, name_en, fontsize=9, ha='center', va='center', 
            color='#555', style='italic')
    ax.text(x + 1.05, 4.25, desc, fontsize=8.5, ha='center', va='center', 
            color='#666', linespacing=1.3)

# 数据层
data_box = FancyBboxPatch((9.5, 3.6), 8, 2.2,
                         boxstyle="round,pad=0.12",
                         edgecolor='#ad1457', facecolor='#fce4ec', linewidth=3)
ax.add_patch(data_box)
ax.text(10.2, 5.5, '数据持久化层', fontsize=18, fontweight='bold', color='#880e4f')
ax.text(10.2, 5.1, 'Data Persistence Layer', fontsize=11, color='#c2185b', style='italic')

# 数据子模块 - 更详细的规格
data_modules = [
    ('地理空间数据', 'Geospatial Data', 'DEM高程(30m)\nLULC土地覆盖\nOSM路网\nUTM 32630投影'),
    ('轨迹数据集', 'Trajectory Dataset', '100,000+样本\nFAS三阶段划分\n模型检查点')
]
for i, (name_cn, name_en, desc) in enumerate(data_modules):
    x = 10.2 + i * 3.5
    rect = FancyBboxPatch((x, 3.8), 3.3, 1.5,
                          boxstyle="round,pad=0.06",
                          edgecolor='#c2185b', facecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.65, 5.05, name_cn, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(x + 1.65, 4.75, name_en, fontsize=9, ha='center', va='center', 
            color='#555', style='italic')
    ax.text(x + 1.65, 4.25, desc, fontsize=8.5, ha='center', va='center', 
            color='#666', linespacing=1.3)

# ============ 第4层：基础设施层 ============
infra_box = FancyBboxPatch((0.5, 1.8), 17, 1.4,
                          boxstyle="round,pad=0.12",
                          edgecolor='#424242', facecolor='#f5f5f5', linewidth=3)
ax.add_patch(infra_box)
ax.text(9, 2.9, '计算基础设施层', fontsize=16, fontweight='bold', ha='center', color='#212121')
ax.text(9, 2.55, 'Computing Infrastructure Layer', fontsize=11, ha='center', 
        color='#616161', style='italic')
ax.text(9, 2.15, 'Ubuntu 24.04 LTS  •  NVIDIA CUDA 12.0  •  PyTorch 2.0+  •  GDAL 3.8 & Proj 9.3  •  Conda Environment', 
        fontsize=11.5, ha='center', color='#616161', fontweight='500')

# ============ 添加数据流箭头 ============
arrow_props = dict(arrowstyle='->', lw=3, color='#37474f')

# UI -> Application
arrow1 = FancyArrowPatch((9, 8.8), (9, 8.4), **arrow_props)
ax.add_patch(arrow1)
ax.text(9.5, 8.6, '用户请求', fontsize=10, color='#37474f', style='italic')

# Application -> Model
arrow2 = FancyArrowPatch((4.5, 6.2), (4.5, 5.8), **arrow_props)
ax.add_patch(arrow2)
ax.text(5.2, 6, '训练/推理', fontsize=10, color='#37474f', style='italic')

# Application -> Data
arrow3 = FancyArrowPatch((13.5, 6.2), (13.5, 5.8), **arrow_props)
ax.add_patch(arrow3)
ax.text(14.2, 6, '数据读写', fontsize=10, color='#37474f', style='italic')

# Model <-> Data (双向箭头)
arrow4 = FancyArrowPatch((8.5, 4.7), (9.5, 4.7), 
                        arrowstyle='<->', lw=2.5, color='#37474f')
ax.add_patch(arrow4)
ax.text(9, 5, '数据交换', fontsize=10, ha='center', color='#37474f', style='italic')

# Model/Data -> Infrastructure
arrow5 = FancyArrowPatch((4.5, 3.6), (4.5, 3.2), **arrow_props)
ax.add_patch(arrow5)
arrow6 = FancyArrowPatch((13.5, 3.6), (13.5, 3.2), **arrow_props)
ax.add_patch(arrow6)

# ============ 添加图例 ============
legend_y = 1.1
legend_items = [
    ('● 用户交互层', '#1565c0', 2),
    ('● 应用逻辑层', '#2e7d32', 6.5),
    ('● 深度学习模型层', '#e65100', 11),
    ('● 数据持久化层', '#ad1457', 15.5)
]
for text, color, x_pos in legend_items:
    ax.text(x_pos, legend_y, text, fontsize=11, color=color, fontweight='bold')

# ============ 数据流描述 ============
ax.text(9, 0.4, '数据流向：用户交互 → 应用逻辑 → 模型计算/数据存储 → 基础设施支撑 → 结果返回', 
        fontsize=12, style='italic', ha='center', color='#37474f', fontweight='500',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#eceff1', edgecolor='#90a4ae', linewidth=1.5))

# 保存图片
plt.tight_layout()
output_path = '/home/zmc/文档/programwork/docs/architecture_diagram_cn_v2.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ 优化版中文架构图已保存到: {output_path}")
print(f"  - 分辨率: 300 DPI")
print(f"  - 尺寸: 18×12 英寸")
print(f"  - 字体优化: 标题32pt, 层级标题18pt, 模块名称12-14pt")
print(f"  - 双语标注: 中文主标题 + 英文副标题")
plt.close()
