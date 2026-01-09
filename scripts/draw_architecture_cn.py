#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制系统架构图（中文版）
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
import warnings

# 配置中文字体 - 使用与UI相同的配置
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

# 创建图形
fig, ax = plt.subplots(figsize=(16, 11), facecolor='white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

# 标题
ax.text(8, 10.3, 'TerraTNT 系统架构设计', 
        fontsize=28, fontweight='bold', ha='center', va='center')

# 第1层：交互层 (UI Layer)
ui_box = FancyBboxPatch((0.5, 8.2), 15, 1.6, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='#1976d2', facecolor='#e3f2fd', linewidth=2.5)
ax.add_patch(ui_box)
ax.text(1, 9.5, '交互层 (PyQt5 用户界面)', fontsize=16, fontweight='bold', color='#0d47a1')

# UI 子模块
ui_modules = ['主窗口', '卫星配置', '数据加载', '轨迹查看', '训练控制台']
for i, module in enumerate(ui_modules):
    x = 1.2 + i * 2.8
    rect = FancyBboxPatch((x, 8.4), 2.5, 0.7, 
                          boxstyle="round,pad=0.05",
                          edgecolor='#1976d2', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1.25, 8.75, module, fontsize=11, ha='center', va='center', fontweight='600')

# 第2层：应用层 (Application Layer)
app_box = FancyBboxPatch((0.5, 5.8), 15, 2, 
                        boxstyle="round,pad=0.1",
                        edgecolor='#388e3c', facecolor='#e8f5e9', linewidth=2.5)
ax.add_patch(app_box)
ax.text(1, 7.5, '应用层 (业务逻辑)', fontsize=16, fontweight='bold', color='#1b5e20')

# 应用层子模块
app_modules = [
    ('数据生成引擎', '分层A*算法\n并行计算'),
    ('模型编排器', 'FAS 1/2/3阶段\n早停机制'),
    ('评估框架', 'ADE/FDE/MR\n基准对比')
]
for i, (name, desc) in enumerate(app_modules):
    x = 1.2 + i * 4.7
    rect = FancyBboxPatch((x, 6), 4.2, 1.4,
                          boxstyle="round,pad=0.08",
                          edgecolor='#388e3c', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 2.1, 6.9, name, fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(x + 2.1, 6.3, desc, fontsize=9, ha='center', va='center', color='#555', style='italic')

# 第3层：模型层 & 数据层
# 模型层
model_box = FancyBboxPatch((0.5, 3.5), 7, 2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#f57c00', facecolor='#fff3e0', linewidth=2.5)
ax.add_patch(model_box)
ax.text(1, 5.2, '模型层 (深度学习)', fontsize=16, fontweight='bold', color='#e65100')

# 模型子模块
models = [
    ('TerraTNT\n核心模型', 'CNN+LSTM\n目标驱动'),
    ('基线模型', 'YNet, PECNet\nTrajectron++'),
    ('速度预测器', 'XGBoost\nOORD训练')
]
for i, (name, desc) in enumerate(models):
    x = 1.2 + i * 2
    rect = FancyBboxPatch((x, 3.7), 1.8, 1.3,
                          boxstyle="round,pad=0.05",
                          edgecolor='#f57c00', facecolor='#ffe0b2', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.9, 4.7, name, fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(x + 0.9, 4.1, desc, fontsize=8, ha='center', va='center', color='#555')

# 数据层
data_box = FancyBboxPatch((8.5, 3.5), 7, 2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#c2185b', facecolor='#fce4ec', linewidth=2.5)
ax.add_patch(data_box)
ax.text(9, 5.2, '数据持久化层', fontsize=16, fontweight='bold', color='#880e4f')

# 数据子模块
data_modules = [
    ('GIS资产', 'DEM, LULC\nOSM道路\nUTM 32630'),
    ('轨迹存储', '10万+PKL\nFAS划分\n检查点')
]
for i, (name, desc) in enumerate(data_modules):
    x = 9.2 + i * 3
    rect = FancyBboxPatch((x, 3.7), 2.8, 1.3,
                          boxstyle="round,pad=0.05",
                          edgecolor='#c2185b', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1.4, 4.7, name, fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(x + 1.4, 4.1, desc, fontsize=8, ha='center', va='center', color='#555')

# 第4层：基础设施层
infra_box = FancyBboxPatch((0.5, 1.8), 15, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#616161', facecolor='#f5f5f5', linewidth=2.5)
ax.add_patch(infra_box)
ax.text(8, 2.7, '基础设施层', fontsize=14, fontweight='bold', ha='center', color='#424242')
ax.text(8, 2.2, 'Ubuntu 24.04 LTS  |  NVIDIA CUDA  |  PyTorch 2.0+  |  GDAL & Proj  |  Anaconda环境', 
        fontsize=11, ha='center', color='#616161')

# 添加箭头连接
arrow_props = dict(arrowstyle='->', lw=2.5, color='#546e7a')

# UI -> Application
arrow1 = FancyArrowPatch((8, 8.2), (8, 7.8), **arrow_props)
ax.add_patch(arrow1)

# Application -> Model
arrow2 = FancyArrowPatch((4, 5.8), (4, 5.5), **arrow_props)
ax.add_patch(arrow2)

# Application -> Data
arrow3 = FancyArrowPatch((12, 5.8), (12, 5.5), **arrow_props)
ax.add_patch(arrow3)

# Model <-> Data (双向箭头)
arrow4 = FancyArrowPatch((7.5, 4.5), (8.5, 4.5), 
                        arrowstyle='<->', lw=2, color='#546e7a')
ax.add_patch(arrow4)

# 添加图例
legend_y = 1.2
ax.text(1.5, legend_y, '● 交互层: 用户界面与可视化', 
        fontsize=10, color='#1976d2', fontweight='600')
ax.text(5.5, legend_y, '● 应用层: 业务逻辑与编排', 
        fontsize=10, color='#388e3c', fontweight='600')
ax.text(10, legend_y, '● 模型层: AI/ML组件', 
        fontsize=10, color='#f57c00', fontweight='600')
ax.text(13.5, legend_y, '● 数据层: 存储', 
        fontsize=10, color='#c2185b', fontweight='600')

# 数据流描述
ax.text(8, 0.5, '数据流向: 用户 → UI → 应用层 → 模型/数据层 → 结果返回', 
        fontsize=11, style='italic', ha='center', color='#546e7a', fontweight='500')

# 保存图片
plt.tight_layout()
plt.savefig('/home/zmc/文档/programwork/docs/architecture_diagram_cn.png', 
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ 中文架构图已保存到: /home/zmc/文档/programwork/docs/architecture_diagram_cn.png")
plt.close()
