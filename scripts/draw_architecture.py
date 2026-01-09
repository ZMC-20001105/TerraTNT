#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制系统架构图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 设置中文字体（使用系统中确实可用的字体）
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Droid Sans Fallback', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 清除字体缓存
try:
    fm._load_fontmanager(try_read_cache=False)
except:
    pass

# 创建图形
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(7, 9.5, 'TerraTNT 系统架构设计', 
        fontsize=24, fontweight='bold', ha='center', va='center')

# 1. 交互层 (UI Layer)
ui_box = FancyBboxPatch((0.5, 7.5), 13, 1.5, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='#1976d2', facecolor='#e3f2fd', linewidth=2)
ax.add_patch(ui_box)
ax.text(1, 8.8, '交互层 (UI Layer)', fontsize=14, fontweight='bold', color='#0d47a1')

# UI 子模块
ui_modules = ['主窗口', '卫星配置', '数据加载', '轨迹预测', '可视化']
for i, module in enumerate(ui_modules):
    x = 1.5 + i * 2.3
    rect = FancyBboxPatch((x, 7.7), 2, 0.6, 
                          boxstyle="round,pad=0.05",
                          edgecolor='#1976d2', facecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 1, 8, module, fontsize=10, ha='center', va='center')

# 2. 应用层 (Application Layer)
app_box = FancyBboxPatch((0.5, 5.5), 13, 1.5,
                        boxstyle="round,pad=0.1",
                        edgecolor='#388e3c', facecolor='#e8f5e9', linewidth=2)
ax.add_patch(app_box)
ax.text(1, 6.8, '应用层 (Application Layer)', fontsize=14, fontweight='bold', color='#1b5e20')

# 应用层子模块
app_modules = [
    ('数据生成引擎', 'A* + 并行'),
    ('模型编排器', 'FAS 1/2/3'),
    ('评估框架', 'ADE/FDE')
]
for i, (name, desc) in enumerate(app_modules):
    x = 1.5 + i * 3.8
    rect = FancyBboxPatch((x, 5.7), 3.5, 1,
                          boxstyle="round,pad=0.05",
                          edgecolor='#388e3c', facecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 1.75, 6.4, name, fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(x + 1.75, 6, desc, fontsize=8, ha='center', va='center', color='gray')

# 3. 模型层 & 数据层
# 模型层
model_box = FancyBboxPatch((0.5, 3.5), 6, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='#f57c00', facecolor='#fff3e0', linewidth=2)
ax.add_patch(model_box)
ax.text(1, 4.8, '模型层 (Model Layer)', fontsize=14, fontweight='bold', color='#e65100')

# 模型子模块
models = ['TerraTNT', '基线模型', 'XGBoost']
for i, model in enumerate(models):
    x = 1.5 + i * 1.5
    rect = FancyBboxPatch((x, 3.7), 1.3, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor='#f57c00', facecolor='#ffe0b2', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 0.65, 4, model, fontsize=9, ha='center', va='center')

# 数据层
data_box = FancyBboxPatch((7, 3.5), 6, 1.5,
                         boxstyle="round,pad=0.1",
                         edgecolor='#c2185b', facecolor='#fce4ec', linewidth=2)
ax.add_patch(data_box)
ax.text(7.5, 4.8, '数据层 (Data Layer)', fontsize=14, fontweight='bold', color='#880e4f')

# 数据子模块
data_modules = ['GIS 资产', '轨迹存储', 'FAS 划分']
for i, module in enumerate(data_modules):
    x = 7.5 + i * 1.8
    rect = FancyBboxPatch((x, 3.7), 1.6, 0.6,
                          boxstyle="round,pad=0.05",
                          edgecolor='#c2185b', facecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 0.8, 4, module, fontsize=9, ha='center', va='center')

# 4. 基础设施层
infra_box = FancyBboxPatch((0.5, 2), 13, 1,
                          boxstyle="round,pad=0.1",
                          edgecolor='#616161', facecolor='#f5f5f5', linewidth=2)
ax.add_patch(infra_box)
ax.text(7, 2.7, '基础设施 (Infrastructure)', fontsize=12, fontweight='bold', ha='center', color='#424242')
ax.text(7, 2.3, 'Ubuntu 24.04 | NVIDIA CUDA | PyTorch 2.0+ | GDAL | Conda', 
        fontsize=10, ha='center', color='#616161')

# 添加箭头连接
arrow_props = dict(arrowstyle='->', lw=2, color='#546e7a')

# UI -> Application
arrow1 = FancyArrowPatch((7, 7.5), (7, 7), **arrow_props)
ax.add_patch(arrow1)

# Application -> Model & Data
arrow2 = FancyArrowPatch((4, 5.5), (4, 5), **arrow_props)
ax.add_patch(arrow2)
arrow3 = FancyArrowPatch((10, 5.5), (10, 5), **arrow_props)
ax.add_patch(arrow3)

# Model <-> Data
arrow4 = FancyArrowPatch((6.5, 4.2), (7, 4.2), arrowstyle='<->', lw=1.5, color='#546e7a')
ax.add_patch(arrow4)

# 添加图例说明
legend_y = 1.2
ax.text(1, legend_y, '● 交互层: PyQt5 用户界面', fontsize=9, color='#1976d2')
ax.text(4.5, legend_y, '● 应用层: 业务逻辑控制', fontsize=9, color='#388e3c')
ax.text(8, legend_y, '● 模型层: 深度学习模型', fontsize=9, color='#f57c00')
ax.text(11, legend_y, '● 数据层: 数据管理', fontsize=9, color='#c2185b')

ax.text(1, 0.6, '数据流向: 用户 → UI → 应用层 → 模型/数据层 → 结果返回', 
        fontsize=10, style='italic', color='#546e7a')

# 保存图片
plt.tight_layout()
plt.savefig('/home/zmc/文档/programwork/docs/architecture_diagram.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ 架构图已保存到: /home/zmc/文档/programwork/docs/architecture_diagram.png")
plt.close()
