#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制系统架构图（中文版 - 修复字体问题并增大字号）
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
from matplotlib import font_manager
import warnings

# 抑制警告
warnings.filterwarnings('ignore')

# 直接指定字体文件路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 创建图形 - 增大尺寸
fig, ax = plt.subplots(figsize=(18, 12), facecolor='white')
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# 标题 - 增大字号
ax.text(9, 11.2, 'TerraTNT 系统架构设计', 
        fontsize=32, fontweight='bold', ha='center', va='center', fontproperties=prop)

# 第1层：交互层 (UI Layer)
ui_box = FancyBboxPatch((0.5, 9), 17, 1.8, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='#1976d2', facecolor='#e3f2fd', linewidth=3)
ax.add_patch(ui_box)
ax.text(1, 10.5, '交互层 (PyQt5 用户界面)', fontsize=18, fontweight='bold', 
        color='#0d47a1', fontproperties=prop)

# UI 子模块
ui_modules = ['主窗口', '卫星配置', '数据加载', '轨迹查看', '训练控制台']
for i, module in enumerate(ui_modules):
    x = 1.5 + i * 3.1
    rect = FancyBboxPatch((x, 9.2), 2.8, 0.8, 
                          boxstyle="round,pad=0.05",
                          edgecolor='#1976d2', facecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.4, 9.6, module, fontsize=13, ha='center', va='center', 
            fontweight='600', fontproperties=prop)

# 第2层：应用层 (Application Layer)
app_box = FancyBboxPatch((0.5, 6.3), 17, 2.2, 
                        boxstyle="round,pad=0.1",
                        edgecolor='#388e3c', facecolor='#e8f5e9', linewidth=3)
ax.add_patch(app_box)
ax.text(1, 8.2, '应用层 (业务逻辑)', fontsize=18, fontweight='bold', 
        color='#1b5e20', fontproperties=prop)

# 应用层子模块
app_modules = [
    ('数据生成引擎', '分层A*算法\n并行计算'),
    ('模型编排器', 'FAS 1/2/3阶段\n早停机制'),
    ('评估框架', 'ADE/FDE/MR\n基准对比')
]
for i, (name, desc) in enumerate(app_modules):
    x = 1.5 + i * 5.3
    rect = FancyBboxPatch((x, 6.5), 4.8, 1.6,
                          boxstyle="round,pad=0.08",
                          edgecolor='#388e3c', facecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 2.4, 7.6, name, fontsize=14, ha='center', va='center', 
            fontweight='bold', fontproperties=prop)
    ax.text(x + 2.4, 6.95, desc, fontsize=11, ha='center', va='center', 
            color='#555', style='italic', fontproperties=prop)

# 第3层：模型层 & 数据层
# 模型层
model_box = FancyBboxPatch((0.5, 3.8), 8, 2.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='#f57c00', facecolor='#fff3e0', linewidth=3)
ax.add_patch(model_box)
ax.text(1, 5.7, '模型层 (深度学习)', fontsize=18, fontweight='bold', 
        color='#e65100', fontproperties=prop)

# 模型子模块
models = [
    ('TerraTNT\n核心模型', 'CNN+LSTM\n目标驱动'),
    ('基线模型', 'YNet, PECNet\nTrajectron++'),
    ('速度预测器', 'XGBoost\nOORD训练')
]
for i, (name, desc) in enumerate(models):
    x = 1.5 + i * 2.3
    rect = FancyBboxPatch((x, 4), 2.1, 1.5,
                          boxstyle="round,pad=0.05",
                          edgecolor='#f57c00', facecolor='#ffe0b2', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.05, 5.1, name, fontsize=12, ha='center', va='center', 
            fontweight='bold', fontproperties=prop)
    ax.text(x + 1.05, 4.4, desc, fontsize=10, ha='center', va='center', 
            color='#555', fontproperties=prop)

# 数据层
data_box = FancyBboxPatch((9.5, 3.8), 8, 2.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#c2185b', facecolor='#fce4ec', linewidth=3)
ax.add_patch(data_box)
ax.text(10, 5.7, '数据持久化层', fontsize=18, fontweight='bold', 
        color='#880e4f', fontproperties=prop)

# 数据子模块
data_modules = [
    ('GIS资产', 'DEM, LULC\nOSM道路\nUTM 32630'),
    ('轨迹存储', '10万+PKL\nFAS划分\n检查点')
]
for i, (name, desc) in enumerate(data_modules):
    x = 10.5 + i * 3.4
    rect = FancyBboxPatch((x, 4), 3.2, 1.5,
                          boxstyle="round,pad=0.05",
                          edgecolor='#c2185b', facecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + 1.6, 5.1, name, fontsize=12, ha='center', va='center', 
            fontweight='bold', fontproperties=prop)
    ax.text(x + 1.6, 4.4, desc, fontsize=10, ha='center', va='center', 
            color='#555', fontproperties=prop)

# 第4层：基础设施层
infra_box = FancyBboxPatch((0.5, 1.9), 17, 1.4,
                          boxstyle="round,pad=0.1",
                          edgecolor='#616161', facecolor='#f5f5f5', linewidth=3)
ax.add_patch(infra_box)
ax.text(9, 2.9, '基础设施层', fontsize=16, fontweight='bold', ha='center', 
        color='#424242', fontproperties=prop)
ax.text(9, 2.35, 'Ubuntu 24.04 LTS  |  NVIDIA CUDA  |  PyTorch 2.0+  |  GDAL & Proj  |  Anaconda环境', 
        fontsize=13, ha='center', color='#616161', fontproperties=prop)

# 添加箭头连接
arrow_props = dict(arrowstyle='->', lw=3, color='#546e7a')

# UI -> Application
arrow1 = FancyArrowPatch((9, 9), (9, 8.5), **arrow_props)
ax.add_patch(arrow1)

# Application -> Model
arrow2 = FancyArrowPatch((4.5, 6.3), (4.5, 6), **arrow_props)
ax.add_patch(arrow2)

# Application -> Data
arrow3 = FancyArrowPatch((13.5, 6.3), (13.5, 6), **arrow_props)
ax.add_patch(arrow3)

# Model <-> Data (双向箭头)
arrow4 = FancyArrowPatch((8.5, 4.9), (9.5, 4.9), 
                        arrowstyle='<->', lw=2.5, color='#546e7a')
ax.add_patch(arrow4)

# 添加图例
legend_y = 1.3
ax.text(1.8, legend_y, '● 交互层: 用户界面与可视化', 
        fontsize=12, color='#1976d2', fontweight='600', fontproperties=prop)
ax.text(6.2, legend_y, '● 应用层: 业务逻辑与编排', 
        fontsize=12, color='#388e3c', fontweight='600', fontproperties=prop)
ax.text(11.2, legend_y, '● 模型层: AI/ML组件', 
        fontsize=12, color='#f57c00', fontweight='600', fontproperties=prop)
ax.text(15.2, legend_y, '● 数据层: 存储', 
        fontsize=12, color='#c2185b', fontweight='600', fontproperties=prop)

# 数据流描述
ax.text(9, 0.5, '数据流向: 用户 → UI → 应用层 → 模型/数据层 → 结果返回', 
        fontsize=13, style='italic', ha='center', color='#546e7a', 
        fontweight='500', fontproperties=prop)

# 保存图片
plt.tight_layout()
plt.savefig('/home/zmc/文档/programwork/docs/architecture_diagram_cn.png', 
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ 中文架构图已保存（字体修复，字号增大）")
plt.close()
