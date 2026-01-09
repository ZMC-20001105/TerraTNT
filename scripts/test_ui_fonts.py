#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 UI 中文字体配置
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

print("="*60)
print("检查系统中文字体")
print("="*60)

# 列出所有可用的中文字体
chinese_fonts = []
for font in fm.fontManager.ttflist:
    if 'CJK' in font.name or 'Noto' in font.name or 'WenQuanYi' in font.name:
        chinese_fonts.append(font.name)

if chinese_fonts:
    print(f"找到 {len(set(chinese_fonts))} 个中文字体:")
    for font in sorted(set(chinese_fonts))[:10]:
        print(f"  - {font}")
else:
    print("警告：未找到中文字体")

print("\n" + "="*60)
print("测试 Matplotlib 中文显示")
print("="*60)

# 配置字体
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

# 创建测试图表
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', label='测试数据')
ax.set_title('中文标题测试：轨迹可视化')
ax.set_xlabel('时间（分钟）')
ax.set_ylabel('距离（公里）')
ax.legend()
ax.grid(True, alpha=0.3)

# 保存测试图像
output_file = '/home/zmc/文档/programwork/docs/font_test.png'
plt.savefig(output_file, dpi=100, bbox_inches='tight')
print(f"✓ 测试图像已保存到: {output_file}")
print("  请检查图像中的中文是否正常显示")

print("\n" + "="*60)
print("当前 Matplotlib 字体配置")
print("="*60)
print(f"font.sans-serif: {matplotlib.rcParams['font.sans-serif']}")
print(f"font.family: {matplotlib.rcParams['font.family']}")
print(f"axes.unicode_minus: {matplotlib.rcParams['axes.unicode_minus']}")

print("\n✓ 字体测试完成")
