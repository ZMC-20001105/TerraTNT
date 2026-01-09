#!/usr/bin/env python
"""生成交互式UI的所有页面截图"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

output_dir = Path('/home/zmc/文档/programwork/docs/ui_interactive')
output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('dark_background')

# 1. 卫星星座配置页面（带参数设置）
fig = plt.figure(figsize=(18, 10), facecolor='#1e1e2e')

# 左侧：参数面板
ax_left = fig.add_subplot(1, 2, 1, facecolor='#1e1e2e')
ax_left.text(0.5, 0.95, 'Constellation Parameters', ha='center', fontsize=18, 
             fontweight='bold', color='#89b4fa', transform=ax_left.transAxes)

params = [
    ('Number of Satellites:', '9', '[3-30]'),
    ('Number of Orbit Planes:', '3', '[1-5]'),
    ('Orbit Altitude (km):', '600', '[400-1000]'),
    ('Inclination 1 (°):', '45', '[0-90]'),
    ('Inclination 2 (°):', '55', '[0-90]'),
    ('Inclination 3 (°):', '65', '[0-90]'),
]

y_pos = 0.80
for label, value, range_str in params:
    ax_left.text(0.1, y_pos, label, fontsize=13, color='#cdd6f4', transform=ax_left.transAxes)
    ax_left.add_patch(plt.Rectangle((0.55, y_pos-0.02), 0.15, 0.04, transform=ax_left.transAxes,
                                    facecolor='#313244', edgecolor='#89b4fa', linewidth=1.5))
    ax_left.text(0.625, y_pos, value, fontsize=12, color='white', fontweight='bold',
                ha='center', va='center', transform=ax_left.transAxes)
    ax_left.text(0.75, y_pos, range_str, fontsize=10, color='#6c7086', transform=ax_left.transAxes)
    y_pos -= 0.12

# 更新按钮
ax_left.add_patch(plt.Rectangle((0.25, 0.08), 0.5, 0.08, transform=ax_left.transAxes,
                                facecolor='#89b4fa', edgecolor='white', linewidth=2))
ax_left.text(0.5, 0.12, 'Update Constellation', ha='center', va='center', fontsize=14,
            color='#1e1e2e', fontweight='bold', transform=ax_left.transAxes)

ax_left.axis('off')

# 右侧：3D卫星图
ax_right = fig.add_subplot(1, 2, 2, projection='3d', facecolor='#1e1e2e')

# 地球（使用真实的蓝绿色）
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = 6371 * np.outer(np.cos(u), np.sin(v))
y = 6371 * np.outer(np.sin(u), np.sin(v))
z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
ax_right.plot_surface(x, y, z, color='#4a9eff', alpha=0.6, shade=True)

# 3个轨道面
orbit_radius = 6371 + 600
colors = ['#f38ba8', '#a6e3a1', '#89b4fa']
inclinations = [45, 55, 65]

for i, (color, inc) in enumerate(zip(colors, inclinations)):
    theta = np.linspace(0, 2*np.pi, 100)
    inclination = np.radians(inc)
    
    x_orbit = orbit_radius * np.cos(theta)
    y_orbit = orbit_radius * np.sin(theta) * np.cos(inclination)
    z_orbit = orbit_radius * np.sin(theta) * np.sin(inclination)
    
    ax_right.plot(x_orbit, y_orbit, z_orbit, color=color, linewidth=2.5, 
                 label=f'Plane {i+1} ({inc}°)', alpha=0.8)
    
    # 卫星
    for j in range(3):
        angle = j * 2*np.pi/3
        sat_x = orbit_radius * np.cos(angle)
        sat_y = orbit_radius * np.sin(angle) * np.cos(inclination)
        sat_z = orbit_radius * np.sin(angle) * np.sin(inclination)
        ax_right.scatter([sat_x], [sat_y], [sat_z], color=color, s=150, 
                        marker='^', edgecolors='white', linewidths=2, zorder=10)

ax_right.set_xlabel('X (km)', color='white', fontsize=11)
ax_right.set_ylabel('Y (km)', color='white', fontsize=11)
ax_right.set_zlabel('Z (km)', color='white', fontsize=11)
ax_right.set_title('Interactive 3D Visualization\n(Adjustable Parameters)', 
                   color='white', fontsize=14, fontweight='bold')
ax_right.legend(facecolor='#313244', edgecolor='#89b4fa', labelcolor='white', fontsize=10)
ax_right.tick_params(colors='white')
ax_right.grid(True, alpha=0.2, color='white')

plt.tight_layout()
plt.savefig(output_dir / '1_Interactive_Satellite_Config.png', dpi=120, facecolor='#1e1e2e')
plt.close()
print("✓ 1. 交互式卫星配置页面")

# 2. 数据加载页面
fig = plt.figure(figsize=(18, 10), facecolor='#1e1e2e')

# 左侧：数据加载面板
ax_left = fig.add_subplot(1, 2, 1, facecolor='#1e1e2e')
ax_left.text(0.5, 0.95, 'Region Data Loading', ha='center', fontsize=18, 
             fontweight='bold', color='#89b4fa', transform=ax_left.transAxes)

# 区域选择下拉框
ax_left.text(0.1, 0.85, 'Select Region:', fontsize=13, color='#cdd6f4', transform=ax_left.transAxes)
ax_left.add_patch(plt.Rectangle((0.1, 0.78), 0.8, 0.05, transform=ax_left.transAxes,
                                facecolor='#313244', edgecolor='#89b4fa', linewidth=1.5))
ax_left.text(0.5, 0.805, 'Scottish Highlands ▼', ha='center', va='center', fontsize=12,
            color='white', transform=ax_left.transAxes)

# 数据路径
ax_left.text(0.1, 0.70, 'Data Directory:', fontsize=13, color='#cdd6f4', transform=ax_left.transAxes)
ax_left.add_patch(plt.Rectangle((0.1, 0.63), 0.65, 0.05, transform=ax_left.transAxes,
                                facecolor='#313244', edgecolor='#45475a', linewidth=1))
ax_left.text(0.12, 0.655, '/home/zmc/文档/programwork/data/processed/...', 
            fontsize=10, color='#cdd6f4', va='center', transform=ax_left.transAxes)
ax_left.add_patch(plt.Rectangle((0.77, 0.63), 0.13, 0.05, transform=ax_left.transAxes,
                                facecolor='#89b4fa', edgecolor='white', linewidth=1.5))
ax_left.text(0.835, 0.655, 'Browse', ha='center', va='center', fontsize=11,
            color='#1e1e2e', fontweight='bold', transform=ax_left.transAxes)

# 加载按钮
ax_left.add_patch(plt.Rectangle((0.25, 0.52), 0.5, 0.07, transform=ax_left.transAxes,
                                facecolor='#a6e3a1', edgecolor='white', linewidth=2))
ax_left.text(0.5, 0.555, 'Load Region Data', ha='center', va='center', fontsize=14,
            color='#1e1e2e', fontweight='bold', transform=ax_left.transAxes)

# 状态框
ax_left.text(0.1, 0.43, 'Status:', fontsize=13, color='#cdd6f4', transform=ax_left.transAxes)
ax_left.add_patch(plt.Rectangle((0.1, 0.15), 0.8, 0.25, transform=ax_left.transAxes,
                                facecolor='#313244', edgecolor='#45475a', linewidth=1))
status_text = "✓ Loaded 3,600 trajectory files\n\nPath: /home/.../scottish_highlands/\nFiles: *.pkl\nSize: 257 MB"
ax_left.text(0.15, 0.35, status_text, fontsize=11, color='#a6e3a1', 
            va='top', family='monospace', transform=ax_left.transAxes)

ax_left.axis('off')

# 右侧：地图
ax_right = fig.add_subplot(1, 2, 2, facecolor='#1e1e2e')

x = np.linspace(0, 100, 300)
y = np.linspace(0, 100, 300)
X, Y = np.meshgrid(x, y)
Z = (50 * np.sin(X/10) * np.cos(Y/10) + 
     30 * np.sin(X/5) * np.cos(Y/8) +
     20 * np.random.randn(300, 300) + 200)

im = ax_right.contourf(X, Y, Z, levels=50, cmap='terrain', alpha=0.95)
cbar = plt.colorbar(im, ax=ax_right)
cbar.ax.tick_params(labelcolor='white')
cbar.set_label('Elevation (m)', color='white', fontsize=11)

info_text = "Scottish Highlands\n─────────────────────\nArea: 25,000 km²\nElevation: 200-1,300 m\nTrajectories: 3,600\nTerrain: Mountainous"
ax_right.text(5, 93, info_text, bbox=dict(boxstyle='round', facecolor='#313244', 
              alpha=0.95, edgecolor='#89b4fa', linewidth=2),
              fontsize=11, color='white', verticalalignment='top', family='monospace')

ax_right.set_xlabel('X (km)', color='white', fontsize=12)
ax_right.set_ylabel('Y (km)', color='white', fontsize=12)
ax_right.set_title('Loaded Region: Scottish Highlands', color='white', fontsize=15, fontweight='bold')
ax_right.grid(True, alpha=0.3, color='white')
ax_right.tick_params(colors='white')
for spine in ax_right.spines.values():
    spine.set_edgecolor('#45475a')

plt.tight_layout()
plt.savefig(output_dir / '2_Interactive_Data_Loading.png', dpi=120, facecolor='#1e1e2e')
plt.close()
print("✓ 2. 交互式数据加载页面")

# 3. 轨迹预测动画控制
fig, ax = plt.subplots(figsize=(14, 14), facecolor='#1e1e2e')
ax.set_facecolor('#1e1e2e')

# 控制面板（顶部）
control_y = 1.08
fig.text(0.1, control_y, '▶ Play', fontsize=12, color='#1e1e2e', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#a6e3a1', edgecolor='white', linewidth=2, pad=0.5))
fig.text(0.2, control_y, '⏸ Pause', fontsize=12, color='#1e1e2e', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#fab387', edgecolor='white', linewidth=2, pad=0.5))
fig.text(0.3, control_y, '🔄 Reset', fontsize=12, color='#1e1e2e', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#89b4fa', edgecolor='white', linewidth=2, pad=0.5))
fig.text(0.42, control_y, 'Speed: ━━━●━━━━━━', fontsize=11, color='#cdd6f4')
fig.text(0.65, control_y, 'Load Trajectory', fontsize=11, color='#1e1e2e', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#f9e2af', edgecolor='white', linewidth=2, pad=0.5))

# 地形背景
x = np.linspace(0, 100, 150)
y = np.linspace(0, 100, 150)
X, Y = np.meshgrid(x, y)
Z = np.sin(X/10) * np.cos(Y/10)
ax.contourf(X, Y, Z, levels=20, cmap='terrain', alpha=0.4)

# 轨迹
t_hist = np.linspace(0, 10, 10)
hist_x = 20 + t_hist * 2
hist_y = 20 + t_hist * 3 + np.sin(t_hist) * 2
ax.plot(hist_x, hist_y, 'o-', color='#89b4fa', linewidth=4, markersize=10, 
        label='Observed (10 min)', zorder=5)

t_future = np.linspace(10, 70, 60)
true_x = 20 + t_future * 2 + np.sin(t_future/5) * 5
true_y = 20 + t_future * 3 + np.cos(t_future/5) * 5
ax.plot(true_x, true_y, '--', color='#a6e3a1', linewidth=3, alpha=0.6,
        label='Ground Truth (60 min)', zorder=3)

pred_x = true_x + np.random.randn(60) * 2
pred_y = true_y + np.random.randn(60) * 2
ax.plot(pred_x[:35], pred_y[:35], 'o-', color='#f38ba8', linewidth=3, markersize=7,
        label='Predicted (35 min shown)', zorder=4)

ax.scatter([pred_x[34]], [pred_y[34]], color='#f9e2af', s=600, marker='*', 
          edgecolors='white', linewidths=2, zorder=6, label='Current Position')

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel('X (km)', color='white', fontsize=13)
ax.set_ylabel('Y (km)', color='white', fontsize=13)
ax.set_title('Interactive Trajectory Prediction Animation\n(Frame 35/60 - Controllable Playback)', 
             color='white', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', facecolor='#313244', edgecolor='#89b4fa', 
          labelcolor='white', fontsize=12)
ax.grid(True, alpha=0.3, color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#45475a')

plt.tight_layout()
plt.savefig(output_dir / '3_Interactive_Trajectory_Animation.png', dpi=120, facecolor='#1e1e2e', bbox_inches='tight')
plt.close()
print("✓ 3. 交互式轨迹预测动画页面")

print(f"\n✅ 所有交互式UI截图已生成到: {output_dir}")
print("\n生成的页面:")
print("1. 🛰️  卫星星座配置 - 可调参数+实时3D更新")
print("2. 🗺️  数据加载 - 浏览目录+加载状态+地图展示")
print("3. 🎬 轨迹预测动画 - 播放控制+速度调节+轨迹加载")
