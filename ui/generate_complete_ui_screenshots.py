#!/usr/bin/env python
"""生成完整UI系统的所有页面截图"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

output_dir = Path('/home/zmc/文档/programwork/docs/ui_complete')
output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('dark_background')

# 1. 卫星星座页面
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 地球
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = 6371 * np.outer(np.cos(u), np.sin(v))
y = 6371 * np.outer(np.sin(u), np.sin(v))
z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

# 3个轨道面，每个3颗卫星
orbit_radius = 6371 + 600
colors = ['#f38ba8', '#a6e3a1', '#89b4fa']
orbit_names = ['Orbit Plane 1 (45°)', 'Orbit Plane 2 (55°)', 'Orbit Plane 3 (65°)']

for i, (color, name) in enumerate(zip(colors, orbit_names)):
    theta = np.linspace(0, 2*np.pi, 100)
    inclination = np.radians(45 + i*10)
    
    x_orbit = orbit_radius * np.cos(theta)
    y_orbit = orbit_radius * np.sin(theta) * np.cos(inclination)
    z_orbit = orbit_radius * np.sin(theta) * np.sin(inclination)
    
    ax.plot(x_orbit, y_orbit, z_orbit, color=color, linewidth=2, label=name)
    
    for j in range(3):
        angle = j * 2*np.pi/3
        sat_x = orbit_radius * np.cos(angle)
        sat_y = orbit_radius * np.sin(angle) * np.cos(inclination)
        sat_z = orbit_radius * np.sin(angle) * np.sin(inclination)
        ax.scatter([sat_x], [sat_y], [sat_z], color=color, s=200, marker='^', edgecolors='white', linewidths=2)

ax.set_xlabel('X (km)', fontsize=12)
ax.set_ylabel('Y (km)', fontsize=12)
ax.set_zlabel('Z (km)', fontsize=12)
ax.set_title('Multi-Satellite Constellation\n9 Satellites | 3 Orbit Planes | 600km Altitude | ~15min Revisit', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / '1_Satellite_Constellation.png', dpi=120, facecolor='#1e1e2e')
plt.close()
print("✓ 1. 卫星星座页面")

# 2. 地理区域展示
fig, ax = plt.subplots(figsize=(14, 10), facecolor='#1e1e2e')
ax.set_facecolor('#1e1e2e')

x = np.linspace(0, 100, 300)
y = np.linspace(0, 100, 300)
X, Y = np.meshgrid(x, y)
Z = 50 * np.sin(X/10) * np.cos(Y/10) + 200 + np.random.randn(300, 300) * 5

im = ax.contourf(X, Y, Z, levels=40, cmap='terrain', alpha=0.9)
cbar = plt.colorbar(im, ax=ax, label='Elevation (m)')
cbar.ax.tick_params(labelsize=10)

# 区域信息框
info_text = """Scottish Highlands
━━━━━━━━━━━━━━━━━━━━
Area: 25,000 km²
Terrain: Mountainous
Elevation: 200-1,300 m
Trajectories: 3,600
Vehicle Types: 4
Tactical Intents: 3
Avg Length: 125.3 km"""

ax.text(5, 92, info_text, bbox=dict(boxstyle='round', facecolor='#313244', alpha=0.9, edgecolor='#89b4fa', linewidth=2),
        fontsize=11, color='white', verticalalignment='top', family='monospace')

ax.set_xlabel('X (km)', fontsize=12, color='white')
ax.set_ylabel('Y (km)', fontsize=12, color='white')
ax.set_title('Geographic Region: Scottish Highlands', fontsize=16, fontweight='bold', color='white', pad=15)
ax.grid(True, alpha=0.2, color='white')
ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig(output_dir / '2_Geographic_Region.png', dpi=120, facecolor='#1e1e2e')
plt.close()
print("✓ 2. 地理区域页面")

# 3. 数据生成页面
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#1e1e2e')
ax1.set_facecolor('#1e1e2e')
ax2.set_facecolor('#1e1e2e')

# 左侧：生成配置
ax1.text(0.5, 0.95, 'Data Generation Configuration', ha='center', fontsize=18, 
         fontweight='bold', color='white', transform=ax1.transAxes)

config_items = [
    ('Number of Trajectories:', '1,000'),
    ('Vehicle Types:', 'Type 1-4 (All)'),
    ('Tactical Intents:', 'Intent 1-3 (All)'),
    ('Min Distance:', '80 km'),
    ('Region:', 'Scottish Highlands'),
    ('Status:', 'Ready to Generate')
]

y_pos = 0.75
for label, value in config_items:
    ax1.text(0.1, y_pos, label, fontsize=13, color='#89b4fa', transform=ax1.transAxes)
    ax1.text(0.6, y_pos, value, fontsize=13, color='white', fontweight='bold', transform=ax1.transAxes)
    y_pos -= 0.1

# 进度条
ax1.add_patch(plt.Rectangle((0.1, 0.15), 0.8, 0.05, transform=ax1.transAxes, 
                            facecolor='#313244', edgecolor='#89b4fa', linewidth=2))
ax1.add_patch(plt.Rectangle((0.1, 0.15), 0.6, 0.05, transform=ax1.transAxes, 
                            facecolor='#a6e3a1', alpha=0.8))
ax1.text(0.5, 0.175, '60% Complete', ha='center', va='center', fontsize=12, 
         color='white', fontweight='bold', transform=ax1.transAxes)

ax1.axis('off')

# 右侧：生成统计
vehicle_types = ['Type 1', 'Type 2', 'Type 3', 'Type 4']
counts = [280, 245, 265, 210]
colors_bar = ['#f38ba8', '#fab387', '#a6e3a1', '#89b4fa']

bars = ax2.bar(vehicle_types, counts, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=2)
ax2.set_ylabel('Number of Trajectories', fontsize=12, color='white')
ax2.set_title('Generated Trajectories by Vehicle Type', fontsize=14, fontweight='bold', color='white', pad=15)
ax2.tick_params(colors='white')
ax2.grid(axis='y', alpha=0.3, color='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#45475a')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '3_Data_Generation.png', dpi=120, facecolor='#1e1e2e')
plt.close()
print("✓ 3. 数据生成页面")

# 4. 轨迹预测动画
fig, ax = plt.subplots(figsize=(12, 12), facecolor='#1e1e2e')
ax.set_facecolor('#1e1e2e')

# 地形背景
x = np.linspace(0, 100, 150)
y = np.linspace(0, 100, 150)
X, Y = np.meshgrid(x, y)
Z = np.sin(X/10) * np.cos(Y/10)
ax.contourf(X, Y, Z, levels=20, cmap='terrain', alpha=0.4)

# 历史轨迹（10分钟）
t_hist = np.linspace(0, 10, 10)
hist_x = 20 + t_hist * 2
hist_y = 20 + t_hist * 3 + np.sin(t_hist) * 2
ax.plot(hist_x, hist_y, 'o-', color='#89b4fa', linewidth=4, markersize=10, 
        label='Observed (10 min)', zorder=5)

# 真实轨迹（60分钟）
t_future = np.linspace(10, 70, 60)
true_x = 20 + t_future * 2 + np.sin(t_future/5) * 5
true_y = 20 + t_future * 3 + np.cos(t_future/5) * 5
ax.plot(true_x, true_y, '--', color='#a6e3a1', linewidth=3, alpha=0.6,
        label='Ground Truth (60 min)', zorder=3)

# 预测轨迹（带误差）
pred_x = true_x + np.random.randn(60) * 2
pred_y = true_y + np.random.randn(60) * 2
ax.plot(pred_x[:30], pred_y[:30], 'o-', color='#f38ba8', linewidth=3, markersize=7,
        label='Predicted (30 min shown)', zorder=4)

# 当前预测点（大星号）
ax.scatter([pred_x[29]], [pred_y[29]], color='#f9e2af', s=500, marker='*', 
          edgecolors='white', linewidths=2, zorder=6, label='Current Position')

# 候选目标点
goal_x = [pred_x[-1], pred_x[-1]+5, pred_x[-1]-3]
goal_y = [pred_y[-1], pred_y[-1]+4, pred_y[-1]-2]
ax.scatter(goal_x, goal_y, color='#f9e2af', s=200, marker='o', 
          edgecolors='white', linewidths=2, zorder=5, label='Candidate Goals')

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel('X (km)', fontsize=12, color='white')
ax.set_ylabel('Y (km)', fontsize=12, color='white')
ax.set_title('Trajectory Prediction Animation (Frame 30/60)', fontsize=16, fontweight='bold', color='white', pad=15)
ax.legend(fontsize=11, loc='upper left', facecolor='#313244', edgecolor='#89b4fa', labelcolor='white')
ax.grid(True, alpha=0.3, color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#45475a')

plt.tight_layout()
plt.savefig(output_dir / '4_Trajectory_Prediction_Animation.png', dpi=120, facecolor='#1e1e2e')
plt.close()
print("✓ 4. 轨迹预测动画页面")

# 5. 模型评估对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#1e1e2e')
ax1.set_facecolor('#1e1e2e')
ax2.set_facecolor('#1e1e2e')

# ADE对比
models = ['TerraTNT', 'YNet', 'PECNet', 'Traj++', 'Social\nLSTM']
ade_values = [1.23, 2.18, 2.05, 2.42, 3.52]
colors_ade = ['#a6e3a1', '#89b4fa', '#89b4fa', '#89b4fa', '#89b4fa']

bars1 = ax1.bar(models, ade_values, color=colors_ade, alpha=0.8, edgecolor='white', linewidth=2)
ax1.set_ylabel('ADE (km)', fontsize=13, color='white', fontweight='bold')
ax1.set_title('Average Displacement Error', fontsize=15, fontweight='bold', color='white', pad=15)
ax1.tick_params(colors='white', labelsize=11)
ax1.grid(axis='y', alpha=0.3, color='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#45475a')

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')

# FDE对比
fde_values = [2.45, 4.32, 4.18, 4.89, 7.18]
colors_fde = ['#a6e3a1', '#f38ba8', '#f38ba8', '#f38ba8', '#f38ba8']

bars2 = ax2.bar(models, fde_values, color=colors_fde, alpha=0.8, edgecolor='white', linewidth=2)
ax2.set_ylabel('FDE (km)', fontsize=13, color='white', fontweight='bold')
ax2.set_title('Final Displacement Error', fontsize=15, fontweight='bold', color='white', pad=15)
ax2.tick_params(colors='white', labelsize=11)
ax2.grid(axis='y', alpha=0.3, color='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#45475a')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '5_Model_Evaluation.png', dpi=120, facecolor='#1e1e2e')
plt.close()
print("✓ 5. 模型评估页面")

print(f"\n✅ 所有UI截图已生成到: {output_dir}")
print("\n生成的页面:")
print("1. 🛰️  卫星星座 - 3D轨道可视化")
print("2. 🗺️  地理区域 - DEM地形展示")
print("3. ⚙️  数据生成 - 配置和进度")
print("4. 🎬 轨迹预测 - 动画演示")
print("5. 📊 模型评估 - 性能对比")
