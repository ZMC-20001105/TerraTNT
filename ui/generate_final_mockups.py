import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_modern_ui_mockups():
    """使用Matplotlib生成现代化的UI视觉原型图"""
    output_dir = Path('/home/zmc/文档/programwork/docs/ui_new_design')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 概览主界面 (Overview)
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1e1e2e')
    ax.set_facecolor('#1e1e2e')
    plt.title('TerraTNT: Ground Target Trajectory Prediction System', color='white', fontsize=20, pad=20)
    
    # 模拟仪表盘
    circles = [
        {'pos': (0.2, 0.6), 'val': '95%', 'label': 'GPU Load', 'color': '#f38ba8'},
        {'pos': (0.5, 0.6), 'val': '1.2GB', 'label': 'VRAM Used', 'color': '#fab387'},
        {'pos': (0.8, 0.6), 'val': '12/15', 'label': 'Models Ready', 'color': '#a6e3a1'}
    ]
    
    for c in circles:
        circle = plt.Circle(c['pos'], 0.12, color=c['color'], alpha=0.2)
        ax.add_artist(circle)
        ax.text(c['pos'][0], c['pos'][1], c['val'], color='white', ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(c['pos'][0], c['pos'][1]-0.18, c['label'], color='#cdd6f4', ha='center', va='center', fontsize=14)

    # 模拟最近活动列表
    ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.3, color='#313244', alpha=0.5))
    ax.text(0.15, 0.35, 'Recent Activities', color='white', fontsize=16, fontweight='bold')
    activities = [
        '● [17:45] Social-LSTM Training Epoch 11 completed. Val ADE: 3623m',
        '● [17:47] Chapter 3 Experiments: XGBoost K-fold verification done.',
        '● [17:31] Training session started on NVIDIA GeForce RTX 5060.'
    ]
    for i, act in enumerate(activities):
        ax.text(0.15, 0.28 - i*0.06, act, color='#cdd6f4', fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / '1_Overview.png', dpi=100)
    plt.close()

    # 2. 模型训练界面 (Training)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), facecolor='#1e1e2e')
    ax1.set_facecolor('#1e1e2e')
    ax2.set_facecolor('#1e1e2e')
    
    # 左侧：配置参数预览
    ax1.text(0.1, 0.9, 'Training Configuration', color='white', fontsize=18, fontweight='bold')
    params = [
        ('Model:', 'TerraTNT (Main)'),
        ('Optimizer:', 'AdamW'),
        ('Learning Rate:', '0.0003'),
        ('Batch Size:', '64'),
        ('Loss Function:', 'TrajectoryMSE + GoalCE')
    ]
    for i, (k, v) in enumerate(params):
        ax1.text(0.1, 0.75 - i*0.1, k, color='#89b4fa', fontsize=14)
        ax1.text(0.5, 0.75 - i*0.1, v, color='white', fontsize=14)
    ax1.axis('off')

    # 右侧：实时Loss曲线模拟
    epochs = np.arange(1, 21)
    train_loss = 50 * np.exp(-epochs/5) + np.random.normal(0, 1, 20) + 5
    val_loss = 55 * np.exp(-epochs/6) + np.random.normal(0, 1, 20) + 8
    
    ax2.plot(epochs, train_loss, color='#89b4fa', label='Train Loss', linewidth=2)
    ax2.plot(epochs, val_loss, color='#f38ba8', label='Val Loss', linewidth=2)
    ax2.set_title('Real-time Loss Curve', color='white', fontsize=16)
    ax2.set_xlabel('Epoch', color='#cdd6f4')
    ax2.set_ylabel('Loss', color='#cdd6f4')
    ax2.legend()
    ax2.tick_params(colors='#cdd6f4')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#45475a')

    plt.tight_layout()
    plt.savefig(output_dir / '2_Training.png', dpi=100)
    plt.close()

    # 3. 轨迹预测界面 (Prediction)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#1e1e2e')
    ax.set_facecolor('#1e1e2e')
    
    # 模拟地图背景
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    ax.contourf(X, Y, Z, levels=15, cmap='terrain', alpha=0.4)
    
    # 历史轨迹
    hx = [2, 2.5, 3, 3.2, 3.5]
    hy = [2, 2.8, 3.5, 4.2, 5.0]
    ax.plot(hx, hy, 'o-', color='#89b4fa', label='Observed (10 min)', markersize=8, linewidth=3)
    
    # 预测轨迹 (多模态)
    px = [3.5, 4.0, 4.8, 5.5, 6.2]
    py = [5.0, 5.8, 6.5, 7.0, 7.5]
    ax.plot(px, py, '--', color='#f38ba8', label='Predicted (60 min)', linewidth=3)
    
    # 候选目标点
    ax.scatter([6.2, 5.8, 7.0], [7.5, 8.2, 6.8], color='#f9e2af', s=100, label='Candidate Goals', zorder=5)

    ax.set_title('Multi-modal Trajectory Prediction Visualization', color='white', fontsize=18)
    ax.legend(facecolor='#313244', labelcolor='white')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_Prediction.png', dpi=100)
    plt.close()

    print(f"Successfully generated 3 new UI mockups in {output_dir}")

if __name__ == '__main__':
    generate_modern_ui_mockups()
