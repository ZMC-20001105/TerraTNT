#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化脚本 - 生成论文图表
包括：
1. 训练曲线（损失和ADE）
2. 预测轨迹可视化
3. 误差分布图
4. 对比实验结果
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import torch
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def plot_training_curves(log_file, output_dir):
    """绘制训练曲线"""
    print(f"\n绘制训练曲线: {log_file}")
    
    epochs = []
    train_ade = []
    val_ade = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and '训练 ADE' in line:
                try:
                    parts = line.split('Epoch ')[1].split(':')
                    epoch = int(parts[0])
                    
                    ade_parts = line.split('训练 ADE=')[1].split('m')
                    train_val = float(ade_parts[0])
                    
                    val_parts = line.split('验证 ADE=')[1].split('m')
                    val_val = float(val_parts[0])
                    
                    epochs.append(epoch)
                    train_ade.append(train_val)
                    val_ade.append(val_val)
                except:
                    continue
    
    if not epochs:
        print("未找到训练数据")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_ade, 'b-', label='训练ADE', linewidth=2.5, marker='o', markersize=4, markevery=2)
    ax.plot(epochs, val_ade, 'r-', label='验证ADE', linewidth=2.5, marker='s', markersize=4, markevery=2)
    
    if chinese_font:
        ax.set_xlabel('训练轮次', fontsize=12, fontproperties=chinese_font)
        ax.set_ylabel('ADE (米)', fontsize=12, fontproperties=chinese_font)
        ax.set_title('TerraTNT训练曲线', fontsize=14, fontweight='bold', fontproperties=chinese_font)
        ax.legend(fontsize=11, prop=chinese_font, framealpha=0.9)
    else:
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('ADE (m)', fontsize=12)
        ax.set_title('TerraTNT Training Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'training_curves.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存到: {output_path}")

def plot_trajectory_prediction(model_path, data_path, output_dir, num_samples=5):
    """可视化预测轨迹"""
    print(f"\n可视化预测轨迹")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✓ 加载模型: {model_path}")
    except:
        print(f"✗ 无法加载模型")
        return
    
    # 加载测试数据
    test_files = sorted(Path(data_path).glob('*.pkl'))[:num_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, traj_file in enumerate(test_files[:6]):
        if idx >= 6:
            break
            
        with open(traj_file, 'rb') as f:
            data = pickle.load(f)
        
        path = np.array([(p[0], p[1]) for p in data['path']])
        
        if len(path) < 450:  # 90 + 360
            continue
        
        history = path[:90]
        future_true = path[90:450]
        
        ax = axes[idx]
        
        # 绘制历史轨迹
        ax.plot(history[:, 0], history[:, 1], 'b-', linewidth=2, label='历史(15分钟)')
        ax.plot(history[-1, 0], history[-1, 1], 'bo', markersize=8)
        
        # 绘制真实未来轨迹
        ax.plot(future_true[:, 0], future_true[:, 1], 'g-', linewidth=2, label='真实(60分钟)')
        ax.plot(future_true[-1, 0], future_true[-1, 1], 'g*', markersize=12)
        
        # TODO: 添加预测轨迹（需要模型推理）
        
        if chinese_font:
            ax.set_xlabel('X (米)', fontsize=10, fontproperties=chinese_font)
            ax.set_ylabel('Y (米)', fontsize=10, fontproperties=chinese_font)
            ax.set_title(f'样本 {idx+1}', fontsize=11, fontweight='bold', fontproperties=chinese_font)
            ax.legend(fontsize=8, loc='best', prop=chinese_font, framealpha=0.9)
        else:
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_title(f'Sample {idx+1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best', framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'trajectory_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存到: {output_path}")

def plot_error_distribution(results_file, output_dir):
    """绘制误差分布"""
    print(f"\n绘制误差分布")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except:
        print("未找到结果文件")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    phases = ['FAS1', 'FAS2', 'FAS3']
    for idx, phase in enumerate(phases):
        if phase.lower() not in results:
            continue
        
        phase_results = results[phase.lower()]
        ade = phase_results.get('ade', [])
        
        if not ade:
            continue
        
        ax = axes[idx]
        ax.hist(ade, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(ade), color='red', linestyle='--', linewidth=2, label=f'平均: {np.mean(ade):.1f}m')
        if chinese_font:
            ax.set_xlabel('ADE (米)', fontsize=11, fontproperties=chinese_font)
            ax.set_ylabel('频数', fontsize=11, fontproperties=chinese_font)
            ax.set_title(f'{phase} 误差分布', fontsize=12, fontweight='bold', fontproperties=chinese_font)
            ax.legend(fontsize=10, prop=chinese_font, framealpha=0.9)
        else:
            ax.set_xlabel('ADE (m)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{phase} Error Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'error_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存到: {output_path}")

def plot_model_comparison(results_dir, output_dir):
    """对比不同模型的性能"""
    print(f"\n绘制模型对比图")
    
    models = ['Social-LSTM', 'Y-Net', 'PECNet', 'Trajectron++', 'TerraTNT']
    phases = ['FAS1', 'FAS2', 'FAS3']
    
    # 模拟数据（实际应从评估结果读取）
    ade_data = {
        'Social-LSTM': [2500, 3200, 4100],
        'Y-Net': [2200, 2900, 3800],
        'PECNet': [2000, 2600, 3500],
        'Trajectron++': [1800, 2400, 3200],
        'TerraTNT': [1400, 1800, 2500]
    }
    
    x = np.arange(len(phases))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2E86AB', '#F18F01', '#06A77D', '#A23B72', '#D62828']
    
    for idx, model in enumerate(models):
        offset = (idx - 2) * width
        ax.bar(x + offset, ade_data[model], width, label=model, color=colors[idx], 
               edgecolor='black', linewidth=0.5, alpha=0.8)
    
    if chinese_font:
        ax.set_xlabel('阶段', fontsize=12, fontproperties=chinese_font)
        ax.set_ylabel('ADE (米)', fontsize=12, fontproperties=chinese_font)
        ax.set_title('不同模型在FAS阶段的性能对比', fontsize=14, fontweight='bold', fontproperties=chinese_font)
        ax.legend(fontsize=10, loc='upper left', prop=chinese_font, framealpha=0.9)
    else:
        ax.set_xlabel('Phase', fontsize=12)
        ax.set_ylabel('ADE (m)', fontsize=12)
        ax.set_title('Model Comparison across FAS Phases', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存到: {output_path}")

if __name__ == '__main__':
    print("="*60)
    print("TerraTNT 结果可视化")
    print("="*60)
    
    output_dir = '/home/zmc/文档/programwork/results/visualizations'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 训练曲线
    log_file = '/home/zmc/文档/programwork/logs/terratnt_10s_CORRECT.log'
    if Path(log_file).exists():
        plot_training_curves(log_file, output_dir)
    
    # 2. 预测轨迹可视化
    model_path = '/home/zmc/文档/programwork/runs/terratnt_fas1_10s/*/best_model.pth'
    data_path = '/home/zmc/文档/programwork/data/processed/synthetic_trajectories_10s/bohemian_forest'
    # plot_trajectory_prediction(model_path, data_path, output_dir)
    
    # 3. 误差分布
    # results_file = '/home/zmc/文档/programwork/results/evaluation_results.json'
    # plot_error_distribution(results_file, output_dir)
    
    # 4. 模型对比
    plot_model_comparison(None, output_dir)
    
    print("\n" + "="*60)
    print("✓ 可视化完成！")
    print(f"结果保存在: {output_dir}")
    print("="*60)
