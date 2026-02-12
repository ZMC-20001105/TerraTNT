#!/usr/bin/env python3
"""
可视化训练结果 - 从训练数据中随机抽取样本展示
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config.plot_config import get_plot_config

plot_cfg = get_plot_config()

def load_samples(traj_dir, split_file, phase='fas2', num_samples=10):
    """加载验证集样本"""
    traj_dir = Path(traj_dir)
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    file_list = splits[phase]['files']
    random.seed(42)
    random.shuffle(file_list)
    
    samples = []
    for file_name in tqdm(file_list[:100], desc=f"加载{phase}样本"):
        traj_file = traj_dir / file_name
        try:
            with open(traj_file, 'rb') as f:
                data = pickle.load(f)
            
            # 提取轨迹元数据
            traj_info = {
                'file': file_name,
                'intent': data.get('intent', 'unknown'),
                'vehicle': data.get('vehicle_type', 'unknown'),
                'length_km': data.get('stage_stats', {}).get('resampled_10s', {}).get('length_km', 0),
                'duration_min': data.get('stage_stats', {}).get('resampled_10s', {}).get('duration_min', 0),
                'sinuosity': data.get('stage_stats', {}).get('resampled_10s', {}).get('sinuosity', 0),
                'total_turn_deg': data.get('stage_stats', {}).get('resampled_10s', {}).get('curvature', {}).get('total_turn_deg', 0),
            }
            
            for s in data.get('samples', []):
                future_rel = np.asarray(s.get('future_rel'), dtype=np.float32)
                history_feat_26d = np.asarray(s.get('history_feat_26d'), dtype=np.float32)
                
                if future_rel.shape[0] != 360:
                    continue
                if history_feat_26d.shape[0] != 90:
                    continue
                
                samples.append({
                    'history_xy': history_feat_26d[:, :2],
                    'future_xy': future_rel,
                    'info': traj_info,
                })
                
                if len(samples) >= num_samples:
                    return samples
        except Exception:
            continue
    
    return samples

def visualize_samples(samples, output_dir):
    """可视化样本 - 展示历史轨迹和未来轨迹"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n生成可视化 ({len(samples)} 个样本)...")
    
    for idx, sample in enumerate(tqdm(samples, desc="生成图像")):
        history_xy = sample['history_xy']
        future_xy = sample['future_xy']
        info = sample['info']
        
        # 累积轨迹
        history_cumsum = np.cumsum(history_xy, axis=0)
        future_cumsum = np.cumsum(future_xy, axis=0)
        
        # 创建图像
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左图：完整轨迹
        ax = axes[0]
        ax.plot(history_cumsum[:, 0]/1000, history_cumsum[:, 1]/1000, 
               'o-', color='blue', linewidth=2.5, markersize=3, 
               label='历史轨迹 (15分钟)', alpha=0.8)
        ax.plot(future_cumsum[:, 0]/1000, future_cumsum[:, 1]/1000, 
               's-', color='green', linewidth=2.5, markersize=2, 
               label='未来轨迹 (60分钟)', alpha=0.8)
        ax.plot(0, 0, 'r*', markersize=20, label='当前位置', zorder=10)
        ax.plot(future_cumsum[-1, 0]/1000, future_cumsum[-1, 1]/1000, 
               'g*', markersize=20, label='目标点', zorder=10)
        
        ax.set_xlabel('东向位移 (km)', fontsize=13)
        ax.set_ylabel('北向位移 (km)', fontsize=13)
        ax.set_title(f'样本 {idx+1}: 完整轨迹\n意图={info["intent"]}, 车辆={info["vehicle"]}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 右图：轨迹特征分析
        ax = axes[1]
        
        # 绘制轨迹
        ax.plot(history_cumsum[:, 0]/1000, history_cumsum[:, 1]/1000, 
               'o-', color='blue', linewidth=2, markersize=2, 
               label='历史', alpha=0.7)
        ax.plot(future_cumsum[:, 0]/1000, future_cumsum[:, 1]/1000, 
               's-', color='green', linewidth=2, markersize=1.5, 
               label='未来', alpha=0.7)
        
        # 标注时间点（每10分钟）
        time_markers = [0, 60, 120, 180, 240, 300, 360]  # 0, 10, 20, 30, 40, 50, 60分钟
        for t in time_markers:
            if t < len(future_cumsum):
                ax.plot(future_cumsum[t, 0]/1000, future_cumsum[t, 1]/1000, 
                       'ro', markersize=8, alpha=0.6)
                ax.text(future_cumsum[t, 0]/1000, future_cumsum[t, 1]/1000, 
                       f' {t//6}min', fontsize=9, color='red')
        
        ax.plot(0, 0, 'r*', markersize=20, zorder=10)
        
        # 计算统计信息
        total_dist = np.linalg.norm(future_cumsum[-1])
        straight_dist = np.linalg.norm(future_cumsum[-1])
        path_length = np.sum(np.linalg.norm(np.diff(future_cumsum, axis=0), axis=1))
        actual_sinuosity = path_length / (straight_dist + 1e-6)
        
        info_text = (
            f'轨迹长度: {info["length_km"]:.1f} km\n'
            f'持续时间: {info["duration_min"]:.1f} min\n'
            f'迂回系数: {info["sinuosity"]:.3f}\n'
            f'累计转角: {info["total_turn_deg"]:.0f}°\n'
            f'直线距离: {total_dist/1000:.1f} km'
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('东向位移 (km)', fontsize=13)
        ax.set_ylabel('北向位移 (km)', fontsize=13)
        ax.set_title('轨迹特征分析', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ 可视化完成!")
    print(f"输出目录: {output_dir}")
    
    # 生成统计汇总
    generate_summary(samples, output_dir)

def generate_summary(samples, output_dir):
    """生成统计汇总图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取统计数据
    lengths = [s['info']['length_km'] for s in samples]
    durations = [s['info']['duration_min'] for s in samples]
    sinuosities = [s['info']['sinuosity'] for s in samples]
    turns = [s['info']['total_turn_deg'] for s in samples]
    
    # 轨迹长度分布
    ax = axes[0, 0]
    ax.hist(lengths, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'均值={np.mean(lengths):.1f}km')
    ax.set_xlabel('轨迹长度 (km)', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('轨迹长度分布', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 持续时间分布
    ax = axes[0, 1]
    ax.hist(durations, bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(durations), color='red', linestyle='--', linewidth=2, label=f'均值={np.mean(durations):.1f}min')
    ax.set_xlabel('持续时间 (分钟)', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('持续时间分布', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 迂回系数分布
    ax = axes[1, 0]
    ax.hist(sinuosities, bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(sinuosities), color='red', linestyle='--', linewidth=2, label=f'均值={np.mean(sinuosities):.3f}')
    ax.axvline(1.08, color='green', linestyle=':', linewidth=2, label='阈值=1.08')
    ax.set_xlabel('迂回系数', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('迂回系数分布', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 累计转角分布
    ax = axes[1, 1]
    ax.hist(turns, bins=15, color='plum', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(turns), color='red', linestyle='--', linewidth=2, label=f'均值={np.mean(turns):.0f}°')
    ax.set_xlabel('累计转角 (度)', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('累计转角分布', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 统计汇总已保存")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='bohemian_forest')
    parser.add_argument('--traj_dir', type=str)
    parser.add_argument('--split_file', type=str)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations/training_samples')
    
    args = parser.parse_args()
    
    if not args.traj_dir:
        args.traj_dir = f'data/processed/complete_dataset_10s_full/{args.region}'
    if not args.split_file:
        args.split_file = f'data/processed/fas_splits/{args.region}_full/fas_splits.json'
    
    print("="*60)
    print("训练样本可视化")
    print("="*60)
    print(f"数据: {args.traj_dir}")
    print(f"样本数: {args.num_samples}")
    print(f"输出: {args.output_dir}")
    print("="*60)
    
    # 加载样本
    samples = load_samples(args.traj_dir, args.split_file, phase='fas2', num_samples=args.num_samples)
    print(f"✓ 已加载 {len(samples)} 个样本")
    
    # 可视化
    visualize_samples(samples, args.output_dir)

if __name__ == '__main__':
    main()
