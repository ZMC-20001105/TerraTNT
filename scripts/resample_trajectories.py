#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重采样轨迹数据到固定时间间隔
将23秒间隔的轨迹重采样到10秒间隔
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import pickle
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from tqdm import tqdm

def resample_trajectory(timestamps, path, speeds, target_interval=10.0):
    """
    重采样轨迹到目标时间间隔
    
    Args:
        timestamps: 原始时间戳数组
        path: 原始路径点 [(x, y), ...]
        speeds: 原始速度数组
        target_interval: 目标时间间隔（秒）
    
    Returns:
        new_timestamps, new_path, new_speeds
    """
    # 转换为numpy数组
    timestamps = np.array(timestamps)
    path = np.array([(p[0], p[1]) for p in path])
    speeds = np.array(speeds)
    
    # 创建插值函数
    interp_x = interp1d(timestamps, path[:, 0], kind='linear', fill_value='extrapolate')
    interp_y = interp1d(timestamps, path[:, 1], kind='linear', fill_value='extrapolate')
    interp_speed = interp1d(timestamps, speeds, kind='linear', fill_value='extrapolate')
    
    # 生成新的时间戳
    start_time = timestamps[0]
    end_time = timestamps[-1]
    new_timestamps = np.arange(start_time, end_time, target_interval)
    
    # 插值得到新的路径和速度
    new_x = interp_x(new_timestamps)
    new_y = interp_y(new_timestamps)
    new_path = [(float(x), float(y)) for x, y in zip(new_x, new_y)]
    new_speeds = interp_speed(new_timestamps).tolist()
    
    return new_timestamps.tolist(), new_path, new_speeds


def resample_dataset(input_dir, output_dir, target_interval=10.0):
    """
    重采样整个数据集
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(input_dir.glob('*.pkl'))
    print(f"找到 {len(files)} 个轨迹文件")
    print(f"目标采样间隔: {target_interval} 秒")
    
    stats = {
        'original_intervals': [],
        'original_points': [],
        'resampled_points': []
    }
    
    for traj_file in tqdm(files, desc="重采样轨迹"):
        try:
            with open(traj_file, 'rb') as f:
                data = pickle.load(f)
            
            # 记录原始统计
            original_interval = data['timestamps'][1] - data['timestamps'][0] if len(data['timestamps']) > 1 else 0
            stats['original_intervals'].append(original_interval)
            stats['original_points'].append(len(data['path']))
            
            # 重采样
            new_timestamps, new_path, new_speeds = resample_trajectory(
                data['timestamps'],
                data['path'],
                data['speeds'],
                target_interval
            )
            
            # 更新数据
            data['timestamps'] = new_timestamps
            data['path'] = new_path
            data['speeds'] = new_speeds
            data['num_points'] = len(new_path)
            data['duration'] = new_timestamps[-1] - new_timestamps[0]
            
            # 重新计算长度
            total_length = 0
            for i in range(len(new_path) - 1):
                p1 = np.array(new_path[i])
                p2 = np.array(new_path[i + 1])
                total_length += np.linalg.norm(p2 - p1)
            data['length'] = total_length
            
            stats['resampled_points'].append(len(new_path))
            
            # 保存
            output_file = output_dir / traj_file.name
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            print(f"处理 {traj_file.name} 失败: {e}")
            continue
    
    # 打印统计信息
    print(f"\n=== 重采样统计 ===")
    print(f"平均原始间隔: {np.mean(stats['original_intervals']):.2f} 秒")
    print(f"平均原始点数: {np.mean(stats['original_points']):.0f}")
    print(f"平均重采样点数: {np.mean(stats['resampled_points']):.0f}")
    print(f"点数变化: {np.mean(stats['resampled_points']) / np.mean(stats['original_points']):.2f}x")
    print(f"\n保存到: {output_dir}")


if __name__ == '__main__':
    regions = ['bohemian_forest', 'scottish_highlands']
    
    for region in regions:
        input_dir = f'/home/zmc/文档/programwork/data/processed/synthetic_trajectories/{region}'
        output_dir = f'/home/zmc/文档/programwork/data/processed/synthetic_trajectories_10s/{region}'
        
        if Path(input_dir).exists():
            print(f"\n处理区域: {region}")
            resample_dataset(input_dir, output_dir, target_interval=10.0)
        else:
            print(f"\n跳过 {region}: 目录不存在")
    
    print("\n✓ 重采样完成！")
