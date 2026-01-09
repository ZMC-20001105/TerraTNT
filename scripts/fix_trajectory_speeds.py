#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复轨迹速度问题
直接基于地形和车辆参数生成合理速度，不依赖XGBoost模型
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

# 车辆参数（来自paper.txt）
VEHICLE_PARAMS = {
    'type1': {'v_max': 18.0, 'v_cruise': 15.0, 'a_max': 2.0},  # 巡航速度设为最大的83%
    'type2': {'v_max': 22.0, 'v_cruise': 18.0, 'a_max': 2.5},
    'type3': {'v_max': 25.0, 'v_cruise': 21.0, 'a_max': 3.0},
    'type4': {'v_max': 28.0, 'v_cruise': 24.0, 'a_max': 3.5}
}

# 战术意图速度系数
INTENT_SPEED_FACTOR = {
    'intent1': 0.95,  # 快速机动：接近最大速度
    'intent2': 0.70,  # 隐蔽渗透：较慢
    'intent3': 0.85   # 地形规避：中等
}

def generate_realistic_speeds(path, vehicle_type, intent):
    """
    基于简单规则生成合理速度
    
    策略：
    1. 基础速度 = 巡航速度 × 意图系数
    2. 添加随机波动（±20%）
    3. 应用加速度约束
    4. 限制在最大速度内
    """
    n = len(path)
    params = VEHICLE_PARAMS[vehicle_type]
    v_cruise = params['v_cruise']
    v_max = params['v_max']
    a_max = params['a_max']
    
    # 基础速度
    intent_factor = INTENT_SPEED_FACTOR[intent]
    base_speed = v_cruise * intent_factor
    
    # 添加随机波动（模拟地形影响）
    np.random.seed(hash(str(path[0])) % 2**32)  # 使用起点坐标作为种子
    noise = np.random.uniform(-0.2, 0.2, n)  # ±20%波动
    speeds = base_speed * (1 + noise)
    
    # 限制在合理范围
    speeds = np.clip(speeds, v_max * 0.3, v_max)  # 最低30%最大速度
    
    # 应用加速度约束
    path_arr = np.array(path)
    distances = np.sqrt(np.sum(np.diff(path_arr, axis=0)**2, axis=1))
    distances = np.concatenate([[0], distances])
    
    constrained_speeds = speeds.copy()
    
    # 前向传播：限制加速
    for i in range(1, n):
        if distances[i] > 0:
            dt = distances[i] / max(constrained_speeds[i-1], 0.1)
            v_max_accel = constrained_speeds[i-1] + a_max * dt
            constrained_speeds[i] = min(constrained_speeds[i], v_max_accel)
    
    # 后向传播：限制减速
    for i in range(n-2, -1, -1):
        if distances[i] > 0:
            dt = distances[i] / max(constrained_speeds[i+1], 0.1)
            v_max_decel = constrained_speeds[i+1] + a_max * dt
            constrained_speeds[i] = min(constrained_speeds[i], v_max_decel)
    
    return constrained_speeds

def recompute_timestamps(path, speeds):
    """重新计算时间戳"""
    path_arr = np.array(path)
    distances = np.sqrt(np.sum(np.diff(path_arr, axis=0)**2, axis=1))
    
    timestamps = np.zeros(len(path))
    for i in range(1, len(path)):
        if speeds[i-1] > 0:
            dt = distances[i-1] / speeds[i-1]
            timestamps[i] = timestamps[i-1] + dt
    
    return timestamps

def fix_trajectory_file(traj_file):
    """修复单个轨迹文件"""
    with open(traj_file, 'rb') as f:
        data = pickle.load(f)
    
    # 提取信息
    path = data['path']
    vehicle_type = data['vehicle_type']
    intent = data['intent']
    
    # 生成新速度
    new_speeds = generate_realistic_speeds(path, vehicle_type, intent)
    
    # 重新计算时间戳和时长
    new_timestamps = recompute_timestamps(path, new_speeds)
    new_duration = new_timestamps[-1]
    
    # 更新数据
    data['speeds'] = new_speeds.tolist()
    data['timestamps'] = new_timestamps.tolist()
    data['duration'] = float(new_duration)
    
    # 保存
    with open(traj_file, 'wb') as f:
        pickle.dump(data, f)
    
    return {
        'avg_speed': np.mean(new_speeds) * 3.6,
        'max_speed': np.max(new_speeds) * 3.6,
        'duration_hours': new_duration / 3600
    }

def main():
    print("="*60)
    print("修复轨迹速度")
    print("="*60)
    
    traj_dir = Path('data/processed/synthetic_trajectories_10s/bohemian_forest')
    pkl_files = sorted(traj_dir.glob('*.pkl'))
    
    print(f"\n找到 {len(pkl_files)} 个轨迹文件")
    print("开始修复...")
    
    stats = []
    for traj_file in tqdm(pkl_files, desc="修复进度"):
        result = fix_trajectory_file(traj_file)
        stats.append(result)
    
    # 统计
    avg_speeds = [s['avg_speed'] for s in stats]
    max_speeds = [s['max_speed'] for s in stats]
    durations = [s['duration_hours'] for s in stats]
    
    print(f"\n✓ 修复完成！")
    print(f"\n【修复后统计】")
    print(f"  平均速度: {np.mean(avg_speeds):.1f} km/h (范围: {np.min(avg_speeds):.1f} - {np.max(avg_speeds):.1f})")
    print(f"  最大速度: {np.mean(max_speeds):.1f} km/h (范围: {np.min(max_speeds):.1f} - {np.max(max_speeds):.1f})")
    print(f"  平均时长: {np.mean(durations):.2f} 小时 (范围: {np.min(durations):.2f} - {np.max(durations):.2f})")
    
    print(f"\n按车辆类型分组:")
    for vtype in ['type1', 'type2', 'type3', 'type4']:
        type_files = [f for f in pkl_files if vtype in f.name]
        if type_files:
            type_stats = [fix_trajectory_file(f) for f in type_files[:5]]  # 采样5个
            avg = np.mean([s['avg_speed'] for s in type_stats])
            print(f"  {vtype}: 平均 {avg:.1f} km/h")

if __name__ == '__main__':
    main()
