#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
校准XGBoost速度预测结果
保留XGBoost的相对变化趋势，但将绝对值校准到合理范围

原理：
1. XGBoost捕捉了地形对速度的影响（相对关系正确）
2. 但绝对值偏低（可能是训练数据问题）
3. 使用线性变换校准：v_calibrated = a × v_xgboost + b
4. 参数a, b根据车辆类型和意图确定
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

# 车辆参数（来自paper.txt）
VEHICLE_PARAMS = {
    'type1': {'v_max': 18.0, 'v_target': 15.0, 'a_max': 2.0},  # 目标巡航速度
    'type2': {'v_max': 22.0, 'v_target': 18.0, 'a_max': 2.5},
    'type3': {'v_max': 25.0, 'v_target': 21.0, 'a_max': 3.0},
    'type4': {'v_max': 28.0, 'v_target': 24.0, 'a_max': 3.5}
}

# 战术意图速度系数
INTENT_FACTOR = {
    'intent1': 0.95,  # 快速机动
    'intent2': 0.70,  # 隐蔽渗透
    'intent3': 0.85   # 地形规避
}

def compute_calibration_params(v_xgboost_mean, vehicle_type, intent):
    """
    计算线性校准参数
    
    目标：v_calibrated = a × v_xgboost + b
    使得校准后的平均速度接近目标速度
    
    Args:
        v_xgboost_mean: XGBoost预测的平均速度
        vehicle_type: 车辆类型
        intent: 战术意图
    
    Returns:
        (a, b): 线性变换参数
    """
    params = VEHICLE_PARAMS[vehicle_type]
    v_target = params['v_target'] * INTENT_FACTOR[intent]
    v_max = params['v_max']
    
    # 线性变换：让XGBoost的平均值映射到目标值
    # v_target = a × v_xgboost_mean + b
    # 同时保证最大值不超过v_max
    
    # 策略：固定b=0，只缩放
    # a = v_target / v_xgboost_mean
    a = v_target / v_xgboost_mean
    b = 0.0
    
    return a, b

def calibrate_speeds(speeds_xgboost, vehicle_type, intent):
    """
    校准XGBoost预测的速度
    
    保留XGBoost捕捉的相对变化，但调整绝对值
    """
    speeds = np.array(speeds_xgboost)
    
    # 计算XGBoost预测的平均速度
    v_xgboost_mean = np.mean(speeds)
    
    # 计算校准参数
    a, b = compute_calibration_params(v_xgboost_mean, vehicle_type, intent)
    
    # 线性变换
    speeds_calibrated = a * speeds + b
    
    # 限制在合理范围
    params = VEHICLE_PARAMS[vehicle_type]
    v_max = params['v_max']
    speeds_calibrated = np.clip(speeds_calibrated, v_max * 0.3, v_max)
    
    return speeds_calibrated, a, b

def apply_kinematic_constraints(speeds, distances, a_max):
    """应用加速度约束"""
    constrained_speeds = speeds.copy()
    
    # 前向传播：限制加速
    for i in range(1, len(speeds)):
        if distances[i-1] > 0:
            dt = distances[i-1] / max(constrained_speeds[i-1], 0.1)
            v_max_accel = constrained_speeds[i-1] + a_max * dt
            constrained_speeds[i] = min(constrained_speeds[i], v_max_accel)
    
    # 后向传播：限制减速
    for i in range(len(speeds) - 2, -1, -1):
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

def calibrate_trajectory_file(traj_file):
    """校准单个轨迹文件"""
    with open(traj_file, 'rb') as f:
        data = pickle.load(f)
    
    # 提取原始XGBoost速度
    speeds_xgboost = np.array(data['speeds'])
    path = np.array(data['path'])
    vehicle_type = data['vehicle_type']
    intent = data['intent']
    
    # 校准速度
    speeds_calibrated, a, b = calibrate_speeds(speeds_xgboost, vehicle_type, intent)
    
    # 应用运动学约束
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    distances = np.concatenate([[0], distances])
    params = VEHICLE_PARAMS[vehicle_type]
    speeds_final = apply_kinematic_constraints(speeds_calibrated, distances, params['a_max'])
    
    # 重新计算时间戳
    timestamps = recompute_timestamps(path, speeds_final)
    
    # 更新数据
    data['speeds'] = speeds_final.tolist()
    data['timestamps'] = timestamps.tolist()
    data['duration'] = float(timestamps[-1])
    
    # 保存校准参数（用于论文说明）
    data['speed_calibration'] = {
        'method': 'linear_transform',
        'formula': 'v_calibrated = a × v_xgboost + b',
        'params': {'a': float(a), 'b': float(b)},
        'xgboost_mean': float(np.mean(speeds_xgboost) * 3.6),
        'calibrated_mean': float(np.mean(speeds_final) * 3.6)
    }
    
    # 保存
    with open(traj_file, 'wb') as f:
        pickle.dump(data, f)
    
    return {
        'xgboost_mean': np.mean(speeds_xgboost) * 3.6,
        'calibrated_mean': np.mean(speeds_final) * 3.6,
        'calibration_factor': a,
        'max_speed': np.max(speeds_final) * 3.6
    }

def main():
    print("="*60)
    print("校准XGBoost速度预测")
    print("="*60)
    print("\n方法：线性变换 v_calibrated = a × v_xgboost + b")
    print("目的：保留XGBoost的相对变化，调整绝对值到合理范围")
    
    traj_dir = Path('data/processed/synthetic_trajectories_10s/bohemian_forest')
    pkl_files = sorted(traj_dir.glob('*.pkl'))
    
    print(f"\n找到 {len(pkl_files)} 个轨迹文件")
    print("开始校准...")
    
    stats = []
    calibration_factors = []
    
    for traj_file in tqdm(pkl_files, desc="校准进度"):
        result = calibrate_trajectory_file(traj_file)
        stats.append(result)
        calibration_factors.append(result['calibration_factor'])
    
    # 统计
    xgboost_means = [s['xgboost_mean'] for s in stats]
    calibrated_means = [s['calibrated_mean'] for s in stats]
    max_speeds = [s['max_speed'] for s in stats]
    
    print(f"\n✓ 校准完成！")
    print(f"\n【校准前后对比】")
    print(f"  XGBoost原始平均速度: {np.mean(xgboost_means):.1f} km/h")
    print(f"  校准后平均速度: {np.mean(calibrated_means):.1f} km/h")
    print(f"  平均校准系数: {np.mean(calibration_factors):.2f}×")
    print(f"  校准后最大速度: {np.mean(max_speeds):.1f} km/h")
    
    print(f"\n【校准系数分布】")
    print(f"  最小: {np.min(calibration_factors):.2f}×")
    print(f"  最大: {np.max(calibration_factors):.2f}×")
    print(f"  中位数: {np.median(calibration_factors):.2f}×")
    
    print(f"\n【论文说明】")
    print("XGBoost速度模型在OORD数据集上训练，捕捉了地形对速度的影响。")
    print("但由于OORD为越野慢速场景，预测值偏低。")
    print("因此对预测结果进行线性校准：v_calibrated = a × v_xgboost")
    print(f"其中校准系数a根据车辆类型和战术意图确定（平均{np.mean(calibration_factors):.2f}×）。")
    print("该方法保留了XGBoost捕捉的地形影响，同时将速度调整到合理范围。")

if __name__ == '__main__':
    main()
