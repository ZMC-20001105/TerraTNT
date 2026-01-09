#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
准备 FAS1/2/3 三阶段评估数据集

根据论文定义：
- Phase 1 (FAS1): 域内目标 + 完备候选集（理想条件）
- Phase 2 (FAS2): 域外目标（训练时未见过的终点）+ 完备候选集（泛化能力）
- Phase 3 (FAS3): 域内目标 + 不完备候选集（候选集不包含真值，测试鲁棒性）
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm

def extract_goals_from_trajectories(traj_dir: Path):
    """
    从所有轨迹中提取终点坐标
    
    Returns:
        goals: list of (x, y) tuples
        traj_to_goal: dict mapping trajectory file to goal
    """
    print(f"从 {traj_dir} 提取终点...")
    
    traj_files = sorted(list(traj_dir.glob('*.pkl')))
    print(f"找到 {len(traj_files)} 个轨迹文件")
    
    goals = []
    traj_to_goal = {}
    
    for traj_file in tqdm(traj_files, desc="提取终点"):
        try:
            with open(traj_file, 'rb') as f:
                data = pickle.load(f)
            
            # 获取路径
            path = data.get('path', data.get('path_utm', []))
            if len(path) == 0:
                continue
            
            # 终点是路径的最后一个点
            goal = (float(path[-1][0]), float(path[-1][1]))
            goals.append(goal)
            traj_to_goal[str(traj_file)] = goal
            
        except Exception as e:
            print(f"警告: 无法处理 {traj_file.name}: {e}")
            continue
    
    print(f"成功提取 {len(goals)} 个终点")
    return goals, traj_to_goal


def cluster_goals(goals, distance_threshold=1000.0):
    """
    将相近的终点聚类（简化版本：基于距离阈值）
    
    Args:
        goals: list of (x, y) tuples
        distance_threshold: 距离阈值（米），小于此距离的终点视为同一个
    
    Returns:
        goal_clusters: dict mapping cluster_id to list of goal indices
        goal_to_cluster: dict mapping goal index to cluster_id
    """
    print(f"聚类终点（阈值={distance_threshold}m）...")
    
    goals = np.array(goals)
    n = len(goals)
    
    goal_to_cluster = {}
    goal_clusters = defaultdict(list)
    cluster_id = 0
    
    for i in tqdm(range(n), desc="聚类"):
        if i in goal_to_cluster:
            continue
        
        # 创建新簇
        goal_clusters[cluster_id].append(i)
        goal_to_cluster[i] = cluster_id
        
        # 找到所有距离小于阈值的点
        for j in range(i+1, n):
            if j in goal_to_cluster:
                continue
            
            dist = np.linalg.norm(goals[i] - goals[j])
            if dist < distance_threshold:
                goal_clusters[cluster_id].append(j)
                goal_to_cluster[j] = cluster_id
        
        cluster_id += 1
    
    print(f"聚类完成：{len(goal_clusters)} 个簇")
    return goal_clusters, goal_to_cluster


def split_train_test_goals(goal_clusters, train_ratio=0.7):
    """
    将终点簇划分为训练集和测试集
    
    Args:
        goal_clusters: dict mapping cluster_id to list of goal indices
        train_ratio: 训练集比例
    
    Returns:
        train_clusters: set of cluster_ids for training
        test_clusters: set of cluster_ids for testing
    """
    print(f"划分训练集和测试集（训练比例={train_ratio}）...")
    
    cluster_ids = list(goal_clusters.keys())
    np.random.shuffle(cluster_ids)
    
    split_idx = int(len(cluster_ids) * train_ratio)
    train_clusters = set(cluster_ids[:split_idx])
    test_clusters = set(cluster_ids[split_idx:])
    
    print(f"训练集簇数: {len(train_clusters)}")
    print(f"测试集簇数: {len(test_clusters)}")
    
    return train_clusters, test_clusters


def prepare_fas_splits(traj_dir: Path, output_dir: Path, 
                       distance_threshold=1000.0, train_ratio=0.7):
    """
    准备 FAS1/2/3 三阶段数据集划分
    
    Args:
        traj_dir: 轨迹数据目录
        output_dir: 输出目录
        distance_threshold: 终点聚类距离阈值
        train_ratio: 训练集比例
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 提取所有终点
    goals, traj_to_goal = extract_goals_from_trajectories(traj_dir)
    
    if len(goals) == 0:
        print("错误：没有找到有效的终点")
        return
    
    # 2. 聚类终点
    goal_clusters, goal_to_cluster = cluster_goals(goals, distance_threshold)
    
    # 3. 划分训练集和测试集簇
    train_clusters, test_clusters = split_train_test_goals(goal_clusters, train_ratio)
    
    # 4. 为每个轨迹分配阶段标签
    traj_files = sorted(list(traj_dir.glob('*.pkl')))
    
    fas1_files = []  # 域内目标
    fas2_files = []  # 域外目标
    
    for i, traj_file in enumerate(traj_files):
        traj_path = str(traj_file)
        if traj_path not in traj_to_goal:
            continue
        
        cluster_id = goal_to_cluster[i]
        
        if cluster_id in train_clusters:
            fas1_files.append(str(traj_file.relative_to(traj_dir)))
        else:
            fas2_files.append(str(traj_file.relative_to(traj_dir)))
    
    # FAS3 使用与 FAS1 相同的轨迹，但候选集不包含真值
    fas3_files = fas1_files.copy()
    
    # 5. 保存划分结果
    splits = {
        'fas1': {
            'description': 'Phase 1: 域内目标 + 完备候选集',
            'files': fas1_files,
            'num_samples': len(fas1_files)
        },
        'fas2': {
            'description': 'Phase 2: 域外目标 + 完备候选集',
            'files': fas2_files,
            'num_samples': len(fas2_files)
        },
        'fas3': {
            'description': 'Phase 3: 域内目标 + 不完备候选集（不含真值）',
            'files': fas3_files,
            'num_samples': len(fas3_files)
        },
        'metadata': {
            'total_trajectories': len(traj_files),
            'total_goals': len(goals),
            'num_goal_clusters': len(goal_clusters),
            'train_clusters': len(train_clusters),
            'test_clusters': len(test_clusters),
            'distance_threshold': distance_threshold,
            'train_ratio': train_ratio
        }
    }
    
    output_file = output_dir / 'fas_splits.json'
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FAS 数据集划分完成")
    print('='*60)
    print(f"FAS1 (域内目标): {len(fas1_files)} 个样本")
    print(f"FAS2 (域外目标): {len(fas2_files)} 个样本")
    print(f"FAS3 (域内目标+不完备候选): {len(fas3_files)} 个样本")
    print(f"\n结果已保存到: {output_file}")
    
    return splits


def main():
    # 配置
    region = 'bohemian_forest'
    traj_dir = Path(f'/home/zmc/文档/programwork/data/processed/synthetic_trajectories/{region}')
    output_dir = Path(f'/home/zmc/文档/programwork/data/processed/fas_splits/{region}')
    
    print("="*60)
    print("准备 FAS1/2/3 三阶段评估数据集")
    print("="*60)
    print(f"区域: {region}")
    print(f"轨迹目录: {traj_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 准备数据集划分
    splits = prepare_fas_splits(
        traj_dir=traj_dir,
        output_dir=output_dir,
        distance_threshold=1000.0,  # 1km 阈值
        train_ratio=0.7  # 70% 训练，30% 测试
    )
    
    print("\n✓ 数据集准备完成！")


if __name__ == '__main__':
    main()
