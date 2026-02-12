#!/usr/bin/env python3
"""
Phase System V2 评估框架

核心改动：所有模型通过env_map第18通道（goal_prior_map）接收统一的区域先验热力图。
- Phase 1：σ≈1km的尖锐高斯 → 精确终点
- Phase 2：σ≈10km的宽高斯 → 潜在区域（覆盖~20km）
- Phase 3：沿运动方向的扇形分布 → 无终点先验

子实验：
- 1a 域内, 1b OOD
- 2a σ=10km, 2b σ=15km, 2c 中心偏移5km
- 3a 直行, 3b 转弯
"""

import sys
import os
import json
import pickle
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from config.plot_config import get_plot_config
plot_cfg = get_plot_config()

# ============================================================
#  先验热力图生成
# ============================================================

def generate_prior_heatmap_phase1(goal_rel_km, env_coverage_km=140.0, map_size=128, sigma_km=1.0):
    """Phase 1: 精确终点 — 尖锐高斯热力图，中心=GT终点"""
    coverage_m = env_coverage_km * 1000.0
    resolution_m = coverage_m / map_size
    half = coverage_m / 2.0

    # 像素中心坐标 (m)
    px_x = (np.arange(map_size) + 0.5) * resolution_m - half  # (128,)
    px_y = half - (np.arange(map_size) + 0.5) * resolution_m  # (128,)

    # GT终点坐标 (m)
    gx_m = goal_rel_km[0] * 1000.0
    gy_m = goal_rel_km[1] * 1000.0

    # 计算每个像素到GT的距离
    dx = px_x[None, :] - gx_m  # (1, 128)
    dy = px_y[:, None] - gy_m  # (128, 1)
    dist_sq = dx ** 2 + dy ** 2  # (128, 128)

    sigma_m = sigma_km * 1000.0
    heatmap = np.exp(-dist_sq / (2.0 * sigma_m ** 2))

    # 归一化到 [0, 1]
    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    return heatmap.astype(np.float32)


def generate_prior_heatmap_phase2(goal_rel_km, env_map_np, env_coverage_km=140.0,
                                   map_size=128, sigma_km=10.0, center_offset_km=None,
                                   use_road_mask=True):
    """Phase 2: 区域先验 — 宽高斯热力图
    
    Args:
        goal_rel_km: GT终点相对坐标 (2,) km
        env_map_np: (18, 128, 128) 环境地图
        sigma_km: 高斯标准差 km
        center_offset_km: 中心偏移 (2,) km，None表示中心=GT
        use_road_mask: 是否乘以道路掩膜
    """
    coverage_m = env_coverage_km * 1000.0
    resolution_m = coverage_m / map_size
    half = coverage_m / 2.0

    px_x = (np.arange(map_size) + 0.5) * resolution_m - half
    px_y = half - (np.arange(map_size) + 0.5) * resolution_m

    # 中心坐标
    center_km = goal_rel_km.copy()
    if center_offset_km is not None:
        center_km = center_km + np.asarray(center_offset_km, dtype=np.float64)

    cx_m = center_km[0] * 1000.0
    cy_m = center_km[1] * 1000.0

    dx = px_x[None, :] - cx_m
    dy = px_y[:, None] - cy_m
    dist_sq = dx ** 2 + dy ** 2

    sigma_m = sigma_km * 1000.0
    heatmap = np.exp(-dist_sq / (2.0 * sigma_m ** 2))

    # 可选：乘以道路掩膜
    if use_road_mask:
        road = env_map_np[15]
        heatmap = heatmap * (road > 0.5).astype(np.float32)

    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    return heatmap.astype(np.float32)


def generate_prior_heatmap_phase3(history_xy_km, env_map_np, env_coverage_km=140.0,
                                   map_size=128, future_len=360,
                                   angle_range_deg=60.0, dist_range_factor=(0.3, 2.0)):
    """Phase 3: 无先验 — 运动方向扇形分布
    
    Args:
        history_xy_km: (T_hist, 2) 历史轨迹 km
        env_map_np: (18, 128, 128) 环境地图
        angle_range_deg: 扇形半角（度）
        dist_range_factor: 距离范围因子 (min_factor, max_factor)
    """
    coverage_m = env_coverage_km * 1000.0
    resolution_m = coverage_m / map_size
    half = coverage_m / 2.0

    px_x = (np.arange(map_size) + 0.5) * resolution_m - half
    px_y = half - (np.arange(map_size) + 0.5) * resolution_m

    # 从历史轨迹推断运动方向和速度
    if len(history_xy_km) >= 10:
        # 用最后10步估计方向，更稳定
        vel = history_xy_km[-1] - history_xy_km[-10]
        vel = vel / 10.0  # 每步速度
    elif len(history_xy_km) >= 2:
        vel = history_xy_km[-1] - history_xy_km[-2]
    else:
        vel = np.array([0.01, 0.0])

    speed = np.linalg.norm(vel)
    if speed > 1e-6:
        heading = np.arctan2(vel[1], vel[0])
    else:
        heading = 0.0
        speed = 0.01

    # 估计未来距离
    est_distance_km = speed * future_len
    est_distance_km = np.clip(est_distance_km, 10.0, 80.0)

    est_distance_m = est_distance_km * 1000.0
    min_dist_m = est_distance_m * dist_range_factor[0]
    max_dist_m = est_distance_m * dist_range_factor[1]

    # 像素网格
    xx = px_x[None, :]  # (1, 128)
    yy = px_y[:, None]  # (128, 1)

    # 距离约束
    dist = np.sqrt(xx ** 2 + yy ** 2)
    dist_mask = (dist >= min_dist_m) & (dist <= max_dist_m)

    # 方向约束
    angles = np.arctan2(yy, xx)
    angle_diff = np.abs(np.arctan2(np.sin(angles - heading), np.cos(angles - heading)))
    angle_mask = angle_diff < np.radians(angle_range_deg)

    # 道路约束
    road = env_map_np[15]
    road_mask = road > 0.5

    # 组合
    heatmap = (dist_mask & angle_mask & road_mask).astype(np.float32)

    # 如果扇形内没有道路，放宽约束
    if heatmap.max() < 0.5:
        heatmap = (dist_mask & angle_mask).astype(np.float32)
    if heatmap.max() < 0.5:
        heatmap = dist_mask.astype(np.float32)

    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    return heatmap.astype(np.float32)


def sample_candidates_from_heatmap(heatmap, env_map_np, env_coverage_km=140.0,
                                    map_size=128, num_candidates=6,
                                    include_gt=False, goal_rel_km=None,
                                    rng=None):
    """从热力图中按权重采样候选终点
    
    Returns:
        candidates_km: (K, 2) 候选坐标 km
        target_idx: 最接近GT的候选索引
    """
    if rng is None:
        rng = np.random.default_rng(42)

    coverage_m = env_coverage_km * 1000.0
    resolution_m = coverage_m / map_size
    half = coverage_m / 2.0

    px_x = (np.arange(map_size) + 0.5) * resolution_m - half
    px_y = half - (np.arange(map_size) + 0.5) * resolution_m

    # 获取非零权重的像素
    weights = heatmap.flatten()
    total_w = weights.sum()

    if total_w > 1e-8:
        probs = weights / total_w
        indices = rng.choice(len(probs), size=num_candidates, replace=True, p=probs)
        rows = indices // map_size
        cols = indices % map_size
        candidates_km = np.stack([
            px_x[cols] / 1000.0,
            px_y[rows] / 1000.0,
        ], axis=1)
    else:
        # Fallback: 均匀随机
        angles = rng.uniform(0, 2 * np.pi, num_candidates)
        dists = rng.uniform(10.0, 50.0, num_candidates)
        candidates_km = np.stack([
            dists * np.cos(angles),
            dists * np.sin(angles),
        ], axis=1)

    # 可选：插入GT
    if include_gt and goal_rel_km is not None and num_candidates > 0:
        ins = int(rng.integers(0, num_candidates))
        candidates_km[ins] = goal_rel_km

    # 计算target_idx
    target_idx = -1
    if goal_rel_km is not None and num_candidates > 0:
        dists_to_gt = np.linalg.norm(candidates_km - goal_rel_km.reshape(1, 2), axis=1)
        target_idx = int(np.argmin(dists_to_gt))

    return candidates_km.astype(np.float32), target_idx


def extract_goal_from_heatmap(heatmap, env_coverage_km=140.0, map_size=128):
    """从热力图中提取"最可能终点"坐标（用于LSTM_Env_Goal等模型）
    
    返回热力图加权质心坐标 (km)
    """
    coverage_m = env_coverage_km * 1000.0
    resolution_m = coverage_m / map_size
    half = coverage_m / 2.0

    px_x = (np.arange(map_size) + 0.5) * resolution_m - half
    px_y = half - (np.arange(map_size) + 0.5) * resolution_m

    total_w = heatmap.sum()
    if total_w < 1e-8:
        return np.zeros(2, dtype=np.float32)

    # 加权质心
    cx_m = (heatmap * px_x[None, :]).sum() / total_w
    cy_m = (heatmap * px_y[:, None]).sum() / total_w

    return np.array([cx_m / 1000.0, cy_m / 1000.0], dtype=np.float32)


# ============================================================
#  Phase V2 数据集
# ============================================================

PHASE_V2_CONFIGS = {
    # Phase 1: 精确终点
    'P1a': {
        'name': 'Phase1a: 精确终点(域内)',
        'split_key': 'fas1',
        'heatmap_type': 'phase1',
        'sigma_km': 1.0,
        'include_gt_in_candidates': True,
        'center_offset_km': None,
        'subset_filter': None,
    },
    'P1b': {
        'name': 'Phase1b: 精确终点(OOD)',
        'split_key': 'fas2',
        'heatmap_type': 'phase1',
        'sigma_km': 1.0,
        'include_gt_in_candidates': True,
        'center_offset_km': None,
        'subset_filter': None,
    },
    # Phase 2: 区域先验
    'P2a': {
        'name': 'Phase2a: 区域先验(σ=10km)',
        'split_key': 'fas1',
        'heatmap_type': 'phase2',
        'sigma_km': 10.0,
        'include_gt_in_candidates': False,
        'center_offset_km': None,
        'subset_filter': None,
    },
    'P2b': {
        'name': 'Phase2b: 区域先验(σ=15km)',
        'split_key': 'fas1',
        'heatmap_type': 'phase2',
        'sigma_km': 15.0,
        'include_gt_in_candidates': False,
        'center_offset_km': None,
        'subset_filter': None,
    },
    'P2c': {
        'name': 'Phase2c: 区域先验(σ=10km,偏移5km)',
        'split_key': 'fas1',
        'heatmap_type': 'phase2',
        'sigma_km': 10.0,
        'include_gt_in_candidates': False,
        'center_offset_km': 5.0,  # 随机方向偏移5km
        'subset_filter': None,
    },
    # Phase 3: 无先验
    'P3a': {
        'name': 'Phase3a: 无先验(直行)',
        'split_key': 'fas1',
        'heatmap_type': 'phase3',
        'sigma_km': None,
        'include_gt_in_candidates': False,
        'center_offset_km': None,
        'subset_filter': 'straight',  # 历史方向与终点方向夹角<30°
    },
    'P3b': {
        'name': 'Phase3b: 无先验(转弯)',
        'split_key': 'fas1',
        'heatmap_type': 'phase3',
        'sigma_km': None,
        'include_gt_in_candidates': False,
        'center_offset_km': None,
        'subset_filter': 'turning',  # 历史方向与终点方向夹角>60°
    },
}


class PhaseV2Dataset(Dataset):
    """Phase V2 数据集：统一先验热力图生成"""

    def __init__(self, traj_dir, fas_split_file, phase_config,
                 history_len=90, future_len=360, num_candidates=6,
                 region='bohemian_forest', sample_fraction=1.0, seed=42,
                 env_coverage_km=140.0, coord_scale=1.0):
        self.traj_dir = Path(traj_dir)
        self.history_len = history_len
        self.future_len = future_len
        self.num_candidates = num_candidates
        self.region = region
        self.seed = int(seed)
        self.env_coverage_km = float(env_coverage_km)
        self.coord_scale = float(coord_scale)
        self.phase_config = phase_config

        with open(fas_split_file, 'r') as f:
            splits = json.load(f)

        split_key = phase_config['split_key']
        phase_spec = splits.get(split_key, {})

        self.samples_meta = []
        self._file_cache = {}

        # 格式1: val_samples/samples 列表 [{file, sample_idx}, ...]
        val_samples = phase_spec.get('val_samples', [])
        if not val_samples:
            val_samples = phase_spec.get('samples', [])

        if val_samples and isinstance(val_samples[0], dict):
            for item in val_samples:
                rel_file = item.get('file')
                sample_idx = item.get('sample_idx')
                if rel_file is None or sample_idx is None:
                    continue
                traj_file = self.traj_dir / str(rel_file)
                if traj_file.exists():
                    self.samples_meta.append((str(traj_file), int(sample_idx)))

        # 格式2: files 列表 [filename, ...] — 每个文件取第一个样本(快速)
        # 或展开所有样本(需要加载文件)
        elif 'files' in phase_spec:
            file_list = phase_spec['files']
            existing_files = []
            for fname in file_list:
                traj_file = self.traj_dir / str(fname)
                if traj_file.exists():
                    existing_files.append(str(traj_file))
            # 标记为需要展开的文件列表模式
            self._files_mode = True
            self._all_files = existing_files
            # 先用每文件1个样本估算, 实际在__getitem__中按需加载
            # 展开: 每个文件的所有样本
            print(f"    扫描 {len(existing_files)} 个文件中的样本...")
            for fpath in existing_files:
                try:
                    with open(fpath, 'rb') as f:
                        data = pickle.load(f)
                    n_samples = len(data.get('samples', []))
                    for si in range(n_samples):
                        self.samples_meta.append((fpath, si))
                except Exception:
                    pass
            print(f"    共 {len(self.samples_meta)} 个样本")

        # 子集过滤（直行/转弯）在加载时执行
        subset_filter = phase_config.get('subset_filter')
        if subset_filter in ('straight', 'turning'):
            self.samples_meta = self._filter_by_direction(subset_filter)

        if sample_fraction < 1.0 and len(self.samples_meta) > 0:
            import random
            random.seed(seed)
            n = max(1, int(round(len(self.samples_meta) * sample_fraction)))
            self.samples_meta = random.sample(self.samples_meta, n)
            self.samples_meta.sort()

        print(f"  {phase_config['name']}: {len(self.samples_meta)} 验证样本")

    def _load_sample(self, file_path, sample_idx):
        if file_path in self._file_cache:
            data = self._file_cache[file_path]
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if len(self._file_cache) > 5:
                self._file_cache.clear()
            self._file_cache[file_path] = data
        return data['samples'][sample_idx]

    def _filter_by_direction(self, filter_type):
        """根据历史方向与终点方向的夹角过滤样本"""
        filtered = []
        for file_path, sample_idx in tqdm(self.samples_meta,
                                           desc=f"过滤{filter_type}样本", leave=False):
            sample = self._load_sample(file_path, sample_idx)
            history = np.asarray(sample['history_feat_26d'], dtype=np.float64)
            goal_rel = np.asarray(sample['goal_rel'], dtype=np.float64)

            # 历史运动方向
            if len(history) >= 10:
                vel = history[-1, :2] - history[-10, :2]
            elif len(history) >= 2:
                vel = history[-1, :2] - history[-2, :2]
            else:
                continue

            speed = np.linalg.norm(vel)
            if speed < 1e-6:
                continue

            heading = np.arctan2(vel[1], vel[0])

            # 终点方向
            goal_dist = np.linalg.norm(goal_rel)
            if goal_dist < 1e-6:
                continue
            goal_heading = np.arctan2(goal_rel[1], goal_rel[0])

            # 夹角
            angle_diff = np.abs(np.arctan2(
                np.sin(goal_heading - heading),
                np.cos(goal_heading - heading)
            ))
            angle_deg = np.degrees(angle_diff)

            if filter_type == 'straight' and angle_deg < 30.0:
                filtered.append((file_path, sample_idx))
            elif filter_type == 'turning' and angle_deg > 60.0:
                filtered.append((file_path, sample_idx))

        self._file_cache.clear()
        return filtered

    @staticmethod
    def normalize_env_map(env_map_np):
        normalized = np.asarray(env_map_np, dtype=np.float32).copy()
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        dem = normalized[0]
        if float(np.nanmax(np.abs(dem))) > 50.0:
            dem_min, dem_max = float(np.nanmin(dem)), float(np.nanmax(dem))
            if dem_max > dem_min:
                normalized[0] = (dem - dem_min) / (dem_max - dem_min)
            else:
                normalized[0] = 0.0
        normalized[1] = np.clip(normalized[1], 0.0, 1.0)
        normalized[2] = np.clip(normalized[2], -1.0, 1.0)
        normalized[3] = np.clip(normalized[3], -1.0, 1.0)
        normalized[4:18] = np.clip(normalized[4:18], 0.0, 1.0)
        return normalized

    def __len__(self):
        return len(self.samples_meta)

    def __getitem__(self, idx):
        file_path, sample_idx = self.samples_meta[idx]
        sample = self._load_sample(file_path, sample_idx)

        current_pos_abs = np.asarray(sample['current_pos_abs'], dtype=np.float64)
        env_map_np = self.normalize_env_map(np.asarray(sample['env_map_100km'], dtype=np.float32))
        history_feat = torch.FloatTensor(sample['history_feat_26d'])
        history = history_feat.clone()
        history[:, 0:2] *= self.coord_scale

        future_rel = torch.FloatTensor(sample['future_rel'])
        future_delta = torch.diff(future_rel, dim=0, prepend=torch.zeros(1, 2))
        future = future_delta * self.coord_scale

        goal_rel_km = np.asarray(sample['goal_rel'], dtype=np.float64)
        goal = torch.as_tensor(goal_rel_km * self.coord_scale, dtype=torch.float32)

        history_xy_km = history_feat[:, :2].numpy()

        # ── 生成先验热力图 ──
        cfg = self.phase_config
        heatmap_type = cfg['heatmap_type']

        if heatmap_type == 'phase1':
            heatmap = generate_prior_heatmap_phase1(
                goal_rel_km, self.env_coverage_km, sigma_km=cfg['sigma_km'])
        elif heatmap_type == 'phase2':
            center_offset = None
            if cfg.get('center_offset_km') is not None:
                # 随机方向偏移
                rng = np.random.default_rng((self.seed * 1000003 + idx) % (2**32))
                angle = rng.uniform(0, 2 * np.pi)
                offset_mag = float(cfg['center_offset_km'])
                center_offset = np.array([
                    offset_mag * np.cos(angle),
                    offset_mag * np.sin(angle)
                ])
            heatmap = generate_prior_heatmap_phase2(
                goal_rel_km, env_map_np, self.env_coverage_km,
                sigma_km=cfg['sigma_km'], center_offset_km=center_offset)
        elif heatmap_type == 'phase3':
            heatmap = generate_prior_heatmap_phase3(
                history_xy_km, env_map_np, self.env_coverage_km,
                future_len=self.future_len)
        else:
            heatmap = np.zeros((128, 128), dtype=np.float32)

        # 写入env_map第18通道
        env_map_np[-1] = heatmap
        env_map = torch.from_numpy(env_map_np).float()

        # ── 从热力图采样候选终点（TerraTNT系列用）──
        rng = np.random.default_rng((self.seed * 1000003 + idx) % (2**32))
        candidates_km, target_idx = sample_candidates_from_heatmap(
            heatmap, env_map_np, self.env_coverage_km,
            num_candidates=self.num_candidates,
            include_gt=cfg['include_gt_in_candidates'],
            goal_rel_km=goal_rel_km,
            rng=rng,
        )
        candidates = torch.as_tensor(candidates_km, dtype=torch.float32) * self.coord_scale

        # ── 从热力图提取goal坐标（LSTM_Env_Goal等用）──
        heatmap_goal_km = extract_goal_from_heatmap(heatmap, self.env_coverage_km)
        heatmap_goal = torch.as_tensor(heatmap_goal_km, dtype=torch.float32) * self.coord_scale

        return {
            'history': history,
            'future': future,
            'candidates': candidates,
            'env_map': env_map,
            'target_goal_idx': target_idx,
            'goal': goal,                    # GT goal (仅用于计算metrics)
            'heatmap_goal': heatmap_goal,    # 从热力图提取的goal (用于LSTM_EG等)
            'current_pos_abs': torch.as_tensor(current_pos_abs, dtype=torch.float32),
            'heatmap': torch.from_numpy(heatmap).float(),
        }


# ============================================================
#  评估逻辑
# ============================================================

def compute_metrics(pred_pos_km, gt_pos_km):
    """计算ADE/FDE及分段指标 (meters)"""
    err = np.linalg.norm(pred_pos_km - gt_pos_km, axis=-1) * 1000.0  # (T,) meters
    T = len(err)
    third = T // 3

    return {
        'ade': float(np.mean(err)),
        'fde': float(err[-1]),
        'early_ade': float(np.mean(err[:third])) if third > 0 else 0.0,
        'mid_ade': float(np.mean(err[third:2*third])) if third > 0 else 0.0,
        'late_ade': float(np.mean(err[2*third:])) if third > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_all_models(models_dict, dataset, device, batch_size=16):
    """评估所有模型，返回 {model_name: [metrics_per_sample]}"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    all_metrics = {name: [] for name in models_dict}
    all_metrics['ConstantVelocity'] = []

    for batch in tqdm(loader, desc="评估中", leave=False):
        history = batch['history'].to(device)
        future = batch['future'].to(device)
        env_map = batch['env_map'].to(device)
        candidates = batch['candidates'].to(device)
        target_idx = batch['target_goal_idx'].to(device)
        goal_gt = batch['goal'].to(device)
        heatmap_goal = batch['heatmap_goal'].to(device)

        B = history.size(0)
        history_xy = history[:, :, :2]
        gt_pos = torch.cumsum(future, dim=1)  # (B, T, 2) km
        T = gt_pos.size(1)

        # ── Constant Velocity baseline ──
        vel = history_xy[:, -1, :] - history_xy[:, -2, :]
        steps = torch.arange(1, T + 1, device=device).float().unsqueeze(0).unsqueeze(-1)
        cv_pos = vel.unsqueeze(1) * steps
        for b in range(B):
            m = compute_metrics(cv_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy())
            all_metrics['ConstantVelocity'].append(m)

        # ── 各模型 ──
        for name, model_info in models_dict.items():
            model = model_info['model']
            model_type = model_info['type']

            try:
                with autocast('cuda', enabled=True):
                    if model_type == 'baseline_with_goal':
                        # LSTM_Env_Goal: 使用热力图提取的goal（公平对比）
                        out = model(history_xy, env_map, goal=heatmap_goal)
                        if isinstance(out, tuple):
                            pred_pos = torch.cumsum(out[0], dim=1)
                        else:
                            pred_pos = out

                    elif model_type == 'baseline_no_goal':
                        # LSTMOnly, MLP, Seq2Seq: 不使用goal
                        out = model(history_xy, env_map)
                        if isinstance(out, tuple):
                            pred_pos = torch.cumsum(out[0], dim=1)
                        else:
                            pred_pos = out

                    elif model_type == 'terratnt':
                        # TerraTNT原版: 返回delta，需要cumsum
                        current_pos = torch.zeros(B, 2, device=device)
                        pred_delta, goal_logits = model(
                            env_map, history, candidates, current_pos,
                            teacher_forcing_ratio=0.0,
                            target_goal_idx=target_idx,
                            use_gt_goal=False,
                        )
                        pred_pos = torch.cumsum(pred_delta, dim=1)

                    elif model_type == 'terratnt_v6':
                        # V6系列: 返回累积位置，不需要cumsum
                        current_pos = torch.zeros(B, 2, device=device)
                        pred_pos, goal_logits = model(
                            env_map, history, candidates, current_pos,
                            teacher_forcing_ratio=0.0,
                            target_goal_idx=target_idx,
                            use_gt_goal=False,
                        )

                    elif model_type == 'terratnt_v7':
                        # V7: 返回累积位置 + alpha
                        current_pos = torch.zeros(B, 2, device=device)
                        pred_pos, goal_logits, alpha = model(
                            env_map, history, candidates, current_pos,
                            teacher_forcing_ratio=0.0,
                            target_goal_idx=target_idx,
                            use_gt_goal=False,
                        )

                    elif model_type == 'incremental':
                        # V3/V4: LSTM-based incremental models
                        out = model(history_xy, env_map, goal=heatmap_goal)
                        if isinstance(out, tuple):
                            pred_pos = torch.cumsum(out[0], dim=1)
                        else:
                            pred_pos = out

                    else:
                        continue

                for b in range(B):
                    m = compute_metrics(pred_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy())
                    all_metrics[name].append(m)

            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                for b in range(B):
                    all_metrics[name].append({
                        'ade': float('nan'), 'fde': float('nan'),
                        'early_ade': float('nan'), 'mid_ade': float('nan'),
                        'late_ade': float('nan'),
                    })

    return all_metrics


# ============================================================
#  模型加载
# ============================================================

def load_all_models(device):
    """加载所有可用模型"""
    from scripts.train_eval_all_baselines import (
        LSTMOnly, LSTMEnvGoal, Seq2SeqAttention, MLPBaseline,
    )
    from scripts.train_incremental_models import (
        LSTMEnvGoalWaypoint, LSTMEnvGoalWaypointSpatial,
        TerraTNTFusionV5, TerraTNTAutoregV6,
    )

    runs = PROJECT_ROOT / 'runs'
    models = {}
    FUTURE_LEN = 360
    HISTORY_LEN = 90

    # ── Baselines ──
    baseline_configs = {
        'LSTM_only': {
            'cls': LSTMOnly,
            'kwargs': dict(hidden_dim=256, future_len=FUTURE_LEN),
            'ckpt': runs / 'LSTM_only_d1' / 'best_model.pth',
            'type': 'baseline_no_goal',
            'state_dict_key': None,
        },
        'LSTM_Env_Goal': {
            'cls': LSTMEnvGoal,
            'kwargs': dict(hidden_dim=256, future_len=FUTURE_LEN),
            'ckpt': runs / 'LSTM_Env_Goal_d1' / 'best_model.pth',
            'type': 'baseline_with_goal',
            'state_dict_key': None,
        },
        'Seq2Seq_Attn': {
            'cls': Seq2SeqAttention,
            'kwargs': dict(hidden_dim=256, future_len=FUTURE_LEN),
            'ckpt': runs / 'Seq2Seq_Attn_d1' / 'best_model.pth',
            'type': 'baseline_no_goal',
            'state_dict_key': None,
        },
        'MLP': {
            'cls': MLPBaseline,
            'kwargs': dict(hidden_dim=512, future_len=FUTURE_LEN, history_len=HISTORY_LEN),
            'ckpt': runs / 'MLP_d1' / 'best_model.pth',
            'type': 'baseline_no_goal',
            'state_dict_key': None,
        },
    }

    for name, cfg in baseline_configs.items():
        if not cfg['ckpt'].exists():
            print(f"  [SKIP] {name}: {cfg['ckpt']}")
            continue
        m = cfg['cls'](**cfg['kwargs']).to(device)
        state = torch.load(cfg['ckpt'], map_location=device)
        if cfg['state_dict_key']:
            state = state[cfg['state_dict_key']]
        m.load_state_dict(state)
        m.eval()
        models[name] = {'model': m, 'type': cfg['type']}
        print(f"  [OK] {name}")

    # ── Incremental V3/V4 ──
    incr_configs = {
        'V3_Waypoint': {
            'cls': LSTMEnvGoalWaypoint,
            'kwargs': dict(hidden_dim=256, future_len=FUTURE_LEN, num_waypoints=10),
            'ckpt': runs / 'incremental_models' / 'V3_best.pth',
            'type': 'incremental',
        },
        'V4_WP_Spatial': {
            'cls': LSTMEnvGoalWaypointSpatial,
            'kwargs': dict(hidden_dim=256, future_len=FUTURE_LEN, num_waypoints=10, env_coverage_km=140.0),
            'ckpt': runs / 'incremental_models' / 'V4_best.pth',
            'type': 'incremental',
        },
    }

    for name, cfg in incr_configs.items():
        if not cfg['ckpt'].exists():
            print(f"  [SKIP] {name}")
            continue
        m = cfg['cls'](**cfg['kwargs']).to(device)
        state = torch.load(cfg['ckpt'], map_location=device)['model_state_dict']
        m.load_state_dict(state)
        m.eval()
        models[name] = {'model': m, 'type': cfg['type']}
        print(f"  [OK] {name}")

    # ── TerraTNT ──
    terratnt_ckpts = {
        'TerraTNT': runs / 'd1_optimal_fas1' / 'terratnt_fas1_10s' / '20260208_172730' / 'best_model.pth',
        'V6_Autoreg': runs / 'incremental_models' / 'V6_best.pth',
        'V6R_Robust': runs / 'incremental_models_v6r' / 'V6_best.pth',
        'V7_ConfGate': runs / 'incremental_models_v7' / 'V7_best.pth',
    }

    for name, ckpt_path in terratnt_ckpts.items():
        if not ckpt_path.exists():
            print(f"  [SKIP] {name}: {ckpt_path}")
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model_cfg = ckpt.get('config', {})
            sd = ckpt.get('model_state_dict', ckpt)

            if name == 'TerraTNT':
                from models.terratnt import TerraTNT
                m = TerraTNT(
                    env_input_channels=18,
                    history_input_dim=26,
                    env_feature_dim=128,
                    history_feature_dim=128,
                    hidden_dim=256,
                    output_length=FUTURE_LEN,
                    num_goals=model_cfg.get('num_candidates', 6),
                    paper_mode=model_cfg.get('paper_mode', True),
                    paper_decoder=model_cfg.get('paper_decoder', 'hierarchical'),
                    waypoint_stride=model_cfg.get('waypoint_stride', 90),
                    env_coverage_km=model_cfg.get('env_coverage_km', 140.0),
                ).to(device)
                mtype = 'terratnt'
            elif name == 'V7_ConfGate':
                from scripts.train_incremental_models import ConfidenceGatedV7
                h_key = 'history_encoder.lstm.weight_ih_l0'
                inferred_hidden = sd[h_key].shape[0] // 4 if h_key in sd else 128
                m = ConfidenceGatedV7(
                    hidden_dim=inferred_hidden,
                    future_len=FUTURE_LEN,
                    num_waypoints=10,
                    env_coverage_km=140.0,
                    num_candidates=model_cfg.get('num_candidates', 6),
                ).to(device)
                mtype = 'terratnt_v7'
            else:
                # V5/V6/V6R — 从权重推断hidden_dim
                h_key = 'history_encoder.lstm.weight_ih_l0'
                if h_key in sd:
                    inferred_hidden = sd[h_key].shape[0] // 4
                else:
                    inferred_hidden = 128
                m = TerraTNTAutoregV6(
                    hidden_dim=inferred_hidden,
                    future_len=FUTURE_LEN,
                    num_waypoints=10,
                    env_coverage_km=140.0,
                    num_candidates=model_cfg.get('num_candidates', 6),
                ).to(device)
                mtype = 'terratnt_v6'

            sd = ckpt.get('model_state_dict', ckpt)
            m.load_state_dict(sd, strict=False)
            m.eval()
            models[name] = {'model': m, 'type': mtype}
            print(f"  [OK] {name} (type={mtype})")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    return models


# ============================================================
#  结果汇总与输出
# ============================================================

def summarize_results(phase_results, output_dir):
    """汇总所有Phase结果并输出"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for phase_id, (phase_cfg, metrics_dict) in phase_results.items():
        summary[phase_id] = {'name': phase_cfg['name'], 'models': {}}

        for model_name, metrics_list in metrics_dict.items():
            if not metrics_list:
                continue
            ades = [m['ade'] for m in metrics_list if not np.isnan(m['ade'])]
            fdes = [m['fde'] for m in metrics_list if not np.isnan(m['fde'])]
            if not ades:
                continue

            summary[phase_id]['models'][model_name] = {
                'ade_mean': float(np.mean(ades)),
                'ade_std': float(np.std(ades)),
                'fde_mean': float(np.mean(fdes)),
                'fde_std': float(np.std(fdes)),
                'n_samples': len(ades),
                'early_ade': float(np.mean([m['early_ade'] for m in metrics_list if not np.isnan(m['early_ade'])])),
                'mid_ade': float(np.mean([m['mid_ade'] for m in metrics_list if not np.isnan(m['mid_ade'])])),
                'late_ade': float(np.mean([m['late_ade'] for m in metrics_list if not np.isnan(m['late_ade'])])),
            }

    # 保存JSON
    with open(output_dir / 'phase_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印表格
    print("\n" + "=" * 100)
    print("Phase V2 评估结果汇总")
    print("=" * 100)

    # 收集所有模型名
    all_models = set()
    for pid, data in summary.items():
        all_models.update(data['models'].keys())
    all_models = sorted(all_models)

    # 打印表头
    header = f"{'Phase':<35}"
    for mn in all_models:
        header += f" {mn:>14}"
    print(header)
    print("-" * len(header))

    for phase_id in sorted(summary.keys()):
        data = summary[phase_id]
        row = f"{data['name']:<35}"
        for mn in all_models:
            if mn in data['models']:
                ade = data['models'][mn]['ade_mean']
                row += f" {ade:>13.0f}m"
            else:
                row += f" {'N/A':>14}"
        print(row)

    print("=" * 100)

    # 保存文本报告
    report_lines = []
    for phase_id in sorted(summary.keys()):
        data = summary[phase_id]
        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"{data['name']}")
        report_lines.append(f"{'='*60}")

        # 按ADE排序
        sorted_models = sorted(data['models'].items(), key=lambda x: x[1]['ade_mean'])
        for rank, (mn, ms) in enumerate(sorted_models, 1):
            report_lines.append(
                f"  #{rank} {mn:<20} ADE={ms['ade_mean']:.0f}m (±{ms['ade_std']:.0f}) "
                f"FDE={ms['fde_mean']:.0f}m  "
                f"Early/Mid/Late={ms['early_ade']:.0f}/{ms['mid_ade']:.0f}/{ms['late_ade']:.0f}m  "
                f"(n={ms['n_samples']})"
            )

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(output_dir / 'phase_v2_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    return summary


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Phase V2 评估')
    parser.add_argument('--phases', nargs='+', default=None,
                        help='要评估的Phase ID列表，如 P1a P2a P3a。默认全部')
    parser.add_argument('--traj_dir', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo'))
    parser.add_argument('--fas_split_file', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo' / 'fas_splits_full_phases.json'))
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'evaluation' / 'phase_v2'))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sample_fraction', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 选择要评估的Phase
    phase_ids = args.phases if args.phases else list(PHASE_V2_CONFIGS.keys())
    print(f"\n要评估的Phase: {phase_ids}")

    # 加载模型
    print("\n加载模型...")
    models_dict = load_all_models(device)
    print(f"已加载 {len(models_dict)} 个模型")

    # 逐Phase评估
    phase_results = {}

    for phase_id in phase_ids:
        if phase_id not in PHASE_V2_CONFIGS:
            print(f"  [SKIP] 未知Phase: {phase_id}")
            continue

        phase_cfg = PHASE_V2_CONFIGS[phase_id]
        print(f"\n{'='*60}")
        print(f"评估 {phase_cfg['name']}")
        print(f"{'='*60}")

        dataset = PhaseV2Dataset(
            traj_dir=args.traj_dir,
            fas_split_file=args.fas_split_file,
            phase_config=phase_cfg,
            sample_fraction=args.sample_fraction,
            seed=args.seed,
        )

        if len(dataset) == 0:
            print(f"  [SKIP] 无样本")
            continue

        metrics = evaluate_all_models(models_dict, dataset, device, args.batch_size)
        phase_results[phase_id] = (phase_cfg, metrics)

    # 汇总
    summarize_results(phase_results, args.output_dir)
    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
