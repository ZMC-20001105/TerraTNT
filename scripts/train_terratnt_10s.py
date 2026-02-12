#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TerraTNT 训练脚本 - 10秒间隔版本
使用重采样后的密集轨迹数据
- 采样间隔: 10秒
- 观测长度: 60点 = 10分钟
- 预测长度: 180点 = 30分钟
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import json
import pickle
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 引入环境地图生成器用于实时提取高清局部图
from utils.data_processing.trajectory_preprocessor import EnvironmentMapGenerator

# 训练配置 - 10秒间隔优化
BATCH_SIZE = 32  # 减小batch size以降低内存占用
NUM_WORKERS = 2  # 减少workers防止多进程内存累积
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
HISTORY_LEN = 90  # 20分钟观测 (20min × 60s ÷ 10s = 120点)
FUTURE_LEN = 360   # 60分钟预测 (60min × 60s ÷ 10s = 360点)


def _print_config(batch_size: int, num_workers: int, num_epochs: int, sample_fraction: float, lr: float, patience: int, env_coverage_km: float, goal_map_scale: float, grad_accum_steps: int, amp: bool):
    print(f"=" * 60)
    print(f"TerraTNT 训练 - 10秒间隔版本")
    print(f"=" * 60)
    print(f"配置:")
    print(f"  采样间隔: 10秒")
    print(f"  观测: {HISTORY_LEN}点 = 15分钟")
    print(f"  预测: {FUTURE_LEN}点 = 60分钟")
    print(f"  Env coverage: {env_coverage_km:.1f} km")
    print(f"  Goal map scale: {goal_map_scale:.2f}")
    print(f"  Sample fraction: {sample_fraction:.2f}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accum: {int(grad_accum_steps)}")
    print(f"  AMP: {bool(amp)}")
    print(f"  Workers: {num_workers}")
    print(f"  Epochs: {num_epochs}")
    print(f"  LR: {lr}")
    print(f"  Patience: {patience}")
    print(f"=" * 60)

class FASDataset(Dataset):
    def __init__(self, traj_dir, fas_split_file, phase='fas1', 
                 history_len=90, future_len=360, num_candidates=6, region='bohemian_forest', sample_fraction=0.5, seed=42, env_coverage_km=140.0, env_local_coverage_km=10.0, coord_scale=1.0, candidate_radius_km=3.0, candidate_center='goal', phase3_missing_goal=False, filter_cfg: str = None, use_dual_scale: bool = False, goal_map_scale: float = 1.0):
        self.traj_dir = Path(traj_dir)
        self.phase = phase
        self.history_len = history_len
        self.future_len = future_len
        self.num_candidates = num_candidates
        self.region = region
        self.sample_fraction = float(sample_fraction)
        self.seed = int(seed)
        self.env_coverage_km = float(env_coverage_km)
        self.env_local_coverage_km = float(env_local_coverage_km)
        self.coord_scale = float(coord_scale)
        self.use_dual_scale = bool(use_dual_scale)
        self.goal_map_scale = float(goal_map_scale)

        # 如果启用双尺度，初始化实时地图生成器
        self.map_generator = None
        if self.use_dual_scale:
            # 10km 局部图，128x128 分辨率 => ~78m/px
            local_cov_m = float(self.env_local_coverage_km) * 1000.0
            self.map_generator = EnvironmentMapGenerator(region, map_size=128, pixel_resolution=local_cov_m / 128.0)

        self.candidate_radius_km = float(candidate_radius_km)
        self.candidate_center = str(candidate_center)
        if self.candidate_center not in ('current', 'goal'):
            raise ValueError(f"Invalid candidate_center: {self.candidate_center}. Must be 'current' or 'goal'.")
        self.phase3_missing_goal = bool(phase3_missing_goal)
        self.filter_cfg = str(filter_cfg) if filter_cfg is not None and str(filter_cfg).strip() else None

        coverage_m = self.env_coverage_km * 1000.0
        resolution_m = coverage_m / 128.0
        half = coverage_m / 2.0
        self._pixel_x_offsets_m = (np.arange(128, dtype=np.float64) + 0.5) * resolution_m - half
        self._pixel_y_offsets_m = half - (np.arange(128, dtype=np.float64) + 0.5) * resolution_m
        
        with open(fas_split_file, 'r') as f:
            splits = json.load(f)

        phase_spec = splits.get(phase, {})
        samples_list = phase_spec.get('samples', None)
        files_list = phase_spec.get('files', None)

        self._file_cache = {}  # 简单缓存：{file_path: data}

        # sample-level split: explicit (file, sample_idx)
        if isinstance(samples_list, list) and len(samples_list) > 0:
            self.samples_meta = []
            for item in samples_list:
                if not isinstance(item, dict):
                    continue
                rel_file = item.get('file', None)
                sample_idx = item.get('sample_idx', None)
                if rel_file is None or sample_idx is None:
                    continue
                traj_file = self.traj_dir / str(rel_file)
                if self.filter_cfg is not None and (not str(traj_file.name).endswith(f"_{self.filter_cfg}.pkl")):
                    continue
                self.samples_meta.append((str(traj_file), int(sample_idx)))

            if self.sample_fraction < 1.0 and len(self.samples_meta) > 0:
                import random
                random.seed(self.seed)
                sample_size = max(1, int(round(len(self.samples_meta) * self.sample_fraction)))
                self.samples_meta = random.sample(self.samples_meta, sample_size)
                self.samples_meta.sort()

            print(f"{phase.upper()}(sample-level): 总计 {len(self.samples_meta)} 样本 (已开启延迟加载)")
            self.file_list = None
        else:
            # file-level split: index all samples inside each file
            self.file_list = list(files_list) if isinstance(files_list, list) else []
            if self.filter_cfg is not None:
                self.file_list = [
                    f for f in self.file_list
                    if str(f).endswith(f"_{self.filter_cfg}.pkl")
                ]
            print(f"{phase.upper()}: {len(self.file_list)} 文件")
            self.samples = []
            self._prepare_samples()
    
    def _prepare_samples(self):
        self.samples_meta = []
        for file_name in tqdm(self.file_list, desc=f"索引{self.phase}"):
            traj_file = self.traj_dir / file_name
            try:
                # 只读取元数据，不保留大数组
                with open(traj_file, 'rb') as f:
                    data = pickle.load(f)
                
                num_samples = len(data.get('samples', []))
                for i in range(num_samples):
                    self.samples_meta.append((str(traj_file), i))
            except Exception:
                continue

        if len(self.samples_meta) == 0:
            return

        if self.sample_fraction < 1.0:
            import random
            random.seed(self.seed)
            sample_size = max(1, int(round(len(self.samples_meta) * self.sample_fraction)))
            self.samples_meta = random.sample(self.samples_meta, sample_size)
            self.samples_meta.sort() # 保持一定程度的文件读取连续性

        print(f"{self.phase.upper()}: 总计 {len(self.samples_meta)} 样本 (已开启延迟加载)")
    
    def __len__(self):
        return len(self.samples_meta)
    
    @staticmethod
    def normalize_env_map(env_map_np: np.ndarray) -> np.ndarray:
        """
        清理环境地图数值，确保没有 NaN/Inf，并对已知范围的通道做合理裁剪。
        
        通道说明：
        0: DEM (高程)
        1: Slope (坡度)
        2-3: Aspect sin/cos
        4-13: LULC one-hot (10 classes)
        14: Tree cover
        15: Road
        16: History heatmap
        17: Goal map (candidates)
        
        Args:
            env_map_np: (18, 128, 128) 原始环境地图
            
        Returns:
            normalized: (18, 128, 128) 归一化后的环境地图
        """
        normalized = np.asarray(env_map_np, dtype=np.float32).copy()
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # Channel 0: DEM
        dem = normalized[0]
        if float(np.nanmax(np.abs(dem))) > 50.0:
            dem_min = float(np.nanmin(dem))
            dem_max = float(np.nanmax(dem))
            if dem_max > dem_min:
                normalized[0] = (dem - dem_min) / (dem_max - dem_min)
            else:
                normalized[0] = 0.0

        # Channel 1: Slope (expect [0,1])
        normalized[1] = np.clip(normalized[1], 0.0, 1.0)

        # Channels 2-3: Aspect sin/cos (expect [-1,1])
        normalized[2] = np.clip(normalized[2], -1.0, 1.0)
        normalized[3] = np.clip(normalized[3], -1.0, 1.0)

        # Channels 4-17: one-hot / masks / heatmaps (expect [0,1])
        normalized[4:18] = np.clip(normalized[4:18], 0.0, 1.0)

        return normalized
    
    def __getitem__(self, idx):
        file_path, sample_idx = self.samples_meta[idx]
        
        # 简单的缓存机制，避免频繁开关同一个文件
        if file_path in self._file_cache:
            data = self._file_cache[file_path]
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            # 限制缓存大小，防止内存再次溢出
            if len(self._file_cache) > 2: 
                self._file_cache.clear()
            self._file_cache[file_path] = data
            
        sample = data['samples'][sample_idx]
        current_pos_abs = np.asarray(sample['current_pos_abs'], dtype=np.float64)

        env_map_np = np.asarray(sample['env_map_100km'], dtype=np.float32)
        # 归一化环境地图到 [0, 1] 范围
        env_map_np = self.normalize_env_map(env_map_np)
        env_map = torch.from_numpy(env_map_np).float()

        history_feat_26d = torch.FloatTensor(sample['history_feat_26d'])
        history = history_feat_26d.clone()
        # 对坐标维度应用coord_scale
        history[:, 0:2] = history[:, 0:2] * self.coord_scale
        
        # 将future_rel（累积位置）转换为增量（每步相对于上一步的位移）
        future_rel = torch.FloatTensor(sample['future_rel'])
        # 计算增量：第一步是相对于起点[0,0]，后续是相对于前一步
        future_delta = torch.diff(future_rel, dim=0, prepend=torch.zeros(1, 2))
        future = future_delta * self.coord_scale  # 应用coord_scale
        
        goal_rel_km = np.asarray(sample['goal_rel'], dtype=np.float64)
        goal = torch.as_tensor(goal_rel_km * self.coord_scale, dtype=torch.float32)

        if self.phase in ('fas1', 'fas2'):
            cand_rel_km = goal_rel_km.reshape(1, 2)
            candidates = torch.as_tensor(cand_rel_km.astype(np.float32), dtype=torch.float32) * self.coord_scale
            target_idx = 0
        else:
            rng = np.random.default_rng((int(self.seed) * 1000003 + int(idx)) % (2**32))
            radius_m = float(self.candidate_radius_km) * 1000.0
            center_offset_m = np.zeros(2, dtype=np.float64)
            if self.candidate_center == 'goal':
                center_offset_m = goal_rel_km * 1000.0

            road = env_map_np[15]
            dx = self._pixel_x_offsets_m[None, :] - center_offset_m[0]
            dy = self._pixel_y_offsets_m[:, None] - center_offset_m[1]
            valid = (road > 0.5) & ((dx * dx + dy * dy) <= (radius_m * radius_m))
            rows, cols = np.where(valid)
            if rows.size > 0:
                sel = rng.choice(rows.size, size=int(self.num_candidates), replace=(rows.size < int(self.num_candidates)))
                cand_rel_km = np.stack(
                    [
                        self._pixel_x_offsets_m[cols[sel]] / 1000.0,
                        self._pixel_y_offsets_m[rows[sel]] / 1000.0,
                    ],
                    axis=1,
                )
            else:
                angles = rng.uniform(0.0, 2.0 * np.pi, size=int(self.num_candidates))
                r = np.sqrt(rng.uniform(0.0, 1.0, size=int(self.num_candidates))) * float(self.candidate_radius_km)
                cand_rel_km = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1) + (center_offset_m / 1000.0)

            include_gt = (self.phase != 'fas3') or (not self.phase3_missing_goal)
            if include_gt and int(self.num_candidates) > 0:
                ins = int(rng.integers(0, int(self.num_candidates)))
                cand_rel_km[ins] = goal_rel_km

            candidates = torch.as_tensor(cand_rel_km.astype(np.float32), dtype=torch.float32) * self.coord_scale
            if int(self.num_candidates) > 0:
                d = torch.norm(candidates - goal.view(1, 2), dim=1)
                target_idx = int(torch.argmin(d).item())
            else:
                target_idx = -1

        # 将候选目标写入最后 1 个通道(Goal map)。覆盖原先生成时的空白 goal_map
        # coverage=100km, patch=128 => ~781.25m/px
        goal_map = torch.zeros(128, 128, dtype=torch.float32)
        coverage_m = self.env_coverage_km * 1000.0
        resolution_m = coverage_m / 128.0
        half = coverage_m / 2.0
        cand_abs_x = current_pos_abs[0] + cand_rel_km[:, 0] * 1000.0
        cand_abs_y = current_pos_abs[1] + cand_rel_km[:, 1] * 1000.0
        for x, y in zip(cand_abs_x.tolist(), cand_abs_y.tolist()):
            col = int((float(x) - (current_pos_abs[0] - half)) / resolution_m)
            row = int(((current_pos_abs[1] + half) - float(y)) / resolution_m)
            if 0 <= row < 128 and 0 <= col < 128:
                r0, r1 = max(0, row - 2), min(128, row + 3)
                c0, c1 = max(0, col - 2), min(128, col + 3)
                goal_map[r0:r1, c0:c1] = 1.0

        scale = float(self.goal_map_scale)
        if scale <= 0.0:
            goal_map.zero_()
        elif scale != 1.0:
            goal_map = (goal_map * scale).clamp(0.0, 1.0)

        env_map[-1] = goal_map
        
        # 实时提取 10km 高清局部图
        env_map_local = torch.zeros_like(env_map)
        if self.map_generator is not None:
            # map_generator 返回 (18, 128, 128)
            local_np = self.map_generator.extract_local_map(
                center_utm=(float(current_pos_abs[0]), float(current_pos_abs[1])),
                history_points=sample['history_feat_26d'][:, 0:2] * 1000.0 + current_pos_abs, # 转换回绝对坐标
                goal_utm=(float(current_pos_abs[0] + goal_rel_km[0] * 1000.0), 
                          float(current_pos_abs[1] + goal_rel_km[1] * 1000.0))
            )
            # 归一化局部环境地图
            local_np = self.normalize_env_map(local_np)
            env_map_local = torch.from_numpy(local_np).float()

        return {
            'history': history,
            'future': future,
            'candidates': candidates,
            'env_map': env_map,
            'env_map_local': env_map_local,
            'target_goal_idx': target_idx,
            'goal': goal,
            'current_pos_abs': torch.as_tensor(current_pos_abs, dtype=torch.float32),
            'file_path': str(file_path),
            'sample_idx': int(sample_idx),
        }

def train_phase(
    phase,
    region='bohemian_forest',
    traj_dir=None,
    fas_split_file=None,
    save_root='/home/zmc/文档/programwork/runs',
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_num_workers=8,
    learning_rate=LEARNING_RATE,
    sample_fraction=0.5,
    seed=42,
    patience=5,
    num_candidates=6,
    env_coverage_km=140.0,
    env_local_coverage_km=10.0,
    history_feature_mode: str = 'full26',
    standardize_history: bool = False,
    hist_stats_samples: int = 2000,
    goal_mode: str = 'joint',
    cls_weight: float = 1.0,
    candidate_radius_km: float = 3.0,
    candidate_center: str = 'goal',
    phase3_missing_goal: bool = False,
    goal_map_scale: float = 1.0,
    paper_mode: bool = False,
    paper_decoder: str = 'hierarchical',
    waypoint_stride: int = 0,
    closed_loop_env_sampling: bool = False,
    wp_weight: float = 1.0,
    traj_wp_align_weight: float = 1.0,
    var_weight: float = 1.0,
    end_loss_weight: float = 0.0,
    waypoint_tf_ratio: float = -1.0,
    tf_start: float = 0.5,
    tf_end: float = 0.05,
    tf_decay_epochs: int = 15,
    grad_accum_steps: int = 1,
    amp: bool = False,
    filter_cfg: str = None,
    resume_checkpoint: str = None,
    use_dual_scale: bool = True,
    override_goal_vec_scale: float = None,
    freeze_goal_vec_scale: bool = False,
    override_env_local_scale: float = None,
    freeze_env_local_scale: bool = False,
    override_env_local_scale2: float = None,
    freeze_env_local_scale2: bool = False,
    goal_vec_scale_start: float = None,
    goal_vec_scale_end: float = None,
    goal_vec_scale_anneal_epochs: int = 0,
    env_local_scale2_start: float = None,
    env_local_scale2_end: float = None,
    env_local_scale2_anneal_epochs: int = 0,
    goal_vec_use_waypoint_after_epoch: int = -1,
    force_goal_vec_use_waypoint: int = None,
    env_loss_weight: float = 0.0,
    env_imp_weight: float = 1.0,
    env_unknown_weight: float = 0.2,
    env_slope_weight: float = 0.5,
    env_slope_thr_deg: float = 30.0,
    env_sample_stride: int = 4,
    curved_norm_dev_weight: float = 0.0,
    curved_norm_dev_threshold: float = 0.08,
    curved_norm_dev_ratio: float = 0.5,
    curved_norm_dev_beta: float = 0.0,
    straight_keep_weight: float = 0.0,
    aux_seg_weight: float = 0.0,
    use_heading: bool = False,
    heading_loss_weight: float = 1.0,
):
    print(f"\n训练 {phase.upper()} (Region: {region})")
    
    if traj_dir is None:
        cand_full = Path(f'/home/zmc/文档/programwork/data/processed/complete_dataset_10s_full/{region}')
        cand_default = Path(f'/home/zmc/文档/programwork/data/processed/complete_dataset_10s/{region}')
        if cand_full.exists():
            traj_dir = str(cand_full)
        else:
            traj_dir = str(cand_default)
        print(f"✓ traj_dir={traj_dir}", flush=True)

    if fas_split_file is None:
        # Prefer trajectory-level split if available
        traj_level = f'/home/zmc/文档/programwork/data/processed/fas_splits/{region}/fas_splits_trajlevel.json'
        if os.path.exists(traj_level):
            fas_split_file = traj_level
            print(f"✓ Using trajectory-level split: {fas_split_file}", flush=True)
        else:
            fas_split_file = f'/home/zmc/文档/programwork/data/processed/fas_splits/{region}/fas_splits.json'

    # Check if trajectory-level split is available
    with open(fas_split_file) as _sf:
        _split_data = json.load(_sf)
    _phase_spec = _split_data.get(phase, {})
    _has_traj_split = 'train_samples' in _phase_spec and 'val_samples' in _phase_spec

    _ds_kwargs = dict(
        traj_dir=traj_dir, fas_split_file=fas_split_file, phase=phase,
        history_len=HISTORY_LEN, future_len=FUTURE_LEN,
        num_candidates=num_candidates, region=region,
        sample_fraction=sample_fraction, seed=seed,
        env_coverage_km=env_coverage_km, env_local_coverage_km=env_local_coverage_km,
        candidate_radius_km=float(candidate_radius_km),
        candidate_center=str(candidate_center),
        phase3_missing_goal=bool(phase3_missing_goal),
        filter_cfg=filter_cfg, use_dual_scale=bool(use_dual_scale),
        goal_map_scale=float(goal_map_scale),
    )

    if _has_traj_split:
        print(f"✓ Trajectory-level split detected, 0 trajectory overlap", flush=True)
        # Train dataset
        train_dataset = FASDataset(**_ds_kwargs)
        train_dataset.samples_meta = []
        for item in _phase_spec['train_samples']:
            tf = str(Path(traj_dir) / item['file'])
            train_dataset.samples_meta.append((tf, int(item['sample_idx'])))
        if float(sample_fraction) < 1.0:
            import random as _rnd
            _rnd.seed(int(seed))
            _k = max(1, int(len(train_dataset.samples_meta) * float(sample_fraction)))
            train_dataset.samples_meta = _rnd.sample(train_dataset.samples_meta, _k)
            train_dataset.samples_meta.sort()

        # Val dataset
        val_dataset = FASDataset(**{**_ds_kwargs, 'sample_fraction': 1.0})
        val_dataset.samples_meta = []
        for item in _phase_spec['val_samples']:
            tf = str(Path(traj_dir) / item['file'])
            val_dataset.samples_meta.append((tf, int(item['sample_idx'])))

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        dataset = train_dataset  # alias for checkpoint saving
    else:
        dataset = FASDataset(**_ds_kwargs)

        if int(len(dataset)) <= 0:
            raise RuntimeError(
                f"Empty dataset: phase={phase}, region={region}, traj_dir={traj_dir}, fas_split_file={fas_split_file}. "
                f"Please check split file paths or pass --traj_dir to the correct dataset directory."
            )

        n_total = int(len(dataset))
        val_size = int(n_total * 0.2)
        if n_total >= 2:
            val_size = max(1, val_size)
            train_size = n_total - val_size
            if train_size <= 0:
                train_size = 1
                val_size = n_total - train_size
        else:
            train_size = n_total
            val_size = 0
        split_gen = torch.Generator()
        split_gen.manual_seed(int(seed))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=split_gen)
    
    train_loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': int(train_num_workers),
        'pin_memory': True,
    }
    if int(train_num_workers) > 0:
        train_loader_kwargs['prefetch_factor'] = 2  # 降低预加载减少内存占用

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"训练batches: {len(train_loader)}, 验证batches: {len(val_loader)}")
    
    from models.terratnt import TerraTNT

    # 若从旧 checkpoint resume，且未显式指定 waypoint_stride，则尝试从 checkpoint 推断
    # 以避免 decoder.wp_query 等参数形状不匹配导致 load_state_dict 报错
    if resume_checkpoint is not None and int(waypoint_stride) <= 0:
        try:
            ckpt_probe = torch.load(str(resume_checkpoint), map_location='cpu')
            inferred = int(waypoint_stride)
            if isinstance(ckpt_probe, dict):
                if ckpt_probe.get('waypoint_stride', None) is not None:
                    inferred = int(ckpt_probe.get('waypoint_stride'))
                elif isinstance(ckpt_probe.get('config', None), dict) and ckpt_probe['config'].get('waypoint_stride', None) is not None:
                    inferred = int(ckpt_probe['config'].get('waypoint_stride'))

                if inferred <= 0:
                    sd = ckpt_probe.get('model_state_dict', None)
                    if isinstance(sd, dict):
                        wpq = sd.get('decoder.wp_query', None)
                        if isinstance(wpq, torch.Tensor) and wpq.dim() == 2:
                            num_waypoints = int(wpq.shape[0]) + 1
                            if num_waypoints > 0:
                                inferred = int(FUTURE_LEN // num_waypoints)
                        if inferred <= 0:
                            wte = sd.get('decoder.waypoint_time_embed', None)
                            if isinstance(wte, torch.Tensor) and wte.dim() == 2:
                                num_waypoints = int(wte.shape[0]) + 1
                                if num_waypoints > 0:
                                    inferred = int(FUTURE_LEN // num_waypoints)

            if int(inferred) > 0:
                waypoint_stride = int(inferred)
                print(f"✓ inferred waypoint_stride={int(waypoint_stride)} from resume_checkpoint for shape-compat", flush=True)
        except Exception as e:
            print(f"⚠️ waypoint_stride 推断失败（继续使用参数 waypoint_stride={int(waypoint_stride)}）: {e}", flush=True)
    history_feature_mode = str(history_feature_mode)
    if history_feature_mode == 'xy':
        history_input_dim = 2
    elif history_feature_mode == 'kin10':
        history_input_dim = 10
    else:
        history_input_dim = 26
    model = TerraTNT({
        'history_len': HISTORY_LEN,
        'future_len': FUTURE_LEN,
        'history_feature_dim': 128, # LSTM hidden
        'history_input_dim': int(history_input_dim),
        'env_feature_dim': 128,     # CNN output (对齐论文)
        'decoder_hidden_dim': 256,
        'num_goals': int(num_candidates),
        'env_channels': 18,
        'output_length': FUTURE_LEN,
        'env_coverage_km': float(env_coverage_km),
        'env_local_coverage_km': float(env_local_coverage_km),
        'paper_mode': bool(paper_mode),
        'paper_decoder': str(paper_decoder),
        'waypoint_stride': int(waypoint_stride),
        'closed_loop_env_sampling': bool(closed_loop_env_sampling),
        'use_dual_scale': bool(use_dual_scale),
        'goal_vec_use_waypoint': True,
        'use_heading': bool(use_heading),
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    start_epoch = 0
    resume_val_ade = None
    resume_val_fde = None
    resume_standardize_history = None
    resume_hist_mean = None
    resume_hist_std = None
    resume_optimizer_state = None
    if resume_checkpoint is not None:
        try:
            ckpt = torch.load(str(resume_checkpoint), map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                resume_optimizer_state = ckpt.get('optimizer_state_dict', None)
                if ckpt.get('epoch', None) is not None:
                    start_epoch = int(ckpt.get('epoch')) + 1
                if ckpt.get('val_ade', None) is not None:
                    resume_val_ade = float(ckpt.get('val_ade'))
                if ckpt.get('val_fde', None) is not None:
                    resume_val_fde = float(ckpt.get('val_fde'))

                resume_standardize_history = ckpt.get('standardize_history', None)
                resume_hist_mean = ckpt.get('hist_mean', None)
                resume_hist_std = ckpt.get('hist_std', None)
            else:
                model.load_state_dict(ckpt, strict=False)
            msg = f"✓ resume from {resume_checkpoint} (start_epoch={start_epoch})"
            if resume_val_ade is not None or resume_val_fde is not None:
                msg += f" (prev ADE={resume_val_ade}, FDE={resume_val_fde})"
            print(msg, flush=True)
        except Exception as e:
            print(f"⚠️ resume_checkpoint 加载失败: {e}", flush=True)

    if override_goal_vec_scale is not None and getattr(model, 'decoder', None) is not None and hasattr(model.decoder, 'goal_vec_scale'):
        model.decoder.goal_vec_scale.data.fill_(float(override_goal_vec_scale))
        if bool(freeze_goal_vec_scale):
            model.decoder.goal_vec_scale.requires_grad_(False)
        print(f"✓ override goal_vec_scale={float(model.decoder.goal_vec_scale.detach().cpu().item()):.4f} (freeze={bool(freeze_goal_vec_scale)})", flush=True)

    if override_env_local_scale is not None and getattr(model, 'decoder', None) is not None:
        if hasattr(model.decoder, 'env_local_scale_step'):
            model.decoder.env_local_scale_step.data.fill_(float(override_env_local_scale))
            if bool(freeze_env_local_scale):
                model.decoder.env_local_scale_step.requires_grad_(False)
            print(f"✓ override env_local_scale_step={float(model.decoder.env_local_scale_step.detach().cpu().item()):.4f} (freeze={bool(freeze_env_local_scale)})", flush=True)
        if hasattr(model.decoder, 'env_local_scale_wp'):
            model.decoder.env_local_scale_wp.data.fill_(float(override_env_local_scale))
            if bool(freeze_env_local_scale):
                model.decoder.env_local_scale_wp.requires_grad_(False)
            print(f"✓ override env_local_scale_wp={float(model.decoder.env_local_scale_wp.detach().cpu().item()):.4f} (freeze={bool(freeze_env_local_scale)})", flush=True)
        if (not hasattr(model.decoder, 'env_local_scale_step')) and (not hasattr(model.decoder, 'env_local_scale_wp')) and hasattr(model.decoder, 'env_local_scale'):
            model.decoder.env_local_scale.data.fill_(float(override_env_local_scale))
            if bool(freeze_env_local_scale):
                model.decoder.env_local_scale.requires_grad_(False)
            print(f"✓ override env_local_scale={float(model.decoder.env_local_scale.detach().cpu().item()):.4f} (freeze={bool(freeze_env_local_scale)})", flush=True)

    if override_env_local_scale2 is not None and getattr(model, 'decoder', None) is not None and hasattr(model.decoder, 'env_local_scale2'):
        model.decoder.env_local_scale2.data.fill_(float(override_env_local_scale2))
        if bool(freeze_env_local_scale2):
            model.decoder.env_local_scale2.requires_grad_(False)
        print(f"✓ override env_local_scale2={float(model.decoder.env_local_scale2.detach().cpu().item()):.4f} (freeze={bool(freeze_env_local_scale2)})", flush=True)

    if force_goal_vec_use_waypoint is not None and getattr(model, 'decoder', None) is not None:
        model.decoder.goal_vec_use_waypoint = bool(int(force_goal_vec_use_waypoint))
        print(f"✓ force goal_vec_use_waypoint={bool(model.decoder.goal_vec_use_waypoint)}", flush=True)

    def _sched_value(start_v: float, end_v: float, k: int, total_k: int) -> float:
        if total_k <= 0:
            return float(start_v)
        kk = max(0, int(k))
        tt = max(1, int(total_k))
        prog = min(1.0, float(kk) / float(tt))
        return float(start_v) + (float(end_v) - float(start_v)) * prog

    use_goal_vec_schedule = (goal_vec_scale_start is not None) and (goal_vec_scale_end is not None) and (int(goal_vec_scale_anneal_epochs) > 0)
    use_env_local2_schedule = (env_local_scale2_start is not None) and (env_local_scale2_end is not None) and (int(env_local_scale2_anneal_epochs) > 0)
    
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_l = n.lower()
        if ('spatial_in' in n_l) or n.endswith('.bias') or (p.dim() <= 1) or ('bn' in n_l) or ('norm' in n_l):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.Adam(
        [
            {'params': decay_params, 'weight_decay': 1e-3},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate,
    )
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            for g in optimizer.param_groups:
                g['lr'] = float(learning_rate)
            print("✓ resume optimizer_state_dict", flush=True)
        except Exception as e:
            print(f"⚠️ resume optimizer_state_dict failed: {e}", flush=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    accum_steps = max(1, int(grad_accum_steps))
    amp_enabled = bool(amp) and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled)
    
    # 论文逻辑：双任务损失函数
    traj_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.BCEWithLogitsLoss()

    def norm_dev(traj_xy: torch.Tensor) -> torch.Tensor:
        end = traj_xy[:, -1, :]
        seg = end
        seg_len = torch.norm(seg, dim=-1).clamp_min(1e-6)
        seg_unit = seg / seg_len.unsqueeze(-1)
        proj = (traj_xy * seg_unit.unsqueeze(1)).sum(dim=-1)
        perp = traj_xy - proj.unsqueeze(-1) * seg_unit.unsqueeze(1)
        perp_dist = torch.norm(perp, dim=-1)
        beta = float(curved_norm_dev_beta)
        if beta > 0.0:
            max_dev = torch.logsumexp(perp_dist * beta, dim=1) / beta
        else:
            max_dev = perp_dist.max(dim=1).values
        return max_dev / seg_len

    def curved_hinge_loss(pred_cumsum: torch.Tensor, gt_cumsum: torch.Tensor) -> torch.Tensor:
        if float(curved_norm_dev_weight) <= 0.0:
            return torch.zeros((), device=pred_cumsum.device, dtype=pred_cumsum.dtype)
        gt_nd = norm_dev(gt_cumsum)
        pred_nd = norm_dev(pred_cumsum)
        thr = float(curved_norm_dev_threshold)
        mask = gt_nd >= thr
        if not bool(mask.any().detach().cpu().item()):
            return torch.zeros((), device=pred_cumsum.device, dtype=pred_cumsum.dtype)
        target = torch.maximum(
            gt_nd.new_full(gt_nd.shape, float(thr)),
            float(curved_norm_dev_ratio) * gt_nd,
        )
        per = F.relu(target - pred_nd)
        per = per * mask.to(per.dtype)
        denom = mask.to(per.dtype).sum().clamp_min(1.0)
        return per.sum() / denom

    def straight_keep_loss(pred_cumsum: torch.Tensor, gt_cumsum: torch.Tensor) -> torch.Tensor:
        if float(straight_keep_weight) <= 0.0:
            return torch.zeros((), device=pred_cumsum.device, dtype=pred_cumsum.dtype)
        gt_nd = norm_dev(gt_cumsum)
        pred_nd = norm_dev(pred_cumsum)
        thr = float(curved_norm_dev_threshold)
        mask = gt_nd < thr
        if not bool(mask.any().detach().cpu().item()):
            return torch.zeros((), device=pred_cumsum.device, dtype=pred_cumsum.dtype)
        per = F.relu(pred_nd - thr)
        per = per * mask.to(per.dtype)
        denom = mask.to(per.dtype).sum().clamp_min(1.0)
        return per.sum() / denom

    def env_consistency_loss(pred_cumsum: torch.Tensor, env_map: torch.Tensor) -> torch.Tensor:
        if float(env_loss_weight) <= 0.0:
            return torch.zeros((), device=pred_cumsum.device, dtype=pred_cumsum.dtype)
        stride = max(1, int(env_sample_stride))
        pos = pred_cumsum[:, ::stride, :]
        half = float(env_coverage_km) * 0.5
        half = max(1e-6, half)
        gx = (pos[..., 0] / half).clamp(-1.0, 1.0)
        gy = (-pos[..., 1] / half).clamp(-1.0, 1.0)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)
        chans = env_map[:, [1, 8, 9, 12, 13], :, :]
        samp = F.grid_sample(chans, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        samp = samp.squeeze(-1)
        slope = samp[:, 0, :]
        wetland = samp[:, 1, :]
        water = samp[:, 2, :]
        ice = samp[:, 3, :]
        unknown = samp[:, 4, :]

        imp = (wetland + water + ice).clamp_min(0.0)
        thr = float(env_slope_thr_deg) / 90.0
        thr = max(0.0, min(1.0, thr))
        slope_pen = F.relu(slope - thr)

        loss_imp = imp.mean()
        loss_unk = unknown.mean()
        loss_slope = slope_pen.mean()
        return (
            float(env_imp_weight) * loss_imp +
            float(env_unknown_weight) * loss_unk +
            float(env_slope_weight) * loss_slope
        )
    
    # 轨迹变化损失：鼓励预测的增量有变化，避免退化成常数速度
    def trajectory_variation_loss(pred_delta, target_delta):
        """
        计算预测增量的变化损失
        pred_delta: (batch, seq_len, 2) 预测的增量
        target_delta: (batch, seq_len, 2) 真实的增量
        """
        # 计算增量的二阶差分（加速度）
        batch_size = pred_delta.size(0)
        prepend_zeros = torch.zeros(batch_size, 1, 2, device=pred_delta.device, dtype=pred_delta.dtype)
        pred_accel = torch.diff(pred_delta, dim=1, prepend=prepend_zeros)  # (batch, seq_len, 2)
        target_accel = torch.diff(target_delta, dim=1, prepend=prepend_zeros)
        
        # 加速度的MSE损失
        accel_loss = F.mse_loss(pred_accel, target_accel)
        
        # 增量标准差损失：鼓励预测的增量有足够的变化
        pred_std = pred_delta.std(dim=1).mean()  # 每个样本内部的标准差
        target_std = target_delta.std(dim=1).mean()
        std_loss = F.mse_loss(pred_std, target_std)
        
        return accel_loss + 0.5 * std_loss

    def curvature_consistency_loss(pred_delta, target_delta):
        """
        计算曲率一致性损失，防止轨迹为了对齐终点而变得过直
        """
        # 计算角度变化
        pred_eps = 1e-6
        pred_angles = torch.atan2(pred_delta[..., 1], pred_delta[..., 0])
        target_angles = torch.atan2(target_delta[..., 1], target_delta[..., 0])
        
        # 计算角速度（方向变化率）
        pred_angle_diff = torch.diff(pred_angles, dim=1)
        target_angle_diff = torch.diff(target_angles, dim=1)
        
        # 强制要求方向变化的幅度（能量）一致
        pred_energy = torch.mean(pred_angle_diff**2, dim=1)
        target_energy = torch.mean(target_angle_diff**2, dim=1)
        
        return F.mse_loss(pred_energy, target_energy)
    
    save_dir = Path(save_root) / f'terratnt_{phase}_10s' / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_ade = float('inf')
    best_val_ade_saved = float('inf')
    if resume_val_ade is not None:
        best_val_ade = resume_val_ade
        best_val_ade_saved = resume_val_ade
    patience_counter = 0

    first_train_exception = None
    first_val_exception = None
    
    initial_tf_ratio = float(tf_start)
    final_tf_ratio = float(tf_end)
    tf_decay_epochs = int(tf_decay_epochs)

    def _slice_history(h: torch.Tensor) -> torch.Tensor:
        if history_feature_mode == 'xy':
            return h[:, :, 0:2]
        if history_feature_mode == 'kin10':
            return h[:, :, 0:10]
        return h

    hist_mean = None
    hist_std = None

    if bool(resume_standardize_history) and isinstance(resume_hist_mean, torch.Tensor) and isinstance(resume_hist_std, torch.Tensor):
        standardize_history = True
        hist_mean = resume_hist_mean.to(device)
        hist_std = resume_hist_std.to(device)
        print("✓ inherit hist standardization from resume checkpoint", flush=True)

    if bool(standardize_history):
        stats_bs = min(128, int(batch_size))
        stats_loader = DataLoader(
            train_dataset,
            batch_size=stats_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        sum_x = None
        sum_x2 = None
        count = 0
        for b in stats_loader:
            h = b['history']
            h = _slice_history(h)
            h = h.reshape(-1, h.shape[-1]).double()
            if sum_x is None:
                sum_x = torch.zeros(h.shape[-1], dtype=torch.float64)
                sum_x2 = torch.zeros(h.shape[-1], dtype=torch.float64)
            sum_x += h.sum(dim=0)
            sum_x2 += (h * h).sum(dim=0)
            count += int(h.shape[0])
            if int(hist_stats_samples) > 0 and count >= int(hist_stats_samples):
                break
        mean = sum_x / max(1, count)
        var = (sum_x2 / max(1, count)) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        cont_dims = min(14, int(mean.shape[0]))
        mean_all = torch.zeros_like(mean)
        std_all = torch.ones_like(std)
        mean_all[:cont_dims] = mean[:cont_dims]
        std_all[:cont_dims] = std[:cont_dims]
        hist_mean = mean_all.float().view(1, 1, -1).to(device)
        hist_std = std_all.float().view(1, 1, -1).to(device)
    
    goal_mode = str(goal_mode)
    if goal_mode not in ('given', 'joint'):
        raise ValueError(f"Invalid goal_mode: {goal_mode}. Must be 'given' or 'joint'.")

    effective_goal_mode = goal_mode

    for epoch in range(int(start_epoch), int(start_epoch) + int(num_epochs)):
        model.train()

        rel_ep = int(epoch) - int(start_epoch)

        # 1) progressively enable waypoint-based goal guidance
        if (force_goal_vec_use_waypoint is None) and getattr(model, 'decoder', None) is not None and hasattr(model.decoder, 'goal_vec_use_waypoint'):
            if int(goal_vec_use_waypoint_after_epoch) >= 0 and rel_ep >= int(goal_vec_use_waypoint_after_epoch):
                if not bool(model.decoder.goal_vec_use_waypoint):
                    model.decoder.goal_vec_use_waypoint = True
                    print(f"[schedule] epoch={epoch} enable goal_vec_use_waypoint=True", flush=True)

        # 2) anneal goal_vec_scale down
        if use_goal_vec_schedule and getattr(model, 'decoder', None) is not None and hasattr(model.decoder, 'goal_vec_scale') and (not bool(freeze_goal_vec_scale)):
            gv = _sched_value(float(goal_vec_scale_start), float(goal_vec_scale_end), rel_ep, int(goal_vec_scale_anneal_epochs))
            model.decoder.goal_vec_scale.data.fill_(float(gv))
            print(f"[schedule] epoch={epoch} goal_vec_scale={float(gv):.4f}", flush=True)

        # 3) ramp env_local_scale2 up
        if use_env_local2_schedule and getattr(model, 'decoder', None) is not None and hasattr(model.decoder, 'env_local_scale2') and (not bool(freeze_env_local_scale2)):
            ev = _sched_value(float(env_local_scale2_start), float(env_local_scale2_end), rel_ep, int(env_local_scale2_anneal_epochs))
            model.decoder.env_local_scale2.data.fill_(float(ev))
            print(f"[schedule] epoch={epoch} env_local_scale2={float(ev):.4f}", flush=True)
        train_loss = 0
        train_loss_traj = 0
        train_loss_var = 0
        train_loss_cls = 0
        train_loss_wp = 0
        train_loss_traj_wp = 0
        train_loss_env = 0
        train_loss_curved = 0
        train_loss_straight_keep = 0
        train_loss_seg = 0
        train_pos_mse = 0
        train_pred_mean = torch.zeros(2, device=device)
        train_pred_std = torch.zeros(2, device=device)
        train_ade = 0
        train_fde = 0
        train_ok_batches = 0
        
        # 计算当前 epoch 的教师强制比率 (线性衰减)
        denom = max(1, int(tf_decay_epochs))
        current_tf_ratio = max(
            float(final_tf_ratio),
            float(initial_tf_ratio) - (float(initial_tf_ratio) - float(final_tf_ratio)) * (float(rel_ep) / float(denom))
        )
        
        wp_tf_ratio = None if float(waypoint_tf_ratio) < 0.0 else float(waypoint_tf_ratio)
        accum_counter = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} (TF={current_tf_ratio:.2f})", file=sys.stdout)):
            history = batch['history'].to(device)
            history = _slice_history(history)
            if hist_mean is not None and hist_std is not None:
                history = (history - hist_mean) / hist_std
            future = batch['future'].to(device)
            candidates = batch['candidates'].to(device)
            env_map = batch['env_map'].to(device)
            env_map_local = batch.get('env_map_local', None)
            if env_map_local is not None:
                env_map_local = env_map_local.to(device)
            target_goal_idx = batch['target_goal_idx'].to(device)
            goal = batch['goal'].to(device)
            
            if epoch == 0 and batch_idx == 0:
                print(f"\n✓ 数据检查:")
                print(f"  history: {history.shape} (batch, {HISTORY_LEN}, {history.shape[-1]})")
                print(f"  future: {future.shape} (batch, {FUTURE_LEN}, 2)")
                print(f"  env_map: {env_map.shape}")
                print(f"  非零元素: {(env_map != 0).sum().item() / env_map.numel() * 100:.1f}%")
            
            if accum_counter == 0:
                optimizer.zero_grad(set_to_none=True)
            current_pos = torch.zeros(history.size(0), 2).to(device)
            
            try:
                # model 返回 (predictions, goal_logits)
                with autocast(enabled=amp_enabled):
                    if effective_goal_mode == 'given':
                        pred, goal_logits, aux = model(
                            env_map,
                            history,
                            candidates,
                            current_pos,
                            teacher_forcing_ratio=current_tf_ratio,
                            waypoint_teacher_forcing_ratio=wp_tf_ratio,
                            ground_truth=future,
                            goal=goal,
                            return_aux=True,
                            env_map_local=env_map_local,
                        )
                    else:
                        pred, goal_logits, aux = model(
                            env_map,
                            history,
                            candidates,
                            current_pos,
                            teacher_forcing_ratio=current_tf_ratio,
                            waypoint_teacher_forcing_ratio=wp_tf_ratio,
                            ground_truth=future,
                            target_goal_idx=target_goal_idx,
                            use_gt_goal=True,
                            return_aux=True,
                            env_map_local=env_map_local,
                        )
                    
                    # 1. 轨迹回归损失 (Kilometers) - Delta MSE
                    loss_traj = traj_criterion(pred, future)
                    
                    # 2. 累积轨迹损失 - ADE 和 FDE
                    pred_cumsum = torch.cumsum(pred, dim=1)  # (batch, T, 2)
                    future_cumsum = torch.cumsum(future, dim=1)
                    
                    # ADE: Average Displacement Error (平均路径误差)
                    dist = torch.norm(pred_cumsum - future_cumsum, dim=-1)  # (batch, T)
                    loss_ade = torch.mean(dist)
                    
                    # FDE: Final Displacement Error (终点误差)
                    loss_fde = torch.mean(dist[:, -1])
                    
                    # 3. 轨迹变化损失：鼓励预测有变化的轨迹
                    if not paper_mode:
                        loss_var = trajectory_variation_loss(pred, future)
                        loss_curv = curvature_consistency_loss(pred, future)
                    else:
                        loss_var = torch.zeros((), device=device)
                        loss_curv = torch.zeros((), device=device)
                    
                    loss_env = env_consistency_loss(pred_cumsum, env_map)
                    
                    loss_curved = curved_hinge_loss(pred_cumsum, future_cumsum)
                    loss_straight_keep = straight_keep_loss(pred_cumsum, future_cumsum)
                    
                    loss_end = torch.mean(torch.norm(pred_cumsum[:, -1, :] - goal, dim=-1))
                
                # 4. 目标分类损失 (仅 Phase3)
                if effective_goal_mode == 'joint':
                    if goal_logits is None:
                        raise RuntimeError("goal_logits is None in joint mode")
                    loss_cls = cls_criterion(goal_logits, target_goal_idx)
                else:
                    loss_cls = torch.zeros((), device=device)

                # 5. Waypoint 监督损失
                loss_wp = torch.zeros((), device=device)
                if isinstance(aux, dict) and ((not paper_mode) or float(wp_weight) > 0.0):
                    pred_wp = aux.get('pred_waypoints', None)
                    wp_idx = aux.get('waypoint_indices', None)
                    if isinstance(pred_wp, torch.Tensor) and pred_wp.numel() > 0 and wp_idx is not None:
                        gt_pos = future_cumsum  # 复用已计算的累积轨迹
                        wp_idx_t = torch.as_tensor(wp_idx, device=gt_pos.device, dtype=torch.long)
                        if wp_idx_t.numel() > 0 and int(wp_idx_t.max().item()) < gt_pos.size(1):
                            gt_wp = gt_pos.index_select(1, wp_idx_t)
                            if gt_wp.shape == pred_wp.shape:
                                loss_wp = F.mse_loss(pred_wp, gt_wp)

                loss_traj_wp = torch.zeros((), device=device)
                if isinstance(aux, dict) and float(traj_wp_align_weight) > 0.0:
                    wp_idx = aux.get('waypoint_indices', None)
                    if wp_idx is not None:
                        wp_idx_t = torch.as_tensor(wp_idx, device=pred_cumsum.device, dtype=torch.long)
                        if wp_idx_t.numel() > 0 and int(wp_idx_t.max().item()) < pred_cumsum.size(1):
                            pred_wp_pos = pred_cumsum.index_select(1, wp_idx_t)
                            gt_wp_pos = future_cumsum.index_select(1, wp_idx_t)
                            if pred_wp_pos.shape == gt_wp_pos.shape:
                                loss_traj_wp = F.mse_loss(pred_wp_pos, gt_wp_pos)

                # 6. Auxiliary Segmentation Loss
                loss_seg = torch.zeros((), device=device)
                if isinstance(aux, dict) and ((not paper_mode) or float(aux_seg_weight) > 0.0):
                    local_seg_logits = aux.get('local_seg_logits', None)
                    if local_seg_logits is not None and env_map_local is not None:
                        # Index 15 is Road channel. Target shape (B, 1, H, W)
                        seg_target = env_map_local[:, 15:16, :, :]
                        loss_seg = seg_criterion(local_seg_logits, seg_target)

                # 7. Heading (转向角) Loss
                loss_heading = torch.zeros((), device=device)
                if bool(use_heading) and float(heading_loss_weight) > 0.0 and isinstance(aux, dict):
                    heading_preds = aux.get('heading_preds', None)
                    if heading_preds is not None:
                        # GT heading changes from future delta (match dtype for AMP)
                        _fut = future.float()
                        gt_angles = torch.atan2(_fut[:, :, 1], _fut[:, :, 0])  # (B, T)
                        gt_dtheta = torch.diff(gt_angles, dim=1, prepend=torch.zeros(future.size(0), 1, device=device))
                        import math
                        gt_dtheta = (gt_dtheta + math.pi) % (2 * math.pi) - math.pi
                        gt_sincos = torch.stack([torch.sin(gt_dtheta), torch.cos(gt_dtheta)], dim=-1)  # (B, T, 2)
                        loss_heading = F.mse_loss(heading_preds.float(), gt_sincos)

                # 损失函数组合
                if paper_mode:
                    # Paper Mode: 仅使用 Delta MSE 和 Classification Loss
                    # 论文公式: L = L_cls + L_traj
                    # 注意: loss_traj 是 Delta MSE
                    loss = (
                        loss_traj +
                        loss_cls +
                        float(env_loss_weight) * loss_env +
                        float(curved_norm_dev_weight) * loss_curved +
                        float(straight_keep_weight) * loss_straight_keep +
                        float(wp_weight) * loss_wp +
                        float(traj_wp_align_weight) * loss_traj_wp +
                        float(aux_seg_weight) * loss_seg +
                        float(end_loss_weight) * loss_end +
                        float(heading_loss_weight) * loss_heading
                    )
                else:
                    # Legacy / Complex Mode: 多任务损失
                    loss = (
                        1.0 * loss_traj +      # Delta MSE
                        10.0 * loss_ade +      # 路径平均误差
                        50.0 * loss_fde +      # 终点误差（最重要）
                        0.1 * loss_cls +       # 分类损失
                        20.0 * loss_wp +       # Waypoint 监督
                        1.0 * loss_curv +      # 曲率一致性
                        float(traj_wp_align_weight) * loss_traj_wp +
                        float(env_loss_weight) * loss_env +
                        float(curved_norm_dev_weight) * loss_curved +
                        float(straight_keep_weight) * loss_straight_keep +
                        float(aux_seg_weight) * loss_seg +
                        float(end_loss_weight) * loss_end +
                        float(heading_loss_weight) * loss_heading
                    )
                
                loss_scaled = loss / float(accum_steps)
                if amp_enabled:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                accum_counter += 1
                if accum_counter >= accum_steps:
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    accum_counter = 0
                
                train_loss += loss.item()
                train_loss_traj += loss_traj.item()
                train_loss_var += loss_var.item()
                train_loss_cls += loss_cls.item()
                train_loss_wp += loss_wp.item()
                train_loss_traj_wp += loss_traj_wp.item()
                train_loss_env += loss_env.item()
                train_loss_curved += loss_curved.item()
                train_loss_straight_keep += loss_straight_keep.item()
                train_loss_seg += loss_seg.item()
                
                # 将预测的增量累加得到累积轨迹，用于计算ADE/FDE
                pred_cumsum = torch.cumsum(pred, dim=1)  # (batch, 360, 2) 累积轨迹
                future_cumsum = torch.cumsum(future, dim=1)  # (batch, 360, 2) 真实累积轨迹
                train_pos_mse += torch.mean((pred_cumsum.detach() - future_cumsum.detach()) ** 2).item()
                train_pred_mean += pred.detach().mean(dim=(0, 1))
                train_pred_std += pred.detach().std(dim=(0, 1))
                
                # ADE/FDE 结果显示为米 (乘以 1000)
                dist = torch.norm(pred_cumsum - future_cumsum, dim=-1)
                train_ade += torch.mean(dist).item() * 1000.0
                train_fde += torch.mean(dist[:, -1]).item() * 1000.0
                train_ok_batches += 1
            except Exception as e:
                if first_train_exception is None:
                    first_train_exception = e
                    import traceback
                    print("\n[TRAIN] First exception traceback:")
                    traceback.print_exc()
                continue
        
        if accum_counter > 0:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            accum_counter = 0

        if train_ok_batches == 0:
            raise RuntimeError("No successful training batches. See first exception traceback above.")

        avg_train_loss = train_loss / train_ok_batches
        avg_train_loss_traj = train_loss_traj / train_ok_batches
        avg_train_loss_var = train_loss_var / train_ok_batches
        avg_train_loss_cls = train_loss_cls / train_ok_batches
        avg_train_loss_wp = train_loss_wp / train_ok_batches
        avg_train_loss_traj_wp = train_loss_traj_wp / train_ok_batches
        avg_train_loss_env = train_loss_env / train_ok_batches
        avg_train_loss_curved = train_loss_curved / train_ok_batches
        avg_train_loss_straight_keep = train_loss_straight_keep / train_ok_batches
        avg_train_loss_seg = train_loss_seg / train_ok_batches
        avg_train_pos_mse = train_pos_mse / train_ok_batches
        avg_train_pred_mean = (train_pred_mean / train_ok_batches).detach().cpu().numpy()
        avg_train_pred_std = (train_pred_std / train_ok_batches).detach().cpu().numpy()
        avg_train_ade = train_ade / train_ok_batches
        avg_train_fde = train_fde / train_ok_batches
        
        model.eval()
        val_loss = 0
        val_loss_traj = 0
        val_loss_var = 0
        val_loss_cls = 0
        val_loss_wp = 0
        val_loss_traj_wp = 0
        val_loss_env = 0
        val_loss_curved = 0
        val_loss_straight_keep = 0
        val_loss_seg = 0
        val_pos_mse = 0
        val_pred_mean = torch.zeros(2, device=device)
        val_pred_std = torch.zeros(2, device=device)
        val_ade = 0
        val_fde = 0
        val_ok_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                history = _slice_history(history)
                if hist_mean is not None and hist_std is not None:
                    history = (history - hist_mean) / hist_std
                future = batch['future'].to(device)
                candidates = batch['candidates'].to(device)
                env_map = batch['env_map'].to(device)
                env_map_local = batch.get('env_map_local', None)
                if env_map_local is not None:
                    env_map_local = env_map_local.to(device)
                target_goal_idx = batch['target_goal_idx'].to(device)
                goal = batch['goal'].to(device)
                current_pos = torch.zeros(history.size(0), 2).to(device)
                
                try:
                    if effective_goal_mode == 'given':
                        pred, goal_logits, aux = model(
                            env_map,
                            history,
                            candidates,
                            current_pos,
                            teacher_forcing_ratio=0.0,
                            ground_truth=future,
                            goal=goal,
                            return_aux=True,
                            env_map_local=env_map_local,
                        )
                    else:
                        pred, goal_logits, aux = model(
                            env_map,
                            history,
                            candidates,
                            current_pos,
                            teacher_forcing_ratio=0.0,
                            ground_truth=future,
                            target_goal_idx=target_goal_idx,
                            use_gt_goal=True,
                            return_aux=True,
                            env_map_local=env_map_local,
                        )

                    # 1. Delta MSE
                    loss_traj = traj_criterion(pred, future)
                    
                    # 2. 累积轨迹损失 - ADE 和 FDE
                    pred_cumsum = torch.cumsum(pred, dim=1)
                    future_cumsum = torch.cumsum(future, dim=1)
                    
                    dist = torch.norm(pred_cumsum - future_cumsum, dim=-1)
                    loss_ade = torch.mean(dist)
                    loss_fde = torch.mean(dist[:, -1])
                    
                    # 3. 轨迹变化损失
                    if not paper_mode:
                        loss_var = trajectory_variation_loss(pred, future)
                        loss_curv = curvature_consistency_loss(pred, future)
                    else:
                        loss_var = torch.zeros((), device=device)
                        loss_curv = torch.zeros((), device=device)

                    loss_env = env_consistency_loss(pred_cumsum, env_map)

                    loss_curved = curved_hinge_loss(pred_cumsum, future_cumsum)
                    loss_straight_keep = straight_keep_loss(pred_cumsum, future_cumsum)

                    loss_end = torch.mean(torch.norm(pred_cumsum[:, -1, :] - goal, dim=-1))
                    
                    # 4. 目标分类损失 (仅 Phase3)
                    if effective_goal_mode == 'joint':
                        if goal_logits is None:
                            raise RuntimeError("goal_logits is None in joint mode")
                        loss_cls = cls_criterion(goal_logits, target_goal_idx)
                    else:
                        loss_cls = torch.zeros((), device=device)

                    # 5. Waypoint 监督损失
                    loss_wp = torch.zeros((), device=device)
                    # 6. Seg Loss
                    loss_seg = torch.zeros((), device=device)
                    
                    if isinstance(aux, dict) and ((not paper_mode) or float(wp_weight) > 0.0 or float(aux_seg_weight) > 0.0):
                        pred_wp = aux.get('pred_waypoints', None)
                        wp_idx = aux.get('waypoint_indices', None)
                        if isinstance(pred_wp, torch.Tensor) and pred_wp.numel() > 0 and wp_idx is not None:
                            gt_pos = future_cumsum  # 复用已计算的累积轨迹
                            wp_idx_t = torch.as_tensor(wp_idx, device=gt_pos.device, dtype=torch.long)
                            if wp_idx_t.numel() > 0 and int(wp_idx_t.max().item()) < gt_pos.size(1):
                                gt_wp = gt_pos.index_select(1, wp_idx_t)
                                if gt_wp.shape == pred_wp.shape:
                                    loss_wp = F.mse_loss(pred_wp, gt_wp)
                        
                        local_seg_logits = aux.get('local_seg_logits', None)
                        if local_seg_logits is not None and env_map_local is not None:
                            seg_target = env_map_local[:, 15:16, :, :]
                            loss_seg = seg_criterion(local_seg_logits, seg_target)

                    loss_traj_wp = torch.zeros((), device=device)
                    if isinstance(aux, dict) and float(traj_wp_align_weight) > 0.0:
                        wp_idx = aux.get('waypoint_indices', None)
                        if wp_idx is not None:
                            wp_idx_t = torch.as_tensor(wp_idx, device=pred_cumsum.device, dtype=torch.long)
                            if wp_idx_t.numel() > 0 and int(wp_idx_t.max().item()) < pred_cumsum.size(1):
                                pred_wp_pos = pred_cumsum.index_select(1, wp_idx_t)
                                gt_wp_pos = future_cumsum.index_select(1, wp_idx_t)
                                if pred_wp_pos.shape == gt_wp_pos.shape:
                                    loss_traj_wp = F.mse_loss(pred_wp_pos, gt_wp_pos)
                    
                    # 损失函数组合
                    if paper_mode:
                        # Paper Mode: 仅使用 Delta MSE 和 Classification Loss
                        loss = (
                            loss_traj +
                            loss_cls +
                            float(env_loss_weight) * loss_env +
                            float(curved_norm_dev_weight) * loss_curved +
                            float(straight_keep_weight) * loss_straight_keep +
                            float(wp_weight) * loss_wp +
                            float(traj_wp_align_weight) * loss_traj_wp +
                            float(aux_seg_weight) * loss_seg +
                            float(end_loss_weight) * loss_end
                        )
                    else:
                        # Legacy / Complex Mode: 多任务损失
                        loss = (
                            1.0 * loss_traj +
                            10.0 * loss_ade +
                            50.0 * loss_fde +
                            0.1 * loss_cls +
                            20.0 * loss_wp +
                            1.0 * loss_curv +
                            float(traj_wp_align_weight) * loss_traj_wp +
                            float(env_loss_weight) * loss_env +
                            float(curved_norm_dev_weight) * loss_curved +
                            float(straight_keep_weight) * loss_straight_keep +
                            float(aux_seg_weight) * loss_seg +
                            float(end_loss_weight) * loss_end
                        )
                    
                    # 统计指标
                    val_pos_mse += torch.mean((pred_cumsum.detach() - future_cumsum.detach()) ** 2).item()
                    val_pred_mean += pred.detach().mean(dim=(0, 1))
                    val_pred_std += pred.detach().std(dim=(0, 1))
                    val_ade += torch.mean(dist).item() * 1000.0
                    val_fde += torch.mean(dist[:, -1]).item() * 1000.0
                    
                    val_loss += loss.item()
                    val_loss_traj += loss_traj.item()
                    val_loss_var += loss_var.item()
                    val_loss_cls += loss_cls.item()
                    val_loss_wp += loss_wp.item()
                    val_loss_traj_wp += loss_traj_wp.item()
                    val_loss_env += loss_env.item()
                    val_loss_curved += loss_curved.item()
                    val_loss_straight_keep += loss_straight_keep.item()
                    val_loss_seg += loss_seg.item()
                    
                    val_ok_batches += 1
                except Exception as e:
                    if first_val_exception is None:
                        first_val_exception = e
                        import traceback
                        print("\n[VAL] First exception traceback:")
                        traceback.print_exc()
                    continue
        
        if val_ok_batches == 0:
            raise RuntimeError("No successful validation batches. See first exception traceback above.")

        avg_val_loss = val_loss / val_ok_batches
        avg_val_loss_traj = val_loss_traj / val_ok_batches
        avg_val_loss_var = val_loss_var / val_ok_batches
        avg_val_loss_cls = val_loss_cls / val_ok_batches
        avg_val_loss_wp = val_loss_wp / val_ok_batches
        avg_val_loss_traj_wp = val_loss_traj_wp / val_ok_batches
        avg_val_pos_mse = val_pos_mse / val_ok_batches
        avg_val_pred_mean = (val_pred_mean / val_ok_batches).detach().cpu().numpy()
        avg_val_pred_std = (val_pred_std / val_ok_batches).detach().cpu().numpy()
        avg_val_ade = val_ade / val_ok_batches
        avg_val_fde = val_fde / val_ok_batches
        avg_val_loss_env = val_loss_env / val_ok_batches
        avg_val_loss_curved = val_loss_curved / val_ok_batches
        avg_val_loss_straight_keep = val_loss_straight_keep / val_ok_batches
        avg_val_loss_seg = val_loss_seg / val_ok_batches

        scheduler.step(avg_val_ade)

        print(f"Epoch {epoch+1}: Train loss={avg_train_loss:.4f} (traj={avg_train_loss_traj:.4f}, var={avg_train_loss_var:.4f}, cls={avg_train_loss_cls:.4f}, wp={avg_train_loss_wp:.4f}, traj_wp={avg_train_loss_traj_wp:.4f}, env={avg_train_loss_env:.4f}, curved={avg_train_loss_curved:.4f}, straight_keep={avg_train_loss_straight_keep:.4f}, seg={avg_train_loss_seg:.4f}), pos_mse={avg_train_pos_mse:.6f}, pred_mean=({avg_train_pred_mean[0]:.4f},{avg_train_pred_mean[1]:.4f}), pred_std=({avg_train_pred_std[0]:.4f},{avg_train_pred_std[1]:.4f}), ADE={avg_train_ade:.1f}m, FDE={avg_train_fde:.1f}m | Val loss={avg_val_loss:.4f} (traj={avg_val_loss_traj:.4f}, var={avg_val_loss_var:.4f}, cls={avg_val_loss_cls:.4f}, wp={avg_val_loss_wp:.4f}, traj_wp={avg_val_loss_traj_wp:.4f}, env={avg_val_loss_env:.4f}, curved={avg_val_loss_curved:.4f}, straight_keep={avg_val_loss_straight_keep:.4f}, seg={avg_val_loss_seg:.4f}), pos_mse={avg_val_pos_mse:.6f}, pred_mean=({avg_val_pred_mean[0]:.4f},{avg_val_pred_mean[1]:.4f}), pred_std=({avg_val_pred_std[0]:.4f},{avg_val_pred_std[1]:.4f}), ADE={avg_val_ade:.1f}m, FDE={avg_val_fde:.1f}m")

        goal_norm_denom = float(env_coverage_km * 0.5)
        
        ckpt = {
            'epoch': epoch,
            'phase': str(phase),
            'region': str(region),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_ade': float(best_val_ade),
            'val_ade': float(avg_val_ade),
            'val_fde': float(avg_val_fde),
            'standardize_history': bool(standardize_history),
            # 顶层关键配置（用于可视化脚本快速访问）
            'coord_scale': float(dataset.coord_scale),
            'env_coverage_km': float(env_coverage_km),
            'env_local_coverage_km': float(env_local_coverage_km),
            'goal_norm_denom': float(goal_norm_denom),
            'num_candidates': int(num_candidates),
            'candidate_radius_km': float(candidate_radius_km),
            'candidate_center': str(candidate_center),
            'phase3_missing_goal': bool(phase3_missing_goal),
            'goal_map_scale': float(goal_map_scale),
            'paper_mode': bool(paper_mode),
            'paper_decoder': str(paper_decoder),
            'use_dual_scale': bool(use_dual_scale),
            'waypoint_stride': int(getattr(getattr(model, 'decoder', None), 'waypoint_stride', int(waypoint_stride))),
            'traj_wp_align_weight': float(traj_wp_align_weight),
            'waypoint_tf_ratio': float(waypoint_tf_ratio),
            'history_feature_mode': str(history_feature_mode),
            'goal_mode': str(effective_goal_mode),
            'config': {
                'phase': str(phase),
                'region': str(region),
                'batch_size': int(batch_size),
                'num_workers': int(num_workers),
                'train_num_workers': int(train_num_workers),
                'lr': float(learning_rate),
                'sample_fraction': float(sample_fraction),
                'env_coverage_km': float(env_coverage_km),
                'env_local_coverage_km': float(env_local_coverage_km),
                'coord_scale': float(dataset.coord_scale),
                'goal_norm_denom': float(goal_norm_denom),
                'use_dual_scale': bool(use_dual_scale),
                'history_feature_mode': str(history_feature_mode),
                'standardize_history': bool(standardize_history),
                'hist_stats_samples': int(hist_stats_samples),
                'split_seed': int(seed),
                'goal_mode': str(effective_goal_mode),
                'grad_accum_steps': int(accum_steps),
                'amp': bool(amp_enabled),
                'cls_weight': float(cls_weight),
                'num_candidates': int(num_candidates),
                'candidate_radius_km': float(candidate_radius_km),
                'candidate_center': str(candidate_center),
                'phase3_missing_goal': bool(phase3_missing_goal),
                'goal_map_scale': float(goal_map_scale),
                'paper_mode': bool(paper_mode),
                'paper_decoder': str(paper_decoder),
                'waypoint_stride': int(getattr(getattr(model, 'decoder', None), 'waypoint_stride', int(waypoint_stride))),
                'closed_loop_env_sampling': bool(getattr(getattr(model, 'decoder', None), 'closed_loop_env_sampling', bool(closed_loop_env_sampling))),
                'wp_weight': float(wp_weight),
                'traj_wp_align_weight': float(traj_wp_align_weight),
                'var_weight': float(var_weight),
                'end_loss_weight': float(end_loss_weight),
                'waypoint_tf_ratio': float(waypoint_tf_ratio),
                'tf_start': float(initial_tf_ratio),
                'tf_end': float(final_tf_ratio),
                'tf_decay_epochs': int(tf_decay_epochs),
                'goal_vec_scale_start': goal_vec_scale_start,
                'goal_vec_scale_end': goal_vec_scale_end,
                'goal_vec_scale_anneal_epochs': int(goal_vec_scale_anneal_epochs),
                'env_local_scale2_start': env_local_scale2_start,
                'env_local_scale2_end': env_local_scale2_end,
                'env_local_scale2_anneal_epochs': int(env_local_scale2_anneal_epochs),
                'goal_vec_use_waypoint_after_epoch': int(goal_vec_use_waypoint_after_epoch),
                'force_goal_vec_use_waypoint': force_goal_vec_use_waypoint,
                'env_loss_weight': float(env_loss_weight),
                'env_imp_weight': float(env_imp_weight),
                'env_unknown_weight': float(env_unknown_weight),
                'env_slope_weight': float(env_slope_weight),
                'env_slope_thr_deg': float(env_slope_thr_deg),
                'env_sample_stride': int(env_sample_stride),
                'curved_norm_dev_weight': float(curved_norm_dev_weight),
                'curved_norm_dev_threshold': float(curved_norm_dev_threshold),
                'curved_norm_dev_ratio': float(curved_norm_dev_ratio),
                'curved_norm_dev_beta': float(curved_norm_dev_beta),
                'straight_keep_weight': float(straight_keep_weight),
                'aux_seg_weight': float(aux_seg_weight),
            },
        }
        if getattr(model, 'decoder', None) is not None and hasattr(model.decoder, 'goal_vec_use_waypoint'):
            ckpt['goal_vec_use_waypoint'] = bool(model.decoder.goal_vec_use_waypoint)
        if hist_mean is not None and hist_std is not None:
            ckpt['hist_mean'] = hist_mean.detach().cpu()
            ckpt['hist_std'] = hist_std.detach().cpu()
        torch.save(ckpt, save_dir / 'last_model.pth')

        if avg_val_ade < best_val_ade:
            best_val_ade = avg_val_ade
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停")
                break

        if avg_val_ade < best_val_ade_saved:
            best_val_ade_saved = avg_val_ade
            torch.save(ckpt, save_dir / 'best_model.pth')
            print(f"  ✓ 保存本次训练最佳模型 (ADE={avg_val_ade:.1f}m)")

    print(f"✓ {phase.upper()} 完成，本次训练最佳 ADE: {best_val_ade_saved:.1f}m")
    return best_val_ade

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, choices=['fas1', 'fas2', 'fas3'], help='Specific phase to train')
    parser.add_argument('--region', type=str, default='bohemian_forest', help='Region name')
    parser.add_argument('--traj_dir', type=str, help='Trajectory directory override')
    parser.add_argument('--fas_split_file', type=str, default=None)
    parser.add_argument('--save_root', type=str, default='/home/zmc/文档/programwork/runs')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--train_num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--sample_fraction', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_candidates', type=int, default=6)
    parser.add_argument('--env_coverage_km', type=float, default=140.0)
    parser.add_argument('--env_local_coverage_km', type=float, default=10.0)
    parser.add_argument('--history_feature_mode', type=str, default='full26', choices=['xy', 'kin10', 'full26'])
    parser.add_argument('--standardize_history', action='store_true')
    parser.add_argument('--hist_stats_samples', type=int, default=2000)
    parser.add_argument('--goal_mode', type=str, default='joint', choices=['given', 'joint'])
    parser.add_argument('--cls_weight', type=float, default=1.0)
    parser.add_argument('--candidate_radius_km', type=float, default=3.0)
    parser.add_argument('--candidate_center', type=str, default='goal', choices=['current', 'goal'])
    parser.add_argument('--phase3_missing_goal', action='store_true')
    parser.add_argument('--goal_map_scale', type=float, default=1.0, help='候选目标通道缩放系数（0=关闭）')
    parser.add_argument('--paper_mode', dest='paper_mode', action='store_true', help='启用 Paper Mode (简化架构)')
    parser.add_argument('--no_paper_mode', dest='paper_mode', action='store_false', help='关闭 Paper Mode (使用复杂解码器)')
    parser.set_defaults(paper_mode=True)
    parser.add_argument('--paper_decoder', type=str, default='hierarchical', choices=['hierarchical', 'flat'], help='Paper Mode decoder type')
    parser.add_argument('--waypoint_stride', type=int, default=0)
    parser.add_argument('--closed_loop_env_sampling', action='store_true', help='启用闭环环境采样：基于实际预测位置采样环境特征，而非waypoint线性插值')
    parser.add_argument('--wp_weight', type=float, default=1.0)
    parser.add_argument('--traj_wp_align_weight', type=float, default=1.0, help='Trajectory alignment loss at waypoint indices weight')
    parser.add_argument('--var_weight', type=float, default=1.0)
    parser.add_argument('--end_loss_weight', type=float, default=0.0, help='可选：终点约束损失权重（>0 启用），loss_end=||pred_end-goal||，单位 km')
    parser.add_argument('--waypoint_tf_ratio', type=float, default=-1.0, help='Waypoint teacher forcing ratio: <0 跟随 teacher_forcing_ratio；=0 始终用预测waypoints；>0 指定概率使用GT waypoints')
    parser.add_argument('--tf_start', type=float, default=0.5)
    parser.add_argument('--tf_end', type=float, default=0.05)
    parser.add_argument('--tf_decay_epochs', type=int, default=15)
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数（增大有效 batch）')
    parser.add_argument('--amp', action='store_true', help='启用混合精度训练 (AMP)')
    parser.add_argument('--filter_cfg', type=str, default=None, help='可选：只使用指定 intent/type 的轨迹文件，例如 intent2_type1')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='从已有 best_model.pth 继续训练')
    parser.add_argument('--use_dual_scale', action='store_true')
    parser.add_argument('--override_goal_vec_scale', type=float, default=None, help='可选：强制 decoder.goal_vec_scale（用于缓解直线化）')
    parser.add_argument('--freeze_goal_vec_scale', action='store_true', help='若设置 override_goal_vec_scale，则冻结该参数不参与训练')
    parser.add_argument('--override_env_local_scale', type=float, default=None, help='可选：强制 decoder.env_local_scale（用于强制环境注入）')
    parser.add_argument('--freeze_env_local_scale', action='store_true', help='若设置 override_env_local_scale，则冻结该参数不参与训练')
    parser.add_argument('--override_env_local_scale2', type=float, default=None, help='可选：强制 decoder.env_local_scale2（用于强制双尺度局部环境注入）')
    parser.add_argument('--freeze_env_local_scale2', action='store_true', help='若设置 override_env_local_scale2，则冻结该参数不参与训练')
    parser.add_argument('--goal_vec_scale_start', type=float, default=None, help='可选：goal_vec_scale 调度起始值')
    parser.add_argument('--goal_vec_scale_end', type=float, default=None, help='可选：goal_vec_scale 调度终止值')
    parser.add_argument('--goal_vec_scale_anneal_epochs', type=int, default=0, help='可选：goal_vec_scale 线性调度 epoch 数（>0启用）')
    parser.add_argument('--env_local_scale2_start', type=float, default=None, help='可选：env_local_scale2 调度起始值')
    parser.add_argument('--env_local_scale2_end', type=float, default=None, help='可选：env_local_scale2 调度终止值')
    parser.add_argument('--env_local_scale2_anneal_epochs', type=int, default=0, help='可选：env_local_scale2 线性调度 epoch 数（>0启用）')
    parser.add_argument('--goal_vec_use_waypoint_after_epoch', type=int, default=-1, help='可选：从第 N 个 epoch(相对resume起点, 0-based) 开始启用 goal_vec_use_waypoint')
    parser.add_argument('--force_goal_vec_use_waypoint', type=int, default=None, choices=[0, 1], help='可选：强制设置 goal_vec_use_waypoint（0=禁用, 1=启用），会覆盖 checkpoint 与 schedule')
    parser.add_argument('--env_loss_weight', type=float, default=0.0, help='环境一致性损失总权重（>0启用）')
    parser.add_argument('--env_imp_weight', type=float, default=1.0, help='环境一致性：不可通行(LULC湿地/水域/冰雪)惩罚权重（loss_env内部）')
    parser.add_argument('--env_unknown_weight', type=float, default=0.2, help='环境一致性：unknown LULC(255)惩罚权重（loss_env内部）')
    parser.add_argument('--env_slope_weight', type=float, default=0.5, help='环境一致性：坡度超阈值惩罚权重（loss_env内部）')
    parser.add_argument('--env_slope_thr_deg', type=float, default=30.0, help='环境一致性：坡度阈值（度）')
    parser.add_argument('--env_sample_stride', type=int, default=4, help='环境一致性：沿轨迹采样步长（越大越省算力）')
    parser.add_argument('--curved_norm_dev_weight', type=float, default=0.0, help='仅对GT弯样本的直线惩罚权重（>0启用）')
    parser.add_argument('--curved_norm_dev_threshold', type=float, default=0.08, help='判定GT为弯的阈值：max_perp_dev/line_len')
    parser.add_argument('--curved_norm_dev_ratio', type=float, default=1.0, help='目标下限：pred_norm_dev >= ratio * gt_norm_dev (仅对GT弯样本)')
    parser.add_argument('--curved_norm_dev_beta', type=float, default=10.0, help='softmax近似max的温度参数beta（>0启用logsumexp平滑）')
    parser.add_argument('--straight_keep_weight', type=float, default=0.0)
    parser.add_argument('--aux_seg_weight', type=float, default=0.0, help='Auxiliary segmentation loss weight')
    parser.add_argument('--use_heading', action='store_true', help='启用显式转向角建模')
    parser.add_argument('--heading_loss_weight', type=float, default=1.0, help='转向角辅助loss权重')
    args = parser.parse_args()

    _print_config(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        sample_fraction=args.sample_fraction,
        lr=args.lr,
        patience=args.patience,
        env_coverage_km=args.env_coverage_km,
        goal_map_scale=args.goal_map_scale,
        grad_accum_steps=args.grad_accum_steps,
        amp=args.amp,
    )
    
    phases = [args.phase] if args.phase else ['fas1', 'fas2', 'fas3']
    
    for phase in phases:
        try:
            train_phase(
                phase=phase,
                region=args.region,
                traj_dir=args.traj_dir,
                fas_split_file=args.fas_split_file,
                save_root=args.save_root,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_num_workers=args.train_num_workers,
                learning_rate=args.lr,
                sample_fraction=args.sample_fraction,
                seed=args.seed,
                patience=args.patience,
                num_candidates=args.num_candidates,
                env_coverage_km=args.env_coverage_km,
                env_local_coverage_km=args.env_local_coverage_km,
                history_feature_mode=str(args.history_feature_mode),
                standardize_history=bool(args.standardize_history),
                hist_stats_samples=int(args.hist_stats_samples),
                goal_mode=str(args.goal_mode),
                cls_weight=args.cls_weight,
                candidate_radius_km=args.candidate_radius_km,
                candidate_center=args.candidate_center,
                phase3_missing_goal=args.phase3_missing_goal,
                goal_map_scale=float(args.goal_map_scale),
                paper_mode=args.paper_mode,
                paper_decoder=args.paper_decoder,
                waypoint_stride=args.waypoint_stride,
                closed_loop_env_sampling=bool(args.closed_loop_env_sampling),
                wp_weight=args.wp_weight,
                traj_wp_align_weight=float(args.traj_wp_align_weight),
                var_weight=args.var_weight,
                end_loss_weight=float(args.end_loss_weight),
                waypoint_tf_ratio=float(args.waypoint_tf_ratio),
                tf_start=float(args.tf_start),
                tf_end=float(args.tf_end),
                tf_decay_epochs=int(args.tf_decay_epochs),
                grad_accum_steps=int(args.grad_accum_steps),
                amp=bool(args.amp),
                filter_cfg=args.filter_cfg,
                resume_checkpoint=args.resume_checkpoint,
                use_dual_scale=bool(args.use_dual_scale),
                override_goal_vec_scale=args.override_goal_vec_scale,
                freeze_goal_vec_scale=bool(args.freeze_goal_vec_scale),
                override_env_local_scale=args.override_env_local_scale,
                freeze_env_local_scale=bool(args.freeze_env_local_scale),
                override_env_local_scale2=args.override_env_local_scale2,
                freeze_env_local_scale2=bool(args.freeze_env_local_scale2),
                goal_vec_scale_start=args.goal_vec_scale_start,
                goal_vec_scale_end=args.goal_vec_scale_end,
                goal_vec_scale_anneal_epochs=int(args.goal_vec_scale_anneal_epochs),
                env_local_scale2_start=args.env_local_scale2_start,
                env_local_scale2_end=args.env_local_scale2_end,
                env_local_scale2_anneal_epochs=int(args.env_local_scale2_anneal_epochs),
                goal_vec_use_waypoint_after_epoch=int(args.goal_vec_use_waypoint_after_epoch),
                force_goal_vec_use_waypoint=args.force_goal_vec_use_waypoint,
                env_loss_weight=float(args.env_loss_weight),
                env_imp_weight=float(args.env_imp_weight),
                env_unknown_weight=float(args.env_unknown_weight),
                env_slope_weight=float(args.env_slope_weight),
                env_slope_thr_deg=float(args.env_slope_thr_deg),
                env_sample_stride=int(args.env_sample_stride),
                curved_norm_dev_weight=args.curved_norm_dev_weight,
                curved_norm_dev_threshold=args.curved_norm_dev_threshold,
                curved_norm_dev_ratio=args.curved_norm_dev_ratio,
                curved_norm_dev_beta=args.curved_norm_dev_beta,
                straight_keep_weight=args.straight_keep_weight,
                aux_seg_weight=args.aux_seg_weight,
                use_heading=bool(args.use_heading),
                heading_loss_weight=float(args.heading_loss_weight),
            )
        except Exception as e:
            print(f"Error training {phase}: {e}")
            import traceback
            traceback.print_exc()
