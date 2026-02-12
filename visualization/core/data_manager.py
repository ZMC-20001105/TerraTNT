#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理器 - 统一加载环境栅格、数据集样本、模型预测
"""
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class RegionData:
    """单个区域的环境栅格数据"""

    def __init__(self, region: str):
        self.region = region
        self.utm_dir = PROJECT_ROOT / 'data' / 'processed' / 'utm_grid' / region
        self.dem = None
        self.slope = None
        self.aspect = None
        self.lulc = None
        self.road = None
        self.transform = None
        self.crs = None
        self.bounds = None
        self.shape = None

    def load(self):
        """加载所有环境栅格 (多线程并行)"""
        import rasterio
        from concurrent.futures import ThreadPoolExecutor

        def _read_tif(name, dtype=np.float32):
            fpath = self.utm_dir / name
            if not fpath.exists():
                return name, None, None
            with rasterio.open(fpath) as src:
                arr = src.read(1).astype(dtype)
                meta = {'transform': src.transform, 'crs': src.crs,
                        'bounds': src.bounds, 'shape': (src.height, src.width)}
            return name, arr, meta

        tasks = [
            ('dem_utm.tif', np.float32),
            ('slope_utm.tif', np.float32),
            ('aspect_utm.tif', np.float32),
            ('lulc_utm.tif', np.uint8),
            ('road_utm.tif', np.float32),
        ]
        results = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(_read_tif, n, d): n for n, d in tasks}
            for fut in futs:
                name, arr, meta = fut.result()
                results[name] = (arr, meta)

        # 赋值
        dem_arr, dem_meta = results['dem_utm.tif']
        if dem_arr is not None:
            self.dem = dem_arr
            self.transform = dem_meta['transform']
            self.crs = dem_meta['crs']
            self.bounds = dem_meta['bounds']
            self.shape = dem_meta['shape']

        slope_arr, _ = results['slope_utm.tif']
        if slope_arr is not None:
            self.slope = slope_arr

        aspect_arr, _ = results['aspect_utm.tif']
        if aspect_arr is not None:
            self.aspect = aspect_arr

        lulc_arr, _ = results['lulc_utm.tif']
        if lulc_arr is not None:
            self.lulc = lulc_arr

        road_arr, _ = results['road_utm.tif']
        self.road = road_arr if road_arr is not None else np.zeros(self.shape, dtype=np.float32)

        # 处理DEM无效值
        if self.dem is not None:
            self.dem = np.where(self.dem == -32768, np.nan, self.dem)

    def utm_to_pixel(self, easting: float, northing: float) -> Tuple[int, int]:
        """UTM坐标转像素坐标 (row, col)"""
        col, row = ~self.transform * (easting, northing)
        return int(row), int(col)

    def pixel_to_utm(self, row: int, col: int) -> Tuple[float, float]:
        """像素坐标转UTM坐标"""
        easting, northing = self.transform * (col, row)
        return float(easting), float(northing)

    def extract_patch(self, center_utm: Tuple[float, float],
                      coverage_km: float = 140.0,
                      out_size: int = 512) -> Dict[str, np.ndarray]:
        """提取以UTM坐标为中心的环境patch"""
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling

        cx, cy = center_utm
        half_m = coverage_km * 1000.0 * 0.5
        left, right = cx - half_m, cx + half_m
        bottom, top = cy - half_m, cy + half_m

        patches = {}
        layer_files = {
            'dem': ('dem_utm.tif', False),
            'slope': ('slope_utm.tif', False),
            'lulc': ('lulc_utm.tif', True),
            'road': ('road_utm.tif', True),
            'road_graded': ('road_graded_utm.tif', True),
        }

        for name, (fname, is_categorical) in layer_files.items():
            fpath = self.utm_dir / fname
            if not fpath.exists():
                if name == 'road_graded':
                    continue  # 可选图层, 不存在则跳过
                patches[name] = np.zeros((out_size, out_size), dtype=np.float32)
                continue
            with rasterio.open(fpath) as src:
                nodata = src.nodata
                window = from_bounds(left, bottom, right, top, transform=src.transform)
                resamp = Resampling.nearest if is_categorical else Resampling.bilinear
                arr = src.read(1, window=window, out_shape=(out_size, out_size),
                               resampling=resamp, boundless=True, fill_value=0)
            arr = arr.astype(np.float32)
            # DEM: 将nodata值(-32768等)及插值伪影转为NaN
            # 双线性插值可能在nodata边界产生中间值(如-9972), 用-500m阈值过滤
            if name == 'dem':
                arr = np.where(arr < -500, np.nan, arr)
            patches[name] = arr

        return patches

    def hillshade(self, dem_patch: np.ndarray,
                  azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
        """计算山体阴影"""
        dem = np.nan_to_num(dem_patch, nan=float(np.nanmean(dem_patch)))
        gy, gx = np.gradient(dem)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gx, gy)
        az = np.deg2rad(azimuth)
        alt = np.deg2rad(altitude)
        shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        return np.clip((shaded + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


class SampleData:
    """单个样本的数据"""

    def __init__(self, sample_id: str, sample_dict: dict, traj_meta: dict = None):
        self.sample_id = sample_id

        # 历史特征 (90, 26) — 前2列是相对位置 (km), 原点=观测点(0,0)
        hist_feat = np.array(sample_dict.get('history_feat_26d', []), dtype=np.float32)
        if hist_feat.ndim == 2 and hist_feat.shape[1] >= 2:
            # 前2列已经是相对坐标 (km), 最后一点=(0,0)
            self.history_rel = hist_feat[:, :2].copy()
            self.history_feat = hist_feat
        else:
            self.history_rel = np.zeros((0, 2), dtype=np.float32)
            self.history_feat = hist_feat

        # 未来轨迹 (360, 2) 累积相对坐标 km
        self.future_rel = np.array(sample_dict.get('future_rel', []), dtype=np.float32)

        # 目标点 (相对坐标, km)
        goal_rel = sample_dict.get('goal_rel', None)
        if goal_rel is not None:
            self.goal_rel = np.array(goal_rel, dtype=np.float32)
        else:
            self.goal_rel = np.zeros(2, dtype=np.float32)

        # 候选终点 — 数据集中无此字段，用goal_rel代替
        self.candidates_rel = self.goal_rel.reshape(1, 2) if self.goal_rel.size == 2 else np.zeros((0, 2), dtype=np.float32)

        # 最后观测点UTM坐标
        pos_abs = sample_dict.get('current_pos_abs', None)
        if pos_abs is not None:
            self.last_obs_utm = (float(pos_abs[0]), float(pos_abs[1]))
        else:
            self.last_obs_utm = (0.0, 0.0)

        # 环境地图 (18, 128, 128)
        env = sample_dict.get('env_map_100km', sample_dict.get('env_map', None))
        if env is not None:
            self.env_map = np.array(env, dtype=np.float32)
        else:
            self.env_map = np.zeros((0,), dtype=np.float32)

        # 如果 ch17 (Goal/Prior) 全空, 根据 goal_rel 写入高斯标记
        if (self.env_map.ndim == 3 and self.env_map.shape[0] >= 18
                and self.env_map[17].max() < 1e-6 and np.linalg.norm(self.goal_rel) > 0.1):
            # goal_rel (km) → 像素坐标 (128x128, 覆盖100km)
            gx_pix = int((self.goal_rel[0] / 100.0 + 0.5) * 128)
            gy_pix = int((0.5 - self.goal_rel[1] / 100.0) * 128)
            if 0 <= gx_pix < 128 and 0 <= gy_pix < 128:
                yy, xx = np.mgrid[0:128, 0:128]
                sigma = 3.0  # 像素
                gauss = np.exp(-((xx - gx_pix)**2 + (yy - gy_pix)**2) / (2 * sigma**2))
                self.env_map[17] = gauss.astype(np.float32)

        # 元数据 (来自轨迹级别)
        if traj_meta:
            self.intent = traj_meta.get('intent', 'unknown')
            self.vehicle_type = traj_meta.get('vehicle_type', 'unknown')
            self.traj_id = traj_meta.get('traj_id', sample_id)
            self.goal_utm = traj_meta.get('goal_utm', None)
        else:
            self.intent = sample_dict.get('intent', 'unknown')
            self.vehicle_type = sample_dict.get('vehicle_type', 'unknown')
            self.traj_id = sample_dict.get('traj_id', sample_id)
            self.goal_utm = None

    @property
    def sinuosity(self) -> float:
        """计算未来轨迹曲折度"""
        if len(self.future_rel) < 2:
            return 1.0
        deltas = np.diff(self.future_rel, axis=0)
        path_len = np.sum(np.linalg.norm(deltas, axis=1))
        disp = np.linalg.norm(self.future_rel[-1])
        return float(path_len / max(disp, 1e-6))

    @property
    def is_axis_aligned(self) -> bool:
        """检测是否为直角贴边脏数据 (轨迹几乎只沿X或Y轴移动)"""
        fut = self.future_rel
        if fut is None or len(fut) < 10:
            return False
        diffs = np.diff(fut, axis=0)
        dx = np.abs(diffs[:, 0])
        dy = np.abs(diffs[:, 1])
        total = dx + dy + 1e-9
        axis_ratio = np.maximum(dx / total, dy / total)
        return float(np.mean(axis_ratio > 0.95)) > 0.5

    @property
    def total_distance_km(self) -> float:
        """总位移距离(km)"""
        if len(self.future_rel) < 1:
            return 0.0
        return float(np.linalg.norm(self.future_rel[-1]))

    def future_abs_utm(self) -> np.ndarray:
        """未来轨迹的绝对UTM坐标"""
        ox, oy = self.last_obs_utm
        return self.future_rel * 1000.0 + np.array([[ox, oy]])

    def history_abs_utm(self) -> np.ndarray:
        """历史轨迹的绝对UTM坐标"""
        ox, oy = self.last_obs_utm
        return self.history_rel * 1000.0 + np.array([[ox, oy]])

    def candidates_abs_utm(self) -> np.ndarray:
        """候选终点的绝对UTM坐标"""
        ox, oy = self.last_obs_utm
        return self.candidates_rel * 1000.0 + np.array([[ox, oy]])


class ModelPrediction:
    """单个模型对单个样本的预测"""

    def __init__(self, model_name: str, pred_rel: np.ndarray,
                 goal_logits: Optional[np.ndarray] = None,
                 alpha: Optional[float] = None):
        self.model_name = model_name
        self.pred_rel = pred_rel  # (T, 2) 相对坐标 km
        self.goal_logits = goal_logits
        self.alpha = alpha  # V7 gate value

    @property
    def selected_goal_idx(self) -> int:
        if self.goal_logits is not None:
            return int(np.argmax(self.goal_logits))
        return -1


def _read_one_pkl(pkl_path):
    """读取单个pkl文件的原始数据 (仅I/O, 线程安全)"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return (pkl_path, data)
    except Exception:
        return None


class DataManager:
    """统一数据管理器"""

    # 模型颜色方案
    MODEL_COLORS = {
        'TerraTNT':     '#1f77b4',  # 蓝
        'V3_Waypoint':  '#ff7f0e',  # 橙
        'V4_WP_Spatial':'#2ca02c',  # 绿
        'V6_GoalRefine':'#d62728',  # 红
        'V6R_Region':   '#9467bd',  # 紫
        'V7_ConfGate':  '#8c564b',  # 棕
        'LSTM_only':    '#e377c2',  # 粉
        'LSTM_Env_Goal':'#7f7f7f',  # 灰
        'Seq2Seq':      '#bcbd22',  # 黄绿
        'MLP':          '#17becf',  # 青
        'CV':           '#aec7e8',  # 浅蓝
    }

    PHASE_CONFIGS = {
        '1a': {'name': 'Phase 1a: 精确终点(域内)', 'sigma_km': 1.0, 'offset_km': 0.0},
        '1b': {'name': 'Phase 1b: 精确终点(OOD)', 'sigma_km': 1.0, 'offset_km': 0.0},
        '2a': {'name': 'Phase 2a: 区域先验(σ=10km)', 'sigma_km': 10.0, 'offset_km': 0.0},
        '2b': {'name': 'Phase 2b: 区域先验(σ=15km)', 'sigma_km': 15.0, 'offset_km': 0.0},
        '2c': {'name': 'Phase 2c: 区域先验(偏移5km)', 'sigma_km': 10.0, 'offset_km': 5.0},
        '3a': {'name': 'Phase 3a: 无先验(直行)', 'sigma_km': None},
        '3b': {'name': 'Phase 3b: 无先验(转弯)', 'sigma_km': None},
    }

    def __init__(self):
        self.regions: Dict[str, RegionData] = {}
        self.current_region: Optional[str] = None
        self.samples: List[SampleData] = []
        self.predictions: Dict[str, Dict[str, ModelPrediction]] = {}  # {sample_id: {model: pred}}

    def available_regions(self) -> List[str]:
        """列出可用区域"""
        utm_dir = PROJECT_ROOT / 'data' / 'processed' / 'utm_grid'
        regions = []
        for d in sorted(utm_dir.iterdir()):
            if d.is_dir() and (d / 'dem_utm.tif').exists():
                regions.append(d.name)
        return regions

    def load_region(self, region: str) -> RegionData:
        """加载区域环境数据"""
        if region not in self.regions:
            rd = RegionData(region)
            rd.load()
            self.regions[region] = rd
        self.current_region = region
        return self.regions[region]

    def get_region(self) -> Optional[RegionData]:
        """获取当前区域数据"""
        if self.current_region:
            return self.regions.get(self.current_region)
        return None

    def load_dataset(self, region: str, dataset_dir: Optional[str] = None,
                     max_samples: int = 5000) -> List[SampleData]:
        """加载数据集样本 (多核并行)"""
        if dataset_dir:
            data_dir = Path(dataset_dir)
        else:
            candidates = [
                PROJECT_ROOT / 'data' / 'processed' / 'final_dataset_v1' / region,
                PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / region,
            ]
            data_dir = None
            for c in candidates:
                if c.exists() and list(c.glob('*.pkl')):
                    data_dir = c
                    break

        if data_dir is None or not data_dir.exists():
            print(f"未找到 {region} 的数据集")
            return []

        pkl_files = sorted(data_dir.glob('*.pkl'))

        # 均匀采样: 先打乱文件顺序, 确保不同intent/vehicle_type均匀覆盖
        import random
        rng = random.Random(42)
        shuffled_files = list(pkl_files)
        rng.shuffle(shuffled_files)

        # 限制文件数量 (每个pkl约30-50样本, 估算需要多少文件)
        avg_samples_per_file = 30
        max_files = min(len(shuffled_files), max(max_samples // avg_samples_per_file + 10, 50))
        files_to_load = shuffled_files[:max_files]

        # 阶段1: 多线程并行读取pkl文件 (纯I/O, 线程安全)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os, time
        n_workers = min(max(os.cpu_count() or 4, 2), 8)
        print(f"并行加载 {len(files_to_load)} 个pkl文件 ({n_workers} threads)...")

        t0 = time.time()
        raw_data = []  # [(pkl_path, data), ...]
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_read_one_pkl, f) for f in files_to_load]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    raw_data.append(result)
        t_io = time.time() - t0

        # 阶段2: 主线程创建SampleData对象 (CPU密集, 避免GIL竞争)
        self.samples = []
        for pkl_path, data in raw_data:
            if 'samples' in data and isinstance(data['samples'], list):
                traj_meta = {
                    'intent': data.get('intent', 'unknown'),
                    'vehicle_type': data.get('vehicle_type', 'unknown'),
                    'traj_id': pkl_path.stem,
                    'goal_utm': data.get('goal_utm', None),
                }
                for i, sample in enumerate(data['samples']):
                    sid = f"{pkl_path.stem}_s{i}"
                    self.samples.append(SampleData(sid, sample, traj_meta))
            else:
                self.samples.append(SampleData(pkl_path.stem, data))
            if len(self.samples) >= max_samples * 2:
                break
        # 过滤脏数据: 直角贴边轨迹
        n_before = len(self.samples)
        self.samples = [s for s in self.samples if not s.is_axis_aligned]
        n_dirty = n_before - len(self.samples)
        if len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        elapsed = time.time() - t0
        dirty_msg = f", 过滤{n_dirty}条脏数据" if n_dirty > 0 else ""
        print(f"加载 {len(self.samples)} 个样本 from {data_dir} (I/O {t_io:.1f}s, 总计 {elapsed:.1f}s{dirty_msg})")
        return self.samples

    def load_evaluation_results(self, phase: str, results_dir: Optional[str] = None):
        """加载Phase评估结果"""
        if results_dir:
            rdir = Path(results_dir)
        else:
            rdir = PROJECT_ROOT / 'outputs' / 'evaluation' / 'phase_v2'

        results_file = rdir / f'phase_{phase}_results.json'
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        return None

    def compute_ade(self, pred_rel: np.ndarray, gt_rel: np.ndarray) -> float:
        """计算ADE (km → m)"""
        if len(pred_rel) == 0 or len(gt_rel) == 0:
            return float('inf')
        min_len = min(len(pred_rel), len(gt_rel))
        diff = pred_rel[:min_len] - gt_rel[:min_len]
        return float(np.mean(np.linalg.norm(diff, axis=1)) * 1000.0)

    def compute_fde(self, pred_rel: np.ndarray, gt_rel: np.ndarray) -> float:
        """计算FDE (km → m)"""
        if len(pred_rel) == 0 or len(gt_rel) == 0:
            return float('inf')
        return float(np.linalg.norm(pred_rel[-1] - gt_rel[-1]) * 1000.0)
