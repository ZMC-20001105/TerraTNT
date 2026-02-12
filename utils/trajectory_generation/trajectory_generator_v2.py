"""
轨迹生成器 V2 (完全重新设计)

新特性：
1. 轨迹长度 ≥ 60km
2. 速度使用 XGBoost + 线性变换（校准到目标平均速度）
3. 在生成时提取所有特征：
   - 26维历史特征（每个时间步）
   - 140km×140km环境地图（以最后观测点为中心）
4. 10秒采样间隔
5. 避免批量重复（每次生成使用唯一随机种子）
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pickle
from time import perf_counter
import torch

from .hierarchical_astar import HierarchicalAStarPlanner
from .path_smoothing import smooth_path
from .sampling import sample_start_goal_v2
from config import cfg, get_path

logger = logging.getLogger(__name__)

# 论文表3.2：车辆运动学参数
VEHICLE_PARAMS = {
    'type1': {'v_max': 18.0, 'a_max': 2.0, 'R_min': 8.0, 'target_speed': 16.7},
    'type2': {'v_max': 22.0, 'a_max': 2.5, 'R_min': 6.0, 'target_speed': 18.0},
    'type3': {'v_max': 25.0, 'a_max': 3.0, 'R_min': 5.0, 'target_speed': 19.5},
    'type4': {'v_max': 28.0, 'a_max': 3.5, 'R_min': 10.0, 'target_speed': 21.0}
}


def _path_length_m(path_xy_m: np.ndarray) -> float:
    diffs = np.diff(path_xy_m, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _path_disp_m(path_xy_m: np.ndarray) -> float:
    return float(np.linalg.norm(path_xy_m[-1] - path_xy_m[0]))


def _path_sinuosity(path_xy_m: np.ndarray) -> float:
    return float(_path_length_m(path_xy_m) / (_path_disp_m(path_xy_m) + 1e-9))


def _curvature_stats_heading(path_xy_m: np.ndarray) -> dict:
    if path_xy_m.shape[0] < 3:
        return {
            'max_kappa_1pm': 0.0,
            'mean_kappa_1pm': 0.0,
            'p95_kappa_1pm': 0.0,
            'frac_kappa_gt_0p01': 0.0,
            'frac_kappa_gt_0p02': 0.0,
            'total_turn_deg': 0.0,
        }

    v = np.diff(path_xy_m, axis=0)
    theta = np.arctan2(v[:, 1], v[:, 0])
    dtheta = np.diff(theta)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    ds = np.linalg.norm(v[:-1], axis=1)
    ds = np.where(ds < 1e-3, 1e-3, ds)
    kappa = np.abs(dtheta / ds)

    return {
        'max_kappa_1pm': float(np.max(kappa)) if kappa.size else 0.0,
        'mean_kappa_1pm': float(np.mean(kappa)) if kappa.size else 0.0,
        'p95_kappa_1pm': float(np.quantile(kappa, 0.95)) if kappa.size else 0.0,
        'frac_kappa_gt_0p01': float(np.mean(kappa > 0.01)) if kappa.size else 0.0,
        'frac_kappa_gt_0p02': float(np.mean(kappa > 0.02)) if kappa.size else 0.0,
        'total_turn_deg': float(np.sum(np.abs(dtheta)) * (180.0 / np.pi)) if dtheta.size else 0.0,
    }


def _is_too_straight(path_xy_m: List[Tuple[float, float]], intent: str) -> Tuple[bool, Dict]:
    stats = _path_stats(path_xy_m)

    qcfg = cfg.get('trajectory_generation.quality_filter', {})
    sinuosity_min_by_intent = qcfg.get('sinuosity_min_by_intent', {})
    turn_min_deg_by_intent = qcfg.get('total_turn_min_deg_by_intent', {})

    sinuosity_min = float(sinuosity_min_by_intent.get(intent, qcfg.get('sinuosity_min', 1.0)))
    turn_min_deg = float(turn_min_deg_by_intent.get(intent, qcfg.get('total_turn_min_deg', 0.0)))

    sinuosity = float(stats.get('sinuosity', 0.0))
    total_turn_deg = float(stats.get('curvature', {}).get('total_turn_deg', 0.0))

    too_straight = (sinuosity < sinuosity_min) and (total_turn_deg < turn_min_deg)
    return too_straight, {
        'sinuosity': sinuosity,
        'sinuosity_min': sinuosity_min,
        'total_turn_deg': total_turn_deg,
        'total_turn_min_deg': turn_min_deg,
    }


def _path_stats(path: List[Tuple[float, float]]) -> dict:
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
        return {
            'num_points': int(arr.shape[0]) if arr.ndim == 2 else 0,
            'length_km': 0.0,
            'disp_km': 0.0,
            'sinuosity': 0.0,
            'curvature': _curvature_stats_heading(np.zeros((0, 2), dtype=np.float64)),
        }

    length_m = _path_length_m(arr)
    disp_m = _path_disp_m(arr)
    return {
        'num_points': int(arr.shape[0]),
        'length_km': float(length_m / 1000.0),
        'disp_km': float(disp_m / 1000.0),
        'sinuosity': float(length_m / (disp_m + 1e-9)),
        'curvature': _curvature_stats_heading(arr),
    }


class TrajectoryGeneratorV2:
    """轨迹生成器 V2"""
    
    def __init__(self, region: str = 'bohemian_forest'):
        self.region = region
        self.last_failure = None
        
        # 加载速度预测模型
        model_path = Path(get_path('paths.models.speed_predictor')) / 'speed_model.pkl'
        logger.info(f"加载速度预测模型: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.speed_model = model_data['model']
            self.feature_names = model_data['feature_names']
        
        # 加载环境栅格
        utm_dir = Path(get_path('paths.processed.utm_grid')) / region
        
        import rasterio
        self.dem_src = rasterio.open(utm_dir / 'dem_utm.tif')
        self.slope_src = rasterio.open(utm_dir / 'slope_utm.tif')
        self.aspect_src = rasterio.open(utm_dir / 'aspect_utm.tif')
        self.lulc_src = rasterio.open(utm_dir / 'lulc_utm.tif')
        
        road_path = utm_dir / 'road_utm.tif'
        if road_path.exists():
            self.road_src = rasterio.open(road_path)
            self.road = self.road_src.read(1)
        else:
            self.road_src = None
            self.road = None
        
        self.dem = self.dem_src.read(1)
        self.slope = self.slope_src.read(1)
        self.aspect = self.aspect_src.read(1)
        self.lulc = self.lulc_src.read(1)

        shapes = [self.dem.shape, self.slope.shape, self.aspect.shape, self.lulc.shape]
        if self.road is not None:
            shapes.append(self.road.shape)
        if len({s for s in shapes}) != 1:
            h = min(s[0] for s in shapes)
            w = min(s[1] for s in shapes)
            self.dem = self.dem[:h, :w]
            self.slope = self.slope[:h, :w]
            self.aspect = self.aspect[:h, :w]
            self.lulc = self.lulc[:h, :w]
            if self.road is not None:
                self.road = self.road[:h, :w]
        
        self.dem = np.where(self.dem == -32768, np.nan, self.dem)
        self.transform = self.dem_src.transform
        
        logger.info("✓ 速度预测模型和环境栅格已加载")
    
    def extract_features_for_path(self, path: List[Tuple[float, float]]) -> np.ndarray:
        """提取路径的20维特征（用于速度预测）"""
        n = len(path)
        features = np.zeros((n, 20))
        
        x = np.array([p[0] for p in path])
        y = np.array([p[1] for p in path])
        
        # 提取环境特征
        for i, (px, py) in enumerate(path):
            col, row = ~self.transform * (px, py)
            col, row = int(col), int(row)
            
            if 0 <= row < self.dem.shape[0] and 0 <= col < self.dem.shape[1]:
                features[i, 0] = self.dem[row, col] if not np.isnan(self.dem[row, col]) else 0
                features[i, 1] = self.slope[row, col]
                features[i, 2] = np.sin(np.deg2rad(self.aspect[row, col]))
                features[i, 3] = np.cos(np.deg2rad(self.aspect[row, col]))
                
                lulc_val = self.lulc[row, col]
                lulc_classes = [10, 20, 30, 40, 50, 60, 80, 90, 100, 255]
                for j, lulc_class in enumerate(lulc_classes):
                    features[i, 4+j] = 1.0 if lulc_val == lulc_class else 0.0
                
                features[i, 14] = 1.0 if lulc_val == 10 else 0.0
                features[i, 15] = self.slope[row, col] * np.cos(np.deg2rad(self.aspect[row, col]))
        
        # 计算曲率
        curvature = self.compute_curvature(path)
        features[:, 16] = curvature
        
        # 过去10m平均曲率
        for i in range(n):
            j = i - 1
            dist_back = 0
            curv_sum = 0
            count = 0
            while j >= 0 and dist_back < 10:
                dist_back += np.sqrt((x[j+1]-x[j])**2 + (y[j+1]-y[j])**2)
                curv_sum += curvature[j]
                count += 1
                j -= 1
            features[i, 17] = curv_sum / count if count > 0 else 0
        
        # 未来10m最大曲率
        for i in range(n):
            j = i + 1
            dist_fwd = 0
            curv_max = 0
            while j < n and dist_fwd < 10:
                dist_fwd += np.sqrt((x[j]-x[j-1])**2 + (y[j]-y[j-1])**2)
                curv_max = max(curv_max, curvature[j])
                j += 1
            features[i, 18] = curv_max
        
        # On road
        for i, (px, py) in enumerate(path):
            col, row = ~self.transform * (px, py)
            col, row = int(col), int(row)
            if 0 <= row < self.lulc.shape[0] and 0 <= col < self.lulc.shape[1]:
                features[i, 19] = 1.0 if self.lulc[row, col] == 80 else 0.0
        
        return features
    
    def compute_curvature(self, path: List[Tuple[float, float]]) -> np.ndarray:
        """计算路径曲率"""
        n = len(path)
        if n < 3:
            return np.zeros(n)
        
        x = np.array([p[0] for p in path])
        y = np.array([p[1] for p in path])
        
        dx = np.diff(x)
        dy = np.diff(y)
        theta = np.arctan2(dy, dx)
        
        dtheta = np.diff(theta)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        ds = np.sqrt(dx[:-1]**2 + dy[:-1]**2)
        ds = np.where(ds < 0.1, 0.1, ds)
        
        curvature = np.abs(dtheta / ds)
        
        result = np.zeros(n)
        result[1:-1] = curvature
        result[0] = curvature[0] if len(curvature) > 0 else 0
        result[-1] = curvature[-1] if len(curvature) > 0 else 0
        
        return result
    
    def predict_speeds_with_calibration(
        self,
        path: List[Tuple[float, float]],
        vehicle_type: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        使用XGBoost预测速度并进行线性变换校准
        
        Returns:
            (speeds, calibration_info)
        """
        features = self.extract_features_for_path(path)
        features = np.nan_to_num(features, nan=0.0)
        
        # XGBoost预测 log(1+v)
        y_pred = self.speed_model.predict(features)
        speeds_xgb = np.exp(y_pred) - 1
        
        # 应用车辆最大速度限制
        params = VEHICLE_PARAMS[vehicle_type]
        v_max = params['v_max']
        speeds_xgb = np.clip(speeds_xgb, 0.5, v_max)
        
        # 线性变换校准到目标平均速度
        target_speed = params['target_speed']
        xgb_mean = np.mean(speeds_xgb)
        
        # v_calibrated = a × v_xgboost + b
        # 约束：mean(v_calibrated) = target_speed
        # 简化：a = target_speed / xgb_mean, b = 0
        a = target_speed / xgb_mean if xgb_mean > 0 else 1.0
        b = 0.0
        
        speeds_calibrated = a * speeds_xgb + b
        speeds_calibrated = np.clip(speeds_calibrated, 0.5, v_max)
        
        # 轻微平滑
        from scipy.ndimage import gaussian_filter1d
        speeds_calibrated = gaussian_filter1d(speeds_calibrated, sigma=1.0)
        
        calibration_info = {
            'method': 'linear_transform',
            'formula': 'v_calibrated = a × v_xgboost + b',
            'params': {'a': float(a), 'b': float(b)},
            'xgboost_mean': float(xgb_mean),
            'calibrated_mean': float(np.mean(speeds_calibrated)),
            'target_speed': float(target_speed)
        }
        
        return speeds_calibrated, calibration_info
    
    def apply_kinematic_constraints(
        self,
        speeds: np.ndarray,
        distances: np.ndarray,
        vehicle_type: str
    ) -> np.ndarray:
        """应用运动学约束"""
        params = VEHICLE_PARAMS[vehicle_type]
        a_max = params['a_max']
        
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
    
    def compute_timestamps(
        self,
        path: List[Tuple[float, float]],
        speeds: np.ndarray
    ) -> np.ndarray:
        """计算时间戳"""
        n = len(path)
        timestamps = np.zeros(n)
        
        for i in range(1, n):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            ds = np.sqrt(dx**2 + dy**2)
            
            v_avg = (speeds[i-1] + speeds[i]) / 2.0
            v_avg = max(v_avg, 0.1)
            
            dt = ds / v_avg
            timestamps[i] = timestamps[i-1] + dt
        
        return timestamps
    
    def resample_to_10s(
        self,
        path: List[Tuple[float, float]],
        speeds: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """重采样到10秒间隔"""
        duration = timestamps[-1]
        num_samples = int(duration / 10.0) + 1
        
        target_times = np.arange(num_samples) * 10.0
        target_times = target_times[target_times <= duration]
        
        # 插值
        path_array = np.array(path)
        x_interp = np.interp(target_times, timestamps, path_array[:, 0])
        y_interp = np.interp(target_times, timestamps, path_array[:, 1])
        speeds_interp = np.interp(target_times, timestamps, speeds)
        
        resampled_path = np.column_stack([x_interp, y_interp])
        
        return resampled_path, speeds_interp, target_times
    
    def extract_26d_history_features(
        self,
        history_abs: np.ndarray
    ) -> np.ndarray:
        """
        提取26维历史特征（对齐论文表4.2）
        
        Args:
            history_abs: (T, 2) 绝对坐标
        
        Returns:
            (T, 26) 特征矩阵
        """
        T = len(history_abs)
        features_26 = np.zeros((T, 26))
        
        # 提取16维环境特征
        for i, (px, py) in enumerate(history_abs):
            col, row = ~self.transform * (px, py)
            col, row = int(col), int(row)
            
            if 0 <= row < self.dem.shape[0] and 0 <= col < self.dem.shape[1]:
                # DEM, slope, aspect
                dem_val = self.dem[row, col] if not np.isnan(self.dem[row, col]) else 0
                features_26[i, 10] = dem_val / 1000.0  # 归一化到km尺度
                features_26[i, 11] = self.slope[row, col] / 90.0  # 归一化到[0,1]
                features_26[i, 12] = np.sin(np.deg2rad(self.aspect[row, col]))
                features_26[i, 13] = np.cos(np.deg2rad(self.aspect[row, col]))
                
                # LULC one-hot (10维)
                lulc_val = self.lulc[row, col]
                lulc_classes = [10, 20, 30, 40, 50, 60, 80, 90, 100, 255]
                for j, lulc_class in enumerate(lulc_classes):
                    features_26[i, 14+j] = 1.0 if lulc_val == lulc_class else 0.0
                
                # Tree cover
                features_26[i, 24] = 1.0 if lulc_val == 10 else 0.0
                
                # Road
                if self.road is not None:
                    road_val = float(self.road[row, col])
                    features_26[i, 25] = 1.0 if road_val > 0.5 else 0.0
                else:
                    features_26[i, 25] = 1.0 if lulc_val == 80 else 0.0
        
        # 计算10维运动学特征（相对坐标）
        current_pos = history_abs[-1]
        history_rel = (history_abs - current_pos) / 1000.0  # 转换为km
        features_26[:, 0:2] = history_rel
        
        # 速度 (vx, vy, speed) - 归一化到合理范围
        dt = 10.0
        vel = np.diff(history_abs, axis=0) / dt  # m/s
        vel = np.vstack([vel[0:1], vel])
        features_26[:, 2:4] = vel / 30.0  # 假设最大速度约30m/s，归一化到±1左右
        features_26[:, 9] = np.linalg.norm(vel, axis=1) / 30.0  # 速度模归一化
        
        # 加速度 (ax, ay) - 归一化到合理范围
        acc = np.diff(vel, axis=0) / dt  # m/s^2
        acc = np.vstack([acc[0:1], acc])
        features_26[:, 4:6] = acc / 3.0  # 假设最大加速度约3m/s^2，归一化到±1左右
        
        # 航向 (h_sin, h_cos)
        headings = np.arctan2(vel[:, 1], vel[:, 0])
        features_26[:, 6] = np.sin(headings)
        features_26[:, 7] = np.cos(headings)
        
        # 曲率
        dx, dy = vel[:, 0], vel[:, 1]
        ddx, ddy = acc[:, 0], acc[:, 1]
        numer = np.abs(dx * ddy - dy * ddx)
        denom = (dx**2 + dy**2)**(1.5) + 1e-6
        features_26[:, 8] = numer / denom
        
        return features_26
    
    def extract_100km_env_map(
        self,
        center_utm: Tuple[float, float],
        history_abs: np.ndarray,
        patch_size: int = 128
    ) -> torch.Tensor:
        """
        提取140km×140km环境地图（以最后观测点为中心）
        
        Args:
            center_utm: 中心点UTM坐标（最后观测点）
            history_abs: 历史轨迹绝对坐标
            patch_size: 地图尺寸（像素）
        
        Returns:
            (18, patch_size, patch_size) 环境特征张量
        """
        # 140km×140km，分辨率约1094m/pixel
        coverage = 140000.0  # 140km
        resolution = coverage / patch_size  # ~1094m
        
        col, row = ~self.transform * center_utm
        col, row = int(col), int(row)
        
        # 计算提取范围（原始30m分辨率）
        half_size = int(coverage / 2 / 30)
        row_start = max(0, row - half_size)
        row_end = min(self.dem.shape[0], row + half_size)
        col_start = max(0, col - half_size)
        col_end = min(self.dem.shape[1], col + half_size)
        
        # 提取数据块
        dem_patch = self.dem[row_start:row_end, col_start:col_end]
        slope_patch = self.slope[row_start:row_end, col_start:col_end]
        aspect_patch = self.aspect[row_start:row_end, col_start:col_end]
        lulc_patch = self.lulc[row_start:row_end, col_start:col_end]
        
        # 缩放到目标尺寸
        from scipy.ndimage import zoom
        
        def resize_patch(p, size, order=1):
            if p.shape[0] == 0 or p.shape[1] == 0:
                return np.zeros((size, size), dtype=p.dtype)
            if p.shape[0] != size or p.shape[1] != size:
                zf = (size / p.shape[0], size / p.shape[1])
                return zoom(p, zf, order=order)
            return p
        
        dem_patch = resize_patch(dem_patch, patch_size, order=1)
        slope_patch = resize_patch(slope_patch, patch_size, order=1)
        aspect_patch = resize_patch(aspect_patch, patch_size, order=1)
        lulc_patch = resize_patch(lulc_patch, patch_size, order=0)

        # 统一处理类型/非法值，避免 one-hot 全 0 或 NaN 传播
        if np.issubdtype(lulc_patch.dtype, np.floating):
            lulc_patch = np.rint(lulc_patch)
        lulc_patch = lulc_patch.astype(np.int32, copy=False)
        
        # 构建18通道特征
        channels = []
        
        # DEM (局部归一化)
        # 注意：dem_patch 内可能包含 nodata(=NaN)，直接 mean/std 会导致整图变 NaN
        dem_valid = np.isfinite(dem_patch)
        if np.any(dem_valid):
            dem_mean = float(np.nanmean(dem_patch))
            dem_std = float(np.nanstd(dem_patch))
            dem_std = dem_std if dem_std > 1e-6 else 1.0
            dem_filled = np.where(dem_valid, dem_patch, dem_mean)
            dem_norm = (dem_filled - dem_mean) / (dem_std + 1e-6)
        else:
            dem_norm = np.zeros((patch_size, patch_size), dtype=np.float32)
        channels.append(dem_norm)
        
        # Slope
        slope_norm = slope_patch / 90.0
        channels.append(slope_norm)
        
        # Aspect sin/cos
        aspect_rad = np.deg2rad(aspect_patch)
        channels.append(np.sin(aspect_rad))
        channels.append(np.cos(aspect_rad))
        
        # LULC one-hot (10通道)
        lulc_classes = [10, 20, 30, 40, 50, 60, 80, 90, 100, 255]
        # 对未覆盖的 LULC 值统一映射到 255(unknown)，避免 one-hot 全 0
        known_mask = np.isin(lulc_patch, np.array(lulc_classes, dtype=np.int32))
        lulc_patch = np.where(known_mask, lulc_patch, 255)
        for lulc_val in lulc_classes:
            lulc_channel = (lulc_patch == lulc_val).astype(np.float32)
            channels.append(lulc_channel)
        
        # Tree cover
        tree_cover = (lulc_patch == 10).astype(np.float32)
        channels.append(tree_cover)
        
        # Road
        if self.road is not None:
            road_patch = self.road[row_start:row_end, col_start:col_end]
            road_patch = resize_patch(road_patch, patch_size, order=0)
        else:
            road_patch = (lulc_patch == 80).astype(np.float32)
        channels.append(road_patch)
        
        # History heatmap
        history_heatmap = np.zeros((patch_size, patch_size), dtype=np.float32)
        for pt in history_abs:
            p_col, p_row = ~self.transform * (pt[0], pt[1])
            local_row = int((p_row - (row - half_size)) * (patch_size / (half_size * 2)))
            local_col = int((p_col - (col - half_size)) * (patch_size / (half_size * 2)))
            if 0 <= local_row < patch_size and 0 <= local_col < patch_size:
                history_heatmap[max(0, local_row-1):min(patch_size, local_row+2),
                                max(0, local_col-1):min(patch_size, local_col+2)] = 1.0
        channels.append(history_heatmap)
        
        # Candidate goal map (空白，训练时动态生成)
        goal_map = np.zeros((patch_size, patch_size), dtype=np.float32)
        channels.append(goal_map)
        
        # 转换为tensor
        env_map = np.stack(channels, axis=0).astype(np.float32)
        return torch.from_numpy(env_map)
    
    def generate_complete_trajectory(
        self,
        intent: str,
        vehicle_type: str,
        min_distance: float = 60000.0,
        trajectory_id: int = 0,
        planning_overrides: Optional[Dict] = None,
        smoothing_overrides: Optional[Dict] = None,
        sampling_overrides: Optional[Dict] = None,
        start_goal_override: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> Optional[Dict]:
        """
        生成完整轨迹（包含所有特征）
        
        Returns:
            轨迹字典，包含：
            - 基本信息（region, intent, vehicle_type, start_utm, goal_utm）
            - 路径和速度（path, speeds, timestamps）
            - 统计信息（duration, length, num_points）
            - 速度校准信息（speed_calibration）
            - 完整特征（samples: list of {history_feat_26d, env_map_100km, future_rel, current_pos_abs}）
        """
        logger.info(f"生成完整轨迹 ID={trajectory_id} - 意图: {intent}, 车辆: {vehicle_type}")

        self.last_failure = None

        timings: Dict[str, float] = {}
        t_total_start = perf_counter()

        qcfg = cfg.get('trajectory_generation.quality_filter', {})
        quality_enabled = bool(qcfg.get('enabled', False))
        quality_max_attempts = int(qcfg.get('max_attempts', 1)) if quality_enabled else 1
        quality_max_attempts = max(1, quality_max_attempts)
        
        # 1. 采样起终点
        # 为了能产生训练样本，需要至少 75min (450 个 10s 点)。
        # 对于更快的车辆，同样的直线距离会导致时长更短，从而在后续被跳过。
        # 因此在采样阶段按车型 target_speed 动态提高最小直线距离，减少无效计算。
        params = VEHICLE_PARAMS[vehicle_type]
        target_speed = float(params['target_speed'])
        required_length_m = target_speed * 4500.0  # 75min
        # 路径长度通常大于直线距离，使用保守的迂回系数预估，避免过度提高阈值
        sampling_overrides = sampling_overrides or {}
        estimated_circuity = float(sampling_overrides.get(
            'estimated_circuity',
            cfg.get('trajectory_generation.sampling.estimated_circuity', 1.25),
        ))
        estimated_circuity = max(1e-6, estimated_circuity)
        effective_min_distance = max(float(min_distance), required_length_m / estimated_circuity)

        prefer_road = bool(sampling_overrides.get('prefer_road', cfg.get('trajectory_generation.sampling.prefer_road', False)))
        road_buffer_m = float(sampling_overrides.get('road_buffer_m', cfg.get('trajectory_generation.sampling.road_buffer_m', 0.0)))
        force_on_road = bool(sampling_overrides.get('force_on_road', cfg.get('trajectory_generation.sampling.force_on_road', False)))

        attempt_records: List[Dict] = []
        timings['sample_start_goal_s'] = 0.0
        timings['planning_s'] = 0.0

        accepted_start_utm: Optional[Tuple[float, float]] = None
        accepted_goal_utm: Optional[Tuple[float, float]] = None
        accepted_path: Optional[List[Tuple[float, float]]] = None

        planner = HierarchicalAStarPlanner(self.region, intent, vehicle_type)
        planning_overrides = planning_overrides or {}
        downsample_factor = int(planning_overrides.get('downsample_factor', 10))
        waypoint_interval = float(planning_overrides.get('waypoint_interval', 5000.0))
        corridor_width = float(planning_overrides.get('corridor_width', 2000.0))
        refine_mode = str(planning_overrides.get('refine_mode', 'repair'))
        densify_step_m = float(planning_overrides.get('densify_step_m', 200.0))
        high_cost_threshold = float(planning_overrides.get('high_cost_threshold', 0.5))
        high_cost_fraction = float(planning_overrides.get('high_cost_fraction', 0.2))

        for attempt_idx in range(quality_max_attempts):
            attempt_info: Dict[str, object] = {'attempt': int(attempt_idx + 1)}

            t0 = perf_counter()
            if start_goal_override is not None and attempt_idx == 0:
                result = start_goal_override
            else:
                result = sample_start_goal_v2(
                    self.region,
                    vehicle_type,
                    effective_min_distance,
                    prefer_road=prefer_road,
                    road_buffer_m=road_buffer_m,
                    force_on_road=force_on_road,
                )
            dt_sample = perf_counter() - t0
            timings['sample_start_goal_s'] = float(timings.get('sample_start_goal_s', 0.0)) + float(dt_sample)
            attempt_info['sample_s'] = float(dt_sample)
            if result is None:
                attempt_info['result'] = 'sample_failed'
                attempt_records.append(attempt_info)
                continue

            start_utm, goal_utm = result
            attempt_info['start_utm'] = (float(start_utm[0]), float(start_utm[1]))
            attempt_info['goal_utm'] = (float(goal_utm[0]), float(goal_utm[1]))

            t0 = perf_counter()
            path = planner.hierarchical_plan(
                start_utm,
                goal_utm,
                downsample_factor=downsample_factor,
                waypoint_interval=waypoint_interval,
                corridor_width=corridor_width,
                refine_mode=refine_mode,
                densify_step_m=densify_step_m,
                high_cost_threshold=high_cost_threshold,
                high_cost_fraction=high_cost_fraction,
            )
            dt_plan = perf_counter() - t0
            timings['planning_s'] = float(timings.get('planning_s', 0.0)) + float(dt_plan)
            attempt_info['planning_s'] = float(dt_plan)
            if path is None:
                attempt_info['result'] = 'planning_failed'
                attempt_records.append(attempt_info)
                continue

            planned_path_list = [(float(x), float(y)) for x, y in path]
            if quality_enabled:
                too_straight, qdetail = _is_too_straight(planned_path_list, intent)
                attempt_info['quality'] = qdetail
                if too_straight and (attempt_idx + 1) < quality_max_attempts:
                    attempt_info['result'] = 'too_straight_retry'
                    attempt_records.append(attempt_info)
                    continue
                if too_straight:
                    attempt_info['result'] = 'too_straight'
                    attempt_records.append(attempt_info)
                    break

            attempt_info['result'] = 'ok'
            attempt_records.append(attempt_info)
            accepted_start_utm = start_utm
            accepted_goal_utm = goal_utm
            accepted_path = path
            break

        timings['attempts'] = float(len(attempt_records))
        if len(attempt_records) > 0:
            timings['quality_enabled'] = float(1.0 if quality_enabled else 0.0)

        if accepted_path is None:
            results = [str(r.get('result')) for r in attempt_records]
            any_sample_ok = any(r in ('planning_failed', 'too_straight', 'too_straight_retry', 'ok') for r in results)
            any_plan_ok = any(r in ('too_straight', 'too_straight_retry', 'ok') for r in results)
            any_too_straight = any(r in ('too_straight', 'too_straight_retry') for r in results)

            if not any_sample_ok:
                reason = 'sample_failed'
                logger.error("起终点采样失败")
            elif not any_plan_ok:
                reason = 'planning_failed'
                logger.error("路径规划失败")
            elif any_too_straight:
                reason = 'too_straight'
                logger.error("轨迹过直，已放弃")
            else:
                reason = 'planning_failed'
                logger.error("路径规划失败")

            timings['total_s'] = perf_counter() - t_total_start
            self.last_failure = {
                'reason': reason,
                'detail': {
                    'attempts': attempt_records,
                },
                'timings': timings,
            }
            logger.info(
                "  耗时(s): "
                f"sample={timings.get('sample_start_goal_s', 0.0):.2f}, "
                f"plan={timings.get('planning_s', 0.0):.2f}, "
                f"total={timings.get('total_s', 0.0):.2f}"
            )
            return None

        start_utm = accepted_start_utm
        goal_utm = accepted_goal_utm
        path = accepted_path
        
        # 3. 路径平滑
        t0 = perf_counter()
        smoothing_overrides = smoothing_overrides or {}
        smoothing_factor = float(smoothing_overrides.get('smoothing_factor', 0.0))
        resample_max_dist = float(smoothing_overrides.get('resample_max_dist', 30.0))
        smoothed_path = smooth_path(path, smoothing_factor=smoothing_factor, resample_max_dist=resample_max_dist)
        timings['smoothing_s'] = perf_counter() - t0
        
        # 4. 速度预测（带校准）
        t0 = perf_counter()
        speeds, calibration_info = self.predict_speeds_with_calibration(smoothed_path, vehicle_type)
        timings['speed_pred_s'] = perf_counter() - t0
        
        # 5. 计算距离
        t0 = perf_counter()
        distances = np.zeros(len(smoothed_path))
        for i in range(1, len(smoothed_path)):
            dx = smoothed_path[i][0] - smoothed_path[i-1][0]
            dy = smoothed_path[i][1] - smoothed_path[i-1][1]
            distances[i] = np.sqrt(dx**2 + dy**2)
        timings['distance_calc_s'] = perf_counter() - t0
        
        total_length = np.sum(distances)
        
        # 检查长度要求
        if total_length < min_distance:
            logger.warning(f"轨迹长度不足: {total_length/1000:.2f} km < {min_distance/1000:.2f} km")
            timings['total_s'] = perf_counter() - t_total_start
            self.last_failure = {
                'reason': 'length_too_short',
                'detail': {
                    'length_m': float(total_length),
                    'min_distance_m': float(min_distance),
                },
                'timings': timings,
            }
            logger.info(
                "  耗时(s): "
                f"sample={timings.get('sample_start_goal_s', 0.0):.2f}, "
                f"plan={timings.get('planning_s', 0.0):.2f}, "
                f"smooth={timings.get('smoothing_s', 0.0):.2f}, "
                f"speed={timings.get('speed_pred_s', 0.0):.2f}, "
                f"total={timings.get('total_s', 0.0):.2f}"
            )
            return None
        
        # 6. 运动学约束
        t0 = perf_counter()
        constrained_speeds = self.apply_kinematic_constraints(speeds, distances, vehicle_type)
        timings['kinematic_s'] = perf_counter() - t0
        
        # 7. 时间戳
        t0 = perf_counter()
        timestamps = self.compute_timestamps(smoothed_path, constrained_speeds)
        timings['timestamps_s'] = perf_counter() - t0
        duration = timestamps[-1]
        
        # 8. 重采样到10秒间隔
        t0 = perf_counter()
        path_10s, speeds_10s, timestamps_10s = self.resample_to_10s(
            smoothed_path, constrained_speeds, timestamps
        )
        timings['resample_10s_s'] = perf_counter() - t0

        history_len = 90
        future_len = 360
        total_len = history_len + future_len

        # 训练样本需要 90 历史 + 360 未来（共 450 个 10s 采样点）=> 至少 4500s (75min)
        if duration < 4500.0:
            logger.warning(
                f"轨迹时长不足以生成样本: {duration/60:.1f} min < 75.0 min，跳过"
            )
            timings['total_s'] = perf_counter() - t_total_start
            self.last_failure = {
                'reason': 'duration_too_short',
                'detail': {
                    'duration_s': float(duration),
                    'min_duration_s': 4500.0,
                },
                'timings': timings,
            }
            logger.info(
                "  耗时(s): "
                f"sample={timings.get('sample_start_goal_s', 0.0):.2f}, "
                f"plan={timings.get('planning_s', 0.0):.2f}, "
                f"smooth={timings.get('smoothing_s', 0.0):.2f}, "
                f"speed={timings.get('speed_pred_s', 0.0):.2f}, "
                f"kin={timings.get('kinematic_s', 0.0):.2f}, "
                f"ts={timings.get('timestamps_s', 0.0):.2f}, "
                f"total={timings.get('total_s', 0.0):.2f}"
            )
            return None

        # 训练样本需要 90 历史 + 360 未来（共 450 个 10s 采样点）
        if len(path_10s) < total_len:
            logger.warning(
                f"轨迹可用点数不足以生成样本: {len(path_10s)} < {total_len} (history={history_len}, future={future_len})，跳过"
            )
            timings['total_s'] = perf_counter() - t_total_start
            self.last_failure = {
                'reason': 'points_too_few',
                'detail': {
                    'num_points': int(len(path_10s)),
                    'required_points': int(total_len),
                },
                'timings': timings,
            }
            logger.info(
                "  耗时(s): "
                f"sample={timings.get('sample_start_goal_s', 0.0):.2f}, "
                f"plan={timings.get('planning_s', 0.0):.2f}, "
                f"smooth={timings.get('smoothing_s', 0.0):.2f}, "
                f"speed={timings.get('speed_pred_s', 0.0):.2f}, "
                f"kin={timings.get('kinematic_s', 0.0):.2f}, "
                f"ts={timings.get('timestamps_s', 0.0):.2f}, "
                f"res10s={timings.get('resample_10s_s', 0.0):.2f}, "
                f"total={timings.get('total_s', 0.0):.2f}"
            )
            return None
        
        # 9. 生成训练样本（滑动窗口：90历史 + 360未来）
        t0 = perf_counter()
        t_hist_feat = 0.0
        t_env_map = 0.0
        windows_considered = 0
        samples = []
        step = int(cfg.get('trajectory_generation.sample_window_step', 30))
        goal_linf_km_list: List[float] = []
        goal_l2_km_list: List[float] = []
        goal_in_bounds = 0
        goal_out_bounds = 0

        def _histogram(values: List[float], bin_edges: List[float]) -> Dict[str, int]:
            if len(values) == 0:
                return {f"{bin_edges[i]}-{bin_edges[i+1]}": 0 for i in range(len(bin_edges) - 1)}
            arr = np.asarray(values, dtype=np.float32)
            counts, _ = np.histogram(arr, bins=np.asarray(bin_edges, dtype=np.float32))
            return {f"{bin_edges[i]}-{bin_edges[i+1]}": int(counts[i]) for i in range(len(bin_edges) - 1)}
        
        for start_idx in range(0, len(path_10s) - total_len, step):
            windows_considered += 1
            history_abs = path_10s[start_idx:start_idx + history_len]
            future_abs = path_10s[start_idx + history_len:start_idx + total_len]
            current_pos_abs = history_abs[-1]
            
            # 提取26维历史特征
            t1 = perf_counter()
            history_feat_26d = self.extract_26d_history_features(history_abs)
            t_hist_feat += perf_counter() - t1
            
            # 提取140km×140km环境地图
            t1 = perf_counter()
            env_map_100km = self.extract_100km_env_map(
                center_utm=(float(current_pos_abs[0]), float(current_pos_abs[1])),
                history_abs=history_abs,
                patch_size=128
            )
            t_env_map += perf_counter() - t1
            
            # 未来轨迹（相对坐标，km）
            future_rel = (future_abs - current_pos_abs) / 1000.0

            abs_xy = np.abs(future_rel)
            max_abs_dx_km = float(abs_xy[:, 0].max())
            max_abs_dy_km = float(abs_xy[:, 1].max())
            max_future_linf_km = float(np.maximum(abs_xy[:, 0], abs_xy[:, 1]).max())
            max_future_l2_km = float(np.linalg.norm(future_rel, axis=1).max())

            goal_rel = future_rel[-1]
            goal_linf_km = float(np.maximum(abs(float(goal_rel[0])), abs(float(goal_rel[1]))))
            goal_l2_km = float(np.linalg.norm(goal_rel))
            goal_linf_km_list.append(goal_linf_km)
            goal_l2_km_list.append(goal_l2_km)
            if (abs(float(goal_rel[0])) > 70.0) or (abs(float(goal_rel[1])) > 70.0):
                goal_out_bounds += 1
                continue
            goal_in_bounds += 1
            
            samples.append({
                'history_feat_26d': history_feat_26d.astype(np.float32),
                'env_map_100km': env_map_100km.numpy().astype(np.float16),
                'future_rel': future_rel.astype(np.float32),
                'current_pos_abs': current_pos_abs.astype(np.float64),
                'goal_rel': goal_rel.astype(np.float32),
                'max_abs_dx_km': max_abs_dx_km,
                'max_abs_dy_km': max_abs_dy_km,
                'max_future_linf_km': max_future_linf_km,
                'max_future_l2_km': max_future_l2_km
            })

        timings['sample_windows_total_s'] = perf_counter() - t0
        timings['history_feat_total_s'] = float(t_hist_feat)
        timings['env_map_total_s'] = float(t_env_map)
        timings['windows_considered'] = float(windows_considered)
        timings['windows_kept'] = float(len(samples))

        window_stats = {
            'windows_considered': int(windows_considered),
            'goal_in_bounds': int(goal_in_bounds),
            'goal_out_bounds': int(goal_out_bounds),
        }
        if len(goal_linf_km_list) > 0:
            linf_arr = np.asarray(goal_linf_km_list, dtype=np.float32)
            l2_arr = np.asarray(goal_l2_km_list, dtype=np.float32)
            bin_edges = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 150.0]
            window_stats.update({
                'goal_linf_km': {
                    'min': float(linf_arr.min()),
                    'p50': float(np.percentile(linf_arr, 50)),
                    'p90': float(np.percentile(linf_arr, 90)),
                    'max': float(linf_arr.max()),
                    'mean': float(linf_arr.mean()),
                },
                'goal_l2_km': {
                    'min': float(l2_arr.min()),
                    'p50': float(np.percentile(l2_arr, 50)),
                    'p90': float(np.percentile(l2_arr, 90)),
                    'max': float(l2_arr.max()),
                    'mean': float(l2_arr.mean()),
                },
                'goal_linf_km_hist': _histogram(goal_linf_km_list, bin_edges),
                'goal_l2_km_hist': _histogram(goal_l2_km_list, bin_edges),
            })
        
        # 10. 构建完整轨迹字典
        planned_path_list = [(float(x), float(y)) for x, y in path]
        smoothed_path_list = [(float(x), float(y)) for x, y in smoothed_path]
        resampled_path_list = [(float(x), float(y)) for x, y in path_10s]

        stage_stats = {
            'planned': _path_stats(planned_path_list),
            'smoothed': _path_stats(smoothed_path_list),
            'resampled_10s': _path_stats(resampled_path_list),
            'params': {
                'planning': {
                    'downsample_factor': int(downsample_factor),
                    'waypoint_interval_m': float(waypoint_interval),
                    'corridor_width_m': float(corridor_width),
                    'refine_mode': str(refine_mode),
                    'densify_step_m': float(densify_step_m),
                    'high_cost_threshold': float(high_cost_threshold),
                    'high_cost_fraction': float(high_cost_fraction),
                },
                'smoothing_factor': float(smoothing_factor),
                'resample_max_dist_m': float(resample_max_dist),
            },
        }
        trajectory = {
            'region': self.region,
            'intent': intent,
            'vehicle_type': vehicle_type,
            'start_utm': start_utm,
            'goal_utm': goal_utm,
            'path_planned': planned_path_list,
            'path_smoothed': smoothed_path_list,
            'path': path_10s.tolist(),
            'speeds': speeds_10s.tolist(),
            'timestamps': timestamps_10s.tolist(),
            'duration': float(duration),
            'length': float(total_length),
            'num_points': len(path_10s),
            'speed_calibration': calibration_info,
            'samples': samples,
            'num_samples': len(samples),
            'window_stats': window_stats,
            'stage_stats': stage_stats,
        }

        timings['total_s'] = perf_counter() - t_total_start
        trajectory['timings'] = timings

        if trajectory['num_samples'] == 0:
            logger.warning("轨迹样本全部超出 env_map_100km 覆盖范围(±50km)，跳过")
            timings['total_s'] = perf_counter() - t_total_start
            self.last_failure = {
                'reason': 'env_map_filtered',
                'detail': {
                    'windows_considered': int(timings.get('windows_considered', 0.0)),
                    'windows_kept': int(timings.get('windows_kept', 0.0)),
                    'window_stats': window_stats,
                },
                'timings': timings,
            }
            return None
        
        logger.info(f"✅ 轨迹生成完成")
        logger.info(f"  长度: {total_length/1000:.2f} km")
        logger.info(f"  时长: {duration/60:.2f} 分钟")
        logger.info(f"  点数: {len(path_10s)}")
        logger.info(f"  样本数: {len(samples)}")
        logger.info(f"  平均速度: {calibration_info['calibrated_mean']:.2f} m/s")
        logger.info(
            "  耗时(s): "
            f"sample={timings.get('sample_start_goal_s', 0.0):.2f}, "
            f"plan={timings.get('planning_s', 0.0):.2f}, "
            f"smooth={timings.get('smoothing_s', 0.0):.2f}, "
            f"speed={timings.get('speed_pred_s', 0.0):.2f}, "
            f"kin={timings.get('kinematic_s', 0.0):.2f}, "
            f"ts={timings.get('timestamps_s', 0.0):.2f}, "
            f"res10s={timings.get('resample_10s_s', 0.0):.2f}, "
            f"samples={timings.get('sample_windows_total_s', 0.0):.2f} "
            f"(hist={timings.get('history_feat_total_s', 0.0):.2f}, env={timings.get('env_map_total_s', 0.0):.2f}, "
            f"kept={int(timings.get('windows_kept', 0.0))}/{int(timings.get('windows_considered', 0.0))}), "
            f"total={timings.get('total_s', 0.0):.2f}"
        )
        
        return trajectory
