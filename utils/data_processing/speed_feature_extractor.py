"""
速度预测训练样本构建

按论文表3.6提取20维特征：
1. DEM elevation (1)
2. Slope (1)
3. Aspect sin/cos (2)
4. LULC one-hot (10)
5. Tree cover (1)
6. Effective slope (1)
7. Curvature (1)
8. Past 10m avg curvature (1)
9. Future 10m max curvature (1)
10. On road (1)

目标: y = log(1 + v)
"""
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import pickle
import rasterio
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from config import cfg, get_path

logger = logging.getLogger(__name__)

# LULC类别（GlobeLand30）
LULC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class SpeedFeatureExtractor:
    """速度预测特征提取器"""
    
    def __init__(self, region: str = 'scottish_highlands'):
        self.region = region
        
        # 加载UTM栅格
        utm_dir = Path(get_path('paths.processed.utm_grid')) / region
        
        logger.info(f"加载 {region} UTM栅格数据...")
        self.dem_src = rasterio.open(utm_dir / 'dem_utm.tif')
        self.slope_src = rasterio.open(utm_dir / 'slope_utm.tif')
        self.aspect_src = rasterio.open(utm_dir / 'aspect_utm.tif')
        self.lulc_src = rasterio.open(utm_dir / 'lulc_utm.tif')
        
        # 读取数据到内存（加速查询）
        self.dem = self.dem_src.read(1)
        self.slope = self.slope_src.read(1)
        self.aspect = self.aspect_src.read(1)
        self.lulc = self.lulc_src.read(1)
        
        # 处理nodata
        self.dem = np.where(self.dem == -32768, np.nan, self.dem)
        
        logger.info(f"  DEM: {self.dem.shape}, 值域 {np.nanmin(self.dem):.1f}~{np.nanmax(self.dem):.1f}m")
        logger.info(f"  Slope: {self.slope.shape}, 值域 {self.slope.min():.1f}~{self.slope.max():.1f}°")
        logger.info(f"  LULC类别: {np.unique(self.lulc).tolist()}")
    
    def __del__(self):
        """关闭栅格文件"""
        if hasattr(self, 'dem_src'):
            self.dem_src.close()
        if hasattr(self, 'slope_src'):
            self.slope_src.close()
        if hasattr(self, 'aspect_src'):
            self.aspect_src.close()
        if hasattr(self, 'lulc_src'):
            self.lulc_src.close()
    
    def utm_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """UTM坐标转像素坐标"""
        row, col = self.dem_src.index(x, y)
        return row, col
    
    def sample_raster(self, raster: np.ndarray, x: float, y: float) -> float:
        """
        从栅格采样值（双线性插值）
        
        Args:
            raster: 栅格数组
            x, y: UTM坐标
        
        Returns:
            采样值
        """
        try:
            row, col = self.utm_to_pixel(x, y)
            
            # 边界检查
            if row < 0 or row >= raster.shape[0] or col < 0 or col >= raster.shape[1]:
                return np.nan
            
            return float(raster[row, col])
        except:
            return np.nan
    
    def compute_curvature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算路径曲率 κ = |dθ/ds|
        
        Args:
            x, y: 轨迹坐标数组
        
        Returns:
            曲率数组（与输入同长度）
        """
        n = len(x)
        if n < 3:
            return np.zeros(n)
        
        # 计算方向角
        dx = np.diff(x)
        dy = np.diff(y)
        theta = np.arctan2(dy, dx)
        
        # 计算角度变化（处理跳变）
        dtheta = np.diff(theta)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))  # 归一化到[-π, π]
        
        # 计算弧长
        ds = np.sqrt(dx[:-1]**2 + dy[:-1]**2)
        ds = np.where(ds < 0.1, 0.1, ds)  # 避免除零
        
        # 曲率
        curvature = np.abs(dtheta / ds)
        
        # 填充到原始长度（首尾用邻近值）
        result = np.zeros(n)
        result[1:-1] = curvature
        result[0] = curvature[0] if len(curvature) > 0 else 0
        result[-1] = curvature[-1] if len(curvature) > 0 else 0
        
        return result
    
    def compute_effective_slope(self, x: np.ndarray, y: np.ndarray, dem: np.ndarray) -> np.ndarray:
        """
        计算有效坡度（沿运动方向的坡度）
        
        Args:
            x, y: 轨迹坐标
            dem: DEM高程值
        
        Returns:
            有效坡度数组（度）
        """
        n = len(x)
        if n < 2:
            return np.zeros(n)
        
        # 计算高程差和水平距离
        dh = np.diff(dem)
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        ds = np.where(ds < 0.1, 0.1, ds)
        
        # 有效坡度（度）
        eff_slope = np.rad2deg(np.arctan(dh / ds))
        
        # 填充
        result = np.zeros(n)
        result[:-1] = eff_slope
        result[-1] = eff_slope[-1] if len(eff_slope) > 0 else 0
        
        return result
    
    def extract_features(self, traj) -> Tuple[np.ndarray, np.ndarray]:
        """
        从单条轨迹提取特征
        
        Args:
            traj: Trajectory对象
        
        Returns:
            (features, targets)
            features: (N, 20)
            targets: (N,) - log(1+v)
        """
        # 从Trajectory对象获取数据
        x = traj.positions_xy[:, 0]  # UTM x
        y = traj.positions_xy[:, 1]  # UTM y
        speed = traj.speed
        
        n = len(x)
        features = np.zeros((n, 20))
        
        # 采样环境栅格
        dem_vals = np.array([self.sample_raster(self.dem, x[i], y[i]) for i in range(n)])
        slope_vals = np.array([self.sample_raster(self.slope, x[i], y[i]) for i in range(n)])
        aspect_vals = np.array([self.sample_raster(self.aspect, x[i], y[i]) for i in range(n)])
        lulc_vals = np.array([self.sample_raster(self.lulc, x[i], y[i]) for i in range(n)])
        
        # 特征1: DEM
        features[:, 0] = dem_vals
        
        # 特征2: Slope
        features[:, 1] = slope_vals
        
        # 特征3-4: Aspect sin/cos
        aspect_rad = np.deg2rad(aspect_vals)
        features[:, 2] = np.sin(aspect_rad)
        features[:, 3] = np.cos(aspect_rad)
        
        # 特征5-14: LULC one-hot (10类)
        for i, lulc_class in enumerate(LULC_CLASSES):
            features[:, 4 + i] = (lulc_vals == lulc_class).astype(float)
        
        # 特征15: Tree cover（简化：森林=1，其他=0）
        features[:, 14] = (lulc_vals == 20).astype(float)
        
        # 特征16: Effective slope
        eff_slope = self.compute_effective_slope(x, y, dem_vals)
        features[:, 15] = eff_slope
        
        # 特征17: Curvature
        curvature = self.compute_curvature(x, y)
        features[:, 16] = curvature
        
        # 特征18: Past 10m avg curvature
        past_curv = np.zeros(n)
        for i in range(n):
            # 找过去10m内的点
            if i == 0:
                past_curv[i] = curvature[i]
            else:
                dist_back = 0
                j = i - 1
                curv_sum = 0
                count = 0
                while j >= 0 and dist_back < 10:
                    dist_back += np.sqrt((x[j+1]-x[j])**2 + (y[j+1]-y[j])**2)
                    curv_sum += curvature[j]
                    count += 1
                    j -= 1
                past_curv[i] = curv_sum / count if count > 0 else curvature[i]
        features[:, 17] = past_curv
        
        # 特征19: Future 10m max curvature
        future_curv = np.zeros(n)
        for i in range(n):
            if i == n - 1:
                future_curv[i] = curvature[i]
            else:
                dist_fwd = 0
                j = i + 1
                curv_max = 0
                while j < n and dist_fwd < 10:
                    dist_fwd += np.sqrt((x[j]-x[j-1])**2 + (y[j]-y[j-1])**2)
                    curv_max = max(curv_max, curvature[j])
                    j += 1
                future_curv[i] = curv_max
        features[:, 18] = future_curv
        
        # 特征20: On road（简化：人造地表=1，其他=0）
        features[:, 19] = (lulc_vals == 80).astype(float)
        
        # 目标值: log(1 + v)
        targets = np.log(1 + speed)
        
        return features, targets
    
    def build_training_data(self, trajectories_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        从OORD轨迹构建训练数据
        
        Args:
            trajectories_path: 轨迹pickle文件路径
        
        Returns:
            (X, y, traj_ids)
            X: (N, 20) 特征矩阵
            y: (N,) 目标值
            traj_ids: (N,) 每个样本所属轨迹ID
        """
        if trajectories_path is None:
            trajectories_path = Path(get_path('paths.processed.trajectories')) / 'oord_trajectories.pkl'
        
        logger.info(f"加载轨迹数据: {trajectories_path}")
        with open(trajectories_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取轨迹列表
        trajectories = data['trajectories']
        
        logger.info(f"共 {len(trajectories)} 条轨迹")
        
        all_features = []
        all_targets = []
        all_traj_ids = []
        
        for traj_id, traj in enumerate(trajectories):
            logger.info(f"  处理轨迹 {traj_id + 1}/{len(trajectories)} ({traj.run_id}): {len(traj.positions_xy)} 点")
            
            try:
                features, targets = self.extract_features(traj)
                
                # 过滤无效样本（NaN）
                valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(targets)
                features = features[valid_mask]
                targets = targets[valid_mask]
                
                logger.info(f"    有效样本: {len(features)}/{len(valid_mask)}")
                
                all_features.append(features)
                all_targets.append(targets)
                all_traj_ids.extend([traj_id] * len(features))
                
            except Exception as e:
                logger.error(f"    处理失败: {e}")
                continue
        
        X = np.vstack(all_features)
        y = np.hstack(all_targets)
        traj_ids = np.array(all_traj_ids)
        
        logger.info(f"\n总样本数: {len(X)}")
        logger.info(f"特征维度: {X.shape}")
        logger.info(f"目标值范围: {y.min():.3f} ~ {y.max():.3f}")
        logger.info(f"速度范围: {(np.exp(y.min())-1):.2f} ~ {(np.exp(y.max())-1):.2f} m/s")
        
        return X, y, traj_ids
    
    def save_training_data(self, X: np.ndarray, y: np.ndarray, traj_ids: np.ndarray, output_dir: Optional[Path] = None):
        """保存训练数据"""
        if output_dir is None:
            output_dir = Path(get_path('paths.processed.speed_training'))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为npz
        output_file = output_dir / 'speed_training_data.npz'
        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            traj_ids=traj_ids,
            feature_names=[
                'dem', 'slope', 'aspect_sin', 'aspect_cos',
                'lulc_10', 'lulc_20', 'lulc_30', 'lulc_40', 'lulc_50',
                'lulc_60', 'lulc_70', 'lulc_80', 'lulc_90', 'lulc_100',
                'tree_cover', 'effective_slope', 'curvature',
                'past_10m_avg_curv', 'future_10m_max_curv', 'on_road'
            ]
        )
        
        logger.info(f"\n✅ 训练数据已保存: {output_file}")
        logger.info(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 保存统计信息
        stats_file = output_dir / 'data_statistics.txt'
        with open(stats_file, 'w') as f:
            f.write("速度预测训练数据统计\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"样本数: {len(X)}\n")
            f.write(f"特征维度: {X.shape[1]}\n")
            f.write(f"轨迹数: {len(np.unique(traj_ids))}\n\n")
            
            f.write("目标值统计 (log(1+v)):\n")
            f.write(f"  均值: {y.mean():.3f}\n")
            f.write(f"  标准差: {y.std():.3f}\n")
            f.write(f"  范围: {y.min():.3f} ~ {y.max():.3f}\n\n")
            
            f.write("速度统计 (m/s):\n")
            v = np.exp(y) - 1
            f.write(f"  均值: {v.mean():.2f}\n")
            f.write(f"  标准差: {v.std():.2f}\n")
            f.write(f"  范围: {v.min():.2f} ~ {v.max():.2f}\n\n")
            
            f.write("特征统计:\n")
            feature_names = [
                'dem', 'slope', 'aspect_sin', 'aspect_cos',
                'lulc_10', 'lulc_20', 'lulc_30', 'lulc_40', 'lulc_50',
                'lulc_60', 'lulc_70', 'lulc_80', 'lulc_90', 'lulc_100',
                'tree_cover', 'effective_slope', 'curvature',
                'past_10m_avg_curv', 'future_10m_max_curv', 'on_road'
            ]
            for i, name in enumerate(feature_names):
                f.write(f"  {name:20s}: {X[:, i].mean():8.3f} ± {X[:, i].std():8.3f}\n")
        
        logger.info(f"  统计信息: {stats_file}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    extractor = SpeedFeatureExtractor('scottish_highlands')
    X, y, traj_ids = extractor.build_training_data()
    extractor.save_training_data(X, y, traj_ids)
