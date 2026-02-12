#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境数据加载器 - 加载真实的 DEM/Slope/Aspect/LULC 数据
"""
import numpy as np
import rasterio
import torch
from pathlib import Path
from typing import Tuple, Optional
import warnings

class EnvironmentDataLoader:
    """加载和处理真实环境数据"""
    
    def __init__(self, region='bohemian_forest'):
        """
        初始化环境数据加载器
        
        Args:
            region: 区域名称 ('bohemian_forest' 或 'scottish_highlands')
        """
        self.region = region
        self.data_dir = Path(f'/home/zmc/文档/programwork/data/processed/utm_grid/{region}')
        
        # 加载所有环境数据
        self.dem = None
        self.slope = None
        self.aspect = None
        self.lulc = None
        self.road = None
        
        self._load_data()
    
    def _load_data(self):
        """加载所有环境栅格数据"""
        try:
            # 加载 DEM
            with rasterio.open(self.data_dir / 'dem_utm.tif') as src:
                self.dem = src.read(1).astype(np.float32)
                self.transform = src.transform
                self.crs = src.crs
            
            # 加载 Slope
            with rasterio.open(self.data_dir / 'slope_utm.tif') as src:
                self.slope = src.read(1).astype(np.float32)
            
            # 加载 Aspect
            with rasterio.open(self.data_dir / 'aspect_utm.tif') as src:
                self.aspect = src.read(1).astype(np.float32)
            
            # 加载 LULC
            with rasterio.open(self.data_dir / 'lulc_utm.tif') as src:
                self.lulc = src.read(1).astype(np.uint8)
            
            # 加载 Road
            road_path = self.data_dir / 'road_utm.tif'
            if road_path.exists():
                with rasterio.open(road_path) as src:
                    self.road = src.read(1).astype(np.float32)
            else:
                self.road = (self.lulc == 80).astype(np.float32)
            
            print(f"✓ 成功加载 {self.region} 环境数据")
            print(f"  - DEM shape: {self.dem.shape}")
            print(f"  - Slope shape: {self.slope.shape}")
            print(f"  - Aspect shape: {self.aspect.shape}")
            print(f"  - LULC shape: {self.lulc.shape}")
            
        except Exception as e:
            raise RuntimeError(f"加载环境数据失败: {e}")
    
    def extract_patch(self, center_utm: Tuple[float, float], 
                     history_abs: Optional[np.ndarray] = None,
                     candidates_abs: Optional[np.ndarray] = None,
                     patch_size: int = 128, 
                     resolution: float = 30.0) -> torch.Tensor:
        """
        提取以给定 UTM 坐标为中心的环境数据块
        
        Args:
            center_utm: (easting, northing) UTM 坐标 (中心点)
            history_abs: (seq_len, 2) 绝对 UTM 坐标历史轨迹 (用于生成热力图)
            candidates_abs: (num_goals, 2) 绝对 UTM 坐标候选目标 (用于生成目标地图)
            patch_size: 输出块大小（像素）
            resolution: 分辨率（米/像素）
        
        Returns:
            env_map: (18, patch_size, patch_size) 环境特征张量
        """
        # 计算像素坐标
        col, row = ~self.transform * center_utm
        col, row = int(col), int(row)
        
        # 计算提取范围
        half_size = int(patch_size * resolution / 2 / 30)  # 30m 是原始分辨率
        row_start = max(0, row - half_size)
        row_end = min(self.dem.shape[0], row + half_size)
        col_start = max(0, col - half_size)
        col_end = min(self.dem.shape[1], col + half_size)
        
        # 提取数据块
        dem_patch = self.dem[row_start:row_end, col_start:col_end]
        slope_patch = self.slope[row_start:row_end, col_start:col_end]
        aspect_patch = self.aspect[row_start:row_end, col_start:col_end]
        lulc_patch = self.lulc[row_start:row_end, col_start:col_end]
        
        # 统一大小到 patch_size (处理边缘不一致情况)
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
        
        # 构建 18 通道特征
        channels = []
        
        # 1. DEM (1 通道)
        dem_norm = (dem_patch - dem_patch.mean()) / (dem_patch.std() + 1e-6)
        channels.append(dem_norm)
        
        # 2. Slope (1 通道)
        slope_norm = slope_patch / 90.0  # 归一化到 [0, 1]
        channels.append(slope_norm)
        
        # 3. Aspect sin/cos (2 通道)
        aspect_rad = np.deg2rad(aspect_patch)
        channels.append(np.sin(aspect_rad))
        channels.append(np.cos(aspect_rad))
        
        # 4. LULC one-hot (10 通道)
        lulc_classes = [10, 20, 30, 40, 50, 60, 80, 90, 100, 255]  # 10个类别
        for lulc_val in lulc_classes:
            lulc_channel = (lulc_patch == lulc_val).astype(np.float32)
            channels.append(lulc_channel)
        
        # 5. Tree cover (1 通道) - LULC=10
        tree_cover = (lulc_patch == 10).astype(np.float32)
        channels.append(tree_cover)
        
        # 6. Road (1 通道) - 从road_utm.tif读取
        road_patch = self.road[row_start:row_end, col_start:col_end]
        road_patch = resize_patch(road_patch, patch_size, order=0)
        channels.append(road_patch)
        
        # 7. History heatmap (1 通道) - 动态生成 (带高斯模糊增强泛化)
        history_heatmap = np.zeros((patch_size, patch_size), dtype=np.float32)
        if history_abs is not None:
            for pt in history_abs:
                p_col, p_row = ~self.transform * (pt[0], pt[1])
                local_row = int((p_row - (row - half_size)) * (patch_size / (half_size * 2)))
                local_col = int((p_col - (col - half_size)) * (patch_size / (half_size * 2)))
                if 0 <= local_row < patch_size and 0 <= local_col < patch_size:
                    # 绘制 3x3 响应区域而非单点，提升 CNN 捕捉能力
                    history_heatmap[max(0, local_row-1):min(patch_size, local_row+2), 
                                    max(0, local_col-1):min(patch_size, local_col+2)] = 1.0
        channels.append(history_heatmap)
        
        # 8. Candidate goal map (1 通道) - 动态生成 (带高斯模糊增强泛化)
        goal_map = np.zeros((patch_size, patch_size), dtype=np.float32)
        if candidates_abs is not None:
            for pt in candidates_abs:
                p_col, p_row = ~self.transform * (pt[0], pt[1])
                local_row = int((p_row - (row - half_size)) * (patch_size / (half_size * 2)))
                local_col = int((p_col - (col - half_size)) * (patch_size / (half_size * 2)))
                if 0 <= local_row < patch_size and 0 <= local_col < patch_size:
                    # 目标点使用更大的 5x5 响应区域，作为强空间引导
                    goal_map[max(0, local_row-2):min(patch_size, local_row+3), 
                             max(0, local_col-2):min(patch_size, local_col+3)] = 1.0
        channels.append(goal_map)
        
        # 堆叠成 (18, H, W)
        env_map = np.stack(channels, axis=0).astype(np.float32)
        
        return torch.from_numpy(env_map)
    
    def get_features_at_coords(self, coords_utm: np.ndarray) -> np.ndarray:
        """
        提取给定坐标点的 26 维环境特征 (用于对齐论文表 4.2)
        
        Args:
            coords_utm: (seq_len, 2) UTM 坐标
            
        Returns:
            features: (seq_len, 26) 特征矩阵
        """
        seq_len = coords_utm.shape[0]
        features = np.zeros((seq_len, 26), dtype=np.float32)
        
        # 1. 提取基础运动学特征 (需要计算)
        # 注意：这里只填充坐标，速度/加速度/航向在 Dataset 中计算
        features[:, 0:2] = coords_utm  # x, y
        
        # 2. 提取环境栅格特征
        for i in range(seq_len):
            col, row = ~self.transform * (coords_utm[i, 0], coords_utm[i, 1])
            col, row = int(col), int(row)
            
            # 边界检查
            if 0 <= row < self.dem.shape[0] and 0 <= col < self.dem.shape[1]:
                # 地形 (4维)
                features[i, 10] = (self.dem[row, col] - self.dem.mean()) / (self.dem.std() + 1e-6) # dem_agg
                features[i, 11] = self.slope[row, col] / 90.0 # slope_agg
                aspect_rad = np.deg2rad(self.aspect[row, col])
                features[i, 12] = np.sin(aspect_rad) # aspect_sin_agg
                features[i, 13] = np.cos(aspect_rad) # aspect_cos_agg
                
                # 地表 (11维)
                lulc_val = self.lulc[row, col]
                lulc_classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 255]
                for idx, cls in enumerate(lulc_classes):
                    if lulc_val == cls:
                        features[i, 14 + idx] = 1.0
                
                # 道路 (1维)
                if self.road[row, col] > 0:
                    features[i, 25] = 1.0
                    
        return features


# 全局缓存
_env_loaders = {}

def get_env_loader(region='bohemian_forest'):
    """获取环境数据加载器（带缓存）"""
    if region not in _env_loaders:
        _env_loaders[region] = EnvironmentDataLoader(region)
    return _env_loaders[region]
