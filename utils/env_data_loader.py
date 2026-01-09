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
            
            print(f"✓ 成功加载 {self.region} 环境数据")
            print(f"  - DEM shape: {self.dem.shape}")
            print(f"  - Slope shape: {self.slope.shape}")
            print(f"  - Aspect shape: {self.aspect.shape}")
            print(f"  - LULC shape: {self.lulc.shape}")
            
        except Exception as e:
            raise RuntimeError(f"加载环境数据失败: {e}")
    
    def extract_patch(self, center_utm: Tuple[float, float], 
                     patch_size: int = 128, 
                     resolution: float = 30.0) -> torch.Tensor:
        """
        提取以给定 UTM 坐标为中心的环境数据块
        
        Args:
            center_utm: (easting, northing) UTM 坐标
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
        
        # 调整大小到 patch_size
        from scipy.ndimage import zoom
        if dem_patch.shape[0] != patch_size or dem_patch.shape[1] != patch_size:
            zoom_factor = (patch_size / dem_patch.shape[0], patch_size / dem_patch.shape[1])
            dem_patch = zoom(dem_patch, zoom_factor, order=1)
            slope_patch = zoom(slope_patch, zoom_factor, order=1)
            aspect_patch = zoom(aspect_patch, zoom_factor, order=1)
            lulc_patch = zoom(lulc_patch, zoom_factor, order=0)  # 最近邻插值
        
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
        
        # 6. Road (1 通道) - 暂时用 LULC=80 (人工表面) 近似
        road = (lulc_patch == 80).astype(np.float32)
        channels.append(road)
        
        # 7. History heatmap (1 通道) - 训练时会动态添加，这里先填充0
        history_heatmap = np.zeros_like(dem_patch)
        channels.append(history_heatmap)
        
        # 8. Candidate goal map (1 通道) - 训练时会动态添加，这里先填充0
        goal_map = np.zeros_like(dem_patch)
        channels.append(goal_map)
        
        # 堆叠成 (18, H, W)
        env_map = np.stack(channels, axis=0).astype(np.float32)
        
        return torch.from_numpy(env_map)
    
    def get_batch_env_maps(self, positions: np.ndarray, 
                          patch_size: int = 128) -> torch.Tensor:
        """
        批量提取环境数据
        
        Args:
            positions: (batch_size, 2) UTM 坐标数组
            patch_size: 块大小
        
        Returns:
            env_maps: (batch_size, 18, patch_size, patch_size)
        """
        batch_size = positions.shape[0]
        env_maps = []
        
        for i in range(batch_size):
            try:
                env_map = self.extract_patch(
                    center_utm=(positions[i, 0], positions[i, 1]),
                    patch_size=patch_size
                )
                env_maps.append(env_map)
            except Exception as e:
                warnings.warn(f"提取环境数据失败 (位置 {i}): {e}，使用零填充")
                env_maps.append(torch.zeros(18, patch_size, patch_size))
        
        return torch.stack(env_maps, dim=0)


# 全局缓存
_env_loaders = {}

def get_env_loader(region='bohemian_forest'):
    """获取环境数据加载器（带缓存）"""
    if region not in _env_loaders:
        _env_loaders[region] = EnvironmentDataLoader(region)
    return _env_loaders[region]
