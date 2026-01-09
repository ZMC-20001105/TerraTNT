"""
土地利用土地覆盖(LULC)数据处理器
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from collections import Counter

from ..utils.coordinate_transform import CoordinateTransformer
from ..utils.config_loader import config

logger = logging.getLogger(__name__)


class LULCProcessor:
    """土地利用土地覆盖处理器"""
    
    def __init__(self, transformer: CoordinateTransformer = None):
        """
        初始化LULC处理器
        
        Args:
            transformer: 坐标转换器
        """
        self.transformer = transformer or CoordinateTransformer()
        self.data_config = config.get_data_config()
        self.resolution = config.get('environment.resolution', 30)
        self.lulc_classes = config.get_lulc_classes()
    
    def load_lulc(self, lulc_path: str) -> Tuple[np.ndarray, rasterio.Affine, str]:
        """
        加载LULC数据
        
        Args:
            lulc_path: LULC文件路径
            
        Returns:
            (lulc_data, transform, crs): LULC数据、仿射变换、坐标系
        """
        try:
            with rasterio.open(lulc_path) as src:
                lulc = src.read(1)
                transform = src.transform
                crs = src.crs
                
                logger.info(f"加载LULC: {lulc_path}, 尺寸: {lulc.shape}, CRS: {crs}")
                return lulc, transform, str(crs)
                
        except Exception as e:
            logger.error(f"加载LULC失败: {e}")
            raise
    
    def reproject_lulc(self, lulc: np.ndarray, src_transform: rasterio.Affine,
                      src_crs: str, bounds: Tuple[float, float, float, float]) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        重投影LULC到目标坐标系和分辨率
        
        Args:
            lulc: 原始LULC数据
            src_transform: 原始仿射变换
            src_crs: 原始坐标系
            bounds: 目标边界 (min_x, min_y, max_x, max_y) - 投影坐标
            
        Returns:
            (reprojected_lulc, new_transform): 重投影后的数据和变换
        """
        min_x, min_y, max_x, max_y = bounds
        
        # 计算目标尺寸
        width = int((max_x - min_x) / self.resolution)
        height = int((max_y - min_y) / self.resolution)
        
        # 计算目标变换
        dst_transform = rasterio.transform.from_bounds(
            min_x, min_y, max_x, max_y, width, height
        )
        
        # 创建目标数组
        dst_lulc = np.zeros((height, width), dtype=lulc.dtype)
        
        # 重投影（使用最近邻插值保持分类值）
        reproject(
            source=lulc,
            destination=dst_lulc,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=self.transformer.target_crs.to_string(),
            resampling=Resampling.nearest
        )
        
        logger.info(f"LULC重投影完成: {dst_lulc.shape}, 分辨率: {self.resolution}m")
        return dst_lulc, dst_transform
    
    def analyze_lulc_distribution(self, lulc: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        分析LULC分布
        
        Args:
            lulc: LULC数据
            
        Returns:
            各类别的统计信息
        """
        # 统计各类别像元数量
        class_counts = Counter(lulc.flatten())
        total_pixels = lulc.size
        
        distribution = {}
        for class_id, count in class_counts.items():
            if class_id in self.lulc_classes:
                class_info = self.lulc_classes[class_id]
                distribution[class_id] = {
                    'name': class_info['name'],
                    'count': count,
                    'percentage': (count / total_pixels) * 100,
                    'area_km2': (count * self.resolution**2) / 1e6,
                    'passable': class_info['passable'],
                    'cost': class_info.get('cost', None)
                }
        
        # 按面积排序
        distribution = dict(sorted(distribution.items(), 
                                 key=lambda x: x[1]['count'], reverse=True))
        
        logger.info(f"LULC分布分析完成: {len(distribution)} 个类别")
        for class_id, info in list(distribution.items())[:5]:  # 显示前5个主要类别
            logger.info(f"  {info['name']}: {info['percentage']:.1f}% ({info['area_km2']:.1f} km²)")
        
        return distribution
    
    def create_passability_mask(self, lulc: np.ndarray) -> np.ndarray:
        """
        创建可通行性掩膜
        
        Args:
            lulc: LULC数据
            
        Returns:
            掩膜数组（1=可通行，0=不可通行）
        """
        passability_mask = np.zeros_like(lulc, dtype=np.uint8)
        
        for class_id, class_info in self.lulc_classes.items():
            if class_info['passable']:
                passability_mask[lulc == class_id] = 1
        
        passable_ratio = np.sum(passability_mask) / passability_mask.size
        logger.info(f"可通行性掩膜创建完成: 可通行区域占比 {passable_ratio:.2%}")
        
        return passability_mask
    
    def create_cost_map(self, lulc: np.ndarray) -> np.ndarray:
        """
        创建代价地图
        
        Args:
            lulc: LULC数据
            
        Returns:
            代价地图（值越大通行代价越高）
        """
        cost_map = np.full_like(lulc, fill_value=np.inf, dtype=np.float32)
        
        for class_id, class_info in self.lulc_classes.items():
            if class_info['passable'] and class_info['cost'] is not None:
                cost_map[lulc == class_id] = class_info['cost']
        
        # 不可通行区域保持无穷大
        logger.info(f"代价地图创建完成: 代价范围 {np.min(cost_map[cost_map != np.inf]):.2f} - {np.max(cost_map[cost_map != np.inf]):.2f}")
        
        return cost_map
    
    def create_one_hot_encoding(self, lulc: np.ndarray) -> np.ndarray:
        """
        创建LULC的独热编码
        
        Args:
            lulc: LULC数据
            
        Returns:
            独热编码数组 (height, width, num_classes)
        """
        height, width = lulc.shape
        class_ids = list(self.lulc_classes.keys())
        num_classes = len(class_ids)
        
        one_hot = np.zeros((height, width, num_classes), dtype=np.float32)
        
        for i, class_id in enumerate(class_ids):
            one_hot[:, :, i] = (lulc == class_id).astype(np.float32)
        
        logger.info(f"独热编码创建完成: {one_hot.shape}")
        return one_hot
    
    def create_multi_channel_features(self, lulc: np.ndarray) -> np.ndarray:
        """
        创建多通道LULC特征
        
        Args:
            lulc: LULC数据
            
        Returns:
            多通道特征数组 (num_channels, height, width)
        """
        height, width = lulc.shape
        
        # 通道1: 可通行性
        passability = self.create_passability_mask(lulc).astype(np.float32)
        
        # 通道2: 代价值（归一化）
        cost_map = self.create_cost_map(lulc)
        cost_normalized = np.where(
            cost_map != np.inf,
            (cost_map - np.min(cost_map[cost_map != np.inf])) / 
            (np.max(cost_map[cost_map != np.inf]) - np.min(cost_map[cost_map != np.inf])),
            1.0  # 不可通行区域设为最大值
        )
        
        # 通道3-N: 主要类别的二值掩膜
        major_classes = [10, 20, 30, 40, 60, 80, 90]  # 选择主要的LULC类别
        class_channels = []
        
        for class_id in major_classes:
            if class_id in self.lulc_classes:
                class_mask = (lulc == class_id).astype(np.float32)
                class_channels.append(class_mask)
        
        # 组合所有通道
        all_channels = [passability, cost_normalized] + class_channels
        features = np.stack(all_channels, axis=0)
        
        logger.info(f"多通道LULC特征创建完成: {features.shape}")
        return features
    
    def calculate_vegetation_density(self, lulc: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        计算植被密度
        
        Args:
            lulc: LULC数据
            window_size: 计算窗口大小
            
        Returns:
            植被密度数组 (0-1)
        """
        from scipy import ndimage
        
        # 定义植被类别
        vegetation_classes = [20, 30, 40, 70]  # 森林、草地、灌木地、苔原
        
        # 创建植被掩膜
        vegetation_mask = np.zeros_like(lulc, dtype=np.float32)
        for class_id in vegetation_classes:
            vegetation_mask[lulc == class_id] = 1.0
        
        # 计算局部植被密度
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        vegetation_density = ndimage.convolve(vegetation_mask, kernel, mode='constant')
        
        logger.info(f"植被密度计算完成: 窗口大小 {window_size}")
        return vegetation_density
    
    def calculate_urbanization_level(self, lulc: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        计算城市化水平
        
        Args:
            lulc: LULC数据
            window_size: 计算窗口大小
            
        Returns:
            城市化水平数组 (0-1)
        """
        from scipy import ndimage
        
        # 定义人工建设类别
        urban_classes = [80]  # 人造地表
        
        # 创建城市掩膜
        urban_mask = np.zeros_like(lulc, dtype=np.float32)
        for class_id in urban_classes:
            urban_mask[lulc == class_id] = 1.0
        
        # 计算局部城市化水平
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        urbanization_level = ndimage.convolve(urban_mask, kernel, mode='constant')
        
        logger.info(f"城市化水平计算完成: 窗口大小 {window_size}")
        return urbanization_level
    
    def process_lulc_for_region(self, lulc_path: str, bounds: Tuple[float, float, float, float],
                               output_dir: str = None) -> Dict[str, np.ndarray]:
        """
        处理指定区域的LULC数据
        
        Args:
            lulc_path: LULC文件路径
            bounds: 边界范围 (min_x, min_y, max_x, max_y) - 投影坐标
            output_dir: 输出目录
            
        Returns:
            包含各种LULC特征的字典
        """
        # 加载LULC
        lulc, src_transform, src_crs = self.load_lulc(lulc_path)
        
        # 重投影到目标坐标系
        lulc, dst_transform = self.reproject_lulc(lulc, src_transform, src_crs, bounds)
        
        # 分析分布
        distribution = self.analyze_lulc_distribution(lulc)
        
        # 创建各种特征
        passability_mask = self.create_passability_mask(lulc)
        cost_map = self.create_cost_map(lulc)
        multi_channel_features = self.create_multi_channel_features(lulc)
        vegetation_density = self.calculate_vegetation_density(lulc)
        urbanization_level = self.calculate_urbanization_level(lulc)
        
        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path / 'lulc.npy', lulc)
            np.save(output_path / 'passability_mask.npy', passability_mask)
            np.save(output_path / 'cost_map.npy', cost_map)
            np.save(output_path / 'lulc_features.npy', multi_channel_features)
            np.save(output_path / 'vegetation_density.npy', vegetation_density)
            np.save(output_path / 'urbanization_level.npy', urbanization_level)
            
            # 保存分布统计
            import json
            with open(output_path / 'lulc_distribution.json', 'w', encoding='utf-8') as f:
                json.dump(distribution, f, ensure_ascii=False, indent=2)
        
        result = {
            'lulc': lulc,
            'passability_mask': passability_mask,
            'cost_map': cost_map,
            'lulc_features': multi_channel_features,
            'vegetation_density': vegetation_density,
            'urbanization_level': urbanization_level,
            'distribution': distribution,
            'transform': dst_transform
        }
        
        logger.info("LULC处理完成")
        return result


def process_region_lulc(region_name: str, bounds: Tuple[float, float, float, float],
                       lulc_path: str = None, output_dir: str = None) -> Dict[str, np.ndarray]:
    """
    处理指定区域的LULC数据
    
    Args:
        region_name: 区域名称
        bounds: 边界范围 (min_x, min_y, max_x, max_y) - 投影坐标
        lulc_path: LULC文件路径
        output_dir: 输出目录
        
    Returns:
        LULC特征字典
    """
    if output_dir is None:
        output_dir = Path(config.get('data.output_dir', 'data/processed')) / region_name
    
    if lulc_path is None:
        lulc_dir = Path(config.get('data.lulc_dir', 'data/lulc'))
        lulc_path = lulc_dir / f"{region_name}_lulc.tif"
    
    processor = LULCProcessor()
    
    try:
        result = processor.process_lulc_for_region(lulc_path, bounds, output_dir)
        logger.info(f"区域 {region_name} LULC数据处理完成")
        return result
        
    except Exception as e:
        logger.error(f"处理区域 {region_name} LULC数据失败: {e}")
        raise
