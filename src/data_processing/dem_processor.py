"""
数字高程模型(DEM)数据处理器
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
from scipy import ndimage
from skimage.filters import gaussian

from ..utils.coordinate_transform import CoordinateTransformer
from ..utils.config_loader import config

logger = logging.getLogger(__name__)


class DEMProcessor:
    """数字高程模型处理器"""
    
    def __init__(self, transformer: CoordinateTransformer = None):
        """
        初始化DEM处理器
        
        Args:
            transformer: 坐标转换器
        """
        self.transformer = transformer or CoordinateTransformer()
        self.data_config = config.get_data_config()
        self.resolution = config.get('environment.resolution', 30)
    
    def load_dem(self, dem_path: str) -> Tuple[np.ndarray, rasterio.Affine, str]:
        """
        加载DEM数据
        
        Args:
            dem_path: DEM文件路径
            
        Returns:
            (elevation_data, transform, crs): 高程数据、仿射变换、坐标系
        """
        try:
            with rasterio.open(dem_path) as src:
                elevation = src.read(1)
                transform = src.transform
                crs = src.crs
                
                logger.info(f"加载DEM: {dem_path}, 尺寸: {elevation.shape}, CRS: {crs}")
                return elevation, transform, str(crs)
                
        except Exception as e:
            logger.error(f"加载DEM失败: {e}")
            raise
    
    def reproject_dem(self, elevation: np.ndarray, src_transform: rasterio.Affine,
                     src_crs: str, bounds: Tuple[float, float, float, float]) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        重投影DEM到目标坐标系和分辨率
        
        Args:
            elevation: 原始高程数据
            src_transform: 原始仿射变换
            src_crs: 原始坐标系
            bounds: 目标边界 (min_x, min_y, max_x, max_y) - 投影坐标
            
        Returns:
            (reprojected_elevation, new_transform): 重投影后的数据和变换
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
        dst_elevation = np.zeros((height, width), dtype=np.float32)
        
        # 重投影
        reproject(
            source=elevation,
            destination=dst_elevation,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=self.transformer.target_crs.to_string(),
            resampling=Resampling.bilinear
        )
        
        logger.info(f"DEM重投影完成: {dst_elevation.shape}, 分辨率: {self.resolution}m")
        return dst_elevation, dst_transform
    
    def calculate_slope(self, elevation: np.ndarray, resolution: float = None) -> np.ndarray:
        """
        计算坡度
        
        Args:
            elevation: 高程数据
            resolution: 像元分辨率（米）
            
        Returns:
            坡度数组（度）
        """
        if resolution is None:
            resolution = self.resolution
        
        # 计算梯度
        dy, dx = np.gradient(elevation, resolution)
        
        # 计算坡度（弧度）
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        
        # 转换为度
        slope_deg = np.degrees(slope_rad)
        
        # 处理无效值
        slope_deg = np.nan_to_num(slope_deg, nan=0.0, posinf=90.0, neginf=0.0)
        
        logger.info(f"坡度计算完成: 最大坡度 {np.max(slope_deg):.2f}°")
        return slope_deg.astype(np.float32)
    
    def calculate_aspect(self, elevation: np.ndarray, resolution: float = None) -> np.ndarray:
        """
        计算坡向
        
        Args:
            elevation: 高程数据
            resolution: 像元分辨率（米）
            
        Returns:
            坡向数组（度，北为0°，顺时针）
        """
        if resolution is None:
            resolution = self.resolution
        
        # 计算梯度
        dy, dx = np.gradient(elevation, resolution)
        
        # 计算坡向（弧度）
        aspect_rad = np.arctan2(-dx, dy)
        
        # 转换为度并调整范围到0-360
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (aspect_deg + 360) % 360
        
        # 处理平坦区域（坡度接近0的地方）
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        flat_mask = slope_rad < np.radians(1.0)  # 坡度小于1度视为平坦
        aspect_deg[flat_mask] = -1  # 平坦区域坡向设为-1
        
        logger.info(f"坡向计算完成")
        return aspect_deg.astype(np.float32)
    
    def calculate_curvature(self, elevation: np.ndarray, resolution: float = None) -> Dict[str, np.ndarray]:
        """
        计算曲率
        
        Args:
            elevation: 高程数据
            resolution: 像元分辨率（米）
            
        Returns:
            包含不同曲率的字典
        """
        if resolution is None:
            resolution = self.resolution
        
        # 计算一阶导数
        fy, fx = np.gradient(elevation, resolution)
        
        # 计算二阶导数
        fyy, fyx = np.gradient(fy, resolution)
        fxy, fxx = np.gradient(fx, resolution)
        
        # 平面曲率 (Plan Curvature)
        denominator = (fx**2 + fy**2)**(3/2)
        plan_curvature = np.where(
            denominator != 0,
            (fxx * fy**2 - 2 * fxy * fx * fy + fyy * fx**2) / denominator,
            0
        )
        
        # 剖面曲率 (Profile Curvature)  
        denominator = (fx**2 + fy**2)**(3/2)
        profile_curvature = np.where(
            denominator != 0,
            (fxx * fx**2 + 2 * fxy * fx * fy + fyy * fy**2) / denominator,
            0
        )
        
        # 总曲率 (Total Curvature)
        total_curvature = np.sqrt(plan_curvature**2 + profile_curvature**2)
        
        # 处理无效值
        for curvature in [plan_curvature, profile_curvature, total_curvature]:
            curvature = np.nan_to_num(curvature, nan=0.0)
        
        logger.info("曲率计算完成")
        
        return {
            'plan_curvature': plan_curvature.astype(np.float32),
            'profile_curvature': profile_curvature.astype(np.float32),
            'total_curvature': total_curvature.astype(np.float32)
        }
    
    def calculate_roughness(self, elevation: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        计算地形粗糙度
        
        Args:
            elevation: 高程数据
            window_size: 窗口大小
            
        Returns:
            粗糙度数组
        """
        # 使用标准差作为粗糙度指标
        roughness = ndimage.generic_filter(
            elevation, 
            np.std, 
            size=window_size,
            mode='constant',
            cval=0
        )
        
        logger.info(f"粗糙度计算完成: 窗口大小 {window_size}")
        return roughness.astype(np.float32)
    
    def calculate_tpi(self, elevation: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        计算地形位置指数 (Topographic Position Index, TPI)
        
        Args:
            elevation: 高程数据
            radius: 分析半径（像元数）
            
        Returns:
            TPI数组
        """
        # 创建圆形核
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        kernel = x**2 + y**2 <= radius**2
        
        # 计算邻域平均高程
        mean_elevation = ndimage.convolve(
            elevation.astype(np.float64), 
            kernel.astype(np.float64) / np.sum(kernel),
            mode='constant',
            cval=0
        )
        
        # TPI = 中心点高程 - 邻域平均高程
        tpi = elevation - mean_elevation
        
        logger.info(f"TPI计算完成: 半径 {radius} 像元")
        return tpi.astype(np.float32)
    
    def create_slope_mask(self, slope: np.ndarray, max_slope: float = 30.0) -> np.ndarray:
        """
        创建坡度掩膜
        
        Args:
            slope: 坡度数组（度）
            max_slope: 最大可通行坡度（度）
            
        Returns:
            掩膜数组（1=可通行，0=不可通行）
        """
        slope_mask = (slope <= max_slope).astype(np.uint8)
        
        passable_ratio = np.sum(slope_mask) / slope_mask.size
        logger.info(f"坡度掩膜创建完成: 可通行区域占比 {passable_ratio:.2%}")
        
        return slope_mask
    
    def smooth_elevation(self, elevation: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        平滑高程数据
        
        Args:
            elevation: 高程数据
            sigma: 高斯核标准差
            
        Returns:
            平滑后的高程数据
        """
        smoothed = gaussian(elevation, sigma=sigma, preserve_range=True)
        
        logger.info(f"高程数据平滑完成: sigma={sigma}")
        return smoothed.astype(np.float32)
    
    def process_dem_for_region(self, dem_path: str, bounds: Tuple[float, float, float, float],
                              output_dir: str = None) -> Dict[str, np.ndarray]:
        """
        处理指定区域的DEM数据
        
        Args:
            dem_path: DEM文件路径
            bounds: 边界范围 (min_x, min_y, max_x, max_y) - 投影坐标
            output_dir: 输出目录
            
        Returns:
            包含各种地形参数的字典
        """
        # 加载DEM
        elevation, src_transform, src_crs = self.load_dem(dem_path)
        
        # 重投影到目标坐标系
        elevation, dst_transform = self.reproject_dem(elevation, src_transform, src_crs, bounds)
        
        # 计算地形参数
        slope = self.calculate_slope(elevation)
        aspect = self.calculate_aspect(elevation)
        curvatures = self.calculate_curvature(elevation)
        roughness = self.calculate_roughness(elevation)
        tpi = self.calculate_tpi(elevation)
        
        # 创建坡度掩膜
        max_slope = config.get('environment.max_slope', 30.0)
        slope_mask = self.create_slope_mask(slope, max_slope)
        
        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path / 'elevation.npy', elevation)
            np.save(output_path / 'slope.npy', slope)
            np.save(output_path / 'aspect.npy', aspect)
            np.save(output_path / 'roughness.npy', roughness)
            np.save(output_path / 'tpi.npy', tpi)
            np.save(output_path / 'slope_mask.npy', slope_mask)
            
            for name, data in curvatures.items():
                np.save(output_path / f'{name}.npy', data)
        
        result = {
            'elevation': elevation,
            'slope': slope,
            'aspect': aspect,
            'roughness': roughness,
            'tpi': tpi,
            'slope_mask': slope_mask,
            'transform': dst_transform
        }
        result.update(curvatures)
        
        logger.info("DEM处理完成")
        return result


def process_region_dem(region_name: str, bounds: Tuple[float, float, float, float],
                      dem_path: str = None, output_dir: str = None) -> Dict[str, np.ndarray]:
    """
    处理指定区域的DEM数据
    
    Args:
        region_name: 区域名称
        bounds: 边界范围 (min_x, min_y, max_x, max_y) - 投影坐标
        dem_path: DEM文件路径
        output_dir: 输出目录
        
    Returns:
        地形参数字典
    """
    if output_dir is None:
        output_dir = Path(config.get('data.output_dir', 'data/processed')) / region_name
    
    if dem_path is None:
        dem_dir = Path(config.get('data.dem_dir', 'data/dem'))
        dem_path = dem_dir / f"{region_name}_dem.tif"
    
    processor = DEMProcessor()
    
    try:
        result = processor.process_dem_for_region(dem_path, bounds, output_dir)
        logger.info(f"区域 {region_name} DEM数据处理完成")
        return result
        
    except Exception as e:
        logger.error(f"处理区域 {region_name} DEM数据失败: {e}")
        raise
