"""
坐标系统转换工具
"""
import numpy as np
from pyproj import Transformer, CRS
from typing import Tuple, Union, List
import logging

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """坐标系统转换器"""
    
    def __init__(self, source_crs: str = "EPSG:4326", target_crs: str = "EPSG:32630"):
        """
        初始化坐标转换器
        
        Args:
            source_crs: 源坐标系，默认为WGS84 (EPSG:4326)
            target_crs: 目标坐标系，默认为UTM 30N (EPSG:32630)
        """
        self.source_crs = CRS.from_string(source_crs)
        self.target_crs = CRS.from_string(target_crs)
        
        # 创建转换器
        self.transformer = Transformer.from_crs(
            self.source_crs, 
            self.target_crs, 
            always_xy=True
        )
        
        # 创建反向转换器
        self.inverse_transformer = Transformer.from_crs(
            self.target_crs, 
            self.source_crs, 
            always_xy=True
        )
        
        logger.info(f"坐标转换器初始化: {source_crs} -> {target_crs}")
    
    def transform_point(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        转换单个点坐标
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            (x, y): 投影坐标
        """
        x, y = self.transformer.transform(lon, lat)
        return x, y
    
    def transform_points(self, lons: Union[List[float], np.ndarray], 
                        lats: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量转换点坐标
        
        Args:
            lons: 经度数组
            lats: 纬度数组
            
        Returns:
            (xs, ys): 投影坐标数组
        """
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        
        xs, ys = self.transformer.transform(lons, lats)
        return xs, ys
    
    def inverse_transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        反向转换单个点坐标
        
        Args:
            x: 投影坐标X
            y: 投影坐标Y
            
        Returns:
            (lon, lat): 地理坐标
        """
        lon, lat = self.inverse_transformer.transform(x, y)
        return lon, lat
    
    def inverse_transform_points(self, xs: Union[List[float], np.ndarray], 
                               ys: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量反向转换点坐标
        
        Args:
            xs: 投影坐标X数组
            ys: 投影坐标Y数组
            
        Returns:
            (lons, lats): 地理坐标数组
        """
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        
        lons, lats = self.inverse_transformer.transform(xs, ys)
        return lons, lats
    
    def get_utm_zone(self, lon: float) -> int:
        """
        根据经度计算UTM带号
        
        Args:
            lon: 经度
            
        Returns:
            UTM带号
        """
        return int((lon + 180) / 6) + 1
    
    def get_optimal_utm_crs(self, lon: float, lat: float) -> str:
        """
        根据经纬度获取最优的UTM坐标系
        
        Args:
            lon: 经度
            lat: 纬度
            
        Returns:
            UTM坐标系EPSG代码
        """
        utm_zone = self.get_utm_zone(lon)
        
        # 判断南北半球
        if lat >= 0:
            # 北半球
            epsg_code = f"EPSG:326{utm_zone:02d}"
        else:
            # 南半球
            epsg_code = f"EPSG:327{utm_zone:02d}"
        
        return epsg_code
    
    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        计算两点间的欧氏距离（投影坐标系下）
        
        Args:
            x1, y1: 第一个点的投影坐标
            x2, y2: 第二个点的投影坐标
            
        Returns:
            距离（米）
        """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def calculate_bearing(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        计算两点间的方位角（投影坐标系下）
        
        Args:
            x1, y1: 起点投影坐标
            x2, y2: 终点投影坐标
            
        Returns:
            方位角（度，北为0度，顺时针为正）
        """
        dx = x2 - x1
        dy = y2 - y1
        
        # 计算角度（弧度）
        angle_rad = np.arctan2(dx, dy)
        
        # 转换为度数
        angle_deg = np.degrees(angle_rad)
        
        # 确保角度在0-360度范围内
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg


def create_transformer_for_region(bounds: Tuple[float, float, float, float]) -> CoordinateTransformer:
    """
    为指定区域创建最优的坐标转换器
    
    Args:
        bounds: 区域边界 (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        坐标转换器
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # 使用区域中心点确定最优UTM坐标系
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    
    transformer = CoordinateTransformer()
    optimal_crs = transformer.get_optimal_utm_crs(center_lon, center_lat)
    
    logger.info(f"为区域 {bounds} 选择坐标系: {optimal_crs}")
    
    return CoordinateTransformer(target_crs=optimal_crs)
