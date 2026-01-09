"""
从DEM生成slope和aspect数据
"""
import numpy as np
import rasterio
from rasterio.transform import Affine
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_slope_aspect(dem_path, slope_path, aspect_path):
    """从DEM计算slope和aspect"""
    logger.info(f"读取DEM: {dem_path}")
    
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        profile = src.profile.copy()
        transform = src.transform
        
    # 获取分辨率
    pixel_size = transform.a  # 30m
    
    logger.info(f"DEM尺寸: {dem.shape}, 分辨率: {pixel_size}m")
    
    # 计算梯度
    dy, dx = np.gradient(dem, pixel_size)
    
    # 计算坡度（度）
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
    
    # 计算坡向（度，0-360）
    aspect = np.arctan2(-dx, dy) * 180 / np.pi
    aspect = (aspect + 360) % 360
    
    # 处理平地（坡度为0的地方，坡向设为-1）
    aspect[slope == 0] = -1
    
    logger.info(f"Slope范围: {slope.min():.2f}° ~ {slope.max():.2f}°")
    logger.info(f"Aspect范围: {aspect.min():.2f}° ~ {aspect.max():.2f}°")
    
    # 保存slope
    logger.info(f"保存slope: {slope_path}")
    profile.update(dtype=rasterio.float32, compress='lzw')
    with rasterio.open(slope_path, 'w', **profile) as dst:
        dst.write(slope.astype(np.float32), 1)
    
    # 保存aspect
    logger.info(f"保存aspect: {aspect_path}")
    with rasterio.open(aspect_path, 'w', **profile) as dst:
        dst.write(aspect.astype(np.float32), 1)
    
    logger.info("✓ 完成")

def main():
    region = 'bohemian_forest'
    utm_dir = Path(f'/home/zmc/文档/programwork/data/processed/utm_grid/{region}')
    
    dem_path = utm_dir / 'dem_utm.tif'
    slope_path = utm_dir / 'slope_utm.tif'
    aspect_path = utm_dir / 'aspect_utm.tif'
    
    logger.info("=" * 60)
    logger.info(f"生成 {region} 的 slope 和 aspect 数据")
    logger.info("=" * 60)
    
    calculate_slope_aspect(dem_path, slope_path, aspect_path)
    
    logger.info("\n✅ 所有数据生成完成！")

if __name__ == '__main__':
    main()
