"""
统一环境栅格到 UTM30N EPSG:32630 + 30m 同网格

按论文要求：
- 目标坐标系：EPSG:32630 (UTM Zone 30N)
- 目标分辨率：30m
- 所有栅格（DEM/Slope/Aspect/LULC）对齐到同一网格
"""
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum

from config import cfg, get_path

logger = logging.getLogger(__name__)

TARGET_CRS = "EPSG:32630"  # UTM Zone 30N
TARGET_RESOLUTION = 30  # meters


def get_common_bounds_utm(file_paths: list) -> Tuple[float, float, float, float]:
    """
    计算所有输入文件在目标UTM坐标系下的公共范围
    
    Returns:
        (left, bottom, right, top) in UTM coordinates
    """
    all_bounds = []
    
    for fp in file_paths:
        with rasterio.open(fp) as src:
            # 计算在目标CRS下的bounds
            transform, width, height = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height, *src.bounds
            )
            
            # 从transform和尺寸计算bounds
            left = transform.c
            top = transform.f
            right = left + width * transform.a
            bottom = top + height * transform.e
            
            all_bounds.append((left, bottom, right, top))
    
    # 计算公共范围（交集）
    left = max(b[0] for b in all_bounds)
    bottom = max(b[1] for b in all_bounds)
    right = min(b[2] for b in all_bounds)
    top = min(b[3] for b in all_bounds)
    
    logger.info(f"公共UTM范围: left={left:.2f}, bottom={bottom:.2f}, right={right:.2f}, top={top:.2f}")
    logger.info(f"范围大小: {(right-left)/1000:.2f} km × {(top-bottom)/1000:.2f} km")
    
    return left, bottom, right, top


def align_to_grid(bounds: Tuple[float, float, float, float], resolution: float) -> Tuple[rasterio.Affine, int, int]:
    """
    将bounds对齐到规则网格
    
    Args:
        bounds: (left, bottom, right, top)
        resolution: 分辨率（米）
    
    Returns:
        (transform, width, height)
    """
    left, bottom, right, top = bounds
    
    # 对齐到resolution的整数倍
    left = np.floor(left / resolution) * resolution
    bottom = np.floor(bottom / resolution) * resolution
    right = np.ceil(right / resolution) * resolution
    top = np.ceil(top / resolution) * resolution
    
    width = int((right - left) / resolution)
    height = int((top - bottom) / resolution)
    
    # 创建仿射变换矩阵
    transform = rasterio.Affine(
        resolution, 0.0, left,
        0.0, -resolution, top
    )
    
    logger.info(f"对齐后网格: {width} × {height} @ {resolution}m")
    
    return transform, width, height


def reproject_raster(
    input_path: Path,
    output_path: Path,
    target_transform: rasterio.Affine,
    target_width: int,
    target_height: int,
    resampling_method: ResamplingEnum = ResamplingEnum.bilinear
) -> None:
    """
    重投影单个栅格到目标网格
    
    Args:
        input_path: 输入文件
        output_path: 输出文件
        target_transform: 目标仿射变换
        target_width: 目标宽度
        target_height: 目标高度
        resampling_method: 重采样方法
    """
    with rasterio.open(input_path) as src:
        # 准备输出profile
        profile = src.profile.copy()
        profile.update({
            'crs': TARGET_CRS,
            'transform': target_transform,
            'width': target_width,
            'height': target_height,
            'compress': 'lzw'
        })
        
        # 创建输出数组
        dst_array = np.empty((src.count, target_height, target_width), dtype=profile['dtype'])
        
        # 逐波段重投影
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=dst_array[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=TARGET_CRS,
                resampling=resampling_method
            )
        
        # 写入输出
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_array)
        
        logger.info(f"✓ 重投影完成: {output_path.name}")


def reproject_all_to_utm(region: str = 'scottish_highlands') -> dict:
    """
    将所有环境栅格重投影到统一UTM网格
    
    Args:
        region: 区域名称
    
    Returns:
        重投影后文件路径字典
    """
    logger.info("=" * 60)
    logger.info(f"开始重投影 {region} 环境栅格到 UTM30N")
    logger.info("=" * 60)
    
    # 输入路径
    merged_dir = Path(cfg.get('paths.processed.merged_gee')) / region
    
    input_files = {
        'dem': merged_dir / 'dem_merged.tif',
        'slope': merged_dir / 'slope_merged.tif',
        'aspect': merged_dir / 'aspect_merged.tif',
        'lulc': merged_dir / 'lulc_merged.tif'  # 使用原始合并的，不用resampled版本
    }
    
    # 检查文件存在
    for name, path in input_files.items():
        if not path.exists():
            logger.error(f"输入文件不存在: {path}")
            raise FileNotFoundError(f"Missing {name}: {path}")
    
    # 计算公共范围
    logger.info("\n步骤1: 计算公共UTM范围...")
    common_bounds = get_common_bounds_utm(list(input_files.values()))
    
    # 对齐到规则网格
    logger.info("\n步骤2: 对齐到30m网格...")
    target_transform, target_width, target_height = align_to_grid(
        common_bounds, TARGET_RESOLUTION
    )
    
    # 输出路径
    utm_dir = get_path('paths.processed.utm_grid') / region
    utm_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # 重投影各个栅格
    logger.info("\n步骤3: 重投影各栅格...")
    
    # DEM - 双线性插值
    logger.info("  [1/4] DEM...")
    output_files['dem'] = utm_dir / 'dem_utm.tif'
    reproject_raster(
        input_files['dem'],
        output_files['dem'],
        target_transform,
        target_width,
        target_height,
        ResamplingEnum.bilinear
    )
    
    # Slope - 双线性插值
    logger.info("  [2/4] Slope...")
    output_files['slope'] = utm_dir / 'slope_utm.tif'
    reproject_raster(
        input_files['slope'],
        output_files['slope'],
        target_transform,
        target_width,
        target_height,
        ResamplingEnum.bilinear
    )
    
    # Aspect - 双线性插值
    logger.info("  [3/4] Aspect...")
    output_files['aspect'] = utm_dir / 'aspect_utm.tif'
    reproject_raster(
        input_files['aspect'],
        output_files['aspect'],
        target_transform,
        target_width,
        target_height,
        ResamplingEnum.bilinear
    )
    
    # LULC - 最近邻（保持类别完整性）
    logger.info("  [4/4] LULC...")
    output_files['lulc'] = utm_dir / 'lulc_utm.tif'
    reproject_raster(
        input_files['lulc'],
        output_files['lulc'],
        target_transform,
        target_width,
        target_height,
        ResamplingEnum.nearest
    )
    
    # 验证结果
    logger.info("\n步骤4: 验证结果...")
    for name, path in output_files.items():
        with rasterio.open(path) as src:
            logger.info(f"  {name.upper()}:")
            logger.info(f"    CRS: {src.crs}")
            logger.info(f"    尺寸: {src.width} × {src.height}")
            logger.info(f"    分辨率: {src.res[0]:.2f}m × {src.res[1]:.2f}m")
            logger.info(f"    范围: {src.bounds}")
            
            data = src.read(1)
            logger.info(f"    值域: {data.min():.2f} ~ {data.max():.2f}")
            if name == 'lulc':
                unique = np.unique(data)
                logger.info(f"    LULC类别: {unique.tolist()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ UTM重投影完成")
    logger.info("=" * 60)
    
    return {k: str(v) for k, v in output_files.items()}


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    results = reproject_all_to_utm('scottish_highlands')
    
    print("\n输出文件:")
    for name, path in results.items():
        print(f"  {name}: {path}")
