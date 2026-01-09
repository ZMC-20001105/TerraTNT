"""
处理Bohemian Forest区域数据

将下载的DEM和LULC tiles合并并投影到UTM
"""
import logging
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

sys.path.append(str(Path(__file__).parent.parent))

from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_tiles(tile_dir: Path, output_path: Path, data_type: str):
    """
    合并GEE下载的tiles
    
    Args:
        tile_dir: tiles目录
        output_path: 输出文件路径
        data_type: 数据类型（dem或lulc）
    """
    logger.info(f"合并 {data_type.upper()} tiles...")
    logger.info(f"  输入目录: {tile_dir}")
    
    # 查找所有tile文件
    tile_files = sorted(tile_dir.glob('tile_*.tif'))
    if not tile_files:
        raise FileNotFoundError(f"未找到tile文件: {tile_dir}/tile_*.tif")
    
    logger.info(f"  找到 {len(tile_files)} 个tiles")
    
    # 打开所有tiles
    src_files = [rasterio.open(f) for f in tile_files]
    
    # 合并
    logger.info("  正在合并...")
    mosaic, out_trans = merge(src_files)
    
    # 获取元数据
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # 关闭文件
    for src in src_files:
        src.close()
    
    logger.info(f"  ✓ 合并完成: {output_path}")
    logger.info(f"  尺寸: {mosaic.shape[1]} x {mosaic.shape[2]}")
    
    return output_path


def reproject_to_utm(input_path: Path, output_path: Path, target_epsg: int):
    """
    投影到UTM坐标系
    
    Args:
        input_path: 输入文件（WGS84）
        output_path: 输出文件（UTM）
        target_epsg: 目标EPSG代码
    """
    logger.info(f"投影到UTM (EPSG:{target_epsg})...")
    logger.info(f"  输入: {input_path}")
    
    with rasterio.open(input_path) as src:
        # 计算目标变换
        transform, width, height = calculate_default_transform(
            src.crs, f'EPSG:{target_epsg}',
            src.width, src.height,
            *src.bounds,
            resolution=30  # 30m分辨率
        )
        
        # 更新元数据
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': f'EPSG:{target_epsg}',
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw'
        })
        
        # 投影
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=f'EPSG:{target_epsg}',
                    resampling=Resampling.bilinear
                )
    
    logger.info(f"  ✓ 投影完成: {output_path}")
    logger.info(f"  尺寸: {height} x {width}")
    
    return output_path


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("处理Bohemian Forest区域数据")
    logger.info("=" * 80)
    
    # 路径配置
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data/raw/gee"
    merged_dir = project_root / "data/processed/merged_gee/bohemian_forest"
    utm_dir = project_root / "data/processed/utm_grid/bohemian_forest"
    
    # Bohemian Forest配置
    region_config = cfg.get('regions.bohemian_forest')
    target_epsg = region_config['epsg']  # 32633 (UTM 33N)
    
    logger.info(f"目标坐标系: EPSG:{target_epsg} (UTM 33N)")
    logger.info("")
    
    # 步骤1: 合并DEM tiles
    logger.info("步骤1: 合并DEM tiles")
    dem_tiles_dir = raw_dir / "bohemian_forest_srtm_dem_tiles"
    dem_merged = merged_dir / "dem.tif"
    merge_tiles(dem_tiles_dir, dem_merged, "dem")
    logger.info("")
    
    # 步骤2: 合并LULC tiles
    logger.info("步骤2: 合并LULC tiles")
    lulc_tiles_dir = raw_dir / "bohemian_forest_worldcover_lulc_tiles"
    lulc_merged = merged_dir / "lulc.tif"
    merge_tiles(lulc_tiles_dir, lulc_merged, "lulc")
    logger.info("")
    
    # 步骤3: 投影DEM到UTM
    logger.info("步骤3: 投影DEM到UTM")
    dem_utm = utm_dir / "dem_utm.tif"
    reproject_to_utm(dem_merged, dem_utm, target_epsg)
    logger.info("")
    
    # 步骤4: 投影LULC到UTM
    logger.info("步骤4: 投影LULC到UTM")
    lulc_utm = utm_dir / "lulc_utm.tif"
    reproject_to_utm(lulc_merged, lulc_utm, target_epsg)
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("✅ Bohemian Forest区域数据处理完成！")
    logger.info("=" * 80)
    logger.info(f"DEM (UTM): {dem_utm}")
    logger.info(f"LULC (UTM): {lulc_utm}")
    logger.info("")
    logger.info("下一步：")
    logger.info("1. 下载OSM道路数据（可选）")
    logger.info("2. 运行轨迹生成脚本，将自动生成代价图和可通行域")


if __name__ == '__main__':
    main()
