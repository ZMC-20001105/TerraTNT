#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理新区域的GEE原始数据 → utm_grid格式

支持两种输入格式:
  1. 单文件: data/raw/gee/{region}_dem.tif (GEE直接导出)
  2. 分块tiles: data/raw/gee/{region}_srtm_dem_tiles/tile_*.tif

输出: data/processed/utm_grid/{region}/
  - dem_utm.tif, lulc_utm.tif, slope_utm.tif, aspect_utm.tif

用法:
  python scripts/process_new_regions.py --region donbas carpathians
  python scripts/process_new_regions.py --all
  python scripts/process_new_regions.py --list
"""
import sys
import argparse
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(PROJECT_ROOT / 'config' / 'config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_raw_file(raw_dir, region, data_type):
    """查找原始数据文件（单文件或tiles目录或chunks子目录）"""
    # 单文件格式
    single = raw_dir / f'{region}_{data_type}.tif'
    if single.exists():
        return 'single', single

    # tiles目录格式
    for pattern in [f'{region}_srtm_{data_type}_tiles', f'{region}_{data_type}_tiles',
                    f'{region}_worldcover_{data_type}_tiles']:
        tiles_dir = raw_dir / pattern
        if tiles_dir.exists() and list(tiles_dir.glob('tile_*.tif')):
            return 'tiles', tiles_dir

    # chunks子目录格式 (download_regions_direct.py 输出)
    chunks_dir = raw_dir / region / data_type
    if chunks_dir.exists() and list(chunks_dir.glob('chunk_*.tif')):
        return 'chunks', chunks_dir

    return None, None


def merge_tiles(tiles_dir):
    """合并GEE分块tiles/chunks为单个文件"""
    tile_files = sorted(tiles_dir.glob('tile_*.tif'))
    if not tile_files:
        tile_files = sorted(tiles_dir.glob('chunk_*.tif'))
    logger.info(f"  合并 {len(tile_files)} 个tiles...")
    src_files = [rasterio.open(f) for f in tile_files]
    mosaic, out_trans = merge(src_files)
    meta = src_files[0].meta.copy()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2],
                 "transform": out_trans, "driver": "GTiff", "compress": "lzw"})
    for s in src_files:
        s.close()
    return mosaic, meta


def reproject_to_utm(input_path_or_data, output_path, target_epsg, resolution=30,
                     resampling=Resampling.bilinear, src_meta=None):
    """投影到UTM坐标系"""
    if isinstance(input_path_or_data, (str, Path)):
        src = rasterio.open(input_path_or_data)
        data = src.read()
        src_meta = src.meta.copy()
        src.close()
    else:
        data = input_path_or_data

    transform, width, height = calculate_default_transform(
        src_meta['crs'], f'EPSG:{target_epsg}',
        src_meta['width'], src_meta['height'],
        *rasterio.transform.array_bounds(src_meta['height'], src_meta['width'], src_meta['transform']),
        resolution=resolution
    )

    kwargs = src_meta.copy()
    kwargs.update({'crs': f'EPSG:{target_epsg}', 'transform': transform,
                   'width': width, 'height': height, 'compress': 'lzw', 'driver': 'GTiff'})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        for i in range(data.shape[0]):
            reproject(
                source=data[i], destination=rasterio.band(dst, i + 1),
                src_transform=src_meta['transform'], src_crs=src_meta['crs'],
                dst_transform=transform, dst_crs=f'EPSG:{target_epsg}',
                resampling=resampling
            )

    logger.info(f"  ✓ {output_path.name}: {height}x{width} px")
    return output_path


def compute_slope_aspect(dem_path, slope_path, aspect_path):
    """从DEM计算坡度和坡向"""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        res = src.res[0]  # 像素分辨率(m)

    # 梯度
    dy, dx = np.gradient(dem, res)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    aspect_rad = np.arctan2(-dx, dy)
    aspect_deg = np.degrees(aspect_rad).astype(np.float32)
    aspect_deg[aspect_deg < 0] += 360.0

    meta.update({'dtype': 'float32', 'compress': 'lzw', 'driver': 'GTiff'})
    for path, data in [(slope_path, slope_deg), (aspect_path, aspect_deg)]:
        with rasterio.open(path, 'w', **meta) as dst:
            dst.write(data, 1)
        logger.info(f"  ✓ {path.name}: {data.shape}")


def process_region(region_key, region_cfg):
    """处理单个区域"""
    epsg = region_cfg['epsg']
    raw_dir = PROJECT_ROOT / 'data' / 'raw' / 'gee'
    utm_dir = PROJECT_ROOT / 'data' / 'processed' / 'utm_grid' / region_key

    logger.info(f"\n{'='*60}")
    logger.info(f"处理区域: {region_cfg['name']} (EPSG:{epsg})")
    logger.info(f"{'='*60}")

    # 检查是否已有utm_grid数据
    if (utm_dir / 'dem_utm.tif').exists():
        logger.info(f"  ⚠ {utm_dir} 已存在，跳过（使用 --force 覆盖）")
        return False

    processed = False
    for data_type, utm_name, res, resamp in [
        ('dem', 'dem_utm.tif', 30, Resampling.bilinear),
        ('lulc', 'lulc_utm.tif', 30, Resampling.nearest),
    ]:
        fmt, source = find_raw_file(raw_dir, region_key, data_type)
        if fmt is None:
            logger.warning(f"  ⚠ 未找到 {region_key} 的 {data_type} 原始数据")
            continue

        logger.info(f"\n--- {data_type.upper()} ---")
        if fmt in ('tiles', 'chunks'):
            mosaic, meta = merge_tiles(source)
            reproject_to_utm(mosaic, utm_dir / utm_name, epsg, res, resamp, meta)
        else:
            reproject_to_utm(source, utm_dir / utm_name, epsg, res, resamp)
        processed = True

    # slope/aspect: 优先从GEE下载的文件，否则从DEM计算
    dem_utm = utm_dir / 'dem_utm.tif'
    if dem_utm.exists():
        slope_utm = utm_dir / 'slope_utm.tif'
        aspect_utm = utm_dir / 'aspect_utm.tif'

        # 检查GEE下载的slope/aspect
        for data_type, utm_name, res in [('slope', 'slope_utm.tif', 30), ('aspect', 'aspect_utm.tif', 30)]:
            fmt, source = find_raw_file(raw_dir, region_key, data_type)
            if fmt is not None:
                logger.info(f"\n--- {data_type.upper()} (from GEE) ---")
                if fmt in ('tiles', 'chunks'):
                    mosaic, meta = merge_tiles(source)
                    reproject_to_utm(mosaic, utm_dir / utm_name, epsg, res, Resampling.bilinear, meta)
                else:
                    reproject_to_utm(source, utm_dir / utm_name, epsg, res, Resampling.bilinear)

        # 如果GEE没有slope/aspect，从DEM计算
        if not slope_utm.exists() or not aspect_utm.exists():
            logger.info("\n--- Slope/Aspect (从DEM计算) ---")
            compute_slope_aspect(dem_utm, slope_utm, aspect_utm)

        processed = True

    if processed:
        logger.info(f"\n✅ {region_cfg['name']} 处理完成 → {utm_dir}")
    return processed


def main():
    parser = argparse.ArgumentParser(description='处理新区域GEE数据')
    parser.add_argument('--region', nargs='+', help='要处理的区域')
    parser.add_argument('--all', action='store_true', help='处理所有区域')
    parser.add_argument('--force', action='store_true', help='覆盖已有数据')
    parser.add_argument('--list', action='store_true', help='列出所有区域')
    args = parser.parse_args()

    cfg = load_config()
    regions = cfg.get('regions', {})

    if args.list:
        print("配置的区域:")
        for key, rcfg in regions.items():
            utm_exists = (PROJECT_ROOT / 'data' / 'processed' / 'utm_grid' / key / 'dem_utm.tif').exists()
            status = '✓' if utm_exists else '✗'
            print(f"  {status} {key:25s} {rcfg['name']:25s} EPSG:{rcfg['epsg']}")
        return

    if args.all:
        targets = list(regions.keys())
    elif args.region:
        targets = args.region
    else:
        parser.print_help()
        return

    for key in targets:
        if key not in regions:
            logger.warning(f"未知区域: {key}")
            continue
        if args.force:
            utm_dir = PROJECT_ROOT / 'data' / 'processed' / 'utm_grid' / key
            if utm_dir.exists():
                import shutil
                shutil.rmtree(utm_dir)
                logger.info(f"已删除旧数据: {utm_dir}")
        process_region(key, regions[key])


if __name__ == '__main__':
    main()
