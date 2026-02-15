#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载新区域(Donbas, Carpathians, 及更多)的环境数据
使用Google Earth Engine API

区域来源: config/config.yaml
数据: DEM (SRTM 30m) + LULC (ESA WorldCover 10m)
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import ee
from pathlib import Path
import time
import os
import argparse

# 设置代理（国内网络需要）
os.environ.setdefault('https_proxy', 'socks5://127.0.0.1:7897')
os.environ.setdefault('http_proxy', 'socks5://127.0.0.1:7897')

# 初始化Earth Engine - 使用服务账号
credentials_path = '/home/zmc/文档/programwork/gen-lang-client-0843667030-72e96d89711d.json'

try:
    import json
    with open(credentials_path) as f:
        key_data = json.load(f)
    service_account = key_data['client_email']
    credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
    ee.Initialize(credentials)
    print(f"✓ Earth Engine 已初始化（服务账号: {service_account}）")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    print("  请确认服务账号密钥文件存在且有效")
    print("  请确认代理已启动（socks5://127.0.0.1:7897）")
    sys.exit(1)

# 从config.yaml读取区域配置
import yaml

def load_regions_from_config():
    """从config.yaml读取所有区域配置"""
    config_path = Path('/home/zmc/文档/programwork/config/config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    regions = {}
    for key, rcfg in cfg.get('regions', {}).items():
        b = rcfg['bounds']
        regions[key] = {
            'name': rcfg['name'],
            'bounds': [b['lon_min'], b['lat_min'], b['lon_max'], b['lat_max']],
            'epsg': rcfg['epsg'],
            'description': f"{rcfg['name']} (UTM {rcfg['utm_zone']})",
        }
    return regions

REGIONS = load_regions_from_config()


def download_region_data(region_key, region_config, drive_folder='GEE_TerraTNT'):
    """下载单个区域的DEM和LULC数据"""
    print(f"\n{'='*60}")
    print(f"下载区域: {region_config['name']}")
    print(f"描述: {region_config['description']}")
    print(f"范围: {region_config['bounds']}")
    print(f"EPSG: {region_config['epsg']}")
    print(f"{'='*60}\n")

    bounds = region_config['bounds']
    roi = ee.Geometry.Rectangle(bounds)
    epsg = region_config['epsg']

    tasks = {}

    # 1. DEM (SRTM 30m)
    print("1. 准备下载 DEM (SRTM 30m)...")
    dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
    dem_task = ee.batch.Export.image.toDrive(
        image=dem.select('elevation'),
        description=f'{region_key}_dem',
        folder=drive_folder,
        fileNamePrefix=f'{region_key}_dem',
        region=roi,
        scale=30,
        crs=f'EPSG:{epsg}',
        maxPixels=1e13
    )
    dem_task.start()
    tasks['dem'] = dem_task
    print(f"   ✓ DEM 导出任务已启动: {dem_task.id}")

    # 2. LULC (ESA WorldCover 10m)
    print("2. 准备下载 LULC (ESA WorldCover 10m)...")
    lulc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi)
    lulc_task = ee.batch.Export.image.toDrive(
        image=lulc.select('Map'),
        description=f'{region_key}_lulc',
        folder=drive_folder,
        fileNamePrefix=f'{region_key}_lulc',
        region=roi,
        scale=10,
        crs=f'EPSG:{epsg}',
        maxPixels=1e13
    )
    lulc_task.start()
    tasks['lulc'] = lulc_task
    print(f"   ✓ LULC 导出任务已启动: {lulc_task.id}")

    # 3. Slope (从DEM派生)
    print("3. 准备下载 Slope...")
    slope = ee.Terrain.slope(dem)
    slope_task = ee.batch.Export.image.toDrive(
        image=slope,
        description=f'{region_key}_slope',
        folder=drive_folder,
        fileNamePrefix=f'{region_key}_slope',
        region=roi,
        scale=30,
        crs=f'EPSG:{epsg}',
        maxPixels=1e13
    )
    slope_task.start()
    tasks['slope'] = slope_task
    print(f"   ✓ Slope 导出任务已启动: {slope_task.id}")

    # 4. Aspect (从DEM派生)
    print("4. 准备下载 Aspect...")
    aspect = ee.Terrain.aspect(dem)
    aspect_task = ee.batch.Export.image.toDrive(
        image=aspect,
        description=f'{region_key}_aspect',
        folder=drive_folder,
        fileNamePrefix=f'{region_key}_aspect',
        region=roi,
        scale=30,
        crs=f'EPSG:{epsg}',
        maxPixels=1e13
    )
    aspect_task.start()
    tasks['aspect'] = aspect_task
    print(f"   ✓ Aspect 导出任务已启动: {aspect_task.id}")

    return tasks


def check_all_tasks(all_tasks):
    """检查所有任务状态"""
    print(f"\n{'='*60}")
    print("检查导出任务状态")
    print(f"{'='*60}\n")

    for region_key, tasks in all_tasks.items():
        print(f"{region_key.upper()}:")
        for data_type, task in tasks.items():
            status = task.status()
            state = status['state']
            icon = '✓' if state == 'COMPLETED' else ('⏳' if state == 'RUNNING' else '❌')
            print(f"  {icon} {data_type:8s}: {state}")
            if 'error_message' in status:
                print(f"    错误: {status['error_message']}")
        print()


def main():
    parser = argparse.ArgumentParser(description='下载新区域环境数据')
    parser.add_argument('--regions', nargs='+', default=None,
                        help='要下载的区域 (默认: 全部)')
    parser.add_argument('--list', action='store_true',
                        help='列出可用区域')
    parser.add_argument('--check', action='store_true',
                        help='仅检查任务状态')
    parser.add_argument('--folder', default='GEE_TerraTNT',
                        help='Google Drive文件夹名')
    args = parser.parse_args()

    if args.list:
        print("可用区域:")
        for key, cfg in REGIONS.items():
            print(f"  {key:20s} {cfg['name']:20s} {cfg['description']}")
        return

    regions_to_download = args.regions or list(REGIONS.keys())

    print("=" * 60)
    print("TerraTNT - 下载新区域环境数据")
    print("=" * 60)
    print(f"目标区域: {', '.join(regions_to_download)}")
    print(f"Drive文件夹: {args.folder}")
    print()

    all_tasks = {}
    for region_key in regions_to_download:
        if region_key not in REGIONS:
            print(f"⚠ 未知区域: {region_key}, 跳过")
            continue
        all_tasks[region_key] = download_region_data(
            region_key, REGIONS[region_key], args.folder)
        time.sleep(2)

    # 检查状态
    time.sleep(5)
    check_all_tasks(all_tasks)

    print(f"\n{'='*60}")
    print("✓ 所有导出任务已提交")
    print(f"{'='*60}\n")
    print("任务将在Google Earth Engine后台运行（通常10-30分钟）")
    print("完成后文件会出现在Google Drive的 '{}' 文件夹中".format(args.folder))
    print("\n下载完成后，请：")
    print("1. 从Google Drive下载 {region}_dem.tif, {region}_lulc.tif,")
    print("   {region}_slope.tif, {region}_aspect.tif")
    print("2. 放到 data/raw/gee/{region}/ 目录下")
    print("3. 运行处理脚本:")
    print("   python scripts/process_new_region.py --region {region}")
    print("\n监控任务状态：")
    print("  https://code.earthengine.google.com/tasks")
    print("  或重新运行: python scripts/download_new_regions.py --check")


if __name__ == '__main__':
    main()
