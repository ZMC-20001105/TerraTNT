#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载Garmisch和Hohenfels区域的环境数据
使用Google Earth Engine API
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import ee
from pathlib import Path
import time

# 初始化Earth Engine - 使用服务账号
import os
credentials_path = '/home/zmc/文档/programwork/gen-lang-client-0843667030-72e96d89711d.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

try:
    ee.Initialize()
    print("✓ Earth Engine 已初始化（服务账号）")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit(1)

# 区域配置
REGIONS = {
    'garmisch': {
        'name': 'Garmisch-Partenkirchen',
        'bounds': [10.8, 47.4, 11.2, 47.6],  # [min_lon, min_lat, max_lon, max_lat]
        'epsg': 32632,  # UTM 32N
        'description': '德国加米施-帕滕基兴地区'
    },
    'hohenfels': {
        'name': 'Hohenfels',
        'bounds': [11.7, 49.1, 12.0, 49.3],
        'epsg': 32632,  # UTM 32N
        'description': '德国霍恩费尔斯训练区'
    }
}

def download_region_data(region_key, region_config):
    """
    下载单个区域的DEM和LULC数据
    """
    print(f"\n{'='*60}")
    print(f"下载区域: {region_config['name']}")
    print(f"描述: {region_config['description']}")
    print(f"{'='*60}\n")
    
    # 定义区域
    bounds = region_config['bounds']
    roi = ee.Geometry.Rectangle(bounds)
    
    output_dir = Path(f'/home/zmc/文档/programwork/data/raw/gee/{region_key}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 下载DEM (SRTM 30m)
    print("1. 准备下载 DEM (SRTM 30m)...")
    dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
    
    dem_task = ee.batch.Export.image.toDrive(
        image=dem.select('elevation'),
        description=f'{region_key}_dem',
        folder='GEE_TerraTNT',
        fileNamePrefix=f'{region_key}_dem',
        region=roi,
        scale=30,
        crs=f'EPSG:{region_config["epsg"]}',
        maxPixels=1e13
    )
    dem_task.start()
    print(f"   ✓ DEM 导出任务已启动: {dem_task.id}")
    
    # 2. 下载LULC (ESA WorldCover 10m)
    print("2. 准备下载 LULC (ESA WorldCover 10m)...")
    lulc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi)
    
    lulc_task = ee.batch.Export.image.toDrive(
        image=lulc.select('Map'),
        description=f'{region_key}_lulc',
        folder='GEE_TerraTNT',
        fileNamePrefix=f'{region_key}_lulc',
        region=roi,
        scale=10,
        crs=f'EPSG:{region_config["epsg"]}',
        maxPixels=1e13
    )
    lulc_task.start()
    print(f"   ✓ LULC 导出任务已启动: {lulc_task.id}")
    
    return {
        'dem_task': dem_task,
        'lulc_task': lulc_task
    }

def check_task_status(tasks):
    """
    检查任务状态
    """
    print(f"\n{'='*60}")
    print("检查导出任务状态")
    print(f"{'='*60}\n")
    
    for region_key, region_tasks in tasks.items():
        print(f"\n{region_key.upper()}:")
        
        dem_status = region_tasks['dem_task'].status()
        print(f"  DEM:  {dem_status['state']}")
        if 'error_message' in dem_status:
            print(f"    错误: {dem_status['error_message']}")
        
        lulc_status = region_tasks['lulc_task'].status()
        print(f"  LULC: {lulc_status['state']}")
        if 'error_message' in lulc_status:
            print(f"    错误: {lulc_status['error_message']}")

def main():
    print("=" * 60)
    print("TerraTNT - 下载Garmisch和Hohenfels环境数据")
    print("=" * 60)
    
    tasks = {}
    
    # 下载所有区域
    for region_key, region_config in REGIONS.items():
        tasks[region_key] = download_region_data(region_key, region_config)
        time.sleep(2)  # 避免API限流
    
    # 检查状态
    time.sleep(5)
    check_task_status(tasks)
    
    print(f"\n{'='*60}")
    print("✓ 所有导出任务已提交")
    print(f"{'='*60}\n")
    print("任务将在Google Earth Engine后台运行")
    print("完成后文件会出现在Google Drive的 'GEE_TerraTNT' 文件夹中")
    print("\n下载完成后，请：")
    print("1. 从Google Drive下载文件到 data/raw/gee/<region_name>/")
    print("2. 运行处理脚本生成UTM格式和派生数据")
    print("\n监控任务状态：")
    print("  https://code.earthengine.google.com/tasks")

if __name__ == '__main__':
    main()
