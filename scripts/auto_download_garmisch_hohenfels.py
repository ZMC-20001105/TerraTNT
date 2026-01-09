#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动下载Garmisch和Hohenfels环境数据
使用服务账号认证，完全自动化，无需手动操作
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import ee
import json
import requests
from pathlib import Path
import time

print("="*60)
print("自动下载Garmisch和Hohenfels环境数据")
print("="*60)

# 1. 初始化GEE
print("\n1. 初始化Google Earth Engine...")
key_file = '/home/zmc/文档/programwork/gen-lang-client-0843667030-72e96d89711d.json'

with open(key_file) as f:
    key_data = json.load(f)

credentials = ee.ServiceAccountCredentials(
    key_data['client_email'],
    key_file
)
ee.Initialize(credentials, project=key_data['project_id'])
print(f"✓ GEE初始化成功")
print(f"  服务账号: {key_data['client_email']}")
print(f"  项目ID: {key_data['project_id']}")

# 2. 区域配置
REGIONS = {
    'garmisch': {
        'name': 'Garmisch-Partenkirchen',
        'bounds': [10.8, 47.4, 11.2, 47.6],
        'epsg': 32632,
    },
    'hohenfels': {
        'name': 'Hohenfels',
        'bounds': [11.7, 49.1, 12.0, 49.3],
        'epsg': 32632,
    }
}

def download_file(url, output_path):
    """下载文件"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size * 100
                    print(f"\r  下载进度: {progress:.1f}%", end='', flush=True)
    
    print(f"\r  ✓ 下载完成: {downloaded/1024/1024:.2f} MB")

def download_region(region_key, region_config):
    """下载单个区域的数据"""
    print(f"\n{'='*60}")
    print(f"区域: {region_config['name']}")
    print(f"{'='*60}")
    
    bounds = region_config['bounds']
    roi = ee.Geometry.Rectangle(bounds)
    epsg = f"EPSG:{region_config['epsg']}"
    
    output_dir = Path(f'/home/zmc/文档/programwork/data/raw/gee/{region_key}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载DEM
    print("\n📥 DEM (SRTM 30m)...")
    dem = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(roi)
    dem_url = dem.getDownloadURL({
        'scale': 30,
        'crs': epsg,
        'region': roi,
        'format': 'GEO_TIFF'
    })
    dem_path = output_dir / 'dem.tif'
    download_file(dem_url, dem_path)
    
    # 下载LULC
    print("\n📥 LULC (ESA WorldCover 10m)...")
    lulc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').clip(roi)
    lulc_url = lulc.getDownloadURL({
        'scale': 10,
        'crs': epsg,
        'region': roi,
        'format': 'GEO_TIFF'
    })
    lulc_path = output_dir / 'lulc.tif'
    download_file(lulc_url, lulc_path)
    
    print(f"\n✓ {region_config['name']} 下载完成")
    print(f"  保存位置: {output_dir}")

# 3. 下载所有区域
for region_key, region_config in REGIONS.items():
    try:
        download_region(region_key, region_config)
        time.sleep(2)  # 避免API限流
    except Exception as e:
        print(f"\n✗ {region_config['name']} 下载失败: {e}")
        continue

print(f"\n{'='*60}")
print("✓ 所有下载任务完成")
print(f"{'='*60}")
print("\n下一步:")
print("1. 运行处理脚本生成slope和aspect")
print("2. 重投影到UTM格式")
print("3. 生成cost_map和passable_mask")
