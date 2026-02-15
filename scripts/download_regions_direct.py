#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接下载区域环境数据到本地（不经过Google Drive）。
使用 ee.Image.getDownloadURL 分块下载大区域数据。

数据源:
  - DEM: USGS/SRTMGL1_003 (SRTM 30m)
  - LULC: ESA/WorldCover/v200 (10m)
  - Slope/Aspect: 从DEM派生

用法:
  python scripts/download_regions_direct.py --regions donbas carpathians
  python scripts/download_regions_direct.py --regions donbas --data dem lulc
  python scripts/download_regions_direct.py --list
"""

import sys, os, json, argparse, time, zipfile, io, tempfile
from pathlib import Path
import requests
import numpy as np

# 代理
os.environ.setdefault('https_proxy', 'socks5://127.0.0.1:7897')
os.environ.setdefault('http_proxy', 'socks5://127.0.0.1:7897')

import ee
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── GEE 初始化 ──
CREDENTIALS_PATH = str(PROJECT_ROOT / 'gen-lang-client-0843667030-72e96d89711d.json')

def init_ee():
    with open(CREDENTIALS_PATH) as f:
        key_data = json.load(f)
    sa = key_data['client_email']
    credentials = ee.ServiceAccountCredentials(sa, CREDENTIALS_PATH)
    ee.Initialize(credentials)
    print(f"✓ Earth Engine 已初始化 (服务账号: {sa})")


def load_regions():
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    regions = {}
    for key, rcfg in cfg.get('regions', {}).items():
        b = rcfg['bounds']
        regions[key] = {
            'name': rcfg['name'],
            'bounds': [b['lon_min'], b['lat_min'], b['lon_max'], b['lat_max']],
            'epsg': rcfg['epsg'],
            'utm_zone': rcfg['utm_zone'],
        }
    return regions


def download_tif_from_url(url, output_path, timeout=300):
    """从GEE URL下载GeoTIFF（可能是zip包含tif）"""
    print(f"    下载中...", end='', flush=True)
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()

    content_type = resp.headers.get('Content-Type', '')
    data = resp.content
    size_mb = len(data) / 1024 / 1024
    print(f" {size_mb:.1f}MB", end='', flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if 'zip' in content_type or data[:2] == b'PK':
        # ZIP格式，解压出tif
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            tif_names = [n for n in zf.namelist() if n.endswith('.tif')]
            if tif_names:
                with zf.open(tif_names[0]) as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                print(f" ✓ (解压)", flush=True)
                return True
            else:
                print(f" ❌ ZIP中无tif文件", flush=True)
                return False
    else:
        with open(output_path, 'wb') as f:
            f.write(data)
        print(f" ✓", flush=True)
        return True


def download_chunked(image, band, region_bounds, output_dir, data_type,
                     scale, epsg, chunk_deg=0.5):
    """
    分块下载大区域影像。
    将区域按 chunk_deg x chunk_deg 度分块，逐块下载。
    """
    min_lon, min_lat, max_lon, max_lat = region_bounds
    out_dir = output_dir / data_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已完成
    done_flag = out_dir / 'download_complete.txt'
    if done_flag.exists():
        print(f"  {data_type}: 已完成，跳过")
        return True

    # 计算分块
    lon_chunks = np.arange(min_lon, max_lon, chunk_deg)
    lat_chunks = np.arange(min_lat, max_lat, chunk_deg)
    total = len(lon_chunks) * len(lat_chunks)

    print(f"  {data_type.upper()}: {len(lon_chunks)}x{len(lat_chunks)} = {total} 块 "
          f"(scale={scale}m, chunk={chunk_deg}°)")

    success_count = 0
    fail_count = 0

    for ri, lat in enumerate(lat_chunks):
        for ci, lon in enumerate(lon_chunks):
            chunk_name = f"chunk_r{ri}_c{ci}.tif"
            chunk_path = out_dir / chunk_name

            if chunk_path.exists() and chunk_path.stat().st_size > 100:
                success_count += 1
                continue

            lat_end = min(lat + chunk_deg, max_lat)
            lon_end = min(lon + chunk_deg, max_lon)
            chunk_region = ee.Geometry.Rectangle([lon, lat, lon_end, lat_end])

            try:
                url = image.select(band).getDownloadURL({
                    'name': f'{data_type}_{ri}_{ci}',
                    'region': chunk_region,
                    'scale': scale,
                    'crs': f'EPSG:{epsg}',
                    'format': 'GEO_TIFF',
                })
                if download_tif_from_url(url, chunk_path):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"    ❌ chunk_r{ri}_c{ci}: {e}")
                fail_count += 1
                time.sleep(2)

            # 避免API限流
            time.sleep(0.5)

    print(f"  {data_type}: {success_count}/{total} 成功, {fail_count} 失败")

    if fail_count == 0:
        done_flag.write_text(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                             f"chunks: {total}\n")
    return fail_count == 0


def download_region(region_key, region_config, data_types=None):
    """下载单个区域的所有环境数据"""
    if data_types is None:
        data_types = ['dem', 'slope', 'aspect', 'lulc']

    bounds = region_config['bounds']
    epsg = region_config['epsg']
    output_dir = PROJECT_ROOT / 'data' / 'raw' / 'gee' / region_key

    print(f"\n{'='*60}")
    print(f"区域: {region_config['name']} ({region_key})")
    print(f"范围: {bounds}")
    print(f"EPSG: {epsg}")
    print(f"输出: {output_dir}")
    print(f"{'='*60}")

    roi = ee.Geometry.Rectangle(bounds)
    dem = ee.Image('USGS/SRTMGL1_003').clip(roi)

    results = {}

    for dt in data_types:
        if dt == 'dem':
            results[dt] = download_chunked(
                dem, 'elevation', bounds, output_dir, 'dem',
                scale=30, epsg=epsg, chunk_deg=0.5)
        elif dt == 'slope':
            slope = ee.Terrain.slope(dem)
            results[dt] = download_chunked(
                slope, 'slope', bounds, output_dir, 'slope',
                scale=30, epsg=epsg, chunk_deg=0.5)
        elif dt == 'aspect':
            aspect = ee.Terrain.aspect(dem)
            results[dt] = download_chunked(
                aspect, 'aspect', bounds, output_dir, 'aspect',
                scale=30, epsg=epsg, chunk_deg=0.5)
        elif dt == 'lulc':
            lulc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi)
            # LULC 10m分辨率，数据量大，用更小的块
            results[dt] = download_chunked(
                lulc, 'Map', bounds, output_dir, 'lulc',
                scale=10, epsg=epsg, chunk_deg=0.25)

    return results


def download_osm_for_region(region_key, region_config):
    """下载区域的OSM道路数据"""
    try:
        import osmnx as ox
    except ImportError:
        print("  ⚠ osmnx未安装，跳过OSM下载")
        return False

    bounds = region_config['bounds']
    min_lon, min_lat, max_lon, max_lat = bounds

    osm_dir = PROJECT_ROOT / 'data' / 'osm' / region_key
    osm_dir.mkdir(parents=True, exist_ok=True)

    edges_file = osm_dir / 'edges_drive.geojson'
    if edges_file.exists():
        print(f"  OSM: 已有 {edges_file.name}，跳过")
        return True

    print(f"  OSM: 下载 {region_config['name']} 道路网络...")
    try:
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(osm_dir / 'cache')

        G = ox.graph_from_bbox(
            bbox=(max_lat, min_lat, max_lon, min_lon),
            network_type='drive', simplify=True, retain_all=False)

        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        edges_gdf.to_file(edges_file, driver='GeoJSON')
        nodes_gdf.to_file(osm_dir / 'nodes_drive.geojson', driver='GeoJSON')
        ox.save_graphml(G, osm_dir / 'network_drive.graphml')

        print(f"  OSM: ✓ {len(G.nodes)} 节点, {len(G.edges)} 边, "
              f"总长 {edges_gdf['length'].sum()/1000:.0f}km")
        return True
    except Exception as e:
        print(f"  OSM: ❌ {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='直接下载区域环境数据到本地')
    parser.add_argument('--regions', nargs='+', default=['donbas', 'carpathians'],
                        help='要下载的区域')
    parser.add_argument('--data', nargs='+', default=None,
                        help='数据类型 (dem slope aspect lulc osm)')
    parser.add_argument('--list', action='store_true', help='列出可用区域')
    parser.add_argument('--chunk_deg', type=float, default=0.5,
                        help='分块大小(度)')
    parser.add_argument('--skip_osm', action='store_true', help='跳过OSM下载')
    args = parser.parse_args()

    REGIONS = load_regions()

    if args.list:
        print("可用区域:")
        for k, v in REGIONS.items():
            existing = (PROJECT_ROOT / 'data' / 'raw' / 'gee' / k).exists()
            status = "✓ 已有数据" if existing else "  无数据"
            print(f"  {k:22s} {v['name']:22s} EPSG:{v['epsg']}  {status}")
        return

    init_ee()

    gee_types = [d for d in (args.data or ['dem', 'slope', 'aspect', 'lulc'])
                 if d != 'osm']
    do_osm = not args.skip_osm and (args.data is None or 'osm' in args.data)

    all_results = {}
    for region_key in args.regions:
        if region_key not in REGIONS:
            print(f"⚠ 未知区域: {region_key}")
            continue

        results = download_region(region_key, REGIONS[region_key], gee_types)
        if do_osm:
            results['osm'] = download_osm_for_region(region_key, REGIONS[region_key])
        all_results[region_key] = results

    # 汇总
    print(f"\n{'='*60}")
    print("下载汇总")
    print(f"{'='*60}")
    for region_key, results in all_results.items():
        print(f"\n{region_key}:")
        for dt, ok in results.items():
            print(f"  {dt:8s}: {'✓' if ok else '❌'}")

    print(f"\n下一步:")
    print(f"  python scripts/process_new_regions.py --regions {' '.join(args.regions)}")


if __name__ == '__main__':
    main()
