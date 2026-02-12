#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XYZ瓦片地图获取器 — 支持卫星影像/街道地图底图
从在线瓦片服务下载并缓存瓦片, 拼接为指定UTM范围的RGB图像
"""
import math
import hashlib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# 瓦片缓存目录
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'cache' / 'tiles'

# 瓦片源定义
TILE_SOURCES = {
    '卫星影像': {
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'attribution': 'Esri World Imagery',
        'max_zoom': 18,
    },
    '街道地图': {
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        'attribution': 'Esri World Street Map',
        'max_zoom': 18,
    },
    '地形地图': {
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        'attribution': 'Esri World Topo Map',
        'max_zoom': 18,
    },
}


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """WGS84经纬度 → 瓦片坐标 (x, y)"""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(x, n - 1)), max(0, min(y, n - 1))


def _tile_to_lonlat(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """瓦片坐标左上角 → WGS84经纬度"""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def _fetch_tile(source_name: str, z: int, x: int, y: int) -> Optional[np.ndarray]:
    """获取单个瓦片 (优先缓存, 否则下载)"""
    source = TILE_SOURCES.get(source_name)
    if source is None:
        return None

    # 缓存路径
    cache_subdir = CACHE_DIR / source_name / str(z) / str(x)
    cache_file = cache_subdir / f'{y}.png'

    if cache_file.exists():
        try:
            from PIL import Image
            img = Image.open(cache_file).convert('RGB')
            return np.array(img, dtype=np.uint8)
        except Exception:
            cache_file.unlink(missing_ok=True)

    # 下载
    url = source['url'].format(z=z, x=x, y=y)
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={
            'User-Agent': 'TerraTNT-Viz/1.0 (research project)'
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()

        # 缓存到磁盘
        cache_subdir.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)

        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data)).convert('RGB')
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        logger.debug(f"瓦片下载失败 {url}: {e}")
        return None


def _utm_to_lonlat(easting: float, northing: float, crs) -> Tuple[float, float]:
    """UTM坐标 → WGS84经纬度"""
    import pyproj
    transformer = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lon, lat


def fetch_satellite_image(center_utm: Tuple[float, float],
                          coverage_km: float,
                          out_size: int = 512,
                          crs=None,
                          source_name: str = '卫星影像') -> Optional[np.ndarray]:
    """
    获取指定UTM范围的卫星/街道地图图像

    Args:
        center_utm: (easting, northing) UTM中心坐标
        coverage_km: 覆盖范围 (km, 正方形边长)
        out_size: 输出图像尺寸 (像素)
        crs: 源CRS (pyproj兼容格式, 如 'EPSG:32633')
        source_name: 瓦片源名称

    Returns:
        (out_size, out_size, 3) uint8 RGB图像, 或 None
    """
    if crs is None:
        logger.warning("未指定CRS, 无法获取卫星影像")
        return None

    source = TILE_SOURCES.get(source_name)
    if source is None:
        return None

    half_m = coverage_km * 500.0  # km → m, half
    cx, cy = center_utm
    # 四角UTM坐标
    corners_utm = [
        (cx - half_m, cy - half_m),  # SW
        (cx + half_m, cy + half_m),  # NE
    ]

    # UTM → WGS84
    import pyproj
    transformer = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    sw_lon, sw_lat = transformer.transform(corners_utm[0][0], corners_utm[0][1])
    ne_lon, ne_lat = transformer.transform(corners_utm[1][0], corners_utm[1][1])

    # 选择合适的zoom级别
    # 目标: 每个像素约 coverage_km*1000/out_size 米
    meters_per_pixel = coverage_km * 1000.0 / out_size
    # 在赤道处, zoom z 的分辨率 ≈ 156543 / 2^z m/px
    # 在纬度lat处, ≈ 156543 * cos(lat) / 2^z
    mid_lat = (sw_lat + ne_lat) / 2
    for z in range(source['max_zoom'], 0, -1):
        tile_res = 156543.0 * math.cos(math.radians(mid_lat)) / (2 ** z)
        if tile_res <= meters_per_pixel * 2:
            break
    zoom = min(z, source['max_zoom'])

    # 计算需要的瓦片范围
    tx_min, ty_min = _lonlat_to_tile(sw_lon, ne_lat, zoom)  # NW corner
    tx_max, ty_max = _lonlat_to_tile(ne_lon, sw_lat, zoom)  # SE corner

    # 限制瓦片数量 (避免下载过多)
    n_tiles = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
    if n_tiles > 100:
        # 降低zoom
        while n_tiles > 100 and zoom > 1:
            zoom -= 1
            tx_min, ty_min = _lonlat_to_tile(sw_lon, ne_lat, zoom)
            tx_max, ty_max = _lonlat_to_tile(ne_lon, sw_lat, zoom)
            n_tiles = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)

    # 并行下载瓦片
    tile_coords = [(z, x, y) for z in [zoom]
                   for x in range(tx_min, tx_max + 1)
                   for y in range(ty_min, ty_max + 1)]

    tiles = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch_tile, source_name, z, x, y): (x, y)
                for z, x, y in tile_coords}
        for fut in futs:
            xy = futs[fut]
            result = fut.result()
            if result is not None:
                tiles[xy] = result

    if not tiles:
        return None

    # 拼接瓦片
    tile_size = 256
    canvas_w = (tx_max - tx_min + 1) * tile_size
    canvas_h = (ty_max - ty_min + 1) * tile_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for (tx, ty), tile_img in tiles.items():
        px = (tx - tx_min) * tile_size
        py = (ty - ty_min) * tile_size
        h, w = tile_img.shape[:2]
        canvas[py:py + h, px:px + w] = tile_img[:min(h, tile_size), :min(w, tile_size)]

    # 计算目标范围在拼接画布中的像素坐标
    # 瓦片左上角和右下角的经纬度
    canvas_lon_min, canvas_lat_max = _tile_to_lonlat(tx_min, ty_min, zoom)
    canvas_lon_max, canvas_lat_min = _tile_to_lonlat(tx_max + 1, ty_max + 1, zoom)

    # 目标范围在画布中的归一化位置
    def _lon_to_px(lon):
        return (lon - canvas_lon_min) / (canvas_lon_max - canvas_lon_min) * canvas_w

    def _lat_to_py(lat):
        # Mercator Y
        def merc(la):
            return math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
        merc_min = merc(canvas_lat_min)
        merc_max = merc(canvas_lat_max)
        return (1.0 - (merc(lat) - merc_min) / (merc_max - merc_min)) * canvas_h

    crop_x0 = int(_lon_to_px(sw_lon))
    crop_x1 = int(_lon_to_px(ne_lon))
    crop_y0 = int(_lat_to_py(ne_lat))
    crop_y1 = int(_lat_to_py(sw_lat))

    # 裁切并缩放到目标尺寸
    crop_x0 = max(0, min(crop_x0, canvas_w - 1))
    crop_x1 = max(crop_x0 + 1, min(crop_x1, canvas_w))
    crop_y0 = max(0, min(crop_y0, canvas_h - 1))
    crop_y1 = max(crop_y0 + 1, min(crop_y1, canvas_h))

    cropped = canvas[crop_y0:crop_y1, crop_x0:crop_x1]

    from PIL import Image
    img = Image.fromarray(cropped)
    img = img.resize((out_size, out_size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)
