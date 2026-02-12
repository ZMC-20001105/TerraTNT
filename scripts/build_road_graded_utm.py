#!/usr/bin/env python
"""
生成分级道路栅格 road_graded_utm.tif

像素值编码道路等级：
  0 = 无道路
  1 = 高等级 (motorway, trunk, primary)
  2 = 中等级 (secondary, tertiary)
  3 = 低等级 (unclassified, residential, service, living_street, road, track)

同时生成 road_utm.tif（二值兼容旧代码）。
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, box
from shapely.ops import transform as shp_transform
import fiona

ROAD_GRADES = {
    'high':   ['motorway', 'trunk', 'primary'],
    'medium': ['secondary', 'tertiary'],
    'low':    ['unclassified', 'residential', 'service', 'living_street', 'road', 'track'],
}

GRADE_VALUE = {'high': 1, 'medium': 2, 'low': 3}


def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def _osmium_tags_filter(in_pbf, out_pbf, highway_types):
    filters = [f"w/highway={h}" for h in highway_types]
    _run(["osmium", "tags-filter", "--overwrite", "-o", str(out_pbf), str(in_pbf), *filters])


def _osmium_merge(in_pbfs, out_pbf):
    _run(["osmium", "merge", "--overwrite", "-o", str(out_pbf), *[str(p) for p in in_pbfs]])


def _osmium_export_geojson(in_pbf, out_geojson):
    _run(["osmium", "export", "--overwrite", "--geometry-types=linestring",
          "-f", "geojson", "-o", str(out_geojson), str(in_pbf)])


def _osmium_extract_bbox(in_pbf, out_pbf, bbox_wgs84):
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    _run(["osmium", "extract", "--overwrite",
          f"--bbox={min_lon},{min_lat},{max_lon},{max_lat}",
          "-o", str(out_pbf), str(in_pbf)])


def _iter_shapes(geojson_path, src_crs, dst_crs, clip_box, buffer_m, grade_value):
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    def _proj(x, y, z=None):
        return transformer.transform(x, y)
    with fiona.open(geojson_path) as src:
        for feat in src:
            geom_mapping = feat.get("geometry")
            if not geom_mapping:
                continue
            geom = shape(geom_mapping)
            if geom.is_empty:
                continue
            try:
                geom_dst = shp_transform(_proj, geom)
            except Exception:
                continue
            if geom_dst.is_empty:
                continue
            gb = geom_dst.bounds
            cb = clip_box.bounds
            if gb[2] < cb[0] or gb[0] > cb[2] or gb[3] < cb[1] or gb[1] > cb[3]:
                continue
            try:
                geom_buf = geom_dst.buffer(buffer_m)
            except Exception:
                continue
            if geom_buf.is_empty:
                continue
            yield (geom_buf, grade_value)


def main():
    parser = argparse.ArgumentParser(description="Build graded road_utm.tif from OSM PBF")
    parser.add_argument("--region", type=str, default="bohemian_forest")
    parser.add_argument("--pbf", type=str, action="append", required=True)
    parser.add_argument("--buffer-m", type=float, default=7.5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dem_path = project_root / "data/processed/utm_grid" / args.region / "dem_utm.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    out_graded = project_root / "data/processed/utm_grid" / args.region / "road_graded_utm.tif"
    out_binary = project_root / "data/processed/utm_grid" / args.region / "road_utm.tif"

    pbfs = [Path(p) for p in args.pbf]
    for p in pbfs:
        if not p.exists():
            raise FileNotFoundError(f"PBF not found: {p}")

    with rasterio.open(dem_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)
        ref_bounds = ref.bounds
        profile = ref.profile.copy()

    dst_to_wgs84 = pyproj.Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
    lon1, lat1 = dst_to_wgs84.transform(ref_bounds.left, ref_bounds.bottom)
    lon2, lat2 = dst_to_wgs84.transform(ref_bounds.right, ref_bounds.top)
    bbox_wgs84 = (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

    clip_box = box(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)

    # 初始化分级栅格（高等级覆盖低等级）
    graded = np.zeros(ref_shape, dtype=np.uint8)

    with tempfile.TemporaryDirectory(prefix="osmium_graded_") as td:
        td_path = Path(td)

        # 按等级从低到高处理（高等级覆盖低等级）
        for grade_name in ['low', 'medium', 'high']:
            highway_types = ROAD_GRADES[grade_name]
            grade_val = GRADE_VALUE[grade_name]
            print(f"\n处理 {grade_name} 等级道路: {highway_types}")

            filtered = []
            for i, pbf in enumerate(pbfs):
                fp = td_path / f"filtered_{grade_name}_{i}.osm.pbf"
                _osmium_tags_filter(pbf, fp, highway_types)
                filtered.append(fp)

            merged = td_path / f"roads_{grade_name}.osm.pbf"
            if len(filtered) == 1:
                shutil.copyfile(filtered[0], merged)
            else:
                _osmium_merge(filtered, merged)

            clipped = td_path / f"roads_{grade_name}_clip.osm.pbf"
            _osmium_extract_bbox(merged, clipped, bbox_wgs84)

            geojson = td_path / f"roads_{grade_name}.geojson"
            _osmium_export_geojson(clipped, geojson)

            shapes = list(_iter_shapes(
                geojson, "EPSG:4326", ref_crs.to_string(),
                clip_box, float(args.buffer_m), grade_val
            ))
            print(f"  {grade_name}: {len(shapes)} 个几何体")

            if shapes:
                layer = rasterize(shapes, out_shape=ref_shape, transform=ref_transform,
                                  fill=0, dtype=np.uint8, all_touched=False)
                # 高等级覆盖低等级
                graded[layer > 0] = grade_val

    # 统计
    for grade_name, grade_val in GRADE_VALUE.items():
        count = int(np.sum(graded == grade_val))
        pct = count / graded.size * 100
        print(f"  {grade_name}(={grade_val}): {count} 像素 ({pct:.2f}%)")
    total_road = int(np.sum(graded > 0))
    print(f"  总道路: {total_road} 像素 ({total_road/graded.size*100:.2f}%)")

    # 保存分级栅格
    profile.update({"dtype": "uint8", "count": 1, "nodata": 0, "compress": "lzw"})
    with rasterio.open(out_graded, "w", **profile) as dst:
        dst.write(graded, 1)
    print(f"\n✓ 分级道路栅格: {out_graded}")

    # 同时生成二值兼容版本
    binary = (graded > 0).astype(np.uint8)
    with rasterio.open(out_binary, "w", **profile) as dst:
        dst.write(binary, 1)
    print(f"✓ 二值道路栅格: {out_binary}")


if __name__ == "__main__":
    main()
