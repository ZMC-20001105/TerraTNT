#!/usr/bin/env python
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


_DEFAULT_HIGHWAYS = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "road",
    "track",
]


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def _osmium_tags_filter(in_pbf: Path, out_pbf: Path, highway_types: list[str]) -> None:
    filters = [f"w/highway={h}" for h in highway_types]
    _run([
        "osmium",
        "tags-filter",
        "--overwrite",
        "-o",
        str(out_pbf),
        str(in_pbf),
        *filters,
    ])


def _osmium_merge(in_pbfs: list[Path], out_pbf: Path) -> None:
    _run([
        "osmium",
        "merge",
        "--overwrite",
        "-o",
        str(out_pbf),
        *[str(p) for p in in_pbfs],
    ])


def _osmium_export_geojson(in_pbf: Path, out_geojson: Path) -> None:
    _run([
        "osmium",
        "export",
        "--overwrite",
        "--geometry-types=linestring",
        "-f",
        "geojson",
        "-o",
        str(out_geojson),
        str(in_pbf),
    ])


def _osmium_extract_bbox(in_pbf: Path, out_pbf: Path, bbox_wgs84: tuple[float, float, float, float]) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    _run([
        "osmium",
        "extract",
        "--overwrite",
        f"--bbox={min_lon},{min_lat},{max_lon},{max_lat}",
        "-o",
        str(out_pbf),
        str(in_pbf),
    ])


def _iter_buffered_shapes_from_geojson(
    geojson_path: Path,
    src_crs: str,
    dst_crs: str,
    clip_box_dst,
    buffer_m: float,
):
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

            # quick bbox reject
            gb = geom_dst.bounds
            cb = clip_box_dst.bounds
            if (gb[2] < cb[0]) or (gb[0] > cb[2]) or (gb[3] < cb[1]) or (gb[1] > cb[3]):
                continue

            try:
                geom_buf = geom_dst.buffer(buffer_m)
            except Exception:
                continue
            if geom_buf.is_empty:
                continue

            yield (geom_buf, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build road_utm.tif from local OSM PBF using osmium-tool")
    parser.add_argument("--region", type=str, default="bohemian_forest")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GeoTIFF path (default: data/processed/utm_grid/{region}/road_utm.tif)",
    )
    parser.add_argument(
        "--pbf",
        type=str,
        action="append",
        required=True,
        help="Input .osm.pbf files (can be given multiple times)",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=7.5,
        help="Buffer distance (meters) applied to road centerlines before rasterize",
    )
    parser.add_argument(
        "--highway",
        type=str,
        action="append",
        default=None,
        help="Highway types to include (repeatable). Default is a drivable-road allowlist.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="If set, rasterize will burn all pixels touched by geometry. Default is false.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Prioritize DEM as reference for geometry alignment
    dem_path = project_root / "data/processed/utm_grid" / args.region / "dem_utm.tif"
    lulc_path = project_root / "data/processed/utm_grid" / args.region / "lulc_utm.tif"
    
    ref_path = None
    if dem_path.exists():
        ref_path = dem_path
        print(f"Using DEM as reference: {ref_path}")
    elif lulc_path.exists():
        ref_path = lulc_path
        print(f"Using LULC as reference: {ref_path}")
    else:
        raise FileNotFoundError(f"Neither dem_utm.tif nor lulc_utm.tif found in {dem_path.parent}")

    out_tif = Path(args.output) if args.output else (project_root / "data/processed/utm_grid" / args.region / "road_utm.tif")
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    pbfs = [Path(p) for p in args.pbf]
    for p in pbfs:
        if not p.exists():
            raise FileNotFoundError(f"PBF not found: {p}")

    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)
        ref_bounds = ref.bounds
        profile = ref.profile.copy()

    with tempfile.TemporaryDirectory(prefix="osmium_roads_") as td:
        td_path = Path(td)

        highway_types = list(_DEFAULT_HIGHWAYS if args.highway is None else args.highway)

        filtered = []
        for i, pbf in enumerate(pbfs):
            fp = td_path / f"filtered_{i}.osm.pbf"
            _osmium_tags_filter(pbf, fp, highway_types)
            filtered.append(fp)

        merged = td_path / "roads_merged.osm.pbf"
        if len(filtered) == 1:
            shutil.copyfile(filtered[0], merged)
        else:
            _osmium_merge(filtered, merged)

        # 将lulc范围从投影坐标转换成WGS84 bbox，用于裁剪PBF，避免后续导出/读取过大
        # 注意：这里用bbox裁剪足够，后续再在投影坐标下严格clip
        dst_to_wgs84 = pyproj.Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = ref_bounds
        lon1, lat1 = dst_to_wgs84.transform(minx, miny)
        lon2, lat2 = dst_to_wgs84.transform(maxx, maxy)
        min_lon, max_lon = (lon1, lon2) if lon1 < lon2 else (lon2, lon1)
        min_lat, max_lat = (lat1, lat2) if lat1 < lat2 else (lat2, lat1)

        clipped = td_path / "roads_clipped.osm.pbf"
        _osmium_extract_bbox(merged, clipped, (min_lon, min_lat, max_lon, max_lat))

        roads_geojson = td_path / "roads.geojson"
        _osmium_export_geojson(clipped, roads_geojson)

        clip_box = box(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
        shapes = _iter_buffered_shapes_from_geojson(
            roads_geojson,
            src_crs="EPSG:4326",
            dst_crs=ref_crs.to_string(),
            clip_box_dst=clip_box,
            buffer_m=float(args.buffer_m),
        )

        road = rasterize(
            shapes,
            out_shape=ref_shape,
            transform=ref_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=bool(args.all_touched),
        )

        profile.update(
            {
                "dtype": "uint8",
                "count": 1,
                "nodata": 0,
                "compress": "lzw",
            }
        )

        tmp_out = out_tif.with_suffix(out_tif.suffix + ".tmp")
        with rasterio.open(tmp_out, "w", **profile) as dst:
            dst.write(road, 1)

        if out_tif.exists():
            bak = out_tif.with_suffix(out_tif.suffix + ".bak")
            try:
                out_tif.replace(bak)
            except Exception:
                pass

        tmp_out.replace(out_tif)

    ratio = float(np.mean(road > 0))
    print(f"✓ road_utm.tif written: {out_tif}")
    print(f"  road pixel ratio: {ratio*100:.4f}%")


if __name__ == "__main__":
    main()
