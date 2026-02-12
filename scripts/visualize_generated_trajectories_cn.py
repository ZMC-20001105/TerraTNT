#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.plot_config import get_plot_config
from config import get_path


def _pick_traj_file(data_dir: Path, traj_file: str | None) -> Path:
    if traj_file:
        p = Path(traj_file)
        if not p.is_absolute():
            p = data_dir / p
        return p

    files = sorted(data_dir.glob('traj_*.pkl'))
    if not files:
        raise FileNotFoundError(f'未找到轨迹文件: {data_dir}/traj_*.pkl')
    return files[min(2, len(files) - 1)]


def _to_relative_km(path_utm_m: np.ndarray) -> np.ndarray:
    path_km = path_utm_m / 1000.0
    origin = np.array([path_km[:, 0].min(), path_km[:, 1].min()], dtype=np.float64)
    return path_km - origin


def _square_axis(ax, enabled: bool):
    if not enabled:
        return
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect(1)


def _infer_region_from_data_dir(data_dir: Path) -> str:
    return data_dir.name


def _read_processed_raster_patch(
    region: str,
    layer: str,
    bounds_utm_m: tuple[float, float, float, float],
    out_size: int = 512,
):
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling

    utm_dir = Path(get_path('paths.processed.utm_grid')) / region
    layer_to_file = {
        'dem': 'dem_utm.tif',
        'slope': 'slope_utm.tif',
        'aspect': 'aspect_utm.tif',
        'lulc': 'lulc_utm.tif',
    }
    tif_name = layer_to_file.get(layer, 'dem_utm.tif')
    tif_path = utm_dir / tif_name

    left, bottom, right, top = bounds_utm_m
    with rasterio.open(tif_path) as src:
        # clamp to raster bounds to avoid empty windows
        left = max(left, src.bounds.left)
        right = min(right, src.bounds.right)
        bottom = max(bottom, src.bounds.bottom)
        top = min(top, src.bounds.top)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data = src.read(
            1,
            window=window,
            out_shape=(out_size, out_size),
            resampling=Resampling.bilinear if layer != 'lulc' else Resampling.nearest,
        )
        nodata = src.nodata

    data = data.astype(np.float32)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    # TerraTNT 的 DEM nodata 常见为 -32768
    if layer == 'dem':
        data = np.where(data <= -32000, np.nan, data)
    return data


def _map_lulc_values_to_index(lulc_values: np.ndarray, lulc_classes: list[int]) -> np.ndarray:
    """Map raw LULC codes to 0..(K-1) indices for colormap display."""
    lut = {int(v): i for i, v in enumerate(lulc_classes)}
    unknown_idx = lut.get(255, len(lulc_classes) - 1)
    out = np.full(lulc_values.shape, unknown_idx, dtype=np.int16)
    for v, i in lut.items():
        out[lulc_values == v] = i
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed/complete_dataset_10s/bohemian_forest')
    parser.add_argument('--traj-file', default=None)
    parser.add_argument('--out', default='outputs/figures/tests/latest_traj_viz_cn.png')
    args = parser.parse_args()

    plot_cfg = get_plot_config()

    data_dir = Path(args.data_dir)
    traj_path = _pick_traj_file(data_dir, args.traj_file)

    with open(traj_path, 'rb') as f:
        traj = pickle.load(f)

    region = str(traj.get('region', _infer_region_from_data_dir(data_dir)))

    path_abs_m = np.asarray(traj['path'], dtype=np.float64)
    path_rel_km = _to_relative_km(path_abs_m)

    speeds_mps = np.asarray(traj['speeds'], dtype=np.float64)
    speeds_kmh = speeds_mps * 3.6

    ts_min = np.asarray(traj['timestamps'], dtype=np.float64) / 60.0

    s0 = traj['samples'][0]
    env = np.asarray(s0['env_map_100km'], dtype=np.float32)

    square_panels = bool(getattr(plot_cfg, 'SQUARE_PANELS', True))
    panel_in = float(getattr(plot_cfg, 'PANEL_SIZE_INCH', 4.5))
    figsize = (2 * panel_in, 2 * panel_in) if square_panels else (18, 10)
    bg_cfg = (getattr(plot_cfg, 'TRAJ_PLOT', {}) or {}).get('background', {})
    bg_enabled = bool(bg_cfg.get('enabled', True))
    bg_alpha = float(bg_cfg.get('alpha', 0.55))

    # 背景覆盖范围：以“观测段末点(current_pos_abs)”为中心，自动扩展到覆盖整条轨迹
    coverage_mode = str(bg_cfg.get('coverage_mode', 'auto'))
    min_coverage_km = float(bg_cfg.get('min_coverage_km', 100.0))
    padding_km = float(bg_cfg.get('padding_km', 5.0))
    center_abs = np.asarray(s0['current_pos_abs'], dtype=np.float64)
    dx_km = np.abs(path_abs_m[:, 0] - center_abs[0]) / 1000.0
    dy_km = np.abs(path_abs_m[:, 1] - center_abs[1]) / 1000.0
    radius_km = float(max(dx_km.max(initial=0.0), dy_km.max(initial=0.0)))
    if coverage_mode == 'auto':
        patch_coverage_km = max(min_coverage_km, 2.0 * (radius_km + padding_km))
    else:
        patch_coverage_km = min_coverage_km
    half_km = patch_coverage_km / 2.0
    patch_origin_abs_m = center_abs - np.array([half_km * 1000.0, half_km * 1000.0], dtype=np.float64)
    path_rel_km = (path_abs_m - patch_origin_abs_m) / 1000.0

    # Background bounds (UTM meters)
    left = float(center_abs[0] - half_km * 1000.0)
    right = float(center_abs[0] + half_km * 1000.0)
    bottom = float(center_abs[1] - half_km * 1000.0)
    top = float(center_abs[1] + half_km * 1000.0)

    # Read processed raster patches once and reuse across panels to avoid mismatch
    layer = str(bg_cfg.get('layer', 'dem')).lower()
    dem_bg = None
    lulc_bg = None
    if bg_enabled:
        dem_bg = _read_processed_raster_patch(region, layer, (left, bottom, right, top), out_size=512)
        lulc_bg = _read_processed_raster_patch(region, 'lulc', (left, bottom, right, top), out_size=512)
        if lulc_bg is not None:
            # 将 NaN/无效像素映射为 unknown(255)，再转 int，避免警告与错误分类
            lulc_bg = np.where(np.isfinite(lulc_bg), lulc_bg, 255.0)
            lulc_bg = np.rint(lulc_bg).astype(np.int32, copy=False)

    fig = plt.figure(figsize=figsize, dpi=plot_cfg.DPI)

    ax1 = fig.add_subplot(2, 2, 1)
    # 背景：使用经过处理的环境栅格（这里用 env_map_100km 的 DEM 通道作为底图）
    if bg_enabled:
        dem = dem_bg if dem_bg is not None else env[0]
        vmin, vmax = np.nanpercentile(dem, [2, 98]) if np.any(np.isfinite(dem)) else (0.0, 1.0)
        ax1.imshow(
            dem,
            cmap='terrain',
            interpolation='bilinear',
            alpha=bg_alpha,
            origin='upper',
            extent=[0.0, patch_coverage_km, 0.0, patch_coverage_km],
            vmin=vmin,
            vmax=vmax,
        )

    sc = ax1.scatter(path_rel_km[:, 0], path_rel_km[:, 1], c=speeds_kmh, cmap='viridis', s=6, alpha=0.90)
    ax1.plot(path_rel_km[:, 0], path_rel_km[:, 1], color='k', alpha=0.35, lw=0.8)
    ax1.scatter([path_rel_km[0, 0]], [path_rel_km[0, 1]], c='g', s=80, label='起点', edgecolors='white', linewidths=1)
    ax1.scatter([path_rel_km[-1, 0]], [path_rel_km[-1, 1]], c='r', s=80, label='终点', edgecolors='white', linewidths=1)
    fig.colorbar(sc, ax=ax1, label='速度 (km/h)')
    ax1.set_title('轨迹路径（相对坐标，单位 km）', fontweight='bold')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_xlim(0.0, patch_coverage_km)
    ax1.set_ylim(0.0, patch_coverage_km)
    ax1.set_aspect('equal', adjustable='box')
    _square_axis(ax1, square_panels)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='best')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(ts_min, speeds_kmh, color=plot_cfg.PRIMARY, lw=1.4)
    cal = traj.get('speed_calibration', {})
    cal_mean_kmh = float(cal.get('calibrated_mean', np.nan)) * 3.6
    target_kmh = float(cal.get('target_speed', np.nan)) * 3.6
    if np.isfinite(cal_mean_kmh):
        ax2.axhline(cal_mean_kmh, color=plot_cfg.ERROR, ls='--', lw=2, label=f'均值 {cal_mean_kmh:.1f} km/h')
    if np.isfinite(target_kmh):
        ax2.axhline(target_kmh, color=plot_cfg.SUCCESS, ls='--', lw=2, label=f'目标 {target_kmh:.1f} km/h')
    ax2.set_title('速度曲线（km/h）', fontweight='bold')
    ax2.set_xlabel('时间 (分钟)')
    ax2.set_ylabel('速度 (km/h)')
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='best')
    _square_axis(ax2, square_panels)

    ax3 = fig.add_subplot(2, 2, 3)
    if bg_enabled and dem_bg is not None:
        dem_panel = dem_bg
        title = f'环境：DEM（覆盖 {patch_coverage_km:.0f}km×{patch_coverage_km:.0f}km）'
    else:
        dem_panel = env[0]
        title = '环境：DEM（100km×100km）'
    im3 = ax3.imshow(dem_panel, cmap='terrain', interpolation='bilinear', origin='upper')
    ax3.set_title(title, fontweight='bold')
    ax3.set_axis_off()
    fig.colorbar(im3, ax=ax3)
    _square_axis(ax3, square_panels)

    ax4 = fig.add_subplot(2, 2, 4)
    lulc_cmap, lulc_classes = plot_cfg.get_lulc_cmap()
    if bg_enabled and lulc_bg is not None:
        # Unknown values map to 255
        known_mask = np.isin(lulc_bg, np.array(lulc_classes, dtype=np.int32))
        lulc_vals = np.where(known_mask, lulc_bg, 255)
        lulc_idx = _map_lulc_values_to_index(lulc_vals, lulc_classes)
        title = f'环境：LULC（覆盖 {patch_coverage_km:.0f}km×{patch_coverage_km:.0f}km）'
    else:
        # fallback: use env_map one-hot
        lulc_idx = np.argmax(env[4:14], axis=0)
        title = '环境：LULC（含 unknown=255）'
    im4 = ax4.imshow(lulc_idx, cmap=lulc_cmap, interpolation='nearest', origin='upper', vmin=0, vmax=len(lulc_classes) - 1)
    ax4.set_title(title, fontweight='bold')
    ax4.set_axis_off()
    cb4 = fig.colorbar(im4, ax=ax4, ticks=list(range(len(lulc_classes))))
    cb4.ax.set_yticklabels([str(c) for c in lulc_classes])
    _square_axis(ax4, square_panels)

    length_km = float(traj['length']) / 1000.0
    duration_min = float(traj['duration']) / 60.0
    fig.suptitle(f"{traj_path.stem} | 长度 {length_km:.2f} km | 时长 {duration_min:.1f} 分钟", fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(str(out_path))


if __name__ == '__main__':
    main()
