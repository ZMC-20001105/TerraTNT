#!/usr/bin/env python3
"""生成图3.2b: 成本地图四子图(DEM/LULC/Slope/Cost)"""
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / 'outputs' / 'paper_final'
UTM = ROOT / 'data/processed/utm_grid/bohemian_forest'

try:
    import rasterio
except ImportError:
    print('需要 rasterio: pip install rasterio')
    exit(1)

def save(fig, name):
    fig.savefig(FIGS / f'{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {name}')

def load_tif(name):
    fp = UTM / name
    if not fp.exists():
        return None, None
    with rasterio.open(fp) as src:
        data = src.read(1)
        bounds = src.bounds
    return data, bounds

# Subsample for display (full resolution too large)
def subsample(arr, factor=10):
    return arr[::factor, ::factor]

print('--- 图3.2b: 成本地图生成流程 ---')

dem, bounds = load_tif('dem_utm.tif')
slope, _ = load_tif('slope_utm.tif')
lulc, _ = load_tif('lulc_utm.tif')
cost, _ = load_tif('cost_map_intent1_type1.tif')

if dem is None:
    print('  ⚠️ 无DEM数据')
    exit(0)

# Subsample
f = 10
dem_s = subsample(dem, f)
slope_s = subsample(slope, f) if slope is not None else None
lulc_s = subsample(lulc, f) if lulc is not None else None
cost_s = subsample(cost, f) if cost is not None else None

# Replace nodata
dem_s = np.where(dem_s < -1000, np.nan, dem_s)
if slope_s is not None:
    slope_s = np.where(slope_s < 0, np.nan, slope_s)
if cost_s is not None:
    cost_s = np.where(cost_s < 0, np.nan, cost_s)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (a) DEM
ax = axes[0, 0]
im = ax.imshow(dem_s, cmap='terrain', aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='高程 (m)')
ax.set_title('(a) DEM高程数据', fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])

# (b) LULC
ax = axes[0, 1]
if lulc_s is not None:
    # GlobeLand30 color scheme
    lulc_colors = {
        10: ('#006400', '农田'), 20: ('#FFD700', '森林'),
        30: ('#90EE90', '草地'), 40: ('#8B4513', '灌木'),
        50: ('#FF6347', '湿地'), 60: ('#0000FF', '水体'),
        70: ('#808080', '苔原'), 80: ('#FFFFFF', '裸地'),
        90: ('#FF0000', '冰雪'), 100: ('#A9A9A9', '不透水面'),
    }
    im = ax.imshow(lulc_s, cmap='Set3', aspect='auto', vmin=0, vmax=110)
    plt.colorbar(im, ax=ax, shrink=0.8, label='LULC类别')
else:
    ax.text(0.5, 0.5, '无LULC数据', transform=ax.transAxes, ha='center')
ax.set_title('(b) 地表覆盖分类 (GlobeLand30)', fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])

# (c) Slope
ax = axes[1, 0]
if slope_s is not None:
    im = ax.imshow(np.clip(slope_s, 0, 40), cmap='YlOrRd', aspect='auto', vmin=0, vmax=40)
    plt.colorbar(im, ax=ax, shrink=0.8, label='坡度 (°)')
else:
    ax.text(0.5, 0.5, '无坡度数据', transform=ax.transAxes, ha='center')
ax.set_title('(c) 地形坡度', fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])

# (d) Cost map
ax = axes[1, 1]
if cost_s is not None:
    # Clip to reasonable range
    vmax = np.nanpercentile(cost_s[cost_s < 1e6], 95) if np.any(cost_s < 1e6) else 1.0
    im = ax.imshow(np.clip(cost_s, 0, vmax), cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, label='通行成本')
else:
    ax.text(0.5, 0.5, '无成本地图', transform=ax.transAxes, ha='center')
ax.set_title('(d) 综合通行成本地图', fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('环境数据处理流程 (波西米亚森林区域)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, 'fig3_2b_cost_map')

print('\n完成!')
