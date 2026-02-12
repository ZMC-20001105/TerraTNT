#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""颜色方案和LULC着色"""
import numpy as np

# LULC类别颜色 (ESA WorldCover v200)
LULC_COLORS = {
    0:   (0, 0, 0),         # NoData - 黑
    10:  (0, 100, 0),       # Tree cover - 深绿
    20:  (255, 187, 34),    # Shrubland - 黄
    30:  (255, 255, 76),    # Grassland - 浅黄
    40:  (240, 150, 255),   # Cropland - 粉紫
    50:  (250, 0, 0),       # Built-up - 红
    60:  (180, 180, 180),   # Bare/sparse - 灰
    80:  (0, 100, 200),     # Water - 蓝
    90:  (0, 150, 160),     # Wetland - 青
    100: (250, 230, 160),   # Moss/lichen - 米
    255: (100, 100, 100),   # Unknown - 深灰
}

LULC_NAMES = {
    0: 'NoData', 10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland',
    40: 'Cropland', 50: 'Built-up', 60: 'Bare/sparse', 80: 'Water',
    90: 'Wetland', 100: 'Moss/lichen', 255: 'Unknown',
}

# 模型颜色 — 键名必须与 model_manager.discover_checkpoints() 一致
MODEL_COLORS = {
    'TerraTNT':      '#1f77b4',
    'V3_Waypoint':   '#ff7f0e',
    'V4_WP_Spatial': '#2ca02c',
    'V6_Autoreg':    '#d62728',
    'V6R_Robust':    '#9467bd',
    'V7_ConfGate':   '#8c564b',
    'LSTM_only':     '#e377c2',
    'LSTM_Env_Goal': '#7f7f7f',
    'Seq2Seq_Attn':  '#bcbd22',
    'MLP':           '#17becf',
    'ConstantVelocity': '#aec7e8',
}


def lulc_to_rgb(lulc: np.ndarray) -> np.ndarray:
    """将LULC分类栅格转为RGB图像 (H, W, 3) uint8"""
    h, w = lulc.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in LULC_COLORS.items():
        mask = lulc == val
        rgb[mask] = color
    return rgb


def hex_to_rgb(hex_color: str):
    """'#1f77b4' -> (31, 119, 180)"""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def terrain_hillshade(dem: np.ndarray, hillshade: np.ndarray) -> np.ndarray:
    """DEM高程着色 + hillshade混合, 生成美观的地形底图 (H,W,3) uint8
    
    使用自然地形配色: 绿(低) → 黄 → 棕 → 白(高)
    """
    valid = dem[np.isfinite(dem)]
    if len(valid) == 0:
        return np.stack([hillshade * 255] * 3, axis=-1).astype(np.uint8)
    vmin, vmax = np.nanpercentile(valid, [2, 98])
    if vmax - vmin < 1:
        vmax = vmin + 1
    norm = np.clip((np.nan_to_num(dem, nan=vmin) - vmin) / (vmax - vmin), 0, 1)

    # 自然地形色带: 深绿 → 浅绿 → 黄 → 棕 → 灰白
    stops = np.array([
        [30, 80, 30],     # 0.0 深绿 (低海拔)
        [60, 140, 50],    # 0.2 绿
        [170, 190, 80],   # 0.4 黄绿
        [200, 170, 100],  # 0.6 棕黄
        [170, 150, 130],  # 0.8 灰棕
        [230, 230, 230],  # 1.0 浅灰 (高海拔)
    ], dtype=np.float32)
    t = norm * (len(stops) - 1)
    idx = np.clip(t.astype(int), 0, len(stops) - 2)
    frac = (t - idx)[..., np.newaxis]
    color = stops[idx] * (1 - frac) + stops[idx + 1] * frac  # (H, W, 3)

    # 与hillshade混合: 70%色彩 + 30%明暗
    hs3 = hillshade[..., np.newaxis]  # (H, W, 1)
    blended = color * (0.5 + 0.5 * hs3)  # 明暗调制
    return np.clip(blended, 0, 255).astype(np.uint8)


def cost_map_colormap(slope: np.ndarray, lulc: np.ndarray,
                      road: np.ndarray = None) -> np.ndarray:
    """综合代价地图着色: 融合坡度代价 + LULC通行代价 + 道路折扣
    
    基于论文Table3-1的LULC代价权重和坡度约束
    输出: (H, W, 3) uint8, 绿(低代价/易通行) → 黄 → 红(高代价) → 黑(不可通行)
    """
    # LULC通行代价 (论文Table3-1)
    LULC_COST = {
        0: 5.0,    # NoData → 高代价
        10: 1.2,   # Cropland
        20: 1.8,   # Forest
        30: 1.0,   # Grassland
        40: 1.5,   # Shrubland
        50: 99.0,  # Wetland → 不可通行
        60: 99.0,  # Water → 不可通行
        80: 0.5,   # Artificial/Built-up
        90: 1.3,   # Bare
        100: 99.0, # Ice/Snow → 不可通行
        255: 5.0,  # Unknown
    }
    
    # 计算LULC代价层
    lulc_cost = np.full(lulc.shape, 5.0, dtype=np.float32)
    for val, cost in LULC_COST.items():
        lulc_cost[lulc == val] = cost
    
    # 坡度代价: 0°→0, 30°→1, >45°→不可通行
    slope_cost = np.clip(slope / 30.0, 0, 1)
    slope_cost[slope > 45] = 99.0
    
    # 综合代价 = LULC代价 * (1 + 坡度代价)
    total_cost = lulc_cost * (1.0 + slope_cost)
    
    # 道路折扣: 道路上代价降低80%
    if road is not None:
        total_cost[road > 0] *= 0.2
    
    # 标记不可通行
    impassable = total_cost > 50.0
    
    # 归一化到 [0, 1] (排除不可通行)
    passable_costs = total_cost[~impassable]
    if len(passable_costs) > 0:
        vmin, vmax_c = np.percentile(passable_costs, [2, 98])
        if vmax_c - vmin < 0.1:
            vmax_c = vmin + 1.0
        norm = np.clip((total_cost - vmin) / (vmax_c - vmin), 0, 1)
    else:
        norm = np.ones_like(total_cost)
    
    # 色带: 深绿(低代价) → 黄绿 → 橙 → 红(高代价)
    stops = np.array([
        [20, 120, 20],    # 0.0 深绿 (最易通行)
        [80, 180, 40],    # 0.25 绿
        [200, 200, 50],   # 0.5 黄
        [220, 130, 30],   # 0.75 橙
        [180, 40, 20],    # 1.0 红 (高代价)
    ], dtype=np.float32)
    t = norm * (len(stops) - 1)
    idx = np.clip(t.astype(int), 0, len(stops) - 2)
    frac = (t - idx)[..., np.newaxis]
    rgb = stops[idx] * (1 - frac) + stops[idx + 1] * frac
    
    # 不可通行区域 → 深灰
    rgb[impassable] = [40, 40, 40]
    
    # 道路高亮叠加
    if road is not None:
        road_mask = road > 0
        rgb[road_mask] = rgb[road_mask] * 0.4 + np.array([255, 220, 100]) * 0.6
    
    return np.clip(rgb, 0, 255).astype(np.uint8)


def slope_colormap(slope: np.ndarray, vmax: float = 45.0) -> np.ndarray:
    """坡度着色: 绿(平) -> 黄 -> 红(陡)"""
    norm = np.clip(slope / vmax, 0, 1)
    r = np.clip(norm * 2, 0, 1)
    g = np.clip(2 - norm * 2, 0, 1)
    b = np.zeros_like(norm)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)
