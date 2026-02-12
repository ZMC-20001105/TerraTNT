"""
为Bohemian Forest生成所有必需的数据文件
包括：passable_mask, cost_map等
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import rasterio
import logging
from config import cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_passable_masks(region='bohemian_forest'):
    """生成可通行域掩码"""
    logger.info("=" * 60)
    logger.info("生成可通行域掩码")
    logger.info("=" * 60)
    
    utm_dir = Path(f'/home/zmc/文档/programwork/data/processed/utm_grid/{region}')
    
    # 读取LULC和slope
    with rasterio.open(utm_dir / 'lulc_utm.tif') as src:
        lulc = src.read(1)
        profile = src.profile.copy()
    
    with rasterio.open(utm_dir / 'slope_utm.tif') as src:
        slope = src.read(1)
    
    logger.info(f"LULC尺寸: {lulc.shape}")
    logger.info(f"Slope尺寸: {slope.shape}")
    
    # 确保尺寸一致（裁剪到较小的尺寸）
    min_height = min(lulc.shape[0], slope.shape[0])
    min_width = min(lulc.shape[1], slope.shape[1])
    lulc = lulc[:min_height, :min_width]
    slope = slope[:min_height, :min_width]
    logger.info(f"统一尺寸: {lulc.shape}")
    
    # 不可通行的LULC类型
    impassable_lulc = [50, 60, 100, 200]  # wetland, water, ice/snow, ocean
    
    # 为每种车辆类型生成掩码
    vehicle_types = {
        'type1': 30,  # 最大坡度30度
        'type2': 25,
        'type3': 20,
        'type4': 15
    }
    
    for vtype, max_slope in vehicle_types.items():
        logger.info(f"\n生成 {vtype} 的可通行域掩码（最大坡度: {max_slope}°）")
        
        # LULC可通行
        lulc_passable = ~np.isin(lulc, impassable_lulc)
        
        # 坡度可通行
        slope_passable = slope <= max_slope
        
        # 组合
        passable = lulc_passable & slope_passable
        
        passable_count = passable.sum()
        total_count = passable.size
        passable_ratio = passable_count / total_count * 100
        
        logger.info(f"  可通行像素: {passable_count}/{total_count} ({passable_ratio:.2f}%)")
        
        # 保存
        output_path = utm_dir / f'passable_mask_{vtype}.tif'
        profile.update(dtype=rasterio.uint8, compress='lzw', height=min_height, width=min_width)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(passable.astype(np.uint8), 1)
        
        logger.info(f"  ✓ 保存: {output_path.name}")

def generate_cost_maps(region='bohemian_forest'):
    """生成代价图"""
    logger.info("\n" + "=" * 60)
    logger.info("生成代价图")
    logger.info("=" * 60)
    
    utm_dir = Path(f'/home/zmc/文档/programwork/data/processed/utm_grid/{region}')
    
    # 读取环境数据
    with rasterio.open(utm_dir / 'lulc_utm.tif') as src:
        lulc = src.read(1)
        profile = src.profile.copy()
    
    with rasterio.open(utm_dir / 'slope_utm.tif') as src:
        slope = src.read(1)
    
    # 读取道路栅格
    road_utm_path = utm_dir / 'road_utm.tif'
    if road_utm_path.exists():
        with rasterio.open(road_utm_path) as src:
            road_raster = src.read(1)
    else:
        logger.warning(f"road_utm.tif不存在，使用LULC==80作为临时道路proxy")
        road_raster = (lulc == 80).astype(np.float32)
    
    # 确保尺寸一致
    min_height = min(lulc.shape[0], slope.shape[0])
    min_width = min(lulc.shape[1], slope.shape[1])
    lulc = lulc[:min_height, :min_width]
    slope = slope[:min_height, :min_width]
    
    # LULC代价
    lulc_cost_map = {
        10: 1.2,   # cropland
        20: 1.8,   # forest
        30: 1.0,   # grassland
        40: 1.5,   # shrubland
        50: np.inf, # wetland
        60: np.inf, # water
        70: 2.0,   # tundra
        80: 0.5,   # artificial
        90: 1.3,   # bare
        100: np.inf, # ice/snow
        200: np.inf  # ocean
    }
    
    # 战术意图权重
    intents = {
        'intent1': {'dist': 1.0, 'slope': 0.5, 'lulc': 1.0, 'exp': 0.0},  # 距离优先
        'intent2': {'dist': 0.3, 'slope': 0.3, 'lulc': 0.5, 'exp': 2.0},  # 隐蔽优先
        'intent3': {'dist': 0.5, 'slope': 2.0, 'lulc': 1.0, 'exp': 0.5}   # 地形优先
    }
    
    vehicle_types = {
        'type1': 30,
        'type2': 25,
        'type3': 20,
        'type4': 15
    }

    # 道路偏好（LULC=80），倍率越小越偏好走道路
    road_mult_by_vehicle = {
        'type1': 1.0,
        'type2': 0.8,
        'type3': 0.6,
        'type4': 0.4,
    }
    
    # 创建LULC代价栅格
    lulc_cost = np.ones_like(lulc, dtype=float)
    for code, cost in lulc_cost_map.items():
        lulc_cost[lulc == code] = cost
    
    # 为每种配置生成代价图
    for intent_name, weights in intents.items():
        for vtype, max_slope in vehicle_types.items():
            logger.info(f"\n生成 {intent_name}_{vtype} 代价图")
            
            # 基础代价（LULC）
            cost = lulc_cost * weights['lulc']
            
            # 坡度代价
            slope_cost = slope * weights['slope']
            cost = cost + slope_cost

            # 道路偏好：降低道路像素代价，鼓励走道路（不同车型强度不同）
            road_mult = float(road_mult_by_vehicle.get(vtype, 1.0))
            road_mask = (road_raster > 0)
            if np.any(road_mask):
                cost[road_mask] = cost[road_mask] * road_mult
            
            # 不可通行区域设为无穷
            cost[slope > max_slope] = np.inf
            cost[np.isin(lulc, [50, 60, 100, 200])] = np.inf
            
            passable_ratio = np.isfinite(cost).sum() / cost.size * 100
            logger.info(f"  可通行区域: {passable_ratio:.2f}%")
            logger.info(f"  代价范围: {cost[np.isfinite(cost)].min():.2f} ~ {cost[np.isfinite(cost)].max():.2f}")
            
            # 保存
            output_path = utm_dir / f'cost_map_{intent_name}_{vtype}.tif'
            profile.update(dtype=rasterio.float32, compress='lzw', height=min_height, width=min_width)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(cost.astype(np.float32), 1)
            
            logger.info(f"  ✓ 保存: {output_path.name}")

def main():
    region = 'bohemian_forest'
    
    logger.info("=" * 60)
    logger.info(f"准备 {region} 区域数据")
    logger.info("=" * 60)
    
    # 生成可通行域掩码
    generate_passable_masks(region)
    
    # 生成代价图
    generate_cost_maps(region)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 所有数据准备完成！")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
