"""
环境代价模型 (Cost Map)

按论文第3章要求：
总代价 = w_dist * f_dist + w_slope * f_slope + w_lulc * f_lulc + w_exp * f_exp

1. 距离代价：f_dist = 欧氏距离
2. 坡度代价：f_slope = |h_i - h_{i+1}| / d * slope_factor
3. 地表代价：f_lulc = avg(c_lulc)，基于表3.1
4. 暴露代价：f_exp = (1 - (veg_i + veg_{i+1})/2) * exp_factor

3种战术意图权重（表3.3）：
- Intent1（最短时间）: dist=1.0, slope=0.5, lulc=1.0, exp=0.0
- Intent2（隐蔽）: dist=0.3, slope=0.3, lulc=0.5, exp=2.0
- Intent3（避陡坡）: dist=0.5, slope=2.0, lulc=1.0, exp=0.5
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import rasterio
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_dilation

from config import cfg, get_path
from utils.trajectory_generation.passable_mask import VEHICLE_MAX_SLOPE

logger = logging.getLogger(__name__)

# 论文表3.1：LULC基础代价系数
LULC_BASE_COST = {
    10: 1.2,   # 耕地
    20: 1.8,   # 森林
    30: 1.0,   # 草地
    40: 1.5,   # 灌木
    50: 999.0, # 湿地（不可通行）
    60: 999.0, # 水体（不可通行）
    70: 2.0,   # 苔原
    80: 0.5,   # 人造地表
    90: 1.3,   # 裸地
    100: 999.0 # 冰雪（不可通行）
}

# 论文表3.3：战术意图权重
INTENT_WEIGHTS = {
    'intent1': {'dist': 1.0, 'slope': 0.5, 'lulc': 1.0, 'exp': 0.0},  # 最短时间
    'intent2': {'dist': 0.3, 'slope': 0.3, 'lulc': 0.5, 'exp': 2.0},  # 隐蔽
    'intent3': {'dist': 0.5, 'slope': 2.0, 'lulc': 1.0, 'exp': 0.5}   # 避陡坡
}


@dataclass
class CostFactors:
    """代价因子"""
    slope_factor: float = 1.0
    exposure_factor: float = 1.0


class CostMapGenerator:
    """环境代价图生成器"""
    
    def __init__(self, region: str = 'scottish_highlands'):
        self.region = region
        
        # 加载UTM栅格
        utm_dir = Path(get_path('paths.processed.utm_grid')) / region
        
        logger.info(f"加载 {region} 环境栅格数据...")
        self.dem_src = rasterio.open(utm_dir / 'dem_utm.tif')
        self.slope_src = rasterio.open(utm_dir / 'slope_utm.tif')
        self.lulc_src = rasterio.open(utm_dir / 'lulc_utm.tif')
        
        # 读取数据
        self.dem = self.dem_src.read(1)
        self.slope = self.slope_src.read(1)
        self.lulc = self.lulc_src.read(1)

        shapes = [self.dem.shape, self.slope.shape, self.lulc.shape]
        if len({s for s in shapes}) != 1:
            h = min(s[0] for s in shapes)
            w = min(s[1] for s in shapes)
            logger.warning(f"栅格shape不一致，将裁剪到共同尺寸: {(h, w)}; shapes={shapes}")
            self.dem = self.dem[:h, :w]
            self.slope = self.slope[:h, :w]
            self.lulc = self.lulc[:h, :w]
        
        # 处理nodata
        self.dem = np.where(self.dem == -32768, np.nan, self.dem)

        self.shape = self.dem.shape
        self.transform = self.dem_src.transform
        self.crs = self.dem_src.crs
        self.resolution = self.transform.a  # 30m
        
        logger.info(f"  栅格尺寸: {self.shape}")
        logger.info(f"  分辨率: {self.resolution}m")
    
    def __del__(self):
        """关闭栅格文件"""
        if hasattr(self, 'dem_src'):
            self.dem_src.close()
        if hasattr(self, 'slope_src'):
            self.slope_src.close()
        if hasattr(self, 'lulc_src'):
            self.lulc_src.close()
    
    def compute_lulc_cost_map(self) -> np.ndarray:
        """
        计算LULC基础代价图
        
        Returns:
            代价图（归一化到0-1）
        """
        logger.info("计算LULC基础代价图...")
        
        cost_map = np.zeros(self.shape, dtype=np.float32)
        
        for lulc_class, cost in LULC_BASE_COST.items():
            mask = self.lulc == lulc_class
            count = np.sum(mask)
            if count > 0:
                cost_map[mask] = cost
                lulc_name = {
                    10: '耕地', 20: '森林', 30: '草地', 40: '灌木',
                    50: '湿地', 60: '水体', 70: '苔原', 80: '人造',
                    90: '裸地', 100: '冰雪'
                }.get(lulc_class, str(lulc_class))
                logger.info(f"  {lulc_name}({lulc_class}): 代价={cost:.1f}, {count}像素")
        
        # 不归一化，保留原始代价值，并添加最小基础代价确保高于道路
        passable_mask = cost_map < 900
        min_terrain_cost = 0.1  # 最小地形代价，确保高于道路代价0.001
        if np.any(passable_mask):
            # 将原始代价(0.5-2.0)映射到(0.1-2.0)范围，确保最低代价仍为0.1
            min_cost = cost_map[passable_mask].min()
            max_cost = cost_map[passable_mask].max()
            if max_cost > min_cost:
                # 归一化到0-1，然后映射到[min_terrain_cost, max_cost]
                normalized = (cost_map[passable_mask] - min_cost) / (max_cost - min_cost)
                cost_map[passable_mask] = min_terrain_cost + normalized * (max_cost - min_terrain_cost)
            else:
                cost_map[passable_mask] = min_terrain_cost
        
        logger.info(f"  LULC代价范围: {cost_map[passable_mask].min():.3f} ~ {cost_map[passable_mask].max():.3f}")
        
        return cost_map
    
    def compute_slope_cost_map(self, factors: CostFactors, vehicle_type: str = 'type1') -> np.ndarray:
        """
        计算坡度代价图
        
        Args:
            factors: 代价因子
            vehicle_type: 车辆类型
        
        Returns:
            坡度代价图（归一化到0-1）
        """
        logger.info("计算坡度代价图...")
        
        # 坡度代价：与坡度成正比
        slope_cost = self.slope * factors.slope_factor
        
        # 映射到[0.1, 1.0]范围，确保最小坡度代价也高于道路
        # 注意：全图 max_slope 可能被极少数异常/极端值(如 90°)主导，导致大部分坡度惩罚被“压扁”
        # 这里使用鲁棒分位数（p99）并用车辆最大可爬坡角做上限，避免量纲失真
        min_terrain_cost = 0.1
        slope_vals = slope_cost[np.isfinite(slope_cost)]
        if slope_vals.size > 0:
            p99 = float(np.percentile(slope_vals, 99))
        else:
            p99 = 0.0
        
        v_max = float(VEHICLE_MAX_SLOPE.get(vehicle_type, 30.0)) * float(factors.slope_factor)
        max_slope_ref = max(0.0, min(v_max, p99))
        if max_slope_ref > 1e-6:
            normalized = np.clip(slope_cost / max_slope_ref, 0.0, 1.0)
            slope_cost = min_terrain_cost + normalized * (1.0 - min_terrain_cost)
        else:
            slope_cost = np.full_like(slope_cost, min_terrain_cost)
        
        logger.info(f"  坡度代价范围: {slope_cost.min():.3f} ~ {slope_cost.max():.3f}")
        
        return slope_cost
    
    def compute_exposure_cost_map(self, factors: CostFactors) -> np.ndarray:
        """
        计算暴露代价图
        
        暴露度 = 1 - 植被覆盖度
        植被覆盖：森林(20)=1.0, 灌木(40)=0.6, 草地(30)=0.3, 其他=0.0
        
        Args:
            factors: 代价因子
        
        Returns:
            暴露代价图（0-1）
        """
        logger.info("计算暴露代价图...")
        
        # 植被覆盖度
        vegetation = np.zeros(self.shape, dtype=np.float32)
        vegetation[self.lulc == 20] = 1.0  # 森林
        vegetation[self.lulc == 40] = 0.6  # 灌木
        vegetation[self.lulc == 30] = 0.3  # 草地
        
        # 暴露度 = 1 - 植被覆盖
        min_terrain_cost = 0.1
        raw_exposure = 1.0 - vegetation
        exposure_cost = min_terrain_cost + raw_exposure * (1.0 - min_terrain_cost)
        
        logger.info(f"  暴露代价范围: {exposure_cost.min():.3f} ~ {exposure_cost.max():.3f}")
        logger.info(f"  平均植被覆盖: {vegetation.mean():.3f}")
        
        return exposure_cost
    
    def generate_cost_map(
        self,
        intent: str = 'intent1',
        vehicle_type: str = 'type1',
        factors: Optional[CostFactors] = None
    ) -> np.ndarray:
        """
        生成综合代价图
        
        Args:
            intent: 战术意图 (intent1/intent2/intent3)
            vehicle_type: 车辆类型
            factors: 代价因子
        
        Returns:
            综合代价图
        """
        if factors is None:
            factors = CostFactors()
        
        logger.info("=" * 60)
        logger.info(f"生成环境代价图 - 意图: {intent}, 车辆: {vehicle_type}")
        logger.info("=" * 60)
        
        # 获取意图权重
        weights = INTENT_WEIGHTS.get(intent, INTENT_WEIGHTS['intent1'])
        logger.info(f"权重: dist={weights['dist']}, slope={weights['slope']}, lulc={weights['lulc']}, exp={weights['exp']}")
        
        # 计算各项代价
        lulc_cost = self.compute_lulc_cost_map()
        slope_cost = self.compute_slope_cost_map(factors, vehicle_type=vehicle_type)
        exposure_cost = self.compute_exposure_cost_map(factors)

        # 坡度非线性增强（使山地代价更高，更符合“避陡坡”意图）
        # 论文里 slope 在 intent3 权重更大，这里进一步引入幂次增强差异
        slope_exp_by_intent = cfg.get('trajectory_generation.cost_map.slope_penalty_exp_by_intent', {})
        slope_exp = float(slope_exp_by_intent.get(intent, 1.0))
        if slope_exp != 1.0:
            slope_cost = np.power(np.clip(slope_cost, 0.0, 1.0), slope_exp)
        
        # 加载可通行域掩码
        passable_mask_path = Path(get_path('paths.processed.utm_grid')) / self.region / f'passable_mask_{vehicle_type}.tif'
        with rasterio.open(passable_mask_path) as src:
            passable_mask = src.read(1).astype(bool)

        shapes = [passable_mask.shape, lulc_cost.shape, slope_cost.shape, exposure_cost.shape]
        if len({s for s in shapes}) != 1:
            h = min(s[0] for s in shapes)
            w = min(s[1] for s in shapes)
            logger.warning(f"代价/掩码shape不一致，将裁剪到共同尺寸: {(h, w)}; shapes={shapes}")
            passable_mask = passable_mask[:h, :w]
            lulc_cost = lulc_cost[:h, :w]
            slope_cost = slope_cost[:h, :w]
            exposure_cost = exposure_cost[:h, :w]
        
        logger.info(f"\n加载可通行域掩码: {passable_mask_path.name}")
        logger.info(f"  可通行像素: {np.sum(passable_mask)} ({np.sum(passable_mask)/passable_mask.size*100:.2f}%)")
        
        # 综合代价（不包含距离项，距离在A*搜索时计算）
        # 这里只计算静态环境代价
        cost_map = (
            weights['slope'] * slope_cost +
            weights['lulc'] * lulc_cost +
            weights['exp'] * exposure_cost
        )
        
        # 不做归一化，保留原始加权代价

        # 道路偏好（分级道路）：
        # 优先使用 road_graded_utm.tif（1=高等级, 2=中等级, 3=低等级）
        # 不同等级给不同代价折扣，低等级道路不再享受极低代价
        road_graded_path = Path(get_path('paths.processed.utm_grid')) / self.region / 'road_graded_utm.tif'
        road_utm_path = Path(get_path('paths.processed.utm_grid')) / self.region / 'road_utm.tif'

        road_pref_cfg = cfg.get('trajectory_generation.cost_map.road_preference', {})
        road_pref_enabled = bool(road_pref_cfg.get('enabled', False))

        if road_pref_enabled and road_graded_path.exists():
            with rasterio.open(road_graded_path) as src:
                road_graded = src.read(1)
            if road_graded.shape != cost_map.shape:
                h = min(road_graded.shape[0], cost_map.shape[0])
                w = min(road_graded.shape[1], cost_map.shape[1])
                road_graded = road_graded[:h, :w]
                cost_map = cost_map[:h, :w]
                passable_mask = passable_mask[:h, :w]

            # 分级代价折扣（可通过config覆盖）
            graded_floor = road_pref_cfg.get('graded_cost_floor', {})
            high_floor = float(graded_floor.get('high', 0.001))    # motorway/trunk/primary
            medium_floor = float(graded_floor.get('medium', 0.15)) # secondary/tertiary
            low_floor = float(graded_floor.get('low', 0.5))        # track/residential/etc

            mask_high = (road_graded == 1)
            mask_medium = (road_graded == 2)
            mask_low = (road_graded == 3)

            if np.any(mask_high):
                cost_map[mask_high] = np.minimum(cost_map[mask_high], high_floor)
            if np.any(mask_medium):
                cost_map[mask_medium] = np.minimum(cost_map[mask_medium], medium_floor)
            if np.any(mask_low):
                cost_map[mask_low] = np.minimum(cost_map[mask_low], low_floor)

            logger.info(f"  分级道路代价: high={high_floor}, medium={medium_floor}, low={low_floor}")
            logger.info(f"  道路像素: high={mask_high.sum()}, medium={mask_medium.sum()}, low={mask_low.sum()}")

        elif road_pref_enabled and road_utm_path.exists():
            # 回退到旧的二值道路（兼容）
            with rasterio.open(road_utm_path) as src:
                road_raster = src.read(1)
            if road_raster.shape != cost_map.shape:
                h = min(road_raster.shape[0], cost_map.shape[0])
                w = min(road_raster.shape[1], cost_map.shape[1])
                road_raster = road_raster[:h, :w]
                cost_map = cost_map[:h, :w]
                passable_mask = passable_mask[:h, :w]
            road_mask = (road_raster > 0)
            floor_by_vehicle = road_pref_cfg.get('road_cost_floor_by_vehicle', {})
            road_floor = float(floor_by_vehicle.get(vehicle_type, 0.0))
            if road_floor > 0.0 and np.any(road_mask):
                cost_map[road_mask] = road_floor
        
        # 不可通行区域设为无穷大
        cost_map[~passable_mask] = np.inf
        
        logger.info(f"\n综合代价统计:")
        logger.info(f"  可通行区域代价: {cost_map[passable_mask].min():.3f} ~ {cost_map[passable_mask].max():.3f}")
        logger.info(f"  平均代价: {cost_map[passable_mask].mean():.3f}")
        logger.info(f"  标准差: {cost_map[passable_mask].std():.3f}")
        
        logger.info("\n✅ 环境代价图生成完成")
        
        return cost_map
    
    def save_cost_map(
        self,
        cost_map: np.ndarray,
        intent: str,
        vehicle_type: str,
        output_dir: Optional[Path] = None
    ):
        """保存代价图为GeoTIFF"""
        if output_dir is None:
            output_dir = Path(get_path('paths.processed.utm_grid')) / self.region
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'cost_map_{intent}_{vehicle_type}.tif'
        
        # 转换为float32
        cost_map_f32 = cost_map.astype(np.float32)
        
        # 保存
        profile = {
            'driver': 'GTiff',
            'height': self.shape[0],
            'width': self.shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': self.crs,
            'transform': self.transform,
            'compress': 'lzw',
            'nodata': np.inf
        }
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(cost_map_f32, 1)
        
        logger.info(f"\n✓ 代价图已保存: {output_file}")
        logger.info(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    def generate_all_cost_maps(self):
        """为所有意图和车辆类型生成代价图"""
        logger.info("\n" + "=" * 60)
        logger.info("为所有意图和车辆类型生成代价图")
        logger.info("=" * 60 + "\n")
        
        results = {}
        
        for intent in ['intent1', 'intent2', 'intent3']:
            for vehicle_type in ['type1', 'type2', 'type3', 'type4']:
                key = f"{intent}_{vehicle_type}"
                cost_map = self.generate_cost_map(intent, vehicle_type)
                self.save_cost_map(cost_map, intent, vehicle_type)
                results[key] = cost_map
                logger.info("")
        
        # 生成统计报告
        output_dir = Path(get_path('paths.processed.utm_grid')) / self.region
        report_file = output_dir / 'cost_map_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("环境代价图生成报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"区域: {self.region}\n")
            f.write(f"栅格尺寸: {self.shape}\n")
            f.write(f"分辨率: {self.resolution}m\n\n")
            
            f.write("战术意图权重:\n")
            for intent, weights in INTENT_WEIGHTS.items():
                f.write(f"  {intent}: dist={weights['dist']}, slope={weights['slope']}, lulc={weights['lulc']}, exp={weights['exp']}\n")
            
            f.write("\nLULC基础代价:\n")
            for lulc_class, cost in LULC_BASE_COST.items():
                if cost < 900:
                    lulc_name = {
                        10: '耕地', 20: '森林', 30: '草地', 40: '灌木',
                        70: '苔原', 80: '人造', 90: '裸地'
                    }.get(lulc_class, str(lulc_class))
                    f.write(f"  {lulc_name}({lulc_class}): {cost:.1f}\n")
            
            f.write("\n各组合代价统计:\n")
            for key, cost_map in results.items():
                passable = np.isfinite(cost_map)
                if np.any(passable):
                    f.write(f"  {key}:\n")
                    f.write(f"    范围: {cost_map[passable].min():.3f} ~ {cost_map[passable].max():.3f}\n")
                    f.write(f"    均值: {cost_map[passable].mean():.3f}\n")
                    f.write(f"    标准差: {cost_map[passable].std():.3f}\n")
        
        logger.info(f"✅ 统计报告已保存: {report_file}")
        
        return results


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    generator = CostMapGenerator('scottish_highlands')
    results = generator.generate_all_cost_maps()
    
    print(f"\n所有代价图已生成（共{len(results)}个）")
