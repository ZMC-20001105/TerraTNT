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

from config import cfg, get_path

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
        
        # 归一化到0-1（排除不可通行区域）
        passable_mask = cost_map < 900
        if np.any(passable_mask):
            min_cost = cost_map[passable_mask].min()
            max_cost = cost_map[passable_mask].max()
            cost_map[passable_mask] = (cost_map[passable_mask] - min_cost) / (max_cost - min_cost)
        
        logger.info(f"  LULC代价范围: {cost_map[passable_mask].min():.3f} ~ {cost_map[passable_mask].max():.3f}")
        
        return cost_map
    
    def compute_slope_cost_map(self, factors: CostFactors) -> np.ndarray:
        """
        计算坡度代价图
        
        Args:
            factors: 代价因子
        
        Returns:
            坡度代价图（归一化到0-1）
        """
        logger.info("计算坡度代价图...")
        
        # 坡度代价：与坡度成正比
        slope_cost = self.slope * factors.slope_factor
        
        # 归一化到0-1
        max_slope = np.nanmax(slope_cost)
        if max_slope > 0:
            slope_cost = slope_cost / max_slope
        
        logger.info(f"  坡度代价范围: {np.nanmin(slope_cost):.3f} ~ {np.nanmax(slope_cost):.3f}")
        
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
        exposure = (1.0 - vegetation) * factors.exposure_factor
        
        logger.info(f"  暴露代价范围: {exposure.min():.3f} ~ {exposure.max():.3f}")
        logger.info(f"  平均植被覆盖: {vegetation.mean():.3f}")
        
        return exposure
    
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
        slope_cost = self.compute_slope_cost_map(factors)
        exposure_cost = self.compute_exposure_cost_map(factors)
        
        # 加载可通行域掩码
        passable_mask_path = Path(get_path('paths.processed.utm_grid')) / self.region / f'passable_mask_{vehicle_type}.tif'
        with rasterio.open(passable_mask_path) as src:
            passable_mask = src.read(1).astype(bool)
        
        logger.info(f"\n加载可通行域掩码: {passable_mask_path.name}")
        logger.info(f"  可通行像素: {np.sum(passable_mask)} ({np.sum(passable_mask)/passable_mask.size*100:.2f}%)")
        
        # 综合代价（不包含距离项，距离在A*搜索时计算）
        # 这里只计算静态环境代价
        cost_map = (
            weights['slope'] * slope_cost +
            weights['lulc'] * lulc_cost +
            weights['exp'] * exposure_cost
        )
        
        # 归一化到0-1
        if np.any(passable_mask):
            min_cost = cost_map[passable_mask].min()
            max_cost = cost_map[passable_mask].max()
            if max_cost > min_cost:
                cost_map[passable_mask] = (cost_map[passable_mask] - min_cost) / (max_cost - min_cost)
        
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
