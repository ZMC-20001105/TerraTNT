"""
可通行域提取 (Passable Mask)

按论文第3章要求：
M_passable = M_LULC ∧ M_slope

1. LULC黑名单：湿地(50)、水体(60)、冰雪(100)
2. 坡度阈值：根据车辆类型的最大爬坡角
3. 最大连通域：确保起终点可达
"""
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import rasterio
from scipy.ndimage import label, binary_dilation

from config import cfg, get_path

logger = logging.getLogger(__name__)

# 论文表3.1：LULC不可通行类别
LULC_IMPASSABLE = [50, 60, 100]  # 湿地、水体、冰雪

# 论文表3.2：车辆类型最大爬坡角（度）
VEHICLE_MAX_SLOPE = {
    'type1': 30.0,
    'type2': 25.0,
    'type3': 20.0,
    'type4': 15.0
}


class PassableMaskGenerator:
    """可通行域生成器"""
    
    def __init__(self, region: str = 'scottish_highlands'):
        self.region = region
        
        # 加载UTM栅格
        utm_dir = Path(get_path('paths.processed.utm_grid')) / region
        
        logger.info(f"加载 {region} UTM栅格数据...")
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
        
        logger.info(f"  栅格尺寸: {self.shape}")
        logger.info(f"  分辨率: {self.transform.a}m × {-self.transform.e}m")
        logger.info(f"  LULC类别: {np.unique(self.lulc).tolist()}")
    
    def __del__(self):
        """关闭栅格文件"""
        if hasattr(self, 'dem_src'):
            self.dem_src.close()
        if hasattr(self, 'slope_src'):
            self.slope_src.close()
        if hasattr(self, 'lulc_src'):
            self.lulc_src.close()
    
    def create_lulc_mask(self) -> np.ndarray:
        """
        创建LULC可通行掩码
        
        Returns:
            M_LULC: bool数组，True表示可通行
        """
        logger.info("创建LULC可通行掩码...")
        
        # 初始化为全部可通行
        M_LULC = np.ones(self.shape, dtype=bool)
        
        # 标记不可通行区域
        for lulc_class in LULC_IMPASSABLE:
            impassable_count = np.sum(self.lulc == lulc_class)
            if impassable_count > 0:
                M_LULC[self.lulc == lulc_class] = False
                lulc_name = {50: '湿地', 60: '水体', 100: '冰雪'}.get(lulc_class, str(lulc_class))
                logger.info(f"  {lulc_name}({lulc_class}): {impassable_count} 像素 ({impassable_count/M_LULC.size*100:.2f}%)")
        
        passable_count = np.sum(M_LULC)
        logger.info(f"  LULC可通行: {passable_count} 像素 ({passable_count/M_LULC.size*100:.2f}%)")
        
        return M_LULC
    
    def create_slope_mask(self, max_slope_deg: float) -> np.ndarray:
        """
        创建坡度可通行掩码
        
        Args:
            max_slope_deg: 最大爬坡角（度）
        
        Returns:
            M_slope: bool数组，True表示可通行
        """
        logger.info(f"创建坡度可通行掩码（阈值: {max_slope_deg}°）...")
        
        M_slope = self.slope <= max_slope_deg
        
        passable_count = np.sum(M_slope)
        logger.info(f"  坡度可通行: {passable_count} 像素 ({passable_count/M_slope.size*100:.2f}%)")
        
        return M_slope
    
    def find_largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        """
        提取最大连通域
        
        Args:
            mask: bool数组
        
        Returns:
            最大连通域掩码
        """
        logger.info("提取最大连通域...")
        
        # 连通域标记（8连通）
        labeled, num_features = label(mask, structure=np.ones((3, 3)))
        
        if num_features == 0:
            logger.warning("  未找到连通域！")
            return np.zeros_like(mask, dtype=bool)
        
        logger.info(f"  找到 {num_features} 个连通域")
        
        # 找最大连通域
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # 忽略背景
        
        largest_component = component_sizes.argmax()
        largest_size = component_sizes[largest_component]
        
        logger.info(f"  最大连通域: {largest_size} 像素 ({largest_size/mask.size*100:.2f}%)")
        
        # 统计其他连通域
        sorted_sizes = np.sort(component_sizes[1:])[::-1]
        if len(sorted_sizes) > 1:
            logger.info(f"  第2大连通域: {sorted_sizes[1]} 像素 ({sorted_sizes[1]/mask.size*100:.2f}%)")
        if len(sorted_sizes) > 2:
            logger.info(f"  第3大连通域: {sorted_sizes[2]} 像素 ({sorted_sizes[2]/mask.size*100:.2f}%)")
        
        return labeled == largest_component
    
    def generate_passable_mask(
        self,
        vehicle_type: str = 'type1',
        use_largest_component: bool = True,
        buffer_pixels: int = 0
    ) -> np.ndarray:
        """
        生成可通行域掩码
        
        Args:
            vehicle_type: 车辆类型 (type1/type2/type3/type4)
            use_largest_component: 是否只保留最大连通域
            buffer_pixels: 膨胀缓冲区（像素）
        
        Returns:
            M_passable: bool数组，True表示可通行
        """
        logger.info("=" * 60)
        logger.info(f"生成可通行域掩码 - 车辆类型: {vehicle_type}")
        logger.info("=" * 60)
        
        # 获取车辆最大爬坡角
        max_slope = VEHICLE_MAX_SLOPE.get(vehicle_type, 30.0)
        logger.info(f"最大爬坡角: {max_slope}°")
        
        # 1. LULC掩码
        M_LULC = self.create_lulc_mask()
        
        # 2. 坡度掩码
        M_slope = self.create_slope_mask(max_slope)
        
        # 3. 合并：M_passable = M_LULC ∧ M_slope
        logger.info("\n合并LULC和坡度掩码...")
        M_passable = M_LULC & M_slope
        
        passable_count = np.sum(M_passable)
        logger.info(f"  合并后可通行: {passable_count} 像素 ({passable_count/M_passable.size*100:.2f}%)")
        
        # 4. 最大连通域
        if use_largest_component:
            M_passable = self.find_largest_connected_component(M_passable)
        
        # 5. 膨胀缓冲（可选）
        if buffer_pixels > 0:
            logger.info(f"\n应用膨胀缓冲（{buffer_pixels}像素）...")
            structure = np.ones((2*buffer_pixels+1, 2*buffer_pixels+1))
            M_passable = binary_dilation(M_passable, structure=structure)
            
            passable_count = np.sum(M_passable)
            logger.info(f"  膨胀后可通行: {passable_count} 像素 ({passable_count/M_passable.size*100:.2f}%)")
        
        logger.info("\n✅ 可通行域生成完成")
        
        return M_passable
    
    def save_passable_mask(
        self,
        mask: np.ndarray,
        vehicle_type: str,
        output_dir: Optional[Path] = None
    ):
        """保存可通行域掩码为GeoTIFF"""
        if output_dir is None:
            output_dir = Path(get_path('paths.processed.utm_grid')) / self.region
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'passable_mask_{vehicle_type}.tif'
        
        # 转换为uint8（0/1）
        mask_uint8 = mask.astype(np.uint8)
        
        # 保存
        profile = {
            'driver': 'GTiff',
            'height': self.shape[0],
            'width': self.shape[1],
            'count': 1,
            'dtype': 'uint8',
            'crs': self.crs,
            'transform': self.transform,
            'compress': 'lzw'
        }
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(mask_uint8, 1)
        
        logger.info(f"\n✓ 可通行域掩码已保存: {output_file}")
        logger.info(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    def generate_all_vehicle_masks(self):
        """为所有车辆类型生成可通行域掩码"""
        logger.info("\n" + "=" * 60)
        logger.info("为所有车辆类型生成可通行域掩码")
        logger.info("=" * 60 + "\n")
        
        results = {}
        
        for vehicle_type in ['type1', 'type2', 'type3', 'type4']:
            mask = self.generate_passable_mask(vehicle_type, use_largest_component=True)
            self.save_passable_mask(mask, vehicle_type)
            results[vehicle_type] = mask
            logger.info("")
        
        # 生成统计报告
        output_dir = Path(get_path('paths.processed.utm_grid')) / self.region
        report_file = output_dir / 'passable_mask_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("可通行域生成报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"区域: {self.region}\n")
            f.write(f"栅格尺寸: {self.shape}\n")
            f.write(f"分辨率: {self.transform.a}m × {-self.transform.e}m\n")
            f.write(f"总像素数: {self.shape[0] * self.shape[1]}\n\n")
            
            f.write("LULC不可通行类别:\n")
            for lulc_class in LULC_IMPASSABLE:
                count = np.sum(self.lulc == lulc_class)
                lulc_name = {50: '湿地', 60: '水体', 100: '冰雪'}.get(lulc_class, str(lulc_class))
                f.write(f"  {lulc_name}({lulc_class}): {count} 像素 ({count/self.shape[0]/self.shape[1]*100:.2f}%)\n")
            
            f.write("\n各车辆类型可通行域:\n")
            for vehicle_type, mask in results.items():
                max_slope = VEHICLE_MAX_SLOPE[vehicle_type]
                count = np.sum(mask)
                f.write(f"  {vehicle_type} (最大爬坡角{max_slope}°): {count} 像素 ({count/mask.size*100:.2f}%)\n")
        
        logger.info(f"✅ 统计报告已保存: {report_file}")
        
        return results


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    generator = PassableMaskGenerator('scottish_highlands')
    results = generator.generate_all_vehicle_masks()
    
    print("\n所有车辆类型可通行域已生成")
