"""
起终点采样工具

从可通行域中随机采样起终点，满足：
1. 在最大连通域内
2. 直线距离 >= min_distance（论文要求80km）
"""
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import rasterio

from config import cfg, get_path

logger = logging.getLogger(__name__)


def sample_start_goal(
    region: str,
    vehicle_type: str,
    min_distance: float = 80000.0,
    max_attempts: int = 100
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    从可通行域随机采样起终点
    
    Args:
        region: 区域名称
        vehicle_type: 车辆类型
        min_distance: 最小直线距离（米）
        max_attempts: 最大尝试次数
    
    Returns:
        (start_utm, goal_utm) 或 None
    """
    # 加载可通行域掩码
    utm_dir = Path(get_path('paths.processed.utm_grid')) / region
    passable_mask_path = utm_dir / f'passable_mask_{vehicle_type}.tif'
    
    with rasterio.open(passable_mask_path) as src:
        passable_mask = src.read(1).astype(bool)
        transform = src.transform
    
    # 获取可通行像素坐标
    passable_pixels = np.argwhere(passable_mask)
    
    if len(passable_pixels) < 2:
        logger.error("可通行像素不足")
        return None
    
    logger.info(f"可通行像素数: {len(passable_pixels)}")
    
    # 随机采样
    for attempt in range(max_attempts):
        # 随机选择两个点
        idx = np.random.choice(len(passable_pixels), 2, replace=False)
        start_pixel = passable_pixels[idx[0]]
        goal_pixel = passable_pixels[idx[1]]
        
        # 转换到UTM
        start_x = transform.c + start_pixel[1] * transform.a
        start_y = transform.f + start_pixel[0] * transform.e
        
        goal_x = transform.c + goal_pixel[1] * transform.a
        goal_y = transform.f + goal_pixel[0] * transform.e
        
        # 计算直线距离
        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        
        if distance >= min_distance:
            logger.info(f"✓ 采样成功（尝试 {attempt+1} 次）")
            logger.info(f"  起点: ({start_x:.2f}, {start_y:.2f})")
            logger.info(f"  终点: ({goal_x:.2f}, {goal_y:.2f})")
            logger.info(f"  直线距离: {distance/1000:.2f} km")
            
            return (start_x, start_y), (goal_x, goal_y)
    
    logger.warning(f"采样失败（{max_attempts} 次尝试）")
    return None


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    result = sample_start_goal('scottish_highlands', 'type1', min_distance=80000.0)
    
    if result:
        start, goal = result
        print(f"\n起点: {start}")
        print(f"终点: {goal}")
    else:
        print("\n采样失败")
