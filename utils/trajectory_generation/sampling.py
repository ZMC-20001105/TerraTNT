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
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
from skimage.measure import block_reduce

from config import cfg, get_path

logger = logging.getLogger(__name__)


_PASSABLE_CACHE = {}


def _get_passable_cache(region: str, vehicle_type: str):
    key = (region, vehicle_type)
    if key in _PASSABLE_CACHE:
        return _PASSABLE_CACHE[key]

    utm_dir = Path(get_path('paths.processed.utm_grid')) / region
    passable_mask_path = utm_dir / f'passable_mask_{vehicle_type}.tif'
    road_path = utm_dir / 'road_utm.tif'

    with rasterio.open(passable_mask_path) as src:
        passable_mask = src.read(1).astype(bool)
        transform = src.transform

    road = None
    if road_path.exists():
        try:
            with rasterio.open(road_path) as src:
                road = src.read(1)
        except Exception:
            road = None

    # 最大连通域（8连通）
    lbl, num = label(passable_mask, structure=np.ones((3, 3), dtype=np.int8))
    if num == 0:
        cache = {
            'transform': transform,
            'passable_pixels': np.empty((0, 2), dtype=np.int32),
            'coarse_lbl': None,
            'downsample_factor': 10,
        }
        _PASSABLE_CACHE[key] = cache
        return cache

    # 找最大连通域 id
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    main_id = int(np.argmax(counts))
    passable_pixels = np.argwhere(lbl == main_id).astype(np.int32)

    # 粗网格连通域（用于预筛选，减少 coarse_planning 不连通）
    downsample_factor = 10
    coarse_mask = block_reduce(passable_mask.astype(np.uint8), (downsample_factor, downsample_factor), np.max).astype(bool)
    coarse_lbl, _ = label(coarse_mask, structure=np.ones((3, 3), dtype=np.int8))

    cache = {
        'transform': transform,
        'passable_mask_main': (lbl == main_id),
        'passable_pixels': passable_pixels,
        'coarse_lbl': coarse_lbl,
        'downsample_factor': downsample_factor,
        'road': road,
    }
    _PASSABLE_CACHE[key] = cache
    return cache


def sample_start_goal(
    region: str,
    vehicle_type: str,
    min_distance: float = 80000.0,
    max_attempts: int = 100,
    edge_buffer_m: float = 5000.0,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    从可通行域随机采样起终点
    
    Args:
        region: 区域名称
        vehicle_type: 车辆类型
        min_distance: 最小直线距离（米）
        max_attempts: 最大尝试次数
        edge_buffer_m: 距栅格边界最小距离（米），防止贴边轨迹
    
    Returns:
        (start_utm, goal_utm) 或 None
    """
    cache = _get_passable_cache(region, vehicle_type)
    transform = cache['transform']
    passable_pixels = cache['passable_pixels']
    passable_mask_main = cache.get('passable_mask_main')
    coarse_lbl = cache['coarse_lbl']
    downsample_factor = int(cache['downsample_factor'])
    road = cache.get('road')
    
    if len(passable_pixels) < 2:
        logger.error("可通行像素不足")
        return None
    
    # 边界缓冲：排除距栅格边缘过近的像素
    edge_buf_pix = int(edge_buffer_m / max(abs(transform.a), 1e-6))
    h_full = cache['passable_mask_main'].shape[0] if cache.get('passable_mask_main') is not None else passable_pixels[:, 0].max() + 1
    w_full = cache['passable_mask_main'].shape[1] if cache.get('passable_mask_main') is not None else passable_pixels[:, 1].max() + 1
    interior_mask = (
        (passable_pixels[:, 0] >= edge_buf_pix) &
        (passable_pixels[:, 0] < h_full - edge_buf_pix) &
        (passable_pixels[:, 1] >= edge_buf_pix) &
        (passable_pixels[:, 1] < w_full - edge_buf_pix)
    )
    interior_pixels = passable_pixels[interior_mask]
    if len(interior_pixels) < 2:
        interior_pixels = passable_pixels
    
    logger.info(f"可通行像素数(最大连通域): {len(passable_pixels)}, 内部(去边缘): {len(interior_pixels)}")
    
    # 随机采样
    for attempt in range(max_attempts):
        # 随机选择两个点
        idx = np.random.choice(len(interior_pixels), 2, replace=False)
        start_pixel = interior_pixels[idx[0]]
        goal_pixel = interior_pixels[idx[1]]
        
        # 转换到UTM
        start_x = transform.c + start_pixel[1] * transform.a
        start_y = transform.f + start_pixel[0] * transform.e
        
        goal_x = transform.c + goal_pixel[1] * transform.a
        goal_y = transform.f + goal_pixel[0] * transform.e
        
        # 计算直线距离
        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)

        # 粗网格连通性预筛选：避免 coarse_planning 上不连通导致白跑
        if coarse_lbl is not None:
            cs = (int(start_pixel[0]) // downsample_factor, int(start_pixel[1]) // downsample_factor)
            cg = (int(goal_pixel[0]) // downsample_factor, int(goal_pixel[1]) // downsample_factor)
            if (0 <= cs[0] < coarse_lbl.shape[0] and 0 <= cs[1] < coarse_lbl.shape[1] and
                0 <= cg[0] < coarse_lbl.shape[0] and 0 <= cg[1] < coarse_lbl.shape[1]):
                sid = int(coarse_lbl[cs[0], cs[1]])
                gid = int(coarse_lbl[cg[0], cg[1]])
                if sid == 0 or gid == 0 or sid != gid:
                    continue
        
        if distance >= min_distance:
            logger.info(f"✓ 采样成功（尝试 {attempt+1} 次）")
            logger.info(f"  起点: ({start_x:.2f}, {start_y:.2f})")
            logger.info(f"  终点: ({goal_x:.2f}, {goal_y:.2f})")
            logger.info(f"  直线距离: {distance/1000:.2f} km")
            
            return (start_x, start_y), (goal_x, goal_y)
    
    logger.warning(f"采样失败（{max_attempts} 次尝试）")
    return None


def sample_start_goal_v2(
    region: str,
    vehicle_type: str,
    min_distance: float = 60000.0,
    max_attempts: int = 100,
    prefer_road: bool = False,
    road_buffer_m: float = 0.0,
    force_on_road: bool = False,
    edge_buffer_m: float = 5000.0,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    从可通行域随机采样起终点（V2版本，避免批量重复）
    
    注意：调用前应在外部设置唯一的随机种子
    
    Args:
        region: 区域名称
        vehicle_type: 车辆类型
        min_distance: 最小直线距离（米）
        max_attempts: 最大尝试次数
        edge_buffer_m: 距栅格边界最小距离（米），防止贴边轨迹
    
    Returns:
        (start_utm, goal_utm) 或 None
    """
    cache = _get_passable_cache(region, vehicle_type)
    transform = cache['transform']
    passable_pixels = cache['passable_pixels']
    passable_mask_main = cache.get('passable_mask_main')
    coarse_lbl = cache['coarse_lbl']
    downsample_factor = int(cache['downsample_factor'])
    road = cache.get('road')
    
    if len(passable_pixels) < 2:
        logger.error("可通行像素不足")
        return None
    
    # 边界缓冲：排除距栅格边缘过近的像素
    edge_buf_pix = int(edge_buffer_m / max(abs(transform.a), 1e-6))
    h_full = passable_mask_main.shape[0] if passable_mask_main is not None else passable_pixels[:, 0].max() + 1
    w_full = passable_mask_main.shape[1] if passable_mask_main is not None else passable_pixels[:, 1].max() + 1
    interior_mask = (
        (passable_pixels[:, 0] >= edge_buf_pix) &
        (passable_pixels[:, 0] < h_full - edge_buf_pix) &
        (passable_pixels[:, 1] >= edge_buf_pix) &
        (passable_pixels[:, 1] < w_full - edge_buf_pix)
    )
    interior_pixels = passable_pixels[interior_mask]
    if len(interior_pixels) < 2:
        interior_pixels = passable_pixels
    
    logger.info(f"可通行像素数(最大连通域): {len(passable_pixels)}, 内部(去边缘): {len(interior_pixels)}")
    
    # 候选像素：默认内部可通行域；可选道路偏好（道路 buffer 内）
    candidate_pixels = interior_pixels
    if bool(force_on_road) and road is not None and passable_mask_main is not None:
        if road.shape == passable_mask_main.shape:
            mask = passable_mask_main & (road > 0)
            road_pixels = np.argwhere(mask).astype(np.int32)
            if road_pixels.shape[0] >= 2:
                candidate_pixels = road_pixels

    if candidate_pixels is interior_pixels and bool(prefer_road) and road is not None and passable_mask_main is not None:
        if road.shape == passable_mask_main.shape:
            buf_pix = int(max(0.0, float(road_buffer_m)) / max(float(transform.a), 1e-6))
            road_mask = (road > 0)
            if buf_pix > 0:
                road_mask = binary_dilation(road_mask, iterations=buf_pix)
            mask = passable_mask_main & road_mask
            road_pixels = np.argwhere(mask).astype(np.int32)
            if road_pixels.shape[0] >= 2:
                candidate_pixels = road_pixels
        # 若道路像素不足，则回退到全可通行域

    # 随机采样（使用外部设置的随机种子）
    for attempt in range(max_attempts):
        # 随机选择两个点
        idx = np.random.choice(len(candidate_pixels), 2, replace=False)
        start_pixel = candidate_pixels[idx[0]]
        goal_pixel = candidate_pixels[idx[1]]
        
        # 转换到UTM
        start_x = transform.c + start_pixel[1] * transform.a
        start_y = transform.f + start_pixel[0] * transform.e
        
        goal_x = transform.c + goal_pixel[1] * transform.a
        goal_y = transform.f + goal_pixel[0] * transform.e
        
        # 计算直线距离
        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)

        # 粗网格连通性预筛选：避免 coarse_planning 上不连通导致白跑
        if coarse_lbl is not None:
            cs = (int(start_pixel[0]) // downsample_factor, int(start_pixel[1]) // downsample_factor)
            cg = (int(goal_pixel[0]) // downsample_factor, int(goal_pixel[1]) // downsample_factor)
            if (0 <= cs[0] < coarse_lbl.shape[0] and 0 <= cs[1] < coarse_lbl.shape[1] and
                0 <= cg[0] < coarse_lbl.shape[0] and 0 <= cg[1] < coarse_lbl.shape[1]):
                sid = int(coarse_lbl[cs[0], cs[1]])
                gid = int(coarse_lbl[cg[0], cg[1]])
                if sid == 0 or gid == 0 or sid != gid:
                    continue
        
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
