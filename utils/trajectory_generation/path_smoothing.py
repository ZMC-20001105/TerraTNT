"""
路径平滑 (Path Smoothing)

按论文第3章要求：
1. 使用弦长参数化的三次样条插值
2. 重采样：最大距离阈值 d_max = 100m
"""
import logging
from typing import List, Tuple
import numpy as np
from scipy.interpolate import splprep, splev

logger = logging.getLogger(__name__)


def compute_path_length(path: List[Tuple[float, float]]) -> float:
    """计算路径总长度"""
    length = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        length += np.sqrt(dx**2 + dy**2)
    return length


def smooth_path(
    path: List[Tuple[float, float]],
    smoothing_factor: float = 0.0,
    resample_max_dist: float = 100.0
) -> List[Tuple[float, float]]:
    """
    使用三次样条平滑路径
    
    Args:
        path: 原始路径 [(x, y), ...]
        smoothing_factor: 平滑因子（0=无平滑，越大越平滑）
        resample_max_dist: 重采样最大距离（米）
    
    Returns:
        平滑后的路径
    """
    if len(path) < 4:
        logger.warning("路径点太少，无法平滑")
        return path
    
    logger.info(f"路径平滑...")
    logger.info(f"  原始点数: {len(path)}")
    logger.info(f"  原始长度: {compute_path_length(path)/1000:.2f} km")
    
    # 提取x, y坐标
    x = np.array([p[0] for p in path])
    y = np.array([p[1] for p in path])
    
    # 计算弦长参数化
    # u[i] = 累积弦长
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    u = np.zeros(len(path))
    u[1:] = np.cumsum(distances)
    
    # 归一化到[0, 1]
    if u[-1] > 0:
        u = u / u[-1]
    
    # 三次样条插值（弦长参数化）
    try:
        tck, u_fit = splprep([x, y], u=u, s=smoothing_factor, k=3)
    except Exception as e:
        logger.error(f"样条插值失败: {e}")
        return path
    
    # 重采样
    # 计算需要的采样点数
    path_length = compute_path_length(path)
    num_samples = int(np.ceil(path_length / resample_max_dist)) + 1
    
    u_new = np.linspace(0, 1, num_samples)
    x_new, y_new = splev(u_new, tck)
    
    # 构建新路径
    smoothed_path = [(x_new[i], y_new[i]) for i in range(len(x_new))]
    
    logger.info(f"  平滑后点数: {len(smoothed_path)}")
    logger.info(f"  平滑后长度: {compute_path_length(smoothed_path)/1000:.2f} km")
    logger.info(f"  平均采样间隔: {compute_path_length(smoothed_path)/(len(smoothed_path)-1):.2f} m")
    
    return smoothed_path


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 测试：生成一条简单路径并平滑
    test_path = [
        (0, 0),
        (1000, 500),
        (2000, 1500),
        (3000, 2000),
        (4000, 2200),
        (5000, 3000)
    ]
    
    smoothed = smooth_path(test_path, smoothing_factor=0.0, resample_max_dist=100.0)
    
    print(f"\n原始路径: {len(test_path)} 点")
    print(f"平滑路径: {len(smoothed)} 点")
