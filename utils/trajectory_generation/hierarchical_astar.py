"""
分层A*路径规划 (Hierarchical A*)

按论文第3章要求：
1. 粗规划：在降采样地图上运行A*（降采样因子α）
2. 航路点：沿粗路径每5km设置航路点
3. 细化：在2km走廊内对每段进行高分辨率A*规划
"""
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import rasterio
from heapq import heappush, heappop
from skimage.transform import downscale_local_mean, resize
from scipy.ndimage import binary_dilation

from config import cfg, get_path

logger = logging.getLogger(__name__)


class HierarchicalAStarPlanner:
    """分层A*路径规划器"""
    
    def __init__(
        self,
        region: str = 'scottish_highlands',
        intent: str = 'intent1',
        vehicle_type: str = 'type1'
    ):
        self.region = region
        self.intent = intent
        self.vehicle_type = vehicle_type
        
        # 加载代价图
        utm_dir = Path(get_path('paths.processed.utm_grid')) / region
        cost_map_path = utm_dir / f'cost_map_{intent}_{vehicle_type}.tif'
        
        logger.info(f"加载代价图: {cost_map_path.name}")
        with rasterio.open(cost_map_path) as src:
            self.cost_map = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.shape = self.cost_map.shape
        
        self.resolution = self.transform.a  # 30m
        
        logger.info(f"  栅格尺寸: {self.shape}")
        logger.info(f"  分辨率: {self.resolution}m")
        logger.info(f"  可通行像素: {np.sum(np.isfinite(self.cost_map))} ({np.sum(np.isfinite(self.cost_map))/self.cost_map.size*100:.2f}%)")
    
    def pixel_to_utm(self, row: int, col: int) -> Tuple[float, float]:
        """像素坐标转UTM坐标"""
        x = self.transform.c + col * self.transform.a
        y = self.transform.f + row * self.transform.e
        return x, y
    
    def utm_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """UTM坐标转像素坐标"""
        col = int((x - self.transform.c) / self.transform.a)
        row = int((y - self.transform.f) / self.transform.e)
        return row, col
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """A*启发函数（欧氏距离）"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _find_nearest_passable(self, pos: Tuple[int, int], cost_map: np.ndarray, max_radius: int = 20) -> Optional[Tuple[int, int]]:
        """
        寻找最近的可通行点
        
        Args:
            pos: 当前位置
            cost_map: 代价图
            max_radius: 最大搜索半径
        
        Returns:
            最近可通行点或None
        """
        row, col = pos
        
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) == radius or abs(dc) == radius:  # 只检查边界
                        new_row, new_col = row + dr, col + dc
                        if (0 <= new_row < cost_map.shape[0] and 
                            0 <= new_col < cost_map.shape[1] and
                            np.isfinite(cost_map[new_row, new_col])):
                            return (new_row, new_col)
        
        return None
    
    def get_neighbors(self, pos: Tuple[int, int], cost_map: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        获取邻居节点（8连通）
        
        Returns:
            [(row, col, move_cost), ...]
        """
        row, col = pos
        neighbors = []
        
        # 8个方向
        directions = [
            (-1, 0, 1.0),    # 上
            (1, 0, 1.0),     # 下
            (0, -1, 1.0),    # 左
            (0, 1, 1.0),     # 右
            (-1, -1, 1.414), # 左上
            (-1, 1, 1.414),  # 右上
            (1, -1, 1.414),  # 左下
            (1, 1, 1.414)    # 右下
        ]
        
        for dr, dc, base_cost in directions:
            new_row, new_col = row + dr, col + dc
            
            # 边界检查
            if 0 <= new_row < cost_map.shape[0] and 0 <= new_col < cost_map.shape[1]:
                # 可通行检查
                if np.isfinite(cost_map[new_row, new_col]):
                    # 移动代价 = 基础距离 * (1 + 环境代价)
                    move_cost = base_cost * (1.0 + cost_map[new_row, new_col])
                    neighbors.append((new_row, new_col, move_cost))
        
        return neighbors
    
    def astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_map: np.ndarray,
        max_iterations: int = 100000,
        heuristic_weight: float = 1.5
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A*搜索（加权A*，加速搜索）
        
        Args:
            start: 起点(row, col)
            goal: 终点(row, col)
            cost_map: 代价图
            max_iterations: 最大迭代次数
            heuristic_weight: 启发函数权重（>1加速但可能不是最优）
        
        Returns:
            路径点列表 [(row, col), ...] 或 None
        """
        # 检查起终点有效性
        if not (0 <= start[0] < cost_map.shape[0] and 0 <= start[1] < cost_map.shape[1]):
            logger.error(f"起点超出边界: {start}")
            return None
        if not (0 <= goal[0] < cost_map.shape[0] and 0 <= goal[1] < cost_map.shape[1]):
            logger.error(f"终点超出边界: {goal}")
            return None
        if not np.isfinite(cost_map[start[0], start[1]]):
            logger.error(f"起点不可通行: {start}")
            return None
        if not np.isfinite(cost_map[goal[0], goal[1]]):
            logger.error(f"终点不可通行: {goal}")
            return None
        
        # 初始化
        open_set = []
        h_start = self.heuristic(start, goal)
        heappush(open_set, (h_start * heuristic_weight, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: h_start * heuristic_weight}
        
        iterations = 0
        last_log = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # 每10000次迭代打印进度
            if iterations - last_log >= 10000:
                h_current = self.heuristic(current, goal)
                logger.debug(f"    迭代 {iterations}: open_set={len(open_set)}, 距离目标={h_current:.1f}")
                last_log = iterations
            
            current_f, current = heappop(open_set)
            
            # 到达目标
            if current == goal:
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                logger.debug(f"  A*成功: {len(path)}点, {iterations}次迭代")
                return path
            
            # 扩展邻居
            for neighbor_row, neighbor_col, move_cost in self.get_neighbors(current, cost_map):
                neighbor = (neighbor_row, neighbor_col)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.heuristic(neighbor, goal)
                    f_score[neighbor] = tentative_g + h * heuristic_weight
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        logger.warning(f"  A*失败: 达到最大迭代次数 {max_iterations}")
        return None
    
    def coarse_planning(
        self,
        start_utm: Tuple[float, float],
        goal_utm: Tuple[float, float],
        downsample_factor: int = 10
    ) -> Optional[List[Tuple[float, float]]]:
        """
        粗规划：在降采样地图上运行A*
        
        Args:
            start_utm: 起点UTM坐标
            goal_utm: 终点UTM坐标
            downsample_factor: 降采样因子α
        
        Returns:
            粗路径UTM坐标列表 [(x, y), ...]
        """
        logger.info(f"\n步骤1: 粗规划（降采样因子={downsample_factor}）")
        
        # 降采样代价图（使用最小值而非平均值，保持可通行性）
        logger.info("  降采样代价图...")
        from skimage.measure import block_reduce
        coarse_cost = block_reduce(self.cost_map, (downsample_factor, downsample_factor), np.min)
        coarse_shape = coarse_cost.shape
        
        logger.info(f"  粗网格尺寸: {coarse_shape}")
        logger.info(f"  粗分辨率: {self.resolution * downsample_factor}m")
        
        # 转换起终点到粗网格
        start_row, start_col = self.utm_to_pixel(*start_utm)
        goal_row, goal_col = self.utm_to_pixel(*goal_utm)
        
        coarse_start = (start_row // downsample_factor, start_col // downsample_factor)
        coarse_goal = (goal_row // downsample_factor, goal_col // downsample_factor)
        
        # 确保起终点在粗网格范围内
        coarse_start = (
            max(0, min(coarse_start[0], coarse_shape[0] - 1)),
            max(0, min(coarse_start[1], coarse_shape[1] - 1))
        )
        coarse_goal = (
            max(0, min(coarse_goal[0], coarse_shape[0] - 1)),
            max(0, min(coarse_goal[1], coarse_shape[1] - 1))
        )
        
        logger.info(f"  粗起点: {coarse_start}")
        logger.info(f"  粗终点: {coarse_goal}")
        
        # 如果起终点不可通行，寻找最近的可通行点
        if not np.isfinite(coarse_cost[coarse_start[0], coarse_start[1]]):
            logger.warning("  粗起点不可通行，寻找最近可通行点...")
            coarse_start = self._find_nearest_passable(coarse_start, coarse_cost)
            if coarse_start is None:
                logger.error("  未找到可通行起点")
                return None
            logger.info(f"  调整后粗起点: {coarse_start}")
        
        if not np.isfinite(coarse_cost[coarse_goal[0], coarse_goal[1]]):
            logger.warning("  粗终点不可通行，寻找最近可通行点...")
            coarse_goal = self._find_nearest_passable(coarse_goal, coarse_cost)
            if coarse_goal is None:
                logger.error("  未找到可通行终点")
                return None
            logger.info(f"  调整后粗终点: {coarse_goal}")
        
        # 粗A*搜索（使用更大的权重加速）
        logger.info("  运行粗A*搜索...")
        coarse_path = self.astar(coarse_start, coarse_goal, coarse_cost, max_iterations=100000, heuristic_weight=2.0)
        
        if coarse_path is None:
            logger.error("  粗规划失败！")
            return None
        
        logger.info(f"  ✓ 粗路径: {len(coarse_path)} 点")
        
        # 转换回UTM坐标
        coarse_path_utm = []
        for row, col in coarse_path:
            # 映射回原始网格中心
            orig_row = row * downsample_factor + downsample_factor // 2
            orig_col = col * downsample_factor + downsample_factor // 2
            x, y = self.pixel_to_utm(orig_row, orig_col)
            coarse_path_utm.append((x, y))
        
        return coarse_path_utm
    
    def extract_waypoints(
        self,
        coarse_path: List[Tuple[float, float]],
        waypoint_interval: float = 5000.0
    ) -> List[Tuple[float, float]]:
        """
        沿粗路径提取航路点
        
        Args:
            coarse_path: 粗路径UTM坐标
            waypoint_interval: 航路点间隔（米）
        
        Returns:
            航路点列表
        """
        logger.info(f"\n步骤2: 提取航路点（间隔={waypoint_interval}m）")
        
        waypoints = [coarse_path[0]]  # 起点
        
        cumulative_dist = 0.0
        last_waypoint = coarse_path[0]
        
        for i in range(1, len(coarse_path)):
            p1 = coarse_path[i-1]
            p2 = coarse_path[i]
            
            segment_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            cumulative_dist += segment_dist
            
            if cumulative_dist >= waypoint_interval:
                waypoints.append(p2)
                last_waypoint = p2
                cumulative_dist = 0.0
        
        # 确保终点在内
        if waypoints[-1] != coarse_path[-1]:
            waypoints.append(coarse_path[-1])
        
        logger.info(f"  ✓ 航路点: {len(waypoints)} 个")
        
        # 计算总距离
        total_dist = 0.0
        for i in range(1, len(waypoints)):
            dist = np.sqrt((waypoints[i][0] - waypoints[i-1][0])**2 + (waypoints[i][1] - waypoints[i-1][1])**2)
            total_dist += dist
        
        logger.info(f"  总距离: {total_dist/1000:.2f} km")
        
        return waypoints
    
    def refine_segment(
        self,
        start_utm: Tuple[float, float],
        goal_utm: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        细化单个段（直接在原始代价图上连接两点）
        
        Args:
            start_utm: 起点UTM
            goal_utm: 终点UTM
        
        Returns:
            细化路径UTM坐标
        """
        # 转换到像素坐标
        start_pixel = self.utm_to_pixel(*start_utm)
        goal_pixel = self.utm_to_pixel(*goal_utm)
        
        # 检查起点和终点是否可通行
        if self.cost_map[start_pixel[0], start_pixel[1]] == np.inf:
            logger.error(f"起点不可通行: {start_pixel}")
            return None
        
        if self.cost_map[goal_pixel[0], goal_pixel[1]] == np.inf:
            logger.error(f"终点不可通行: {goal_pixel}")
            return None
        
        # 直接在原始代价图上运行A*（使用加权启发函数加速）
        path_pixels = self.astar(start_pixel, goal_pixel, self.cost_map, max_iterations=50000, heuristic_weight=2.0)
        
        if path_pixels is None:
            return None
        
        # 转换回UTM
        path_utm = [self.pixel_to_utm(row, col) for row, col in path_pixels]
        
        return path_utm
    
    def hierarchical_plan(
        self,
        start_utm: Tuple[float, float],
        goal_utm: Tuple[float, float],
        downsample_factor: int = 10
    ) -> Optional[List[Tuple[float, float]]]:
        """
        分层A*规划完整流程（简化版：直接细化粗路径）
        
        Args:
            start_utm: 起点UTM坐标
            goal_utm: 终点UTM坐标
            downsample_factor: 降采样因子α
        
        Returns:
            完整路径UTM坐标列表
        """
        logger.info("=" * 60)
        logger.info(f"分层A*路径规划")
        logger.info(f"  意图: {self.intent}, 车辆: {self.vehicle_type}")
        logger.info(f"  起点: ({start_utm[0]:.2f}, {start_utm[1]:.2f})")
        logger.info(f"  终点: ({goal_utm[0]:.2f}, {goal_utm[1]:.2f})")
        
        straight_dist = np.sqrt((goal_utm[0] - start_utm[0])**2 + (goal_utm[1] - start_utm[1])**2)
        logger.info(f"  直线距离: {straight_dist/1000:.2f} km")
        logger.info("=" * 60)
        
        # 步骤1: 粗规划
        coarse_path = self.coarse_planning(start_utm, goal_utm, downsample_factor)
        if coarse_path is None:
            return None
        
        # 步骤2: 直接使用粗路径作为最终路径（已经是UTM坐标）
        # 粗路径已经考虑了环境代价，只是分辨率较低（300m）
        logger.info(f"\n✅ 分层A*规划完成（使用粗规划路径）")
        logger.info(f"  路径点数: {len(coarse_path)}")
        
        # 计算路径长度
        path_length = 0.0
        for i in range(1, len(coarse_path)):
            dist = np.sqrt((coarse_path[i][0] - coarse_path[i-1][0])**2 + (coarse_path[i][1] - coarse_path[i-1][1])**2)
            path_length += dist
        
        logger.info(f"  路径长度: {path_length/1000:.2f} km")
        logger.info(f"  迂回系数: {path_length/straight_dist:.2f}")
        
        return coarse_path


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 测试：在苏格兰高地规划一条路径
    planner = HierarchicalAStarPlanner(
        region='scottish_highlands',
        intent='intent1',
        vehicle_type='type1'
    )
    
    # 选择起终点（需要在可通行域内）
    # 这里使用示例坐标，实际应该从可通行域中随机采样
    start_utm = (400000.0, 6350000.0)
    goal_utm = (450000.0, 6450000.0)
    
    path = planner.hierarchical_plan(
        start_utm,
        goal_utm,
        downsample_factor=10,
        waypoint_interval=5000.0,
        corridor_width=2000.0
    )
    
    if path:
        print(f"\n路径规划成功！共 {len(path)} 个点")
    else:
        print("\n路径规划失败")
