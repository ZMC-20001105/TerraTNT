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
from scipy.ndimage import label

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
            
            current_f, current = heappop(open_set)

            # 每10000次迭代打印进度
            if iterations - last_log >= 10000:
                h_current = self.heuristic(current, goal)
                logger.debug(f"    迭代 {iterations}: open_set={len(open_set)}, 距离目标={h_current:.1f}")
                last_log = iterations
            
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
        
        if open_set:
            logger.warning(f"  A*失败: 达到最大迭代次数 {max_iterations}")
        else:
            logger.warning("  A*失败: open_set 耗尽（局部区域不可达或被障碍切断）")
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

        # 早期失败：连通域检测（比跑A*便宜很多）
        passable = np.isfinite(coarse_cost)
        lbl, _ = label(passable, structure=np.ones((3, 3), dtype=np.int8))
        if lbl[coarse_start[0], coarse_start[1]] == 0 or lbl[coarse_goal[0], coarse_goal[1]] == 0:
            logger.warning("  粗规划跳过：起终点不在可通行域")
            return None
        if lbl[coarse_start[0], coarse_start[1]] != lbl[coarse_goal[0], coarse_goal[1]]:
            logger.warning("  粗规划跳过：起终点在粗网格上不连通（降采样导致分裂），需要重采样")
            return None
        
        # 粗A*搜索：使用标准A*（权重1.0）以充分利用道路低代价，避免过度贪心走直线
        logger.info("  运行粗A*搜索...")
        coarse_path = self.astar(coarse_start, coarse_goal, coarse_cost, max_iterations=2000000, heuristic_weight=1.0)
        
        if coarse_path is None:
            logger.error("  粗规划失败！")
            return None
        
        logger.info(f"  ✓ 粗路径: {len(coarse_path)} 点")
        
        # 转换回UTM坐标
        coarse_path_utm = []
        for row, col in coarse_path:
            r0 = int(row * downsample_factor)
            c0 = int(col * downsample_factor)
            r1 = int(min(r0 + downsample_factor, self.shape[0]))
            c1 = int(min(c0 + downsample_factor, self.shape[1]))

            block = self.cost_map[r0:r1, c0:c1]
            finite = np.isfinite(block)
            if np.any(finite):
                masked = np.where(finite, block, np.inf)
                rr, cc = np.unravel_index(int(np.argmin(masked)), masked.shape)
                orig_row = int(r0 + rr)
                orig_col = int(c0 + cc)
            else:
                orig_row = int(max(0, min(r0 + downsample_factor // 2, self.shape[0] - 1)))
                orig_col = int(max(0, min(c0 + downsample_factor // 2, self.shape[1] - 1)))

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

        # clamp 到栅格范围
        h_full, w_full = self.cost_map.shape
        start_pixel = (
            max(0, min(int(start_pixel[0]), h_full - 1)),
            max(0, min(int(start_pixel[1]), w_full - 1)),
        )
        goal_pixel = (
            max(0, min(int(goal_pixel[0]), h_full - 1)),
            max(0, min(int(goal_pixel[1]), w_full - 1)),
        )

        # 确保起终点可通行，否则寻找最近可通行点
        if not np.isfinite(self.cost_map[start_pixel[0], start_pixel[1]]):
            nearest = self._find_nearest_passable(start_pixel, self.cost_map, max_radius=60)
            if nearest is None:
                return None
            start_pixel = nearest
        if not np.isfinite(self.cost_map[goal_pixel[0], goal_pixel[1]]):
            nearest = self._find_nearest_passable(goal_pixel, self.cost_map, max_radius=60)
            if nearest is None:
                return None
            goal_pixel = nearest
        
        # 直接在原始代价图上运行A*（降低启发权重以充分利用道路低代价）
        base_max_iter = int(min(3_000_000, max(500_000, self.cost_map.size * 0.05)))
        path_pixels = None
        for heuristic_weight in (1.0, 0.8, 1.1):
            path_pixels = self.astar(
                start_pixel,
                goal_pixel,
                self.cost_map,
                max_iterations=base_max_iter,
                heuristic_weight=heuristic_weight
            )
            if path_pixels is not None:
                break
        
        if path_pixels is None:
            return None
        
        # 转换回UTM
        path_utm = [self.pixel_to_utm(row, col) for row, col in path_pixels]
        
        return path_utm

    def refine_segment_corridor(
        self,
        start_utm: Tuple[float, float],
        goal_utm: Tuple[float, float],
        corridor_width: float = 2000.0,
        coarse_segment: Optional[List[Tuple[float, float]]] = None
    ) -> Optional[List[Tuple[float, float]]]:
        """在走廊窗口内细化单段路径（高分辨率A*）"""
        start_pixel = self.utm_to_pixel(*start_utm)
        goal_pixel = self.utm_to_pixel(*goal_utm)

        # clamp 到栅格范围，避免窗口裁剪后起终点落在窗口外导致越界
        h_full, w_full = self.cost_map.shape
        start_pixel = (
            max(0, min(int(start_pixel[0]), h_full - 1)),
            max(0, min(int(start_pixel[1]), w_full - 1)),
        )
        goal_pixel = (
            max(0, min(int(goal_pixel[0]), h_full - 1)),
            max(0, min(int(goal_pixel[1]), w_full - 1)),
        )

        # 确保起终点可通行，否则寻找最近可通行点
        if not np.isfinite(self.cost_map[start_pixel[0], start_pixel[1]]):
            nearest = self._find_nearest_passable(start_pixel, self.cost_map, max_radius=60)
            if nearest is None:
                return None
            start_pixel = nearest
        if not np.isfinite(self.cost_map[goal_pixel[0], goal_pixel[1]]):
            nearest = self._find_nearest_passable(goal_pixel, self.cost_map, max_radius=60)
            if nearest is None:
                return None
            goal_pixel = nearest

        pad_pix = int(corridor_width / self.resolution) + 5
        if coarse_segment is not None and len(coarse_segment) >= 2:
            seg_pixels = [self.utm_to_pixel(x, y) for x, y in coarse_segment]
            seg_rows = [p[0] for p in seg_pixels]
            seg_cols = [p[1] for p in seg_pixels]
            min_row0 = min(seg_rows)
            max_row0 = max(seg_rows)
            min_col0 = min(seg_cols)
            max_col0 = max(seg_cols)
        else:
            min_row0 = min(start_pixel[0], goal_pixel[0])
            max_row0 = max(start_pixel[0], goal_pixel[0])
            min_col0 = min(start_pixel[1], goal_pixel[1])
            max_col0 = max(start_pixel[1], goal_pixel[1])

        min_row = max(0, min_row0 - pad_pix)
        max_row = min(self.cost_map.shape[0] - 1, max_row0 + pad_pix)
        min_col = max(0, min_col0 - pad_pix)
        max_col = min(self.cost_map.shape[1] - 1, max_col0 + pad_pix)

        local_cost = self.cost_map[min_row:max_row + 1, min_col:max_col + 1].copy()
        local_cost = np.where(np.isfinite(local_cost), local_cost, np.inf)

        # 早期失败检测：窗口过大或可通行比例太低会导致 A* 极慢/无解
        if local_cost.size > 1_200_000:
            logger.debug(f"  跳过走廊细化：窗口过大 size={local_cost.size}")
            return None
        passable_ratio = float(np.isfinite(local_cost).mean())
        if passable_ratio < 0.02:
            logger.debug(f"  跳过走廊细化：可通行比例过低 ratio={passable_ratio:.4f}")
            return None

        s_local = (start_pixel[0] - min_row, start_pixel[1] - min_col)
        g_local = (goal_pixel[0] - min_row, goal_pixel[1] - min_col)

        # 防御：确保局部索引不越界
        if not (0 <= s_local[0] < local_cost.shape[0] and 0 <= s_local[1] < local_cost.shape[1]):
            return None
        if not (0 <= g_local[0] < local_cost.shape[0] and 0 <= g_local[1] < local_cost.shape[1]):
            return None

        # 早期失败：连通域检测（避免A*白跑max_iter）
        # local_cost 已将不可通行设为 inf；用 isfinite 判断可通行
        if local_cost.size <= 600_000:
            passable_local = np.isfinite(local_cost)
            lbl, _ = label(passable_local, structure=np.ones((3, 3), dtype=np.int8))
            sid = lbl[s_local[0], s_local[1]]
            gid = lbl[g_local[0], g_local[1]]
            if sid == 0 or gid == 0 or sid != gid:
                return None

        # 走廊实现说明：论文中的走廊是围绕粗路径的可行区域。
        # 这里为了鲁棒性，使用“窗口走廊”（bounding box + buffer）限制搜索范围，
        # 避免用直线距离 mask 造成图被切断而无解。

        # 动态迭代上限：与窗口规模成比例，避免超大窗口无限耗时
        max_iter = int(min(300000, max(50000, local_cost.size * 1.5)))
        path_local = self.astar(s_local, g_local, local_cost, max_iterations=max_iter, heuristic_weight=1.8)
        if path_local is None:
            return None

        path_utm = []
        for r, c in path_local:
            fr = int(r + min_row)
            fc = int(c + min_col)
            path_utm.append(self.pixel_to_utm(fr, fc))
        return path_utm
    
    def hierarchical_plan(
        self,
        start_utm: Tuple[float, float],
        goal_utm: Tuple[float, float],
        downsample_factor: int = 10,
        waypoint_interval: float = 5000.0,
        corridor_width: float = 2000.0,
        refine_mode: str = 'repair',
        densify_step_m: float = 200.0,
        high_cost_threshold: float = 0.5,
        high_cost_fraction: float = 0.2
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

        # 步骤2: 航路点（沿粗路径每5km）
        waypoints = self.extract_waypoints(coarse_path, waypoint_interval=waypoint_interval)
        if waypoints is None or len(waypoints) < 2:
            return None

        def _nearest_coarse_idx(pt: Tuple[float, float]) -> int:
            arr = np.asarray(coarse_path, dtype=np.float64)
            d2 = (arr[:, 0] - pt[0]) ** 2 + (arr[:, 1] - pt[1]) ** 2
            return int(np.argmin(d2))

        def _densify_polyline(poly: List[Tuple[float, float]], max_step_m: float = 200.0) -> List[Tuple[float, float]]:
            if poly is None or len(poly) < 2:
                return poly
            out: List[Tuple[float, float]] = [poly[0]]
            for i in range(1, len(poly)):
                x0, y0 = out[-1]
                x1, y1 = poly[i]
                dx = float(x1 - x0)
                dy = float(y1 - y0)
                dist = float(np.hypot(dx, dy))
                if dist <= max_step_m:
                    out.append((x1, y1))
                    continue
                n = int(np.ceil(dist / max_step_m))
                for k in range(1, n + 1):
                    t = k / n
                    out.append((x0 + dx * t, y0 + dy * t))
            return out

        def _snap_polyline_to_passable(
            poly: List[Tuple[float, float]],
            max_radius_pix: int = 50,
            max_snap_fraction: float = 0.95
        ) -> Optional[List[Tuple[float, float]]]:
            if poly is None or len(poly) < 2:
                return None
            snapped: List[Tuple[float, float]] = []
            snap_cnt = 0
            checked = 0
            for x, y in poly:
                r, c = self.utm_to_pixel(x, y)
                if r < 0 or r >= self.cost_map.shape[0] or c < 0 or c >= self.cost_map.shape[1]:
                    return None
                v = self.cost_map[r, c]
                checked += 1
                if np.isfinite(v):
                    snapped.append((x, y))
                    continue

                nearest = self._find_nearest_passable((r, c), self.cost_map, max_radius=max_radius_pix)
                if nearest is None:
                    return None
                snap_cnt += 1
                xr, yr = self.pixel_to_utm(nearest[0], nearest[1])
                snapped.append((xr, yr))

            if checked > 0 and (snap_cnt / checked) > max_snap_fraction:
                return None

            out: List[Tuple[float, float]] = [snapped[0]]
            for i in range(1, len(snapped)):
                if snapped[i] != out[-1]:
                    out.append(snapped[i])
            if len(out) < 2:
                return None
            return out

        def _segment_needs_repair(
            poly: List[Tuple[float, float]],
            cost_threshold: float,
            fraction_threshold: float
        ) -> bool:
            if poly is None or len(poly) < 2:
                return True
            if cost_threshold >= 1.0:
                return False
            high_cost_cnt = 0
            checked = 0
            for idx, (x, y) in enumerate(poly):
                if idx % 3 != 0 and idx != len(poly) - 1:
                    continue
                r, c = self.utm_to_pixel(x, y)
                if r < 0 or r >= self.cost_map.shape[0] or c < 0 or c >= self.cost_map.shape[1]:
                    return True
                v = self.cost_map[r, c]
                if not np.isfinite(v):
                    return True
                checked += 1
                if float(v) > cost_threshold:
                    high_cost_cnt += 1
            if checked <= 0:
                return False
            return (high_cost_cnt / checked) >= fraction_threshold

        # 步骤3: 分段细化（高分辨率A*，2km走廊）
        logger.info(f"\n步骤3: 分段细化（走廊宽度={corridor_width}m）")
        refined_path: List[Tuple[float, float]] = []
        direct_cnt = 0
        repaired_cnt = 0
        astar_cnt = 0
        for i in range(len(waypoints) - 1):
            seg_start = waypoints[i]
            seg_goal = waypoints[i + 1]

            # 取粗路径中对应段的点列，用于构建更贴合粗路径的窗口走廊
            i0 = _nearest_coarse_idx(seg_start)
            i1 = _nearest_coarse_idx(seg_goal)
            if i0 <= i1:
                coarse_seg = coarse_path[i0:i1 + 1]
            else:
                coarse_seg = list(reversed(coarse_path[i1:i0 + 1]))

            seg = None

            if refine_mode in ('densify_only', 'repair'):
                if coarse_seg is None or len(coarse_seg) < 2:
                    candidate = [seg_start, seg_goal]
                else:
                    candidate = list(coarse_seg)
                    candidate[0] = seg_start
                    candidate[-1] = seg_goal

                candidate = _densify_polyline(candidate, max_step_m=float(densify_step_m))
                if candidate is not None and len(candidate) >= 2:
                    if refine_mode == 'densify_only':
                        seg = candidate
                        direct_cnt += 1
                    else:
                        snapped = _snap_polyline_to_passable(candidate)
                        if snapped is not None and not _segment_needs_repair(snapped, float(high_cost_threshold), float(high_cost_fraction)):
                            seg = snapped
                            direct_cnt += 1
                        else:
                            repaired_cnt += 1

            if seg is None:
                for cw in (corridor_width, corridor_width * 2.0, corridor_width * 4.0):
                    seg = self.refine_segment_corridor(seg_start, seg_goal, corridor_width=cw, coarse_segment=coarse_seg)
                    if seg is not None:
                        astar_cnt += 1
                        break

            if seg is None:
                for cw in (corridor_width * 4.0, corridor_width * 8.0, corridor_width * 12.0):
                    seg = self.refine_segment_corridor(seg_start, seg_goal, corridor_width=cw, coarse_segment=None)
                    if seg is not None:
                        astar_cnt += 1
                        break
            if seg is None:
                # 走廊失败则退化为全图高分辨率细化
                seg = self.refine_segment(seg_start, seg_goal)
                if seg is not None:
                    astar_cnt += 1
            if seg is None:
                logger.warning(
                    f"  段细化失败，使用粗路径段fallback: seg={i+1}/{len(waypoints)-1}, start={seg_start}, goal={seg_goal}"
                )
                if coarse_seg is None or len(coarse_seg) < 2:
                    seg = [seg_start, seg_goal]
                else:
                    seg = list(coarse_seg)
                    seg[0] = seg_start
                    seg[-1] = seg_goal
                seg = _densify_polyline(seg, max_step_m=float(densify_step_m))
                if seg is None or len(seg) < 2:
                    return None

            if len(refined_path) == 0:
                refined_path.extend(seg)
            else:
                refined_path.extend(seg[1:])

        logger.info(
            f"  细化模式: {refine_mode}, direct={direct_cnt}, repair_triggered={repaired_cnt}, astar_used={astar_cnt}"
        )

        logger.info(f"\n✅ 分层A*规划完成（粗规划+航路点+细化）")
        logger.info(f"  路径点数: {len(refined_path)}")

        path_length = 0.0
        for i in range(1, len(refined_path)):
            dist = np.sqrt((refined_path[i][0] - refined_path[i-1][0])**2 + (refined_path[i][1] - refined_path[i-1][1])**2)
            path_length += dist

        logger.info(f"  路径长度: {path_length/1000:.2f} km")
        logger.info(f"  迂回系数: {path_length/straight_dist:.2f}")

        return refined_path


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
