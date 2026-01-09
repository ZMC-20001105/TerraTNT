"""
轨迹生成器 (Trajectory Generator)

整合完整流程：
1. 分层A*路径规划
2. 路径平滑（三次样条）
3. 速度预测（XGBoost）
4. 运动学约束
5. 时间戳积分
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pickle

from .hierarchical_astar import HierarchicalAStarPlanner
from .path_smoothing import smooth_path
from .sampling import sample_start_goal
from config import cfg, get_path

logger = logging.getLogger(__name__)

# 论文表3.2：车辆运动学参数
VEHICLE_PARAMS = {
    'type1': {'v_max': 18.0, 'a_max': 2.0, 'R_min': 8.0},
    'type2': {'v_max': 22.0, 'a_max': 2.5, 'R_min': 6.0},
    'type3': {'v_max': 25.0, 'a_max': 3.0, 'R_min': 5.0},
    'type4': {'v_max': 28.0, 'a_max': 3.5, 'R_min': 10.0}
}


class TrajectoryGenerator:
    """轨迹生成器"""
    
    def __init__(self, region: str = 'scottish_highlands'):
        self.region = region
        
        # 加载速度预测模型
        model_path = Path(get_path('paths.models.speed_predictor')) / 'speed_model.pkl'
        logger.info(f"加载速度预测模型: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.speed_model = model_data['model']
            self.feature_names = model_data['feature_names']
        
        # 加载环境栅格用于特征提取
        utm_dir = Path(get_path('paths.processed.utm_grid')) / region
        
        import rasterio
        self.dem_src = rasterio.open(utm_dir / 'dem_utm.tif')
        self.slope_src = rasterio.open(utm_dir / 'slope_utm.tif')
        self.aspect_src = rasterio.open(utm_dir / 'aspect_utm.tif')
        self.lulc_src = rasterio.open(utm_dir / 'lulc_utm.tif')
        
        self.dem = self.dem_src.read(1)
        self.slope = self.slope_src.read(1)
        self.aspect = self.aspect_src.read(1)
        self.lulc = self.lulc_src.read(1)
        
        # 处理nodata
        self.dem = np.where(self.dem == -32768, np.nan, self.dem)
        
        self.transform = self.dem_src.transform
        
        logger.info("✓ 速度预测模型和环境栅格已加载")
    
    def compute_curvature(self, path: List[Tuple[float, float]]) -> np.ndarray:
        """计算路径曲率"""
        n = len(path)
        if n < 3:
            return np.zeros(n)
        
        x = np.array([p[0] for p in path])
        y = np.array([p[1] for p in path])
        
        # 计算方向角
        dx = np.diff(x)
        dy = np.diff(y)
        theta = np.arctan2(dy, dx)
        
        # 计算角度变化
        dtheta = np.diff(theta)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        # 计算弧长
        ds = np.sqrt(dx[:-1]**2 + dy[:-1]**2)
        ds = np.where(ds < 0.1, 0.1, ds)
        
        # 曲率
        curvature = np.abs(dtheta / ds)
        
        # 填充
        result = np.zeros(n)
        result[1:-1] = curvature
        result[0] = curvature[0] if len(curvature) > 0 else 0
        result[-1] = curvature[-1] if len(curvature) > 0 else 0
        
        return result
    
    def utm_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """UTM坐标转像素坐标"""
        col = int((x - self.transform.c) / self.transform.a)
        row = int((y - self.transform.f) / self.transform.e)
        return row, col
    
    def sample_raster(self, raster: np.ndarray, x: float, y: float) -> float:
        """从栅格采样值"""
        try:
            row, col = self.utm_to_pixel(x, y)
            if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
                return float(raster[row, col])
        except:
            pass
        return np.nan
    
    def extract_features_for_path(self, path: List[Tuple[float, float]]) -> np.ndarray:
        """
        为路径上的每个点提取20维特征
        
        特征（论文表3.6）：
        1. DEM (1)
        2. Slope (1)
        3. Aspect sin/cos (2)
        4. LULC one-hot (10)
        5. Tree cover (1)
        6. Effective slope (1)
        7. Curvature (1)
        8. Past 10m avg curvature (1)
        9. Future 10m max curvature (1)
        10. On road (1)
        """
        n = len(path)
        features = np.zeros((n, 20))
        
        x = np.array([p[0] for p in path])
        y = np.array([p[1] for p in path])
        
        # 采样环境栅格
        dem_vals = np.array([self.sample_raster(self.dem, x[i], y[i]) for i in range(n)])
        slope_vals = np.array([self.sample_raster(self.slope, x[i], y[i]) for i in range(n)])
        aspect_vals = np.array([self.sample_raster(self.aspect, x[i], y[i]) for i in range(n)])
        lulc_vals = np.array([self.sample_raster(self.lulc, x[i], y[i]) for i in range(n)])
        
        # 特征1: DEM
        features[:, 0] = dem_vals
        
        # 特征2: Slope
        features[:, 1] = slope_vals
        
        # 特征3-4: Aspect sin/cos
        aspect_rad = np.deg2rad(aspect_vals)
        features[:, 2] = np.sin(aspect_rad)
        features[:, 3] = np.cos(aspect_rad)
        
        # 特征5-14: LULC one-hot
        LULC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i, lulc_class in enumerate(LULC_CLASSES):
            features[:, 4 + i] = (lulc_vals == lulc_class).astype(float)
        
        # 特征15: Tree cover
        features[:, 14] = (lulc_vals == 20).astype(float)
        
        # 特征16: Effective slope
        if n > 1:
            dh = np.diff(dem_vals)
            ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            ds = np.where(ds < 0.1, 0.1, ds)
            eff_slope = np.rad2deg(np.arctan(dh / ds))
            features[:-1, 15] = eff_slope
            features[-1, 15] = eff_slope[-1] if len(eff_slope) > 0 else 0
        
        # 特征17: Curvature
        curvature = self.compute_curvature(path)
        features[:, 16] = curvature
        
        # 特征18: Past 10m avg curvature
        for i in range(n):
            if i == 0:
                features[i, 17] = curvature[i]
            else:
                dist_back = 0
                j = i - 1
                curv_sum = 0
                count = 0
                while j >= 0 and dist_back < 10:
                    dist_back += np.sqrt((x[j+1]-x[j])**2 + (y[j+1]-y[j])**2)
                    curv_sum += curvature[j]
                    count += 1
                    j -= 1
                features[i, 17] = curv_sum / count if count > 0 else curvature[i]
        
        # 特征19: Future 10m max curvature
        for i in range(n):
            if i == n - 1:
                features[i, 18] = curvature[i]
            else:
                dist_fwd = 0
                j = i + 1
                curv_max = 0
                while j < n and dist_fwd < 10:
                    dist_fwd += np.sqrt((x[j]-x[j-1])**2 + (y[j]-y[j-1])**2)
                    curv_max = max(curv_max, curvature[j])
                    j += 1
                features[i, 18] = curv_max
        
        # 特征20: On road
        features[:, 19] = (lulc_vals == 80).astype(float)
        
        return features
    
    def predict_speeds(
        self,
        path: List[Tuple[float, float]],
        vehicle_type: str
    ) -> np.ndarray:
        """
        使用XGBoost模型预测路径上每点的速度
        
        Returns:
            速度数组（m/s）
        """
        n = len(path)
        
        # 提取20维特征
        features = self.extract_features_for_path(path)
        
        # 处理NaN值（用0填充）
        features = np.nan_to_num(features, nan=0.0)
        
        # 使用XGBoost模型预测 log(1+v)
        y_pred = self.speed_model.predict(features)
        
        # 转换回速度 v = exp(y) - 1
        speeds = np.exp(y_pred) - 1
        
        # 应用车辆最大速度限制
        params = VEHICLE_PARAMS[vehicle_type]
        v_max = params['v_max']
        speeds = np.clip(speeds, 0.5, v_max)
        
        # 轻微平滑（避免突变）
        from scipy.ndimage import gaussian_filter1d
        speeds = gaussian_filter1d(speeds, sigma=1.0)
        
        return speeds
    
    def apply_kinematic_constraints(
        self,
        speeds: np.ndarray,
        distances: np.ndarray,
        vehicle_type: str
    ) -> np.ndarray:
        """
        应用运动学约束（加速度限制）
        
        Args:
            speeds: 初始速度
            distances: 相邻点距离
            vehicle_type: 车辆类型
        
        Returns:
            约束后的速度
        """
        params = VEHICLE_PARAMS[vehicle_type]
        a_max = params['a_max']
        
        constrained_speeds = speeds.copy()
        
        # 前向传播：限制加速
        for i in range(1, len(speeds)):
            if distances[i-1] > 0:
                # 最大可达速度（考虑加速度限制）
                dt = distances[i-1] / max(constrained_speeds[i-1], 0.1)
                v_max_accel = constrained_speeds[i-1] + a_max * dt
                
                constrained_speeds[i] = min(constrained_speeds[i], v_max_accel)
        
        # 后向传播：限制减速
        for i in range(len(speeds) - 2, -1, -1):
            if distances[i] > 0:
                dt = distances[i] / max(constrained_speeds[i+1], 0.1)
                v_max_decel = constrained_speeds[i+1] + a_max * dt
                
                constrained_speeds[i] = min(constrained_speeds[i], v_max_decel)
        
        return constrained_speeds
    
    def compute_timestamps(
        self,
        path: List[Tuple[float, float]],
        speeds: np.ndarray
    ) -> np.ndarray:
        """
        通过前向积分计算时间戳
        
        Δt = Δs / v_avg
        
        Returns:
            时间戳数组（秒）
        """
        n = len(path)
        timestamps = np.zeros(n)
        
        for i in range(1, n):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            ds = np.sqrt(dx**2 + dy**2)
            
            v_avg = (speeds[i-1] + speeds[i]) / 2.0
            v_avg = max(v_avg, 0.1)  # 避免除零
            
            dt = ds / v_avg
            timestamps[i] = timestamps[i-1] + dt
        
        return timestamps
    
    def generate_trajectory(
        self,
        intent: str,
        vehicle_type: str,
        start_utm: Optional[Tuple[float, float]] = None,
        goal_utm: Optional[Tuple[float, float]] = None,
        min_distance: float = 80000.0
    ) -> Optional[Dict]:
        """
        生成完整轨迹
        
        Args:
            intent: 战术意图
            vehicle_type: 车辆类型
            start_utm: 起点（None则随机采样）
            goal_utm: 终点（None则随机采样）
            min_distance: 最小直线距离
        
        Returns:
            轨迹字典或None
        """
        logger.info("=" * 60)
        logger.info(f"生成轨迹 - 意图: {intent}, 车辆: {vehicle_type}")
        logger.info("=" * 60)
        
        # 采样起终点
        if start_utm is None or goal_utm is None:
            logger.info("\n步骤1: 采样起终点")
            result = sample_start_goal(self.region, vehicle_type, min_distance)
            if result is None:
                logger.error("起终点采样失败")
                return None
            start_utm, goal_utm = result
        else:
            logger.info(f"\n使用指定起终点")
            logger.info(f"  起点: {start_utm}")
            logger.info(f"  终点: {goal_utm}")
        
        # 路径规划
        logger.info("\n步骤2: 分层A*路径规划")
        planner = HierarchicalAStarPlanner(self.region, intent, vehicle_type)
        
        path = planner.hierarchical_plan(
            start_utm,
            goal_utm,
            downsample_factor=10
        )
        
        if path is None:
            logger.error("路径规划失败")
            return None
        
        # 路径平滑
        logger.info("\n步骤3: 路径平滑")
        smoothed_path = smooth_path(path, smoothing_factor=0.0, resample_max_dist=100.0)
        
        # 速度预测
        logger.info("\n步骤4: 速度预测")
        speeds = self.predict_speeds(smoothed_path, vehicle_type)
        logger.info(f"  速度范围: {speeds.min():.2f} ~ {speeds.max():.2f} m/s")
        logger.info(f"  平均速度: {speeds.mean():.2f} m/s")
        
        # 计算距离
        distances = np.zeros(len(smoothed_path))
        for i in range(1, len(smoothed_path)):
            dx = smoothed_path[i][0] - smoothed_path[i-1][0]
            dy = smoothed_path[i][1] - smoothed_path[i-1][1]
            distances[i] = np.sqrt(dx**2 + dy**2)
        
        # 运动学约束
        logger.info("\n步骤5: 应用运动学约束")
        constrained_speeds = self.apply_kinematic_constraints(speeds, distances, vehicle_type)
        logger.info(f"  约束后速度: {constrained_speeds.min():.2f} ~ {constrained_speeds.max():.2f} m/s")
        
        # 时间戳
        logger.info("\n步骤6: 计算时间戳")
        timestamps = self.compute_timestamps(smoothed_path, constrained_speeds)
        duration = timestamps[-1]
        logger.info(f"  轨迹时长: {duration/60:.2f} 分钟")
        
        # 构建轨迹字典
        trajectory = {
            'region': self.region,
            'intent': intent,
            'vehicle_type': vehicle_type,
            'start_utm': start_utm,
            'goal_utm': goal_utm,
            'path': smoothed_path,
            'speeds': constrained_speeds,
            'timestamps': timestamps,
            'duration': duration,
            'length': np.sum(distances),
            'num_points': len(smoothed_path)
        }
        
        logger.info(f"\n✅ 轨迹生成完成")
        logger.info(f"  点数: {len(smoothed_path)}")
        logger.info(f"  长度: {trajectory['length']/1000:.2f} km")
        logger.info(f"  时长: {duration/60:.2f} 分钟")
        logger.info(f"  平均速度: {trajectory['length']/duration:.2f} m/s")
        
        return trajectory
    
    def save_trajectory(self, trajectory: Dict, output_path: Path):
        """保存轨迹"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(trajectory, f)
        
        logger.info(f"✓ 轨迹已保存: {output_path}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 测试：生成一条轨迹
    generator = TrajectoryGenerator('scottish_highlands')
    
    trajectory = generator.generate_trajectory(
        intent='intent1',
        vehicle_type='type1',
        min_distance=80000.0
    )
    
    if trajectory:
        # 保存
        output_dir = Path(get_path('paths.processed.synthetic_trajectories'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'test_trajectory.pkl'
        generator.save_trajectory(trajectory, output_path)
        
        print(f"\n✅ 测试轨迹已生成并保存")
    else:
        print("\n❌ 轨迹生成失败")
