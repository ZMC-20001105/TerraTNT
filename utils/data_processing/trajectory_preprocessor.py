"""
轨迹数据预处理模块
生成TerraTNT模型所需的18通道环境地图和训练数据
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pickle
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentMapGenerator:
    """生成18通道环境地图"""
    
    def __init__(self, region: str, map_size: int = 128, pixel_resolution: float = 781.25):
        """
        Args:
            region: 区域名称
            map_size: 地图尺寸 (128x128)
            pixel_resolution: 像素分辨率 (m), 100km / 128 = 781.25m
        """
        self.region = region
        self.map_size = map_size
        self.pixel_resolution = pixel_resolution
        self.utm_dir = Path(f'/home/zmc/文档/programwork/data/processed/utm_grid/{region}')
        
        # 加载环境栅格
        self._load_environment_rasters()
        
    def _load_environment_rasters(self):
        """加载环境栅格数据"""
        logger.info(f"加载 {self.region} 环境栅格数据...")
        
        with rasterio.open(self.utm_dir / 'dem_utm.tif') as src:
            self.dem = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            
        with rasterio.open(self.utm_dir / 'slope_utm.tif') as src:
            self.slope = src.read(1)
            
        with rasterio.open(self.utm_dir / 'aspect_utm.tif') as src:
            self.aspect = src.read(1)
            
        with rasterio.open(self.utm_dir / 'lulc_utm.tif') as src:
            self.lulc = src.read(1)
        
        logger.info(f"  DEM尺寸: {self.dem.shape}")
        logger.info(f"  分辨率: {self.transform.a}m")
        
    def extract_local_map(self, center_utm: Tuple[float, float], 
                         history_points: Optional[np.ndarray] = None,
                         goal_utm: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        提取以center_utm为中心的局部地图
        
        Args:
            center_utm: 中心点UTM坐标 (x, y)
            history_points: 历史轨迹点 (N, 2) UTM坐标
            goal_utm: 目标点UTM坐标 (x, y)
            
        Returns:
            18通道地图 (18, 128, 128)
        """
        # 计算地图范围 (100km x 100km)
        map_extent = self.map_size * self.pixel_resolution  # 100,000m
        half_extent = map_extent / 2
        
        x_min = center_utm[0] - half_extent
        x_max = center_utm[0] + half_extent
        y_min = center_utm[1] - half_extent
        y_max = center_utm[1] + half_extent
        
        # 转换为栅格坐标
        col_min = int((x_min - self.transform.c) / self.transform.a)
        col_max = int((x_max - self.transform.c) / self.transform.a)
        row_min = int((y_max - self.transform.f) / self.transform.e)  # 注意y轴反向
        row_max = int((y_min - self.transform.f) / self.transform.e)
        
        # 确保在边界内
        col_min = max(0, col_min)
        col_max = min(self.dem.shape[1], col_max)
        row_min = max(0, row_min)
        row_max = min(self.dem.shape[0], row_max)
        
        # 提取局部栅格
        dem_local = self.dem[row_min:row_max, col_min:col_max]
        slope_local = self.slope[row_min:row_max, col_min:col_max]
        aspect_local = self.aspect[row_min:row_max, col_min:col_max]
        lulc_local = self.lulc[row_min:row_max, col_min:col_max]
        
        # 调整大小到128x128
        from scipy.ndimage import zoom
        zoom_factor = (self.map_size / dem_local.shape[0], 
                      self.map_size / dem_local.shape[1])
        
        dem_resized = zoom(dem_local, zoom_factor, order=1)
        slope_resized = zoom(slope_local, zoom_factor, order=1)
        aspect_resized = zoom(aspect_local, zoom_factor, order=1)
        lulc_resized = zoom(lulc_local, zoom_factor, order=0)  # 最近邻插值
        
        # 生成18通道地图
        channels = []
        
        # 通道1: DEM (归一化)
        dem_norm = (dem_resized - dem_resized.mean()) / (dem_resized.std() + 1e-6)
        channels.append(dem_norm)
        
        # 通道2: Slope (归一化到0-1)
        slope_norm = slope_resized / 90.0
        channels.append(slope_norm)
        
        # 通道3-4: Aspect (sin, cos)
        aspect_rad = np.deg2rad(aspect_resized)
        aspect_rad[aspect_resized < 0] = 0  # 平地设为0
        channels.append(np.sin(aspect_rad))
        channels.append(np.cos(aspect_rad))
        
        # 通道5-14: LULC one-hot编码 (10类)
        lulc_codes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for code in lulc_codes:
            lulc_channel = (lulc_resized == code).astype(np.float32)
            channels.append(lulc_channel)
        
        # 通道15: 道路层 (LULC=80为人工表面)
        road_channel = (lulc_resized == 80).astype(np.float32)
        channels.append(road_channel)
        
        # 通道16: 历史轨迹热力图
        history_heatmap = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        if history_points is not None and len(history_points) > 0:
            for point in history_points:
                # 转换到局部地图坐标
                local_x = (point[0] - x_min) / self.pixel_resolution
                local_y = (y_max - point[1]) / self.pixel_resolution
                
                if 0 <= local_x < self.map_size and 0 <= local_y < self.map_size:
                    col = int(local_x)
                    row = int(local_y)
                    # 高斯模糊效果
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            r, c = row + dr, col + dc
                            if 0 <= r < self.map_size and 0 <= c < self.map_size:
                                dist = np.sqrt(dr**2 + dc**2)
                                history_heatmap[r, c] += np.exp(-dist**2 / 2)
        
        history_heatmap = np.clip(history_heatmap, 0, 1)
        channels.append(history_heatmap)
        
        # 通道17: 候选目标地图
        goal_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        if goal_utm is not None:
            local_x = (goal_utm[0] - x_min) / self.pixel_resolution
            local_y = (y_max - goal_utm[1]) / self.pixel_resolution
            
            if 0 <= local_x < self.map_size and 0 <= local_y < self.map_size:
                col = int(local_x)
                row = int(local_y)
                # 高斯峰值
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        r, c = row + dr, col + dc
                        if 0 <= r < self.map_size and 0 <= c < self.map_size:
                            dist = np.sqrt(dr**2 + dc**2)
                            goal_map[r, c] = np.exp(-dist**2 / 4)
        
        channels.append(goal_map)
        
        # 通道18: 缺失值标记 (全0表示无缺失)
        missing_mask = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        channels.append(missing_mask)
        
        # 堆叠为 (18, 128, 128)
        env_map = np.stack(channels, axis=0)
        
        return env_map.astype(np.float32)


class TrajectoryDataset(Dataset):
    """轨迹数据集"""
    
    def __init__(self, 
                 region: str,
                 trajectory_dir: Path,
                 fas_stage: str = 'FAS1',
                 history_length: int = 10,
                 future_length: int = 60,
                 sampling_interval: int = 60):
        """
        Args:
            region: 区域名称
            trajectory_dir: 轨迹数据目录
            fas_stage: 实验阶段 (FAS1, FAS2, FAS3)
            history_length: 历史轨迹长度 (10分钟)
            future_length: 未来轨迹长度 (60分钟)
            sampling_interval: 采样间隔 (秒)
        """
        self.region = region
        self.trajectory_dir = Path(trajectory_dir)
        self.fas_stage = fas_stage
        self.history_length = history_length
        self.future_length = future_length
        self.sampling_interval = sampling_interval
        
        # 加载所有轨迹文件
        self.trajectory_files = sorted(list(self.trajectory_dir.glob('*.pkl')))
        logger.info(f"加载 {len(self.trajectory_files)} 条轨迹")
        
        # 初始化环境地图生成器
        self.map_generator = EnvironmentMapGenerator(region)
        
        # 预处理：为每条轨迹生成训练样本
        self.samples = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        """准备训练样本"""
        logger.info("准备训练样本...")
        
        for traj_file in self.trajectory_files:
            with open(traj_file, 'rb') as f:
                traj_data = pickle.load(f)
            
            path_utm = np.array(traj_data.get('path_utm', traj_data.get('path')))
            timestamps = np.array(traj_data.get('timestamps_s', traj_data.get('timestamps')))
            
            # 按采样间隔重采样
            sampled_indices = []
            for i in range(0, len(timestamps), self.sampling_interval):
                sampled_indices.append(i)
            
            if len(sampled_indices) < self.history_length + self.future_length:
                continue  # 轨迹太短，跳过
            
            # 滑动窗口生成样本
            for i in range(len(sampled_indices) - self.history_length - self.future_length + 1):
                history_idx = sampled_indices[i:i + self.history_length]
                future_idx = sampled_indices[i + self.history_length:
                                            i + self.history_length + self.future_length]
                
                sample = {
                    'trajectory_id': traj_data.get('trajectory_id', traj_file.stem),
                    'history_points': path_utm[history_idx],
                    'future_points': path_utm[future_idx],
                    'goal_point': path_utm[future_idx[-1]],
                    'intent': traj_data['intent'],
                    'vehicle_type': traj_data['vehicle_type'],
                    'full_path_utm': path_utm
                }
                
                self.samples.append(sample)
        
        logger.info(f"生成 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 当前位置（历史轨迹最后一点）
        current_pos = sample['history_points'][-1]
        
        # 根据FAS阶段准备目标点信息
        goal_utm = sample['goal_point']
        candidate_goals = []
        
        if self.fas_stage == 'FAS3':
            # 场景FAS3: 提供候选目标点集合 (1个真值 + N个负采样)
            candidate_goals.append(goal_utm)
            # 模拟候选目标生成: 在真值周围5km内随机偏移，实际应从地理POI或代价地图局部极小点采样
            for _ in range(5):
                offset = np.random.uniform(-5000, 5000, size=2)
                candidate_goals.append(goal_utm + offset)
            np.random.shuffle(candidate_goals)
            
        # 生成18通道环境地图
        env_map = self.map_generator.extract_local_map(
            center_utm=current_pos,
            history_points=sample['history_points'],
            goal_utm=goal_utm
        )
        
        # 归一化历史轨迹（相对于当前位置）
        history_rel = sample['history_points'] - current_pos
        history_rel = history_rel / 1000.0  # 归一化到km
        
        # 归一化未来轨迹（相对于当前位置）
        future_rel = sample['future_points'] - current_pos
        future_rel = future_rel / 1000.0
        
        res = {
            'env_map': torch.from_numpy(env_map),  # (18, 128, 128)
            'history': torch.from_numpy(history_rel).float(),  # (10, 2)
            'future': torch.from_numpy(future_rel).float(),  # (60, 2)
            'goal': torch.from_numpy(sample['goal_point'] - current_pos).float() / 1000.0,  # (2,)
            'current_pos': torch.from_numpy(current_pos).float(),  # (2,)
            'trajectory_id': sample['trajectory_id']
        }
        
        if self.fas_stage == 'FAS3':
            res['candidate_goals'] = torch.from_numpy(np.array(candidate_goals) - current_pos).float() / 1000.0
            
        return res


def create_dataloaders(region: str, 
                      trajectory_dir: Path,
                      batch_size: int = 32,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      num_workers: int = 4):
    """
    创建训练/验证/测试数据加载器
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, random_split
    
    # 创建完整数据集
    full_dataset = TrajectoryDataset(region, trajectory_dir)
    
    # 划分数据集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"数据集划分: 训练={train_size}, 验证={val_size}, 测试={test_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据预处理
    region = 'scottish_highlands'
    trajectory_dir = Path(f'/home/zmc/文档/programwork/data/processed/synthetic_trajectories/{region}')
    
    logger.info("=" * 60)
    logger.info("测试数据预处理管道")
    logger.info("=" * 60)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        region=region,
        trajectory_dir=trajectory_dir,
        batch_size=4,
        num_workers=0  # 测试时使用单进程
    )
    
    # 测试一个batch
    logger.info("\n测试数据加载...")
    for batch in train_loader:
        logger.info(f"Batch shapes:")
        logger.info(f"  env_map: {batch['env_map'].shape}")
        logger.info(f"  history: {batch['history'].shape}")
        logger.info(f"  future: {batch['future'].shape}")
        logger.info(f"  goal: {batch['goal'].shape}")
        logger.info(f"  current_pos: {batch['current_pos'].shape}")
        break
    
    logger.info("\n✅ 数据预处理管道测试成功！")
