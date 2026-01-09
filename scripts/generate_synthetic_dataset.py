"""
生成大规模合成轨迹数据集

按论文要求生成：
- 多个区域
- 4种车辆类型
- 3种战术意图
- 每种组合生成多条轨迹
"""
import logging
import sys
from pathlib import Path
import argparse
from datetime import datetime
import pickle
import json
from typing import Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.trajectory_generation.trajectory_generator import TrajectoryGenerator
from config import cfg, get_path

logger = logging.getLogger(__name__)


def generate_dataset(
    region: str,
    num_trajectories_per_config: int = 10,
    min_distance: float = 80000.0,
    output_dir: Optional[Path] = None
):
    """
    为指定区域生成完整数据集
    
    Args:
        region: 区域名称
        num_trajectories_per_config: 每种配置生成的轨迹数
        min_distance: 最小直线距离（米）
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = Path(get_path('paths.processed.synthetic_trajectories')) / region
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"生成合成轨迹数据集 - 区域: {region}")
    logger.info("=" * 60)
    logger.info(f"每种配置轨迹数: {num_trajectories_per_config}")
    logger.info(f"最小直线距离: {min_distance/1000:.1f} km")
    logger.info(f"输出目录: {output_dir}")
    logger.info("")
    
    # 初始化生成器
    generator = TrajectoryGenerator(region)
    
    # 配置组合
    intents = ['intent1', 'intent2', 'intent3']
    vehicle_types = ['type1', 'type2', 'type3', 'type4']
    
    total_configs = len(intents) * len(vehicle_types)
    total_trajectories = total_configs * num_trajectories_per_config
    
    logger.info(f"配置组合数: {total_configs}")
    logger.info(f"目标轨迹总数: {total_trajectories}")
    logger.info("")
    
    # 统计信息
    stats = {
        'region': region,
        'start_time': datetime.now().isoformat(),
        'num_trajectories_per_config': num_trajectories_per_config,
        'min_distance': min_distance,
        'total_configs': total_configs,
        'total_trajectories': total_trajectories,
        'generated': 0,
        'failed': 0,
        'configs': {}
    }
    
    # 生成轨迹
    trajectory_id = 0
    
    for intent in intents:
        for vehicle_type in vehicle_types:
            config_key = f"{intent}_{vehicle_type}"
            
            logger.info("=" * 60)
            logger.info(f"配置: {config_key}")
            logger.info("=" * 60)
            
            config_stats = {
                'intent': intent,
                'vehicle_type': vehicle_type,
                'generated': 0,
                'failed': 0,
                'trajectories': []
            }
            
            for i in range(num_trajectories_per_config):
                logger.info(f"\n[{trajectory_id+1}/{total_trajectories}] 生成轨迹 {i+1}/{num_trajectories_per_config}")
                
                try:
                    trajectory = generator.generate_trajectory(
                        intent=intent,
                        vehicle_type=vehicle_type,
                        min_distance=min_distance
                    )
                    
                    if trajectory is not None:
                        # 保存轨迹
                        traj_filename = f"traj_{trajectory_id:06d}_{config_key}.pkl"
                        traj_path = output_dir / traj_filename
                        generator.save_trajectory(trajectory, traj_path)
                        
                        # 更新统计
                        config_stats['generated'] += 1
                        config_stats['trajectories'].append({
                            'id': trajectory_id,
                            'filename': traj_filename,
                            'length_km': trajectory['length'] / 1000,
                            'duration_min': trajectory['duration'] / 60,
                            'num_points': trajectory['num_points']
                        })
                        
                        stats['generated'] += 1
                        trajectory_id += 1
                        
                        logger.info(f"✓ 轨迹 {trajectory_id} 已保存")
                    else:
                        config_stats['failed'] += 1
                        stats['failed'] += 1
                        logger.warning(f"✗ 轨迹生成失败")
                
                except Exception as e:
                    logger.error(f"✗ 轨迹生成异常: {e}")
                    config_stats['failed'] += 1
                    stats['failed'] += 1
            
            stats['configs'][config_key] = config_stats
            
            logger.info(f"\n配置 {config_key} 完成:")
            logger.info(f"  成功: {config_stats['generated']}")
            logger.info(f"  失败: {config_stats['failed']}")
    
    # 保存统计信息
    stats['end_time'] = datetime.now().isoformat()
    stats_path = output_dir / 'dataset_stats.json'
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("数据集生成完成")
    logger.info("=" * 60)
    logger.info(f"总轨迹数: {stats['generated']}")
    logger.info(f"失败数: {stats['failed']}")
    logger.info(f"成功率: {stats['generated']/total_trajectories*100:.1f}%")
    logger.info(f"统计信息已保存: {stats_path}")
    logger.info("")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='生成合成轨迹数据集')
    parser.add_argument('--region', type=str, default='scottish_highlands',
                        help='区域名称')
    parser.add_argument('--num-per-config', type=int, default=10,
                        help='每种配置生成的轨迹数')
    parser.add_argument('--min-distance', type=float, default=80.0,
                        help='最小直线距离（km）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 设置日志
    log_dir = Path(get_path('paths.outputs.logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"日志文件: {log_file}")
    
    # 生成数据集
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    stats = generate_dataset(
        region=args.region,
        num_trajectories_per_config=args.num_per_config,
        min_distance=args.min_distance * 1000,  # km to m
        output_dir=output_dir
    )
    
    print(f"\n✅ 数据集生成完成")
    print(f"成功: {stats['generated']}/{stats['total_trajectories']}")
    print(f"输出目录: {output_dir or Path(get_path('paths.processed.synthetic_trajectories')) / args.region}")


if __name__ == '__main__':
    main()
