"""
多核并行生成合成轨迹数据集

使用multiprocessing充分利用CPU核心
"""
import logging
import sys
from pathlib import Path
import argparse
from datetime import datetime
import pickle
import json
from typing import Optional, Tuple
from multiprocessing import Pool, cpu_count
import traceback

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.trajectory_generation.trajectory_generator import TrajectoryGenerator
from utils.visualization.trajectory_plotter import TrajectoryPlotter
from config import cfg, get_path

logger = logging.getLogger(__name__)


def generate_single_trajectory(args: Tuple) -> Optional[dict]:
    """
    生成单条轨迹（用于多进程）
    
    Args:
        args: (trajectory_id, region, intent, vehicle_type, min_distance, output_dir)
    
    Returns:
        轨迹统计信息或None
    """
    trajectory_id, region, intent, vehicle_type, min_distance, output_dir = args
    
    try:
        # 每个进程创建自己的生成器
        generator = TrajectoryGenerator(region)
        
        # 生成轨迹
        trajectory = generator.generate_trajectory(
            intent=intent,
            vehicle_type=vehicle_type,
            min_distance=min_distance
        )
        
        if trajectory is None:
            return None
        
        # 保存轨迹
        config_key = f"{intent}_{vehicle_type}"
        traj_filename = f"traj_{trajectory_id:06d}_{config_key}.pkl"
        traj_path = output_dir / traj_filename
        
        with open(traj_path, 'wb') as f:
            pickle.dump(trajectory, f)
        
        # 返回统计信息
        return {
            'id': trajectory_id,
            'filename': traj_filename,
            'intent': intent,
            'vehicle_type': vehicle_type,
            'length_km': trajectory['length'] / 1000,
            'duration_min': trajectory['duration'] / 60,
            'num_points': trajectory['num_points'],
            'success': True
        }
        
    except Exception as e:
        logger.error(f"轨迹 {trajectory_id} 生成失败: {e}")
        logger.debug(traceback.format_exc())
        return {
            'id': trajectory_id,
            'intent': intent,
            'vehicle_type': vehicle_type,
            'success': False,
            'error': str(e)
        }


def generate_dataset_parallel(
    region: str,
    num_trajectories_per_config: int = 10,
    min_distance: float = 80000.0,
    output_dir: Optional[Path] = None,
    num_workers: Optional[int] = None
):
    """
    并行生成数据集
    
    Args:
        region: 区域名称
        num_trajectories_per_config: 每种配置生成的轨迹数
        min_distance: 最小直线距离（米）
        output_dir: 输出目录
        num_workers: 工作进程数（None=自动检测）
    """
    if output_dir is None:
        output_dir = Path(get_path('paths.processed.synthetic_trajectories')) / region
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定工作进程数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # 保留2个核心给系统
    
    logger.info("=" * 60)
    logger.info(f"并行生成合成轨迹数据集 - 区域: {region}")
    logger.info("=" * 60)
    logger.info(f"每种配置轨迹数: {num_trajectories_per_config}")
    logger.info(f"最小直线距离: {min_distance/1000:.1f} km")
    logger.info(f"工作进程数: {num_workers}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("")
    
    # 配置组合
    intents = ['intent1', 'intent2', 'intent3']
    vehicle_types = ['type1', 'type2', 'type3', 'type4']
    
    total_configs = len(intents) * len(vehicle_types)
    total_trajectories = total_configs * num_trajectories_per_config
    
    logger.info(f"配置组合数: {total_configs}")
    logger.info(f"目标轨迹总数: {total_trajectories}")
    logger.info("")
    
    # 准备任务列表
    tasks = []
    trajectory_id = 0
    
    for intent in intents:
        for vehicle_type in vehicle_types:
            for i in range(num_trajectories_per_config):
                tasks.append((
                    trajectory_id,
                    region,
                    intent,
                    vehicle_type,
                    min_distance,
                    output_dir
                ))
                trajectory_id += 1
    
    # 统计信息
    stats = {
        'region': region,
        'start_time': datetime.now().isoformat(),
        'num_trajectories_per_config': num_trajectories_per_config,
        'min_distance': min_distance,
        'total_configs': total_configs,
        'total_trajectories': total_trajectories,
        'num_workers': num_workers,
        'generated': 0,
        'failed': 0,
        'trajectories': []
    }
    
    # 并行生成
    logger.info(f"开始并行生成（{num_workers}个工作进程）...")
    start_time = datetime.now()
    
    with Pool(processes=num_workers) as pool:
        # 使用imap_unordered获取结果（更快）
        results = []
        for i, result in enumerate(pool.imap_unordered(generate_single_trajectory, tasks), 1):
            if result is not None:
                results.append(result)
                
                if result['success']:
                    stats['generated'] += 1
                    logger.info(f"[{i}/{total_trajectories}] ✓ 轨迹 {result['id']:06d} "
                              f"({result['intent']}, {result['vehicle_type']}): "
                              f"{result['length_km']:.1f}km, {result['duration_min']:.1f}min")
                else:
                    stats['failed'] += 1
                    logger.warning(f"[{i}/{total_trajectories}] ✗ 轨迹 {result['id']:06d} 失败")
            else:
                stats['failed'] += 1
                logger.warning(f"[{i}/{total_trajectories}] ✗ 轨迹生成失败（返回None）")
            
            # 每10条打印进度
            if i % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_trajectories - i) / rate if rate > 0 else 0
                logger.info(f"  进度: {i}/{total_trajectories} ({i/total_trajectories*100:.1f}%), "
                          f"速度: {rate:.2f}条/秒, 预计剩余: {eta/60:.1f}分钟")
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # 保存统计信息
    stats['end_time'] = end_time.isoformat()
    stats['elapsed_seconds'] = elapsed
    stats['trajectories'] = results
    
    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("数据集生成完成")
    logger.info("=" * 60)
    logger.info(f"总轨迹数: {stats['generated']}")
    logger.info(f"失败数: {stats['failed']}")
    logger.info(f"成功率: {stats['generated']/total_trajectories*100:.1f}%")
    logger.info(f"总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"平均速度: {stats['generated']/elapsed:.2f} 条/秒")
    logger.info(f"统计信息已保存: {stats_path}")
    logger.info("")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='并行生成合成轨迹数据集')
    parser.add_argument('--region', type=str, default='scottish_highlands',
                        help='区域名称')
    parser.add_argument('--num-per-config', type=int, default=10,
                        help='每种配置生成的轨迹数')
    parser.add_argument('--min-distance', type=float, default=80.0,
                        help='最小直线距离（km）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--workers', type=int, default=None,
                        help='工作进程数（默认：CPU核心数-2）')
    
    args = parser.parse_args()
    
    # 设置日志
    log_dir = Path(get_path('paths.outputs.logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f'dataset_parallel_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"日志文件: {log_file}")
    logger.info(f"可用CPU核心数: {cpu_count()}")
    
    # 生成数据集
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    stats = generate_dataset_parallel(
        region=args.region,
        num_trajectories_per_config=args.num_per_config,
        min_distance=args.min_distance * 1000,
        output_dir=output_dir,
        num_workers=args.workers
    )
    
    print(f"\n✅ 数据集生成完成")
    print(f"成功: {stats['generated']}/{stats['total_trajectories']}")
    print(f"耗时: {stats['elapsed_seconds']/60:.1f} 分钟")
    print(f"输出目录: {output_dir or Path(get_path('paths.processed.synthetic_trajectories')) / args.region}")


if __name__ == '__main__':
    main()
