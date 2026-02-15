#!/usr/bin/env python3
"""
为指定区域批量生成轨迹数据集 (complete_dataset_10s 格式)

用法:
  python scripts/generate_region_trajectories.py --region donbas --num_trajectories 200
  python scripts/generate_region_trajectories.py --region carpathians --num_trajectories 200
  python scripts/generate_region_trajectories.py --region donbas carpathians --num_trajectories 200
"""
import sys, os, argparse, pickle, time, json, gc
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.trajectory_generation.trajectory_generator_v2 import TrajectoryGeneratorV2

INTENTS = ['intent1', 'intent2', 'intent3']
VEHICLE_TYPES = ['type1', 'type2', 'type3', 'type4']


def generate_trajectories_for_region(
    region: str,
    num_trajectories: int = 200,
    output_dir: Path = None,
    min_distance: float = 60000.0,
    seed: int = 42,
):
    """为单个区域生成轨迹"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / region
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查已有轨迹
    existing = sorted(output_dir.glob('traj_*.pkl'))
    start_id = len(existing)
    if start_id > 0:
        print(f"  已有 {start_id} 条轨迹，从 ID={start_id} 继续")

    print(f"\n{'='*60}")
    print(f"区域: {region}")
    print(f"目标轨迹数: {num_trajectories}")
    print(f"输出目录: {output_dir}")
    print(f"最小距离: {min_distance/1000:.0f} km")
    print(f"{'='*60}\n")

    generator = TrajectoryGeneratorV2(region)

    # 均匀分配 intent × vehicle_type
    combos = [(intent, vtype) for intent in INTENTS for vtype in VEHICLE_TYPES]
    per_combo = max(1, num_trajectories // len(combos))
    remainder = num_trajectories - per_combo * len(combos)

    traj_id = start_id
    success_count = 0
    fail_count = 0
    stats = defaultdict(int)

    np.random.seed(seed + hash(region) % 10000)

    for ci, (intent, vtype) in enumerate(combos):
        n_target = per_combo + (1 if ci < remainder else 0)
        combo_success = 0
        combo_fail = 0
        max_attempts = n_target * 3  # 允许3倍尝试

        print(f"\n--- {intent} × {vtype}: 目标 {n_target} 条 ---")

        for attempt in range(max_attempts):
            if combo_success >= n_target:
                break

            np.random.seed(seed + traj_id * 1000 + attempt)

            try:
                traj = generator.generate_complete_trajectory(
                    intent=intent,
                    vehicle_type=vtype,
                    min_distance=min_distance,
                    trajectory_id=traj_id,
                )
            except Exception as e:
                print(f"  [ERROR] traj_{traj_id:06d}: {e}")
                combo_fail += 1
                continue

            if traj is None:
                combo_fail += 1
                reason = 'unknown'
                if generator.last_failure:
                    reason = generator.last_failure.get('reason', 'unknown')
                stats[f'fail_{reason}'] += 1
                continue

            # 保存
            fname = f'traj_{traj_id:06d}_{intent}_{vtype}.pkl'
            with open(output_dir / fname, 'wb') as f:
                pickle.dump(traj, f)

            combo_success += 1
            success_count += 1
            traj_id += 1
            stats[f'ok_{intent}_{vtype}'] += 1

            n_samples = traj.get('num_samples', 0)
            length_km = traj.get('length', 0) / 1000
            duration_min = traj.get('duration', 0) / 60
            print(f"  ✓ {fname}: {length_km:.1f}km, {duration_min:.0f}min, {n_samples} samples")

            # 显式释放轨迹数据，防止内存累积
            del traj
            gc.collect()

        fail_count += combo_fail
        print(f"  {intent}×{vtype}: {combo_success}/{n_target} 成功, {combo_fail} 失败")

    # 汇总
    print(f"\n{'='*60}")
    print(f"区域 {region} 轨迹生成完成")
    print(f"{'='*60}")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"总轨迹数: {traj_id}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    return success_count


def create_fas_splits(region: str, traj_dir: Path = None):
    """为区域创建FAS splits"""
    if traj_dir is None:
        traj_dir = PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / region
    output_dir = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n创建 {region} FAS splits...")

    from scripts.prepare_fas_datasets import prepare_fas_splits_samples
    splits = prepare_fas_splits_samples(
        traj_dir=traj_dir,
        output_dir=output_dir,
        distance_threshold=1000.0,
        train_ratio=0.7,
    )

    # 也创建 trajectory-level split
    from scripts.generate_traj_level_split import main as gen_traj_split
    sys.argv = ['', '--region', region, '--traj_dir', str(traj_dir)]
    try:
        gen_traj_split()
    except SystemExit:
        pass
    except Exception as e:
        print(f"  trajectory-level split 失败: {e}")

    return splits


def main():
    parser = argparse.ArgumentParser(description='为区域生成轨迹数据集')
    parser.add_argument('--region', nargs='+', default=['donbas', 'carpathians'])
    parser.add_argument('--num_trajectories', type=int, default=200,
                        help='每个区域的目标轨迹数')
    parser.add_argument('--min_distance', type=float, default=60000.0,
                        help='最小直线距离(m)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_splits', action='store_true')
    args = parser.parse_args()

    for region in args.region:
        t0 = time.time()
        n = generate_trajectories_for_region(
            region=region,
            num_trajectories=args.num_trajectories,
            min_distance=args.min_distance,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        print(f"\n{region}: {n} 条轨迹, 耗时 {elapsed/60:.1f} 分钟")

        if not args.skip_splits and n > 0:
            create_fas_splits(region)

    print("\n全部完成!")


if __name__ == '__main__':
    main()
