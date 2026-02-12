#!/usr/bin/env python3
"""
Generate trajectory-level train/val split to prevent data leakage.

Problem: Current random_split operates at sample level, so samples from the
same trajectory (sliding window) can appear in both train and val sets.
This causes ~100% trajectory overlap and inflated validation metrics.

Solution: Split at trajectory (pkl file) level, ensuring no trajectory
appears in both train and val. Also supports curvature-aware oversampling.

Output: A new split JSON with explicit (file, sample_idx) lists for train/val.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import glob
import os
import argparse
import numpy as np
from collections import defaultdict


def compute_trajectory_curvature(samples):
    """Compute mean normalized deviation across all samples in a trajectory."""
    nds = []
    for s in samples:
        fr = np.array(s['future_rel'])
        start = fr[0]
        end = fr[-1]
        dist = np.linalg.norm(end - start)
        if dist < 1e-6:
            nds.append(0.0)
            continue
        straight = np.linspace(start, end, len(fr))
        dev = np.linalg.norm(fr - straight, axis=1)
        nd = np.mean(dev) / dist
        nds.append(nd)
    return np.mean(nds) if nds else 0.0


def main():
    parser = argparse.ArgumentParser(description='Generate trajectory-level train/val split')
    parser.add_argument('--traj_dir', type=str,
                        default='data/processed/final_dataset_v1/bohemian_forest')
    parser.add_argument('--fas_split_file', type=str,
                        default='data/processed/fas_splits/bohemian_forest/fas_splits.json')
    parser.add_argument('--output', type=str,
                        default='data/processed/fas_splits/bohemian_forest/fas_splits_trajlevel.json')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of trajectories for validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--oversample_curved', action='store_true', default=True,
                        help='Oversample high-curvature samples in training set')
    parser.add_argument('--curve_threshold', type=float, default=0.10,
                        help='Normalized deviation threshold for "curved" trajectories')
    parser.add_argument('--oversample_factor', type=int, default=3,
                        help='How many times to repeat curved samples')
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    with open(args.fas_split_file) as f:
        original_splits = json.load(f)

    rng = np.random.default_rng(args.seed)

    for phase in ['fas1', 'fas2', 'fas3']:
        phase_spec = original_splits.get(phase, {})
        file_list = phase_spec.get('files', [])
        if not file_list:
            continue

        print(f'\n{"="*60}')
        print(f'Processing {phase.upper()}: {len(file_list)} trajectory files')

        # Gather per-trajectory info
        traj_info = []
        for fname in file_list:
            fpath = traj_dir / fname
            if not fpath.exists():
                continue
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            samples = data.get('samples', [])
            n_samples = len(samples)
            if n_samples == 0:
                continue
            curvature = compute_trajectory_curvature(samples)
            traj_info.append({
                'file': fname,
                'n_samples': n_samples,
                'curvature': curvature,
            })

        n_traj = len(traj_info)
        total_samples = sum(t['n_samples'] for t in traj_info)
        print(f'  Valid trajectories: {n_traj}, total samples: {total_samples}')

        # Shuffle and split at trajectory level
        indices = rng.permutation(n_traj)
        n_val_traj = max(1, int(n_traj * args.val_ratio))
        val_indices = set(indices[:n_val_traj])
        train_indices = set(indices[n_val_traj:])

        # Build sample lists
        train_samples = []
        val_samples = []
        train_curved_samples = []

        for i, ti in enumerate(traj_info):
            for si in range(ti['n_samples']):
                entry = {'file': ti['file'], 'sample_idx': si}
                if i in val_indices:
                    val_samples.append(entry)
                else:
                    train_samples.append(entry)
                    if ti['curvature'] >= args.curve_threshold:
                        train_curved_samples.append(entry)

        # Oversample curved trajectories in training
        if args.oversample_curved and train_curved_samples:
            extra = train_curved_samples * (args.oversample_factor - 1)
            train_samples.extend(extra)
            print(f'  Oversampled {len(train_curved_samples)} curved samples '
                  f'x{args.oversample_factor} -> +{len(extra)} extra')

        n_train_traj = len(train_indices)
        n_val_traj_actual = len(val_indices)

        # Verify no overlap
        train_files = set(s['file'] for s in train_samples)
        val_files = set(s['file'] for s in val_samples)
        overlap = train_files & val_files
        assert len(overlap) == 0, f'Trajectory overlap detected: {len(overlap)} files!'

        # Curvature stats
        train_curvatures = [traj_info[i]['curvature'] for i in train_indices]
        val_curvatures = [traj_info[i]['curvature'] for i in val_indices]

        print(f'  Train: {n_train_traj} trajs, {len(train_samples)} samples')
        print(f'  Val:   {n_val_traj_actual} trajs, {len(val_samples)} samples')
        print(f'  Train curvature: mean={np.mean(train_curvatures):.4f}, '
              f'median={np.median(train_curvatures):.4f}')
        print(f'  Val curvature:   mean={np.mean(val_curvatures):.4f}, '
              f'median={np.median(val_curvatures):.4f}')
        print(f'  Trajectory overlap: {len(overlap)} (should be 0)')

        # Update split
        original_splits[phase]['train_samples'] = train_samples
        original_splits[phase]['val_samples'] = val_samples
        original_splits[phase]['split_info'] = {
            'type': 'trajectory_level',
            'n_train_traj': n_train_traj,
            'n_val_traj': n_val_traj_actual,
            'n_train_samples': len(train_samples),
            'n_val_samples': len(val_samples),
            'val_ratio': args.val_ratio,
            'seed': args.seed,
            'oversample_curved': args.oversample_curved,
            'curve_threshold': args.curve_threshold,
            'oversample_factor': args.oversample_factor,
        }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(original_splits, f, indent=2, ensure_ascii=False)
    print(f'\nSaved trajectory-level split to: {output_path}')


if __name__ == '__main__':
    main()
