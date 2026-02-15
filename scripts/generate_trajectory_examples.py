#!/usr/bin/env python3
"""Generate trajectory prediction example figures for the paper.
Shows history, GT future, and predictions from multiple models on selected samples.
"""
import sys, json, pickle, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

FIGS = ROOT / 'outputs' / 'paper_figures'
FIGS.mkdir(parents=True, exist_ok=True)


def load_sample(traj_dir, file_name, sample_idx, coord_scale=1.0, env_coverage_km=140.0):
    """Load a single sample from the dataset."""
    fpath = Path(traj_dir) / file_name
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    sample = data['samples'][sample_idx]

    from scripts.evaluate_phases_v2 import (
        PhaseV2Dataset, generate_prior_heatmap_phase1,
        sample_candidates_from_heatmap, extract_goal_from_heatmap,
    )

    env_map_np = PhaseV2Dataset.normalize_env_map(
        np.asarray(sample['env_map_100km'], dtype=np.float32))
    history_feat = torch.FloatTensor(sample['history_feat_26d'])
    history = history_feat.clone()
    history[:, 0:2] *= coord_scale

    future_rel = torch.FloatTensor(sample['future_rel'])
    future_delta = torch.diff(future_rel, dim=0, prepend=torch.zeros(1, 2))
    future = future_delta * coord_scale

    goal_rel_km = np.asarray(sample['goal_rel'], dtype=np.float64)
    goal = torch.as_tensor(goal_rel_km * coord_scale, dtype=torch.float32)

    # Phase 1a heatmap (precise goal)
    heatmap = generate_prior_heatmap_phase1(goal_rel_km, env_coverage_km, sigma_km=1.0)
    env_map_np[-1] = heatmap
    env_map = torch.from_numpy(env_map_np).float()

    rng = np.random.default_rng(42)
    candidates_km, target_idx = sample_candidates_from_heatmap(
        heatmap, env_map_np, env_coverage_km, num_candidates=6,
        include_gt=True, goal_rel_km=goal_rel_km, rng=rng)
    candidates = torch.as_tensor(candidates_km, dtype=torch.float32) * coord_scale

    heatmap_goal_km = extract_goal_from_heatmap(heatmap, env_coverage_km)
    heatmap_goal = torch.as_tensor(heatmap_goal_km, dtype=torch.float32) * coord_scale

    return {
        'history': history, 'future': future, 'env_map': env_map,
        'candidates': candidates, 'target_goal_idx': target_idx,
        'goal': goal, 'heatmap_goal': heatmap_goal,
        'history_xy_km': history_feat[:, :2].numpy(),
        'future_rel_km': np.asarray(sample['future_rel'], dtype=np.float64),
        'goal_rel_km': goal_rel_km,
    }


def predict_all_models(sample, models_dict, device):
    """Run inference for all models on a single sample."""
    preds = {}
    history = sample['history'].unsqueeze(0).to(device)
    env_map = sample['env_map'].unsqueeze(0).to(device)
    candidates = sample['candidates'].unsqueeze(0).to(device)
    target_idx = torch.tensor([sample['target_goal_idx']]).to(device)
    heatmap_goal = sample['heatmap_goal'].unsqueeze(0).to(device)
    B = 1
    history_xy = history[:, :, :2]

    with torch.no_grad():
        for name, info in models_dict.items():
            model = info['model']
            mtype = info['type']
            try:
                with autocast('cuda', enabled=True):
                    if mtype == 'baseline_with_goal':
                        out = model(history_xy, env_map, goal=heatmap_goal)
                        pred_pos = out[0] if isinstance(out, tuple) else out
                    elif mtype == 'baseline_no_goal':
                        out = model(history_xy, env_map)
                        pred_pos = out[0] if isinstance(out, tuple) else out
                    elif mtype == 'terratnt':
                        current_pos = torch.zeros(B, 2, device=device)
                        pred_delta, _ = model(env_map, history, candidates, current_pos,
                                              teacher_forcing_ratio=0.0, target_goal_idx=target_idx,
                                              use_gt_goal=False)
                        pred_pos = torch.cumsum(pred_delta, dim=1)
                    elif mtype == 'terratnt_v6':
                        current_pos = torch.zeros(B, 2, device=device)
                        pred_pos, _ = model(env_map, history, candidates, current_pos,
                                            teacher_forcing_ratio=0.0, target_goal_idx=target_idx,
                                            use_gt_goal=False)
                    elif mtype == 'terratnt_v7':
                        current_pos = torch.zeros(B, 2, device=device)
                        pred_pos, _, alpha = model(env_map, history, candidates, current_pos,
                                                   teacher_forcing_ratio=0.0, target_goal_idx=target_idx,
                                                   use_gt_goal=False)
                    elif mtype == 'incremental':
                        out = model(history_xy, env_map, goal=heatmap_goal)
                        pred_pos = out[0] if isinstance(out, tuple) else out
                    elif mtype == 'faithful_with_goal':
                        out = model(history_xy, env_map, goal=heatmap_goal)
                        pred_pos = out[0] if isinstance(out, tuple) else out
                    else:
                        continue
                preds[name] = pred_pos[0].cpu().numpy()  # (T, 2) km
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
    return preds


def plot_example(sample, preds, idx, title_suffix=''):
    """Plot a single trajectory example with multiple model predictions."""
    history_xy = sample['history_xy_km']  # (90, 2) km
    gt_future = sample['future_rel_km']   # (360, 2) km cumulative
    goal = sample['goal_rel_km']          # (2,) km

    # Models to show (ordered by importance)
    show_models = [
        ('V6R_Robust', 'TerraTNT(本文)', '#1565C0', 2.5, '-'),
        ('YNet', 'YNet', '#E91E63', 2.0, '--'),
        ('PECNet', 'PECNet', '#9C27B0', 2.0, '-.'),
        ('LSTM_only', 'SimpleLSTM', '#FF9800', 1.8, ':'),
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    # DEM background from env_map channel 0
    if 'env_map' in sample:
        env_np = sample['env_map'].numpy() if hasattr(sample['env_map'], 'numpy') else sample['env_map']
        dem = env_np[0]  # ch0 = DEM
        half_km = 70.0  # 140km / 2
        extent = [-half_km, half_km, -half_km, half_km]
        vmin, vmax = np.nanpercentile(dem, [2, 98]) if np.any(np.isfinite(dem)) else (0, 1)
        ax.imshow(dem, extent=extent, origin='upper', cmap='terrain',
                  alpha=0.3, vmin=vmin, vmax=vmax, zorder=0)

    # History
    ax.plot(history_xy[:, 0], history_xy[:, 1], 'o-', color='#333333',
            linewidth=2, markersize=2, label='History (15 min)', alpha=0.7, zorder=5)

    # GT future
    ax.plot(gt_future[:, 0], gt_future[:, 1], '-', color='#2E7D32',
            linewidth=2.5, label='Ground Truth', alpha=0.9, zorder=4)

    # Predictions
    for key, label, color, lw, ls in show_models:
        if key in preds:
            pred = preds[key]
            ax.plot(pred[:, 0], pred[:, 1], ls, color=color, linewidth=lw,
                    label=f'{label}', alpha=0.85, zorder=3)

    # Markers
    ax.plot(0, 0, 'k*', markersize=15, zorder=10, label='Current Position')
    ax.plot(goal[0], goal[1], 'r*', markersize=15, zorder=10, label='Goal')

    # Time markers on GT (every 2 hours = 120 steps)
    for t_step, t_label in [(120, '2h'), (240, '4h'), (359, '6h')]:
        if t_step < len(gt_future):
            ax.plot(gt_future[t_step, 0], gt_future[t_step, 1], 'o',
                    color='#2E7D32', markersize=6, zorder=6)
            ax.annotate(t_label, (gt_future[t_step, 0], gt_future[t_step, 1]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8, color='#2E7D32')

    ax.set_xlabel('East (km)', fontsize=11)
    ax.set_ylabel('North (km)', fontsize=11)
    ax.set_title(f'Trajectory Prediction Example {idx+1}{title_suffix}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.2)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGS / f'fig_example_{idx+1}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / f'fig_example_{idx+1}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Example {idx+1} saved')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traj_dir = ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo'
    split_file = traj_dir / 'fas_splits_full_phases.json'

    with open(split_file) as f:
        splits = json.load(f)

    # Pick diverse samples from fas1 val set
    val_samples = splits['fas1']['val_samples']
    # Select a few representative ones (spread across the dataset)
    np.random.seed(42)
    indices = np.random.choice(len(val_samples), size=min(6, len(val_samples)), replace=False)
    selected = [val_samples[i] for i in sorted(indices)]

    print(f'Loading models...')
    from scripts.evaluate_phases_v2 import load_all_models
    models_dict = load_all_models(device)
    print(f'Loaded {len(models_dict)} models')

    for idx, item in enumerate(selected):
        print(f'\nProcessing sample {idx+1}/{len(selected)}: {item["file"]} #{item["sample_idx"]}')
        sample = load_sample(traj_dir, item['file'], item['sample_idx'])
        preds = predict_all_models(sample, models_dict, device)

        # Compute ADE for each model
        gt_pos = sample['future_rel_km']
        for name, pred in preds.items():
            ade = np.mean(np.linalg.norm(pred - gt_pos, axis=-1)) * 1000
            print(f'  {name}: ADE={ade:.0f}m')

        plot_example(sample, preds, idx)

    print(f'\nAll examples saved to: {FIGS}')


if __name__ == '__main__':
    main()
