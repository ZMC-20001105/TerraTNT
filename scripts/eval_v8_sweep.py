#!/usr/bin/env python3
"""Evaluate V6 with different goal scaling α values across all phases.
This tests the hypothesis: can we improve Phase3-4 by reducing goal influence?"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_terratnt_10s import FASDataset, HISTORY_LEN, FUTURE_LEN
from scripts.train_incremental_models import AdaptiveGoalV8, ade_fde_m


def eval_phase(model, phase_name, traj_dir, split_file, device, alpha_values):
    """Evaluate model on a phase with different forced alpha values."""
    ds_kw = dict(history_len=HISTORY_LEN, future_len=FUTURE_LEN, num_candidates=6,
                 region='bohemian_forest', env_coverage_km=140.0, coord_scale=1.0)

    phase3_missing = (phase_name in ('fas3', 'fas3b_gaussian', 'fas4'))
    ds = FASDataset(traj_dir, split_file, phase=phase_name if phase_name in ('fas1','fas2','fas3') else 'fas1',
                    phase3_missing_goal=phase3_missing, **ds_kw)

    with open(split_file) as f:
        sp = json.load(f)
    # Use val samples
    phase_key = phase_name if phase_name in sp else 'fas1'
    val_items = sp[phase_key].get('val_samples', sp[phase_key].get('train_samples', []))
    ds.samples_meta = [(str(Path(traj_dir) / i['file']), int(i['sample_idx']))
                        for i in val_items]

    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

    results = {}
    for alpha in alpha_values:
        model.eval()
        ade_sum, fde_sum, n = 0, 0, 0
        with torch.no_grad():
            for b in dl:
                h = b['history'].to(device); fd = b['future'].to(device)
                em = b['env_map'].to(device); c = b['candidates'].to(device)
                gt = torch.cumsum(fd, dim=1)
                cp = torch.zeros(h.size(0), 2, device=device)
                with autocast('cuda', enabled=True):
                    pred, gl, _ = model(em, h, c, cp, use_gt_goal=False,
                                        force_alpha=alpha)
                ad, fd2 = ade_fde_m(pred, gt)
                ade_sum += ad.sum().item(); fde_sum += fd2.sum().item()
                n += h.size(0)
        results[alpha] = {'ade': ade_sum / max(1, n), 'fde': fde_sum / max(1, n), 'n': n}
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traj_dir = 'outputs/dataset_experiments/D1_optimal_combo'
    split_file = 'outputs/dataset_experiments/D1_optimal_combo/fas_splits_trajlevel.json'

    # Load V6 weights into AdaptiveGoalV8 (which is V6 + goal_scale_net)
    model = AdaptiveGoalV8(history_dim=26, hidden_dim=128, env_channels=18,
                           env_feature_dim=128, decoder_hidden_dim=256,
                           future_len=FUTURE_LEN, num_waypoints=10,
                           num_candidates=6, env_coverage_km=140.0).to(device)
    ck = torch.load('runs/incremental_models/V6_best.pth', map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'], strict=False)
    print(f'Loaded V6 weights', flush=True)

    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    phases = ['fas1', 'fas3']

    print(f'\n{"α":>5}', end='')
    for ph in phases:
        print(f'  {ph:>10}', end='')
    print()
    print('-' * 30)

    all_results = {}
    for ph in phases:
        print(f'Evaluating {ph}...', flush=True)
        all_results[ph] = eval_phase(model, ph, traj_dir, split_file, device, alpha_values)

    # Print table
    print(f'\n{"α":>5}', end='')
    for ph in phases:
        print(f'  {ph+" ADE":>10}', end='')
    print()
    print('-' * 30)
    for alpha in alpha_values:
        print(f'{alpha:>5.1f}', end='')
        for ph in phases:
            ade = all_results[ph][alpha]['ade']
            print(f'  {ade:>10.0f}', end='')
        print()

    # Save
    out = {ph: {str(a): r for a, r in all_results[ph].items()} for ph in phases}
    with open('runs/incremental_models_v8/alpha_sweep.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to runs/incremental_models_v8/alpha_sweep.json')


if __name__ == '__main__':
    main()
