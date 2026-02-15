#!/usr/bin/env python3
"""Evaluate confidence-based routing between V6R and V3.
When V6R's classifier is confident → use V6R prediction
When V6R's classifier is uncertain → use V3 prediction (goal-independent)

This tests the hypothesis that we can get best-of-both-worlds by
routing based on classifier confidence.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_terratnt_10s import FASDataset, HISTORY_LEN, FUTURE_LEN
from scripts.train_incremental_models import (
    TerraTNTAutoregV6, LSTMEnvGoalWaypoint, ade_fde_m
)


def load_models(device):
    """Load V6R and V3 models."""
    models = {}

    # V6R (robust)
    m_v6r = TerraTNTAutoregV6(
        history_dim=26, hidden_dim=128, env_channels=18, env_feature_dim=128,
        decoder_hidden_dim=256, future_len=FUTURE_LEN, num_waypoints=10,
        num_candidates=6, env_coverage_km=140.0).to(device)
    ck = torch.load('runs/incremental_models_v6r/V6_best.pth', map_location=device, weights_only=False)
    m_v6r.load_state_dict(ck['model_state_dict'], strict=False)
    m_v6r.eval()
    models['V6R'] = m_v6r
    print(f'V6R loaded (ADE={ck.get("val_ade",0):.0f}m)')

    # V6 (phase1-optimized)
    m_v6 = TerraTNTAutoregV6(
        history_dim=26, hidden_dim=128, env_channels=18, env_feature_dim=128,
        decoder_hidden_dim=256, future_len=FUTURE_LEN, num_waypoints=10,
        num_candidates=6, env_coverage_km=140.0).to(device)
    ck = torch.load('runs/incremental_models/V6_best.pth', map_location=device, weights_only=False)
    m_v6.load_state_dict(ck['model_state_dict'], strict=False)
    m_v6.eval()
    models['V6'] = m_v6
    print(f'V6 loaded (ADE={ck.get("val_ade",0):.0f}m)')

    # V3 (goal-independent waypoint model)
    m_v3 = LSTMEnvGoalWaypoint(hidden_dim=256, future_len=FUTURE_LEN, num_waypoints=10).to(device)
    ck = torch.load('runs/incremental_models/V3_best.pth', map_location=device, weights_only=False)
    m_v3.load_state_dict(ck['model_state_dict'], strict=False)
    m_v3.eval()
    models['V3'] = m_v3
    print(f'V3 loaded (ADE={ck.get("val_ade",0):.0f}m)')

    return models


def eval_routing(models, phase_name, traj_dir, split_file, device, thresholds):
    """For each sample, compute V6R confidence and route to V6R or V3."""
    ds_kw = dict(history_len=HISTORY_LEN, future_len=FUTURE_LEN, num_candidates=6,
                 region='bohemian_forest', env_coverage_km=140.0, coord_scale=1.0)

    phase3_missing = (phase_name in ('fas3', 'fas3b_gaussian', 'fas4'))
    phase_key = phase_name if phase_name in ('fas1', 'fas2', 'fas3') else 'fas1'
    ds = FASDataset(traj_dir, split_file, phase=phase_key,
                    phase3_missing_goal=phase3_missing, **ds_kw)

    with open(split_file) as f:
        sp = json.load(f)
    val_items = sp[phase_key].get('val_samples', [])
    ds.samples_meta = [(str(Path(traj_dir) / i['file']), int(i['sample_idx']))
                        for i in val_items]
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)

    # Collect per-sample predictions from both models + confidence
    all_v6r_ade = []
    all_v6_ade = []
    all_v3_ade = []
    all_conf = []  # max classifier probability

    with torch.no_grad():
        for b in dl:
            h = b['history'].to(device); fd = b['future'].to(device)
            em = b['env_map'].to(device); c = b['candidates'].to(device)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=device)
            B = h.size(0)

            # V6R prediction + confidence
            with autocast('cuda', enabled=True):
                pred_v6r, gl_v6r = models['V6R'](em, h, c, cp, use_gt_goal=False)
            conf = F.softmax(gl_v6r, dim=1).max(dim=1).values  # (B,)
            ade_v6r = (torch.norm(pred_v6r - gt, dim=-1) * 1000).mean(dim=1)  # (B,)

            # V6 prediction
            with autocast('cuda', enabled=True):
                pred_v6, _ = models['V6'](em, h, c, cp, use_gt_goal=False)
            ade_v6 = (torch.norm(pred_v6 - gt, dim=-1) * 1000).mean(dim=1)

            # V3 prediction (uses GT goal from last future point)
            goal = gt[:, -1, :]
            history_xy = h[:, :, :2]
            with autocast('cuda', enabled=True):
                pred_v3 = models['V3'](history_xy, em, goal=goal)
                if isinstance(pred_v3, tuple):
                    pred_v3 = pred_v3[0]
                    if pred_v3.dim() == 3 and pred_v3.size(-1) == 2:
                        pred_v3_pos = torch.cumsum(pred_v3, dim=1)
                    else:
                        pred_v3_pos = pred_v3
                else:
                    pred_v3_pos = pred_v3
            ade_v3 = (torch.norm(pred_v3_pos - gt, dim=-1) * 1000).mean(dim=1)

            all_v6r_ade.append(ade_v6r.cpu().numpy())
            all_v6_ade.append(ade_v6.cpu().numpy())
            all_v3_ade.append(ade_v3.cpu().numpy())
            all_conf.append(conf.cpu().numpy())

    all_v6r_ade = np.concatenate(all_v6r_ade)
    all_v6_ade = np.concatenate(all_v6_ade)
    all_v3_ade = np.concatenate(all_v3_ade)
    all_conf = np.concatenate(all_conf)

    results = {
        'V6R_only': float(all_v6r_ade.mean()),
        'V6_only': float(all_v6_ade.mean()),
        'V3_only': float(all_v3_ade.mean()),
    }

    # For each threshold: if conf >= threshold → use V6R, else → use V3
    for thr in thresholds:
        use_v6r = all_conf >= thr
        routed_ade = np.where(use_v6r, all_v6r_ade, all_v3_ade)
        pct_v6r = use_v6r.mean() * 100
        results[f'route_{thr:.2f}'] = {
            'ade': float(routed_ade.mean()),
            'pct_v6r': float(pct_v6r),
        }

    # Also test V6+V3 routing
    for thr in thresholds:
        use_v6 = all_conf >= thr
        routed_ade = np.where(use_v6, all_v6_ade, all_v3_ade)
        pct_v6 = use_v6.mean() * 100
        results[f'route_v6_{thr:.2f}'] = {
            'ade': float(routed_ade.mean()),
            'pct_v6': float(pct_v6),
        }

    # Oracle: always pick the better model per sample
    oracle_v6r_v3 = np.minimum(all_v6r_ade, all_v3_ade)
    results['oracle_v6r_v3'] = float(oracle_v6r_v3.mean())
    oracle_v6_v3 = np.minimum(all_v6_ade, all_v3_ade)
    results['oracle_v6_v3'] = float(oracle_v6_v3.mean())

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traj_dir = 'outputs/dataset_experiments/D1_optimal_combo'
    split_file = 'outputs/dataset_experiments/D1_optimal_combo/fas_splits_trajlevel.json'

    models = load_models(device)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    phases = ['fas1', 'fas3']

    for ph in phases:
        print(f'\n{"="*60}')
        print(f'Phase: {ph}')
        print(f'{"="*60}')
        r = eval_routing(models, ph, traj_dir, split_file, device, thresholds)

        print(f'  V6R only:  {r["V6R_only"]:.0f}m')
        print(f'  V6 only:   {r["V6_only"]:.0f}m')
        print(f'  V3 only:   {r["V3_only"]:.0f}m')
        print(f'  Oracle V6R+V3: {r["oracle_v6r_v3"]:.0f}m')
        print(f'  Oracle V6+V3:  {r["oracle_v6_v3"]:.0f}m')
        print()
        print(f'  V6R+V3 Routing (conf threshold):')
        for thr in thresholds:
            d = r[f'route_{thr:.2f}']
            print(f'    thr={thr:.2f}: ADE={d["ade"]:.0f}m ({d["pct_v6r"]:.0f}% V6R)')
        print()
        print(f'  V6+V3 Routing (conf threshold):')
        for thr in thresholds:
            d = r[f'route_v6_{thr:.2f}']
            print(f'    thr={thr:.2f}: ADE={d["ade"]:.0f}m ({d["pct_v6"]:.0f}% V6)')


if __name__ == '__main__':
    main()
