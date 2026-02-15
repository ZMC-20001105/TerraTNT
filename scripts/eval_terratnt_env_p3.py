#!/usr/bin/env python3
"""在P3a/P3b上评估TerraTNT-Env，与所有无目标基线公平对比"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.evaluate_phases_v2 import (
    PhaseV2Dataset, PHASE_V2_CONFIGS, compute_metrics
)
from scripts.train_terratnt_env import TerraTNTEnv
from scripts.train_eval_all_baselines import (
    LSTMOnly, Seq2SeqAttention, MLPBaseline
)

FUTURE_LEN = 360
HISTORY_LEN = 90


def load_models(device):
    runs = PROJECT_ROOT / 'runs'
    models = {}

    # TerraTNT-Env (our new model)
    ckpt_path = runs / 'terratnt_env_full' / 'best_model.pth'
    if ckpt_path.exists():
        m = TerraTNTEnv(future_len=FUTURE_LEN, env_coverage_km=140.0).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models['TerraTNT_Env'] = {'model': m, 'type': 'terratnt_env'}
        print(f"  [OK] TerraTNT_Env (val_ade={ckpt.get('val_ade', '?'):.0f}m)")

    # Seq2Seq_Attn
    ckpt_path = runs / 'Seq2Seq_Attn_d1' / 'best_model.pth'
    if ckpt_path.exists():
        m = Seq2SeqAttention(hidden_dim=256, future_len=FUTURE_LEN).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        m.eval()
        models['Seq2Seq_Attn'] = {'model': m, 'type': 'baseline_no_goal'}
        print(f"  [OK] Seq2Seq_Attn")

    # LSTM_only
    ckpt_path = runs / 'LSTM_only_d1' / 'best_model.pth'
    if ckpt_path.exists():
        m = LSTMOnly(hidden_dim=256, future_len=FUTURE_LEN).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        m.eval()
        models['LSTM_only'] = {'model': m, 'type': 'baseline_no_goal'}
        print(f"  [OK] LSTM_only")

    # MLP
    ckpt_path = runs / 'MLP_d1' / 'best_model.pth'
    if ckpt_path.exists():
        m = MLPBaseline(hidden_dim=512, future_len=FUTURE_LEN, history_len=HISTORY_LEN).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        m.eval()
        models['MLP'] = {'model': m, 'type': 'baseline_no_goal'}
        print(f"  [OK] MLP")

    return models


@torch.no_grad()
def evaluate(models_dict, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_metrics = {name: [] for name in models_dict}
    all_metrics['ConstantVelocity'] = []

    for batch in tqdm(loader, desc="评估中", leave=False):
        history = batch['history'].to(device)
        future = batch['future'].to(device)
        env_map = batch['env_map'].to(device)
        B = history.size(0)
        history_xy = history[:, :, :2]
        gt_pos = torch.cumsum(future, dim=1)
        T = gt_pos.size(1)

        # Constant Velocity
        vel = history_xy[:, -1, :] - history_xy[:, -2, :]
        steps = torch.arange(1, T+1, device=device).float().unsqueeze(0).unsqueeze(-1)
        cv_pos = vel.unsqueeze(1) * steps
        for b in range(B):
            all_metrics['ConstantVelocity'].append(
                compute_metrics(cv_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy()))

        for name, info in models_dict.items():
            model = info['model']
            mtype = info['type']
            try:
                with autocast('cuda', enabled=True):
                    if mtype == 'terratnt_env':
                        pred_pos = model(env_map, history)
                    elif mtype == 'baseline_no_goal':
                        out = model(history_xy, env_map)
                        pred_pos = out[0] if isinstance(out, tuple) else out
                    else:
                        continue
                for b in range(B):
                    all_metrics[name].append(
                        compute_metrics(pred_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy()))
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                for b in range(B):
                    all_metrics[name].append({'ade': float('nan'), 'fde': float('nan'),
                        'early_ade': float('nan'), 'mid_ade': float('nan'), 'late_ade': float('nan')})
    return all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phases', nargs='+', default=['P3a', 'P3b', 'P1a'])
    p.add_argument('--traj_dir', default=str(PROJECT_ROOT / 'outputs/dataset_experiments/D1_optimal_combo'))
    p.add_argument('--fas_split_file', default=str(PROJECT_ROOT / 'outputs/dataset_experiments/D1_optimal_combo/fas_splits_full_phases.json'))
    p.add_argument('--output', default=str(PROJECT_ROOT / 'outputs/evaluation/terratnt_env_results.json'))
    p.add_argument('--batch_size', type=int, default=16)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    print("\n加载模型...")
    models = load_models(device)
    print(f"已加载 {len(models)} 个模型")

    results = {}
    for phase_id in args.phases:
        cfg = PHASE_V2_CONFIGS.get(phase_id)
        if not cfg:
            continue
        print(f"\n{'='*60}")
        print(f"评估 {cfg['name']}")
        ds = PhaseV2Dataset(args.traj_dir, args.fas_split_file, cfg)
        if len(ds) == 0:
            continue
        metrics = evaluate(models, ds, device, args.batch_size)

        phase_summary = {}
        for name, mlist in metrics.items():
            if not mlist:
                continue
            ades = [m['ade'] for m in mlist if not np.isnan(m['ade'])]
            fdes = [m['fde'] for m in mlist if not np.isnan(m['fde'])]
            if not ades:
                continue
            phase_summary[name] = {
                'ade_mean': float(np.mean(ades)),
                'ade_std': float(np.std(ades)),
                'fde_mean': float(np.mean(fdes)),
                'fde_std': float(np.std(fdes)),
                'n_samples': len(ades),
                'early_ade': float(np.mean([m['early_ade'] for m in mlist if not np.isnan(m['early_ade'])])),
                'mid_ade': float(np.mean([m['mid_ade'] for m in mlist if not np.isnan(m['mid_ade'])])),
                'late_ade': float(np.mean([m['late_ade'] for m in mlist if not np.isnan(m['late_ade'])])),
            }
        results[phase_id] = {'name': cfg['name'], 'models': phase_summary}

        # 打印排名
        ranked = sorted(phase_summary.items(), key=lambda x: x[1]['ade_mean'])
        print(f"\n  {'排名':4s} {'模型':25s} {'ADE(km)':>10s} {'FDE(km)':>10s} {'Early':>8s} {'Mid':>8s} {'Late':>8s}")
        print(f"  {'-'*75}")
        for rank, (name, ms) in enumerate(ranked, 1):
            ade = ms['ade_mean']/1000; fde = ms['fde_mean']/1000
            e = ms['early_ade']/1000; m = ms['mid_ade']/1000; l = ms['late_ade']/1000
            marker = ' ★' if name == 'TerraTNT_Env' else ''
            print(f"  #{rank:<3d} {name:25s} {ade:10.2f} {fde:10.2f} {e:8.2f} {m:8.2f} {l:8.2f}{marker}")

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
