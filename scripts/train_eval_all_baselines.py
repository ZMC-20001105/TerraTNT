#!/usr/bin/env python3
"""
Unified training & evaluation script for ALL baseline models on D1 dataset.
Trains each model, evaluates on val set, and produces per-sample comparison.

Models:
  1. TerraTNT (ours) - load from checkpoint
  2. PECNet - load from checkpoint
  3. Trajectron++ - train from scratch
  4. Social-LSTM - train from scratch
  5. LSTM-only (no env) - train from scratch (ablation)
  6. Seq2Seq+Attention - train from scratch
  7. MLP baseline - train from scratch
  8. Constant Velocity - no training
  9. Straight Line to Goal - no training
"""
import sys, os, json, pickle, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from scripts.train_terratnt_10s import FASDataset, HISTORY_LEN, FUTURE_LEN
from models.baselines.social_lstm import SocialLSTM
from models.baselines.trajectron import TrajectronPP


# ============================================================
# Additional baseline models (implemented inline)
# ============================================================

class LSTMOnly(nn.Module):
    """Pure LSTM encoder-decoder, no environment map. Ablation baseline."""
    def __init__(self, input_dim=2, hidden_dim=256, future_len=360):
        super().__init__()
        self.future_len = future_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(self, history_xy, env_map=None, goal=None, **kwargs):
        _, (h, c) = self.encoder(history_xy)
        curr = history_xy[:, -1:, :]  # (B, 1, 2)
        preds = []
        for _ in range(self.future_len):
            out, (h, c) = self.decoder(curr, (h, c))
            delta = self.output_fc(out)  # (B, 1, 2)
            curr = curr + delta
            preds.append(curr)
        return torch.cat(preds, dim=1)  # (B, T, 2) cumulative positions


class LSTMEnvGoal(nn.Module):
    """LSTM with environment encoding and goal conditioning. Strong baseline."""
    def __init__(self, input_dim=2, hidden_dim=256, env_channels=18, env_dim=128, future_len=360):
        super().__init__()
        self.future_len = future_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        # Environment encoder (simple CNN)
        self.env_cnn = nn.Sequential(
            nn.Conv2d(env_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.env_fc = nn.Linear(128, env_dim)
        # Goal embedding
        self.goal_fc = nn.Linear(2, 64)
        # Fusion
        self.fusion = nn.Linear(hidden_dim + env_dim + 64, hidden_dim)
        # Decoder
        self.decoder = nn.LSTM(2 + hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(self, history_xy, env_map=None, goal=None, **kwargs):
        B = history_xy.size(0)
        _, (h, c) = self.encoder(history_xy)
        # Encode environment
        if env_map is not None:
            env_feat = self.env_cnn(env_map).view(B, -1)
            env_feat = self.env_fc(env_feat)
        else:
            env_feat = torch.zeros(B, 128, device=history_xy.device)
        # Encode goal
        if goal is not None:
            goal_feat = F.relu(self.goal_fc(goal))
        else:
            goal_feat = torch.zeros(B, 64, device=history_xy.device)
        # Fuse into decoder initial state
        context = F.relu(self.fusion(torch.cat([h[-1], env_feat, goal_feat], dim=1)))
        # Replace top layer hidden state
        h_new = h.clone()
        h_new[-1] = context
        curr = history_xy[:, -1:, :]
        preds = []
        for _ in range(self.future_len):
            dec_in = torch.cat([curr, context.unsqueeze(1).expand(-1, 1, -1)], dim=-1)
            out, (h_new, c) = self.decoder(dec_in, (h_new, c))
            delta = self.output_fc(out)
            curr = curr + delta
            preds.append(curr)
        return torch.cat(preds, dim=1)


class Seq2SeqAttention(nn.Module):
    """Seq2Seq with Bahdanau attention over history + env."""
    def __init__(self, input_dim=2, hidden_dim=256, env_channels=18, future_len=360):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        # Env encoder
        self.env_cnn = nn.Sequential(
            nn.Conv2d(env_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.env_fc = nn.Linear(128, hidden_dim)
        # Attention
        self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1)
        # Decoder
        self.decoder = nn.LSTM(2 + hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(self, history_xy, env_map=None, goal=None, **kwargs):
        B = history_xy.size(0)
        enc_out, (h, c) = self.encoder(history_xy)  # enc_out: (B, T_h, H)
        # Add env as extra "token"
        if env_map is not None:
            env_feat = self.env_cnn(env_map).view(B, -1)
            env_feat = self.env_fc(env_feat).unsqueeze(1)  # (B, 1, H)
            keys = torch.cat([enc_out, env_feat], dim=1)  # (B, T_h+1, H)
        else:
            keys = enc_out
        curr = history_xy[:, -1:, :]
        preds = []
        for _ in range(self.future_len):
            # Attention
            query = h[-1].unsqueeze(1).expand(-1, keys.size(1), -1)  # (B, T, H)
            energy = self.attn_v(torch.tanh(self.attn_W(torch.cat([query, keys], dim=-1))))  # (B, T, 1)
            alpha = F.softmax(energy, dim=1)  # (B, T, 1)
            context = (alpha * keys).sum(dim=1, keepdim=True)  # (B, 1, H)
            dec_in = torch.cat([curr, context], dim=-1)  # (B, 1, 2+H)
            out, (h, c) = self.decoder(dec_in, (h, c))
            delta = self.output_fc(out)
            curr = curr + delta
            preds.append(curr)
        return torch.cat(preds, dim=1)


class MLPBaseline(nn.Module):
    """Simple MLP: flatten history → MLP → future positions."""
    def __init__(self, history_len=90, future_len=360, hidden_dim=512):
        super().__init__()
        self.future_len = future_len
        self.net = nn.Sequential(
            nn.Linear(history_len * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, future_len * 2),
        )

    def forward(self, history_xy, env_map=None, goal=None, **kwargs):
        B = history_xy.size(0)
        x = history_xy[:, :, :2].reshape(B, -1)
        out = self.net(x).reshape(B, self.future_len, 2)
        # Output is cumulative positions (add last history position)
        last_pos = history_xy[:, -1, :2].unsqueeze(1)
        return out + last_pos


# ============================================================
# Training & evaluation utilities
# ============================================================

def ade_fde_m(pred_pos_km, gt_pos_km):
    """ADE/FDE in meters from positions in km."""
    err = torch.norm(pred_pos_km - gt_pos_km, dim=-1)  # (B, T)
    ade = err.mean(dim=1) * 1000  # (B,) meters
    fde = err[:, -1] * 1000
    return ade, fde


def train_one_epoch(model, loader, optimizer, scaler, device, model_name, use_amp=True, tf_ratio=0.0):
    model.train()
    total_loss, total_ade, total_fde, n = 0, 0, 0, 0
    for batch in loader:
        history = batch['history'].to(device)
        future_delta = batch['future'].to(device)
        env_map = batch['env_map'].to(device)
        goal = batch['goal'].to(device)

        gt_pos = torch.cumsum(future_delta, dim=1)  # (B, T, 2) km
        history_xy = history[:, :, :2]  # (B, 90, 2)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            # SocialLSTM/TrajectronPP support teacher forcing via future kwarg
            pred_pos = model(history_xy, env_map, goal=gt_pos[:, -1, :],
                             future=gt_pos, teacher_forcing_ratio=tf_ratio)
            loss = F.mse_loss(pred_pos, gt_pos)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            ade, fde = ade_fde_m(pred_pos, gt_pos)
        total_loss += loss.item() * history.size(0)
        total_ade += ade.sum().item()
        total_fde += fde.sum().item()
        n += history.size(0)

    return total_loss / n, total_ade / n, total_fde / n


@torch.no_grad()
def evaluate(model, loader, device, model_name, use_amp=True):
    model.eval()
    total_ade, total_fde, n = 0, 0, 0
    per_sample = []
    for batch in loader:
        history = batch['history'].to(device)
        future_delta = batch['future'].to(device)
        env_map = batch['env_map'].to(device)
        goal = batch['goal'].to(device)

        gt_pos = torch.cumsum(future_delta, dim=1)
        history_xy = history[:, :, :2]

        with autocast(enabled=use_amp):
            pred_pos = model(history_xy, env_map, goal=gt_pos[:, -1, :])

        ade, fde = ade_fde_m(pred_pos, gt_pos)
        total_ade += ade.sum().item()
        total_fde += fde.sum().item()

        # Per-sample
        for i in range(history.size(0)):
            per_sample.append({
                'ade_m': float(ade[i].item()),
                'fde_m': float(fde[i].item()),
            })
        n += history.size(0)

    return total_ade / n, total_fde / n, per_sample


@torch.no_grad()
def evaluate_terratnt(model, loader, device, cfg):
    """Special evaluation for TerraTNT which has different forward signature."""
    from models.terratnt import TerraTNT
    model.eval()
    total_ade, total_fde, n = 0, 0, 0
    per_sample = []
    for batch in loader:
        history = batch['history'].to(device)
        future_delta = batch['future'].to(device)
        env_map = batch['env_map'].to(device)
        candidates = batch['candidates'].to(device)
        target_idx = batch['target_goal_idx'].to(device)

        gt_pos = torch.cumsum(future_delta, dim=1)
        current_pos = torch.zeros(history.size(0), 2, device=device)

        with autocast(enabled=True):
            predictions, goal_logits = model(
                env_map, history, candidates, current_pos,
                teacher_forcing_ratio=0.0,
                target_goal_idx=target_idx,
                use_gt_goal=False,
            )
        pred_pos = torch.cumsum(predictions, dim=1)
        ade, fde = ade_fde_m(pred_pos, gt_pos)
        total_ade += ade.sum().item()
        total_fde += fde.sum().item()
        for i in range(history.size(0)):
            per_sample.append({
                'ade_m': float(ade[i].item()),
                'fde_m': float(fde[i].item()),
            })
        n += history.size(0)
    return total_ade / n, total_fde / n, per_sample


@torch.no_grad()
def evaluate_analytic_baselines(loader, device):
    """Evaluate constant-velocity and straight-line-to-goal baselines."""
    cv_ade_sum, cv_fde_sum = 0, 0
    line_ade_sum, line_fde_sum = 0, 0
    cv_per_sample, line_per_sample = [], []
    n = 0
    for batch in loader:
        history = batch['history'].to(device)
        future_delta = batch['future'].to(device)
        gt_pos = torch.cumsum(future_delta, dim=1)  # (B, T, 2)
        B, T, _ = gt_pos.shape
        history_xy = history[:, :, :2]

        # Constant velocity
        vel = history_xy[:, -1, :] - history_xy[:, -2, :]  # (B, 2)
        steps = torch.arange(1, T + 1, device=device).float().unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        cv_pos = vel.unsqueeze(1) * steps  # (B, T, 2)
        cv_ade, cv_fde = ade_fde_m(cv_pos, gt_pos)
        cv_ade_sum += cv_ade.sum().item()
        cv_fde_sum += cv_fde.sum().item()

        # Straight line to goal
        goal = gt_pos[:, -1, :]  # (B, 2)
        t_frac = torch.linspace(0, 1, T, device=device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        line_pos = goal.unsqueeze(1) * t_frac  # (B, T, 2)
        line_ade, line_fde = ade_fde_m(line_pos, gt_pos)
        line_ade_sum += line_ade.sum().item()
        line_fde_sum += line_fde.sum().item()

        for i in range(B):
            cv_per_sample.append({'ade_m': float(cv_ade[i].item()), 'fde_m': float(cv_fde[i].item())})
            line_per_sample.append({'ade_m': float(line_ade[i].item()), 'fde_m': float(line_fde[i].item())})
        n += B

    return {
        'ConstantVelocity': (cv_ade_sum / n, cv_fde_sum / n, cv_per_sample),
        'StraightLineToGoal': (line_ade_sum / n, line_fde_sum / n, line_per_sample),
    }


def train_and_evaluate_model(model, model_name, train_loader, val_loader, device,
                             num_epochs=40, lr=1e-3, patience=12, use_amp=True):
    """Train a baseline model and return best val metrics + per-sample results."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    best_ade = float('inf')
    best_state = None
    patience_counter = 0

    print(f'\n{"="*80}')
    print(f'Training {model_name} ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params)')
    print(f'{"="*80}')

    for epoch in range(num_epochs):
        t0 = time.time()
        # Scheduled sampling: TF ratio decays linearly from 0.3 to 0
        tf_ratio = max(0.0, 0.3 * (1.0 - epoch / max(1, num_epochs - 1)))
        train_loss, train_ade, train_fde = train_one_epoch(
            model, train_loader, optimizer, scaler, device, model_name, use_amp, tf_ratio=tf_ratio)
        scheduler.step()

        val_ade, val_fde, _ = evaluate(model, val_loader, device, model_name, use_amp)
        elapsed = time.time() - t0

        print(f'  [{model_name}] ep {epoch+1:2d}/{num_epochs} '
              f'train ADE={train_ade:.0f}m FDE={train_fde:.0f}m | '
              f'val ADE={val_ade:.0f}m FDE={val_fde:.0f}m | {elapsed:.0f}s', flush=True)

        if val_ade < best_ade:
            best_ade = val_ade
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    # Restore best and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    val_ade, val_fde, per_sample = evaluate(model, val_loader, device, model_name, use_amp)
    print(f'  {model_name} BEST: ADE={val_ade:.0f}m FDE={val_fde:.0f}m')
    return val_ade, val_fde, per_sample, model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', default='outputs/dataset_experiments/D1_optimal_combo')
    parser.add_argument('--split_file', default=None)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--output_dir', default='outputs/visualizations/baseline_comparison')
    args = parser.parse_args()

    traj_dir = args.traj_dir
    split_file = args.split_file or os.path.join(traj_dir, 'fas_splits_trajlevel.json')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Build datasets ----
    with open(split_file) as f:
        splits = json.load(f)
    phase_spec = splits['fas1']

    train_ds = FASDataset(
        traj_dir=traj_dir, fas_split_file=split_file, phase='fas1',
        history_len=HISTORY_LEN, future_len=FUTURE_LEN, num_candidates=6,
        region='bohemian_forest', sample_fraction=1.0, seed=42,
        env_coverage_km=140.0, coord_scale=1.0, goal_map_scale=1.0,
    )
    train_ds.samples_meta = []
    for item in phase_spec['train_samples']:
        train_ds.samples_meta.append((str(Path(traj_dir) / item['file']), int(item['sample_idx'])))

    val_ds = FASDataset(
        traj_dir=traj_dir, fas_split_file=split_file, phase='fas1',
        history_len=HISTORY_LEN, future_len=FUTURE_LEN, num_candidates=6,
        region='bohemian_forest', sample_fraction=1.0, seed=42,
        env_coverage_km=140.0, coord_scale=1.0, goal_map_scale=1.0,
    )
    val_ds.samples_meta = []
    for item in phase_spec['val_samples']:
        val_ds.samples_meta.append((str(Path(traj_dir) / item['file']), int(item['sample_idx'])))

    print(f'Train: {len(train_ds)} samples, Val: {len(val_ds)} samples')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_results = {}

    # ---- 1. Analytic baselines (no training) ----
    print('\n' + '='*80)
    print('Evaluating analytic baselines...')
    analytic = evaluate_analytic_baselines(val_loader, device)
    for name, (ade, fde, ps) in analytic.items():
        print(f'  {name}: ADE={ade:.0f}m FDE={fde:.0f}m')
        all_results[name] = {'ade': ade, 'fde': fde, 'per_sample': ps}

    # ---- 2. TerraTNT (load checkpoint) ----
    print('\n' + '='*80)
    print('Evaluating TerraTNT (from checkpoint)...')
    from models.terratnt import TerraTNT
    ckpt_path = 'runs/d1_optimal_fas1/terratnt_fas1_10s/20260208_172730/best_model.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        hfm = cfg.get('history_feature_mode', 'full26')
        hdim = 26 if hfm == 'full26' else 2
        tnt_model = TerraTNT(
            env_channels=18, history_input_dim=hdim, future_len=FUTURE_LEN,
            num_goals=cfg.get('num_candidates', 6),
            env_coverage_km=cfg.get('env_coverage_km', 140.0),
            paper_mode=cfg.get('paper_mode', True),
            paper_decoder=cfg.get('paper_decoder', 'hierarchical'),
            waypoint_stride=cfg.get('waypoint_stride', 90),
            goal_mode=cfg.get('goal_mode', 'joint'),
        )
        tnt_model.load_state_dict(ckpt['model_state_dict'])
        tnt_model = tnt_model.to(device).eval()
        tnt_ade, tnt_fde, tnt_ps = evaluate_terratnt(tnt_model, val_loader, device, cfg)
        print(f'  TerraTNT: ADE={tnt_ade:.0f}m FDE={tnt_fde:.0f}m')
        all_results['TerraTNT'] = {'ade': tnt_ade, 'fde': tnt_fde, 'per_sample': tnt_ps}
        del tnt_model
        torch.cuda.empty_cache()

    # ---- 3. PECNet (load checkpoint) ----
    print('\nEvaluating PECNet (from checkpoint)...')
    from models.baselines.pecnet_faithful import PECNetFaithful
    pecnet_ckpt = 'runs/pecnet_faithful/fas1_20260208_202405/best_model.pth'
    if os.path.exists(pecnet_ckpt):
        pck = torch.load(pecnet_ckpt, map_location='cpu', weights_only=False)
        pecnet = PECNetFaithful(
            history_len=HISTORY_LEN, future_len=FUTURE_LEN, history_dim=2,
            fdim=256, zdim=32, sigma=1.3, dropout=0.1, use_env=True,
            env_channels=18, env_dim=128,
        ).to(device).eval()
        pecnet.load_state_dict(pck['model_state_dict'])

        # PECNet eval: forward(history_full, env_map, goal=None) -> pred_pos, mu, logvar
        pecnet_ps = []
        pecnet_ade_sum, pecnet_fde_sum, pecnet_n = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future_delta = batch['future'].to(device)
                env_map = batch['env_map'].to(device)
                gt_pos = torch.cumsum(future_delta, dim=1)
                # Best-of-20
                B = gt_pos.size(0)
                best_preds = None
                best_ade_per = torch.full((B,), float('inf'), device=device)
                for _ in range(20):
                    pred_pos, _, _ = pecnet(history, env_map, goal=None)
                    ade_k = torch.norm(pred_pos - gt_pos, dim=-1).mean(dim=1) * 1000
                    improved = ade_k < best_ade_per
                    if best_preds is None:
                        best_preds = pred_pos.clone()
                    best_preds[improved] = pred_pos[improved]
                    best_ade_per = torch.min(best_ade_per, ade_k)
                ade, fde = ade_fde_m(best_preds, gt_pos)
                pecnet_ade_sum += ade.sum().item()
                pecnet_fde_sum += fde.sum().item()
                for i in range(B):
                    pecnet_ps.append({'ade_m': float(ade[i].item()), 'fde_m': float(fde[i].item())})
                pecnet_n += B
        pecnet_ade = pecnet_ade_sum / pecnet_n
        pecnet_fde = pecnet_fde_sum / pecnet_n
        print(f'  PECNet (best-of-20): ADE={pecnet_ade:.0f}m FDE={pecnet_fde:.0f}m')
        all_results['PECNet'] = {'ade': pecnet_ade, 'fde': pecnet_fde, 'per_sample': pecnet_ps}
        del pecnet
        torch.cuda.empty_cache()

    # ---- 4-8. Train new baselines ----
    models_to_train = {
        'LSTM_only': LSTMOnly(input_dim=2, hidden_dim=256, future_len=FUTURE_LEN),
        'LSTM_Env_Goal': LSTMEnvGoal(input_dim=2, hidden_dim=256, env_channels=18, env_dim=128, future_len=FUTURE_LEN),
        'Seq2Seq_Attn': Seq2SeqAttention(input_dim=2, hidden_dim=256, env_channels=18, future_len=FUTURE_LEN),
        'MLP': MLPBaseline(history_len=HISTORY_LEN, future_len=FUTURE_LEN, hidden_dim=512),
        'SocialLSTM': SocialLSTM({'history_length': HISTORY_LEN, 'future_length': FUTURE_LEN, 'hidden_dim': 256, 'embedding_dim': 64}),
        'TrajectronPP': TrajectronPP({'history_length': HISTORY_LEN, 'future_length': FUTURE_LEN, 'hidden_dim': 256, 'num_modes': 5, 'in_channels': 18}),
    }

    for model_name, model in models_to_train.items():
        model = model.to(device)
        val_ade, val_fde, per_sample, model = train_and_evaluate_model(
            model, model_name, train_loader, val_loader, device,
            num_epochs=args.num_epochs, lr=args.lr, patience=args.patience,
        )
        all_results[model_name] = {'ade': val_ade, 'fde': val_fde, 'per_sample': per_sample}
        # Save checkpoint
        ckpt_dir = Path('runs') / f'{model_name}_d1'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')
        del model
        torch.cuda.empty_cache()

    # ---- Summary table ----
    print('\n' + '='*80)
    print('FINAL COMPARISON (D1 dataset, fas1 phase, val set)')
    print('='*80)
    print(f'{"Model":<25s} {"ADE (m)":>10s} {"FDE (m)":>10s}')
    print('-'*50)
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['ade'])
    for name, res in sorted_models:
        marker = ' <-- OURS' if name == 'TerraTNT' else ''
        print(f'{name:<25s} {res["ade"]:>10.0f} {res["fde"]:>10.0f}{marker}')

    # ---- Per-sample comparison ----
    print('\n' + '='*80)
    print('PER-SAMPLE COMPARISON: Models that beat TerraTNT')
    print('='*80)
    if 'TerraTNT' in all_results:
        tnt_ps = all_results['TerraTNT']['per_sample']
        n_val = len(tnt_ps)
        for name, res in sorted_models:
            if name == 'TerraTNT':
                continue
            ps = res['per_sample']
            if len(ps) != n_val:
                continue
            wins = sum(1 for i in range(n_val) if ps[i]['ade_m'] < tnt_ps[i]['ade_m'])
            avg_improvement = np.mean([
                (tnt_ps[i]['ade_m'] - ps[i]['ade_m']) / tnt_ps[i]['ade_m'] * 100
                for i in range(n_val) if ps[i]['ade_m'] < tnt_ps[i]['ade_m']
            ]) if wins > 0 else 0
            print(f'  {name}: wins {wins}/{n_val} ({wins/n_val*100:.1f}%), '
                  f'avg improvement on wins: {avg_improvement:+.1f}%')

    # ---- Save all results ----
    save_data = {}
    for name, res in all_results.items():
        save_data[name] = {
            'ade': res['ade'], 'fde': res['fde'],
            'per_sample': res['per_sample'],
        }
    with open(output_dir / 'all_baselines_comparison.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nSaved to {output_dir / "all_baselines_comparison.json"}')

    # ---- Visualization ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('D1 Baseline Comparison', fontsize=14, fontweight='bold')

    # Bar chart
    ax = axes[0]
    names = [n for n, _ in sorted_models]
    ades = [r['ade'] for _, r in sorted_models]
    colors = ['#d62728' if n == 'TerraTNT' else '#1f77b4' for n in names]
    bars = ax.barh(range(len(names)), ades, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('ADE (m)')
    ax.set_title('Val ADE Comparison')
    ax.invert_yaxis()
    for bar, v in zip(bars, ades):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{v:.0f}', va='center', fontsize=7)

    # Per-sample ADE scatter: TerraTNT vs best other
    ax = axes[1]
    if 'TerraTNT' in all_results:
        tnt_ades = [s['ade_m'] for s in all_results['TerraTNT']['per_sample']]
        # Find best non-TerraTNT model
        best_other_name = None
        best_other_ade = float('inf')
        for n, r in all_results.items():
            if n == 'TerraTNT' or n in ('ConstantVelocity', 'StraightLineToGoal'):
                continue
            if r['ade'] < best_other_ade:
                best_other_ade = r['ade']
                best_other_name = n
        if best_other_name:
            other_ades = [s['ade_m'] for s in all_results[best_other_name]['per_sample']]
            if len(other_ades) == len(tnt_ades):
                ax.scatter(tnt_ades, other_ades, s=8, alpha=0.4)
                lim = max(max(tnt_ades), max(other_ades)) * 1.05
                ax.plot([0, lim], [0, lim], 'r--', alpha=0.5, label='y=x')
                ax.set_xlabel('TerraTNT ADE (m)')
                ax.set_ylabel(f'{best_other_name} ADE (m)')
                ax.set_title(f'Per-sample: TerraTNT vs {best_other_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)

    # FDE comparison
    ax = axes[2]
    fdes = [r['fde'] for _, r in sorted_models]
    bars = ax.barh(range(len(names)), fdes, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('FDE (m)')
    ax.set_title('Val FDE Comparison')
    ax.invert_yaxis()
    for bar, v in zip(bars, fdes):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{v:.0f}', va='center', fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / 'baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {output_dir / "baseline_comparison.png"}')


if __name__ == '__main__':
    main()
