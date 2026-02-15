#!/usr/bin/env python3
"""
MLP解码器消融实验：用MLP替代LSTM自回归解码器。
论文表4.13中的"MLP替代LSTM"消融。

MLP解码器：将context+goal+waypoint信息一次性映射为360×2的轨迹，
而不是逐步自回归解码。这验证了自回归解码的必要性。
"""

import sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.train_incremental_models import (
    FASDataset, ade_fde_m, HISTORY_LEN, FUTURE_LEN,
)
from models.terratnt import PaperCNNEnvironmentEncoder, PaperLSTMHistoryEncoder, PaperGoalClassifier
from scripts.evaluate_phases_v2 import PhaseV2Dataset, PHASE_V2_CONFIGS, compute_metrics


class TerraTNTMLPDecoder(nn.Module):
    """V6R变体：用MLP替代LSTM自回归解码器。
    编码器部分完全相同（env_encoder, history_encoder, goal_classifier, fusion）。
    解码器改为：context → MLP → 360×2 位移。
    """
    def __init__(self, history_dim=26, hidden_dim=128, env_channels=18,
                 env_feature_dim=128, decoder_hidden_dim=256,
                 future_len=360, num_candidates=6,
                 env_coverage_km=140.0, goal_norm_denom=70.0):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = decoder_hidden_dim

        # --- 编码器（与V6R完全相同）---
        self.env_encoder = PaperCNNEnvironmentEncoder(
            input_channels=env_channels, feature_dim=env_feature_dim)
        self.history_encoder = PaperLSTMHistoryEncoder(
            input_dim=history_dim, hidden_dim=hidden_dim, num_layers=2)
        self.goal_classifier = PaperGoalClassifier(
            env_feature_dim=env_feature_dim, history_feature_dim=hidden_dim,
            num_goals=num_candidates, goal_norm_denom=goal_norm_denom)

        # --- Fusion（与V6R相同）---
        self.goal_fc = nn.Linear(2, 64)
        self.fusion = nn.Linear(hidden_dim + env_feature_dim + 64, decoder_hidden_dim)

        # --- MLP解码器（替代LSTM自回归）---
        self.decoder_mlp = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, future_len * 2),
        )

    def forward(self, env_map, history, candidates, current_pos,
                teacher_forcing_ratio=0.0, ground_truth=None,
                target_goal_idx=None, use_gt_goal=False, goal=None,
                **kwargs):
        B = env_map.size(0)
        device = env_map.device

        # Encode
        env_global, env_tokens, env_spatial = self.env_encoder(env_map)
        _, (h_n, c_n) = self.history_encoder(history)
        history_features = h_n[-1]

        # Goal classification
        goal_logits = self.goal_classifier(env_global, history_features, candidates)

        # Select goal
        if goal is not None:
            selected_goal = goal
        elif use_gt_goal and target_goal_idx is not None:
            selected_goal = candidates[torch.arange(B, device=device), target_goal_idx]
        else:
            _, top_idx = torch.max(goal_logits, dim=1)
            selected_goal = candidates[torch.arange(B, device=device), top_idx]

        # Fuse
        goal_feat = F.relu(self.goal_fc(selected_goal))
        context = F.relu(self.fusion(
            torch.cat([history_features, env_global, goal_feat], dim=1)))

        # MLP decode: one-shot prediction
        out = self.decoder_mlp(context)  # (B, future_len*2)
        deltas = out.view(B, self.future_len, 2)
        predictions = torch.cumsum(deltas, dim=1)

        if not self.training:
            return predictions, goal_logits

        return deltas, goal_logits


def train_mlp_model(model, train_loader, val_loader, device,
                    num_epochs=25, lr=5e-4, patience=8, save_dir=None):
    """训练MLP解码器模型"""
    model_name = 'V6R_MLP'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)
    scaler = GradScaler('cuda')
    cls_criterion = nn.CrossEntropyLoss()

    best_val_ade = float('inf')
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0, 0

        for batch in train_loader:
            history = batch['history'].to(device)
            future_delta = batch['future'].to(device)
            env_map = batch['env_map'].to(device)
            candidates = batch['candidates'].to(device)
            target_idx = batch['target_goal_idx'].to(device)
            current_pos = torch.zeros(history.size(0), 2, device=device)
            gt_pos = torch.cumsum(future_delta, dim=1)

            optimizer.zero_grad()
            with autocast('cuda', enabled=True):
                pred_delta, goal_logits = model(
                    env_map, history, candidates, current_pos,
                    target_goal_idx=target_idx, use_gt_goal=True)
                pred_pos = torch.cumsum(pred_delta, dim=1)
                loss_traj = F.mse_loss(pred_pos, gt_pos)
                loss_cls = cls_criterion(goal_logits, target_idx)
                loss = loss_traj + loss_cls

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_ade_sum, val_fde_sum, val_n = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future_delta = batch['future'].to(device)
                env_map = batch['env_map'].to(device)
                candidates = batch['candidates'].to(device)
                current_pos = torch.zeros(history.size(0), 2, device=device)
                gt_pos = torch.cumsum(future_delta, dim=1)

                with autocast('cuda', enabled=True):
                    pred_pos, _ = model(env_map, history, candidates, current_pos,
                                        use_gt_goal=False)
                ade, fde = ade_fde_m(pred_pos, gt_pos)
                val_ade_sum += ade.sum().item()
                val_fde_sum += fde.sum().item()
                val_n += history.size(0)

        val_ade = val_ade_sum / max(1, val_n)
        val_fde = val_fde_sum / max(1, val_n)

        improved = ''
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            no_improve = 0
            improved = ' *BEST*'
            if save_dir:
                save_path = Path(save_dir) / f'{model_name}_best.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_ade': val_ade,
                    'val_fde': val_fde,
                    'model_name': model_name,
                }, save_path)
        else:
            no_improve += 1

        cur_lr = optimizer.param_groups[0]['lr']
        print(f'  [{model_name}] Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f} '
              f'val_ADE={val_ade:.0f}m val_FDE={val_fde:.0f}m lr={cur_lr:.2e}{improved}', flush=True)

        if no_improve >= patience:
            print(f'  Early stopping at epoch {epoch+1}', flush=True)
            break

    return best_val_ade


@torch.no_grad()
def evaluate_mlp_model(ckpt_path, device, phase='P1a'):
    """评估MLP解码器模型"""
    model = TerraTNTMLPDecoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    ds = PhaseV2Dataset(
        traj_dir=str(PROJECT_ROOT / 'outputs/dataset_experiments/D1_optimal_combo'),
        fas_split_file=str(PROJECT_ROOT / 'outputs/dataset_experiments/D1_optimal_combo/fas_splits_full_phases.json'),
        phase_config=PHASE_V2_CONFIGS[phase],
        seed=42,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    all_metrics = []
    for batch in loader:
        history = batch['history'].to(device)
        future = batch['future'].to(device)
        env_map = batch['env_map'].to(device)
        candidates = batch['candidates'].to(device)
        gt_pos = torch.cumsum(future, dim=1)
        current_pos = torch.zeros(history.size(0), 2, device=device)

        with autocast('cuda', enabled=True):
            pred_pos, _ = model(env_map, history, candidates, current_pos, use_gt_goal=False)

        for b in range(history.size(0)):
            m = compute_metrics(pred_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy())
            all_metrics.append(m)

    ades = [m['ade'] for m in all_metrics]
    fdes = [m['fde'] for m in all_metrics]
    return {
        'ade_mean': float(np.mean(ades)),
        'fde_mean': float(np.mean(fdes)),
        'n_samples': len(all_metrics),
    }


def main():
    parser = argparse.ArgumentParser(description='MLP解码器消融实验')
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = PROJECT_ROOT / 'runs' / 'mlp_decoder_ablation'
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'V6R_MLP_best.pth'

    if not args.eval_only:
        print(f'{"="*60}')
        print('训练 MLP解码器变体 (替代LSTM自回归解码器)')
        print(f'{"="*60}')

        # 数据集
        split_file = str(PROJECT_ROOT / 'outputs/dataset_experiments/D1_optimal_combo/fas_splits_trajlevel.json')
        traj_dir = str(PROJECT_ROOT / 'outputs/dataset_experiments/D1_optimal_combo')

        with open(split_file) as f:
            splits = json.load(f)

        def _make_ds(phase='fas1'):
            return FASDataset(
                traj_dir=traj_dir, fas_split_file=split_file,
                phase=phase, history_len=HISTORY_LEN, future_len=FUTURE_LEN,
                num_candidates=6, region='bohemian_forest',
                sample_fraction=1.0, seed=42, env_coverage_km=140.0,
                coord_scale=1.0, goal_map_scale=1.0,
            )

        train_ds = _make_ds('fas1')
        val_ds = _make_ds('fas1')
        train_items = splits['fas1']['train_samples']
        train_ds.samples_meta = [(str(Path(traj_dir) / item['file']), int(item['sample_idx']))
                                 for item in train_items]
        val_items = splits['fas1']['val_samples']
        val_ds.samples_meta = [(str(Path(traj_dir) / item['file']), int(item['sample_idx']))
                               for item in val_items]

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=1, pin_memory=True)
        print(f'  Train: {len(train_ds)}, Val: {len(val_ds)}')

        # 创建模型并迁移编码器权重
        model = TerraTNTMLPDecoder().to(device)

        base_ckpt = PROJECT_ROOT / 'runs/incremental_models_v6r/V6_best.pth'
        if base_ckpt.exists():
            ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt)
            model_sd = model.state_dict()
            compatible = {k: v for k, v in sd.items()
                         if k in model_sd and v.shape == model_sd[k].shape}
            model.load_state_dict(compatible, strict=False)
            print(f'  迁移学习: 加载了 {len(compatible)}/{len(model_sd)} 个参数')

        n_params = sum(p.numel() for p in model.parameters())
        print(f'  模型参数: {n_params:,}')

        best_ade = train_mlp_model(
            model, train_loader, val_loader, device,
            num_epochs=args.num_epochs, lr=args.lr,
            patience=args.patience, save_dir=str(save_dir),
        )
        print(f'\n  MLP解码器: best_val_ADE = {best_ade:.0f}m')

    # 评估
    if ckpt_path.exists():
        print(f'\n{"="*60}')
        print('评估 MLP解码器 (Phase P1a)')
        print(f'{"="*60}')
        r = evaluate_mlp_model(ckpt_path, device)
        print(f'  ADE={r["ade_mean"]/1000:.2f}km  FDE={r["fde_mean"]/1000:.2f}km')

        # 保存结果
        out_path = PROJECT_ROOT / 'outputs/evaluation/ablation/mlp_decoder_results.json'
        with open(out_path, 'w') as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f'  结果已保存到: {out_path}')
    else:
        print('  无MLP解码器checkpoint')


if __name__ == '__main__':
    main()
