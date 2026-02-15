#!/usr/bin/env python3
"""
TerraTNT-Env: 无目标先验的环境感知轨迹预测模型
复用TerraTNT的环境编码器+历史编码器，去掉目标分类器，直接自回归解码。
"""
import sys, os, json, pickle, argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

FUTURE_LEN = 360
HISTORY_LEN = 90


class TerraTNTEnv(nn.Module):
    """环境感知轨迹预测（无目标先验）"""
    def __init__(self, history_dim=26, hidden_dim=128, env_channels=18,
                 env_feature_dim=128, decoder_hidden_dim=256,
                 future_len=360, env_coverage_km=140.0):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = decoder_hidden_dim
        self.env_feature_dim = env_feature_dim
        self.env_coverage_km = env_coverage_km

        from models.terratnt import PaperCNNEnvironmentEncoder, PaperLSTMHistoryEncoder
        self.env_encoder = PaperCNNEnvironmentEncoder(
            input_channels=env_channels, feature_dim=env_feature_dim)
        self.history_encoder = PaperLSTMHistoryEncoder(
            input_dim=history_dim, hidden_dim=hidden_dim, num_layers=2)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + env_feature_dim, decoder_hidden_dim), nn.ReLU())
        self.spatial_in = nn.Sequential(
            nn.Linear(env_feature_dim, decoder_hidden_dim), nn.ReLU())
        self.env_local_scale = nn.Parameter(torch.tensor(1.0))
        self.motion_proj = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, decoder_hidden_dim))
        self.pos_proj = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, decoder_hidden_dim))
        self.decoder_lstm = nn.LSTM(
            2 + decoder_hidden_dim, decoder_hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(decoder_hidden_dim, 2)

    def _extract_motion(self, history):
        last_n = min(10, history.size(1))
        r = history[:, -last_n:, :]
        vx = r[:, :, 2].mean(1); vy = r[:, :, 3].mean(1)
        speed = r[:, :, 9].mean(1)
        heading = torch.atan2(r[:, :, 6].mean(1), r[:, :, 7].mean(1))
        return torch.stack([vx, vy, speed, heading], dim=1)

    def forward(self, env_map, history, **kwargs):
        B = env_map.size(0); device = env_map.device
        env_global, _, env_spatial = self.env_encoder(env_map)
        _, (h_n, _) = self.history_encoder(history)
        hist_feat = h_n[-1]
        motion = self.motion_proj(self._extract_motion(history))
        context = self.fusion(torch.cat([hist_feat, env_global], dim=1)) + motion

        h_dec = torch.zeros(2, B, self.hidden_dim, device=device)
        c_dec = torch.zeros(2, B, self.hidden_dim, device=device)
        h_dec[-1] = context
        curr_pos = torch.zeros(B, 1, 2, device=device)
        preds = []
        half = max(1e-6, self.env_coverage_km * 0.5)

        for t in range(self.future_len):
            sc = context.clone()
            if env_spatial is not None:
                p = curr_pos.squeeze(1)
                gx = (p[:, 0] / half).clamp(-1, 1)
                gy = (-p[:, 1] / half).clamp(-1, 1)
                grid = torch.stack([gx, gy], dim=1).view(-1, 1, 1, 2)
                samp = F.grid_sample(env_spatial, grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=True)
                sc = sc + self.env_local_scale * self.spatial_in(samp.squeeze(-1).squeeze(-1))
            sc = sc + self.pos_proj(curr_pos.squeeze(1) / half)
            dec_in = torch.cat([curr_pos, sc.unsqueeze(1)], dim=-1)
            out, (h_dec, c_dec) = self.decoder_lstm(dec_in, (h_dec, c_dec))
            delta = self.output_fc(out)
            curr_pos = curr_pos + delta
            preds.append(curr_pos)
        return torch.cat(preds, dim=1)


class TrainDatasetNoGoal(Dataset):
    def __init__(self, traj_dir, fas_split_file, split_key='fas1',
                 sample_fraction=0.1, seed=42, use_train=True):
        self.traj_dir = Path(traj_dir)
        self.seed = seed
        self._file_cache = {}
        with open(fas_split_file, 'r') as f:
            splits = json.load(f)
        phase_spec = splits.get(split_key, {})
        key = 'train_samples' if use_train else 'val_samples'
        samples_list = phase_spec.get(key, phase_spec.get('samples', []))
        self.samples_meta = []
        if samples_list and isinstance(samples_list[0], dict):
            for item in samples_list:
                fp = self.traj_dir / str(item.get('file', ''))
                si = item.get('sample_idx')
                if fp.exists() and si is not None:
                    self.samples_meta.append((str(fp), int(si)))
        elif 'files' in phase_spec:
            for fname in phase_spec['files']:
                fp = self.traj_dir / str(fname)
                if fp.exists():
                    try:
                        with open(fp, 'rb') as f:
                            data = pickle.load(f)
                        for si in range(len(data.get('samples', []))):
                            self.samples_meta.append((str(fp), si))
                    except: pass
        if sample_fraction < 1.0 and self.samples_meta:
            import random; random.seed(seed)
            n = max(1, int(len(self.samples_meta) * sample_fraction))
            self.samples_meta = random.sample(self.samples_meta, n)
        print(f"  数据集: {len(self.samples_meta)} 样本 ({'训练' if use_train else '验证'})")

    def _load(self, fp, si):
        if fp not in self._file_cache:
            if len(self._file_cache) > 5: self._file_cache.clear()
            with open(fp, 'rb') as f: self._file_cache[fp] = pickle.load(f)
        return self._file_cache[fp]['samples'][si]

    @staticmethod
    def _norm_env(e):
        e = np.nan_to_num(np.asarray(e, dtype=np.float32).copy())
        dem = e[0]
        if float(np.nanmax(np.abs(dem))) > 50:
            mn, mx = float(np.nanmin(dem)), float(np.nanmax(dem))
            e[0] = (dem - mn) / max(mx - mn, 1e-6)
        e[1] = np.clip(e[1], 0, 1)
        e[2:4] = np.clip(e[2:4], -1, 1)
        e[4:18] = np.clip(e[4:18], 0, 1)
        return e

    def __len__(self): return len(self.samples_meta)

    def __getitem__(self, idx):
        s = self._load(*self.samples_meta[idx])
        env_map = torch.from_numpy(self._norm_env(s['env_map_100km'])).float()
        history = torch.FloatTensor(s['history_feat_26d'])
        future_rel = torch.FloatTensor(s['future_rel'])
        future_delta = torch.diff(future_rel, dim=0, prepend=torch.zeros(1, 2))
        gt_pos = torch.cumsum(future_delta, dim=0)
        return {'history': history, 'future_delta': future_delta,
                'gt_pos': gt_pos, 'env_map': env_map}


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    traj_dir = args.traj_dir
    fas_file = args.fas_split_file

    train_ds = TrainDatasetNoGoal(traj_dir, fas_file, 'fas1',
                                   sample_fraction=args.sample_fraction, use_train=True)
    val_ds = TrainDatasetNoGoal(traj_dir, fas_file, 'fas1',
                                 sample_fraction=1.0, use_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TerraTNTEnv(future_len=FUTURE_LEN, env_coverage_km=140.0).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_ade = float('inf')
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0; n_batch = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            h = batch['history'].to(device)
            env = batch['env_map'].to(device)
            gt = batch['gt_pos'].to(device)

            optimizer.zero_grad()
            with autocast('cuda', enabled=True):
                pred = model(env, h)
                loss = F.mse_loss(pred, gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); n_batch += 1

        avg_loss = total_loss / max(n_batch, 1)

        # 验证
        model.eval()
        val_ades = []
        with torch.no_grad():
            for batch in val_loader:
                h = batch['history'].to(device)
                env = batch['env_map'].to(device)
                gt = batch['gt_pos'].to(device)
                with autocast('cuda', enabled=True):
                    pred = model(env, h)
                err = torch.norm(pred - gt, dim=-1).mean(dim=1) * 1000  # meters
                val_ades.extend(err.cpu().tolist())
        val_ade = np.mean(val_ades)
        scheduler.step(val_ade)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}  loss={avg_loss:.4f}  val_ADE={val_ade:.0f}m  lr={lr_now:.1e}")

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch, 'val_ade': val_ade,
                'model_name': 'TerraTNT_Env',
                'n_params': n_params,
            }, save_dir / 'best_model.pth')
            print(f"  ★ 保存最优模型 ADE={val_ade:.0f}m")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  早停 (patience={args.patience})")
                break

    print(f"\n训练完成. 最优验证ADE={best_val_ade:.0f}m")
    print(f"模型保存: {save_dir / 'best_model.pth'}")
    return best_val_ade


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--traj_dir', default=str(PROJECT_ROOT / 'outputs/dataset_experiments/D1_optimal_combo'))
    p.add_argument('--fas_split_file', default=str(PROJECT_ROOT / 'outputs/dataset_experiments/D1_optimal_combo/fas_splits_full_phases.json'))
    p.add_argument('--save_dir', default=str(PROJECT_ROOT / 'runs/terratnt_env'))
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--sample_fraction', type=float, default=0.1)
    args = p.parse_args()
    train(args)
