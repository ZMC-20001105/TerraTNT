#!/usr/bin/env python3
"""方向意图预测器：历史轨迹+环境 → 8方向分类 + 距离回归 + 角度回归"""
import sys, os, json, math, time, argparse, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from torch.amp import GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
HISTORY_LEN, FUTURE_LEN, NUM_DIRS = 90, 360, 8
DIR_LABELS = ['N','NE','E','SE','S','SW','W','NW']

def angle_to_dir(angle_rad):
    compass = (90 - math.degrees(angle_rad)) % 360
    return int((compass + 22.5) / 45) % NUM_DIRS

class DirectionDataset(Dataset):
    """方向预测数据集，支持pkl文件缓存以加速数据加载"""
    def __init__(self, traj_dir, split_file, phase='fas1', split='train', **kw):
        self.traj_dir = Path(traj_dir)
        with open(split_file) as f:
            splits = json.load(f)
        items = splits.get(phase, {}).get(f'{split}_samples', [])
        self.samples_meta = [(str(self.traj_dir / it['file']), int(it['sample_idx'])) for it in items]
        # pkl文件级缓存: {pkl_path: loaded_data}
        self._pkl_cache = {}
        print(f'DirectionDataset: {len(self.samples_meta)} {split} samples from {len(set(m[0] for m in self.samples_meta))} files')

    def __len__(self): return len(self.samples_meta)

    def _load_pkl(self, pkl_path):
        if pkl_path not in self._pkl_cache:
            with open(pkl_path, 'rb') as f:
                self._pkl_cache[pkl_path] = pickle.load(f)
        return self._pkl_cache[pkl_path]

    def __getitem__(self, idx):
        pkl_path, si = self.samples_meta[idx]
        data = self._load_pkl(pkl_path)
        samples = data.get('samples', [data])
        s = samples[min(si, len(samples)-1)]
        hist = s['history_feat_26d'][:HISTORY_LEN].astype(np.float32)
        if hist.shape[0] < HISTORY_LEN:
            hist = np.concatenate([np.zeros((HISTORY_LEN-hist.shape[0], hist.shape[1]), dtype=np.float32), hist])
        env = s['env_map_100km'].astype(np.float32); env[17] = 0.0
        fut = s['future_rel'][:FUTURE_LEN].astype(np.float32)
        goal = fut[-1]; angle = math.atan2(goal[1], goal[0])
        return {
            'history': torch.from_numpy(hist[:,:2]),
            'env_map': torch.from_numpy(env),
            'dir_class': angle_to_dir(angle),
            'distance_km': float(np.linalg.norm(goal)),
            'angle_sincos': torch.tensor([math.sin(angle), math.cos(angle)], dtype=torch.float32),
        }

class DirectionPredictor(nn.Module):
    def __init__(self, hidden_dim=128, env_channels=18, env_dim=128, num_dirs=8):
        super().__init__()
        self.env_cnn = nn.Sequential(
            nn.Conv2d(env_channels,32,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(64,128,3,stride=2,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.env_fc = nn.Linear(128, env_dim)
        self.hist_enc = nn.LSTM(2, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        fuse = hidden_dim + env_dim
        self.dir_head = nn.Sequential(nn.Linear(fuse,256), nn.ReLU(), nn.Dropout(0.2),
                                       nn.Linear(256,128), nn.ReLU(), nn.Linear(128, num_dirs))
        self.dist_head = nn.Sequential(nn.Linear(fuse,128), nn.ReLU(), nn.Linear(128,1), nn.Softplus())
        self.angle_head = nn.Sequential(nn.Linear(fuse,128), nn.ReLU(), nn.Linear(128,2))

    def forward(self, history_xy, env_map):
        B = history_xy.size(0)
        ef = F.relu(self.env_fc(self.env_cnn(env_map).view(B,-1)))
        _, (h, _) = self.hist_enc(history_xy)
        fused = torch.cat([h[-1], ef], dim=1)
        ac = self.angle_head(fused)
        ac = ac / ac.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return self.dir_head(fused), self.dist_head(fused), ac

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', default='data/processed/final_dataset_v1/bohemian_forest')
    parser.add_argument('--split_file', default='data/processed/fas_splits/bohemian_forest/fas_splits_trajlevel.json')
    parser.add_argument('--output_dir', default='runs/direction_predictor')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--region', default='bohemian_forest')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_ds = DirectionDataset(args.traj_dir, args.split_file, split='train')
    val_ds = DirectionDataset(args.traj_dir, args.split_file, split='val')
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DirectionPredictor().to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
    scaler = GradScaler('cuda')
    cls_crit, reg_crit = nn.CrossEntropyLoss(), nn.SmoothL1Loss()
    best_acc, best_ep = 0.0, 0

    for ep in range(args.num_epochs):
        model.train(); tloss=tc=tt=0; t0=time.time()
        for b in tl:
            hx,em,dg,dkg,asg = b['history'].to(device), b['env_map'].to(device), \
                b['dir_class'].to(device), b['distance_km'].to(device).unsqueeze(1), b['angle_sincos'].to(device)
            opt.zero_grad()
            with autocast(device_type='cuda'):
                dl,dp,ap = model(hx, em)
                loss = cls_crit(dl,dg) + 0.01*reg_crit(dp,dkg) + 2.0*reg_crit(ap,asg)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            tloss += loss.item()*hx.size(0); tc += (dl.argmax(1)==dg).sum().item(); tt += hx.size(0)
        sched.step()

        model.eval(); vc=vt=vae=0
        with torch.no_grad():
            for b in vl:
                hx,em,dg,asg = b['history'].to(device), b['env_map'].to(device), \
                    b['dir_class'].to(device), b['angle_sincos'].to(device)
                with autocast(device_type='cuda'): dl,dp,ap = model(hx, em)
                vc += (dl.argmax(1)==dg).sum().item(); vt += hx.size(0)
                pa = torch.atan2(ap[:,0],ap[:,1]); ga = torch.atan2(asg[:,0],asg[:,1])
                ad = torch.abs(pa-ga); ad = torch.min(ad, 2*math.pi-ad)
                vae += torch.rad2deg(ad).sum().item()

        va = vc/max(vt,1); vae /= max(vt,1)
        print(f'Ep {ep+1:3d}/{args.num_epochs} | loss={tloss/max(tt,1):.4f} tacc={tc/max(tt,1):.3f} | '
              f'vacc={va:.3f} angle_err={vae:.1f}° | {time.time()-t0:.1f}s')
        if va > best_acc:
            best_acc, best_ep = va, ep+1
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': va, 'angle_err': vae}, out/'direction_predictor_best.pth')
            print(f'  ★ Best: acc={va:.3f}')

    print(f'\nBest: ep={best_ep}, acc={best_acc:.3f}')
    json.dump({'best_epoch':best_ep,'best_val_acc':best_acc,'dir_labels':DIR_LABELS},
              open(out/'direction_results.json','w'), indent=2)

if __name__ == '__main__':
    main()
