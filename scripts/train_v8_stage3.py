#!/usr/bin/env python3
"""Resume V8 training from Stage 2 checkpoint, run Stage 3 (joint gate) only."""
import sys, json, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_terratnt_10s import FASDataset, HISTORY_LEN, FUTURE_LEN
from scripts.train_incremental_models import DualDecoderV8, ade_fde_m

def val_model(model, vl, dev, mode='full', fg=None):
    model.eval()
    a, f, g, n = 0, 0, 0, 0
    with torch.no_grad():
        for b in vl:
            h = b['history'].to(dev); fd = b['future'].to(dev)
            em = b['env_map'].to(dev); c = b['candidates'].to(dev)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=dev)
            with autocast('cuda', enabled=True):
                o = model(em, h, c, cp, use_gt_goal=False, mode=mode, force_gate=fg)
            ad, fd2 = ade_fde_m(o[0], gt)
            a += ad.sum().item(); f += fd2.sum().item(); n += h.size(0)
            if len(o) > 2 and o[2].numel() > 1:
                g += o[2].sum().item()
    return a / max(1, n), f / max(1, n), g / max(1, n)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--traj_dir', default='outputs/dataset_experiments/D1_optimal_combo')
    pa.add_argument('--split_file', default='outputs/dataset_experiments/D1_optimal_combo/fas_splits_trajlevel.json')
    pa.add_argument('--output_dir', default='runs/incremental_models_v8')
    pa.add_argument('--resume', default='runs/incremental_models_v8/V8_best.pth')
    pa.add_argument('--batch_size', type=int, default=32)
    pa.add_argument('--num_epochs', type=int, default=30)
    pa.add_argument('--lr', type=float, default=3e-4)
    pa.add_argument('--patience', type=int, default=10)
    args = pa.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd = Path(args.output_dir); sd.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds_kw = dict(history_len=HISTORY_LEN, future_len=FUTURE_LEN, num_candidates=6,
                 region='bohemian_forest', env_coverage_km=140.0, coord_scale=1.0)
    tr_ds = FASDataset(args.traj_dir, args.split_file, phase='fas1', **ds_kw)
    va_ds = FASDataset(args.traj_dir, args.split_file, phase='fas1', **ds_kw)
    with open(args.split_file) as f:
        sp = json.load(f)
    tr_ds.samples_meta = [(str(Path(args.traj_dir) / i['file']), int(i['sample_idx']))
                           for i in sp['fas1']['train_samples']]
    va_ds.samples_meta = [(str(Path(args.traj_dir) / i['file']), int(i['sample_idx']))
                           for i in sp['fas1']['val_samples']]
    tl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
    vl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=1, pin_memory=True)
    print(f'Train:{len(tr_ds)} Val:{len(va_ds)}', flush=True)

    # Model
    model = DualDecoderV8(history_dim=26, hidden_dim=128, env_channels=18, env_feature_dim=128,
                          decoder_hidden_dim=256, future_len=FUTURE_LEN, num_waypoints=10,
                          num_candidates=6, env_coverage_km=140.0).to(dev)

    # Load Stage 2 checkpoint
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    print(f'Loaded checkpoint: ADE={ck["val_ade"]:.0f}m epoch={ck["epoch"]}', flush=True)

    # Quick validation of both paths
    va_ga, _, _ = val_model(model, vl, dev, 'goal_aware', 1.0)
    va_gf, _, _ = val_model(model, vl, dev, 'goal_free')
    va_full, _, vg = val_model(model, vl, dev, 'full')
    print(f'Pre-S3: goal_aware={va_ga:.0f}m  goal_free={va_gf:.0f}m  full={va_full:.0f}m  gate={vg:.3f}', flush=True)

    # Stage 3: Joint gate training
    print(f'\n=== Stage 3: Joint Gate ({args.num_epochs} ep) ===', flush=True)
    name = 'V8'
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    scaler = GradScaler('cuda')
    cls_c = nn.CrossEntropyLoss()
    bce_logit = nn.BCEWithLogitsLoss()
    best, ni = float('inf'), 0

    for ep in range(args.num_epochs):
        model.train()
        tl_sum, nb, gs, gtsum = 0, 0, 0, 0
        for b in tl:
            h = b['history'].to(dev); fd = b['future'].to(dev)
            em = b['env_map'].to(dev); c = b['candidates'].to(dev)
            ti = b['target_goal_idx'].to(dev)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=dev)
            B = h.size(0)

            corrupt = (torch.rand(1).item() > 0.5)
            if corrupt:
                ci = torch.randn(B, c.size(1), 2, device=dev) * 20.0
                gtgt = torch.zeros(B, 1, device=dev)
            else:
                ci = c
                gtgt = torch.ones(B, 1, device=dev)

            opt.zero_grad()
            with autocast('cuda', enabled=True):
                tr, gl, wp, al, al_logit = model(
                    em, h, ci, cp, target_goal_idx=ti,
                    use_gt_goal=(not corrupt), mode='full')
                lo = F.mse_loss(tr, gt) + 0.5 * bce_logit(al_logit, gtgt)
                if not corrupt:
                    lo += cls_c(gl, ti)
                    if wp is not None:
                        wi = torch.tensor(model.waypoint_indices, device=dev, dtype=torch.long)
                        lo += 0.5 * F.mse_loss(wp, gt.index_select(1, wi))

            scaler.scale(lo).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl_sum += lo.item(); nb += 1
            gs += al.detach().mean().item()
            gtsum += gtgt.mean().item()

        va, vf, vg = val_model(model, vl, dev, 'full')
        sched.step(va)
        imp = ''
        if va < best:
            best = va; ni = 0; imp = ' *BEST*'
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep,
                        'val_ade': va, 'val_fde': vf, 'model_name': name},
                       sd / f'{name}_best.pth')
        else:
            ni += 1
        print(f'  [S3] {ep+1}/{args.num_epochs} loss={tl_sum/nb:.4f} ADE={va:.0f}m '
              f'gate={gs/nb:.3f}(tgt={gtsum/nb:.2f}) vgate={vg:.3f}{imp}', flush=True)
        if ni >= args.patience:
            print(f'  Early stopping', flush=True)
            break

    # Final eval of both paths
    va_ga, _, _ = val_model(model, vl, dev, 'goal_aware', 1.0)
    va_gf, _, _ = val_model(model, vl, dev, 'goal_free')
    va_full, _, vg = val_model(model, vl, dev, 'full')
    print(f'\nFinal: goal_aware={va_ga:.0f}m  goal_free={va_gf:.0f}m  full={va_full:.0f}m  gate={vg:.3f}')
    print(f'Best ADE={best:.0f}m -> {sd}/{name}_best.pth')

if __name__ == '__main__':
    main()
