#!/usr/bin/env python3
"""Train V8d DecoupledGoalV8 in 2 stages:
  Stage 1: Pure α=1 training (like V6) to converge the goal-aware path
  Stage 2: Introduce curriculum corruption + gate supervision to learn α=0 fallback
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_terratnt_10s import FASDataset, HISTORY_LEN, FUTURE_LEN
from scripts.train_incremental_models import DecoupledGoalV8, ade_fde_m


def validate(model, vl, dev, force_alpha=None):
    model.eval()
    a, f, g, n = 0, 0, 0, 0
    with torch.no_grad():
        for b in vl:
            h = b['history'].to(dev); fd = b['future'].to(dev)
            em = b['env_map'].to(dev); c = b['candidates'].to(dev)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=dev)
            with autocast('cuda', enabled=True):
                pred, gl, alpha = model(em, h, c, cp, use_gt_goal=False,
                                        force_alpha=force_alpha)
            ad, fd2 = ade_fde_m(pred, gt)
            a += ad.sum().item(); f += fd2.sum().item(); n += h.size(0)
            g += alpha.sum().item()
    return a / max(1, n), f / max(1, n), g / max(1, n)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--traj_dir', default='outputs/dataset_experiments/D1_optimal_combo')
    pa.add_argument('--split_file', default='outputs/dataset_experiments/D1_optimal_combo/fas_splits_trajlevel.json')
    pa.add_argument('--output_dir', default='runs/incremental_models_v8d')
    pa.add_argument('--batch_size', type=int, default=32)
    pa.add_argument('--s1_epochs', type=int, default=35, help='Stage 1 epochs (pure α=1)')
    pa.add_argument('--s2_epochs', type=int, default=25, help='Stage 2 epochs (curriculum corruption)')
    pa.add_argument('--lr', type=float, default=1e-3)
    pa.add_argument('--patience', type=int, default=12)
    pa.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (skip Stage 1)')
    args = pa.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd_path = Path(args.output_dir); sd_path.mkdir(parents=True, exist_ok=True)
    print(f'Device: {dev}', flush=True)

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
    model = DecoupledGoalV8(history_dim=26, hidden_dim=128, env_channels=18,
                            env_feature_dim=128, decoder_hidden_dim=256,
                            future_len=FUTURE_LEN, num_waypoints=10,
                            num_candidates=6, env_coverage_km=140.0).to(dev)
    print(f'V8d params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

    name = 'V8d'
    scaler = GradScaler('cuda')
    cls_c = nn.CrossEntropyLoss()
    bce_logit = nn.BCEWithLogitsLoss()
    best = float('inf')
    no_imp = 0

    def _save(va, vf, ep, tag='best'):
        nonlocal best, no_imp
        if va < best:
            best = va; no_imp = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep,
                        'val_ade': va, 'val_fde': vf, 'model_name': name,
                        'config': {'num_candidates': 6, 'env_coverage_km': 140.0}},
                       sd_path / f'{name}_{tag}.pth')
            return True
        no_imp += 1
        return False

    # ==================== STAGE 1: Pure α=1 (like V6 training) ====================
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        src_sd = ck.get('model_state_dict', ck)
        tgt_sd = model.state_dict()
        # Transfer matching layers
        transferred = 0
        for k, v in src_sd.items():
            if k in tgt_sd and tgt_sd[k].shape == v.shape:
                tgt_sd[k] = v
                transferred += 1
        # Special: V6 fusion(320→256) → V8d base_fusion(256→256)
        # V6 fusion input = [history(128) + env(128) + goal(64)] = 320
        # V8d base_fusion input = [history(128) + env(128)] = 256
        # Extract first 256 columns (history+env) from V6 fusion weight
        if 'fusion.weight' in src_sd and 'base_fusion.weight' in tgt_sd:
            v6_fw = src_sd['fusion.weight']  # (256, 320)
            tgt_sd['base_fusion.weight'] = v6_fw[:, :256].clone()  # (256, 256)
            tgt_sd['base_fusion.bias'] = src_sd['fusion.bias'].clone()
            # Initialize goal_proj from the goal portion of V6 fusion
            # V6 fusion goal columns = [:, 256:320] (64 cols) → need to project to (256, 64)
            tgt_sd['goal_proj.weight'] = v6_fw[:, 256:].clone()  # (256, 64)
            tgt_sd['goal_proj.bias'] = torch.zeros(256)  # goal residual starts at 0
            transferred += 3
            print(f'  Transferred V6 fusion → base_fusion + goal_proj', flush=True)
        model.load_state_dict(tgt_sd)
        print(f'Resumed from {args.resume} ({transferred} layers, ADE={ck.get("val_ade",0):.0f}m)', flush=True)
    else:
        print(f'\n=== Stage 1: Pure α=1 ({args.s1_epochs} ep) ===', flush=True)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.s1_epochs, eta_min=args.lr*0.01)

        for ep in range(args.s1_epochs):
            model.train()
            tl_sum, nb = 0, 0
            for batch in tl:
                h = batch['history'].to(dev); fd = batch['future'].to(dev)
                em = batch['env_map'].to(dev); c = batch['candidates'].to(dev)
                ti = batch['target_goal_idx'].to(dev)
                gt = torch.cumsum(fd, dim=1)
                cp = torch.zeros(h.size(0), 2, device=dev)

                opt.zero_grad()
                with autocast('cuda', enabled=True):
                    deltas, gl, wp, alpha, _ = model(
                        em, h, c, cp, target_goal_idx=ti, use_gt_goal=True,
                        force_alpha=1.0)
                    pred_pos = torch.cumsum(deltas, dim=1)
                    loss = F.mse_loss(pred_pos, gt) + cls_c(gl, ti)
                    if wp is not None:
                        wi = torch.tensor(model.waypoint_indices, device=dev, dtype=torch.long)
                        loss += 0.5 * F.mse_loss(wp, gt.index_select(1, wi))

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
                tl_sum += loss.item(); nb += 1

            sched.step()
            va, vf, _ = validate(model, vl, dev, force_alpha=1.0)
            imp = ' *BEST*' if _save(va, vf, ep, 's1_best') else ''
            print(f'  [S1] {ep+1}/{args.s1_epochs} loss={tl_sum/nb:.4f} ADE(α=1)={va:.0f}m{imp}',
                  flush=True)
            if no_imp >= args.patience:
                print(f'  S1 early stopping', flush=True)
                break

        # Load best S1 checkpoint
        ck = torch.load(sd_path / f'{name}_s1_best.pth', map_location=dev, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        s1_best = ck['val_ade']
        print(f'\nStage 1 done. Best ADE(α=1)={s1_best:.0f}m', flush=True)

    # Check α=0 baseline before Stage 2
    va0, _, _ = validate(model, vl, dev, force_alpha=0.0)
    va1, _, _ = validate(model, vl, dev, force_alpha=1.0)
    print(f'Pre-S2: ADE(α=1)={va1:.0f}m  ADE(α=0)={va0:.0f}m', flush=True)

    # ==================== STAGE 2: Curriculum corruption + gate ====================
    print(f'\n=== Stage 2: Curriculum corruption ({args.s2_epochs} ep) ===', flush=True)
    best = float('inf')
    no_imp = 0
    opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.s2_epochs, eta_min=args.lr*0.003)

    for ep in range(args.s2_epochs):
        model.train()
        tl_sum, nb, alpha_sum = 0, 0, 0
        # Corruption ramps from 0.05 to 0.4 over training
        corrupt_prob = 0.05 + 0.35 * min(1.0, ep / max(1, args.s2_epochs * 0.5))

        for batch in tl:
            h = batch['history'].to(dev); fd = batch['future'].to(dev)
            em = batch['env_map'].to(dev); c = batch['candidates'].to(dev)
            ti = batch['target_goal_idx'].to(dev)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=dev)
            B = h.size(0)

            # Per-sample α
            alpha_vals = torch.ones(B, 1, device=dev)
            corrupt_mask = torch.rand(B) < corrupt_prob
            n_corrupt = corrupt_mask.sum().item()
            if n_corrupt > 0:
                low_alpha = torch.from_numpy(
                    np.random.beta(1, 3, size=(n_corrupt, 1)).astype(np.float32)).to(dev)
                alpha_vals[corrupt_mask] = low_alpha
                c[corrupt_mask] = torch.randn(n_corrupt, c.size(1), 2, device=dev) * 20.0

            opt.zero_grad()
            with autocast('cuda', enabled=True):
                deltas, gl, wp, alpha, alpha_logit = model(
                    em, h, c, cp, target_goal_idx=ti, use_gt_goal=True,
                    force_alpha=alpha_vals)
                pred_pos = torch.cumsum(deltas, dim=1)

                loss = F.mse_loss(pred_pos, gt)
                if n_corrupt < B:
                    good_mask = ~corrupt_mask
                    loss += cls_c(gl[good_mask], ti[good_mask])
                if wp is not None:
                    wi = torch.tensor(model.waypoint_indices, device=dev, dtype=torch.long)
                    loss += 0.5 * F.mse_loss(wp, gt.index_select(1, wi))
                # Gate supervision
                loss += 0.3 * bce_logit(alpha_logit, alpha_vals.detach())

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl_sum += loss.item(); nb += 1
            alpha_sum += alpha_vals.mean().item()

        sched.step()
        va, vf, vg = validate(model, vl, dev)
        va1, _, _ = validate(model, vl, dev, force_alpha=1.0)
        va0, _, _ = validate(model, vl, dev, force_alpha=0.0)
        imp = ' *BEST*' if _save(va1, vf, args.s1_epochs + ep, 'best') else ''
        print(f'  [S2] {ep+1}/{args.s2_epochs} loss={tl_sum/nb:.4f} '
              f'ADE(auto)={va:.0f}m ADE(α=1)={va1:.0f}m ADE(α=0)={va0:.0f}m '
              f'val_α={vg:.3f} corrupt={corrupt_prob:.2f}{imp}', flush=True)
        if no_imp >= args.patience:
            print(f'  S2 early stopping', flush=True)
            break

    print(f'\nDone. Best ADE={best:.0f}m -> {sd_path}/{name}_best.pth', flush=True)


if __name__ == '__main__':
    main()
