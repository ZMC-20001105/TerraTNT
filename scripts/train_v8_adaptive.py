#!/usr/bin/env python3
"""Train V8 AdaptiveGoalV8: V6 + learned goal scaling.

2-stage training:
  Stage 1 (70%): Fine-tune from V6 with curriculum random α
    - Start with α=1 (pure V6), gradually increase randomness
    - Random α drawn from Beta(a,b) with shifting parameters
    - Also randomly corrupt candidates (20%) to teach α→0
  Stage 2 (30%): Free the gate, train with mixed good/bad candidates
    - 50% good candidates (α should be high), 50% corrupted (α should be low)
    - Gate supervision via BCEWithLogitsLoss on alpha_logit

Usage:
  python scripts/train_v8_adaptive.py --resume_v6 runs/incremental_models/V6_best.pth
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
from scripts.train_incremental_models import AdaptiveGoalV8, ade_fde_m


def validate(model, vl, dev):
    model.eval()
    a, f, g, n = 0, 0, 0, 0
    with torch.no_grad():
        for b in vl:
            h = b['history'].to(dev); fd = b['future'].to(dev)
            em = b['env_map'].to(dev); c = b['candidates'].to(dev)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=dev)
            with autocast('cuda', enabled=True):
                pred, gl, alpha = model(em, h, c, cp, use_gt_goal=False)
            ad, fd2 = ade_fde_m(pred, gt)
            a += ad.sum().item(); f += fd2.sum().item(); n += h.size(0)
            g += alpha.sum().item()
    return a / max(1, n), f / max(1, n), g / max(1, n)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--traj_dir', default='outputs/dataset_experiments/D1_optimal_combo')
    pa.add_argument('--split_file', default='outputs/dataset_experiments/D1_optimal_combo/fas_splits_trajlevel.json')
    pa.add_argument('--output_dir', default='runs/incremental_models_v8')
    pa.add_argument('--batch_size', type=int, default=32)
    pa.add_argument('--num_epochs', type=int, default=40)
    pa.add_argument('--lr', type=float, default=5e-4)
    pa.add_argument('--resume_v6', type=str, default=None)
    pa.add_argument('--patience', type=int, default=10)
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
    model = AdaptiveGoalV8(history_dim=26, hidden_dim=128, env_channels=18,
                           env_feature_dim=128, decoder_hidden_dim=256,
                           future_len=FUTURE_LEN, num_waypoints=10,
                           num_candidates=6, env_coverage_km=140.0).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'V8 params: {n_params:,}', flush=True)

    # Load V6 weights
    if args.resume_v6:
        ck = torch.load(args.resume_v6, map_location=dev, weights_only=False)
        mi, _ = model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
        print(f'Loaded V6 (ADE={ck.get("val_ade", 0):.0f}m), new layers: {len(mi)}', flush=True)

    name = 'V8'
    scaler = GradScaler('cuda')
    cls_c = nn.CrossEntropyLoss()
    bce_logit = nn.BCEWithLogitsLoss()
    NE = args.num_epochs
    s1e = int(NE * 0.7)
    s2e = NE - s1e
    best = float('inf')
    no_imp = 0

    def _save(va, vf, ep):
        nonlocal best, no_imp
        if va < best:
            best = va; no_imp = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep,
                        'val_ade': va, 'val_fde': vf, 'model_name': name,
                        'config': {'num_candidates': 6, 'env_coverage_km': 140.0}},
                       sd_path / f'{name}_best.pth')
            return True
        no_imp += 1
        return False

    # Quick baseline validation
    va0, vf0, vg0 = validate(model, vl, dev)
    print(f'Baseline: ADE={va0:.0f}m FDE={vf0:.0f}m alpha={vg0:.3f}', flush=True)

    # ==================== STAGE 1: Curriculum random α ====================
    print(f'\n=== Stage 1: Curriculum α ({s1e} ep) ===', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    for ep in range(s1e):
        model.train()
        tl_sum, nb, alpha_sum = 0, 0, 0
        # Curriculum: corruption probability increases from 0 to 0.3
        corrupt_prob = 0.3 * min(1.0, ep / max(1, s1e * 0.5))
        # Alpha randomness: start deterministic (α=1), gradually add noise
        alpha_noise = 0.5 * min(1.0, ep / max(1, s1e * 0.5))

        for batch in tl:
            h = batch['history'].to(dev); fd = batch['future'].to(dev)
            em = batch['env_map'].to(dev); c = batch['candidates'].to(dev)
            ti = batch['target_goal_idx'].to(dev)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=dev)
            B = h.size(0)

            # Random α per sample
            if alpha_noise > 0:
                # Mix of α=1 (good goal) and random α
                alpha_vals = torch.ones(B, 1, device=dev)
                corrupt_mask = torch.rand(B) < corrupt_prob
                if corrupt_mask.any():
                    n_corrupt = corrupt_mask.sum().item()
                    # Corrupted samples: random α from Beta(1, 3) → biased toward 0
                    alpha_vals[corrupt_mask] = torch.from_numpy(
                        np.random.beta(1, 3, size=(n_corrupt, 1)).astype(np.float32)).to(dev)
                    # Also corrupt their candidates
                    c[corrupt_mask] = torch.randn(n_corrupt, c.size(1), 2, device=dev) * 20.0
                force_a = alpha_vals
            else:
                force_a = None  # Pure V6 mode

            opt.zero_grad()
            with autocast('cuda', enabled=True):
                deltas, gl, wp, alpha, _ = model(
                    em, h, c, cp, target_goal_idx=ti,
                    use_gt_goal=(not bool(corrupt_prob > 0 and corrupt_mask.any())),
                    force_alpha=force_a)
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
            alpha_sum += (force_a.mean().item() if force_a is not None else 1.0)

        va, vf, vg = validate(model, vl, dev); sched.step(va)
        imp = ' *BEST*' if _save(va, vf, ep) else ''
        print(f'  [S1] {ep+1}/{s1e} loss={tl_sum/nb:.4f} ADE={va:.0f}m '
              f'train_α={alpha_sum/nb:.3f} val_α={vg:.3f} corrupt={corrupt_prob:.2f}{imp}',
              flush=True)
        if no_imp >= args.patience:
            print(f'  Early stopping Stage 1', flush=True)
            break

    # Reset for Stage 2
    best_s1 = best
    best = float('inf')
    no_imp = 0

    # ==================== STAGE 2: Gate supervision ====================
    print(f'\n=== Stage 2: Gate Supervision ({s2e} ep) ===', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    for ep in range(s2e):
        model.train()
        tl_sum, nb, gate_sum, gt_sum = 0, 0, 0, 0

        for batch in tl:
            h = batch['history'].to(dev); fd = batch['future'].to(dev)
            em = batch['env_map'].to(dev); c = batch['candidates'].to(dev)
            ti = batch['target_goal_idx'].to(dev)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=dev)
            B = h.size(0)

            # 50% corrupt candidates
            corrupt = (torch.rand(1).item() > 0.5)
            if corrupt:
                c_input = torch.randn(B, c.size(1), 2, device=dev) * 20.0
                gate_tgt = torch.zeros(B, 1, device=dev)
                use_gt = False
            else:
                c_input = c
                gate_tgt = torch.ones(B, 1, device=dev)
                use_gt = True

            opt.zero_grad()
            with autocast('cuda', enabled=True):
                # Let the model predict its own α (no force_alpha)
                deltas, gl, wp, alpha, alpha_logit = model(
                    em, h, c_input, cp, target_goal_idx=ti, use_gt_goal=use_gt)
                pred_pos = torch.cumsum(deltas, dim=1)
                loss = F.mse_loss(pred_pos, gt)
                loss += 0.5 * bce_logit(alpha_logit, gate_tgt)
                if not corrupt:
                    loss += cls_c(gl, ti)
                    if wp is not None:
                        wi = torch.tensor(model.waypoint_indices, device=dev, dtype=torch.long)
                        loss += 0.5 * F.mse_loss(wp, gt.index_select(1, wi))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl_sum += loss.item(); nb += 1
            gate_sum += alpha.detach().mean().item()
            gt_sum += gate_tgt.mean().item()

        va, vf, vg = validate(model, vl, dev); sched.step(va)
        imp = ' *BEST*' if _save(va, vf, s1e + ep) else ''
        print(f'  [S2] {ep+1}/{s2e} loss={tl_sum/nb:.4f} ADE={va:.0f}m '
              f'gate={gate_sum/nb:.3f}(tgt={gt_sum/nb:.2f}) val_α={vg:.3f}{imp}',
              flush=True)
        if no_imp >= args.patience:
            print(f'  Early stopping Stage 2', flush=True)
            break

    print(f'\nDone. S1 best={best_s1:.0f}m, S2 best={best:.0f}m -> {sd_path}/{name}_best.pth',
          flush=True)


if __name__ == '__main__':
    main()
