#!/usr/bin/env python3
"""Train V8 DecoupledGoalV8: from scratch with curriculum random α.

Training strategy:
  - Train from scratch (no V6 weight transfer, fusion architecture is different)
  - Curriculum: start with α=1 (pure goal-aware), gradually increase randomness
  - Each batch: some samples get α=1 (good goal), some get random α∈[0,1]
  - Also corrupt candidates for low-α samples to teach the model
  - After main training, fine-tune gate with explicit good/bad candidate supervision
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
    pa.add_argument('--num_epochs', type=int, default=50)
    pa.add_argument('--lr', type=float, default=1e-3)
    pa.add_argument('--patience', type=int, default=12)
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

    # Model (from scratch)
    model = DecoupledGoalV8(history_dim=26, hidden_dim=128, env_channels=18,
                            env_feature_dim=128, decoder_hidden_dim=256,
                            future_len=FUTURE_LEN, num_waypoints=10,
                            num_candidates=6, env_coverage_km=140.0).to(dev)
    print(f'V8d params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

    name = 'V8d'
    scaler = GradScaler('cuda')
    cls_c = nn.CrossEntropyLoss()
    bce_logit = nn.BCEWithLogitsLoss()
    NE = args.num_epochs
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

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NE, eta_min=args.lr * 0.01)

    for ep in range(NE):
        model.train()
        tl_sum, nb, alpha_sum = 0, 0, 0

        # Curriculum: corruption probability ramps up over first 40% of training
        progress = ep / max(1, NE - 1)
        corrupt_prob = min(0.4, 0.4 * progress / 0.4) if progress < 0.4 else 0.4

        for batch in tl:
            h = batch['history'].to(dev); fd = batch['future'].to(dev)
            em = batch['env_map'].to(dev); c = batch['candidates'].to(dev)
            ti = batch['target_goal_idx'].to(dev)
            gt = torch.cumsum(fd, dim=1)
            cp = torch.zeros(h.size(0), 2, device=dev)
            B = h.size(0)

            # Per-sample α: mix of α=1 and random low α
            alpha_vals = torch.ones(B, 1, device=dev)
            corrupt_mask = torch.rand(B) < corrupt_prob
            n_corrupt = corrupt_mask.sum().item()
            if n_corrupt > 0:
                # Low α for corrupted samples (Beta(1,2) biased toward 0)
                low_alpha = torch.from_numpy(
                    np.random.beta(1, 2, size=(n_corrupt, 1)).astype(np.float32)).to(dev)
                alpha_vals[corrupt_mask] = low_alpha
                # Corrupt their candidates
                c[corrupt_mask] = torch.randn(n_corrupt, c.size(1), 2, device=dev) * 20.0

            # For non-corrupt samples, use GT goal; for corrupt, use classifier
            use_gt = True  # GT goal for all (corrupt ones have bad candidates anyway)

            opt.zero_grad()
            with autocast('cuda', enabled=True):
                deltas, gl, wp, alpha, alpha_logit = model(
                    em, h, c, cp, target_goal_idx=ti, use_gt_goal=use_gt,
                    force_alpha=alpha_vals)
                pred_pos = torch.cumsum(deltas, dim=1)

                # Trajectory loss
                loss = F.mse_loss(pred_pos, gt)

                # Classifier loss (only for non-corrupt samples)
                if n_corrupt < B:
                    good_mask = ~corrupt_mask
                    loss += cls_c(gl[good_mask], ti[good_mask])

                # Waypoint loss
                if wp is not None:
                    wi = torch.tensor(model.waypoint_indices, device=dev, dtype=torch.long)
                    loss += 0.5 * F.mse_loss(wp, gt.index_select(1, wi))

                # Gate supervision: teach gate to predict α
                gate_target = alpha_vals.detach()
                loss += 0.3 * bce_logit(alpha_logit, gate_target)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tl_sum += loss.item(); nb += 1
            alpha_sum += alpha_vals.mean().item()

        sched.step()

        # Validate with auto α and with forced α=1
        va, vf, vg = validate(model, vl, dev)
        va1, _, _ = validate(model, vl, dev, force_alpha=1.0)
        va0, _, _ = validate(model, vl, dev, force_alpha=0.0)
        imp = ' *BEST*' if _save(va, vf, ep) else ''
        print(f'  [{name}] {ep+1}/{NE} loss={tl_sum/nb:.4f} '
              f'ADE(auto)={va:.0f}m ADE(α=1)={va1:.0f}m ADE(α=0)={va0:.0f}m '
              f'train_α={alpha_sum/nb:.3f} val_α={vg:.3f} corrupt={corrupt_prob:.2f}{imp}',
              flush=True)
        if no_imp >= args.patience:
            print(f'  Early stopping at epoch {ep+1}', flush=True)
            break

    print(f'\nDone. Best ADE={best:.0f}m -> {sd_path}/{name}_best.pth', flush=True)


if __name__ == '__main__':
    main()
