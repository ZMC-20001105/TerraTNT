#!/usr/bin/env python3
"""
Waypoint数量消融实验：训练 wp=2,4,6,8 的V6R变体并评估。
wp=10 已有 (runs/incremental_models_v6r/V6_best.pth)。

用法:
  python scripts/train_waypoint_ablation.py
  python scripts/train_waypoint_ablation.py --wp_list 2,4
  python scripts/train_waypoint_ablation.py --eval_only
"""

import sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.train_incremental_models import (
    TerraTNTAutoregV6, FASDataset, train_v5, ade_fde_m,
    HISTORY_LEN, FUTURE_LEN,
)
from scripts.evaluate_phases_v2 import PhaseV2Dataset, PHASE_V2_CONFIGS, compute_metrics


def train_wp_variant(num_wp, device, args):
    """训练指定waypoint数量的V6R变体"""
    save_dir = PROJECT_ROOT / 'runs' / 'waypoint_ablation'
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name = f'V6R_wp{num_wp}'
    ckpt_path = save_dir / f'{model_name}_best.pth'

    if ckpt_path.exists() and not args.force_retrain:
        print(f'\n  {model_name}: 已有checkpoint，跳过训练')
        return ckpt_path

    print(f'\n{"="*60}')
    print(f'训练 {model_name} (num_waypoints={num_wp})')
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

    # 从wp=10的V6R初始化（迁移学习）
    base_ckpt = PROJECT_ROOT / 'runs/incremental_models_v6r/V6_best.pth'

    model = TerraTNTAutoregV6(
        history_dim=26, hidden_dim=128, env_channels=18,
        env_feature_dim=128, decoder_hidden_dim=256,
        future_len=FUTURE_LEN, num_waypoints=num_wp,
        num_candidates=6, env_coverage_km=140.0,
    ).to(device)

    # 加载兼容的权重（跳过waypoint相关的不兼容层）
    if base_ckpt.exists():
        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        # 过滤掉形状不兼容的参数
        model_sd = model.state_dict()
        compatible = {}
        skipped = []
        for k, v in sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                compatible[k] = v
            else:
                skipped.append(k)
        model.load_state_dict(compatible, strict=False)
        if skipped:
            print(f'  迁移学习: 跳过 {len(skipped)} 个不兼容参数: {skipped[:5]}...')
        print(f'  加载了 {len(compatible)}/{len(sd)} 个参数')

    best_ade = train_v5(
        model, model_name, train_loader, val_loader, device,
        num_epochs=args.num_epochs, lr=args.lr, patience=args.patience,
        save_dir=str(save_dir), wp_weight=0.5, cls_weight=1.0,
        use_cosine_lr=True, goal_dropout_prob=0.15,
    )
    print(f'  {model_name}: best_val_ADE = {best_ade:.0f}m')
    return ckpt_path


@torch.no_grad()
def evaluate_wp_variant(num_wp, ckpt_path, device, phase='P1a'):
    """评估指定waypoint数量的V6R变体"""
    model = TerraTNTAutoregV6(
        history_dim=26, hidden_dim=128, env_channels=18,
        env_feature_dim=128, decoder_hidden_dim=256,
        future_len=FUTURE_LEN, num_waypoints=num_wp,
        num_candidates=6, env_coverage_km=140.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
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
        target_idx = batch['target_goal_idx'].to(device)

        gt_pos = torch.cumsum(future, dim=1)
        current_pos = torch.zeros(history.size(0), 2, device=device)

        with autocast('cuda', enabled=True):
            pred_pos, goal_logits = model(
                env_map, history, candidates, current_pos,
                teacher_forcing_ratio=0.0, use_gt_goal=False,
            )

        for b in range(history.size(0)):
            m = compute_metrics(pred_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy())
            all_metrics.append(m)

    ades = [m['ade'] for m in all_metrics]
    fdes = [m['fde'] for m in all_metrics]
    return {
        'ade_mean': float(np.mean(ades)),
        'ade_std': float(np.std(ades)),
        'fde_mean': float(np.mean(fdes)),
        'fde_std': float(np.std(fdes)),
        'n_samples': len(all_metrics),
    }


def main():
    parser = argparse.ArgumentParser(description='Waypoint数量消融实验')
    parser.add_argument('--wp_list', default='2,4,6,8',
                        help='要训练的waypoint数量列表(逗号分隔)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--force_retrain', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    wp_list = [int(x) for x in args.wp_list.split(',')]
    save_dir = PROJECT_ROOT / 'runs' / 'waypoint_ablation'
    save_dir.mkdir(parents=True, exist_ok=True)

    # wp=10 已有
    all_wp = sorted(set(wp_list + [10]))

    # 训练
    if not args.eval_only:
        for num_wp in wp_list:
            train_wp_variant(num_wp, device, args)

    # 评估
    print(f'\n{"="*60}')
    print('Waypoint数量消融评估 (Phase P1a)')
    print(f'{"="*60}')

    results = {}
    for num_wp in all_wp:
        if num_wp == 10:
            ckpt = PROJECT_ROOT / 'runs/incremental_models_v6r/V6_best.pth'
        else:
            ckpt = save_dir / f'V6R_wp{num_wp}_best.pth'

        if not ckpt.exists():
            print(f'  wp={num_wp}: 无checkpoint，跳过')
            continue

        print(f'\n  评估 wp={num_wp}...')
        r = evaluate_wp_variant(num_wp, ckpt, device)
        results[str(num_wp)] = r
        print(f'  wp={num_wp}: ADE={r["ade_mean"]/1000:.2f}km  FDE={r["fde_mean"]/1000:.2f}km')

    # 保存结果
    output_path = PROJECT_ROOT / 'outputs/evaluation/control_variables/waypoint_ablation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f'\n{"="*60}')
    print('表4.11: Waypoint数量的影响')
    print(f'{"="*60}')
    print(f'{"Waypoint数":>10} {"ADE(km)":>10} {"FDE(km)":>10}')
    print('-' * 32)
    for wp in sorted(results.keys(), key=int):
        r = results[wp]
        print(f'  {wp:>8} {r["ade_mean"]/1000:>9.2f} {r["fde_mean"]/1000:>9.2f}')

    print(f'\n结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
