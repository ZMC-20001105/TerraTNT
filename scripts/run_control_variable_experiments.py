#!/usr/bin/env python3
"""
控制变量实验脚本
论文表4.9: 候选数量K对性能影响
论文表4.10: 观测时长对性能影响
论文图4.9: Phase3候选敏感性

只评估V6R_Robust（本文模型），在Phase1a条件下。
"""

import sys
import os
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.evaluate_phases_v2 import (
    PhaseV2Dataset, PHASE_V2_CONFIGS, compute_metrics,
    generate_prior_heatmap_phase1, sample_candidates_from_heatmap,
    extract_goal_from_heatmap,
)


def load_v6r_model(device):
    """加载V6R_Robust模型"""
    from scripts.train_incremental_models import TerraTNTAutoregV6
    runs = PROJECT_ROOT / 'runs'
    ckpt_path = runs / 'incremental_models_v6r' / 'V6_best.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    model_cfg = ckpt.get('config', {})

    h_key = 'history_encoder.lstm.weight_ih_l0'
    inferred_hidden = sd[h_key].shape[0] // 4 if h_key in sd else 128

    model = TerraTNTAutoregV6(
        hidden_dim=inferred_hidden,
        future_len=360,
        num_waypoints=10,
        env_coverage_km=140.0,
        num_candidates=model_cfg.get('num_candidates', 6),
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_v6r(model, dataset, device, batch_size=16):
    """评估V6R模型，返回per-sample metrics"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    all_metrics = []

    for batch in tqdm(loader, desc="评估", leave=False):
        history = batch['history'].to(device)
        future = batch['future'].to(device)
        env_map = batch['env_map'].to(device)
        candidates = batch['candidates'].to(device)
        target_idx = batch['target_goal_idx'].to(device)

        B = history.size(0)
        gt_pos = torch.cumsum(future, dim=1)

        with autocast('cuda', enabled=True):
            current_pos = torch.zeros(B, 2, device=device)
            pred_pos, goal_logits = model(
                env_map, history, candidates, current_pos,
                teacher_forcing_ratio=0.0,
                target_goal_idx=target_idx,
                use_gt_goal=False,
            )

        # 计算goal accuracy
        pred_goal_idx = goal_logits.argmax(dim=1)
        goal_correct = (pred_goal_idx == target_idx).float()

        for b in range(B):
            m = compute_metrics(pred_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy())
            m['goal_correct'] = goal_correct[b].item()
            all_metrics.append(m)

    return all_metrics


# ============================================================
#  实验1: 候选数量K (表4.9)
# ============================================================

def experiment_candidate_K(device, model):
    """测试不同候选数量K对性能的影响"""
    K_values = [3, 5, 10, 20]
    results = {}

    for K in K_values:
        print(f"\n--- K={K} ---")
        ds = PhaseV2Dataset(
            traj_dir=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo'),
            fas_split_file=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo' / 'fas_splits_full_phases.json'),
            phase_config=PHASE_V2_CONFIGS['P1a'],
            num_candidates=K,
            seed=42,
        )
        print(f"  样本数: {len(ds)}")

        metrics = evaluate_v6r(model, ds, device)
        ades = [m['ade'] for m in metrics]
        fdes = [m['fde'] for m in metrics]
        ep1 = np.mean([m['goal_correct'] for m in metrics]) * 100

        results[K] = {
            'ade_mean': float(np.mean(ades)),
            'ade_std': float(np.std(ades)),
            'fde_mean': float(np.mean(fdes)),
            'fde_std': float(np.std(fdes)),
            'ep1': float(ep1),
            'n_samples': len(metrics),
        }
        print(f"  ADE={np.mean(ades)/1000:.2f}km  FDE={np.mean(fdes)/1000:.2f}km  EP@1={ep1:.1f}%")

    return results


# ============================================================
#  实验2: 观测时长 (表4.10)
# ============================================================

class TruncatedHistoryDataset(Dataset):
    """包装PhaseV2Dataset，截断历史轨迹到指定长度"""

    def __init__(self, base_dataset, obs_steps):
        self.base = base_dataset
        self.obs_steps = obs_steps

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        H = self.obs_steps
        full_history = sample['history']  # (90, 26)

        if H < full_history.size(0):
            # 取最后H步（最近的观测）
            truncated = full_history[-H:]
            # 用零填充到原始长度（前面补零）
            padded = torch.zeros_like(full_history)
            padded[-H:] = truncated
            sample['history'] = padded

        return sample


def experiment_observation_length(device, model):
    """测试不同观测时长对性能的影响"""
    # 3min=18步, 6min=36步, 9min=54步, 12min=72步, 15min=90步
    obs_configs = [
        (3, 18),
        (6, 36),
        (9, 54),
        (12, 72),
        (15, 90),
    ]
    results = {}

    base_ds = PhaseV2Dataset(
        traj_dir=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo'),
        fas_split_file=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo' / 'fas_splits_full_phases.json'),
        phase_config=PHASE_V2_CONFIGS['P1a'],
        seed=42,
    )

    for minutes, steps in obs_configs:
        print(f"\n--- 观测时长={minutes}min ({steps}步) ---")
        ds = TruncatedHistoryDataset(base_ds, steps)

        metrics = evaluate_v6r(model, ds, device)
        ades = [m['ade'] for m in metrics]
        fdes = [m['fde'] for m in metrics]

        results[minutes] = {
            'steps': steps,
            'ade_mean': float(np.mean(ades)),
            'ade_std': float(np.std(ades)),
            'fde_mean': float(np.mean(fdes)),
            'fde_std': float(np.std(fdes)),
            'n_samples': len(metrics),
        }
        print(f"  ADE={np.mean(ades)/1000:.2f}km  FDE={np.mean(fdes)/1000:.2f}km")

    return results


# ============================================================
#  实验3: Phase3候选敏感性 (图4.9)
# ============================================================

def experiment_phase3_sensitivity(device, model):
    """Phase3条件下，不同候选数量和候选范围的敏感性"""
    # 候选数量敏感性
    K_values = [3, 5, 10, 20, 50]
    k_results = {}

    for K in K_values:
        print(f"\n--- Phase3 K={K} ---")
        ds = PhaseV2Dataset(
            traj_dir=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo'),
            fas_split_file=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo' / 'fas_splits_full_phases.json'),
            phase_config=PHASE_V2_CONFIGS['P3a'],
            num_candidates=K,
            seed=42,
        )
        print(f"  样本数: {len(ds)}")

        metrics = evaluate_v6r(model, ds, device)
        ades = [m['ade'] for m in metrics]
        fdes = [m['fde'] for m in metrics]

        k_results[K] = {
            'ade_mean': float(np.mean(ades)),
            'fde_mean': float(np.mean(fdes)),
            'n_samples': len(metrics),
        }
        print(f"  ADE={np.mean(ades)/1000:.2f}km  FDE={np.mean(fdes)/1000:.2f}km")

    return {'candidate_K': k_results}


def main():
    parser = argparse.ArgumentParser(description='控制变量实验')
    parser.add_argument('--experiments', nargs='+',
                        default=['K', 'obs', 'phase3_sens'],
                        choices=['K', 'obs', 'phase3_sens'],
                        help='要运行的实验')
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'outputs' / 'evaluation' / 'control_variables'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")

    # 加载模型
    print("\n加载V6R_Robust模型...")
    model = load_v6r_model(device)
    print("模型加载完成")

    all_results = {}

    if 'K' in args.experiments:
        print("\n" + "=" * 60)
        print("实验1: 候选数量K对性能影响 (表4.9)")
        print("=" * 60)
        all_results['candidate_K'] = experiment_candidate_K(device, model)

    if 'obs' in args.experiments:
        print("\n" + "=" * 60)
        print("实验2: 观测时长对性能影响 (表4.10)")
        print("=" * 60)
        all_results['observation_length'] = experiment_observation_length(device, model)

    if 'phase3_sens' in args.experiments:
        print("\n" + "=" * 60)
        print("实验3: Phase3候选敏感性 (图4.9)")
        print("=" * 60)
        all_results['phase3_sensitivity'] = experiment_phase3_sensitivity(device, model)

    # 保存结果
    with open(output_dir / 'control_variable_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("\n" + "=" * 60)
    print("控制变量实验结果汇总")
    print("=" * 60)

    if 'candidate_K' in all_results:
        print("\n表4.9: 候选数量K")
        print(f"{'K':>5} {'ADE(km)':>10} {'FDE(km)':>10} {'EP@1(%)':>10}")
        print("-" * 37)
        for k, r in sorted(all_results['candidate_K'].items(), key=lambda x: int(x[0])):
            print(f"{k:>5} {r['ade_mean']/1000:>9.2f} {r['fde_mean']/1000:>9.2f} {r.get('ep1', 0):>9.1f}")

    if 'observation_length' in all_results:
        print("\n表4.10: 观测时长")
        print(f"{'分钟':>5} {'步数':>5} {'ADE(km)':>10} {'FDE(km)':>10}")
        print("-" * 32)
        for mins, r in sorted(all_results['observation_length'].items(), key=lambda x: int(x[0])):
            print(f"{mins:>5} {r['steps']:>5} {r['ade_mean']/1000:>9.2f} {r['fde_mean']/1000:>9.2f}")

    if 'phase3_sensitivity' in all_results:
        print("\n图4.9: Phase3候选敏感性")
        sens = all_results['phase3_sensitivity']
        if 'candidate_K' in sens:
            print(f"{'K':>5} {'ADE(km)':>10} {'FDE(km)':>10}")
            for k, r in sorted(sens['candidate_K'].items(), key=lambda x: int(x[0])):
                print(f"{k:>5} {r['ade_mean']/1000:>9.2f} {r['fde_mean']/1000:>9.2f}")

    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
