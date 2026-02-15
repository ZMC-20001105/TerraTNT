#!/usr/bin/env python3
"""
一次性运行所有剩余实验:
1. 为Donbas创建FAS splits
2. 跨区域评估: BF模型 → Donbas测试集
3. 候选目标K敏感性实验
4. 观测长度敏感性实验

用法:
  python scripts/run_remaining_experiments.py
"""
import sys
import os
import json
import pickle
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

# ============================================================
#  1. 为Donbas创建FAS splits
# ============================================================

def create_fas_splits(region: str, seed: int = 42):
    """为指定区域创建 FAS splits (70/15/15)"""
    data_dir = PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / region
    out_dir = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pkl_files = sorted([f.name for f in data_dir.glob('*.pkl')])
    if not pkl_files:
        print(f"  ⚠️ {region}: 无pkl文件")
        return None
    
    # 按轨迹ID分组
    traj_groups = defaultdict(list)
    for f in pkl_files:
        # traj_000001_intent1_type1.pkl → traj_000001
        parts = f.split('_')
        traj_id = '_'.join(parts[:2])  # traj_XXXXXX
        traj_groups[traj_id].append(f)
    
    traj_ids = sorted(traj_groups.keys())
    random.seed(seed)
    random.shuffle(traj_ids)
    
    n = len(traj_ids)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_ids = traj_ids[:n_train]
    val_ids = traj_ids[n_train:n_train + n_val]
    test_ids = traj_ids[n_train + n_val:]
    
    def collect_files(ids):
        files = []
        for tid in ids:
            files.extend(traj_groups[tid])
        return sorted(files)
    
    def count_samples(file_list):
        total = 0
        for f in file_list:
            try:
                with open(data_dir / f, 'rb') as fp:
                    d = pickle.load(fp)
                    if isinstance(d, dict) and 'samples' in d:
                        total += len(d['samples'])
                    elif isinstance(d, list):
                        total += len(d)
                    else:
                        total += 1
            except:
                total += 1
        return total
    
    fas1_files = collect_files(train_ids)
    fas2_files = collect_files(val_ids)
    fas3_files = collect_files(test_ids)
    
    splits = {
        'fas1': {
            'description': f'{region} training set (70%)',
            'files': fas1_files,
            'num_samples': count_samples(fas1_files),
        },
        'fas2': {
            'description': f'{region} validation set (15%)',
            'files': fas2_files,
            'num_samples': count_samples(fas2_files),
        },
        'fas3': {
            'description': f'{region} test set (15%)',
            'files': fas3_files,
            'num_samples': count_samples(fas3_files),
        },
        'metadata': {
            'region': region,
            'total_trajectories': n,
            'total_samples': count_samples(pkl_files),
            'split_seed': seed,
            'data_dir': str(data_dir),
        }
    }
    
    out_file = out_dir / 'fas_splits.json'
    with open(out_file, 'w') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ {region} FAS splits: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test trajectories")
    print(f"     Samples: {splits['fas1']['num_samples']} / {splits['fas2']['num_samples']} / {splits['fas3']['num_samples']}")
    return out_file


# ============================================================
#  2. 跨区域评估
# ============================================================

def run_cross_region_evaluation(test_region: str):
    """用BF训练的模型在其他区域上评估"""
    print(f"\n{'='*60}")
    print(f"跨区域评估: bohemian_forest → {test_region}")
    print(f"{'='*60}")
    
    fas_file = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / test_region / 'fas_splits.json'
    if not fas_file.exists():
        print(f"  ⚠️ FAS splits不存在: {fas_file}")
        return
    
    traj_dir = PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / test_region
    output_dir = PROJECT_ROOT / 'outputs' / 'evaluation' / f'cross_bohemian_forest_to_{test_region}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 调用 evaluate_phases_v2.py
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'evaluate_phases_v2.py'),
        '--phases', 'P1a', 'P3a',
        '--traj_dir', str(traj_dir),
        '--fas_split_file', str(fas_file),
        '--output_dir', str(output_dir),
        '--batch_size', '32',
    ]
    
    print(f"  命令: {' '.join(cmd)}")
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode == 0:
        print(f"  ✅ 跨区域评估完成: {output_dir}")
    else:
        print(f"  ❌ 评估失败:")
        print(result.stderr[-500:] if result.stderr else "无错误输出")
    
    return output_dir


# ============================================================
#  3. 候选目标K敏感性实验 (复用PhaseV2Dataset预处理)
# ============================================================

def run_k_sensitivity():
    """改变候选目标数K，评估对性能的影响。
    直接复用evaluate_phases_v2的PhaseV2Dataset确保预处理一致。"""
    print(f"\n{'='*60}")
    print(f"候选目标K敏感性实验")
    print(f"{'='*60}")
    
    from scripts.evaluate_phases_v2 import (
        PhaseV2Dataset, PHASE_V2_CONFIGS, compute_metrics,
    )
    from models.terratnt import TerraTNT
    
    # 找到最佳模型
    model_candidates = [
        PROJECT_ROOT / 'runs' / 'terratnt_fas3_10s' / '20260206_203449' / 'best_model.pth',
        PROJECT_ROOT / 'runs' / 'terratnt_fas3_10s' / '20260206_111233' / 'best_model.pth',
    ]
    model_path = None
    for mp in model_candidates:
        if mp.exists():
            model_path = mp
            break
    
    if model_path is None:
        print("  ⚠️ 找不到模型checkpoint")
        return {}
    
    print(f"  模型: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    
    config = ckpt.get('config', {})
    env_coverage_km = float(config.get('env_coverage_km', ckpt.get('env_coverage_km', 140.0)))
    coord_scale = float(config.get('coord_scale', ckpt.get('coord_scale', 1.0)))
    paper_decoder = str(config.get('paper_decoder', ckpt.get('paper_decoder', 'hierarchical')))
    history_feature_mode = str(config.get('history_feature_mode', ckpt.get('history_feature_mode', 'full')))
    
    hist_dim_map = {'xy': 2, 'kin10': 10, 'full': 26}
    history_input_dim = hist_dim_map.get(history_feature_mode, 26)
    
    model = TerraTNT(
        env_channels=18,
        history_input_dim=history_input_dim,
        num_candidates=200,
        output_length=360,
        paper_mode=True,
        paper_decoder=paper_decoder,
        use_dual_scale=False,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    # 使用Phase P1a配置 (精确终点先验, include_gt=True)
    fas_file = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / 'bohemian_forest' / 'fas_splits.json'
    traj_dir = PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / 'bohemian_forest'
    phase_cfg = PHASE_V2_CONFIGS['P1a']
    
    K_values = [6, 10, 20, 50, 100, 200]
    results = {}
    
    for K in K_values:
        print(f"  K={K}...", end=' ', flush=True)
        
        # 为每个K值创建dataset (不同num_candidates)
        dataset = PhaseV2Dataset(
            traj_dir=str(traj_dir),
            fas_split_file=str(fas_file),
            phase_config=phase_cfg,
            num_candidates=K,
            env_coverage_km=env_coverage_km,
            coord_scale=coord_scale,
            sample_fraction=min(1.0, 500.0 / 10000),  # ~500样本
        )
        
        if len(dataset) == 0:
            print("无样本，跳过")
            continue
        
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        
        all_ade = []
        all_fde = []
        
        for batch in tqdm(loader, desc=f'K={K}', leave=False):
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            env_map = batch['env_map'].to(device)
            candidates = batch['candidates'].to(device)
            
            B = history.size(0)
            gt_pos = torch.cumsum(future, dim=1)  # (B, T, 2)
            
            with torch.no_grad():
                current_pos = torch.zeros(B, 2, device=device)
                pred_delta, goal_logits = model(
                    env_map, history, candidates, current_pos,
                    teacher_forcing_ratio=0.0,
                    use_gt_goal=False,
                )
                pred_pos = torch.cumsum(pred_delta, dim=1)
            
            for b in range(B):
                m = compute_metrics(pred_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy())
                all_ade.append(m['ade'])
                all_fde.append(m['fde'])
        
        avg_ade = np.mean(all_ade)
        avg_fde = np.mean(all_fde)
        results[str(K)] = {
            'ade': avg_ade, 'fde': avg_fde,
            'n_samples': len(all_ade),
        }
        print(f"ADE={avg_ade:.0f}m, FDE={avg_fde:.0f}m (n={len(all_ade)})")
        del dataset, loader
    
    out_dir = PROJECT_ROOT / 'outputs' / 'evaluation' / 'control_variables'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'candidate_k_sensitivity.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ K敏感性结果保存: {out_file}")
    return results


# ============================================================
#  4. 观测长度敏感性实验 (复用PhaseV2Dataset预处理)
# ============================================================

def run_observation_length_sensitivity():
    """改变观测长度，评估对性能的影响。
    使用PhaseV2Dataset确保预处理一致，在batch级别截断历史。"""
    print(f"\n{'='*60}")
    print(f"观测长度敏感性实验")
    print(f"{'='*60}")
    
    from scripts.evaluate_phases_v2 import (
        PhaseV2Dataset, PHASE_V2_CONFIGS, compute_metrics,
    )
    from models.terratnt import TerraTNT
    
    # 找到最佳模型
    model_candidates = [
        PROJECT_ROOT / 'runs' / 'terratnt_fas3_10s' / '20260206_203449' / 'best_model.pth',
        PROJECT_ROOT / 'runs' / 'terratnt_fas3_10s' / '20260206_111233' / 'best_model.pth',
    ]
    model_path = None
    for mp in model_candidates:
        if mp.exists():
            model_path = mp
            break
    
    if model_path is None:
        print("  ⚠️ 找不到模型checkpoint")
        return {}
    
    print(f"  模型: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    
    config = ckpt.get('config', {})
    env_coverage_km = float(config.get('env_coverage_km', ckpt.get('env_coverage_km', 140.0)))
    coord_scale = float(config.get('coord_scale', ckpt.get('coord_scale', 1.0)))
    paper_decoder = str(config.get('paper_decoder', ckpt.get('paper_decoder', 'hierarchical')))
    history_feature_mode = str(config.get('history_feature_mode', ckpt.get('history_feature_mode', 'full')))
    num_candidates = int(config.get('num_candidates', ckpt.get('num_candidates', 6)))
    
    hist_dim_map = {'xy': 2, 'kin10': 10, 'full': 26}
    history_input_dim = hist_dim_map.get(history_feature_mode, 26)
    
    model = TerraTNT(
        env_channels=18,
        history_input_dim=history_input_dim,
        num_candidates=num_candidates,
        output_length=360,
        paper_mode=True,
        paper_decoder=paper_decoder,
        use_dual_scale=False,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    # 使用Phase P1a (精确终点先验)
    fas_file = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / 'bohemian_forest' / 'fas_splits.json'
    traj_dir = PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / 'bohemian_forest'
    phase_cfg = PHASE_V2_CONFIGS['P1a']
    
    # 创建一次dataset (用满观测长度)
    dataset = PhaseV2Dataset(
        traj_dir=str(traj_dir),
        fas_split_file=str(fas_file),
        phase_config=phase_cfg,
        num_candidates=num_candidates,
        env_coverage_km=env_coverage_km,
        coord_scale=coord_scale,
        sample_fraction=min(1.0, 500.0 / 10000),
    )
    
    if len(dataset) == 0:
        print("  ⚠️ 无测试样本")
        return {}
    
    print(f"  测试样本数: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 不同观测长度 (原始90步=15分钟, 每步10秒)
    obs_lengths = [18, 30, 42, 54, 66, 78, 90]  # 3/5/7/9/11/13/15 分钟
    results = {}
    
    # 预加载所有batch数据 (避免重复IO)
    all_batches = []
    for batch in loader:
        all_batches.append(batch)
    
    for obs_len in obs_lengths:
        minutes = obs_len * 10 / 60
        print(f"  观测长度={obs_len}步 ({minutes:.0f}min)...", end=' ', flush=True)
        all_ade = []
        all_fde = []
        
        for batch in tqdm(all_batches, desc=f'obs={obs_len}', leave=False):
            history = batch['history'].to(device)  # (B, 90, D)
            future = batch['future'].to(device)
            env_map = batch['env_map'].to(device)
            candidates = batch['candidates'].to(device)
            goal_gt = batch['goal'].to(device)
            
            B = history.size(0)
            gt_pos = torch.cumsum(future, dim=1)
            
            # 截断历史: 将前面的步骤置零
            if obs_len < 90:
                history_masked = history.clone()
                history_masked[:, :90 - obs_len, :] = 0.0
            else:
                history_masked = history
            
            with torch.no_grad():
                current_pos = torch.zeros(B, 2, device=device)
                pred_delta, goal_logits = model(
                    env_map, history_masked, candidates, current_pos,
                    teacher_forcing_ratio=0.0,
                    use_gt_goal=False,
                )
                pred_pos = torch.cumsum(pred_delta, dim=1)
            
            for b in range(B):
                m = compute_metrics(pred_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy())
                all_ade.append(m['ade'])
                all_fde.append(m['fde'])
        
        avg_ade = np.mean(all_ade)
        avg_fde = np.mean(all_fde)
        results[str(obs_len)] = {
            'obs_steps': obs_len,
            'obs_minutes': minutes,
            'ade': avg_ade,
            'fde': avg_fde,
            'n_samples': len(all_ade),
        }
        print(f"ADE={avg_ade:.0f}m, FDE={avg_fde:.0f}m (n={len(all_ade)})")
    
    out_dir = PROJECT_ROOT / 'outputs' / 'evaluation' / 'control_variables'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'observation_length_sensitivity.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ 观测长度敏感性结果保存: {out_file}")
    return results


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-cross', action='store_true', help='跳过跨区域评估')
    parser.add_argument('--skip-k', action='store_true', help='跳过K敏感性')
    parser.add_argument('--skip-obs', action='store_true', help='跳过观测长度')
    args = parser.parse_args()
    
    print("=" * 60)
    print("剩余实验一键运行")
    print("=" * 60)
    
    # Step 1: 创建FAS splits
    print("\n[1/4] 创建FAS splits...")
    for region in ['donbas']:
        fas_dir = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region
        if (fas_dir / 'fas_splits.json').exists():
            print(f"  {region}: 已存在，跳过")
        else:
            create_fas_splits(region)
    
    # Step 2: 跨区域评估
    if not args.skip_cross:
        print("\n[2/4] 跨区域评估...")
        for region in ['donbas']:
            out_dir = PROJECT_ROOT / 'outputs' / 'evaluation' / f'cross_bohemian_forest_to_{region}'
            existing = out_dir / 'phase_v2_results.json'
            if existing.exists() and existing.stat().st_size > 10:
                print(f"  {region}: 结果已存在，跳过")
            else:
                run_cross_region_evaluation(region)
    else:
        print("\n[2/4] 跨区域评估: 跳过")
    
    # Step 3: K敏感性
    if not args.skip_k:
        print("\n[3/4] 候选目标K敏感性...")
        k_file = PROJECT_ROOT / 'outputs' / 'evaluation' / 'control_variables' / 'candidate_k_sensitivity.json'
        if k_file.exists():
            print(f"  结果已存在，跳过")
        else:
            run_k_sensitivity()
    else:
        print("\n[3/4] K敏感性: 跳过")
    
    # Step 4: 观测长度
    if not args.skip_obs:
        print("\n[4/4] 观测长度敏感性...")
        obs_file = PROJECT_ROOT / 'outputs' / 'evaluation' / 'control_variables' / 'observation_length_sensitivity.json'
        if obs_file.exists():
            print(f"  结果已存在，跳过")
        else:
            run_observation_length_sensitivity()
    else:
        print("\n[4/4] 观测长度: 跳过")
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
