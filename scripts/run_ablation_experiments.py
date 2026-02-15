#!/usr/bin/env python3
"""
消融实验脚本 (论文表4.13/4.14)

模块消融 (表4.13):
  - 完整模型 (V6R_Robust) — 已有
  - 无环境编码器 (no_env): 环境编码器输出置零
  - 无目标分类器 (no_goal_cls): 随机选择候选目标
  - MLP替代LSTM (mlp_decoder): 用MLP替代自回归LSTM解码器

环境通道消融 (表4.14):
  - 完整模型 — 已有
  - 无DEM (no_dem): 将DEM通道置零
  - 无LULC (no_lulc): 将LULC通道置零
  - 无OSM (no_osm): 将OSM通道置零

注意: 这些消融实验不需要重新训练模型！
通过在推理时修改输入或模型行为来实现消融效果。
这是标准的"inference-time ablation"方法。
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.evaluate_phases_v2 import (
    PhaseV2Dataset, PHASE_V2_CONFIGS, compute_metrics,
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


# ============================================================
#  环境通道映射 (根据数据集构建逻辑)
# ============================================================

def get_history_dim_ranges():
    """获取26维历史特征的维度分组 (来自 trajectory_generator_v2.extract_26d_history_features)"""
    # dim 0-1:  相对位置 (dx, dy) km
    # dim 2-3:  速度 (vx, vy) / 30 m/s
    # dim 4-5:  加速度 (ax, ay) / 3 m/s²
    # dim 6-7:  航向 (sin θ, cos θ)
    # dim 8:    曲率
    # dim 9:    速度模 ‖v‖ / 30
    # dim 10:   DEM高程 / 1000
    # dim 11:   坡度 / 90
    # dim 12-13: 坡向 (sin, cos)
    # dim 14-23: LULC one-hot (10类)
    # dim 24:   树木覆盖
    # dim 25:   道路
    return {
        'position':     [0, 1],
        'velocity':     [2, 3, 9],       # vx, vy, speed
        'acceleration': [4, 5],
        'heading':      [6, 7],
        'curvature':    [8],
        'kinematics':   [2, 3, 4, 5, 6, 7, 8, 9],  # 全部运动学 (不含位置)
        'traj_env':     list(range(10, 26)),          # 沿轨迹环境特征
    }


def get_channel_ranges():
    """获取环境通道范围 (来自 trajectory_generator_v2.extract_100km_env_map)"""
    # ch0: DEM (归一化高程)
    # ch1: Slope (/90)
    # ch2: Aspect sin
    # ch3: Aspect cos
    # ch4-13: LULC one-hot (10类: 10,20,30,40,50,60,80,90,100,255)
    # ch14: Tree cover
    # ch15: Road (OSM或LULC派生)
    # ch16: History heatmap
    # ch17: Goal prior map
    return {
        'dem': [0, 1, 2, 3],           # DEM + slope + aspect_sin + aspect_cos
        'lulc': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # LULC 10类 + tree_cover
        'osm': [15],                    # Road
        'goal': [17],                   # goal prior map
        'history': [16],                # history heatmap
    }


# ============================================================
#  消融评估包装器
# ============================================================

class AblationWrapper:
    """包装模型，在推理时实施消融"""

    def __init__(self, model, ablation_type='full'):
        self.model = model
        self.ablation_type = ablation_type
        self.channel_ranges = get_channel_ranges()

    def __call__(self, env_map, history, candidates, current_pos, **kwargs):
        """模拟模型forward，但根据消融类型修改输入/行为"""

        if self.ablation_type == 'full':
            return self.model(env_map, history, candidates, current_pos, **kwargs)

        elif self.ablation_type == 'no_env':
            # 将环境编码器输入置零（保留goal通道）
            env_map_ablated = env_map.clone()
            for ch_name in ['dem', 'lulc', 'osm']:
                for ch in self.channel_ranges[ch_name]:
                    if ch < env_map_ablated.size(1):
                        env_map_ablated[:, ch] = 0.0
            return self.model(env_map_ablated, history, candidates, current_pos, **kwargs)

        elif self.ablation_type == 'no_goal_cls':
            # 随机选择候选目标（不使用分类器）
            B = env_map.size(0)
            K = candidates.size(1)
            device = env_map.device
            # 运行正常forward获取pred
            pred_pos, goal_logits = self.model(
                env_map, history, candidates, current_pos, **kwargs)
            # 但用随机目标重新解码
            random_idx = torch.randint(0, K, (B,), device=device)
            random_goal = candidates[torch.arange(B, device=device), random_idx]
            pred_pos_rand, _ = self.model(
                env_map, history, candidates, current_pos,
                goal=random_goal,
                teacher_forcing_ratio=0.0,
            )
            return pred_pos_rand, goal_logits

        elif self.ablation_type == 'no_history':
            # 将历史轨迹输入置零
            history_ablated = torch.zeros_like(history)
            return self.model(env_map, history_ablated, candidates, current_pos, **kwargs)

        elif self.ablation_type == 'no_dem':
            env_map_ablated = env_map.clone()
            for ch in self.channel_ranges['dem']:
                if ch < env_map_ablated.size(1):
                    env_map_ablated[:, ch] = 0.0
            return self.model(env_map_ablated, history, candidates, current_pos, **kwargs)

        elif self.ablation_type == 'no_lulc':
            env_map_ablated = env_map.clone()
            for ch in self.channel_ranges['lulc']:
                if ch < env_map_ablated.size(1):
                    env_map_ablated[:, ch] = 0.0
            return self.model(env_map_ablated, history, candidates, current_pos, **kwargs)

        elif self.ablation_type == 'no_osm':
            env_map_ablated = env_map.clone()
            for ch in self.channel_ranges['osm']:
                if ch < env_map_ablated.size(1):
                    env_map_ablated[:, ch] = 0.0
            return self.model(env_map_ablated, history, candidates, current_pos, **kwargs)

        # --- 运动学特征消融 (修改 history 的对应维度) ---
        elif self.ablation_type.startswith('no_hist_'):
            dim_key = self.ablation_type[len('no_hist_'):]
            hist_ranges = get_history_dim_ranges()
            if dim_key not in hist_ranges:
                raise ValueError(f"Unknown history dim key: {dim_key}")
            dims = hist_ranges[dim_key]
            history_ablated = history.clone()
            for d in dims:
                if d < history_ablated.size(-1):
                    history_ablated[:, :, d] = 0.0
            return self.model(env_map, history_ablated, candidates, current_pos, **kwargs)

        else:
            raise ValueError(f"Unknown ablation type: {self.ablation_type}")


@torch.no_grad()
def evaluate_ablation(wrapper, dataset, device, batch_size=16):
    """评估消融模型"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    all_metrics = []

    for batch in tqdm(loader, desc=f"评估 {wrapper.ablation_type}", leave=False):
        history = batch['history'].to(device)
        future = batch['future'].to(device)
        env_map = batch['env_map'].to(device)
        candidates = batch['candidates'].to(device)
        target_idx = batch['target_goal_idx'].to(device)

        B = history.size(0)
        gt_pos = torch.cumsum(future, dim=1)

        with autocast('cuda', enabled=True):
            current_pos = torch.zeros(B, 2, device=device)
            pred_pos, goal_logits = wrapper(
                env_map, history, candidates, current_pos,
                teacher_forcing_ratio=0.0,
                target_goal_idx=target_idx,
                use_gt_goal=False,
            )

        for b in range(B):
            m = compute_metrics(pred_pos[b].cpu().numpy(), gt_pos[b].cpu().numpy())
            all_metrics.append(m)

    return all_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='消融实验')
    parser.add_argument('--phase', default='P1a', help='评估Phase')
    parser.add_argument('--output_dir', default=str(PROJECT_ROOT / 'outputs' / 'evaluation' / 'ablation'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"设备: {device}")
    print(f"评估Phase: {args.phase}")

    # 加载模型
    print("\n加载V6R_Robust模型...")
    model = load_v6r_model(device)

    # 加载数据集
    ds = PhaseV2Dataset(
        traj_dir=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo'),
        fas_split_file=str(PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo' / 'fas_splits_full_phases.json'),
        phase_config=PHASE_V2_CONFIGS[args.phase],
        seed=42,
    )
    print(f"数据集大小: {len(ds)}")

    # ============================================================
    #  模块消融 (表4.13)
    # ============================================================
    print("\n" + "=" * 60)
    print("表4.13: 模块消融实验")
    print("=" * 60)

    module_ablations = {
        '完整模型': 'full',
        '无环境编码器': 'no_env',
        '无历史编码器': 'no_history',
        '无目标分类器': 'no_goal_cls',
    }

    module_results = {}
    for name, abl_type in module_ablations.items():
        print(f"\n--- {name} ({abl_type}) ---")
        wrapper = AblationWrapper(model, abl_type)
        metrics = evaluate_ablation(wrapper, ds, device)
        ades = [m['ade'] for m in metrics]
        fdes = [m['fde'] for m in metrics]
        module_results[abl_type] = {
            'name': name,
            'ade_mean': float(np.mean(ades)),
            'ade_std': float(np.std(ades)),
            'fde_mean': float(np.mean(fdes)),
            'fde_std': float(np.std(fdes)),
            'n_samples': len(metrics),
        }
        print(f"  ADE={np.mean(ades)/1000:.2f}km  FDE={np.mean(fdes)/1000:.2f}km")

    # ============================================================
    #  运动学特征消融 (表4.15)
    # ============================================================
    print("\n" + "=" * 60)
    print("表4.15: 运动学/历史特征消融实验")
    print("=" * 60)

    kinematics_ablations = {
        '完整模型':       'full',
        '无速度':         'no_hist_velocity',
        '无加速度':       'no_hist_acceleration',
        '无航向':         'no_hist_heading',
        '无曲率':         'no_hist_curvature',
        '仅位置(无运动学)': 'no_hist_kinematics',
        '无沿轨迹环境':   'no_hist_traj_env',
    }

    kinematics_results = {}
    for name, abl_type in kinematics_ablations.items():
        if abl_type == 'full' and 'full' in module_results:
            kinematics_results[abl_type] = module_results['full'].copy()
            kinematics_results[abl_type]['name'] = name
            print(f"\n--- {name} (复用) ---")
            print(f"  ADE={module_results['full']['ade_mean']/1000:.2f}km")
            continue

        print(f"\n--- {name} ({abl_type}) ---")
        wrapper = AblationWrapper(model, abl_type)
        metrics = evaluate_ablation(wrapper, ds, device)
        ades = [m['ade'] for m in metrics]
        fdes = [m['fde'] for m in metrics]
        kinematics_results[abl_type] = {
            'name': name,
            'ade_mean': float(np.mean(ades)),
            'ade_std': float(np.std(ades)),
            'fde_mean': float(np.mean(fdes)),
            'fde_std': float(np.std(fdes)),
            'n_samples': len(metrics),
        }
        print(f"  ADE={np.mean(ades)/1000:.2f}km  FDE={np.mean(fdes)/1000:.2f}km")

    # ============================================================
    #  环境通道消融 (表4.14)
    # ============================================================
    print("\n" + "=" * 60)
    print("表4.14: 环境通道消融实验")
    print("=" * 60)

    channel_ablations = {
        '完整模型': 'full',
        '无DEM': 'no_dem',
        '无LULC': 'no_lulc',
        '无OSM': 'no_osm',
    }

    channel_results = {}
    for name, abl_type in channel_ablations.items():
        if abl_type == 'full' and 'full' in module_results:
            # 复用完整模型结果
            channel_results[abl_type] = module_results['full'].copy()
            channel_results[abl_type]['name'] = name
            print(f"\n--- {name} (复用) ---")
            print(f"  ADE={module_results['full']['ade_mean']/1000:.2f}km")
            continue

        print(f"\n--- {name} ({abl_type}) ---")
        wrapper = AblationWrapper(model, abl_type)
        metrics = evaluate_ablation(wrapper, ds, device)
        ades = [m['ade'] for m in metrics]
        fdes = [m['fde'] for m in metrics]
        channel_results[abl_type] = {
            'name': name,
            'ade_mean': float(np.mean(ades)),
            'ade_std': float(np.std(ades)),
            'fde_mean': float(np.mean(fdes)),
            'fde_std': float(np.std(fdes)),
            'n_samples': len(metrics),
        }
        print(f"  ADE={np.mean(ades)/1000:.2f}km  FDE={np.mean(fdes)/1000:.2f}km")

    # 保存结果
    all_results = {
        'phase': args.phase,
        'module_ablation': module_results,
        'kinematics_ablation': kinematics_results,
        'channel_ablation': channel_results,
    }
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总表格
    print("\n" + "=" * 60)
    print("消融实验结果汇总")
    print("=" * 60)

    print("\n表4.13: 模块消融")
    print(f"{'配置':<20} {'ADE(km)':>10} {'FDE(km)':>10} {'ΔADE':>10}")
    print("-" * 52)
    base_ade = module_results['full']['ade_mean']
    for abl_type, r in module_results.items():
        delta = (r['ade_mean'] - base_ade) / 1000
        print(f"  {r['name']:<18} {r['ade_mean']/1000:>9.2f} {r['fde_mean']/1000:>9.2f} {delta:>+9.2f}")

    print("\n表4.15: 运动学/历史特征消融")
    print(f"{'配置':<22} {'ADE(km)':>10} {'FDE(km)':>10} {'ΔADE':>10}")
    print("-" * 54)
    for abl_type, r in kinematics_results.items():
        delta = (r['ade_mean'] - base_ade) / 1000
        print(f"  {r['name']:<20} {r['ade_mean']/1000:>9.2f} {r['fde_mean']/1000:>9.2f} {delta:>+9.2f}")

    print("\n表4.14: 环境通道消融")
    print(f"{'配置':<20} {'ADE(km)':>10} {'FDE(km)':>10} {'ΔADE':>10}")
    print("-" * 52)
    for abl_type, r in channel_results.items():
        delta = (r['ade_mean'] - base_ade) / 1000
        print(f"  {r['name']:<18} {r['ade_mean']/1000:>9.2f} {r['fde_mean']/1000:>9.2f} {delta:>+9.2f}")

    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
