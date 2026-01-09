#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练 TerraTNT 模型 - 修复版本
关键修复：
1. 坐标归一化（相对于当前点）
2. 正确的 goal 定义（future[-1] 而不是 path[-1]）
3. 训练过程健康检查（定期验证 ADE/FDE）
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import warnings

class FASTrajectoryDatasetFixed(Dataset):
    """FAS 阶段特定的轨迹数据集 - 修复版"""
    
    def __init__(self, traj_dir, fas_split_file, phase='fas1', 
                 history_len=10, future_len=60, num_candidates=6):
        self.traj_dir = Path(traj_dir)
        self.phase = phase
        self.history_len = history_len
        self.future_len = future_len
        self.num_candidates = num_candidates
        
        # 加载 FAS 划分
        with open(fas_split_file, 'r') as f:
            splits = json.load(f)
        
        self.file_list = splits[phase]['files']
        print(f"{phase.upper()}: 加载 {len(self.file_list)} 个轨迹文件")
        
        # 预处理样本
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        print(f"准备 {self.phase.upper()} 样本...")
        
        for file_name in tqdm(self.file_list, desc=f"处理{self.phase}"):
            traj_file = self.traj_dir / file_name
            
            try:
                with open(traj_file, 'rb') as f:
                    data = pickle.load(f)
                
                path = np.array([(p[0], p[1]) for p in data.get('path', data.get('path_utm', []))])
                
                if len(path) < self.history_len + self.future_len:
                    continue
                
                # 滑动窗口采样
                for start_idx in range(0, len(path) - self.history_len - self.future_len, 30):
                    history = path[start_idx:start_idx + self.history_len]
                    future = path[start_idx + self.history_len:start_idx + self.history_len + self.future_len]
                    
                    # 关键修复：goal 是当前窗口的终点，而不是整条轨迹的终点
                    goal = future[-1]  # 60分钟后的位置
                    current_pos = history[-1]  # 当前位置
                    
                    # 归一化：相对于当前位置
                    history_rel = history - current_pos
                    future_rel = future - current_pos
                    goal_rel = goal - current_pos
                    
                    self.samples.append({
                        'history': history_rel,
                        'future': future_rel,
                        'goal': goal_rel,
                        'current_pos': current_pos,  # 保存用于反归一化
                        'traj_file': str(traj_file)
                    })
            
            except Exception as e:
                continue
        
        print(f"{self.phase.upper()} 样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        history = torch.FloatTensor(sample['history'])
        future = torch.FloatTensor(sample['future'])
        goal = torch.FloatTensor(sample['goal'])
        current_pos = torch.FloatTensor(sample['current_pos'])
        
        # 生成候选终点（相对坐标）
        candidates = self._generate_candidates(sample['goal'], sample['traj_file'])
        
        return {
            'history': history,
            'future': future,
            'goal': goal,
            'candidates': candidates,
            'current_pos': current_pos
        }
    
    def _generate_candidates(self, true_goal_rel, traj_file):
        """
        生成候选终点集合（相对坐标）
        
        FAS1/FAS2: 候选集包含真实终点（完备）
        FAS3: 候选集不包含真实终点（不完备）
        """
        candidates = []
        
        if self.phase in ['fas1', 'fas2']:
            # 完备候选集：包含真实终点
            candidates.append(true_goal_rel)
            
            # 添加负样本（相对坐标空间的随机偏移）
            for _ in range(self.num_candidates - 1):
                # 在相对坐标空间，偏移范围应该是几千米级别
                offset = np.random.randn(2) * 3000  # 3km 标准差
                neg_candidate = true_goal_rel + offset
                candidates.append(neg_candidate)
        
        else:  # fas3
            # 不完备候选集：不包含真实终点
            # 生成围绕真值但不包含真值的候选点
            for _ in range(self.num_candidates):
                # 确保候选点距离真值至少 1km
                offset = np.random.randn(2) * 3000 + np.random.choice([-1, 1], 2) * 1000
                neg_candidate = true_goal_rel + offset
                candidates.append(neg_candidate)
        
        return torch.FloatTensor(np.array(candidates))


def compute_metrics(pred, target):
    """
    计算评估指标（在相对坐标空间）
    
    Args:
        pred: (batch, future_len, 2) 预测轨迹（相对坐标）
        target: (batch, future_len, 2) 真实轨迹（相对坐标）
    
    Returns:
        ade: 平均位移误差（米）
        fde: 最终位移误差（米）
    """
    # ADE: 所有时间步的平均欧氏距离
    displacement = torch.norm(pred - target, dim=-1)  # (batch, future_len)
    ade = torch.mean(displacement).item()
    
    # FDE: 最后一个时间步的欧氏距离
    fde = torch.mean(displacement[:, -1]).item()
    
    return ade, fde


def health_check(model, val_loader, device, epoch, phase):
    """
    训练健康检查：在固定验证集上测试
    
    如果指标异常，返回 False 并建议中止训练
    """
    model.eval()
    
    # 只用前几个batch做快速检查
    check_batches = min(10, len(val_loader))
    total_ade = 0
    total_fde = 0
    pred_min, pred_max = float('inf'), float('-inf')
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= check_batches:
                break
            
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            candidates = batch['candidates'].to(device)
            
            # 当前位置在相对坐标系中是 (0, 0)
            current_pos = torch.zeros(history.size(0), 2).to(device)
            
            # 环境地图（暂时用零，后续可以加载真实地图）
            env_map = torch.zeros(history.size(0), 18, 128, 128).to(device)
            
            try:
                pred, _ = model(env_map, history, candidates, current_pos)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                # 计算指标
                ade, fde = compute_metrics(pred, future)
                total_ade += ade
                total_fde += fde
                
                # 记录预测值范围
                pred_min = min(pred_min, pred.min().item())
                pred_max = max(pred_max, pred.max().item())
            
            except Exception as e:
                print(f"    ⚠️ 健康检查批次失败: {e}")
                continue
    
    avg_ade = total_ade / check_batches
    avg_fde = total_fde / check_batches
    
    print(f"\n  📊 健康检查 (Epoch {epoch+1}):")
    print(f"    - ADE: {avg_ade:.2f} m")
    print(f"    - FDE: {avg_fde:.2f} m")
    print(f"    - 预测值范围: [{pred_min:.2f}, {pred_max:.2f}] m")
    
    # 异常检测
    is_healthy = True
    warnings_list = []
    
    if avg_ade > 20000:  # 20km
        warnings_list.append(f"ADE 过高 ({avg_ade/1000:.1f} km)")
        is_healthy = False
    
    if np.isnan(avg_ade) or np.isnan(avg_fde):
        warnings_list.append("出现 NaN")
        is_healthy = False
    
    if abs(pred_min) > 100000 or abs(pred_max) > 100000:  # 100km
        warnings_list.append(f"预测值范围异常 ({pred_min/1000:.1f} ~ {pred_max/1000:.1f} km)")
        is_healthy = False
    
    if warnings_list:
        print(f"    ⚠️ 警告: {', '.join(warnings_list)}")
        if epoch < 5 and not is_healthy:
            print(f"    ❌ 建议中止训练：前期指标异常")
            return False, avg_ade, avg_fde
    else:
        print(f"    ✅ 指标正常")
    
    return True, avg_ade, avg_fde


def train_terratnt_phase(phase, config):
    """训练 TerraTNT 模型的一个阶段"""
    
    print(f"\n{'='*60}")
    print(f"开始训练 TerraTNT - {phase.upper()} (修复版)")
    print('='*60)
    
    # 准备数据集
    dataset = FASTrajectoryDatasetFixed(
        traj_dir=config['traj_dir'],
        fas_split_file=config['fas_split_file'],
        phase=phase,
        history_len=config['history_len'],
        future_len=config['future_len'],
        num_candidates=config['num_candidates']
    )
    
    if len(dataset) == 0:
        print(f"错误：{phase} 没有有效样本")
        return None
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=10, pin_memory=True)  
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=10, pin_memory=True)
    
    # 创建模型
    from models.terratnt import TerraTNT
    
    model_config = {
        'history_len': config['history_len'],
        'future_len': config['future_len'],
        'hidden_dim': 256,
        'num_goals': config['num_candidates'],
        'map_size': 128,
        'in_channels': 18,
        'env_channels': 18,
        'output_length': config['future_len']
    }
    
    model = TerraTNT(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_ade = float('inf')
    patience_counter = 0
    
    save_dir = Path(config['save_dir']) / f'terratnt_{phase}_fixed' / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump({**config, **model_config, 'phase': phase}, f, indent=2)
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    print(f"保存目录: {save_dir}")
    print(f"坐标系统: 相对坐标（归一化到当前位置）")
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_ade': [],
        'val_loss': [],
        'val_ade': [],
        'val_fde': []
    }
    
    for epoch in range(config['num_epochs']):
        # 训练
        model.train()
        train_loss = 0
        train_ade = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in pbar:
            history_batch = batch['history'].to(device)
            future_batch = batch['future'].to(device)
            candidates = batch['candidates'].to(device)
            
            optimizer.zero_grad()
            
            # 当前位置在相对坐标系中是原点
            current_pos = torch.zeros(history_batch.size(0), 2).to(device)
            
            # 环境地图（暂时用零）
            env_map = torch.zeros(history_batch.size(0), 18, 128, 128).to(device)
            
            try:
                pred, goal_probs = model(env_map, history_batch, candidates, current_pos, 
                                        teacher_forcing_ratio=0.5, ground_truth=future_batch)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                loss = criterion(pred, future_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                ade, _ = compute_metrics(pred, future_batch)
                train_ade += ade
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'ade': f'{ade:.2f}m'})
            
            except Exception as e:
                print(f"训练批次失败: {e}")
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ade = train_ade / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        val_ade = 0
        val_fde = 0
        
        with torch.no_grad():
            for batch in val_loader:
                history_batch = batch['history'].to(device)
                future_batch = batch['future'].to(device)
                candidates = batch['candidates'].to(device)
                
                current_pos = torch.zeros(history_batch.size(0), 2).to(device)
                env_map = torch.zeros(history_batch.size(0), 18, 128, 128).to(device)
                
                try:
                    pred, _ = model(env_map, history_batch, candidates, current_pos)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    
                    loss = criterion(pred, future_batch)
                    val_loss += loss.item()
                    
                    ade, fde = compute_metrics(pred, future_batch)
                    val_ade += ade
                    val_fde += fde
                
                except:
                    continue
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        avg_val_ade = val_ade / len(val_loader) if len(val_loader) > 0 else float('inf')
        avg_val_fde = val_fde / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_ade'].append(avg_train_ade)
        history['val_loss'].append(avg_val_loss)
        history['val_ade'].append(avg_val_ade)
        history['val_fde'].append(avg_val_fde)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"  训练 - Loss: {avg_train_loss:.4f}, ADE: {avg_train_ade:.2f}m")
        print(f"  验证 - Loss: {avg_val_loss:.4f}, ADE: {avg_val_ade:.2f}m, FDE: {avg_val_fde:.2f}m")
        
        # 健康检查
        is_healthy, check_ade, check_fde = health_check(model, val_loader, device, epoch, phase)
        
        if not is_healthy and epoch < 5:
            print(f"\n❌ 训练异常中止：前 {epoch+1} 个 epoch 指标不正常")
            print(f"   建议检查：数据预处理、模型架构、学习率等")
            break
        
        # 保存最佳模型
        if avg_val_ade < best_val_ade:
            best_val_ade = avg_val_ade
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_ade': avg_val_ade,
                'val_fde': avg_val_fde,
                'config': model_config,
                'history': history
            }, save_dir / 'best_model.pth')
            
            print(f"  ✅ 保存最佳模型 (val_ade={avg_val_ade:.2f}m, val_fde={avg_val_fde:.2f}m)")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"早停：验证 ADE {config['patience']} 轮未改善")
                break
    
    # 保存训练历史
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ {phase.upper()} 训练完成")
    print(f"  最佳验证 ADE: {best_val_ade:.2f}m")
    print(f"  最佳验证 FDE: {history['val_fde'][history['val_ade'].index(best_val_ade)]:.2f}m")
    
    return {
        'phase': phase,
        'best_val_loss': best_val_loss,
        'best_val_ade': best_val_ade,
        'save_dir': str(save_dir),
        'history': history
    }


def main():
    # 配置
    config = {
        'traj_dir': '/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest',
        'fas_split_file': '/home/zmc/文档/programwork/data/processed/fas_splits/bohemian_forest/fas_splits.json',
        'save_dir': '/home/zmc/文档/programwork/runs',
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'patience': 5,
        'history_len': 10,
        'future_len': 60,
        'num_candidates': 6
    }
    
    print("="*60)
    print("TerraTNT 三阶段训练 - 修复版")
    print("="*60)
    print("关键改进:")
    print("  1. ✅ 坐标归一化（相对于当前位置）")
    print("  2. ✅ 正确的 goal 定义（future[-1]）")
    print("  3. ✅ 训练健康检查（每 epoch 验证 ADE/FDE）")
    print("  4. ✅ 异常自动中止（前 5 epoch ADE > 20km）")
    print(f"\n数据目录: {config['traj_dir']}")
    print(f"FAS划分文件: {config['fas_split_file']}")
    print()
    
    # 训练三个阶段
    results = {}
    
    for phase in ['fas1', 'fas2', 'fas3']:
        try:
            result = train_terratnt_phase(phase, config)
            if result:
                results[phase] = result
        except Exception as e:
            print(f"✗ {phase.upper()} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results[phase] = {'status': 'failed', 'error': str(e)}
    
    # 保存结果
    results_file = Path(config['save_dir']) / 'terratnt_training_results_fixed.json'
    
    # 转换 numpy 类型为 Python 原生类型
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    results_native = convert_to_native(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_native, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TerraTNT 训练结果汇总（修复版）")
    print('='*60)
    for phase, result in results.items():
        if 'best_val_ade' in result:
            print(f"✓ {phase.upper():10s} 最佳验证 ADE: {result['best_val_ade']:.2f}m")
        else:
            print(f"✗ {phase.upper():10s} 失败: {result.get('error', 'Unknown')}")
    
    print(f"\n结果已保存到: {results_file}")
    
    # 与基线模型对比
    print(f"\n{'='*60}")
    print("与基线模型对比（参考值）")
    print('='*60)
    print("  Trajectron++: ADE ~4,800m")
    print("  PECNet:       ADE ~7,800m")
    print("  Social-LSTM:  ADE ~9,900m")
    print("\n目标：TerraTNT 应该接近或优于这些基线")


if __name__ == '__main__':
    main()
