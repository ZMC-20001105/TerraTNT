#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TerraTNT 优化训练脚本
- 大 batch size (128)
- 多 workers (10)
- 坐标归一化
- 训练监控
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

# 训练配置
BATCH_SIZE = 128
NUM_WORKERS = 10
LEARNING_RATE = 0.001  # 恢复正常学习率
NUM_EPOCHS = 30

print(f"=" * 60)
print(f"训练配置:")
print(f"  BATCH_SIZE = {BATCH_SIZE}")
print(f"  NUM_WORKERS = {NUM_WORKERS}")
print(f"  LEARNING_RATE = {LEARNING_RATE}")
print(f"=" * 60)

class FASDataset(Dataset):
    def __init__(self, traj_dir, fas_split_file, phase='fas1', 
                 history_len=10, future_len=60, num_candidates=6):
        self.traj_dir = Path(traj_dir)
        self.phase = phase
        self.history_len = history_len
        self.future_len = future_len
        self.num_candidates = num_candidates
        
        with open(fas_split_file, 'r') as f:
            splits = json.load(f)
        
        self.file_list = splits[phase]['files']
        print(f"{phase.upper()}: {len(self.file_list)} 文件")
        
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        for file_name in tqdm(self.file_list, desc=f"加载{self.phase}"):
            traj_file = self.traj_dir / file_name
            try:
                with open(traj_file, 'rb') as f:
                    data = pickle.load(f)
                
                path = np.array([(p[0], p[1]) for p in data.get('path', data.get('path_utm', []))])
                
                if len(path) < self.history_len + self.future_len:
                    continue
                
                for start_idx in range(0, len(path) - self.history_len - self.future_len, 30):
                    history = path[start_idx:start_idx + self.history_len]
                    future = path[start_idx + self.history_len:start_idx + self.history_len + self.future_len]
                    goal = future[-1]  # 正确的 goal
                    current_pos = history[-1]
                    
                    # 归一化
                    history_rel = history - current_pos
                    future_rel = future - current_pos
                    goal_rel = goal - current_pos
                    
                    self.samples.append({
                        'history': history_rel,
                        'future': future_rel,
                        'goal': goal_rel
                    })
            except:
                continue
        
        print(f"{self.phase.upper()}: {len(self.samples)} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        history = torch.FloatTensor(sample['history'])
        future = torch.FloatTensor(sample['future'])
        goal = torch.FloatTensor(sample['goal'])
        
        # 生成候选点
        candidates = [goal]
        for _ in range(self.num_candidates - 1):
            offset = np.random.randn(2) * 3000
            candidates.append(goal + offset)
        candidates = torch.FloatTensor(np.array(candidates))
        
        return {
            'history': history,
            'future': future,
            'candidates': candidates
        }


def train_phase(phase):
    print(f"\n{'='*60}")
    print(f"训练 {phase.upper()}")
    print(f"{'='*60}")
    
    # 数据集
    dataset = FASDataset(
        traj_dir='/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest',
        fas_split_file='/home/zmc/文档/programwork/data/processed/fas_splits/bohemian_forest/fas_splits.json',
        phase=phase,
        history_len=10,
        future_len=60,
        num_candidates=6
    )
    
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\n创建 DataLoader:")
    print(f"  batch_size = {BATCH_SIZE}")
    print(f"  num_workers = {NUM_WORKERS}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"  训练 batches: {len(train_loader)}")
    print(f"  验证 batches: {len(val_loader)}")
    
    # 模型
    from models.terratnt import TerraTNT
    
    model = TerraTNT({
        'history_len': 10,
        'future_len': 60,
        'hidden_dim': 256,
        'num_goals': 6,
        'map_size': 128,
        'in_channels': 18,
        'env_channels': 18,
        'output_length': 60
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    save_dir = Path('/home/zmc/文档/programwork/runs') / f'terratnt_{phase}_optimized' / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练...")
    print(f"保存目录: {save_dir}")
    
    best_val_ade = float('inf')
    patience_counter = 0
    patience = 5  # 5 个 epoch 不改善就停止
    
    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        train_ade = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            candidates = batch['candidates'].to(device)
            
            # 打印第一个 batch 的信息
            if epoch == 0 and batch_idx == 0:
                print(f"\n第一个 batch 信息:")
                print(f"  history shape: {history.shape}")
                print(f"  future shape: {future.shape}")
                print(f"  实际 batch size: {history.size(0)}")
            
            optimizer.zero_grad()
            
            current_pos = torch.zeros(history.size(0), 2).to(device)
            env_map = torch.zeros(history.size(0), 18, 128, 128).to(device)
            
            try:
                pred, _ = model(env_map, history, candidates, current_pos, 
                               teacher_forcing_ratio=0.5, ground_truth=future)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                loss = criterion(pred, future)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                ade = torch.mean(torch.norm(pred - future, dim=-1)).item()
                train_ade += ade
                
                pbar.set_postfix({'loss': f'{loss.item():.2f}', 'ade': f'{ade:.1f}m'})
            except Exception as e:
                print(f"批次失败: {e}")
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ade = train_ade / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        val_ade = 0
        
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                candidates = batch['candidates'].to(device)
                env_map = batch['env_map'].to(device)  # 使用真实环境数据
                
                current_pos = torch.zeros(history.size(0), 2).to(device)
                
                try:
                    pred, _ = model(env_map, history, candidates, current_pos)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    
                    loss = criterion(pred, future)
                    val_loss += loss.item()
                    val_ade += torch.mean(torch.norm(pred - future, dim=-1)).item()
                except:
                    continue
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ade = val_ade / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  训练 - Loss: {avg_train_loss:.2f}, ADE: {avg_train_ade:.1f}m")
        print(f"  验证 - Loss: {avg_val_loss:.2f}, ADE: {avg_val_ade:.1f}m")
        
        # 保存最佳模型
        if avg_val_ade < best_val_ade:
            best_val_ade = avg_val_ade
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ade': avg_val_ade
            }, save_dir / 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (ADE={avg_val_ade:.1f}m)")
        
        # 早期检查
        if epoch < 3 and avg_val_ade > 10000:
            print(f"\n⚠️ 警告: Epoch {epoch+1} 验证 ADE > 10km，训练可能有问题")
    
    print(f"\n✓ {phase.upper()} 完成，最佳 ADE: {best_val_ade:.1f}m")
    return best_val_ade


if __name__ == '__main__':
    for phase in ['fas1', 'fas2', 'fas3']:
        try:
            best_ade = train_phase(phase)
            print(f"\n{phase.upper()} 最佳 ADE: {best_ade:.1f}m")
        except Exception as e:
            print(f"\n✗ {phase.upper()} 失败: {e}")
            import traceback
            traceback.print_exc()
