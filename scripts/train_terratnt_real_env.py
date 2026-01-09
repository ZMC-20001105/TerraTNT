#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from utils.env_data_loader import get_env_loader

BATCH_SIZE = 128
NUM_WORKERS = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 30

print("=" * 60)
print("使用真实环境数据训练 TerraTNT")
print("=" * 60)

class FASDataset(Dataset):
    def __init__(self, traj_dir, fas_split_file, phase='fas1', 
                 history_len=10, future_len=60, num_candidates=6, region='bohemian_forest'):
        self.traj_dir = Path(traj_dir)
        self.phase = phase
        self.history_len = history_len
        self.future_len = future_len
        self.num_candidates = num_candidates
        self.region = region
        
        print(f"正在加载 {region} 真实环境数据...")
        self.env_loader = get_env_loader(region)
        print(f"✓ 已加载真实环境数据")
        
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
                    goal = future[-1]
                    current_pos = history[-1]
                    
                    history_rel = history - current_pos
                    future_rel = future - current_pos
                    goal_rel = goal - current_pos
                    
                    self.samples.append({
                        'history': history_rel,
                        'future': future_rel,
                        'goal': goal_rel,
                        'current_pos_abs': current_pos
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
        current_pos_abs = sample['current_pos_abs']
        
        candidates = [goal]
        for _ in range(self.num_candidates - 1):
            offset = np.random.randn(2) * 3000
            candidates.append(goal + offset)
        candidates = torch.FloatTensor(np.array(candidates))
        
        try:
            env_map = self.env_loader.extract_patch(
                center_utm=(float(current_pos_abs[0]), float(current_pos_abs[1])),
                patch_size=128
            )
        except Exception as e:
            env_map = torch.zeros(18, 128, 128)
            if idx % 1000 == 0:
                print(f"警告: 样本 {idx} 环境数据提取失败")
        
        return {
            'history': history,
            'future': future,
            'candidates': candidates,
            'env_map': env_map
        }

def train_phase(phase):
    print(f"\n训练 {phase.upper()}")
    
    dataset = FASDataset(
        traj_dir='/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest',
        fas_split_file='/home/zmc/文档/programwork/data/processed/fas_splits/bohemian_forest/fas_splits.json',
        phase=phase,
        region='bohemian_forest'
    )
    
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    from models.terratnt import TerraTNT
    model = TerraTNT({'history_len': 10, 'future_len': 60, 'hidden_dim': 256, 'num_goals': 6, 'map_size': 128, 'in_channels': 18, 'env_channels': 18, 'output_length': 60})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    
    save_dir = Path('/home/zmc/文档/programwork/runs') / f'terratnt_{phase}_real_env' / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_ade = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_ade = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            candidates = batch['candidates'].to(device)
            env_map = batch['env_map'].to(device)
            
            if epoch == 0 and batch_idx == 0:
                print(f"\n✓ 使用真实环境数据:")
                print(f"  env_map shape: {env_map.shape}")
                print(f"  非零元素: {(env_map != 0).sum().item()} / {env_map.numel()}")
                print(f"  非零比例: {(env_map != 0).sum().item() / env_map.numel() * 100:.1f}%")
            
            optimizer.zero_grad()
            current_pos = torch.zeros(history.size(0), 2).to(device)
            
            try:
                pred, _ = model(env_map, history, candidates, current_pos, teacher_forcing_ratio=0.5, ground_truth=future)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                loss = criterion(pred, future)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_ade += torch.mean(torch.norm(pred - future, dim=-1)).item()
            except:
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ade = train_ade / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_ade = 0
        
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                candidates = batch['candidates'].to(device)
                env_map = batch['env_map'].to(device)
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
        
        print(f"Epoch {epoch+1}: 训练 ADE={avg_train_ade:.1f}m, 验证 ADE={avg_val_ade:.1f}m")
        
        scheduler.step(avg_val_ade)
        
        if avg_val_ade < best_val_ade:
            best_val_ade = avg_val_ade
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_ade': avg_val_ade}, save_dir / 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (ADE={avg_val_ade:.1f}m)")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"早停")
                break
    
    print(f"✓ {phase.upper()} 完成，最佳 ADE: {best_val_ade:.1f}m")
    return best_val_ade

if __name__ == '__main__':
    for phase in ['fas1', 'fas2', 'fas3']:
        try:
            best_ade = train_phase(phase)
        except Exception as e:
            print(f"✗ {phase.upper()} 失败: {e}")
            import traceback
            traceback.print_exc()
