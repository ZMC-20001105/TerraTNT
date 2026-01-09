#!/usr/bin/env python
"""单个模型训练脚本"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime

class SimpleTrajectoryDataset(Dataset):
    def __init__(self, data_dir, history_len=10, future_len=60):
        self.data_dir = Path(data_dir)
        self.history_len = history_len
        self.future_len = future_len
        
        self.traj_files = sorted(list(self.data_dir.glob('*.pkl')))[:1000]
        print(f"加载 {len(self.traj_files)} 条轨迹")
        
        self.samples = []
        for idx, traj_file in enumerate(self.traj_files, 1):
            if idx % 200 == 0:
                print(f"处理: {idx}/{len(self.traj_files)}")
            try:
                with open(traj_file, 'rb') as f:
                    data = pickle.load(f)
                
                path = np.array([(p[0], p[1]) for p in data['path']])
                timestamps = data['timestamps']
                
                sampled_idx = []
                current_time = timestamps[0]
                for i, t in enumerate(timestamps):
                    if t >= current_time:
                        sampled_idx.append(i)
                        current_time = t + 60
                
                if len(sampled_idx) < self.history_len + self.future_len:
                    continue
                
                sampled_path = path[sampled_idx]
                
                for i in range(len(sampled_path) - self.history_len - self.future_len + 1):
                    history = sampled_path[i:i+self.history_len]
                    future = sampled_path[i+self.history_len:i+self.history_len+self.future_len]
                    self.samples.append({
                        'history': history.astype(np.float32),
                        'future': future.astype(np.float32)
                    })
            except:
                continue
        
        print(f"生成 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def main(model_name):
    device = torch.device('cuda:0')
    print(f"\n{'='*60}")
    print(f"[{model_name.upper()}] 开始训练")
    print(f"{'='*60}\n")
    
    # 加载数据
    data_dir = '/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest'
    dataset = SimpleTrajectoryDataset(data_dir)
    
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    # 创建模型
    from models.baselines.ynet import YNet
    from models.baselines.pecnet import PECNet
    
    model_config = {
        'history_len': 10,
        'future_len': 60,
        'in_channels': 18,
        'map_size': 128,
        'hidden_dim': 256,
        'num_modes': 5
    }
    
    if model_name == 'ynet':
        model = YNet(model_config)
    else:
        model = PECNet(model_config)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练
    best_val_loss = float('inf')
    save_dir = Path(f'/home/zmc/文档/programwork/runs/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(30):
        model.train()
        train_loss = 0
        train_ade = 0
        
        for batch_idx, batch in enumerate(train_loader):
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            
            optimizer.zero_grad()
            
            env_map = torch.zeros(history.size(0), 18, 128, 128).to(device)
            pred = model(history, env_map)
            
            if isinstance(pred, tuple):
                pred = pred[0]
            
            loss = criterion(pred, future)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            ade = torch.mean(torch.norm(pred - future, dim=-1))
            train_ade += ade.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"[{model_name.upper()}] Epoch {epoch+1}, Step {batch_idx+1}: loss={train_loss/(batch_idx+1):.2f}, ade={train_ade/(batch_idx+1):.2f}")
        
        # 验证
        model.eval()
        val_loss = 0
        val_ade = 0
        val_fde = 0
        
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                
                env_map = torch.zeros(history.size(0), 18, 128, 128).to(device)
                pred = model(history, env_map)
                
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                loss = criterion(pred, future)
                val_loss += loss.item()
                
                ade = torch.mean(torch.norm(pred - future, dim=-1))
                fde = torch.mean(torch.norm(pred[:, -1] - future[:, -1], dim=-1))
                val_ade += ade.item()
                val_fde += fde.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ade = val_ade / len(val_loader)
        avg_val_fde = val_fde / len(val_loader)
        
        print(f"[{model_name.upper()}] Epoch {epoch+1}/30: Val Loss={avg_val_loss:.2f}, ADE={avg_val_ade:.2f}, FDE={avg_val_fde:.2f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"[{model_name.upper()}] ✓ 保存最佳模型")
        
        if epoch >= 5 and avg_val_loss > best_val_loss * 1.1:
            print(f"[{model_name.upper()}] 早停")
            break
    
    print(f"\n[{model_name.upper()}] ✓ 训练完成，最佳 Val Loss={best_val_loss:.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python train_single_model.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    main(model_name)
