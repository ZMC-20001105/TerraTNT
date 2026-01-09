#!/usr/bin/env python
"""
并行训练2个模型 - 充分利用GPU显存
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import pickle
import numpy as np
import json
from datetime import datetime
import multiprocessing as mp
import os

# 简化的数据集类
class SimpleTrajectoryDataset(Dataset):
    def __init__(self, data_dir, history_len=10, future_len=60, sample_interval=60):
        self.data_dir = Path(data_dir)
        self.history_len = history_len
        self.future_len = future_len
        self.sample_interval = sample_interval
        
        self.traj_files = sorted(list(self.data_dir.glob('*.pkl')))
        print(f"[Dataset] 加载 {len(self.traj_files)} 条轨迹文件")
        
        self.samples = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        print("[Dataset] 准备训练样本...")
        max_files = min(1000, len(self.traj_files))
        for idx, traj_file in enumerate(self.traj_files[:max_files], 1):
            if idx % 100 == 0 or idx == max_files:
                print(f"[Dataset]   处理: {idx}/{max_files}")
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
                        current_time = t + self.sample_interval
                
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
        
        print(f"[Dataset] 生成 {len(self.samples)} 个训练样本\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def train_single_model_process(model_name, gpu_id, train_indices, val_indices, dataset_samples, config):
    """在独立进程中训练单个模型"""
    try:
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device('cuda:0')
        
        print(f"\n[{model_name}] 启动训练进程 (GPU {gpu_id})")
        
        # 重建数据集（从共享的samples）
        class SubDataset(Dataset):
            def __init__(self, samples, indices):
                self.samples = [samples[i] for i in indices]
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]
        
        train_dataset = SubDataset(dataset_samples, train_indices)
        val_dataset = SubDataset(dataset_samples, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                 shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                               shuffle=False, num_workers=2, pin_memory=True)
        
        # 创建模型
        from models.baselines.social_lstm import SocialLSTM
        from models.baselines.ynet import YNet
        from models.baselines.pecnet import PECNet
        from models.baselines.trajectron import TrajectronPP
        
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
        elif model_name == 'pecnet':
            model = PECNet(model_config)
        elif model_name == 'trajectron':
            model = TrajectronPP(model_config)
        else:
            model = SocialLSTM(model_config)
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        save_dir = Path(config['save_dir']) / model_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(config['num_epochs']):
            # 训练
            model.train()
            train_loss = 0
            train_ade = 0
            
            for batch_idx, batch in enumerate(train_loader):
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                
                optimizer.zero_grad()
                
                if model_name == 'social_lstm':
                    pred = model(history)
                else:
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
                
                if (batch_idx + 1) % 200 == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    avg_ade = train_ade / (batch_idx + 1)
                    print(f"[{model_name}] Epoch {epoch+1}, Step {batch_idx+1}: loss={avg_loss:.4f}, ade={avg_ade:.4f}")
            
            # 验证
            model.eval()
            val_loss = 0
            val_ade = 0
            val_fde = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    history = batch['history'].to(device)
                    future = batch['future'].to(device)
                    
                    if model_name == 'social_lstm':
                        pred = model(history)
                    else:
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
            
            print(f"[{model_name}] Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Val Loss={avg_val_loss:.4f}, ADE={avg_val_ade:.4f}, FDE={avg_val_fde:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_dir / 'best_model.pth')
                print(f"[{model_name}] ✓ 保存最佳模型")
            
            # 早停
            if epoch >= 5 and avg_val_loss > best_val_loss * 1.1:
                print(f"[{model_name}] 早停于 epoch {epoch+1}")
                break
        
        print(f"[{model_name}] ✓ 训练完成，最佳 Val Loss={best_val_loss:.4f}")
        return {'model_name': model_name, 'status': 'success', 'best_val_loss': best_val_loss}
        
    except Exception as e:
        print(f"[{model_name}] ✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return {'model_name': model_name, 'status': 'failed', 'error': str(e)}


def main():
    config = {
        'data_dir': '/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest',
        'save_dir': '/home/zmc/文档/programwork/runs',
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'val_split': 0.2
    }
    
    print("="*60)
    print("并行训练2个模型")
    print("="*60)
    
    # 加载数据集
    dataset = SimpleTrajectoryDataset(config['data_dir'])
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * config['val_split'])
    train_size = len(dataset) - val_size
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"训练样本: {len(train_indices)}")
    print(f"验证样本: {len(val_indices)}")
    
    # 启动2个并行训练进程
    models_to_train = [
        ('ynet', 0),      # YNet在GPU 0
        ('pecnet', 0),    # PECNet也在GPU 0（显存够用）
    ]
    
    print(f"\n启动并行训练...")
    processes = []
    for model_name, gpu_id in models_to_train:
        p = mp.Process(
            target=train_single_model_process,
            args=(model_name, gpu_id, train_indices, val_indices, dataset.samples, config)
        )
        p.start()
        processes.append((model_name, p))
        print(f"✓ 启动 {model_name} (PID: {p.pid})")
    
    # 等待所有进程完成
    print(f"\n等待训练完成...")
    for model_name, p in processes:
        p.join()
        print(f"✓ {model_name} 进程结束")
    
    print(f"\n{'='*60}")
    print("所有模型训练完成！")
    print('='*60)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
