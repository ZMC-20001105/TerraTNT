#!/usr/bin/env python
"""
并行训练多个基线模型 - 充分利用GPU显存
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
from multiprocessing import Process, Queue
import os

# 简化的数据集类
class SimpleTrajectoryDataset(Dataset):
    def __init__(self, data_dir, history_len=10, future_len=60, sample_interval=60):
        self.data_dir = Path(data_dir)
        self.history_len = history_len
        self.future_len = future_len
        self.sample_interval = sample_interval
        
        # 加载所有轨迹文件
        self.traj_files = sorted(list(self.data_dir.glob('*.pkl')))
        print(f"加载 {len(self.traj_files)} 条轨迹文件")
        
        # 预处理：生成训练样本
        self.samples = []
        self._prepare_samples()
        
    def _prepare_samples(self):
        print("准备训练样本...")
        max_files = min(1000, len(self.traj_files))
        for idx, traj_file in enumerate(self.traj_files[:max_files], 1):
            if idx % 50 == 0 or idx == 1 or idx == max_files:
                print(f"  处理轨迹文件: {idx}/{max_files}")
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
            except Exception as e:
                continue
        
        print(f"生成 {len(self.samples)} 个训练样本\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        history = torch.from_numpy(sample['history'])
        future = torch.from_numpy(sample['future'])
        env_map = torch.zeros(18, 128, 128)
        return history, future, env_map


def train_single_model(model_name, model_class, config, train_dataset, val_dataset, gpu_id, result_queue):
    """在指定GPU上训练单个模型"""
    try:
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        print(f"[{model_name}] 开始训练，使用 GPU {gpu_id}")
        
        # 创建DataLoader - 使用CPU多进程加载数据
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=4,  # CPU多进程加载
            pin_memory=True  # 加速数据传输到GPU
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # 创建模型
        model_config = {
            'history_len': 10,
            'future_len': 60,
            'in_channels': 18,
            'map_size': 128,
            'hidden_dim': 256,
            'num_modes': 5
        }
        model = model_class(model_config).to(device)
        
        # 优化器
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
            
            for batch_idx, (history, future, env_map) in enumerate(train_loader):
                history = history.to(device)
                future = future.to(device)
                env_map = env_map.to(device)
                
                optimizer.zero_grad()
                
                if model_name == 'social_lstm':
                    pred = model(history)
                else:
                    pred = model(history, env_map)
                
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                loss = criterion(pred, future)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                ade = torch.mean(torch.norm(pred - future, dim=-1))
                train_ade += ade.item()
                
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    avg_ade = train_ade / (batch_idx + 1)
                    print(f"[{model_name}] Epoch {epoch+1}, Step {batch_idx+1}/{len(train_loader)}: loss={avg_loss:.4f}, ade={avg_ade:.4f}")
            
            # 验证
            model.eval()
            val_loss = 0
            val_ade = 0
            val_fde = 0
            
            with torch.no_grad():
                for history, future, env_map in val_loader:
                    history = history.to(device)
                    future = future.to(device)
                    env_map = env_map.to(device)
                    
                    if model_name == 'social_lstm':
                        pred = model(history)
                    else:
                        pred = model(history, env_map)
                    
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    
                    loss = criterion(pred, future)
                    val_loss += loss.item()
                    
                    ade = torch.mean(torch.norm(pred - future, dim=-1))
                    fde = torch.mean(torch.norm(pred[:, -1] - future[:, -1], dim=-1))
                    val_ade += ade.item()
                    val_fde += fde.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_ade = train_ade / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_ade = val_ade / len(val_loader)
            avg_val_fde = val_fde / len(val_loader)
            
            print(f"[{model_name}] Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Train Loss={avg_train_loss:.4f}, ADE={avg_train_ade:.4f} | "
                  f"Val Loss={avg_val_loss:.4f}, ADE={avg_val_ade:.4f}, FDE={avg_val_fde:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_dir / 'best_model.pth')
                print(f"[{model_name}] ✓ 保存最佳模型 (val_loss={best_val_loss:.4f})")
            
            # 早停
            if epoch >= 5 and avg_val_loss > best_val_loss * 1.1:
                print(f"[{model_name}] 早停于 epoch {epoch+1}")
                break
        
        result_queue.put({
            'model_name': model_name,
            'status': 'success',
            'best_val_loss': best_val_loss,
            'save_dir': str(save_dir)
        })
        
    except Exception as e:
        print(f"[{model_name}] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put({
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        })


def main():
    # 配置
    config = {
        'data_dir': '/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest',
        'save_dir': '/home/zmc/文档/programwork/runs',
        'batch_size': 64,  # 增大batch size
        'learning_rate': 0.001,
        'num_epochs': 30,
        'val_split': 0.2
    }
    
    print("="*60)
    print("并行训练基线模型")
    print("="*60)
    print(f"Batch Size: {config['batch_size']}")
    print(f"DataLoader Workers: 4 (CPU多进程)")
    print(f"并行模型数: 3")
    print("="*60)
    
    # 加载数据集（主进程）
    print("\n加载数据集...")
    dataset = SimpleTrajectoryDataset(config['data_dir'])
    
    if len(dataset) == 0:
        print("错误：没有生成训练样本")
        return
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * config['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    
    # 导入模型
    from models.baselines.social_lstm import SocialLSTM
    from models.baselines.ynet import YNet
    from models.baselines.pecnet import PECNet
    from models.baselines.trajectron import TrajectronPP
    
    # 定义要并行训练的模型（每个用同一个GPU，显存够用）
    models_to_train = [
        ('social_lstm', SocialLSTM),
        ('ynet', YNet),
        ('pecnet', PECNet),
    ]
    
    # 创建结果队列
    result_queue = Queue()
    
    # 启动并行训练进程
    processes = []
    for idx, (model_name, model_class) in enumerate(models_to_train):
        p = Process(
            target=train_single_model,
            args=(model_name, model_class, config, train_dataset, val_dataset, 0, result_queue)
        )
        p.start()
        processes.append(p)
        print(f"启动训练进程: {model_name} (PID: {p.pid})")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 收集结果
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        results[result['model_name']] = result
    
    # 保存结果
    results_file = Path(config['save_dir']) / 'parallel_training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("训练结果汇总")
    print('='*60)
    for model_name, result in results.items():
        if result['status'] == 'success':
            print(f"✓ {model_name:20s} 最佳验证损失: {result['best_val_loss']:.4f}")
            print(f"  保存位置: {result['save_dir']}")
        else:
            print(f"✗ {model_name:20s} 失败: {result.get('error', 'Unknown')}")
    
    print(f"\n结果已保存到: {results_file}")
    
    # 训练Trajectron++（单独训练，因为它比较大）
    print(f"\n{'='*60}")
    print("训练 Trajectron++")
    print('='*60)
    result_queue2 = Queue()
    train_single_model('trajectron', TrajectronPP, config, train_dataset, val_dataset, 0, result_queue2)
    traj_result = result_queue2.get()
    results['trajectron'] = traj_result
    
    # 更新结果文件
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n所有模型训练完成！")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
