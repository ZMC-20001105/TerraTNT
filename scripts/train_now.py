#!/usr/bin/env python
"""
立即开始训练 - 使用现有Bohemian数据
简化版本，快速启动训练流程
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
        max_files = min(1000, len(self.traj_files))  # 先用1000条测试
        for idx, traj_file in enumerate(self.traj_files[:max_files], 1):
            if idx % 50 == 0 or idx == 1 or idx == max_files:
                print(f"  处理轨迹文件: {idx}/{max_files}")
            try:
                with open(traj_file, 'rb') as f:
                    data = pickle.load(f)
                
                # 提取路径和时间戳
                path = np.array([(p[0], p[1]) for p in data['path']])
                timestamps = data['timestamps']
                
                # 按时间间隔采样
                sampled_idx = []
                current_time = timestamps[0]
                for i, t in enumerate(timestamps):
                    if t >= current_time:
                        sampled_idx.append(i)
                        current_time = t + self.sample_interval
                
                if len(sampled_idx) < self.history_len + self.future_len:
                    continue
                
                sampled_path = path[sampled_idx]
                
                # 滑动窗口生成样本
                for i in range(len(sampled_path) - self.history_len - self.future_len + 1):
                    history = sampled_path[i:i+self.history_len]
                    future = sampled_path[i+self.history_len:i+self.history_len+self.future_len]
                    
                    # 归一化（相对于历史轨迹最后一点）
                    last_pos = history[-1]
                    history_norm = history - last_pos
                    future_norm = future - last_pos
                    
                    self.samples.append({
                        'history': torch.FloatTensor(history_norm),
                        'future': torch.FloatTensor(future_norm)
                    })
            except Exception as e:
                print(f"处理 {traj_file} 失败: {e}")
                continue
        
        print(f"生成 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# 训练函数
def train_model(model, train_loader, val_loader, config, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    save_root = Path(config['save_dir'])
    save_root.mkdir(parents=True, exist_ok=True)
    save_dir = save_root / model_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n开始训练 {model_name}")
    print(f"设备: {device}")
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")
    print(f"保存目录: {save_dir}")
    
    for epoch in range(config['num_epochs']):
        # 训练
        model.train()
        train_loss = 0
        train_ade = 0

        for step, batch in enumerate(train_loader, 1):
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if model_name in ['social_lstm', 'constant_velocity']:
                pred = model(history)
            else:
                # 其他模型需要env_map，这里用零填充
                env_map = torch.zeros(history.size(0), 18, 128, 128).to(device)
                pred = model(history, env_map)
            
            # 处理可能的tuple返回
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # 计算损失
            loss = criterion(pred, future)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ade += torch.mean(torch.norm(pred - future, dim=-1)).item()

            if step % 50 == 0:
                avg_loss = train_loss / step
                avg_ade = train_ade / step
                print(f"  step {step}/{len(train_loader)}: loss={avg_loss:.4f}, ade={avg_ade:.4f}")
        
        train_loss /= len(train_loader)
        train_ade /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        val_ade = 0
        val_fde = 0
        
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                
                if model_name in ['social_lstm', 'constant_velocity']:
                    pred = model(history)
                else:
                    env_map = torch.zeros(history.size(0), 18, 128, 128).to(device)
                    pred = model(history, env_map)
                
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                loss = criterion(pred, future)
                val_loss += loss.item()
                val_ade += torch.mean(torch.norm(pred - future, dim=-1)).item()
                val_fde += torch.mean(torch.norm(pred[:, -1] - future[:, -1], dim=-1)).item()
        
        val_loss /= len(val_loader)
        val_ade /= len(val_loader)
        val_fde /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, ADE={train_ade:.4f} | "
              f"Val Loss={val_loss:.4f}, ADE={val_ade:.4f}, FDE={val_fde:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ade': val_ade,
                'val_fde': val_fde
            }, save_dir / 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"早停：验证损失 {config['patience']} 轮未改善")
                break
    
    print(f"✓ {model_name} 训练完成，最佳验证损失: {best_val_loss:.4f}")
    return best_val_loss

def main():
    # 配置
    config = {
        'data_dir': '/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest',
        'save_dir': '/home/zmc/文档/programwork/runs',
        'batch_size': 64,  # 增大batch size充分利用GPU
        'learning_rate': 0.001,
        'num_epochs': 30,
        'val_split': 0.2,
        'patience': 5
    }
    
    # 加载数据
    print("="*60)
    print("加载数据集")
    print("="*60)
    dataset = SimpleTrajectoryDataset(config['data_dir'])
    
    if len(dataset) == 0:
        print("错误：没有生成训练样本")
        return
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * config['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 使用CPU多进程加载数据，加速训练
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    # 训练所有模型
    from models.baselines.constant_velocity import ConstantVelocity
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
    
    models = [
        # ('constant_velocity', ConstantVelocity(model_config)),  # CV没有可训练参数，跳过
        ('social_lstm', SocialLSTM(model_config)),
        ('ynet', YNet(model_config)),
        ('pecnet', PECNet(model_config)),
        ('trajectron', TrajectronPP(model_config))
    ]
    
    results = {}
    
    for model_name, model in models:
        print(f"\n{'='*60}")
        print(f"训练模型: {model_name.upper()}")
        print('='*60)
        
        try:
            best_loss = train_model(model, train_loader, val_loader, config, model_name)
            results[model_name] = {'best_val_loss': best_loss, 'status': 'success'}
        except Exception as e:
            print(f"✗ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'status': 'failed', 'error': str(e)}
    
    # 保存结果
    results_file = Path(config['save_dir']) / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("训练结果汇总")
    print('='*60)
    for model_name, result in results.items():
        if result['status'] == 'success':
            print(f"✓ {model_name:20s} 最佳验证损失: {result['best_val_loss']:.4f}")
        else:
            print(f"✗ {model_name:20s} 失败: {result.get('error', 'Unknown')}")
    
    print(f"\n结果已保存到: {results_file}")

if __name__ == '__main__':
    main()
