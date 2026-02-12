"""
训练所有基线模型 - 10秒间隔版本
使用Bohemian Forest数据集进行公平对比
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json

# 导入基线模型
from models.baselines.constant_velocity import ConstantVelocity
from models.baselines.social_lstm import SocialLSTM
from models.baselines.ynet import YNet
from models.baselines.pecnet import PECNet
from utils.env_data_loader import EnvironmentDataLoader
from models.baselines.trajectron import TrajectronPP

# 训练配置
BATCH_SIZE = 64
NUM_WORKERS = 4  # 降低 worker 数量防止内存溢出
LEARNING_RATE = 0.001
NUM_EPOCHS = 30  # 基线不需要跑太久
HISTORY_LEN = 90
FUTURE_LEN = 360
EARLY_STOP_PATIENCE = 5

# 数据集路径
LEGACY_DEFAULT_DATA_DIR = Path('/home/zmc/文档/programwork/data/processed/synthetic_trajectories_10s/bohemian_forest')

class SimpleTrajectoryDataset(Dataset):
    """带环境地图的基线轨迹数据集"""
    def __init__(self, traj_dir, history_len=90, future_len=360, load_env_map=True):
        self.traj_dir = Path(traj_dir)
        self.history_len = history_len
        self.future_len = future_len
        self.load_env_map = load_env_map
        if load_env_map:
            self.env_loader = EnvironmentDataLoader('bohemian_forest')
        else:
            self.env_loader = None
        
        # 加载所有轨迹文件
        self.traj_files = sorted(list(self.traj_dir.glob('*.pkl')))
        print(f"找到 {len(self.traj_files)} 个轨迹文件")
        
        # 生成样本
        all_samples = []
        print("生成训练样本...")
        skipped_files = 0
        for traj_file in tqdm(self.traj_files):
            try:
                with open(traj_file, 'rb') as f:
                    traj_data = pickle.load(f)
                
                coords = None
                for key in ['path', 'trajectory', 'coords', 'full_path']:
                    if key in traj_data:
                        coords = np.array(traj_data[key])
                        break
                
                if coords is None:
                    skipped_files += 1
                    continue
                
                total_len = len(coords)
                if total_len <= history_len + future_len:
                    skipped_files += 1
                    continue
                
                # 抽样间隔
                step = 60 
                for i in range(0, total_len - history_len - future_len, step):
                    history = coords[i:i+history_len]
                    future = coords[i+history_len:i+history_len+future_len]
                    
                    all_samples.append({
                        'history': history,
                        'future': future,
                        'current_pos_abs': history[-1].copy()
                    })
            except:
                skipped_files += 1
                continue
        
        # 增加采样量至 10000 以确保基线充分收敛
        import random
        random.seed(42)
        if len(all_samples) > 10000:
            sample_indices = random.sample(range(len(all_samples)), 10000)
            self.samples = [all_samples[i] for i in sorted(sample_indices)]
        else:
            self.samples = all_samples
            
        print(f"最终样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        current_pos_abs = sample['current_pos_abs']
        
        # 相对坐标缩放至 km
        history = torch.FloatTensor(sample['history'] - current_pos_abs) / 1000.0
        future = torch.FloatTensor(sample['future'] - current_pos_abs) / 1000.0
        
        # 仅为需要环境信息的模型提取地图
        if self.load_env_map:
            try:
                env_map = self.env_loader.extract_patch(
                    center_utm=(float(current_pos_abs[0]), float(current_pos_abs[1])),
                    patch_size=128
                )
            except:
                env_map = torch.zeros(18, 128, 128)
        else:
            env_map = torch.zeros(18, 128, 128)  # 占位符
            
        return {
            'history': history,
            'future': future,
            'env_map': env_map,
            'goal': future[-1]
        }


def train_model(model, model_name, train_loader, val_loader, device):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"训练 {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    
    # 检查是否有可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    has_params = len(trainable_params) > 0
    
    if has_params:
        optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    else:
        print(f"ℹ {model_name} 没有可训练参数，将仅进行评估")
    
    criterion = nn.MSELoss()
    
    best_val_ade = float('inf')
    patience_counter = 0
    
    # Teacher Forcing 衰减配置
    tf_ratio = 1.0
    tf_decay = 0.9
    
    run_dir = Path(f'runs/{model_name}_10s') / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        # 训练/评估
        if has_params:
            model.train()
        else:
            model.eval()
            
        train_loss = 0
        train_ade = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            env_map = batch['env_map'].to(device)
            goal = batch['goal'].to(device)
            
            if has_params:
                optimizer.zero_grad()
            
            # 前向传播 - 统一接口处理
            try:
                if model_name == 'cv':
                    pred = model(history)
                elif model_name == 'ynet':
                    pred = model(history, env_map)
                elif model_name == 'trajectron':
                    # Trajectron++ 通常也需要 future 进行训练
                    pred = model(history, env_map, future=future if model.training else None)
                elif model_name == 'social_lstm':
                    # 使用 Teacher Forcing
                    pred = model(history, future=future, teacher_forcing_ratio=tf_ratio)
                elif model_name == 'pecnet':
                    # PECNet 训练阶段返回 (pred, mu, log_var)
                    pred, mu, log_var = model(history, env_map, goal=goal)
                    # CVAE 还需要 KL 散度损失
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / history.size(0)
                else:
                    pred = model(history, goal=goal)
            except Exception as e:
                # 兜底：尝试不带参数调用
                try:
                    pred = model(history)
                except:
                    print(f"Forward error for {model_name}: {e}")
                    continue
            
            # 计算损失
            loss = criterion(pred, future)
            if model_name == 'pecnet':
                loss = loss + 0.01 * kl_loss # 加入 KL 散度约束
            
            # 计算ADE (结果乘以 1000 转换回米进行显示)
            ade = torch.mean(torch.norm(pred - future, dim=-1)) * 1000.0
            
            if has_params and loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_ade += ade.item()
            
            # 如果没有参数，只跑一个 batch 看看训练集表现即可（可选，这里为了进度条跑完）
        
        train_loss /= len(train_loader)
        train_ade /= len(train_loader)
        
        # 每个 Epoch 衰减 Teacher Forcing 比例
        tf_ratio *= tf_decay
        
        # 如果模型没有参数，实际上训练和验证表现是一样的，跑一次 Epoch 即可
        if not has_params and epoch > 0:
            break
        
        # 验证
        model.eval()
        val_loss = 0
        val_ade = 0
        
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                env_map = batch['env_map'].to(device)
                goal = batch['goal'].to(device)
                
                if model_name == 'cv':
                    pred = model(history)
                elif model_name in ['trajectron', 'ynet']:
                    pred = model(history, env_map)
                elif model_name == 'pecnet':
                    # PECNet 推理阶段也返回 (pred, mu, log_var)
                    pred, _, _ = model(history, env_map) 
                else:
                    pred = model(history, goal=goal)
                
                loss = criterion(pred, future)
                # ADE 转换回米进行显示
                ade = torch.mean(torch.norm(pred - future, dim=-1)) * 1000.0
                
                val_loss += loss.item()
                val_ade += ade.item()
        
        val_loss /= len(val_loader)
        val_ade /= len(val_loader)
        
        print(f"Epoch {epoch+1}: 训练Loss={train_loss:.4f}, ADE={train_ade:.1f}m | "
              f"验证Loss={val_loss:.4f}, ADE={val_ade:.1f}m")
        
        # 保存最佳模型
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            patience_counter = 0
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_ade': val_ade
            }
            if has_params:
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
                
            torch.save(save_dict, run_dir / 'best_model.pth')
            
            print(f"  ✓ 保存最佳模型 (ADE={val_ade:.1f}m)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"早停于Epoch {epoch+1}")
                break
    
    print(f"✓ {model_name} 训练完成，最佳验证ADE={best_val_ade:.1f}m")
    
    return {
        'model_name': model_name,
        'best_val_ade': best_val_ade,
        'run_dir': str(run_dir)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=str(LEGACY_DEFAULT_DATA_DIR), help='legacy demo 数据路径（synthetic_trajectories_10s）')
    parser.add_argument('--use_legacy_synth', action='store_true', help='必须显式开启才允许运行 legacy demo 数据管线，避免误用')
    args = parser.parse_args()

    if not bool(args.use_legacy_synth):
        print('❌ 该脚本属于 legacy demo 管线（synthetic_trajectories_10s），默认禁止运行。', flush=True)
        print('   如确需使用，请显式添加 --use_legacy_synth 并确认数据路径。', flush=True)
        print(f"   默认 legacy data_dir: {LEGACY_DEFAULT_DATA_DIR}", flush=True)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 定义要训练的模型（使用config字典初始化）
    models_to_train = [
        ('cv', ConstantVelocity({'history_length': HISTORY_LEN, 'future_length': FUTURE_LEN}), False),  # 不需要环境地图
        ('social_lstm', SocialLSTM({'history_length': HISTORY_LEN, 'future_length': FUTURE_LEN, 'hidden_dim': 128}), False),  # 不需要环境地图
        ('ynet', YNet({'history_length': HISTORY_LEN, 'future_length': FUTURE_LEN, 'in_channels': 18, 'hidden_dim': 256}), True),  # 需要环境地图
        ('pecnet', PECNet({'history_length': HISTORY_LEN, 'future_length': FUTURE_LEN, 'hidden_dim': 512}), True),  # 需要环境地图
        ('trajectron', TrajectronPP({'history_length': HISTORY_LEN, 'future_length': FUTURE_LEN, 'hidden_dim': 128}), True)  # 需要环境地图
    ]
    
    results = []
    
    for model_name, model, need_env_map in models_to_train:
        try:
            # 为每个模型创建专用数据集
            print(f"\n加载数据集 (模型: {model_name}, 环境地图: {need_env_map})...")
            full_dataset = SimpleTrajectoryDataset(Path(str(args.data_dir)), HISTORY_LEN, FUTURE_LEN, load_env_map=need_env_map)
            
            # 划分训练/验证集 (80/20)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            print(f"训练集: {len(train_dataset)} 样本")
            print(f"验证集: {len(val_dataset)} 样本")
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True, 
                num_workers=NUM_WORKERS
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=NUM_WORKERS
            )
            
            result = train_model(model, model_name, train_loader, val_loader, device)
            results.append(result)
        except Exception as e:
            print(f"✗ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果汇总
    print("\n" + "="*60)
    print("训练结果汇总")
    print("="*60)
    
    for result in results:
        print(f"{result['model_name']}: 最佳ADE={result['best_val_ade']:.1f}m")
    
    # 保存到JSON
    with open('results/baseline_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ 所有基线模型训练完成！")


if __name__ == '__main__':
    main()
