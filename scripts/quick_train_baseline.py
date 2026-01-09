#!/usr/bin/env python
"""
快速开始基线模型训练 - 使用Bohemian现有数据
"""
import sys
sys.path.insert(0, '/home/zmc/文档/programwork')

import torch
from pathlib import Path
from training.baseline_trainer import train_baseline

def main():
    # 使用Bohemian现有数据
    data_dir = Path('/home/zmc/文档/programwork/data/processed/synthetic_trajectories/bohemian_forest')
    
    # 检查数据
    pkl_files = list(data_dir.glob('*.pkl'))
    print(f"找到 {len(pkl_files)} 条轨迹数据")
    
    if len(pkl_files) < 100:
        print("数据不足，退出")
        return
    
    # 训练配置
    config = {
        'data_dir': str(data_dir),
        'region': 'bohemian_forest',
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'history_len': 10,
        'future_len': 60,
        'val_split': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '/home/zmc/文档/programwork/runs',
        'early_stopping_patience': 10
    }
    
    print(f"\n训练配置:")
    print(f"  数据: {len(pkl_files)} 条轨迹")
    print(f"  设备: {config['device']}")
    print(f"  批大小: {config['batch_size']}")
    print(f"  训练轮数: {config['num_epochs']}")
    print(f"  验证集比例: {config['val_split']}")
    
    # 按顺序训练所有基线模型
    models = ['constant_velocity', 'social_lstm', 'ynet', 'pecnet', 'trajectron']
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"开始训练: {model_name.upper()}")
        print('='*60)
        
        try:
            train_baseline(model_name, config)
            print(f"✓ {model_name} 训练完成")
        except Exception as e:
            print(f"✗ {model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有基线模型训练完成")
    print('='*60)

if __name__ == '__main__':
    main()
