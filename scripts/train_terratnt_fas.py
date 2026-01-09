import sys
from pathlib import Path
# 自动添加项目根目录到 sys.path
root_dir = Path(__file__).parent.parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from datetime import datetime

from models.terratnt import TerraTNT
from utils.data_processing.trajectory_preprocessor import TrajectoryDataset, create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_fas_stage(stage='FAS1', num_epochs=50, batch_size=32, lr=3e-4):
    logger.info(f"开始训练 {stage} 阶段任务...")
    
    region = 'scottish_highlands'
    traj_dir = Path(f'/home/zmc/文档/programwork/data/processed/synthetic_trajectories/{region}')
    save_dir = Path(f'/home/zmc/文档/programwork/checkpoints/terratnt_{stage.lower()}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 数据准备
    # 为每个阶段创建特定的Dataset实例
    dataset = TrajectoryDataset(
        region=region,
        trajectory_dir=traj_dir,
        fas_stage=stage,
        sampling_interval=10 # 增加采样频率，获取更多样本
    )
    
    if len(dataset) == 0:
        logger.error(f"数据集为空！请检查目录: {traj_dir}")
        return
    
    # 划分训练/验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    if train_size == 0 or val_size == 0:
        logger.error(f"样本量不足以进行划分: 总数={len(dataset)}")
        return
        
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 2. 模型初始化
    config = {
        'env_channels': 18,
        'history_steps': 10,
        'future_steps': 60,
        'hidden_dim': 128,
        'num_candidates': 6 if stage == 'FAS3' else 1
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TerraTNT(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 对于FAS1和FAS2，主要训练轨迹生成（未来坐标预测）
    # 对于FAS3，除了轨迹生成，还要训练目标分类器
    criterion_traj = nn.MSELoss()
    criterion_goal = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 将数据移至GPU
            env_map = batch['env_map'].to(device)
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            goal_gt = batch['goal'].to(device)
            
            # 前向传播
            # 简化逻辑：这里假设model.forward根据输入决定训练哪个头
            outputs = model(env_map, history)
            
            loss = 0
            if stage in ['FAS1', 'FAS2']:
                # 只有轨迹生成损失
                loss = criterion_traj(outputs['traj'], future)
            elif stage == 'FAS3':
                # 包含目标分类和轨迹生成
                # 假设 outputs['goal_logits'] 是对候选目标的打分
                # 真值总是索引0 (在preprocessor里定义的)
                target_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
                loss_goal = criterion_goal(outputs['goal_logits'], target_idx)
                loss_traj = criterion_traj(outputs['traj'], future)
                loss = loss_goal + loss_traj
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证逻辑
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                env_map = batch['env_map'].to(device)
                history = batch['history'].to(device)
                future = batch['future'].to(device)
                
                outputs = model(env_map, history)
                if stage in ['FAS1', 'FAS2']:
                    loss = criterion_traj(outputs['traj'], future)
                elif stage == 'FAS3':
                    target_idx = torch.zeros(env_map.size(0), dtype=torch.long).to(device)
                    loss = criterion_goal(outputs['goal_logits'], target_idx) + criterion_traj(outputs['traj'], future)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            logger.info(f"✓ 保存最佳模型: {save_dir / 'best_model.pth'}")

if __name__ == "__main__":
    # 按顺序执行阶段训练，或者并行启动
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='FAS1', choices=['FAS1', 'FAS2', 'FAS3'])
    args = parser.parse_args()
    
    train_fas_stage(stage=args.stage)
