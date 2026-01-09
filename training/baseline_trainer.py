"""
基线模型统一训练框架
支持 TerraTNT, YNet, PECNet, Trajectron++, Social-LSTM 等模型的统一训练接口
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baselines.base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class TrajectoryLoss(nn.Module):
    """轨迹预测统一损失函数"""
    
    def __init__(self, ade_weight: float = 1.0, fde_weight: float = 1.0):
        super().__init__()
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: 预测轨迹 (B, T, 2)
            target: 真实轨迹 (B, T, 2)
        Returns:
            损失字典
        """
        # ADE: Average Displacement Error
        ade = torch.mean(torch.norm(pred - target, dim=-1))
        
        # FDE: Final Displacement Error
        fde = torch.mean(torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1))
        
        # 总损失
        total_loss = self.ade_weight * ade + self.fde_weight * fde
        
        return {
            'total': total_loss,
            'ade': ade,
            'fde': fde
        }


class BaselineTrainer:
    """基线模型统一训练器"""
    
    def __init__(
        self,
        model: BasePredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 损失函数
        self.criterion = TrajectoryLoss(
            ade_weight=config.get('ade_weight', 1.0),
            fde_weight=config.get('fde_weight', 1.0)
        )
        
        # 训练配置
        self.epochs = config.get('epochs', 100)
        self.early_stop_patience = config.get('early_stop_patience', 15)
        
        # 日志和检查点
        self.model_name = model.__class__.__name__
        self.run_dir = Path(config.get('run_dir', 'runs')) / f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.run_dir / 'tensorboard')
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 保存配置
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Trainer initialized for {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Run directory: {self.run_dir}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            history = batch['history'].to(self.device)
            future = batch['future'].to(self.device)
            env_map = batch['env_map'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            pred = self.model(history, env_map)
            
            # 计算损失
            losses = self.criterion(pred, future)
            
            # 反向传播
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += losses['total'].item()
            total_ade += losses['ade'].item()
            total_fde += losses['fde'].item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                           f"Loss: {losses['total'].item():.4f} "
                           f"ADE: {losses['ade'].item():.2f}m "
                           f"FDE: {losses['fde'].item():.2f}m")
        
        return {
            'loss': total_loss / num_batches,
            'ade': total_ade / num_batches,
            'fde': total_fde / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            history = batch['history'].to(self.device)
            future = batch['future'].to(self.device)
            env_map = batch['env_map'].to(self.device)
            
            pred = self.model(history, env_map)
            losses = self.criterion(pred, future)
            
            total_loss += losses['total'].item()
            total_ade += losses['ade'].item()
            total_fde += losses['fde'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'ade': total_ade / num_batches,
            'fde': total_fde / num_batches
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.run_dir / 'checkpoint_latest.pt')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.run_dir / 'checkpoint_best.pt')
            logger.info(f"✓ Saved best model at epoch {epoch}")
    
    def train(self) -> Dict:
        """完整训练流程"""
        logger.info(f"Starting training for {self.epochs} epochs...")
        
        history = {'train': [], 'val': []}
        
        for epoch in range(1, self.epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            history['train'].append(train_metrics)
            
            # 验证
            val_metrics = self.validate()
            history['val'].append(val_metrics)
            
            # 学习率调度
            self.scheduler.step(val_metrics['loss'])
            
            # TensorBoard 日志
            self.writer.add_scalars('Loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)
            self.writer.add_scalars('ADE', {
                'train': train_metrics['ade'],
                'val': val_metrics['ade']
            }, epoch)
            self.writer.add_scalars('FDE', {
                'train': train_metrics['fde'],
                'val': val_metrics['fde']
            }, epoch)
            
            # 检查是否为最佳模型
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 打印进度
            logger.info(f"Epoch {epoch}/{self.epochs} | "
                       f"Train Loss: {train_metrics['loss']:.4f} | "
                       f"Val Loss: {val_metrics['loss']:.4f} | "
                       f"Val ADE: {val_metrics['ade']:.2f}m | "
                       f"Val FDE: {val_metrics['fde']:.2f}m")
            
            # 早停
            if self.patience_counter >= self.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.writer.close()
        
        # 保存训练历史
        with open(self.run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed. Best val loss: {self.best_val_loss:.4f}")
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch,
            'run_dir': str(self.run_dir)
        }


def train_baseline(
    model_name: str,
    data_dir: str,
    config: Optional[Dict] = None
) -> Dict:
    """
    训练指定的基线模型
    
    Args:
        model_name: 模型名称 ('ynet', 'pecnet', 'trajectron', 'social_lstm', 'cv')
        data_dir: 数据目录
        config: 训练配置
    
    Returns:
        训练结果
    """
    from utils.data_processing.trajectory_preprocessor import create_data_loaders
    
    # 默认配置
    default_config = {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'epochs': 100,
        'early_stop_patience': 15,
        'history_length': 10,
        'future_length': 60,
        'in_channels': 18,
        'hidden_dim': 256
    }
    
    if config:
        default_config.update(config)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=default_config['batch_size'],
        history_length=default_config['history_length'],
        future_length=default_config['future_length']
    )
    
    # 创建模型
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'ynet':
        from models.baselines.ynet import YNet
        model = YNet(default_config)
    elif model_name_lower == 'pecnet':
        from models.baselines.pecnet import PECNet
        model = PECNet(default_config)
    elif model_name_lower == 'trajectron':
        from models.baselines.trajectron import TrajectronPP
        model = TrajectronPP(default_config)
    elif model_name_lower == 'social_lstm':
        from models.baselines.social_lstm import SocialLSTM
        model = SocialLSTM(default_config)
    elif model_name_lower == 'cv':
        from models.baselines.constant_velocity import ConstantVelocity
        model = ConstantVelocity(default_config)
    elif model_name_lower == 'terratnt':
        from models.terratnt import TerraTNT
        model = TerraTNT(default_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 创建训练器并训练
    trainer = BaselineTrainer(model, train_loader, val_loader, default_config)
    result = trainer.train()
    
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # 示例：训练 YNet
    result = train_baseline(
        model_name='ynet',
        data_dir='/home/zmc/文档/programwork/data/processed/synthetic_trajectories/scottish_highlands',
        config={'epochs': 50, 'batch_size': 16}
    )
    print(f"Training result: {result}")
