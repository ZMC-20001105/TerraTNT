"""
TerraTNT模型训练框架
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, Optional, Tuple

from models.terratnt import TerraTNT
from utils.data_processing.trajectory_preprocessor import create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrajectoryLoss(nn.Module):
    """轨迹预测损失函数"""
    
    def __init__(self, goal_weight: float = 0.3, ade_weight: float = 0.5, fde_weight: float = 0.2):
        """
        Args:
            goal_weight: 目标分类损失权重
            ade_weight: 平均位移误差权重
            fde_weight: 最终位移误差权重
        """
        super().__init__()
        self.goal_weight = goal_weight
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        
        self.nll_loss = nn.NLLLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, 
                ground_truth: torch.Tensor,
                goal_probs: torch.Tensor,
                goal_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: (batch, seq_len, 2) 预测轨迹
            ground_truth: (batch, seq_len, 2) 真实轨迹
            goal_probs: (batch, num_goals) 目标概率
            goal_labels: (batch,) 真实目标索引
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 目标分类损失 (NLL)
        goal_loss = self.nll_loss(torch.log(goal_probs + 1e-8), goal_labels)
        
        # ADE (Average Displacement Error)
        displacement = torch.norm(predictions - ground_truth, dim=2)  # (batch, seq_len)
        ade = displacement.mean()
        
        # FDE (Final Displacement Error)
        fde = torch.norm(predictions[:, -1, :] - ground_truth[:, -1, :], dim=1).mean()
        
        # 总损失
        total_loss = (self.goal_weight * goal_loss + 
                     self.ade_weight * ade + 
                     self.fde_weight * fde)
        
        loss_dict = {
            'total': total_loss.item(),
            'goal': goal_loss.item(),
            'ade': ade.item(),
            'fde': fde.item()
        }
        
        return total_loss, loss_dict


class TerraTNTTrainer:
    """TerraTNT训练器"""
    
    def __init__(self,
                 model: TerraTNT,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 num_epochs: int = 100,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'runs'):
        """
        Args:
            model: TerraTNT模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
            learning_rate: 学习率
            num_epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            log_dir: TensorBoard日志目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 损失函数
        self.criterion = TrajectoryLoss()
        
        # 检查点和日志
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'{log_dir}/terratnt_{timestamp}')
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        logger.info("=" * 60)
        logger.info("TerraTNT训练器初始化完成")
        logger.info(f"  设备: {device}")
        logger.info(f"  学习率: {learning_rate}")
        logger.info(f"  训练批次: {len(train_loader)}")
        logger.info(f"  验证批次: {len(val_loader)}")
        logger.info("=" * 60)
    
    def _generate_candidate_goals(self, batch_size: int, num_goals: int = 100) -> torch.Tensor:
        """
        生成候选目标点（简化版本，实际应该基于环境采样）
        
        Returns:
            (batch, num_goals, 2) 候选目标
        """
        # 在[-50, 50]km范围内随机采样
        candidates = torch.randn(batch_size, num_goals, 2) * 25.0
        return candidates.to(self.device)
    
    def _find_closest_goal(self, true_goal: torch.Tensor, 
                          candidates: torch.Tensor) -> torch.Tensor:
        """
        找到最接近真实目标的候选目标索引
        
        Args:
            true_goal: (batch, 2) 真实目标
            candidates: (batch, num_goals, 2) 候选目标
            
        Returns:
            (batch,) 最近候选目标的索引
        """
        # 计算距离
        distances = torch.norm(candidates - true_goal.unsqueeze(1), dim=2)  # (batch, num_goals)
        closest_idx = torch.argmin(distances, dim=1)
        return closest_idx
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'goal': 0.0,
            'ade': 0.0,
            'fde': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            env_map = batch['env_map'].to(self.device)
            history = batch['history'].to(self.device)
            future = batch['future'].to(self.device)
            goal = batch['goal'].to(self.device)
            current_pos = batch['current_pos'].to(self.device)
            
            batch_size = env_map.size(0)
            
            # 生成候选目标
            candidate_goals = self._generate_candidate_goals(batch_size)
            
            # 找到最接近真实目标的候选
            goal_labels = self._find_closest_goal(goal, candidate_goals)
            
            # 前向传播
            predictions, goal_probs = self.model(
                env_map=env_map,
                history=history,
                candidate_goals=candidate_goals,
                current_pos=current_pos,
                teacher_forcing_ratio=0.5,
                ground_truth=future
            )
            
            # 计算损失
            loss, loss_dict = self.criterion(predictions, future, goal_probs, goal_labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累积损失
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            
            # 日志
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch [{batch_idx+1}/{num_batches}] "
                          f"Loss: {loss_dict['total']:.4f} "
                          f"(Goal: {loss_dict['goal']:.4f}, "
                          f"ADE: {loss_dict['ade']:.4f}, "
                          f"FDE: {loss_dict['fde']:.4f})")
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'goal': 0.0,
            'ade': 0.0,
            'fde': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                env_map = batch['env_map'].to(self.device)
                history = batch['history'].to(self.device)
                future = batch['future'].to(self.device)
                goal = batch['goal'].to(self.device)
                current_pos = batch['current_pos'].to(self.device)
                
                batch_size = env_map.size(0)
                
                candidate_goals = self._generate_candidate_goals(batch_size)
                goal_labels = self._find_closest_goal(goal, candidate_goals)
                
                predictions, goal_probs = self.model(
                    env_map=env_map,
                    history=history,
                    candidate_goals=candidate_goals,
                    current_pos=current_pos,
                    teacher_forcing_ratio=0.0,
                    ground_truth=None
                )
                
                loss, loss_dict = self.criterion(predictions, future, goal_probs, goal_labels)
                
                for key in val_losses:
                    val_losses[key] += loss_dict[key]
        
        # 平均损失
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self, early_stopping_patience: int = 10):
        """完整训练流程"""
        logger.info("\n" + "=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)
        
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            
            logger.info(f"\nEpoch [{self.current_epoch}/{self.num_epochs}]")
            
            # 训练
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            logger.info(f"  训练损失: Total={train_losses['total']:.4f}, "
                       f"Goal={train_losses['goal']:.4f}, "
                       f"ADE={train_losses['ade']:.4f}, "
                       f"FDE={train_losses['fde']:.4f}")
            
            # 验证
            val_losses = self.validate()
            self.val_losses.append(val_losses)
            
            logger.info(f"  验证损失: Total={val_losses['total']:.4f}, "
                       f"Goal={val_losses['goal']:.4f}, "
                       f"ADE={val_losses['ade']:.4f}, "
                       f"FDE={val_losses['fde']:.4f}")
            
            # TensorBoard记录
            for key in train_losses:
                self.writer.add_scalar(f'Loss/train_{key}', train_losses[key], self.current_epoch)
                self.writer.add_scalar(f'Loss/val_{key}', val_losses[key], self.current_epoch)
            
            self.writer.add_scalar('Learning_Rate', 
                                  self.optimizer.param_groups[0]['lr'], 
                                  self.current_epoch)
            
            # 学习率调度
            self.scheduler.step(val_losses['total'])
            
            # 保存最佳模型
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pth')
                logger.info(f"  ✓ 保存最佳模型 (验证损失: {self.best_val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (self.current_epoch) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pth')
            
            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f"\n早停触发 (patience={early_stopping_patience})")
                break
        
        logger.info("\n" + "=" * 60)
        logger.info("训练完成")
        logger.info(f"  最佳验证损失: {self.best_val_loss:.4f}")
        logger.info("=" * 60)
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        logger.info(f"✓ 加载检查点: {filename} (Epoch {self.current_epoch})")


def main():
    """主训练函数"""
    # 配置
    region = 'scottish_highlands'
    trajectory_dir = Path(f'/home/zmc/文档/programwork/data/processed/synthetic_trajectories/{region}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders(
        region=region,
        trajectory_dir=trajectory_dir,
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15,
        num_workers=4
    )
    
    # 创建模型
    logger.info("创建TerraTNT模型...")
    model = TerraTNT(
        env_channels=18,
        env_feature_dim=256,
        history_hidden_dim=128,
        decoder_hidden_dim=256,
        num_goals=100,
        output_length=60
    )
    
    # 创建训练器
    trainer = TerraTNTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        num_epochs=100,
        checkpoint_dir='checkpoints/terratnt',
        log_dir='runs'
    )
    
    # 开始训练
    trainer.train(early_stopping_patience=10)


if __name__ == '__main__':
    main()
