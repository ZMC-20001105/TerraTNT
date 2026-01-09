"""
轨迹预测评估指标
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算平均位移误差 (Average Displacement Error)
    
    Args:
        pred: 预测轨迹 (B, T, 2) 或 (B, K, T, 2) 多模态
        target: 真实轨迹 (B, T, 2)
    
    Returns:
        ADE值
    """
    if pred.dim() == 4:
        # 多模态预测，取最佳轨迹
        B, K, T, _ = pred.shape
        target_expanded = target.unsqueeze(1).expand(-1, K, -1, -1)
        errors = torch.norm(pred - target_expanded, dim=-1).mean(dim=-1)  # (B, K)
        best_idx = errors.argmin(dim=1)
        pred = pred[torch.arange(B), best_idx]  # (B, T, 2)
    
    displacement = torch.norm(pred - target, dim=-1)  # (B, T)
    ade = displacement.mean()
    return ade


def compute_fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算最终位移误差 (Final Displacement Error)
    
    Args:
        pred: 预测轨迹 (B, T, 2) 或 (B, K, T, 2) 多模态
        target: 真实轨迹 (B, T, 2)
    
    Returns:
        FDE值
    """
    if pred.dim() == 4:
        B, K, T, _ = pred.shape
        target_expanded = target.unsqueeze(1).expand(-1, K, -1, -1)
        errors = torch.norm(pred[:, :, -1, :] - target_expanded[:, :, -1, :], dim=-1)  # (B, K)
        best_idx = errors.argmin(dim=1)
        pred = pred[torch.arange(B), best_idx]
    
    fde = torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1).mean()
    return fde


def compute_miss_rate(pred: torch.Tensor, target: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
    """
    计算失败率 (Miss Rate)
    FDE超过阈值的比例
    
    Args:
        pred: 预测轨迹
        target: 真实轨迹
        threshold: 阈值 (米)
    
    Returns:
        Miss Rate
    """
    if pred.dim() == 4:
        B, K, T, _ = pred.shape
        target_expanded = target.unsqueeze(1).expand(-1, K, -1, -1)
        errors = torch.norm(pred[:, :, -1, :] - target_expanded[:, :, -1, :], dim=-1)
        min_errors = errors.min(dim=1)[0]
    else:
        min_errors = torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1)
    
    miss_rate = (min_errors > threshold).float().mean()
    return miss_rate


def compute_goal_accuracy(
    pred_goal: torch.Tensor, 
    target_goal: torch.Tensor, 
    threshold: float = 5.0
) -> torch.Tensor:
    """
    计算目标预测准确率
    
    Args:
        pred_goal: 预测目标位置 (B, 2) 或 (B, K, 2)
        target_goal: 真实目标位置 (B, 2)
        threshold: 正确判定阈值 (米)
    
    Returns:
        Goal Accuracy
    """
    if pred_goal.dim() == 3:
        # 多模态，取最近的
        B, K, _ = pred_goal.shape
        target_expanded = target_goal.unsqueeze(1).expand(-1, K, -1)
        errors = torch.norm(pred_goal - target_expanded, dim=-1)
        min_errors = errors.min(dim=1)[0]
    else:
        min_errors = torch.norm(pred_goal - target_goal, dim=-1)
    
    accuracy = (min_errors < threshold).float().mean()
    return accuracy


def compute_all_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor,
    miss_threshold: float = 2.0,
    goal_threshold: float = 5.0
) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        pred: 预测轨迹 (B, T, 2) 或 (B, K, T, 2)
        target: 真实轨迹 (B, T, 2)
        miss_threshold: Miss Rate阈值
        goal_threshold: Goal Accuracy阈值
    
    Returns:
        指标字典
    """
    ade = compute_ade(pred, target).item()
    fde = compute_fde(pred, target).item()
    miss_rate = compute_miss_rate(pred, target, miss_threshold).item()
    
    # Goal accuracy使用最后一个点
    if pred.dim() == 4:
        pred_goal = pred[:, :, -1, :]
    else:
        pred_goal = pred[:, -1, :]
    target_goal = target[:, -1, :]
    goal_acc = compute_goal_accuracy(pred_goal, target_goal, goal_threshold).item()
    
    return {
        'ade': ade,
        'fde': fde,
        'miss_rate': miss_rate,
        'goal_accuracy': goal_acc
    }


class MetricsTracker:
    """指标追踪器，用于累积计算"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_ade = 0.0
        self.total_fde = 0.0
        self.total_miss = 0
        self.total_goal_correct = 0
        self.total_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, 
               miss_threshold: float = 2.0, goal_threshold: float = 5.0):
        """更新指标"""
        batch_size = target.size(0)
        
        metrics = compute_all_metrics(pred, target, miss_threshold, goal_threshold)
        
        self.total_ade += metrics['ade'] * batch_size
        self.total_fde += metrics['fde'] * batch_size
        self.total_miss += metrics['miss_rate'] * batch_size
        self.total_goal_correct += metrics['goal_accuracy'] * batch_size
        self.total_samples += batch_size
    
    def compute(self) -> Dict[str, float]:
        """计算最终指标"""
        if self.total_samples == 0:
            return {'ade': 0, 'fde': 0, 'miss_rate': 0, 'goal_accuracy': 0}
        
        return {
            'ade': self.total_ade / self.total_samples,
            'fde': self.total_fde / self.total_samples,
            'miss_rate': self.total_miss / self.total_samples,
            'goal_accuracy': self.total_goal_correct / self.total_samples
        }
