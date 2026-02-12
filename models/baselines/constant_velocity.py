"""
Constant Velocity (CV) 基线模型
最简单的轨迹预测基线：假设目标以当前速度匀速直线运动
"""
import torch
import torch.nn as nn
from .base_predictor import BasePredictor
from typing import Dict


class ConstantVelocity(BasePredictor):
    """
    匀速直线运动模型
    预测公式: p(t) = p(0) + v * t
    其中 v 由最后两个历史点计算得出
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        # CV模型无需训练参数

    def forward(self, history: torch.Tensor, env_map: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            history: 历史轨迹 (B, T_hist, 2)
            env_map: 环境地图 (未使用)
        Returns:
            预测轨迹 (B, T_fut, 2)
        """
        batch_size = history.size(0)
        
        # 计算当前速度 (最后两点的差)
        velocity = history[:, -1, :] - history[:, -2, :]  # (B, 2)
        
        # 当前位置
        curr_pos = history[:, -1, :]  # (B, 2)
        
        # 生成未来轨迹
        predictions = []
        for t in range(1, self.future_length + 1):
            future_pos = curr_pos + velocity * t
            predictions.append(future_pos.unsqueeze(1))
        
        return torch.cat(predictions, dim=1)

    def predict(self, history: torch.Tensor, env_map: torch.Tensor = None, num_samples: int = 1) -> torch.Tensor:
        """CV模型是确定性的，所有采样结果相同"""
        pred = self.forward(history, env_map)
        return pred.unsqueeze(1).repeat(1, num_samples, 1, 1)
