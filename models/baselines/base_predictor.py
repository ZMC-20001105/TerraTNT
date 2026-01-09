import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

class BasePredictor(nn.Module, ABC):
    """
    轨迹预测模型基类，定义统一的接口
    """
    def __init__(self, config: Dict):
        super(BasePredictor, self).__init__()
        self.config = config
        self.history_length = config.get('history_length', 10)
        self.future_length = config.get('future_length', 60)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def forward(self, history: torch.Tensor, env_map: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播
        Args:
            history: 历史轨迹 (B, T_hist, 2)
            env_map: 环境地图 (B, C, H, W)
        Returns:
            predicted_trajectory: 预测轨迹 (B, T_fut, 2)
        """
        pass

    @abstractmethod
    def predict(self, history: torch.Tensor, env_map: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        推理接口，支持多模态预测
        Args:
            history: 历史轨迹 (B, T_hist, 2)
            env_map: 环境地图 (B, C, H, W)
            num_samples: 采样数量
        Returns:
            samples: 多模态预测结果 (B, num_samples, T_fut, 2)
        """
        pass

    def get_model_info(self) -> Dict:
        """获取模型基本信息"""
        return {
            "name": self.__class__.__name__,
            "parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "history_len": self.history_length,
            "future_len": self.future_length
        }
