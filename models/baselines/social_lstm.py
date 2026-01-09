"""
Social-LSTM: Human Trajectory Prediction in Crowded Spaces
参考论文: Alahi et al. "Social LSTM: Human Trajectory Prediction in Crowded Spaces", CVPR 2016
"""
import torch
import torch.nn as nn
from .base_predictor import BasePredictor
from typing import Dict


class SocialLSTM(BasePredictor):
    """
    Social-LSTM 简化实现
    注意: 原始Social-LSTM主要用于行人轨迹预测，这里适配为单目标预测
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.input_dim = 2
        self.hidden_dim = config.get('hidden_dim', 128)
        self.embedding_dim = config.get('embedding_dim', 64)
        
        # 位置嵌入
        self.input_embedding = nn.Linear(self.input_dim, self.embedding_dim)
        
        # LSTM编码器
        self.encoder_lstm = nn.LSTM(
            self.embedding_dim, 
            self.hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # LSTM解码器
        self.decoder_lstm = nn.LSTM(
            self.embedding_dim, 
            self.hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(self.hidden_dim, 2)

    def forward(self, history: torch.Tensor, env_map: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Social-LSTM不使用环境地图，仅基于历史轨迹预测
        """
        batch_size = history.size(0)
        
        # 嵌入历史轨迹
        embedded = self.input_embedding(history)
        
        # 编码
        _, (h_n, c_n) = self.encoder_lstm(embedded)
        
        # 解码
        curr_pos = history[:, -1, :]
        predictions = []
        
        for _ in range(self.future_length):
            # 嵌入当前位置
            curr_embedded = self.input_embedding(curr_pos).unsqueeze(1)
            
            # LSTM步进
            out, (h_n, c_n) = self.decoder_lstm(curr_embedded, (h_n, c_n))
            
            # 预测位移
            delta = self.output_layer(out.squeeze(1))
            curr_pos = curr_pos + delta
            predictions.append(curr_pos.unsqueeze(1))
        
        return torch.cat(predictions, dim=1)

    def predict(self, history: torch.Tensor, env_map: torch.Tensor = None, num_samples: int = 1) -> torch.Tensor:
        pred = self.forward(history, env_map)
        return pred.unsqueeze(1).repeat(1, num_samples, 1, 1)
