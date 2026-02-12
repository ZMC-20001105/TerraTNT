"""
基线模型实现：SimpleLSTM, Social-LSTM等
用于对比实验，验证是模型问题还是数据问题
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SimpleLSTM(nn.Module):
    """
    简单的LSTM基线模型
    只使用历史轨迹，不使用环境信息和目标信息
    用于验证基本的轨迹预测能力
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_length: int = 360,
                 dropout: float = 0.3):
        """
        Args:
            input_dim: 输入维度（通常是2，表示x,y坐标）
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_length: 预测长度（时间步数）
            dropout: Dropout比率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM编码器
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # LSTM解码器
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, 
                history: torch.Tensor,
                teacher_forcing_ratio: float = 0.0,
                ground_truth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            history: (batch, hist_len, 2) 历史轨迹delta
            teacher_forcing_ratio: 教师强制比率
            ground_truth: (batch, pred_len, 2) 真实未来轨迹（用于teacher forcing）
            
        Returns:
            predictions: (batch, pred_len, 2) 预测的未来轨迹delta
        """
        batch_size = history.size(0)
        device = history.device
        
        # 编码历史轨迹
        hist_embed = self.input_embedding(history)  # (batch, hist_len, hidden_dim)
        _, (h_n, c_n) = self.encoder_lstm(hist_embed)  # h_n: (num_layers, batch, hidden_dim)
        
        # 解码未来轨迹
        predictions = []
        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        hidden = (h_n, c_n)
        
        for t in range(self.output_length):
            # LSTM解码
            decoder_output, hidden = self.decoder_lstm(decoder_input, hidden)
            
            # 预测当前步
            pred_delta = self.output_layer(decoder_output.squeeze(1))  # (batch, 2)
            predictions.append(pred_delta)
            
            # Teacher forcing
            if self.training and ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_input = self.input_embedding(ground_truth[:, t, :]).unsqueeze(1)
            else:
                next_input = self.input_embedding(pred_delta).unsqueeze(1)
            
            decoder_input = next_input
        
        predictions = torch.stack(predictions, dim=1)  # (batch, pred_len, 2)
        return predictions


class SocialLSTM(nn.Module):
    """
    Social LSTM基线模型
    考虑社交交互的LSTM模型
    但在我们的任务中，由于是单个目标预测，简化为只使用历史信息
    """
    
    def __init__(self,
                 input_dim: int = 2,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 output_length: int = 60,
                 dropout: float = 0.0):
        """
        Args:
            input_dim: 输入维度
            embedding_dim: 嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_length: 预测长度
            dropout: Dropout比率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, embedding_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self,
                history: torch.Tensor,
                teacher_forcing_ratio: float = 0.0,
                ground_truth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            history: (batch, hist_len, 2) 历史轨迹delta
            teacher_forcing_ratio: 教师强制比率
            ground_truth: (batch, pred_len, 2) 真实未来轨迹
            
        Returns:
            predictions: (batch, pred_len, 2) 预测的未来轨迹delta
        """
        batch_size = history.size(0)
        device = history.device
        
        # 嵌入历史轨迹
        hist_embed = self.input_embedding(history)  # (batch, hist_len, embedding_dim)
        
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(hist_embed)
        
        # 使用最后一个隐藏状态进行解码
        predictions = []
        hidden = (h_n, c_n)
        prev_output = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        for t in range(self.output_length):
            # LSTM前向
            lstm_input = prev_output.unsqueeze(1)  # (batch, 1, embedding_dim)
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            
            # 预测
            pred_delta = self.output_layer(lstm_out.squeeze(1))  # (batch, 2)
            predictions.append(pred_delta)
            
            # Teacher forcing
            if self.training and ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_output = self.input_embedding(ground_truth[:, t, :])
            else:
                prev_output = self.input_embedding(pred_delta)
        
        predictions = torch.stack(predictions, dim=1)  # (batch, pred_len, 2)
        return predictions


class ConstantVelocity(nn.Module):
    """
    常速度模型（最简单的基线）
    假设目标以最后观测到的速度继续前进
    """
    
    def __init__(self, output_length: int = 60):
        super().__init__()
        self.output_length = output_length
        
    def forward(self, history: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            history: (batch, hist_len, 2) 历史轨迹delta
            
        Returns:
            predictions: (batch, pred_len, 2) 预测的未来轨迹delta
        """
        # 使用最后一个delta作为常速度
        last_delta = history[:, -1, :]  # (batch, 2)
        
        # 重复预测
        predictions = last_delta.unsqueeze(1).repeat(1, self.output_length, 1)
        
        return predictions
