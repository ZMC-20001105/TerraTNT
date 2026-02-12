import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_predictor import BasePredictor
from typing import Dict, Tuple

class UNetEncoder(nn.Module):
    """
    YNet使用的简化UNet场景编码器
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetEncoder, self).__init__()
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        
        return self.final(d1)

class YNet(BasePredictor):
    """
    YNet: Endpoint Conditioned Network
    参考论文: Mangalam et al. "It Is Not the Journey but the Destination: 
    Endpoint Conditioned Trajectory Prediction", ECCV 2020
    """
    def __init__(self, config: Dict):
        super(YNet, self).__init__(config)
        self.in_channels = config.get('in_channels', 18)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.env_coverage_km = float(config.get('env_coverage_km', 140.0))
        
        # 1. 场景语义编码器
        self.scene_encoder = UNetEncoder(self.in_channels, 64)
        
        # 2. 轨迹编码器 (LSTM)
        self.traj_encoder = nn.LSTM(2, self.hidden_dim, batch_first=True)
        
        # 3. 终点预测器 (Goal Predictor)
        self.goal_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim + 64, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 128 * 128) # 映射到地图热力图
        )
        
        # 4. 轨迹解码器 (输入: 当前位置(2) + 目标位置(2) = 4)
        self.traj_decoder = nn.LSTM(4, self.hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, 2)

    def forward(self, history: torch.Tensor, env_map: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = history.size(0)
        
        # 场景特征提取
        scene_feat = self.scene_encoder(env_map) # (B, 64, H, W)
        scene_feat_pooled = F.adaptive_avg_pool2d(scene_feat, (1, 1)).view(batch_size, -1)
        
        # 历史轨迹编码
        _, (h_n, _) = self.traj_encoder(history)
        traj_feat = h_n[-1] # (B, hidden_dim)
        
        # 融合特征
        combined_feat = torch.cat([traj_feat, scene_feat_pooled], dim=1)
        
        # 终点预测 (这里简化为直接预测坐标或高斯中心)
        goal_logits = self.goal_predictor(combined_feat)
        goal_map = goal_logits.view(batch_size, 1, 128, 128)
        
        # 为了演示，我们从goal_map中提取最大值作为预测终点
        # 实际实现中会使用KDE或多模态采样
        goal_coords_norm = self._get_max_coords(goal_map) # [0, 1] 归一化坐标
        
        # 将归一化地图坐标转换为相对物理 km 坐标（env_map 覆盖 env_coverage_km × env_coverage_km，中心在当前点）
        # 归一化 [0,1] -> [-env_coverage_km/2, env_coverage_km/2]
        goal_coords = (goal_coords_norm - 0.5) * self.env_coverage_km
        
        # 轨迹解码 (Endpoint conditioned)
        curr_pos = history[:, -1, :]
        predictions = []
        
        h_dec = traj_feat.unsqueeze(0)
        c_dec = torch.zeros_like(h_dec)
        
        for _ in range(self.future_length):
            # 将当前位置和终点坐标作为输入
            dec_input = torch.cat([curr_pos, goal_coords], dim=1).unsqueeze(1)
            out, (h_dec, c_dec) = self.traj_decoder(dec_input, (h_dec, c_dec))
            delta = self.output_layer(out.squeeze(1))
            curr_pos = curr_pos + delta
            predictions.append(curr_pos.unsqueeze(1))
            
        return torch.cat(predictions, dim=1)

    def predict(self, history: torch.Tensor, env_map: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        # 简化版预测，只返回单一路径
        pred = self.forward(history, env_map)
        return pred.unsqueeze(1).repeat(1, num_samples, 1, 1)

    def _get_max_coords(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """从热力图中提取最大值坐标"""
        B, C, H, W = heatmaps.shape
        flat_maps = heatmaps.view(B, -1)
        _, indices = torch.max(flat_maps, dim=1)
        y = (indices // W).float() / H
        x = (indices % W).float() / W
        return torch.stack([x, y], dim=1).to(heatmaps.device)
