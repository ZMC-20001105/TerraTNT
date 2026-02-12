"""
Trajectron++: Dynamically-Feasible Trajectory Forecasting
参考论文: Salzmann et al. "Trajectron++: Dynamically-Feasible Trajectory Forecasting 
With Heterogeneous Data", ECCV 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_predictor import BasePredictor
from typing import Dict, Tuple, Optional


class NodeHistoryEncoder(nn.Module):
    """节点历史轨迹编码器"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, (h_n, c_n) = self.lstm(x)
        return h_n[-1], output


class MapEncoder(nn.Module):
    """环境地图编码器 (CNN)"""
    def __init__(self, in_channels: int = 18, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class GMM2D(nn.Module):
    """2D高斯混合模型输出层"""
    def __init__(self, input_dim: int, num_modes: int = 5):
        super().__init__()
        self.num_modes = num_modes
        # 每个模式: 2个均值 + 3个协方差参数 (var_x, var_y, corr) + 1个权重
        self.output_dim = num_modes * 6
        self.fc = nn.Linear(input_dim, self.output_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        params = self.fc(x)
        params = params.view(-1, self.num_modes, 6)
        
        # 解析参数
        means = params[:, :, :2]  # (B, K, 2)
        log_vars = params[:, :, 2:4]  # (B, K, 2)
        corrs = torch.tanh(params[:, :, 4])  # (B, K)
        log_weights = F.log_softmax(params[:, :, 5], dim=1)  # (B, K)
        
        return {
            'means': means,
            'log_vars': log_vars,
            'corrs': corrs,
            'log_weights': log_weights
        }


class TrajectronPP(BasePredictor):
    """
    Trajectron++ 简化实现
    核心特点:
    1. 动力学约束解码
    2. 多模态GMM输出
    3. 环境感知
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_modes = config.get('num_modes', 5)
        self.in_channels = config.get('in_channels', 18)
        
        # 编码器
        self.history_encoder = NodeHistoryEncoder(2, self.hidden_dim)
        self.map_encoder = MapEncoder(self.in_channels, self.hidden_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 解码器 (带动力学约束的GRU)
        self.decoder_gru = nn.GRU(self.hidden_dim + 2, self.hidden_dim, batch_first=True)
        
        # GMM输出层
        self.gmm_output = GMM2D(self.hidden_dim, self.num_modes)
        
        # 动力学约束参数
        self.max_accel = config.get('max_accel', 3.0) / 1000.0  # m/s^2 -> km/s^2
        self.dt = config.get('dt', 10.0)  # 采样间隔 (秒), 实际为10s

    def forward(self, history: torch.Tensor, env_map: torch.Tensor, future: torch.Tensor = None, teacher_forcing_ratio: float = 0.0, **kwargs) -> torch.Tensor:
        batch_size = history.size(0)
        
        # 编码历史轨迹
        traj_feat, _ = self.history_encoder(history)
        
        # 编码环境地图
        map_feat = self.map_encoder(env_map)
        
        # 特征融合
        combined = torch.cat([traj_feat, map_feat], dim=1)
        fused = self.fusion(combined)
        
        # 解码 (自回归生成)
        curr_pos = history[:, -1, :]
        prev_vel = history[:, -1, :] - history[:, -2, :]
        
        h = fused.unsqueeze(0)
        predictions = []
        
        for t in range(self.future_length):
            # GRU输入: 融合特征 + 当前位置
            dec_input = torch.cat([fused, curr_pos], dim=1).unsqueeze(1)
            out, h = self.decoder_gru(dec_input, h)
            
            # GMM预测
            gmm_params = self.gmm_output(out.squeeze(1))
            
            # 取最可能的模式作为预测
            best_mode = gmm_params['log_weights'].argmax(dim=1)
            delta = gmm_params['means'][torch.arange(batch_size), best_mode]
            
            # 应用动力学约束
            delta = self._apply_dynamics(delta, prev_vel)
            
            curr_pos = curr_pos + delta
            prev_vel = delta
            predictions.append(curr_pos.unsqueeze(1))
            
            # Teacher Forcing
            if self.training and future is not None and torch.rand(1) < teacher_forcing_ratio:
                curr_pos = future[:, t, :]
                if t > 0:
                    prev_vel = future[:, t, :] - future[:, t-1, :]
                else:
                    prev_vel = future[:, 0, :] - history[:, -1, :]
        
        return torch.cat(predictions, dim=1)

    def predict(self, history: torch.Tensor, env_map: torch.Tensor, num_samples: int = 20) -> torch.Tensor:
        """多模态采样预测"""
        batch_size = history.size(0)
        
        # 编码
        traj_feat, _ = self.history_encoder(history)
        map_feat = self.map_encoder(env_map)
        combined = torch.cat([traj_feat, map_feat], dim=1)
        fused = self.fusion(combined)
        
        all_samples = []
        
        for _ in range(num_samples):
            curr_pos = history[:, -1, :]
            prev_vel = history[:, -1, :] - history[:, -2, :]
            h = fused.unsqueeze(0)
            sample_traj = []
            
            for t in range(self.future_length):
                dec_input = torch.cat([fused, curr_pos], dim=1).unsqueeze(1)
                out, h = self.decoder_gru(dec_input, h)
                gmm_params = self.gmm_output(out.squeeze(1))
                
                # 从GMM采样
                delta = self._sample_gmm(gmm_params)
                delta = self._apply_dynamics(delta, prev_vel)
                
                curr_pos = curr_pos + delta
                prev_vel = delta
                sample_traj.append(curr_pos.unsqueeze(1))
            
            all_samples.append(torch.cat(sample_traj, dim=1).unsqueeze(1))
        
        return torch.cat(all_samples, dim=1)

    def _apply_dynamics(self, delta: torch.Tensor, prev_vel: torch.Tensor) -> torch.Tensor:
        """应用动力学约束 (所有量单位: km, s)"""
        # delta, prev_vel 是每步位移 (km/step)
        # 转换为速度 (km/s)
        vel = delta / self.dt
        prev_v = prev_vel / self.dt
        # 加速度 (km/s^2)
        accel = (vel - prev_v) / self.dt
        accel_mag = torch.norm(accel, dim=-1, keepdim=True)
        
        # 限制加速度 (self.max_accel 已转换为 km/s^2)
        scale = torch.clamp(self.max_accel / (accel_mag + 1e-6), max=1.0)
        constrained_accel = accel * scale
        
        # 重新计算速度和位移
        constrained_vel = prev_v + constrained_accel * self.dt
        return constrained_vel * self.dt

    def _sample_gmm(self, gmm_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从GMM分布采样"""
        batch_size = gmm_params['means'].size(0)
        
        # 采样模式
        weights = torch.exp(gmm_params['log_weights'])
        mode_idx = torch.multinomial(weights, 1).squeeze(1)
        
        # 获取对应模式的参数
        means = gmm_params['means'][torch.arange(batch_size), mode_idx]
        log_vars = gmm_params['log_vars'][torch.arange(batch_size), mode_idx]
        
        # 采样
        std = torch.exp(0.5 * log_vars)
        eps = torch.randn_like(means)
        
        return means + eps * std
