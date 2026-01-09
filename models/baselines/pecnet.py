import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_predictor import BasePredictor
from typing import Dict, Tuple, Optional

class PECNet(BasePredictor):
    """
    PECNet: Predicted Endpoint Conditioned Network
    参考论文: Mangalam et al. "PECNet: Trajectory Prediction with Planning-based 
    Endpoint Conditioned Network", CVPR 2020
    """
    def __init__(self, config: Dict):
        super(PECNet, self).__init__(config)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.latent_dim = config.get('latent_dim', 16)
        
        # 1. 轨迹编码器
        self.traj_encoder = nn.Sequential(
            nn.Linear(self.history_length * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim),
            nn.ReLU()
        )
        
        # 2. CVAE 终点预测器
        # Encoder (q(z|history, goal))
        self.q_z = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim * 2) # mean and log_var
        )
        
        # Decoder (p(goal|history, z))
        self.p_goal = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        # 3. 轨迹预测器 (Predictor)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.future_length * 2)
        )

    def forward(self, history: torch.Tensor, env_map: torch.Tensor, goal: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = history.size(0)
        
        # 编码历史轨迹
        flat_history = history.view(batch_size, -1)
        h_feat = self.traj_encoder(flat_history)
        
        # CVAE 采样或推理
        if goal is not None:
            # 训练阶段: 使用真实终点进行编码
            # 确保goal维度正确 (B, 2) -> (B, 2)
            if goal.dim() == 3:
                goal = goal[:, 0, :]  # 取第一个候选目标
            z_params = self.q_z(torch.cat([h_feat, goal], dim=1))
            mu, log_var = torch.chunk(z_params, 2, dim=1)
            z = self._reparameterize(mu, log_var)
        else:
            # 推理阶段: 从先验 N(0, 1) 采样
            device = h_feat.device
            mu, log_var = torch.zeros(batch_size, self.latent_dim, device=device), torch.zeros(batch_size, self.latent_dim, device=device)
            z = torch.randn(batch_size, self.latent_dim, device=device)
            
        # 预测终点
        pred_goal = self.p_goal(torch.cat([h_feat, z], dim=1))
        
        # 预测完整轨迹
        pred_traj_flat = self.predictor(torch.cat([h_feat, pred_goal], dim=1))
        pred_traj = pred_traj_flat.view(batch_size, self.future_length, 2)
        
        return pred_traj, mu, log_var

    def predict(self, history: torch.Tensor, env_map: torch.Tensor, num_samples: int = 20) -> torch.Tensor:
        batch_size = history.size(0)
        flat_history = history.view(batch_size, -1)
        h_feat = self.traj_encoder(flat_history)
        
        all_samples = []
        for _ in range(num_samples):
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            pred_goal = self.p_goal(torch.cat([h_feat, z], dim=1))
            pred_traj_flat = self.predictor(torch.cat([h_feat, pred_goal], dim=1))
            all_samples.append(pred_traj_flat.view(batch_size, 1, self.future_length, 2))
            
        return torch.cat(all_samples, dim=1)

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
