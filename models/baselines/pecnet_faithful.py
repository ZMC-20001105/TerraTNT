"""
PECNet: Predicted Endpoint Conditioned Network (Faithful Implementation)

Reference: Mangalam et al., "It Is Not the Journey but the Destination:
Endpoint Conditioned Trajectory Prediction", ECCV 2020.

Key architecture:
1. Past trajectory encoder (MLP) -> past feature
2. Destination encoder (MLP) -> dest feature  
3. CVAE: encoder_latent(past_feat, dest_feat) -> mu, logvar -> z
4. Decoder: (past_feat, z) -> predicted destination
5. Trajectory predictor: (past_feat, dest_feat) -> intermediate waypoints
6. Social pooling (optional, omitted for single-agent setting)

Adapted for FASDataset:
- Input: history (B, 90, 26) with xy in first 2 dims, env_map (B, 18, 128, 128)
- Output: predicted positions (B, 360, 2) in cumulative relative km
- Uses env_map via CNN encoder for environment-aware prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, ...] = (512, 256),
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EnvEncoder(nn.Module):
    """Simple CNN to extract a feature vector from the 18-channel env map."""
    def __init__(self, in_channels: int = 18, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        feat = self.conv(x).flatten(1)
        return self.fc(feat)


class PECNetFaithful(nn.Module):
    """
    Faithful PECNet adapted for single-agent, environment-aware trajectory prediction.
    
    Training: provide goal (ground truth destination) for CVAE training.
    Inference: sample z from prior, decode destination, then predict full trajectory.
    """
    def __init__(
        self,
        history_len: int = 90,
        future_len: int = 360,
        history_dim: int = 2,
        fdim: int = 256,
        zdim: int = 32,
        sigma: float = 1.3,
        dropout: float = 0.1,
        use_env: bool = True,
        env_channels: int = 18,
        env_dim: int = 128,
    ):
        super().__init__()
        self.history_len = history_len
        self.future_len = future_len
        self.fdim = fdim
        self.zdim = zdim
        self.sigma = sigma
        self.use_env = use_env

        # Past trajectory encoder
        past_input_dim = history_len * history_dim
        self.encoder_past = MLP(past_input_dim, fdim, (512, 256), dropout=dropout)

        # Environment encoder (optional)
        env_feat_dim = env_dim if use_env else 0
        if use_env:
            self.env_encoder = EnvEncoder(env_channels, env_dim)

        # Destination encoder
        self.encoder_dest = MLP(2, fdim, (256, 128), dropout=dropout)

        # CVAE latent encoder: (past_feat + dest_feat + env_feat) -> mu, logvar
        self.encoder_latent = MLP(
            fdim + fdim + env_feat_dim, zdim * 2, (256, 128), dropout=dropout
        )

        # Destination decoder: (past_feat + z + env_feat) -> predicted destination (2D)
        self.decoder_dest = MLP(
            fdim + zdim + env_feat_dim, 2, (256, 128), dropout=dropout
        )

        # Full trajectory predictor: (past_feat + dest_feat + env_feat) -> intermediate points
        # Predicts (future_len - 1) intermediate points, then appends destination
        self.predictor = MLP(
            fdim + fdim + env_feat_dim,
            (future_len - 1) * 2,
            (1024, 512, 256),
            dropout=dropout,
        )

    def _encode_past(self, history_xy: torch.Tensor) -> torch.Tensor:
        """history_xy: (B, history_len, 2) -> (B, fdim)"""
        B = history_xy.size(0)
        return self.encoder_past(history_xy.reshape(B, -1))

    def _encode_env(self, env_map: torch.Tensor) -> torch.Tensor:
        """env_map: (B, 18, 128, 128) -> (B, env_dim)"""
        if self.use_env:
            return self.env_encoder(env_map)
        return None

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        history: torch.Tensor,
        env_map: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            history: (B, 90, 26) or (B, 90, 2) - history features
            env_map: (B, 18, 128, 128) - environment map
            goal: (B, 2) - ground truth destination (cumulative position, km)
                  If None, sample from prior (inference mode).
        Returns:
            pred_pos: (B, future_len, 2) - predicted cumulative positions (km)
            mu: (B, zdim)
            logvar: (B, zdim)
        """
        B = history.size(0)

        # Extract xy from history (first 2 dims of 26-dim features)
        if history.dim() == 3 and history.size(-1) > 2:
            history_xy = history[:, :, :2]
        else:
            history_xy = history

        past_feat = self._encode_past(history_xy)
        env_feat = self._encode_env(env_map)

        if goal is not None:
            # Training: use ground truth destination for CVAE
            dest_feat = self.encoder_dest(goal)
            cat_list = [past_feat, dest_feat]
            if env_feat is not None:
                cat_list.append(env_feat)
            latent = self.encoder_latent(torch.cat(cat_list, dim=1))
            mu, logvar = torch.chunk(latent, 2, dim=1)
            z = self._reparameterize(mu, logvar)
        else:
            # Inference: sample from prior
            mu = torch.zeros(B, self.zdim, device=history.device)
            logvar = torch.zeros(B, self.zdim, device=history.device)
            z = torch.randn(B, self.zdim, device=history.device) * self.sigma

        # Decode destination
        dec_list = [past_feat, z]
        if env_feat is not None:
            dec_list.append(env_feat)
        pred_dest = self.decoder_dest(torch.cat(dec_list, dim=1))  # (B, 2)

        # Predict full trajectory conditioned on predicted destination
        dest_feat_pred = self.encoder_dest(pred_dest)
        traj_list = [past_feat, dest_feat_pred]
        if env_feat is not None:
            traj_list.append(env_feat)
        pred_mid_flat = self.predictor(torch.cat(traj_list, dim=1))
        pred_mid = pred_mid_flat.view(B, self.future_len - 1, 2)

        # Concatenate intermediate + destination
        pred_pos = torch.cat([pred_mid, pred_dest.unsqueeze(1)], dim=1)

        return pred_pos, mu, logvar

    def predict(self, history: torch.Tensor, env_map: torch.Tensor,
                num_samples: int = 20) -> torch.Tensor:
        """
        Multi-sample inference (best-of-K).
        Returns: (B, num_samples, future_len, 2)
        """
        B = history.size(0)
        if history.dim() == 3 and history.size(-1) > 2:
            history_xy = history[:, :, :2]
        else:
            history_xy = history

        past_feat = self._encode_past(history_xy)
        env_feat = self._encode_env(env_map)

        all_samples = []
        for _ in range(num_samples):
            z = torch.randn(B, self.zdim, device=history.device) * self.sigma
            dec_list = [past_feat, z]
            if env_feat is not None:
                dec_list.append(env_feat)
            pred_dest = self.decoder_dest(torch.cat(dec_list, dim=1))

            dest_feat = self.encoder_dest(pred_dest)
            traj_list = [past_feat, dest_feat]
            if env_feat is not None:
                traj_list.append(env_feat)
            pred_mid_flat = self.predictor(torch.cat(traj_list, dim=1))
            pred_mid = pred_mid_flat.view(B, self.future_len - 1, 2)
            pred_pos = torch.cat([pred_mid, pred_dest.unsqueeze(1)], dim=1)
            all_samples.append(pred_pos.unsqueeze(1))

        return torch.cat(all_samples, dim=1)
