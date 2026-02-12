"""
Y-net: Goals, Waypoints & Paths for Long Term Human Trajectory Forecasting (Faithful Implementation)

Reference: Mangalam et al., "From Goals, Waypoints & Paths To Long Term
Human Trajectory Forecasting", ICCV 2021.

Key architecture:
1. Scene encoder (U-Net backbone) processes semantic segmentation map
2. Goal module: predicts goal heatmap on scene, samples goals
3. Waypoint module: predicts waypoint heatmaps conditioned on goal
4. Path module: interpolates full trajectory through waypoints

Adapted for FASDataset:
- Input: history (B, 90, 26), env_map (B, 18, 128, 128)
- Output: predicted positions (B, 360, 2) in cumulative relative km
- Scene encoder processes 18-channel env map instead of RGB segmentation
- Goal/waypoint predictions in coordinate space (not heatmap) for efficiency
  at 360-step prediction horizon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    """U-Net style encoder for scene features."""
    def __init__(self, in_channels: int = 18, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)       # 128->128
        self.enc2 = ConvBlock(base_ch, base_ch * 2)        # 64->64
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)    # 32->32
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)    # 16->16
        self.pool = nn.MaxPool2d(2)

        # Decoder (for spatial features)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(base_ch * 8, 256)

    def forward(self, x):
        e1 = self.enc1(x)                    # (B, 32, 128, 128)
        e2 = self.enc2(self.pool(e1))         # (B, 64, 64, 64)
        e3 = self.enc3(self.pool(e2))         # (B, 128, 32, 32)
        e4 = self.enc4(self.pool(e3))         # (B, 256, 16, 16)

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))  # (B, 128, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32, 128, 128)

        # Global feature
        g = self.global_pool(e4).flatten(1)
        g = self.global_fc(g)

        return d1, g  # spatial_features, global_feature


class TrajectoryEncoder(nn.Module):
    """LSTM encoder for past trajectory."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class GoalModule(nn.Module):
    """
    Predicts goal (endpoint) as a 2D coordinate.
    Uses trajectory features + scene features.
    CVAE formulation for multimodal goals.
    """
    def __init__(self, traj_dim: int = 128, scene_dim: int = 256, zdim: int = 32, sigma: float = 1.3):
        super().__init__()
        self.zdim = zdim
        self.sigma = sigma

        # Goal encoder (for training with GT goal)
        self.goal_enc = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128),
        )

        # CVAE encoder
        self.latent_enc = nn.Sequential(
            nn.Linear(traj_dim + scene_dim + 128, 256), nn.ReLU(),
            nn.Linear(256, zdim * 2),
        )

        # Goal decoder
        self.goal_dec = nn.Sequential(
            nn.Linear(traj_dim + scene_dim + zdim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, traj_feat, scene_feat, goal_gt=None):
        """
        Returns: pred_goal (B, 2), mu (B, zdim), logvar (B, zdim)
        """
        B = traj_feat.size(0)
        if goal_gt is not None:
            goal_feat = self.goal_enc(goal_gt)
            latent = self.latent_enc(torch.cat([traj_feat, scene_feat, goal_feat], dim=1))
            mu, logvar = torch.chunk(latent, 2, dim=1)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        else:
            mu = torch.zeros(B, self.zdim, device=traj_feat.device)
            logvar = torch.zeros(B, self.zdim, device=traj_feat.device)
            z = torch.randn(B, self.zdim, device=traj_feat.device) * self.sigma

        pred_goal = self.goal_dec(torch.cat([traj_feat, scene_feat, z], dim=1))
        return pred_goal, mu, logvar


class WaypointModule(nn.Module):
    """
    Predicts intermediate waypoints conditioned on goal.
    Uses trajectory features + scene features + goal.
    """
    def __init__(self, traj_dim: int = 128, scene_dim: int = 256,
                 num_waypoints: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.num_waypoints = num_waypoints

        self.goal_enc = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128),
        )

        self.wp_predictor = nn.Sequential(
            nn.Linear(traj_dim + scene_dim + 128, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),
        )

    def forward(self, traj_feat, scene_feat, goal):
        """
        Returns: waypoints (B, num_waypoints, 2)
        """
        goal_feat = self.goal_enc(goal)
        wp_flat = self.wp_predictor(torch.cat([traj_feat, scene_feat, goal_feat], dim=1))
        return wp_flat.view(-1, self.num_waypoints, 2)


class PathModule(nn.Module):
    """
    Generates full trajectory by interpolating through waypoints.
    Uses parallel MLP refinement on top of linear interpolation (fast, no 360-step loop).
    """
    def __init__(self, hidden_dim: int = 256, future_len: int = 360,
                 traj_dim: int = 128, scene_dim: int = 256):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = hidden_dim

        # Context projection
        self.ctx_proj = nn.Sequential(
            nn.Linear(traj_dim + scene_dim, hidden_dim), nn.ReLU(),
        )

        # Refinement MLP: takes linear-interpolated position + context -> residual
        # Input: interp_pos(2) + progress(1) + seg_start(2) + seg_end(2) = 7
        self.refine = nn.Sequential(
            nn.Linear(7 + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, traj_feat, scene_feat, waypoints, goal):
        """
        waypoints: (B, num_wp, 2) - intermediate waypoints (cumulative positions)
        goal: (B, 2) - endpoint
        Returns: (B, future_len, 2) - full trajectory as cumulative positions
        """
        B = waypoints.size(0)
        device = waypoints.device

        # Build waypoint sequence: [origin, wp1, wp2, ..., goal]
        origin = torch.zeros(B, 1, 2, device=device)
        all_wp = torch.cat([origin, waypoints, goal.unsqueeze(1)], dim=1)  # (B, N, 2)
        N = all_wp.size(1)

        # Linear interpolation through waypoints -> baseline trajectory
        # Evenly spaced node indices
        node_times = torch.linspace(0, 1, N, device=device)  # (N,)
        query_times = torch.linspace(0, 1, self.future_len, device=device)  # (T,)

        # For each query time, find segment and interpolate
        # seg_idx: which segment each query belongs to
        seg_idx = torch.searchsorted(node_times, query_times, right=True) - 1
        seg_idx = seg_idx.clamp(0, N - 2)  # (T,)

        t_start = node_times[seg_idx]       # (T,)
        t_end = node_times[seg_idx + 1]     # (T,)
        alpha = ((query_times - t_start) / (t_end - t_start + 1e-8)).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        wp_start = all_wp[:, seg_idx, :]    # (B, T, 2)
        wp_end = all_wp[:, seg_idx + 1, :]  # (B, T, 2)
        interp = wp_start + alpha * (wp_end - wp_start)  # (B, T, 2)

        # Progress feature
        progress = query_times.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)  # (B, T, 1)

        # Context (broadcast over time)
        ctx = self.ctx_proj(torch.cat([traj_feat, scene_feat], dim=1))  # (B, hidden_dim)
        ctx_exp = ctx.unsqueeze(1).expand(-1, self.future_len, -1)  # (B, T, hidden_dim)

        # Refinement input
        refine_in = torch.cat([interp, progress, wp_start, wp_end, ctx_exp], dim=-1)  # (B, T, 7+H)
        residual = self.refine(refine_in)  # (B, T, 2)

        positions = interp + residual
        return positions


class YNetFaithful(nn.Module):
    """
    Faithful Y-net adapted for single-agent, environment-aware trajectory prediction.

    Architecture:
    1. UNet scene encoder -> spatial + global features
    2. LSTM trajectory encoder -> trajectory features
    3. Goal module (CVAE) -> predicted endpoint
    4. Waypoint module -> intermediate waypoints
    5. Path module (LSTM) -> full trajectory

    Training: provide goal for CVAE training.
    Inference: sample goals from prior, predict waypoints and paths.
    """
    def __init__(
        self,
        history_len: int = 90,
        future_len: int = 360,
        history_dim: int = 2,
        env_channels: int = 18,
        traj_hidden: int = 128,
        scene_base_ch: int = 32,
        zdim: int = 32,
        sigma: float = 1.3,
        num_waypoints: int = 4,
        path_hidden: int = 256,
    ):
        super().__init__()
        self.history_len = history_len
        self.future_len = future_len
        self.num_waypoints = num_waypoints

        # Scene encoder
        self.scene_encoder = UNetEncoder(env_channels, scene_base_ch)
        scene_dim = 256  # from global_fc

        # Trajectory encoder
        self.traj_encoder = TrajectoryEncoder(history_dim, traj_hidden)

        # Goal module
        self.goal_module = GoalModule(traj_hidden, scene_dim, zdim, sigma)

        # Waypoint module
        self.waypoint_module = WaypointModule(traj_hidden, scene_dim, num_waypoints)

        # Path module
        self.path_module = PathModule(path_hidden, future_len, traj_hidden, scene_dim)

    def forward(
        self,
        history: torch.Tensor,
        env_map: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            history: (B, 90, 26) or (B, 90, 2)
            env_map: (B, 18, 128, 128)
            goal: (B, 2) ground truth destination (cumulative position km)
        Returns:
            pred_pos: (B, future_len, 2) cumulative positions
            mu, logvar: CVAE parameters
        """
        # Extract xy
        if history.dim() == 3 and history.size(-1) > 2:
            history_xy = history[:, :, :2]
        else:
            history_xy = history

        # Encode scene
        _, scene_global = self.scene_encoder(env_map)

        # Encode trajectory
        traj_feat = self.traj_encoder(history_xy)

        # Predict goal
        pred_goal, mu, logvar = self.goal_module(traj_feat, scene_global, goal)

        # Predict waypoints
        waypoints = self.waypoint_module(traj_feat, scene_global, pred_goal)

        # Generate full path
        pred_pos = self.path_module(traj_feat, scene_global, waypoints, pred_goal)

        return pred_pos, mu, logvar

    def predict(self, history: torch.Tensor, env_map: torch.Tensor,
                num_samples: int = 20) -> torch.Tensor:
        """
        Multi-sample inference.
        Returns: (B, num_samples, future_len, 2)
        """
        if history.dim() == 3 and history.size(-1) > 2:
            history_xy = history[:, :, :2]
        else:
            history_xy = history

        _, scene_global = self.scene_encoder(env_map)
        traj_feat = self.traj_encoder(history_xy)

        all_samples = []
        for _ in range(num_samples):
            pred_goal, _, _ = self.goal_module(traj_feat, scene_global, goal_gt=None)
            waypoints = self.waypoint_module(traj_feat, scene_global, pred_goal)
            pred_pos = self.path_module(traj_feat, scene_global, waypoints, pred_goal)
            all_samples.append(pred_pos.unsqueeze(1))

        return torch.cat(all_samples, dim=1)
