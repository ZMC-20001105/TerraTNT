#!/usr/bin/env python3
"""
增量模型实验：从LSTM_Env_Goal出发，逐步添加结构组件，观察每个组件的贡献。

V1: LSTM_Env_Goal (baseline) — 已训练
V2: + Attention over history (Seq2Seq attention mechanism)
V3: + Waypoint prediction (predict intermediate waypoints, condition decoder)
V4: + Spatial env sampling (sample local env features at predicted positions)
V5: + Segment conditioning (full waypoint-conditioned hierarchical decoding)
"""
import sys, os, json, math, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_terratnt_10s import FASDataset, HISTORY_LEN, FUTURE_LEN


# ============================================================
#  V2: LSTM_Env_Goal + Richer Context (parallel, no per-step loop)
# ============================================================

class LSTMEnvGoalV2(nn.Module):
    """LSTM_Env_Goal enhanced with richer context injection.
    Uses parallel LSTM decoding (no per-step loop) for speed.
    Adds: deeper env encoder, goal-conditioned context at every step."""
    def __init__(self, input_dim=2, hidden_dim=256, env_channels=18, env_dim=128, future_len=360):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.env_cnn = nn.Sequential(
            nn.Conv2d(env_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.env_fc = nn.Linear(128, env_dim)
        self.goal_fc = nn.Linear(2, 64)
        self.fusion = nn.Linear(hidden_dim + env_dim + 64, hidden_dim)
        # Time embedding for each future step
        self.time_embed = nn.Parameter(torch.randn(future_len, hidden_dim) * 0.02)
        # Parallel decoder: input = context + time_embed
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(self, history_xy, env_map=None, goal=None, **kwargs):
        B = history_xy.size(0)
        _, (h, c) = self.encoder(history_xy)
        if env_map is not None:
            env_feat = self.env_cnn(env_map).view(B, -1)
            env_feat = self.env_fc(env_feat)
        else:
            env_feat = torch.zeros(B, 128, device=history_xy.device)
        if goal is not None:
            goal_feat = F.relu(self.goal_fc(goal))
        else:
            goal_feat = torch.zeros(B, 64, device=history_xy.device)
        context = F.relu(self.fusion(torch.cat([h[-1], env_feat, goal_feat], dim=1)))
        # Parallel: repeat context + add time embeddings
        x = context.unsqueeze(1).expand(-1, self.future_len, -1) + self.time_embed.unsqueeze(0)
        h_new = h.clone()
        h_new[-1] = context
        out, _ = self.decoder(x, (h_new, c))
        predictions = self.output_fc(out)  # (B, T, 2) deltas
        return torch.cumsum(predictions, dim=1)  # cumulative positions


# ============================================================
#  V3: + Waypoint Prediction (parallel segment decoding)
# ============================================================

class LSTMEnvGoalWaypoint(nn.Module):
    """LSTM_Env_Goal + waypoint prediction with parallel segment decoding.
    Predicts waypoints first, then decodes each segment in parallel using LSTM."""
    def __init__(self, input_dim=2, hidden_dim=256, env_channels=18, env_dim=128,
                 future_len=360, num_waypoints=10):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        stride = future_len // (num_waypoints + 1)
        self.waypoint_indices = [stride * (i + 1) - 1 for i in range(num_waypoints)]
        if self.waypoint_indices[-1] != future_len - 1:
            self.waypoint_indices[-1] = future_len - 1

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.env_cnn = nn.Sequential(
            nn.Conv2d(env_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.env_fc = nn.Linear(128, env_dim)
        self.goal_fc = nn.Linear(2, 64)
        self.fusion = nn.Linear(hidden_dim + env_dim + 64, hidden_dim)

        # Waypoint predictor
        self.wp_query = nn.Parameter(torch.randn(num_waypoints, hidden_dim) * 0.02)
        self.wp_time_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.wp_out = nn.Linear(hidden_dim, 2)
        self.wp_residual_scale = nn.Parameter(torch.tensor(5.0))

        # Segment-conditioned parallel decoder
        self.seg_proj = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())
        self.time_embed = nn.Parameter(torch.randn(future_len, hidden_dim) * 0.02)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, 2)

    def forward(self, history_xy, env_map=None, goal=None, ground_truth=None,
                teacher_forcing_ratio=0.0, **kwargs):
        B = history_xy.size(0)
        device = history_xy.device
        _, (h, c) = self.encoder(history_xy)
        if env_map is not None:
            env_feat = self.env_cnn(env_map).view(B, -1)
            env_feat = self.env_fc(env_feat)
        else:
            env_feat = torch.zeros(B, 128, device=device)
        if goal is not None:
            goal_feat = F.relu(self.goal_fc(goal))
        else:
            goal_feat = torch.zeros(B, 64, device=device)
        base_feat = F.relu(self.fusion(torch.cat([h[-1], env_feat, goal_feat], dim=1)))

        # Predict waypoints
        goal_raw = goal if goal is not None else torch.zeros(B, 2, device=device)
        frac_list = [float(idx + 1) / float(self.future_len) for idx in self.waypoint_indices]
        frac = torch.tensor(frac_list, device=device, dtype=base_feat.dtype).view(1, -1, 1).expand(B, -1, -1)
        base_guess = goal_raw.unsqueeze(1) * frac
        q = self.wp_query.unsqueeze(0).expand(B, -1, -1)
        t_feat = self.wp_time_proj(frac.reshape(-1, 1)).view(B, -1, self.hidden_dim)
        wp_h = base_feat.unsqueeze(1) + q + t_feat
        resid = torch.tanh(self.wp_out(wp_h)) * self.wp_residual_scale
        pred_waypoints = base_guess + resid  # (B, num_wp, 2)

        # Build per-step segment conditioning (parallel, no loop)
        norm_scale = 50.0
        wp0 = torch.zeros(B, 1, 2, device=device, dtype=base_feat.dtype)
        wp_nodes = torch.cat([wp0, pred_waypoints], dim=1)  # (B, num_wp+1, 2)

        # Map each timestep to its segment start/end waypoint
        seg_cond = torch.zeros(B, self.future_len, 4, device=device, dtype=base_feat.dtype)
        prev_end = -1
        for si, end_t in enumerate(self.waypoint_indices):
            start_t = prev_end + 1
            seg_len = max(1, end_t - start_t + 1)
            for t in range(start_t, end_t + 1):
                if t < self.future_len:
                    seg_cond[:, t, :2] = wp_nodes[:, si, :] / norm_scale
                    seg_cond[:, t, 2:4] = wp_nodes[:, si + 1, :] / norm_scale
            prev_end = end_t

        seg_feat = self.seg_proj(seg_cond)  # (B, T, hidden_dim)

        # Parallel decode: context + segment + time
        x = base_feat.unsqueeze(1).expand(-1, self.future_len, -1) + seg_feat + self.time_embed.unsqueeze(0)
        h_new = h.clone()
        h_new[-1] = base_feat
        out, _ = self.decoder(x, (h_new, c))
        predictions = self.output_fc(out)  # (B, T, 2) deltas

        if not self.training:
            return torch.cumsum(predictions, dim=1)
        return predictions, pred_waypoints


# ============================================================
#  V4: + Spatial Environment Sampling (at waypoint positions)
# ============================================================

class LSTMEnvGoalWaypointSpatial(LSTMEnvGoalWaypoint):
    """V3 + spatial environment features sampled at waypoint positions.
    Samples env features at predicted waypoint locations and injects into segment conditioning."""
    def __init__(self, *args, env_coverage_km=140.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_coverage_km = env_coverage_km
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(18, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.spatial_in = nn.Sequential(nn.Linear(128, self.hidden_dim), nn.ReLU())
        self.env_local_scale = nn.Parameter(torch.tensor(1.0))
        # Override seg_proj to accept 4 + hidden_dim (segment + env)
        self.seg_proj = nn.Sequential(nn.Linear(4 + self.hidden_dim, self.hidden_dim), nn.ReLU())

    def forward(self, history_xy, env_map=None, goal=None, ground_truth=None,
                teacher_forcing_ratio=0.0, **kwargs):
        B = history_xy.size(0)
        device = history_xy.device
        _, (h, c) = self.encoder(history_xy)
        if env_map is not None:
            env_feat = self.env_cnn(env_map).view(B, -1)
            env_feat = self.env_fc(env_feat)
        else:
            env_feat = torch.zeros(B, 128, device=device)
        if goal is not None:
            goal_feat = F.relu(self.goal_fc(goal))
        else:
            goal_feat = torch.zeros(B, 64, device=device)
        base_feat = F.relu(self.fusion(torch.cat([h[-1], env_feat, goal_feat], dim=1)))

        # Extract spatial features
        env_spatial = None
        if env_map is not None:
            env_spatial = self.spatial_conv(env_map)  # (B, 128, H, W)

        # Predict waypoints
        goal_raw = goal if goal is not None else torch.zeros(B, 2, device=device)
        frac_list = [float(idx + 1) / float(self.future_len) for idx in self.waypoint_indices]
        frac = torch.tensor(frac_list, device=device, dtype=base_feat.dtype).view(1, -1, 1).expand(B, -1, -1)
        base_guess = goal_raw.unsqueeze(1) * frac
        q = self.wp_query.unsqueeze(0).expand(B, -1, -1)
        t_feat_wp = self.wp_time_proj(frac.reshape(-1, 1)).view(B, -1, self.hidden_dim)
        wp_h = base_feat.unsqueeze(1) + q + t_feat_wp
        resid = torch.tanh(self.wp_out(wp_h)) * self.wp_residual_scale
        pred_waypoints = base_guess + resid

        # Sample env at waypoint positions
        wp_env_feats = torch.zeros(B, self.num_waypoints + 1, self.hidden_dim, device=device)
        if env_spatial is not None:
            wp0 = torch.zeros(B, 1, 2, device=device, dtype=base_feat.dtype)
            all_wp = torch.cat([wp0, pred_waypoints], dim=1)  # (B, num_wp+1, 2)
            half = max(1e-6, self.env_coverage_km * 0.5)
            for wi in range(all_wp.size(1)):
                pos = all_wp[:, wi, :]  # (B, 2)
                gx = (pos[:, 0] / half).clamp(-1.0, 1.0)
                gy = (-pos[:, 1] / half).clamp(-1.0, 1.0)
                grid = torch.stack([gx, gy], dim=1).view(-1, 1, 1, 2)
                samp = F.grid_sample(env_spatial, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                wp_env_feats[:, wi, :] = self.spatial_in(samp.squeeze(-1).squeeze(-1))

        # Build per-step segment conditioning with env features
        norm_scale = 50.0
        wp0 = torch.zeros(B, 1, 2, device=device, dtype=base_feat.dtype)
        wp_nodes = torch.cat([wp0, pred_waypoints], dim=1)

        seg_cond = torch.zeros(B, self.future_len, 4 + self.hidden_dim, device=device, dtype=base_feat.dtype)
        prev_end = -1
        for si, end_t in enumerate(self.waypoint_indices):
            start_t = prev_end + 1
            env_seg = (wp_env_feats[:, si, :] + wp_env_feats[:, si + 1, :]) * 0.5 * self.env_local_scale
            for t in range(start_t, min(end_t + 1, self.future_len)):
                seg_cond[:, t, :2] = wp_nodes[:, si, :] / norm_scale
                seg_cond[:, t, 2:4] = wp_nodes[:, si + 1, :] / norm_scale
                seg_cond[:, t, 4:] = env_seg
            prev_end = end_t

        seg_feat = self.seg_proj(seg_cond)
        x = base_feat.unsqueeze(1).expand(-1, self.future_len, -1) + seg_feat + self.time_embed.unsqueeze(0)
        h_new = h.clone()
        h_new[-1] = base_feat
        out, _ = self.decoder(x, (h_new, c))
        predictions = self.output_fc(out)

        if not self.training:
            return torch.cumsum(predictions, dim=1)
        return predictions, pred_waypoints


# ============================================================
#  V5: TerraTNT Encoder + Goal Classifier + V4 Decoder (Fusion)
# ============================================================

class TerraTNTFusionV5(nn.Module):
    """Fusion model: TerraTNT's CNN env encoder + 26-dim history encoder + goal classifier,
    combined with V4-style waypoint prediction + spatial env sampling decoder.
    
    Goal: combine TerraTNT's best FDE (639m, goal accuracy) with V4's best ADE (1482m, trajectory quality).
    """
    def __init__(self, history_dim=26, hidden_dim=128, env_channels=18,
                 env_feature_dim=128, decoder_hidden_dim=256,
                 future_len=360, num_waypoints=10, num_candidates=6,
                 env_coverage_km=140.0, goal_norm_denom=70.0):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = decoder_hidden_dim
        self.env_feature_dim = env_feature_dim
        self.num_waypoints = num_waypoints
        self.env_coverage_km = env_coverage_km
        stride = future_len // (num_waypoints + 1)
        self.waypoint_indices = [stride * (i + 1) - 1 for i in range(num_waypoints)]
        if self.waypoint_indices[-1] != future_len - 1:
            self.waypoint_indices[-1] = future_len - 1

        # --- TerraTNT components ---
        from models.terratnt import PaperCNNEnvironmentEncoder, PaperLSTMHistoryEncoder, PaperGoalClassifier
        self.env_encoder = PaperCNNEnvironmentEncoder(
            input_channels=env_channels, feature_dim=env_feature_dim)
        self.history_encoder = PaperLSTMHistoryEncoder(
            input_dim=history_dim, hidden_dim=hidden_dim, num_layers=2)
        self.goal_classifier = PaperGoalClassifier(
            env_feature_dim=env_feature_dim, history_feature_dim=hidden_dim,
            num_goals=num_candidates, goal_norm_denom=goal_norm_denom)

        # --- V4-style decoder components ---
        # Fusion: history + env_global + goal → base_feat
        self.goal_fc = nn.Linear(2, 64)
        self.fusion = nn.Linear(hidden_dim + env_feature_dim + 64, decoder_hidden_dim)

        # Waypoint predictor
        self.wp_query = nn.Parameter(torch.randn(num_waypoints, decoder_hidden_dim) * 0.02)
        self.wp_time_proj = nn.Sequential(nn.Linear(1, decoder_hidden_dim), nn.ReLU())
        self.wp_out = nn.Linear(decoder_hidden_dim, 2)
        self.wp_residual_scale = nn.Parameter(torch.tensor(5.0))

        # Spatial env sampling at waypoints (using TerraTNT's spatial features)
        self.spatial_in = nn.Sequential(nn.Linear(env_feature_dim, decoder_hidden_dim), nn.ReLU())
        self.env_local_scale = nn.Parameter(torch.tensor(1.0))

        # Segment-conditioned parallel decoder
        self.seg_proj = nn.Sequential(
            nn.Linear(4 + decoder_hidden_dim, decoder_hidden_dim), nn.ReLU())
        self.time_embed = nn.Parameter(torch.randn(future_len, decoder_hidden_dim) * 0.02)
        self.decoder_lstm = nn.LSTM(decoder_hidden_dim, decoder_hidden_dim,
                                     num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(decoder_hidden_dim, 2)

    def forward(self, env_map, history, candidates, current_pos,
                teacher_forcing_ratio=0.0, ground_truth=None,
                target_goal_idx=None, use_gt_goal=False, goal=None,
                **kwargs):
        """
        TerraTNT-compatible forward interface.
        Args:
            env_map: (B, 18, 128, 128)
            history: (B, 90, 26) full history features
            candidates: (B, K, 2) candidate goals
            current_pos: (B, 2)
            target_goal_idx: (B,) GT goal index for training
            use_gt_goal: if True, use GT goal for decoding
            goal: (B, 2) optional direct goal override
        Returns:
            predictions: (B, T, 2) delta or cumulative positions
            goal_logits: (B, K) goal classification logits
        """
        B = env_map.size(0)
        device = env_map.device

        # Encode environment (TerraTNT CNN)
        env_global, env_tokens, env_spatial = self.env_encoder(env_map)

        # Encode history (TerraTNT LSTM, 26-dim input)
        _, (h_n, c_n) = self.history_encoder(history)
        history_features = h_n[-1]  # (B, hidden_dim=128)

        # Goal classification
        goal_logits = self.goal_classifier(env_global, history_features, candidates)

        # Select goal for decoding
        if goal is not None:
            selected_goal = goal
        elif use_gt_goal and target_goal_idx is not None:
            selected_goal = candidates[torch.arange(B, device=device), target_goal_idx]
        else:
            _, top_idx = torch.max(goal_logits, dim=1)
            selected_goal = candidates[torch.arange(B, device=device), top_idx]

        # Fuse history + env + goal → base_feat
        goal_feat = F.relu(self.goal_fc(selected_goal))
        base_feat = F.relu(self.fusion(
            torch.cat([history_features, env_global, goal_feat], dim=1)))  # (B, decoder_hidden_dim)

        # Predict waypoints
        frac_list = [float(idx + 1) / float(self.future_len) for idx in self.waypoint_indices]
        frac = torch.tensor(frac_list, device=device, dtype=base_feat.dtype).view(1, -1, 1).expand(B, -1, -1)
        base_guess = selected_goal.unsqueeze(1) * frac
        q = self.wp_query.unsqueeze(0).expand(B, -1, -1)
        t_feat = self.wp_time_proj(frac.reshape(-1, 1)).view(B, -1, self.hidden_dim)
        wp_h = base_feat.unsqueeze(1) + q + t_feat
        resid = torch.tanh(self.wp_out(wp_h)) * self.wp_residual_scale
        pred_waypoints = base_guess + resid  # (B, num_wp, 2)

        # Sample spatial env features at waypoint positions
        half = max(1e-6, self.env_coverage_km * 0.5)
        wp0 = torch.zeros(B, 1, 2, device=device, dtype=base_feat.dtype)
        all_wp = torch.cat([wp0, pred_waypoints], dim=1)  # (B, num_wp+1, 2)
        wp_env_feats = torch.zeros(B, self.num_waypoints + 1, self.hidden_dim, device=device)
        if env_spatial is not None:
            for wi in range(all_wp.size(1)):
                pos = all_wp[:, wi, :]
                gx = (pos[:, 0] / half).clamp(-1.0, 1.0)
                gy = (-pos[:, 1] / half).clamp(-1.0, 1.0)
                grid = torch.stack([gx, gy], dim=1).view(-1, 1, 1, 2)
                samp = F.grid_sample(env_spatial, grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=True)
                wp_env_feats[:, wi, :] = self.spatial_in(samp.squeeze(-1).squeeze(-1))

        # Build per-step segment conditioning with env features
        norm_scale = 50.0
        wp_nodes = all_wp
        seg_cond = torch.zeros(B, self.future_len, 4 + self.hidden_dim,
                               device=device, dtype=base_feat.dtype)
        prev_end = -1
        for si, end_t in enumerate(self.waypoint_indices):
            start_t = prev_end + 1
            env_seg = (wp_env_feats[:, si, :] + wp_env_feats[:, si + 1, :]) * 0.5 * self.env_local_scale
            for t in range(start_t, min(end_t + 1, self.future_len)):
                seg_cond[:, t, :2] = wp_nodes[:, si, :] / norm_scale
                seg_cond[:, t, 2:4] = wp_nodes[:, si + 1, :] / norm_scale
                seg_cond[:, t, 4:] = env_seg
            prev_end = end_t

        seg_feat = self.seg_proj(seg_cond)
        x = base_feat.unsqueeze(1).expand(-1, self.future_len, -1) + seg_feat + self.time_embed.unsqueeze(0)

        # Initialize decoder LSTM with base_feat
        h0 = torch.zeros(2, B, self.hidden_dim, device=device)
        c0 = torch.zeros(2, B, self.hidden_dim, device=device)
        h0[-1] = base_feat
        out, _ = self.decoder_lstm(x, (h0, c0))
        predictions = self.output_fc(out)  # (B, T, 2) deltas

        if not self.training:
            return torch.cumsum(predictions, dim=1), goal_logits

        return predictions, goal_logits, pred_waypoints


# ============================================================
#  V6: TerraTNT Encoder + Goal Classifier + Autoregressive Decoder
#      with Waypoint Milestones + Spatial Env Sampling
# ============================================================

class TerraTNTAutoregV6(nn.Module):
    """V6: Combines TerraTNT encoder+classifier with LSTM_Env_Goal-style autoregressive
    decoding, enhanced with waypoint milestones and spatial env features.
    
    Key insight: LSTM_Env_Goal's autoregressive step-by-step decoding maintains trajectory
    coherence (best Late ADE=1420m), while V5's parallel decoder loses feedback (Late=1686m).
    V6 adds waypoint+spatial conditioning to the autoregressive loop.
    """
    def __init__(self, history_dim=26, hidden_dim=128, env_channels=18,
                 env_feature_dim=128, decoder_hidden_dim=256,
                 future_len=360, num_waypoints=10, num_candidates=6,
                 env_coverage_km=140.0, goal_norm_denom=70.0):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = decoder_hidden_dim
        self.env_feature_dim = env_feature_dim
        self.num_waypoints = num_waypoints
        self.env_coverage_km = env_coverage_km
        stride = future_len // (num_waypoints + 1)
        self.waypoint_indices = [stride * (i + 1) - 1 for i in range(num_waypoints)]
        if self.waypoint_indices[-1] != future_len - 1:
            self.waypoint_indices[-1] = future_len - 1

        # --- TerraTNT components ---
        from models.terratnt import PaperCNNEnvironmentEncoder, PaperLSTMHistoryEncoder, PaperGoalClassifier
        self.env_encoder = PaperCNNEnvironmentEncoder(
            input_channels=env_channels, feature_dim=env_feature_dim)
        self.history_encoder = PaperLSTMHistoryEncoder(
            input_dim=history_dim, hidden_dim=hidden_dim, num_layers=2)
        self.goal_classifier = PaperGoalClassifier(
            env_feature_dim=env_feature_dim, history_feature_dim=hidden_dim,
            num_goals=num_candidates, goal_norm_denom=goal_norm_denom)

        # --- Fusion ---
        self.goal_fc = nn.Linear(2, 64)
        self.fusion = nn.Linear(hidden_dim + env_feature_dim + 64, decoder_hidden_dim)

        # --- Waypoint predictor (same as V5) ---
        self.wp_query = nn.Parameter(torch.randn(num_waypoints, decoder_hidden_dim) * 0.02)
        self.wp_time_proj = nn.Sequential(nn.Linear(1, decoder_hidden_dim), nn.ReLU())
        self.wp_out = nn.Linear(decoder_hidden_dim, 2)
        self.wp_residual_scale = nn.Parameter(torch.tensor(5.0))

        # --- Spatial env sampling ---
        self.spatial_in = nn.Sequential(nn.Linear(env_feature_dim, decoder_hidden_dim), nn.ReLU())
        self.env_local_scale = nn.Parameter(torch.tensor(1.0))

        # --- Autoregressive decoder (LSTM_Env_Goal style) ---
        # Input: prev_pos(2) + context(decoder_hidden_dim) + segment_cond(decoder_hidden_dim)
        self.decoder_lstm = nn.LSTM(
            2 + decoder_hidden_dim + decoder_hidden_dim,
            decoder_hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(decoder_hidden_dim, 2)

        # Segment conditioning projection
        self.seg_proj = nn.Sequential(
            nn.Linear(4 + decoder_hidden_dim, decoder_hidden_dim), nn.ReLU())

    def _get_segment_index(self, t):
        """Return which segment timestep t belongs to."""
        for si, end_t in enumerate(self.waypoint_indices):
            if t <= end_t:
                return si
        return len(self.waypoint_indices) - 1

    def forward(self, env_map, history, candidates, current_pos,
                teacher_forcing_ratio=0.0, ground_truth=None,
                target_goal_idx=None, use_gt_goal=False, goal=None,
                **kwargs):
        B = env_map.size(0)
        device = env_map.device

        # Encode environment
        env_global, env_tokens, env_spatial = self.env_encoder(env_map)

        # Encode history
        _, (h_n, c_n) = self.history_encoder(history)
        history_features = h_n[-1]

        # Goal classification
        goal_logits = self.goal_classifier(env_global, history_features, candidates)

        # Select goal
        if goal is not None:
            selected_goal = goal
        elif use_gt_goal and target_goal_idx is not None:
            selected_goal = candidates[torch.arange(B, device=device), target_goal_idx]
        else:
            _, top_idx = torch.max(goal_logits, dim=1)
            selected_goal = candidates[torch.arange(B, device=device), top_idx]

        # Fuse
        goal_feat = F.relu(self.goal_fc(selected_goal))
        context = F.relu(self.fusion(
            torch.cat([history_features, env_global, goal_feat], dim=1)))

        # Predict waypoints
        frac_list = [float(idx + 1) / float(self.future_len) for idx in self.waypoint_indices]
        frac = torch.tensor(frac_list, device=device, dtype=context.dtype).view(1, -1, 1).expand(B, -1, -1)
        base_guess = selected_goal.unsqueeze(1) * frac
        q = self.wp_query.unsqueeze(0).expand(B, -1, -1)
        t_feat = self.wp_time_proj(frac.reshape(-1, 1)).view(B, -1, self.hidden_dim)
        wp_h = context.unsqueeze(1) + q + t_feat
        resid = torch.tanh(self.wp_out(wp_h)) * self.wp_residual_scale
        pred_waypoints = base_guess + resid  # (B, num_wp, 2)

        # Sample spatial env at waypoints
        half = max(1e-6, self.env_coverage_km * 0.5)
        wp0 = torch.zeros(B, 1, 2, device=device, dtype=context.dtype)
        wp_nodes = torch.cat([wp0, pred_waypoints], dim=1)  # (B, num_wp+1, 2)
        wp_env_feats = torch.zeros(B, self.num_waypoints + 1, self.hidden_dim, device=device)
        if env_spatial is not None:
            for wi in range(wp_nodes.size(1)):
                pos = wp_nodes[:, wi, :]
                gx = (pos[:, 0] / half).clamp(-1.0, 1.0)
                gy = (-pos[:, 1] / half).clamp(-1.0, 1.0)
                grid = torch.stack([gx, gy], dim=1).view(-1, 1, 1, 2)
                samp = F.grid_sample(env_spatial, grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=True)
                wp_env_feats[:, wi, :] = self.spatial_in(samp.squeeze(-1).squeeze(-1))

        # Pre-compute per-segment conditioning vectors
        norm_scale = 50.0
        seg_conds = []  # (num_wp,) each (B, hidden_dim)
        for si in range(self.num_waypoints):
            seg_input = torch.cat([
                wp_nodes[:, si, :] / norm_scale,
                wp_nodes[:, si + 1, :] / norm_scale,
                (wp_env_feats[:, si, :] + wp_env_feats[:, si + 1, :]) * 0.5 * self.env_local_scale,
            ], dim=1)  # (B, 4 + hidden_dim)
            seg_conds.append(self.seg_proj(seg_input))  # (B, hidden_dim)

        # Autoregressive decoding
        h_dec = torch.zeros(2, B, self.hidden_dim, device=device)
        c_dec = torch.zeros(2, B, self.hidden_dim, device=device)
        h_dec[-1] = context

        curr_pos = torch.zeros(B, 1, 2, device=device)  # cumulative position
        preds = []

        for t in range(self.future_len):
            si = self._get_segment_index(t)
            seg_cond = seg_conds[si].unsqueeze(1)  # (B, 1, hidden_dim)

            dec_in = torch.cat([
                curr_pos,
                context.unsqueeze(1),
                seg_cond,
            ], dim=-1)  # (B, 1, 2 + hidden_dim + hidden_dim)

            out, (h_dec, c_dec) = self.decoder_lstm(dec_in, (h_dec, c_dec))
            delta = self.output_fc(out)  # (B, 1, 2)
            curr_pos = curr_pos + delta
            preds.append(curr_pos)

        predictions = torch.cat(preds, dim=1)  # (B, T, 2) cumulative positions

        if not self.training:
            return predictions, goal_logits

        # For training, return deltas for loss computation
        deltas = torch.cat([preds[0]] + [preds[i] - preds[i-1] for i in range(1, len(preds))], dim=1)
        return deltas, goal_logits, pred_waypoints


# ============================================================
#  V7: Confidence-Gated Adaptive Model
#      When classifier is confident → use goal (like V6)
#      When classifier is uncertain → ignore goal (like Seq2Seq)
# ============================================================

class ConfidenceGatedV7(nn.Module):
    """V7: V6 + learned confidence gate.
    
    Core mechanism: After goal classification, compute a scalar gate α ∈ [0,1]
    from classifier logits + history + env features. α controls how much the
    goal information influences the decoder:
      context = α * goal_context + (1-α) * no_goal_context
    
    When trained on mixed Phase data (precise + noisy + no prior), the model
    learns to trust goal info only when the classifier is confident.
    """
    def __init__(self, history_dim=26, hidden_dim=128, env_channels=18,
                 env_feature_dim=128, decoder_hidden_dim=256,
                 future_len=360, num_waypoints=10, num_candidates=6,
                 env_coverage_km=140.0, goal_norm_denom=70.0):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = decoder_hidden_dim
        self.env_feature_dim = env_feature_dim
        self.num_waypoints = num_waypoints
        self.env_coverage_km = env_coverage_km
        stride = future_len // (num_waypoints + 1)
        self.waypoint_indices = [stride * (i + 1) - 1 for i in range(num_waypoints)]
        if self.waypoint_indices[-1] != future_len - 1:
            self.waypoint_indices[-1] = future_len - 1

        # --- TerraTNT components (same as V6) ---
        from models.terratnt import PaperCNNEnvironmentEncoder, PaperLSTMHistoryEncoder, PaperGoalClassifier
        self.env_encoder = PaperCNNEnvironmentEncoder(
            input_channels=env_channels, feature_dim=env_feature_dim)
        self.history_encoder = PaperLSTMHistoryEncoder(
            input_dim=history_dim, hidden_dim=hidden_dim, num_layers=2)
        self.goal_classifier = PaperGoalClassifier(
            env_feature_dim=env_feature_dim, history_feature_dim=hidden_dim,
            num_goals=num_candidates, goal_norm_denom=goal_norm_denom)

        # --- Goal fusion (same as V6) ---
        self.goal_fc = nn.Linear(2, 64)
        self.fusion_with_goal = nn.Linear(hidden_dim + env_feature_dim + 64, decoder_hidden_dim)
        # --- No-goal fusion (history + env only) ---
        self.fusion_no_goal = nn.Linear(hidden_dim + env_feature_dim, decoder_hidden_dim)

        # --- Confidence gate ---
        # Input: classifier entropy + max_prob + history_features + env_features
        # Output: scalar gate α ∈ [0, 1]
        gate_hidden = 128
        self.confidence_gate_fc1 = nn.Linear(2 + hidden_dim + env_feature_dim, gate_hidden)
        self.confidence_gate_fc2 = nn.Linear(gate_hidden, 1)
        # Initialize bias to 0 so sigmoid outputs ~0.5 at start
        nn.init.zeros_(self.confidence_gate_fc2.bias)
        nn.init.zeros_(self.confidence_gate_fc2.weight)

        # --- Waypoint predictor (same as V6) ---
        self.wp_query = nn.Parameter(torch.randn(num_waypoints, decoder_hidden_dim) * 0.02)
        self.wp_time_proj = nn.Sequential(nn.Linear(1, decoder_hidden_dim), nn.ReLU())
        self.wp_out = nn.Linear(decoder_hidden_dim, 2)
        self.wp_residual_scale = nn.Parameter(torch.tensor(5.0))

        # --- Spatial env sampling (same as V6) ---
        self.spatial_in = nn.Sequential(nn.Linear(env_feature_dim, decoder_hidden_dim), nn.ReLU())
        self.env_local_scale = nn.Parameter(torch.tensor(1.0))

        # --- Autoregressive decoder (same as V6) ---
        self.decoder_lstm = nn.LSTM(
            2 + decoder_hidden_dim + decoder_hidden_dim,
            decoder_hidden_dim, num_layers=2, batch_first=True)
        self.output_fc = nn.Linear(decoder_hidden_dim, 2)

        # Segment conditioning projection
        self.seg_proj = nn.Sequential(
            nn.Linear(4 + decoder_hidden_dim, decoder_hidden_dim), nn.ReLU())

    def _get_segment_index(self, t):
        for si, end_t in enumerate(self.waypoint_indices):
            if t <= end_t:
                return si
        return len(self.waypoint_indices) - 1

    def forward(self, env_map, history, candidates, current_pos,
                teacher_forcing_ratio=0.0, ground_truth=None,
                target_goal_idx=None, use_gt_goal=False, goal=None,
                force_gate=None, **kwargs):
        """
        Args:
            force_gate: If not None, override the learned gate with this value (0.0 or 1.0).
                        Useful for ablation: force_gate=1.0 → always use goal (like V6),
                        force_gate=0.0 → never use goal (like Seq2Seq).
        """
        B = env_map.size(0)
        device = env_map.device

        # Encode environment
        env_global, env_tokens, env_spatial = self.env_encoder(env_map)

        # Encode history
        _, (h_n, c_n) = self.history_encoder(history)
        history_features = h_n[-1]

        # Goal classification
        goal_logits = self.goal_classifier(env_global, history_features, candidates)

        # Select goal
        if goal is not None:
            selected_goal = goal
        elif use_gt_goal and target_goal_idx is not None:
            selected_goal = candidates[torch.arange(B, device=device), target_goal_idx]
        else:
            _, top_idx = torch.max(goal_logits, dim=1)
            selected_goal = candidates[torch.arange(B, device=device), top_idx]

        # --- Confidence gate ---
        goal_probs = F.softmax(goal_logits, dim=1)
        max_prob = goal_probs.max(dim=1, keepdim=True).values          # (B, 1)
        entropy = -(goal_probs * (goal_probs + 1e-8).log()).sum(dim=1, keepdim=True)  # (B, 1)
        gate_input = torch.cat([max_prob, entropy, history_features, env_global], dim=1)

        # Always compute raw gate prediction for supervision
        gate_h = F.relu(self.confidence_gate_fc1(gate_input))
        alpha_raw = torch.sigmoid(self.confidence_gate_fc2(gate_h))  # (B, 1)

        if force_gate is not None:
            alpha = torch.full((B, 1), float(force_gate), device=device)
        else:
            alpha = alpha_raw

        # --- Two context paths ---
        goal_feat = F.relu(self.goal_fc(selected_goal))
        context_with_goal = F.relu(self.fusion_with_goal(
            torch.cat([history_features, env_global, goal_feat], dim=1)))
        context_no_goal = F.relu(self.fusion_no_goal(
            torch.cat([history_features, env_global], dim=1)))

        # Blend
        context = alpha * context_with_goal + (1.0 - alpha) * context_no_goal  # (B, hidden)

        # --- Waypoint prediction ---
        # When alpha is low, waypoints should be less goal-directed
        # base_guess is goal-dependent, so scale it by alpha
        frac_list = [float(idx + 1) / float(self.future_len) for idx in self.waypoint_indices]
        frac = torch.tensor(frac_list, device=device, dtype=context.dtype).view(1, -1, 1).expand(B, -1, -1)
        base_guess = selected_goal.unsqueeze(1) * frac * alpha.unsqueeze(-1)  # scale by gate
        q = self.wp_query.unsqueeze(0).expand(B, -1, -1)
        t_feat = self.wp_time_proj(frac.reshape(-1, 1)).view(B, -1, self.hidden_dim)
        wp_h = context.unsqueeze(1) + q + t_feat
        resid = torch.tanh(self.wp_out(wp_h)) * self.wp_residual_scale
        pred_waypoints = base_guess + resid  # (B, num_wp, 2)

        # --- Spatial env sampling at waypoints (same as V6) ---
        half = max(1e-6, self.env_coverage_km * 0.5)
        wp0 = torch.zeros(B, 1, 2, device=device, dtype=context.dtype)
        wp_nodes = torch.cat([wp0, pred_waypoints], dim=1)
        wp_env_feats = torch.zeros(B, self.num_waypoints + 1, self.hidden_dim, device=device)
        if env_spatial is not None:
            for wi in range(wp_nodes.size(1)):
                pos = wp_nodes[:, wi, :]
                gx = (pos[:, 0] / half).clamp(-1.0, 1.0)
                gy = (-pos[:, 1] / half).clamp(-1.0, 1.0)
                grid = torch.stack([gx, gy], dim=1).view(-1, 1, 1, 2)
                samp = F.grid_sample(env_spatial, grid, mode='bilinear',
                                     padding_mode='zeros', align_corners=True)
                wp_env_feats[:, wi, :] = self.spatial_in(samp.squeeze(-1).squeeze(-1))

        # Pre-compute per-segment conditioning vectors
        norm_scale = 50.0
        seg_conds = []
        for si in range(self.num_waypoints):
            seg_input = torch.cat([
                wp_nodes[:, si, :] / norm_scale,
                wp_nodes[:, si + 1, :] / norm_scale,
                (wp_env_feats[:, si, :] + wp_env_feats[:, si + 1, :]) * 0.5 * self.env_local_scale,
            ], dim=1)
            seg_conds.append(self.seg_proj(seg_input))

        # Autoregressive decoding
        h_dec = torch.zeros(2, B, self.hidden_dim, device=device)
        c_dec = torch.zeros(2, B, self.hidden_dim, device=device)
        h_dec[-1] = context

        curr_pos = torch.zeros(B, 1, 2, device=device)
        preds = []

        for t in range(self.future_len):
            si = self._get_segment_index(t)
            seg_cond = seg_conds[si].unsqueeze(1)

            dec_in = torch.cat([
                curr_pos,
                context.unsqueeze(1),
                seg_cond,
            ], dim=-1)

            out, (h_dec, c_dec) = self.decoder_lstm(dec_in, (h_dec, c_dec))
            delta = self.output_fc(out)
            curr_pos = curr_pos + delta
            preds.append(curr_pos)

        predictions = torch.cat(preds, dim=1)  # (B, T, 2) cumulative positions

        if not self.training:
            return predictions, goal_logits, alpha_raw

        # For training, return deltas for loss computation
        deltas = torch.cat([preds[0]] + [preds[i] - preds[i-1] for i in range(1, len(preds))], dim=1)
        return deltas, goal_logits, pred_waypoints, alpha_raw


# ============================================================
#  Training
# ============================================================

def ade_fde_m(pred_pos, gt_pos):
    """Compute ADE/FDE in meters. Input in km."""
    dist = torch.norm(pred_pos - gt_pos, dim=-1) * 1000
    ade = dist.mean(dim=1)
    fde = dist[:, -1]
    return ade, fde


def train_model(model, model_name, train_loader, val_loader, device,
                num_epochs=30, lr=1e-3, patience=10, save_dir=None,
                use_waypoints=False, wp_weight=0.5, tf_start=0.3):
    """Train a model with optional waypoint loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler()

    best_val_ade = float('inf')
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        # Scheduled teacher forcing
        tf_ratio = max(0, tf_start * (1 - epoch / num_epochs))

        for batch in train_loader:
            history = batch['history'].to(device)
            future_delta = batch['future'].to(device)
            env_map = batch['env_map'].to(device)
            gt_pos = torch.cumsum(future_delta, dim=1)
            goal = gt_pos[:, -1, :]
            history_xy = history[:, :, :2]

            optimizer.zero_grad()
            with autocast(enabled=True):
                if use_waypoints:
                    out = model(history_xy, env_map, goal=goal,
                                ground_truth=future_delta, teacher_forcing_ratio=tf_ratio)
                    if isinstance(out, tuple):
                        pred_delta, pred_wp = out
                        pred_pos = torch.cumsum(pred_delta, dim=1)
                        loss_traj = F.mse_loss(pred_pos, gt_pos)
                        # Waypoint loss
                        wp_idx = torch.tensor(model.waypoint_indices, device=device, dtype=torch.long)
                        gt_wp = gt_pos.index_select(1, wp_idx)
                        loss_wp = F.mse_loss(pred_wp, gt_wp)
                        loss = loss_traj + wp_weight * loss_wp
                    else:
                        pred_pos = out
                        loss = F.mse_loss(pred_pos, gt_pos)
                else:
                    pred_pos = model(history_xy, env_map, goal=goal)
                    loss = F.mse_loss(pred_pos, gt_pos)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_ade_sum, val_fde_sum, val_n = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future_delta = batch['future'].to(device)
                env_map = batch['env_map'].to(device)
                gt_pos = torch.cumsum(future_delta, dim=1)
                goal = gt_pos[:, -1, :]
                history_xy = history[:, :, :2]

                with autocast(enabled=True):
                    pred_pos = model(history_xy, env_map, goal=goal)
                    if isinstance(pred_pos, tuple):
                        pred_pos = pred_pos[0]
                        if pred_pos.dim() == 3 and pred_pos.size(-1) == 2:
                            pred_pos = torch.cumsum(pred_pos, dim=1)

                ade, fde = ade_fde_m(pred_pos, gt_pos)
                val_ade_sum += ade.sum().item()
                val_fde_sum += fde.sum().item()
                val_n += history.size(0)

        val_ade = val_ade_sum / max(1, val_n)
        val_fde = val_fde_sum / max(1, val_n)
        scheduler.step(val_ade)

        improved = ''
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            no_improve = 0
            improved = ' *BEST*'
            if save_dir:
                save_path = Path(save_dir) / f'{model_name}_best.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_ade': val_ade,
                    'val_fde': val_fde,
                    'model_name': model_name,
                }, save_path)
        else:
            no_improve += 1

        print(f'  [{model_name}] Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f} '
              f'val_ADE={val_ade:.0f}m val_FDE={val_fde:.0f}m tf={tf_ratio:.2f}{improved}',
              flush=True)

        if no_improve >= patience:
            print(f'  Early stopping at epoch {epoch+1}', flush=True)
            break

    return best_val_ade


def train_v5(model, model_name, train_loader, val_loader, device,
             num_epochs=30, lr=1e-3, patience=10, save_dir=None,
             wp_weight=0.5, cls_weight=1.0, resume_ckpt=None,
             use_cosine_lr=False, goal_dropout_prob=0.0):
    """Train V5/V6 fusion model with TerraTNT-compatible interface.
    Uses full 26-dim history, candidates, goal classifier loss.
    goal_dropout_prob: probability of using predicted goal instead of GT during training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler()
    cls_criterion = nn.CrossEntropyLoss()

    best_val_ade = float('inf')
    no_improve = 0
    start_epoch = 0

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
        orig_ade = ckpt.get('val_ade', float('inf'))
        start_epoch = ckpt.get('epoch', 0) + 1
        # Reset best_val_ade so fine-tuning for a different phase can save checkpoints
        best_val_ade = float('inf')
        print(f'  Resumed from {resume_ckpt} (epoch {start_epoch}, orig_ADE={orig_ade:.0f}m, best_ade reset)', flush=True)
        # Reset optimizer LR to the new lr
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            history = batch['history'].to(device)        # (B, 90, 26)
            future_delta = batch['future'].to(device)    # (B, T, 2)
            env_map = batch['env_map'].to(device)        # (B, 18, 128, 128)
            candidates = batch['candidates'].to(device)  # (B, K, 2)
            target_idx = batch['target_goal_idx'].to(device)  # (B,)
            current_pos = torch.zeros(history.size(0), 2, device=device)

            gt_pos = torch.cumsum(future_delta, dim=1)

            # Goal dropout: schedule probability linearly from 0 to goal_dropout_prob
            rel_ep = epoch - start_epoch
            if goal_dropout_prob > 0 and num_epochs > 1:
                cur_gdp = goal_dropout_prob * min(1.0, rel_ep / max(1, num_epochs * 0.3))
                use_gt = (torch.rand(1).item() > cur_gdp)
            else:
                use_gt = True

            optimizer.zero_grad()
            with autocast(enabled=True):
                out = model(env_map, history, candidates, current_pos,
                            target_goal_idx=target_idx, use_gt_goal=use_gt)
                pred_delta, goal_logits, pred_wp = out

                pred_pos = torch.cumsum(pred_delta, dim=1)
                loss_traj = F.mse_loss(pred_pos, gt_pos)

                # Waypoint loss
                wp_idx = torch.tensor(model.waypoint_indices, device=device, dtype=torch.long)
                gt_wp = gt_pos.index_select(1, wp_idx)
                loss_wp = F.mse_loss(pred_wp, gt_wp)

                # Goal classifier loss
                loss_cls = cls_criterion(goal_logits, target_idx)

                loss = loss_traj + wp_weight * loss_wp + cls_weight * loss_cls

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # Validation (inference mode: use predicted goal, not GT)
        model.eval()
        val_ade_sum, val_fde_sum, val_n = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future_delta = batch['future'].to(device)
                env_map = batch['env_map'].to(device)
                candidates = batch['candidates'].to(device)
                target_idx = batch['target_goal_idx'].to(device)
                current_pos = torch.zeros(history.size(0), 2, device=device)

                gt_pos = torch.cumsum(future_delta, dim=1)

                with autocast(enabled=True):
                    pred_pos, goal_logits = model(
                        env_map, history, candidates, current_pos,
                        use_gt_goal=False)

                ade, fde = ade_fde_m(pred_pos, gt_pos)
                val_ade_sum += ade.sum().item()
                val_fde_sum += fde.sum().item()
                val_n += history.size(0)

        val_ade = val_ade_sum / max(1, val_n)
        val_fde = val_fde_sum / max(1, val_n)
        if use_cosine_lr:
            scheduler.step()
        else:
            scheduler.step(val_ade)

        improved = ''
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            no_improve = 0
            improved = ' *BEST*'
            if save_dir:
                save_path = Path(save_dir) / f'{model_name}_best.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_ade': val_ade,
                    'val_fde': val_fde,
                    'model_name': model_name,
                }, save_path)
        else:
            no_improve += 1

        cur_lr = optimizer.param_groups[0]['lr']
        print(f'  [{model_name}] Epoch {epoch+1}/{start_epoch+num_epochs}: loss={avg_loss:.6f} '
              f'val_ADE={val_ade:.0f}m val_FDE={val_fde:.0f}m lr={cur_lr:.2e}{improved}', flush=True)

        if no_improve >= patience:
            print(f'  Early stopping at epoch {epoch+1}', flush=True)
            break

    return best_val_ade


def train_v7(model, model_name, train_loader, val_loader, device,
             num_epochs=30, lr=1e-3, patience=10, save_dir=None,
             wp_weight=0.5, cls_weight=1.0, resume_ckpt=None,
             use_cosine_lr=False, gate_weight=1.0):
    """Train V7 confidence-gated model in 2 stages.
    
    Stage 1 (60% of epochs): Train goal-free path with force_gate=0.
      - Freeze all V6-inherited layers, only train fusion_no_goal.
      - This teaches the no-goal path to predict from history+env alone.
    
    Stage 2 (40% of epochs): Joint fine-tune with gate supervision.
      - Unfreeze all. Per-batch: 50% good candidates (gate_target=1),
        50% corrupted candidates (gate_target=0). Gate runs free.
      - Both trajectory loss and gate BCE loss.
    """
    scaler = GradScaler()
    cls_criterion = nn.CrossEntropyLoss()
    GATE_DIST_THRESHOLD_KM = 3.0

    # Load V6 weights
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        if 'fusion.weight' in sd and 'fusion_with_goal.weight' not in sd:
            sd['fusion_with_goal.weight'] = sd.pop('fusion.weight')
            sd['fusion_with_goal.bias'] = sd.pop('fusion.bias')
            print('  Mapped V6 fusion -> V7 fusion_with_goal', flush=True)
        missing, _ = model.load_state_dict(sd, strict=False)
        orig_ade = ckpt.get('val_ade', float('inf'))
        print(f'  Resumed from {resume_ckpt} (orig_ADE={orig_ade:.0f}m)', flush=True)
        if missing:
            print(f'  New V7 layers: {missing}', flush=True)

    stage1_epochs = int(num_epochs * 0.6)
    stage2_epochs = num_epochs - stage1_epochs
    best_val_ade = float('inf')
    no_improve = 0
    global_epoch = 0

    # ==================== STAGE 1: Train goal-free path ====================
    print(f'\n  === STAGE 1: Train goal-free path ({stage1_epochs} epochs) ===', flush=True)

    # Freeze everything except fusion_no_goal
    for name, param in model.named_parameters():
        if 'fusion_no_goal' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f'  Trainable params: {sum(p.numel() for p in trainable):,} '
          f'(total: {sum(p.numel() for p in model.parameters()):,})', flush=True)

    opt1 = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-5)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, patience=3, factor=0.5)

    for epoch in range(stage1_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            history = batch['history'].to(device)
            future_delta = batch['future'].to(device)
            env_map = batch['env_map'].to(device)
            candidates = batch['candidates'].to(device)
            target_idx = batch['target_goal_idx'].to(device)
            current_pos = torch.zeros(history.size(0), 2, device=device)
            gt_pos = torch.cumsum(future_delta, dim=1)

            opt1.zero_grad()
            with autocast(enabled=True):
                out = model(env_map, history, candidates, current_pos,
                            target_goal_idx=target_idx, use_gt_goal=True,
                            force_gate=0.0)  # Force goal-free path
                pred_delta, goal_logits, pred_wp, alpha = out

                pred_pos = torch.cumsum(pred_delta, dim=1)
                loss = F.mse_loss(pred_pos, gt_pos)

            scaler.scale(loss).backward()
            scaler.unscale_(opt1)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(opt1)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        # Validate with gate=0 (goal-free)
        model.eval()
        val_ade_sum, val_n = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future_delta = batch['future'].to(device)
                env_map = batch['env_map'].to(device)
                candidates = batch['candidates'].to(device)
                current_pos = torch.zeros(history.size(0), 2, device=device)
                gt_pos = torch.cumsum(future_delta, dim=1)

                with autocast(enabled=True):
                    pred_pos, _, _ = model(env_map, history, candidates, current_pos,
                                           use_gt_goal=False, force_gate=0.0)
                ade, _ = ade_fde_m(pred_pos, gt_pos)
                val_ade_sum += ade.sum().item()
                val_n += history.size(0)

        val_ade = val_ade_sum / max(1, val_n)
        sched1.step(val_ade)
        cur_lr = opt1.param_groups[0]['lr']
        print(f'  [S1] Ep{epoch+1}/{stage1_epochs}: loss={avg_loss:.4f} '
              f'goal_free_ADE={val_ade:.0f}m lr={cur_lr:.1e}', flush=True)
        global_epoch += 1

    # Quick check: what's the goal-aware ADE (should still be ~V6 level)?
    model.eval()
    ga_ade_sum, ga_n = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            history = batch['history'].to(device)
            future_delta = batch['future'].to(device)
            env_map = batch['env_map'].to(device)
            candidates = batch['candidates'].to(device)
            current_pos = torch.zeros(history.size(0), 2, device=device)
            gt_pos = torch.cumsum(future_delta, dim=1)
            with autocast(enabled=True):
                pred_pos, _, _ = model(env_map, history, candidates, current_pos,
                                       use_gt_goal=False, force_gate=1.0)
            ade, _ = ade_fde_m(pred_pos, gt_pos)
            ga_ade_sum += ade.sum().item()
            ga_n += history.size(0)
    print(f'  [S1 done] goal_aware_ADE={ga_ade_sum/max(1,ga_n):.0f}m, '
          f'goal_free_ADE={val_ade:.0f}m', flush=True)

    # ==================== STAGE 2: Train gate network ====================
    print(f'\n  === STAGE 2: Train gate + fine-tune ({stage2_epochs} epochs) ===', flush=True)

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Differential LR: low for pretrained, high for gate
    gate_names = ['confidence_gate_fc1', 'confidence_gate_fc2', 'fusion_no_goal']
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if any(gn in name for gn in gate_names):
            gate_params.append(param)
        else:
            other_params.append(param)

    opt2 = torch.optim.Adam([
        {'params': other_params, 'lr': lr * 0.1},
        {'params': gate_params, 'lr': lr * 2.0},
    ], weight_decay=1e-5)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=3, factor=0.5)
    no_improve = 0

    for epoch in range(stage2_epochs):
        model.train()
        total_loss, total_gate_loss = 0, 0
        total_gate_mean, total_gate_target_mean = 0, 0
        n_batches = 0

        for batch in train_loader:
            history = batch['history'].to(device)
            future_delta = batch['future'].to(device)
            env_map = batch['env_map'].to(device)
            candidates = batch['candidates'].to(device)
            target_idx = batch['target_goal_idx'].to(device)
            goal = batch['goal'].to(device)
            current_pos = torch.zeros(history.size(0), 2, device=device)
            B = history.size(0)
            K = candidates.size(1)

            gt_pos = torch.cumsum(future_delta, dim=1)

            # 50% of batches: corrupt all candidates
            if torch.rand(1).item() < 0.5:
                angles = torch.rand(B, K, device=device) * 2 * 3.14159
                dists = 20.0 + torch.rand(B, K, device=device) * 40.0
                candidates = torch.stack([dists * torch.cos(angles),
                                          dists * torch.sin(angles)], dim=-1)
                target_idx = torch.zeros(B, dtype=torch.long, device=device)

            # Gate target from actual candidate quality
            min_dist = torch.norm(candidates - goal.unsqueeze(1), dim=-1).min(dim=1).values
            gate_target = (min_dist < GATE_DIST_THRESHOLD_KM).float().unsqueeze(1)

            opt2.zero_grad()
            with autocast(enabled=True):
                out = model(env_map, history, candidates, current_pos,
                            target_goal_idx=target_idx, use_gt_goal=False)
                pred_delta, goal_logits, pred_wp, alpha = out

                pred_pos = torch.cumsum(pred_delta, dim=1)
                loss_traj = F.mse_loss(pred_pos, gt_pos)

                wp_idx = torch.tensor(model.waypoint_indices, device=device, dtype=torch.long)
                gt_wp = gt_pos.index_select(1, wp_idx)
                loss_wp = F.mse_loss(pred_wp, gt_wp)

                loss_cls = cls_criterion(goal_logits, target_idx)
                loss_main = loss_traj + wp_weight * loss_wp + cls_weight * loss_cls

            # Gate BCE loss outside autocast
            alpha_safe = alpha.float().clamp(0.01, 0.99)
            loss_gate = F.binary_cross_entropy(alpha_safe, gate_target.float())
            loss = loss_main.float() + gate_weight * loss_gate

            scaler.scale(loss).backward()
            scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt2)
            scaler.update()

            total_loss += loss.item()
            total_gate_loss += loss_gate.item()
            total_gate_mean += alpha.mean().item()
            total_gate_target_mean += gate_target.mean().item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_gate = total_gate_mean / max(1, n_batches)
        avg_gt = total_gate_target_mean / max(1, n_batches)
        avg_gl = total_gate_loss / max(1, n_batches)

        # Validate with free gate (normal inference)
        model.eval()
        val_ade_sum, val_fde_sum, val_n = 0, 0, 0
        val_gate_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(device)
                future_delta = batch['future'].to(device)
                env_map = batch['env_map'].to(device)
                candidates = batch['candidates'].to(device)
                current_pos = torch.zeros(history.size(0), 2, device=device)
                gt_pos = torch.cumsum(future_delta, dim=1)

                with autocast(enabled=True):
                    pred_pos, _, alpha_val = model(
                        env_map, history, candidates, current_pos,
                        use_gt_goal=False)
                ade, fde = ade_fde_m(pred_pos, gt_pos)
                val_ade_sum += ade.sum().item()
                val_fde_sum += fde.sum().item()
                val_gate_sum += alpha_val.sum().item()
                val_n += history.size(0)

        val_ade = val_ade_sum / max(1, val_n)
        val_fde = val_fde_sum / max(1, val_n)
        val_gate = val_gate_sum / max(1, val_n)
        sched2.step(val_ade)

        improved = ''
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            no_improve = 0
            improved = ' *BEST*'
            if save_dir:
                save_path = Path(save_dir) / f'{model_name}_best.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': global_epoch + epoch,
                    'val_ade': val_ade,
                    'val_fde': val_fde,
                    'model_name': model_name,
                }, save_path)
        else:
            no_improve += 1

        cur_lr = opt2.param_groups[0]['lr']
        print(f'  [S2] Ep{epoch+1}/{stage2_epochs}: loss={avg_loss:.4f} '
              f'ADE={val_ade:.0f}m FDE={val_fde:.0f}m '
              f'gate={avg_gate:.3f}(tgt={avg_gt:.2f}) val_gate={val_gate:.3f} '
              f'g_loss={avg_gl:.3f} lr={cur_lr:.1e}{improved}', flush=True)

        if no_improve >= patience:
            print(f'  Early stopping at epoch {epoch+1}', flush=True)
            break

    return best_val_ade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', default='outputs/dataset_experiments/D1_optimal_combo')
    parser.add_argument('--split_file', default='outputs/dataset_experiments/D1_optimal_combo/fas_splits_trajlevel.json')
    parser.add_argument('--output_dir', default='runs/incremental_models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--models', default='V2,V3,V4',
                        help='Comma-separated list of models to train: V2,V3,V4,V5,V6')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (e.g. runs/incremental_models/V6_best.pth)')
    parser.add_argument('--cosine_lr', action='store_true',
                        help='Use cosine annealing LR schedule instead of ReduceLROnPlateau')
    parser.add_argument('--phase', type=str, default='fas1', choices=['fas1', 'fas2', 'fas3'],
                        help='Training phase (fas1=in-domain, fas3=missing GT candidate)')
    parser.add_argument('--phase3_mix', action='store_true',
                        help='Mix Phase1 and Phase3 data for robustness training')
    parser.add_argument('--goal_dropout', type=float, default=0.0,
                        help='Probability of using predicted goal instead of GT during training (0-1)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    print('Building datasets...', flush=True)
    with open(args.split_file) as f:
        splits = json.load(f)

    def _make_ds(phase_name='fas1', phase3_missing=False):
        return FASDataset(
            traj_dir=args.traj_dir, fas_split_file=args.split_file,
            phase=phase_name, history_len=HISTORY_LEN, future_len=FUTURE_LEN,
            num_candidates=6, region='bohemian_forest',
            sample_fraction=1.0, seed=42, env_coverage_km=140.0,
            coord_scale=1.0, goal_map_scale=1.0,
            phase3_missing_goal=phase3_missing,
        )

    phase = args.phase
    phase3_missing = (phase == 'fas3')
    train_ds = _make_ds(phase, phase3_missing)
    val_ds = _make_ds(phase, phase3_missing)

    # Explicitly set train/val samples_meta from split file
    train_items = splits[phase]['train_samples']
    train_ds.samples_meta = [(str(Path(args.traj_dir) / item['file']), int(item['sample_idx']))
                             for item in train_items]

    # Phase3 mixed training: create a separate fas3 dataset and concatenate
    if args.phase3_mix and 'fas3' in splits:
        # Recreate fas1 dataset with phase='fas3' + phase3_missing_goal=False
        # so it generates 6 candidates (with GT included) matching fas3 tensor shape
        train_ds_p1 = _make_ds('fas3', phase3_missing=False)
        train_ds_p1.samples_meta = [(str(Path(args.traj_dir) / item['file']), int(item['sample_idx']))
                                     for item in train_items]
        fas3_train = splits['fas3'].get('train_samples', splits['fas1'].get('train_samples', []))
        if fas3_train:
            fas3_ds = _make_ds('fas3', phase3_missing=True)
            fas3_ds.samples_meta = [(str(Path(args.traj_dir) / item['file']), int(item['sample_idx']))
                                     for item in fas3_train]
            print(f'Phase3 mixed training: {len(train_ds_p1)} fas1(6-cand) + {len(fas3_ds)} fas3 samples', flush=True)
            train_ds = torch.utils.data.ConcatDataset([train_ds_p1, fas3_ds])

    val_items = splits[phase]['val_samples']
    val_ds.samples_meta = [(str(Path(args.traj_dir) / item['file']), int(item['sample_idx']))
                           for item in val_items]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=1, pin_memory=True)

    print(f'Train: {len(train_ds)} samples, Val: {len(val_ds)} samples', flush=True)

    models_to_train = [m.strip() for m in args.models.split(',')]
    results = {}

    for model_name in models_to_train:
        print(f'\n{"="*60}', flush=True)
        print(f'Training {model_name}', flush=True)
        print(f'{"="*60}', flush=True)

        if model_name == 'V2':
            model = LSTMEnvGoalV2(hidden_dim=256, future_len=FUTURE_LEN).to(device)
            best_ade = train_model(model, model_name, train_loader, val_loader, device,
                                   num_epochs=args.num_epochs, lr=args.lr, patience=10,
                                   save_dir=str(output_dir), use_waypoints=False)
        elif model_name == 'V3':
            model = LSTMEnvGoalWaypoint(hidden_dim=256, future_len=FUTURE_LEN,
                                        num_waypoints=10).to(device)
            best_ade = train_model(model, model_name, train_loader, val_loader, device,
                                   num_epochs=args.num_epochs, lr=args.lr, patience=10,
                                   save_dir=str(output_dir), use_waypoints=True,
                                   wp_weight=0.5, tf_start=0.3)
        elif model_name == 'V4':
            model = LSTMEnvGoalWaypointSpatial(hidden_dim=256, future_len=FUTURE_LEN,
                                                num_waypoints=10, env_coverage_km=140.0).to(device)
            best_ade = train_model(model, model_name, train_loader, val_loader, device,
                                   num_epochs=args.num_epochs, lr=args.lr, patience=10,
                                   save_dir=str(output_dir), use_waypoints=True,
                                   wp_weight=0.5, tf_start=0.3)
        elif model_name == 'V5':
            model = TerraTNTFusionV5(
                history_dim=26, hidden_dim=128, env_channels=18,
                env_feature_dim=128, decoder_hidden_dim=256,
                future_len=FUTURE_LEN, num_waypoints=10,
                num_candidates=6, env_coverage_km=140.0,
            ).to(device)
            best_ade = train_v5(model, model_name, train_loader, val_loader, device,
                                num_epochs=args.num_epochs, lr=args.lr, patience=10,
                                save_dir=str(output_dir), wp_weight=0.5, cls_weight=1.0)
        elif model_name == 'V6':
            model = TerraTNTAutoregV6(
                history_dim=26, hidden_dim=128, env_channels=18,
                env_feature_dim=128, decoder_hidden_dim=256,
                future_len=FUTURE_LEN, num_waypoints=10,
                num_candidates=6, env_coverage_km=140.0,
            ).to(device)
            best_ade = train_v5(model, model_name, train_loader, val_loader, device,
                                num_epochs=args.num_epochs, lr=args.lr, patience=10,
                                save_dir=str(output_dir), wp_weight=0.5, cls_weight=1.0,
                                resume_ckpt=args.resume, use_cosine_lr=args.cosine_lr,
                                goal_dropout_prob=args.goal_dropout)
        elif model_name == 'V7':
            model = ConfidenceGatedV7(
                history_dim=26, hidden_dim=128, env_channels=18,
                env_feature_dim=128, decoder_hidden_dim=256,
                future_len=FUTURE_LEN, num_waypoints=10,
                num_candidates=6, env_coverage_km=140.0,
            ).to(device)
            best_ade = train_v7(model, model_name, train_loader, val_loader, device,
                                num_epochs=args.num_epochs, lr=args.lr, patience=10,
                                save_dir=str(output_dir), wp_weight=0.5, cls_weight=1.0,
                                resume_ckpt=args.resume, use_cosine_lr=args.cosine_lr)
        else:
            print(f'  Unknown model: {model_name}', flush=True)
            continue

        results[model_name] = {'best_val_ade_m': best_ade}
        n_params = sum(p.numel() for p in model.parameters())
        results[model_name]['n_params'] = n_params
        print(f'\n  {model_name}: best_val_ADE={best_ade:.0f}m, params={n_params:,}', flush=True)

    # Save summary
    with open(output_dir / 'incremental_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAll results saved to {output_dir}/incremental_results.json', flush=True)

    # Print comparison
    print(f'\n{"="*60}', flush=True)
    print('INCREMENTAL MODEL COMPARISON', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'{"Model":<25} {"ADE (m)":<12} {"Params":<15}', flush=True)
    print(f'{"-"*52}', flush=True)
    print(f'{"V1 (LSTM_Env_Goal)":<25} {"3092":<12} {"~1.5M":<15}', flush=True)
    for name, r in sorted(results.items()):
        print(f'{name:<25} {r["best_val_ade_m"]:<12.0f} {r["n_params"]:,}', flush=True)
    print(f'{"TerraTNT (reference)":<25} {"3108":<12} {"~2.5M":<15}', flush=True)


if __name__ == '__main__':
    main()
