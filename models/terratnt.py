"""
TerraTNT: 目标驱动的地面目标轨迹预测模型
包含CNN环境编码器、LSTM历史编码器、目标分类器和LSTM解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional, List
import logging
import math
from models.pos_encoding import PositionEmbeddingSine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperCNNEnvironmentEncoder(nn.Module):
    def __init__(self, input_channels: int = 18, feature_dim: int = 128, spatial_res: int = 32,
                 dropout: float = 0.0, **kwargs):
        super().__init__()
        self.spatial_res = spatial_res
        self.drop_rate = dropout

        def _block(in_ch: int, out_ch: int, stride: int = 2, dilation: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # 128x128 -> 64x64
        self.conv1 = _block(input_channels, 32, stride=2)
        # 64x64 -> 32x32
        self.conv2 = _block(32, 64, stride=2)
        
        # 保持 32x32 分辨率
        self.conv3 = _block(64, 128, stride=1, dilation=1)
        self.conv4 = _block(128, 128, stride=1, dilation=1)

        # Spatial dropout for cross-domain regularization
        self.spatial_dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # 空间特征投影
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.spatial_dropout(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.spatial_dropout(x)
        
        # 输出 32x32 的特征图
        env_spatial = self.spatial_proj(x)
        if env_spatial.shape[-1] != self.spatial_res:
            env_spatial = F.interpolate(env_spatial, size=(self.spatial_res, self.spatial_res), mode='bilinear', align_corners=True)
        
        pool = self.avgpool(x)
        flat = torch.flatten(pool, 1)
        flat = self.fc_dropout(flat)
        env_global = self.fc(flat)
        
        return env_global, None, env_spatial


class DualScaleEnvironmentEncoder(nn.Module):
    """双尺度环境编码器：同时处理全局宏观视野和局部高精度视野"""
    def __init__(self, input_channels: int = 18, feature_dim: int = 128, spatial_res: int = 32):
        super().__init__()
        
        # 全局编码器 (140km -> 捕捉地形大势)
        self.global_encoder = PaperCNNEnvironmentEncoder(input_channels, feature_dim // 2, spatial_res=spatial_res)
        
        # 局部编码器 (10km -> 捕捉道路细节)
        self.local_encoder = PaperCNNEnvironmentEncoder(input_channels, feature_dim // 2, spatial_res=spatial_res)
        
        # 融合层
        self.fusion = nn.Linear(feature_dim, feature_dim)
        
        # 空间特征压缩层 (用于局部采样，将拼接后的 256 通道压缩回 feature_dim)
        # 注意：PaperCNNEnvironmentEncoder 的 spatial_res 分支输出的是卷积后的通道数 (128)
        # 两个拼接后是 256
        self.spatial_compress = nn.Conv2d(256, feature_dim, kernel_size=1)
        
        # Auxiliary Segmentation Head (Aux Cost Map)
        # Predict road/passable mask from local features (128 -> 1)
        # Input: (B, 128, 32, 32) -> Output: (B, 1, 128, 128)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 128->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64->128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, env_map_global: torch.Tensor, env_map_local: torch.Tensor):
        # env_map_global: (B, 18, 128, 128)
        # env_map_local: (B, 18, 128, 128)
        
        g_feat, _, g_spatial = self.global_encoder(env_map_global)
        l_feat, _, l_spatial = self.local_encoder(env_map_local)
        
        # 全局特征拼接
        combined_global = torch.cat([g_feat, l_feat], dim=1)
        env_global = self.fusion(combined_global)
        
        # 空间特征拼接 (用于局部采样)
        combined_spatial = torch.cat([g_spatial, l_spatial], dim=1) # (B, 256, 32, 32)
        env_spatial = self.spatial_compress(combined_spatial) # (B, 128, 32, 32)
        
        # Aux Task: Predict local road map
        local_seg_logits = self.seg_head(l_spatial)
        
        return env_global, None, env_spatial, l_spatial, local_seg_logits

class PaperLSTMHistoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 26, hidden_dim: int = 128, num_layers: int = 2, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output, (h_n, c_n) = self.lstm(x)
        h_last = self.fc(h_n[-1])
        if h_n.size(0) == 1:
            h_n = h_last.unsqueeze(0)
        else:
            h_n = torch.cat([h_n[:-1], h_last.unsqueeze(0)], dim=0)
        return output, (h_n, c_n)


class PaperGoalClassifier(nn.Module):
    def __init__(
        self,
        env_feature_dim: int = 128,
        history_feature_dim: int = 128,
        goal_embed_dim: int = 64,
        hidden_dim: int = 256,
        num_goals: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.num_goals = num_goals
        self.goal_norm_denom = float(kwargs.get('goal_norm_denom', 70.0))

        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, goal_embed_dim),
        )

        input_dim = env_feature_dim + history_feature_dim + goal_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, env_features: torch.Tensor, history_features: torch.Tensor, candidate_goals: torch.Tensor) -> torch.Tensor:
        num_goals = candidate_goals.size(1)
        denom = float(self.goal_norm_denom) if float(self.goal_norm_denom) > 0 else 1.0
        goal_embeds = self.goal_encoder(candidate_goals / denom)

        H_expanded = history_features.unsqueeze(1).expand(-1, num_goals, -1)
        E_expanded = env_features.unsqueeze(1).expand(-1, num_goals, -1)
        combined = torch.cat([H_expanded, E_expanded, goal_embeds], dim=2)
        logits = self.classifier(combined).squeeze(-1)
        return logits


class PaperTrajectoryDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        env_feature_dim: int = 128,
        history_feature_dim: int = 128,
        goal_feature_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_length: int = 60,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length
        self.waypoint_indices: List[int] = []

        self.goal_norm_denom = float(kwargs.get('goal_norm_denom', 70.0))

        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

        input_size = env_feature_dim + history_feature_dim + 64
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        env_features: torch.Tensor,
        history_features: torch.Tensor,
        goal_features: torch.Tensor,
        current_pos: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
        waypoint_teacher_forcing_ratio: Optional[float] = None,
        ground_truth: Optional[torch.Tensor] = None,
        return_waypoints: bool = False,
    ):
        if isinstance(env_features, (tuple, list)):
            env_global = env_features[0]
        else:
            env_global = env_features

        denom = float(self.goal_norm_denom) if float(self.goal_norm_denom) > 0 else 1.0
        goal_embed = self.goal_encoder(goal_features / denom)
        base_input = torch.cat([history_features, env_global, goal_embed], dim=1)
        x = base_input.unsqueeze(1).expand(-1, int(self.output_length), -1)

        out, _ = self.lstm(x)
        predictions = self.output_layer(out)

        if return_waypoints:
            return predictions, None
        return predictions


class PaperHierarchicalTrajectoryDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        env_feature_dim: int = 128,
        history_feature_dim: int = 128,
        goal_feature_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_length: int = 60,
        waypoint_stride: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.output_length = int(output_length)

        self.waypoint_stride = int(waypoint_stride) if waypoint_stride is not None else 0
        self.waypoint_indices = self._get_waypoint_indices(self.output_length, self.waypoint_stride)
        self.num_waypoints = len(self.waypoint_indices)

        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

        self.base_proj = nn.Sequential(
            nn.Linear(env_feature_dim + history_feature_dim + 64, self.hidden_dim),
            nn.ReLU(),
        )

        self.wp_query = nn.Parameter(torch.randn(max(0, self.num_waypoints - 1), self.hidden_dim) * 0.02)
        self.wp_time_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
        )
        self.wp_residual_scale_v2 = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.waypoint_out = nn.Linear(self.hidden_dim, 2)
        
        # Cross Attention for Waypoint-Environment interaction
        self.wp_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=self.hidden_dim // 2, normalize=True)

        self.segment_proj = nn.Sequential(
            nn.Linear(5, self.hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.output_layer = nn.Linear(self.hidden_dim, 2)

        # 显式转向建模
        self.use_heading = bool(kwargs.get('use_heading', False))
        if self.use_heading:
            # 预测每步的heading change (sin, cos)
            self.heading_pred = nn.Linear(self.hidden_dim, 2)
            # 将当前heading注入解码器输入
            self.heading_proj = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, self.hidden_dim),
            )
            # heading影响力缩放
            self.heading_scale = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

        # 动态输入增强
        self.pos_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim),
        )
        # 目标向量增强：提供当前点到终点的方向和距离感
        self.goal_vec_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim),
        )
        self.goal_vec_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.goal_vec_use_waypoint = bool(kwargs.get('goal_vec_use_waypoint', False))
        self.spatial_in = nn.Sequential(
            nn.Linear(env_feature_dim, self.hidden_dim),
            nn.ReLU(),
        )
        # Architectural Fix: Initialize scales to 1.0 (was 0.5/0.3) to allow strong local environment signal
        self.env_local_scale = nn.Parameter(torch.tensor(1.0))
        self.env_local_scale2 = nn.Parameter(torch.tensor(1.0))
        self.env_coverage_km = float(kwargs.get('env_coverage_km', 140.0))
        self.env_local_coverage_km = float(kwargs.get('env_local_coverage_km', 10.0))
        self.goal_norm_denom = float(kwargs.get('goal_norm_denom', float(self.env_coverage_km * 0.5)))

    def _pos_to_grid(self, pos_xy: torch.Tensor, coverage_km: Optional[float] = None) -> torch.Tensor:
        cov = float(self.env_coverage_km if coverage_km is None else coverage_km)
        half = max(1e-6, cov * 0.5)
        gx = (pos_xy[:, 0] / half).clamp(-1.0, 1.0)
        gy = (-pos_xy[:, 1] / half).clamp(-1.0, 1.0)
        return torch.stack([gx, gy], dim=1)

    def _sample_env(self, env_spatial: torch.Tensor, pos_xy: torch.Tensor, coverage_km: Optional[float] = None) -> torch.Tensor:
        grid = self._pos_to_grid(pos_xy, coverage_km=coverage_km).view(-1, 1, 1, 2)
        samp = F.grid_sample(env_spatial, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return self.spatial_in(samp.squeeze(-1).squeeze(-1))

    def _get_waypoint_indices(self, output_length: int, waypoint_stride: int = 0) -> List[int]:
        if output_length <= 0:
            return []
        if output_length == 1:
            return [0]

        stride = int(waypoint_stride) if waypoint_stride is not None else 0
        if stride and stride > 0:
            idx = list(range(stride - 1, output_length, stride))
            if idx[-1] != (output_length - 1):
                idx.append(output_length - 1)
        else:
            idx = [
                max(0, output_length // 4 - 1),
                max(0, output_length // 2 - 1),
                max(0, (3 * output_length) // 4 - 1),
                output_length - 1,
            ]
        out: List[int] = []
        for i in idx:
            if len(out) == 0 or out[-1] != i:
                out.append(int(i))
        return out

    def forward(
        self,
        env_features: torch.Tensor,
        history_features: torch.Tensor,
        goal_features: torch.Tensor,
        current_pos: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
        waypoint_teacher_forcing_ratio: Optional[float] = None,
        ground_truth: Optional[torch.Tensor] = None,
        return_waypoints: bool = False,
    ):
        env_tokens = None
        env_spatial = None
        env_spatial_local = None
        if isinstance(env_features, (tuple, list)) and len(env_features) >= 3:
            env_global, env_tokens, env_spatial = env_features[:3]
            if len(env_features) >= 4:
                env_spatial_local = env_features[3]
        elif isinstance(env_features, (tuple, list)):
            env_global = env_features[0]
        else:
            env_global = env_features

        batch_size = env_global.size(0)
        denom = float(self.goal_norm_denom) if float(self.goal_norm_denom) > 0 else 1.0
        goal_features_raw = goal_features
        goal_features_norm = goal_features_raw / denom
        goal_embed = self.goal_encoder(goal_features_norm)
        base_in = torch.cat([history_features, env_global, goal_embed], dim=1)
        base_feat = self.base_proj(base_in)

        debug_log = getattr(self, 'debug_log', False)
        debug_info = None
        if debug_log and batch_size > 0:
            import json
            debug_info = {
                'goal_norm_denom': float(denom),
                'goal_features': [float(goal_features_raw[0, 0]), float(goal_features_raw[0, 1])],
                'goal_features_norm': [float(goal_features_norm[0, 0]), float(goal_features_norm[0, 1])],
                'history_norm': float(history_features[0].norm().item()),
                'env_global_norm': float(env_global[0].norm().item()),
                'goal_embed_norm': float(goal_embed[0].norm().item()),
                'base_feat_norm': float(base_feat[0].norm().item()),
                'env_local_scale': float(self.env_local_scale.detach().cpu().item()) if hasattr(self.env_local_scale, 'detach') else float(self.env_local_scale),
                'env_local_scale2': float(self.env_local_scale2.detach().cpu().item()) if hasattr(self.env_local_scale2, 'detach') else float(self.env_local_scale2),
                'goal_vec_scale': float(self.goal_vec_scale.detach().cpu().item()) if hasattr(self.goal_vec_scale, 'detach') else float(self.goal_vec_scale),
                'goal_vec_use_waypoint': bool(self.goal_vec_use_waypoint),
                'debug_stride': int(getattr(self, 'debug_stride', 60)),
                'steps': [],
            }

        # waypoint prediction (hierarchical)
        pred_waypoints = []
        if self.num_waypoints > 1:
            frac_list = [float(i + 1) / float(max(1, self.output_length)) for i in self.waypoint_indices[:-1]]
            frac = torch.as_tensor(frac_list, device=base_feat.device, dtype=base_feat.dtype).view(1, -1, 1)
            frac = frac.expand(batch_size, -1, -1)

            base_guess = goal_features_raw.unsqueeze(1) * frac
            q = self.wp_query.unsqueeze(0).expand(batch_size, -1, -1)
            t_feat = self.wp_time_proj(frac.reshape(-1, 1)).view(batch_size, -1, self.hidden_dim)
            h = base_feat.unsqueeze(1) + q + t_feat
            
            # Cross Attention: Waypoints (Query) attend to Environment Spatial Features (Key/Value)
            if env_spatial is not None:
                # env_spatial: (B, C, H, W)
                b, c, h_map, w_map = env_spatial.shape
                # Flatten spatial dims: (B, H*W, C)
                env_flat = env_spatial.flatten(2).transpose(1, 2)
                # Project to hidden_dim: (B, H*W, hidden_dim)
                env_emb = self.spatial_in(env_flat)
                
                # Generate and add Positional Encodings
                pos_encoding = self.pos_enc(env_spatial) # (B, hidden_dim, H, W)
                pos_flat = pos_encoding.flatten(2).transpose(1, 2) # (B, H*W, hidden_dim)
                
                # Attention (Query=Waypoints, Key=Env+Pos, Value=Env)
                attn_out, attn_w = self.wp_attn(query=h, key=env_emb + pos_flat, value=env_emb, need_weights=True)
                h = h + attn_out # Residual connection
                if debug_log and debug_info is not None and attn_w is not None:
                    # attn_w: (B, num_wp-1, H*W) when average_attn_weights=True
                    if 'attn_weights' not in debug_info:
                        debug_info['attn_shape'] = [int(x) for x in attn_w.shape]
                        debug_info['attn_hw'] = [int(h_map), int(w_map)]
                        debug_info['attn_weights'] = attn_w[0].detach().cpu().tolist()
            
            resid = torch.tanh(self.waypoint_out(h)) * self.wp_residual_scale_v2.to(base_guess.dtype)
            pred_waypoints = base_guess + resid
        else:
            pred_waypoints = torch.zeros(batch_size, 0, 2, device=base_feat.device, dtype=base_feat.dtype)

        # last waypoint is always the goal
        pred_waypoints = torch.cat([pred_waypoints, goal_features_raw.unsqueeze(1)], dim=1)

        waypoints_cond = pred_waypoints
        if self.training and ground_truth is not None and self.num_waypoints > 0:
            wp_tf_ratio = teacher_forcing_ratio if waypoint_teacher_forcing_ratio is None else float(waypoint_teacher_forcing_ratio)
            if torch.rand(1).item() < wp_tf_ratio:
                gt_pos = torch.cumsum(ground_truth, dim=1)
                wp_idx = torch.as_tensor(self.waypoint_indices, device=gt_pos.device, dtype=torch.long)
                if int(wp_idx.max().item()) < gt_pos.size(1):
                    gt_waypoints = gt_pos.index_select(1, wp_idx)
                    if gt_waypoints.size() == pred_waypoints.size():
                        waypoints_cond = gt_waypoints

        # segment-conditioned step decoding
        wp0 = torch.zeros(batch_size, 1, 2, device=base_feat.device, dtype=base_feat.dtype)
        wp_nodes = torch.cat([wp0, waypoints_cond], dim=1)

        # 计算段边界
        seg_bounds = []
        prev = -1
        for end_t in self.waypoint_indices:
            start_t = prev + 1
            seg_bounds.append((start_t, end_t))
            prev = end_t

        predictions = []
        heading_preds_list = []
        hidden = None
        pos_running = torch.zeros(batch_size, 2, device=base_feat.device, dtype=base_feat.dtype)
        prev_delta = torch.zeros(batch_size, 2, device=base_feat.device, dtype=base_feat.dtype)
        pos_heading = torch.zeros(batch_size, device=base_feat.device, dtype=base_feat.dtype)  # current heading angle

        for t in range(self.output_length):
            # 确定当前时间步所属的段
            seg_id = 0
            for si, (s, e) in enumerate(seg_bounds):
                if s <= t <= e:
                    seg_id = si
                    break
            
            s, e = seg_bounds[seg_id]
            seg_len = float(max(1, e - s))
            prog = float(t - s) / seg_len  # 段内进度 [0, 1]
            
            # 段起点和终点 waypoint
            start_wp = wp_nodes[:, seg_id, :] if seg_id < wp_nodes.size(1) else torch.zeros_like(goal_features_raw)
            end_wp = wp_nodes[:, seg_id + 1, :] if (seg_id + 1) < wp_nodes.size(1) else goal_features_raw
            
            # 段特征：[start_wp, end_wp, progress] (归一化)
            # 使用空间尺度 (goal_norm_denom=70km) 而非段长度进行归一化
            norm_scale = self.goal_norm_denom
            start_wp_norm = start_wp / norm_scale
            end_wp_norm = end_wp / norm_scale
            seg_in = torch.cat([
                start_wp_norm,
                end_wp_norm,
                torch.full((batch_size, 1), prog, device=base_feat.device, dtype=base_feat.dtype),
            ], dim=1)
            seg_feat = self.segment_proj(seg_in)
            
            # 目标向量：从当前位置到目标的向量（归一化）
            if getattr(self, 'goal_vec_use_waypoint', False):
                # 使用当前段的终点作为局部目标
                target_pos = end_wp
            else:
                # 使用最终目标
                target_pos = goal_features_raw
                
            goal_vector = target_pos - pos_running
            goal_vector_norm = goal_vector / norm_scale
            goal_vec_feat = self.goal_vec_proj(goal_vector_norm)
            
            # 位置特征：当前累积位置（归一化）
            pos_norm = pos_running / norm_scale
            pos_feat = self.pos_proj(pos_norm)
            
            # 前一步增量特征
            delta_feat = self.delta_proj(prev_delta)
            
            # 组合所有特征
            step_input = base_feat + seg_feat + pos_feat + self.goal_vec_scale * goal_vec_feat + delta_feat

            if debug_log:
                stride = int(getattr(self, 'debug_stride', 60))
                stride = max(1, stride)
                if (t % stride == 0) and debug_info is not None:
                    debug_info['steps'].append({
                        'step': int(t),
                        'pos_running': [float(pos_running[0, 0]), float(pos_running[0, 1])],
                        'prev_delta': [float(prev_delta[0, 0]), float(prev_delta[0, 1])],
                        'delta_feat_norm': float(delta_feat[0].norm().item()),
                        'step_input_norm_pre_env': float(step_input[0].norm().item()),
                    })

            if env_spatial is not None:
                env_local = self._sample_env(env_spatial, pos_running)
                step_input = step_input + self.env_local_scale * env_local

                if debug_log and debug_info is not None and len(debug_info['steps']) > 0:
                    last = debug_info['steps'][-1]
                    if int(last.get('step', -1)) == int(t):
                        last['env_local_norm'] = float(env_local[0].norm().item())
                        last['env_local_contribution'] = float((self.env_local_scale.to(env_local.dtype) * env_local[0]).norm().item())
                        last['step_input_norm_post_env'] = float(step_input[0].norm().item())

            if env_spatial_local is not None:
                env_local2 = self._sample_env(env_spatial_local, pos_running, coverage_km=self.env_local_coverage_km)
                step_input = step_input + self.env_local_scale2 * env_local2

                if debug_log and debug_info is not None and len(debug_info['steps']) > 0:
                    last = debug_info['steps'][-1]
                    if int(last.get('step', -1)) == int(t):
                        last['env_local2_norm'] = float(env_local2[0].norm().item())
                        last['env_local2_contribution'] = float((self.env_local_scale2.to(env_local2.dtype) * env_local2[0]).norm().item())
                        last['step_input_norm_post_env2'] = float(step_input[0].norm().item())

            # 注入heading特征
            if self.use_heading:
                heading_feat = self.heading_proj(torch.stack([torch.cos(pos_heading), torch.sin(pos_heading)], dim=-1))
                step_input = step_input + self.heading_scale * heading_feat

            out, hidden = self.lstm(step_input.unsqueeze(1), hidden)
            out_sq = out.squeeze(1)
            delta = self.output_layer(out_sq)
            predictions.append(delta)

            # 预测转向角变化
            if self.use_heading:
                hc = self.heading_pred(out_sq)  # (B, 2): [sin(dtheta), cos(dtheta)]
                heading_preds_list.append(hc)

            # 自回归更新
            if self.training and ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_delta = ground_truth[:, t, :]
            else:
                prev_delta = delta
            pos_running = pos_running + prev_delta
            # 更新heading
            if self.use_heading:
                pos_heading = torch.atan2(prev_delta[:, 1], prev_delta[:, 0])

        predictions = torch.stack(predictions, dim=1)

        if debug_log and debug_info is not None and batch_size > 0:
            import os
            debug_dir = getattr(self, 'debug_dir', 'debug_logs')
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f'paper_decoder_debug_{torch.randint(0, 100000, (1,)).item()}.json')
            with open(debug_file, 'w') as f:
                json.dump(debug_info, f, indent=2)
        decoder_aux = {}
        if self.use_heading and len(heading_preds_list) > 0:
            decoder_aux['heading_preds'] = torch.stack(heading_preds_list, dim=1)  # (B, T, 2)
        if return_waypoints:
            return predictions, pred_waypoints, decoder_aux
        return predictions, decoder_aux


class CNNEnvironmentEncoder(nn.Module):
    """CNN环境编码器 - 使用ResNet-18提取环境特征"""
    
    def __init__(self, input_channels: int = 18, feature_dim: int = 128, spatial_from_layer: str = 'layer3'):
        """
        Args:
            input_channels: 输入通道数 (18通道环境地图)
            feature_dim: 输出特征维度 (按照论文设定为128)
        """
        super().__init__()
        
        # 使用预训练的ResNet-18作为backbone
        resnet = models.resnet18(pretrained=False)
        
        # 修改第一层以接受18通道输入
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 全局平均池化 - 按照论文 4.1.1 节，先压缩到 4x4
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 投影到特征维度 (512 * 4 * 4 = 8192)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

        self.token_proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.spatial_from_layer = str(spatial_from_layer)

        self.spatial_proj = nn.Sequential(
            nn.Conv2d(256, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_proj_l2 = nn.Sequential(
            nn.Conv2d(128, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 18, 128, 128) 环境地图
            
        Returns:
            (batch, feature_dim) 环境特征
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x = self.layer4(x3)

        if self.spatial_from_layer == 'layer2':
            spatial = self.spatial_proj_l2(x2)
        else:
            spatial = self.spatial_proj(x3)

        x4 = self.avgpool(x)
        b, c, h, w = x4.shape
        tokens = x4.permute(0, 2, 3, 1).reshape(b, h * w, c)
        tokens = self.token_proj(tokens)

        x_vec = torch.flatten(x4, 1)
        x_vec = self.fc(x_vec)

        return x_vec, tokens, spatial


class LSTMHistoryEncoder(nn.Module):
    """LSTM历史轨迹编码器"""
    
    def __init__(self, input_dim: int = 26, hidden_dim: int = 128, num_layers: int = 2):
        """
        Args:
            input_dim: 输入维度 (按照论文表4.2设定为26)
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, 2) 历史轨迹
            
        Returns:
            output: (batch, seq_len, hidden_dim)
            (h_n, c_n): 最终隐藏状态
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class GoalClassifier(nn.Module):
    """目标分类器 - 对候选终点进行概率评分
    
    按照论文第4.1.3节实现：
    1. MLP_goal: 2D坐标 -> 64D嵌入
    2. MLP_cls: [H;E;C_k] -> logit (三层网络)
    """
    
    def __init__(self, env_feature_dim: int = 128, 
                 history_feature_dim: int = 128,
                 goal_embed_dim: int = 64,
                 hidden_dim: int = 256,
                 num_goals: int = 100,
                 goal_norm_denom: float = 70.0):
        """
        Args:
            env_feature_dim: 环境特征维度 (论文: 128)
            history_feature_dim: 历史特征维度 (论文: 128)
            goal_embed_dim: 目标嵌入维度 (论文: 64)
            hidden_dim: 分类器隐藏层维度
            num_goals: 候选目标数量
        """
        super().__init__()
        
        self.num_goals = num_goals
        self.goal_embed_dim = goal_embed_dim
        self.goal_norm_denom = float(goal_norm_denom)
        
        # MLP_goal: 2D坐标 -> 64D嵌入 (论文要求)
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, goal_embed_dim)
        )
        
        # MLP_cls: [H;E;C_k] -> logit (三层网络，论文要求)
        input_dim = env_feature_dim + history_feature_dim + goal_embed_dim  # 128+128+64=320
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, env_features: torch.Tensor, 
                history_features: torch.Tensor,
                candidate_goals: torch.Tensor) -> torch.Tensor:
        """
        按照论文实现：C_k = MLP_goal([dx_k, dy_k]), s_k = MLP_cls([H; E; C_k])
        
        Args:
            env_features: (batch, 128) 环境特征 E
            history_features: (batch, 128) 历史特征 H
            candidate_goals: (batch, num_goals, 2) 候选目标坐标
            
        Returns:
            logits: (batch, num_goals) 目标分类logits
        """
        batch_size = env_features.size(0)
        num_goals = candidate_goals.size(1)
        
        # Step 1: 编码候选目标 C_k = MLP_goal([dx_k, dy_k])
        denom = float(self.goal_norm_denom) if float(self.goal_norm_denom) > 0 else 1.0
        goal_embeds = self.goal_encoder(candidate_goals / denom)  # (batch, num_goals, 64)
        
        # Step 2: 扩展 H 和 E 到所有候选
        H_expanded = history_features.unsqueeze(1).expand(-1, num_goals, -1)  # (batch, num_goals, 128)
        E_expanded = env_features.unsqueeze(1).expand(-1, num_goals, -1)  # (batch, num_goals, 128)
        
        # Step 3: 拼接 [H; E; C_k]
        combined = torch.cat([H_expanded, E_expanded, goal_embeds], dim=2)  # (batch, num_goals, 320)
        
        # Step 4: 分类器计算 logit s_k = MLP_cls([H; E; C_k])
        logits = self.classifier(combined).squeeze(-1)  # (batch, num_goals)
        
        return logits


class CrossAttention(nn.Module):
    """环境特征与轨迹特征的交叉注意力机制"""
    def __init__(self, query_dim: int, key_dim: int, value_dim: int):
        super().__init__()
        self.scale = query_dim ** -0.5
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, value_dim)
        
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, query_dim) - 轨迹/状态特征
            context: (batch, seq_len, key_dim) 或 (batch, key_dim) - 环境特征
        """
        if context.dim() == 2:
            context = context.unsqueeze(1)
            
        q = self.q_proj(query).unsqueeze(1) # (batch, 1, query_dim)
        k = self.k_proj(context) # (batch, L, query_dim)
        v = self.v_proj(context) # (batch, L, value_dim)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (batch, 1, L)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v).squeeze(1) # (batch, value_dim)
        return out

class HierarchicalLSTMDecoder(nn.Module):
    """层次化LSTM解码器
    
    按照论文第4.1.4节实现：
    - 每个时间步重复输入 [H; E; MLP_goal(g*)]
    - 双层LSTM，隐藏维度256
    - 输出2维位移 Δp_t = FC(h_t^dec)
    """
    
    def __init__(self, input_dim: int = 2,
                 env_feature_dim: int = 128,
                 history_feature_dim: int = 128,
                 goal_feature_dim: int = 2,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 output_length: int = 60,
                 waypoint_stride: int = 0,
                 env_coverage_km: float = 140.0,
                 closed_loop_env_sampling: bool = False,
                 goal_norm_denom: Optional[float] = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length
        self.autoregressive = True  # 实验2: 启用自回归
        self.delta_inject_scale = 1.0

        self.use_token_pos_embed = True
        self.use_pos_condition = True
        self.token_pos_embed = nn.Parameter(torch.randn(16, env_feature_dim) * 0.02)
        self.env_attn = CrossAttention(query_dim=hidden_dim, key_dim=env_feature_dim, value_dim=hidden_dim)

        self.env_coverage_km = float(env_coverage_km)
        self.goal_norm_denom = float(goal_norm_denom) if goal_norm_denom is not None else float(self.env_coverage_km * 0.5)
        self.spatial_in = nn.Sequential(
            nn.Linear(env_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Architectural Fix: Initialize scales to 1.0 to allow strong local environment signal
        self.env_local_scale_wp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.env_local_scale_step = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # Goal vector注入缩放因子（可学习）
        self.goal_vec_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.closed_loop_env_sampling = bool(closed_loop_env_sampling)

        self.waypoint_stride = int(waypoint_stride) if waypoint_stride is not None else 0
        self.waypoint_indices = self._get_waypoint_indices(output_length, self.waypoint_stride)
        self.num_waypoints = len(self.waypoint_indices)
        
        # MLP_goal: 2D目标坐标 -> 64D嵌入（与GoalClassifier一致）
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # 实验1: 可学习的时间步位置编码
        self.time_embed = nn.Parameter(torch.randn(output_length, hidden_dim) * 0.02)

        self.waypoint_time_embed = nn.Parameter(torch.randn(max(0, self.num_waypoints - 1), hidden_dim) * 0.02)
        self.waypoint_out = nn.Linear(hidden_dim, 2)

        self.segment_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 输入投影: [H; E; MLP_goal(g*)] -> hidden_dim
        # 128 + 128 + 64 = 320
        self.input_projection = nn.Sequential(
            nn.Linear(env_feature_dim + history_feature_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 双层LSTM解码器（论文要求）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 输出层: Δp_t = FC(h_t^dec)
        self.output_layer = nn.Linear(hidden_dim, 2)

    def _get_waypoint_indices(self, output_length: int, waypoint_stride: int = 0) -> List[int]:
        if output_length <= 0:
            return []
        if output_length == 1:
            return [0]

        stride = int(waypoint_stride) if waypoint_stride is not None else 0
        if stride and stride > 0:
            idx = list(range(stride - 1, output_length, stride))
            if idx[-1] != (output_length - 1):
                idx.append(output_length - 1)
        else:
            idx = [
                max(0, output_length // 4 - 1),
                max(0, output_length // 2 - 1),
                max(0, (3 * output_length) // 4 - 1),
                output_length - 1,
            ]
        out: List[int] = []
        for i in idx:
            if len(out) == 0 or out[-1] != i:
                out.append(i)
        return out
        
    def forward(self, env_features: torch.Tensor,
                history_features: torch.Tensor,
                goal_features: torch.Tensor,
                current_pos: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5,
                waypoint_teacher_forcing_ratio: Optional[float] = None,
                ground_truth: Optional[torch.Tensor] = None,
                return_waypoints: bool = False):
        """
        实验2: 添加自回归机制 + 时间编码
        
        Args:
            env_features: (batch, 128) 环境特征 E
            history_features: (batch, 128) 历史特征 H
            goal_features: (batch, 2) 目标坐标 g*
            current_pos: 未使用
            teacher_forcing_ratio: 教师强制比率
            ground_truth: (batch, T, 2) 真实轨迹用于teacher forcing
            
        Returns:
            predictions: (batch, output_length, 2) 预测的位移序列
        """
        env_tokens = None
        env_spatial = None
        env_spatial_local = None
        env_global = None
        
        if isinstance(env_features, (tuple, list)):
            if len(env_features) == 2:
                env_global, env_tokens = env_features
            elif len(env_features) == 3:
                env_global, env_tokens, env_spatial = env_features
            elif len(env_features) == 4:
                env_global, env_tokens, env_spatial, env_spatial_local = env_features
            else:
                raise ValueError(f"env_features tuple has unexpected length: {len(env_features)}")
        else:
            env_global = env_features
        
        # 确保env_global是tensor
        if env_global is None or not hasattr(env_global, 'size'):
            raise ValueError(f"env_global is invalid: type={type(env_global)}")
        
        batch_size = env_global.size(0)
        
        # Step 1: 编码目标（先归一化，避免goal坐标量级压制其他输入）
        denom = float(self.goal_norm_denom) if float(self.goal_norm_denom) > 0 else 1.0
        goal_features_raw = goal_features
        goal_features_norm = goal_features_raw / denom
        goal_embed = self.goal_encoder(goal_features_norm)  # (batch, 64)
        
        # Step 2: 基础输入 [H; E; MLP_goal(g*)] with learnable goal scale
        # 使用可学习缩放因子控制goal注入强度
        scaled_goal_embed = goal_embed * self.goal_vec_scale
        base_input = torch.cat([history_features, env_global, scaled_goal_embed], dim=1)  # (batch, 320)
        base_input = self.input_projection(base_input)  # (batch, hidden_dim)

        def _pos_to_grid(pos_xy: torch.Tensor) -> torch.Tensor:
            half = max(1e-6, self.env_coverage_km * 0.5)
            gx = (pos_xy[:, 0] / half).clamp(-1.0, 1.0)
            gy = (-pos_xy[:, 1] / half).clamp(-1.0, 1.0)
            return torch.stack([gx, gy], dim=1)

        def _sample_env(pos_xy: torch.Tensor) -> Optional[torch.Tensor]:
            if env_spatial is None:
                return None
            grid = _pos_to_grid(pos_xy).view(-1, 1, 1, 2)
            samp = F.grid_sample(env_spatial, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            samp = samp.squeeze(-1).squeeze(-1)
            return self.spatial_in(samp)

        if env_tokens is not None and self.use_token_pos_embed:
            env_tokens = env_tokens + self.token_pos_embed.unsqueeze(0)

        pred_waypoints = []
        if self.num_waypoints > 1:
            for wi in range(self.num_waypoints - 1):
                h = base_input + self.waypoint_time_embed[wi].unsqueeze(0)
                if env_tokens is not None and self.use_pos_condition:
                    h = h + self.env_attn(h, env_tokens)

                if env_spatial is not None:
                    frac = float(self.waypoint_indices[wi] + 1) / float(max(1, self.output_length))
                    pos_guess = goal_features_raw * frac
                    env_local = _sample_env(pos_guess)
                    if env_local is not None:
                        h = h + self.env_local_scale_wp * env_local
                pred_waypoints.append(self.waypoint_out(h))
        pred_waypoints.append(goal_features)
        pred_waypoints = torch.stack(pred_waypoints, dim=1)

        waypoints_cond = pred_waypoints
        if self.training and ground_truth is not None and self.num_waypoints > 0:
            wp_tf_ratio = teacher_forcing_ratio if waypoint_teacher_forcing_ratio is None else float(waypoint_teacher_forcing_ratio)
            if torch.rand(1).item() < wp_tf_ratio:
                gt_pos = torch.cumsum(ground_truth, dim=1)
                wp_idx = torch.as_tensor(self.waypoint_indices, device=gt_pos.device, dtype=torch.long)
                if int(wp_idx.max().item()) < gt_pos.size(1):
                    gt_waypoints = gt_pos.index_select(1, wp_idx)
                    if gt_waypoints.size() == pred_waypoints.size():
                        waypoints_cond = gt_waypoints
        
        # 实验2: 自回归解码，逐步生成
        predictions = []
        hidden = None
        prev_delta = torch.zeros(batch_size, 2, device=env_global.device)
        pos_running = torch.zeros(batch_size, 2, device=env_global.device, dtype=base_input.dtype)
        
        # 调试日志：记录解码过程
        debug_log = getattr(self, 'debug_log', False)
        if debug_log and batch_size > 0:
            import json
            debug_info = {
                'closed_loop_env_sampling': self.closed_loop_env_sampling,
                'waypoint_stride': self.waypoint_stride,
                'num_waypoints': self.num_waypoints,
                'output_length': self.output_length,
                'env_local_scale_step': float(self.env_local_scale_step.item()) if hasattr(self.env_local_scale_step, 'item') else float(self.env_local_scale_step),
                'goal_vec_scale': float(self.goal_vec_scale.item()),
                'debug_stride': int(getattr(self, 'debug_stride', 60)),
                'goal_norm_denom': float(denom),
                'goal_features': [float(goal_features_raw[0, 0]), float(goal_features_raw[0, 1])],
                'goal_features_norm': [float(goal_features_norm[0, 0]), float(goal_features_norm[0, 1])],
                'history_norm': float(history_features[0].norm().item()),
                'env_global_norm': float(env_global[0].norm().item()),
                'goal_embed_norm': float(goal_embed[0].norm().item()),
                'scaled_goal_embed_norm': float(scaled_goal_embed[0].norm().item()),
                'base_input_norm': float(base_input[0].norm().item()),
                'steps': []
            }
        
        wp0 = torch.zeros(batch_size, 1, 2, device=env_global.device, dtype=base_input.dtype)
        wp_nodes = torch.cat([wp0, waypoints_cond], dim=1)

        seg_bounds = []
        prev = -1
        for end_t in self.waypoint_indices:
            start_t = prev + 1
            seg_bounds.append((start_t, end_t))
            prev = end_t

        for t in range(self.output_length):
            seg_id = 0
            for si, (s, e) in enumerate(seg_bounds):
                if s <= t <= e:
                    seg_id = si
                    break

            s, e = seg_bounds[seg_id]
            denom = float(max(1, e - s))
            prog = float(t - s) / denom

            start_wp = wp_nodes[:, seg_id, :]
            end_wp = wp_nodes[:, seg_id + 1, :]
            seg_in = torch.cat([
                start_wp,
                end_wp,
                torch.full((batch_size, 1), prog, device=env_global.device, dtype=base_input.dtype),
            ], dim=1)
            seg_feat = self.segment_proj(seg_in)

            step_input = base_input + self.time_embed[t].unsqueeze(0) + seg_feat
            attn_out = None
            if env_tokens is not None and self.use_pos_condition:
                attn_out = self.env_attn(step_input, env_tokens)
                step_input = step_input + attn_out

            if env_spatial is not None:
                pos_query = pos_running if self.closed_loop_env_sampling else (start_wp + (end_wp - start_wp) * float(prog))
                env_local = _sample_env(pos_query)
                if env_local is not None:
                    step_input = step_input + self.env_local_scale_step * env_local
                    
                    # 调试日志：记录环境采样信息
                    if debug_log:
                        stride = int(getattr(self, 'debug_stride', 60))
                        stride = max(1, stride)
                    if debug_log and (t % stride == 0):
                        delta_pad_norm = None
                        prev_delta_xy = None
                        if self.autoregressive:
                            delta_pad = torch.zeros(batch_size, self.hidden_dim, device=env_global.device)
                            delta_pad[:, :2] = prev_delta * self.delta_inject_scale
                            delta_pad_norm = float(delta_pad[0].norm().item())
                            prev_delta_xy = [float(prev_delta[0, 0].item()), float(prev_delta[0, 1].item())]
                        debug_info['steps'].append({
                            'step': t,
                            'seg_id': seg_id,
                            'progress': float(prog),
                            'start_wp': [float(start_wp[0, 0]), float(start_wp[0, 1])],
                            'end_wp': [float(end_wp[0, 0]), float(end_wp[0, 1])],
                            'pos_running': [float(pos_running[0, 0]), float(pos_running[0, 1])],
                            'pos_query': [float(pos_query[0, 0]), float(pos_query[0, 1])],
                            'time_embed_norm': float(self.time_embed[t].norm().item()),
                            'seg_feat_norm': float(seg_feat[0].norm().item()),
                            'attn_norm': float(attn_out[0].norm().item()) if attn_out is not None else None,
                            'env_local_norm': float(env_local[0].norm().item()),
                            'env_contribution': float((self.env_local_scale_step * env_local[0]).norm().item()),
                            'delta_pad_norm': delta_pad_norm,
                            'prev_delta_xy': prev_delta_xy,
                            'step_input_norm': float(step_input[0].norm().item()),
                        })
            
            # 注入前一步delta（自回归）
            if self.autoregressive:
                delta_pad = torch.zeros(batch_size, self.hidden_dim, device=env_global.device)
                delta_pad[:, :2] = prev_delta * self.delta_inject_scale
                step_input = step_input + delta_pad
            
            # LSTM前向
            lstm_out, hidden = self.lstm(step_input.unsqueeze(1), hidden)
            
            # 预浌当前步delta
            delta = self.output_layer(lstm_out.squeeze(1))  # (batch, 2)
            predictions.append(delta)
            
            # Teacher forcing
            if self.training and ground_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_delta = ground_truth[:, t, :]
            else:
                prev_delta = delta

            pos_running = pos_running + prev_delta
        
        predictions = torch.stack(predictions, dim=1)  # (batch, T, 2)
        
        # 调试日志：保存到文件
        if debug_log and batch_size > 0:
            import os
            debug_dir = getattr(self, 'debug_dir', 'debug_logs')
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f'decoder_debug_{torch.randint(0, 100000, (1,)).item()}.json')
            with open(debug_file, 'w') as f:
                json.dump(debug_info, f, indent=2)
        
        if return_waypoints:
            return predictions, pred_waypoints
        return predictions


class TerraTNT(nn.Module):
    """TerraTNT完整模型"""
    
    def __init__(self, config: dict = None, **kwargs):
        """
        Args:
            config: 配置字典，包含以下键：
                - env_channels: 环境地图通道数 (默认18)
                - env_feature_dim: 环境特征维度 (默认128)
                - history_hidden_dim: 历史编码器隐藏维度 (默认128)
                - decoder_hidden_dim: 解码器隐藏维度 (默认256)
                - num_goals: 候选目标数量 (默认100)
                - output_length/future_len: 输出轨迹长度 (默认60)
        """
        super().__init__()
        
        # 支持两种初始化方式：config字典或直接参数
        if config is None:
            config = kwargs
        
        # 提取配置参数
        env_channels = config.get('env_channels', 18)
        env_feature_dim = config.get('env_feature_dim', 128)
        history_hidden_dim = config.get('history_hidden_dim', config.get('history_feature_dim', 128))
        history_input_dim = config.get('history_input_dim', 2)
        decoder_hidden_dim = config.get('decoder_hidden_dim', 256)
        num_goals = config.get('num_goals', config.get('num_goals', 100))
        output_length = config.get('output_length', config.get('future_len', 60))
        waypoint_stride = config.get('waypoint_stride', None)
        env_coverage_km = config.get('env_coverage_km', 140.0)
        env_local_coverage_km = config.get('env_local_coverage_km', 10.0)
        goal_norm_denom = config.get('goal_norm_denom', None)
        if goal_norm_denom is None:
            goal_norm_denom = float(env_coverage_km) * 0.5
        goal_norm_denom = float(goal_norm_denom)
        goal_vec_use_waypoint = bool(config.get('goal_vec_use_waypoint', False))
        closed_loop_env_sampling = config.get('closed_loop_env_sampling', False)
        spatial_from_layer = config.get('spatial_from_layer', 'layer3')
        
        # 核心修改: 默认启用 Paper Mode (简化架构)
        paper_mode = bool(config.get('paper_mode', True))
        paper_decoder = str(config.get('paper_decoder', 'flat'))

        if waypoint_stride is None:
            waypoint_stride = 0

        if paper_mode and paper_decoder != 'flat':
            if int(waypoint_stride) <= 0:
                waypoint_stride = max(1, int(math.ceil(float(output_length) / 4.0)))
        
        self.num_goals = num_goals
        self.output_length = output_length
        self.paper_mode = bool(paper_mode)
        self.paper_decoder = str(paper_decoder)
        self.use_dual_scale = bool(config.get('use_dual_scale', False))
        
        # 环境编码器
        if self.use_dual_scale:
            self.env_encoder = DualScaleEnvironmentEncoder(
                input_channels=env_channels,
                feature_dim=env_feature_dim,
                spatial_res=32
            )
        elif self.paper_mode:
            self.env_encoder = PaperCNNEnvironmentEncoder(
                input_channels=env_channels,
                feature_dim=env_feature_dim,
            )
        else:
            self.env_encoder = CNNEnvironmentEncoder(
                input_channels=env_channels,
                feature_dim=env_feature_dim,
                spatial_from_layer=spatial_from_layer,
            )
        
        # 历史轨迹编码器
        if self.paper_mode:
            self.history_encoder = PaperLSTMHistoryEncoder(
                input_dim=history_input_dim,
                hidden_dim=history_hidden_dim,
                num_layers=2,
            )
        else:
            self.history_encoder = LSTMHistoryEncoder(
                input_dim=history_input_dim,
                hidden_dim=history_hidden_dim,
                num_layers=2
            )
        
        # 目标分类器
        if self.paper_mode:
            self.goal_classifier = PaperGoalClassifier(
                env_feature_dim=env_feature_dim,
                history_feature_dim=history_hidden_dim,
                num_goals=num_goals,
                goal_norm_denom=float(goal_norm_denom),
            )
        else:
            self.goal_classifier = GoalClassifier(
                env_feature_dim=env_feature_dim,
                history_feature_dim=history_hidden_dim,
                num_goals=num_goals,
                goal_norm_denom=float(goal_norm_denom),
            )
        
        # 轨迹解码器
        if self.paper_mode:
            if self.paper_decoder == 'flat':
                self.decoder = PaperTrajectoryDecoder(
                    input_dim=2,
                    env_feature_dim=env_feature_dim,
                    history_feature_dim=history_hidden_dim,
                    goal_feature_dim=2,
                    hidden_dim=decoder_hidden_dim,
                    num_layers=2,
                    output_length=output_length,
                    goal_norm_denom=float(goal_norm_denom),
                )
            else:
                self.decoder = PaperHierarchicalTrajectoryDecoder(
                    input_dim=2,
                    env_feature_dim=env_feature_dim,
                    history_feature_dim=history_hidden_dim,
                    goal_feature_dim=2,
                    hidden_dim=decoder_hidden_dim,
                    num_layers=2,
                    output_length=output_length,
                    waypoint_stride=int(waypoint_stride),
                    env_coverage_km=float(env_coverage_km),
                    env_local_coverage_km=float(env_local_coverage_km),
                    goal_vec_use_waypoint=bool(goal_vec_use_waypoint),
                    goal_norm_denom=float(goal_norm_denom),
                    use_heading=bool(config.get('use_heading', False)),
                )
        else:
            self.decoder = HierarchicalLSTMDecoder(
                input_dim=2,
                env_feature_dim=env_feature_dim,
                history_feature_dim=history_hidden_dim,
                goal_feature_dim=2,
                hidden_dim=decoder_hidden_dim,
                num_layers=2,
                output_length=output_length,
                waypoint_stride=waypoint_stride,
                env_coverage_km=env_coverage_km,
                closed_loop_env_sampling=closed_loop_env_sampling,
                goal_norm_denom=float(goal_norm_denom),
            )
        
    def forward(self, env_map: torch.Tensor,
                history: torch.Tensor,
                candidate_goals: torch.Tensor,
                current_pos: torch.Tensor,
                teacher_forcing_ratio: float = 0.5,
                ground_truth: Optional[torch.Tensor] = None,
                target_goal_idx: Optional[torch.Tensor] = None,
                use_gt_goal: bool = False,
                use_soft_goal: bool = False,
                goal_temperature: float = 1.0,
                goal: Optional[torch.Tensor] = None,
                return_aux: bool = False,
                waypoint_teacher_forcing_ratio: Optional[float] = None,
                env_map_local: Optional[torch.Tensor] = None):
        """
        Args:
            env_map: (batch, 18, 128, 128) 全局环境地图 (140km)
            env_map_local: (batch, 18, 128, 128) 局部环境地图 (10km)
            ...
        """
        # 编码环境
        local_seg_logits = None
        if env_map_local is not None and hasattr(self, 'use_dual_scale') and self.use_dual_scale:
            out = self.env_encoder(env_map, env_map_local)
            if len(out) == 5:
                env_global, env_tokens, env_spatial, env_spatial_local, local_seg_logits = out
            else:
                env_global, env_tokens, env_spatial, env_spatial_local = out
        else:
            env_global, env_tokens, env_spatial = self.env_encoder(env_map)
            env_spatial_local = None
        
        # 编码历史轨迹
        history_output, (h_n, c_n) = self.history_encoder(history)
        history_features = h_n[-1]  # 使用最后一层的隐藏状态 (batch, history_hidden_dim)
        
        goal_logits = None
        if goal is not None:
            selected_goals = goal  # (batch, 2)
        else:
            # 目标分类 (返回 Logits)
            goal_logits = self.goal_classifier(env_global, history_features, candidate_goals)

            # 选择用于解码的目标
            if use_gt_goal and target_goal_idx is not None:
                selected_goals = candidate_goals[torch.arange(candidate_goals.size(0)), target_goal_idx]  # (batch, 2)
            elif use_soft_goal:
                probs = F.softmax(goal_logits / goal_temperature, dim=1)  # (batch, num_goals)
                selected_goals = torch.sum(candidate_goals * probs.unsqueeze(-1), dim=1)  # (batch, 2)
            else:
                # 选择最可能的目标 (Logits 不影响 Max 选择)
                _, top_goal_idx = torch.max(goal_logits, dim=1)
                selected_goals = candidate_goals[torch.arange(candidate_goals.size(0)), top_goal_idx]  # (batch, 2)
        
        if return_aux:
            predictions, pred_waypoints, decoder_aux = self.decoder(
                env_features=(env_global, env_tokens, env_spatial, env_spatial_local),
                history_features=history_features,
                goal_features=selected_goals,
                current_pos=current_pos,
                teacher_forcing_ratio=teacher_forcing_ratio,
                waypoint_teacher_forcing_ratio=waypoint_teacher_forcing_ratio,
                ground_truth=ground_truth,
                return_waypoints=True,
            )
            aux = {
                'pred_waypoints': pred_waypoints,
                'waypoint_indices': self.decoder.waypoint_indices,
                'local_seg_logits': local_seg_logits,
            }
            if decoder_aux:
                aux.update(decoder_aux)
            return predictions, goal_logits, aux

        predictions, decoder_aux = self.decoder(
            env_features=(env_global, env_tokens, env_spatial, env_spatial_local),
            history_features=history_features,
            goal_features=selected_goals,
            current_pos=current_pos,
            teacher_forcing_ratio=teacher_forcing_ratio,
            waypoint_teacher_forcing_ratio=waypoint_teacher_forcing_ratio,
            ground_truth=ground_truth
        )

        return predictions, goal_logits
    
    def predict(self, env_map: torch.Tensor,
                history: torch.Tensor,
                candidate_goals: torch.Tensor,
                current_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        推理模式
        
        Returns:
            predictions: (batch, output_length, 2) 预测轨迹
            goal_probs: (batch, num_goals) 目标概率
            selected_goals: (batch, 2) 选择的目标
        """
        self.eval()
        with torch.no_grad():
            predictions, goal_logits = self.forward(
                env_map=env_map,
                history=history,
                candidate_goals=candidate_goals,
                current_pos=current_pos,
                teacher_forcing_ratio=0.0  # 推理时不使用教师强制
            )
            
            goal_probs = F.softmax(goal_logits, dim=1)
            _, top_goal_idx = torch.max(goal_probs, dim=1)
            selected_goals = candidate_goals[torch.arange(candidate_goals.size(0)), top_goal_idx]
            
        return predictions, goal_probs, selected_goals


def test_model():
    """测试模型"""
    logger.info("=" * 60)
    logger.info("测试TerraTNT模型")
    logger.info("=" * 60)
    
    # 创建模型
    model = TerraTNT(
        env_channels=18,
        env_feature_dim=256,
        history_hidden_dim=128,
        decoder_hidden_dim=256,
        num_goals=100,
        output_length=60
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n模型参数:")
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 4
    env_map = torch.randn(batch_size, 18, 128, 128)
    history = torch.randn(batch_size, 10, 2)
    candidate_goals = torch.randn(batch_size, 100, 2)
    current_pos = torch.randn(batch_size, 2)
    ground_truth = torch.randn(batch_size, 60, 2)
    
    logger.info(f"\n测试前向传播:")
    logger.info(f"  输入形状:")
    logger.info(f"    env_map: {env_map.shape}")
    logger.info(f"    history: {history.shape}")
    logger.info(f"    candidate_goals: {candidate_goals.shape}")
    logger.info(f"    current_pos: {current_pos.shape}")
    
    predictions, goal_probs = model(
        env_map=env_map,
        history=history,
        candidate_goals=candidate_goals,
        current_pos=current_pos,
        teacher_forcing_ratio=0.5,
        ground_truth=ground_truth
    )
    
    logger.info(f"\n  输出形状:")
    logger.info(f"    predictions: {predictions.shape}")
    logger.info(f"    goal_probs: {goal_probs.shape}")
    
    logger.info("\n✅ 模型测试成功！")


if __name__ == '__main__':
    test_model()
