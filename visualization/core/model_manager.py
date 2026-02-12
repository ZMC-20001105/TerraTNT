#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器 - 统一加载和管理所有模型，支持推理
"""
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

HISTORY_LEN = 90
FUTURE_LEN = 360


class ModelManager:
    """统一模型管理器"""

    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_types: Dict[str, str] = {}  # name -> type tag
        self.runs_dir = PROJECT_ROOT / 'runs'

    def discover_checkpoints(self) -> Dict[str, Path]:
        """自动发现所有可用的模型checkpoint"""
        found = {}
        patterns = {
            'TerraTNT': ['plan_b_fas1/terratnt_fas1_10s/*/best_model.pth'],
            'V3_Waypoint': ['incremental_models/V3_best.pth'],
            'V4_WP_Spatial': ['incremental_models/V4_best.pth'],
            'V6_Autoreg': ['incremental_models/V6_best.pth'],
            'V6R_Robust': ['incremental_models_v6r/V6_best.pth', 'incremental_models/V6R_best.pth'],
            'V7_ConfGate': ['incremental_models_v7/V7_best.pth'],
            'LSTM_only': ['LSTM_only_d1/best_model.pth'],
            'LSTM_Env_Goal': ['LSTM_Env_Goal_d1/best_model.pth'],
            'Seq2Seq_Attn': ['Seq2Seq_Attn_d1/best_model.pth'],
            'MLP': ['MLP_d1/best_model.pth'],
        }
        for name, pats in patterns.items():
            for pat in pats:
                matches = list(self.runs_dir.glob(pat))
                if matches:
                    found[name] = matches[-1]  # latest
                    break
        return found

    def load_model(self, name: str, checkpoint_path: Path) -> bool:
        """加载单个模型"""
        try:
            if name in ('LSTM_only', 'LSTM_Env_Goal', 'Seq2Seq_Attn', 'MLP'):
                return self._load_baseline(name, checkpoint_path)
            elif name == 'TerraTNT':
                return self._load_terratnt(name, checkpoint_path)
            elif name.startswith('V'):
                return self._load_incremental(name, checkpoint_path)
            return False
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            return False

    def _load_baseline(self, name: str, ckpt_path: Path) -> bool:
        from scripts.train_eval_all_baselines import (
            LSTMOnly, LSTMEnvGoal, Seq2SeqAttention, MLPBaseline,
        )
        cls_map = {
            'LSTM_only': (LSTMOnly, dict(hidden_dim=256, future_len=FUTURE_LEN)),
            'LSTM_Env_Goal': (LSTMEnvGoal, dict(hidden_dim=256, future_len=FUTURE_LEN)),
            'Seq2Seq_Attn': (Seq2SeqAttention, dict(hidden_dim=256, future_len=FUTURE_LEN)),
            'MLP': (MLPBaseline, dict(hidden_dim=512, future_len=FUTURE_LEN, history_len=HISTORY_LEN)),
        }
        cls, kwargs = cls_map[name]
        m = cls(**kwargs).to(self.device)
        m.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False))
        m.eval()
        self.models[name] = m
        self.model_types[name] = 'baseline'
        print(f"  [OK] {name}")
        return True

    def _load_terratnt(self, name: str, ckpt_path: Path) -> bool:
        from models.terratnt import TerraTNT
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get('model_state_dict', {})

        has_base_proj = any(k.startswith('decoder.base_proj.') for k in sd)
        paper_decoder = 'hierarchical' if has_base_proj else 'flat'
        use_dual_scale = any(k.startswith('env_encoder.local_encoder.') for k in sd)
        waypoint_stride = int(ckpt.get('waypoint_stride', 90))
        wte = sd.get('decoder.waypoint_time_embed', None)
        if isinstance(wte, torch.Tensor) and wte.dim() == 2 and wte.shape[0] > 0:
            waypoint_stride = int(FUTURE_LEN // (wte.shape[0] + 1))
        hist_in = 26
        hist_hidden = 128  # default
        wih = sd.get('history_encoder.lstm.weight_ih_l0', None)
        if isinstance(wih, torch.Tensor) and wih.dim() == 2:
            hist_in = int(wih.shape[1])
            hist_hidden = int(wih.shape[0] // 4)  # LSTM: 4*hidden_dim
        num_goals = int(ckpt.get('num_candidates', 6))

        m = TerraTNT(
            history_input_dim=hist_in, history_hidden_dim=hist_hidden,
            env_channels=18, env_feature_dim=hist_hidden,
            future_len=FUTURE_LEN, num_goals=num_goals,
            waypoint_stride=waypoint_stride, paper_decoder=paper_decoder,
            use_dual_scale=use_dual_scale,
        ).to(self.device)
        m.load_state_dict(sd)
        m.eval()
        self.models[name] = m
        self.model_types[name] = 'terratnt'
        print(f"  [OK] {name}")
        return True

    def _load_incremental(self, name: str, ckpt_path: Path) -> bool:
        from scripts.train_incremental_models import (
            LSTMEnvGoalWaypoint, LSTMEnvGoalWaypointSpatial,
            TerraTNTAutoregV6, ConfidenceGatedV7,
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)

        # 自动推断hidden_dim: 从checkpoint的LSTM权重shape推断
        inferred_hidden = 128  # 默认
        for k in sd:
            if 'weight_ih_l0' in k:
                inferred_hidden = sd[k].shape[0] // 4
                break

        cls_map = {
            'V3_Waypoint': (LSTMEnvGoalWaypoint, dict(hidden_dim=inferred_hidden, future_len=FUTURE_LEN, num_waypoints=10)),
            'V4_WP_Spatial': (LSTMEnvGoalWaypointSpatial, dict(hidden_dim=inferred_hidden, future_len=FUTURE_LEN, num_waypoints=10, env_coverage_km=140.0)),
            'V6_Autoreg': (TerraTNTAutoregV6, dict(hidden_dim=inferred_hidden, future_len=FUTURE_LEN, num_candidates=6, env_coverage_km=140.0)),
            'V6R_Robust': (TerraTNTAutoregV6, dict(hidden_dim=inferred_hidden, future_len=FUTURE_LEN, num_candidates=6, env_coverage_km=140.0)),
            'V7_ConfGate': (ConfidenceGatedV7, dict(hidden_dim=inferred_hidden, future_len=FUTURE_LEN, num_candidates=6, env_coverage_km=140.0)),
        }
        if name not in cls_map:
            return False
        cls, kwargs = cls_map[name]
        m = cls(**kwargs).to(self.device)
        m.load_state_dict(sd)
        m.eval()
        self.models[name] = m
        self.model_types[name] = name.split('_')[0].lower()
        print(f"  [OK] {name}")
        return True

    def load_all_available(self) -> int:
        """加载所有可用模型"""
        ckpts = self.discover_checkpoints()
        count = 0
        for name, path in ckpts.items():
            if self.load_model(name, path):
                count += 1
        return count

    @torch.no_grad()
    def predict(self, name: str, history_feat: np.ndarray, env_map: np.ndarray,
                candidates_rel: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """对单个样本推理，返回预测轨迹 (T, 2) km
        
        Args:
            history_feat: (90, 26) 完整历史特征, 前2列是 dx/dy (km)
            env_map: (18, 128, 128) 环境地图
            candidates_rel: (C, 2) 候选终点相对坐标 (km)
        """
        if name not in self.models:
            return None
        if name == 'ConstantVelocity':
            return self._cv_predict(history_feat)

        model = self.models[name]
        mtype = self.model_types[name]

        # 完整26维特征
        h_full = torch.from_numpy(history_feat).unsqueeze(0).float().to(self.device)
        # 仅 dx/dy (2维) — V3/V4/baselines 需要
        h_xy = h_full[:, :, :2]
        e = torch.from_numpy(env_map).unsqueeze(0).float().to(self.device)

        # 候选终点
        if candidates_rel is not None:
            c = torch.from_numpy(candidates_rel).unsqueeze(0).float().to(self.device)
            # 确保至少6个候选点 (padding)
            if c.shape[1] < 6:
                pad = torch.zeros(1, 6 - c.shape[1], 2, device=self.device)
                c = torch.cat([c, pad], dim=1)
        else:
            c = torch.zeros(1, 6, 2, device=self.device)

        # current_pos 设为0 (相对坐标原点)
        current_pos = torch.zeros(1, 2, device=self.device)

        try:
            if mtype == 'baseline' and name == 'LSTM_Env_Goal':
                # LSTM_Env_Goal: 从env_map ch17热力图提取goal (与评估脚本一致)
                hm = env_map[17]  # (128, 128)
                hm_goal = self._extract_goal_from_heatmap(hm)
                goal_t = torch.from_numpy(hm_goal).unsqueeze(0).float().to(self.device)
                pred = model(h_xy, e, goal=goal_t)
            elif mtype == 'baseline':
                # LSTMOnly, MLP, Seq2Seq: 不使用goal
                pred = model(h_xy, e)
            elif mtype == 'terratnt':
                # TerraTNT: forward(history_26d, env_map, candidates)
                pred, _ = model(h_full, e, c)
            elif mtype in ('v3', 'v4'):
                # V3/V4: forward(history_xy, env_map, goal)  input_dim=2
                # goal = 候选终点中的第一个
                goal = c[:, 0, :]  # (1, 2)
                pred = model(h_xy, e, goal)
            elif mtype in ('v6', 'v6r'):
                # V6: forward(env_map, history_26d, candidates, current_pos)
                result = model(e, h_full, c, current_pos)
                if isinstance(result, tuple):
                    pred = result[0]  # (predictions, goal_logits)
                else:
                    pred = result
            elif mtype == 'v7':
                # V7: forward(env_map, history_26d, candidates, current_pos)
                result = model(e, h_full, c, current_pos)
                if isinstance(result, tuple):
                    pred = result[0]  # (predictions, goal_logits, alpha_raw)
                else:
                    pred = result
            else:
                pred = model(h_xy, e)

            result = pred.squeeze(0).cpu().numpy()
            if result.ndim == 2 and result.shape[1] == 2:
                return result
            return result.reshape(-1, 2)
        except Exception as ex:
            print(f"  [PREDICT FAIL] {name} (type={mtype}): {ex}")
            return None

    def _extract_goal_from_heatmap(self, heatmap: np.ndarray,
                                    env_coverage_km: float = 140.0) -> np.ndarray:
        """从热力图提取加权质心坐标 (km), 与evaluate_phases_v2一致"""
        h, w = heatmap.shape
        coverage_m = env_coverage_km * 1000.0
        resolution_m = coverage_m / w
        half = coverage_m / 2.0
        px_x = (np.arange(w) + 0.5) * resolution_m - half
        px_y = half - (np.arange(h) + 0.5) * resolution_m
        total_w = heatmap.sum()
        if total_w < 1e-8:
            return np.zeros(2, dtype=np.float32)
        cx_m = (heatmap * px_x[None, :]).sum() / total_w
        cy_m = (heatmap * px_y[:, None]).sum() / total_w
        return np.array([cx_m / 1000.0, cy_m / 1000.0], dtype=np.float32)

    def _cv_predict(self, history_feat: np.ndarray) -> np.ndarray:
        """Constant Velocity baseline"""
        if history_feat.shape[0] >= 2:
            vel = history_feat[-1, :2] - history_feat[-2, :2]
        else:
            vel = np.array([0.001, 0.0])
        positions = np.cumsum(history_feat[:, :2], axis=0)
        last_pos = positions[-1]
        future = np.zeros((FUTURE_LEN, 2))
        for t in range(FUTURE_LEN):
            future[t] = last_pos + vel * (t + 1)
        # 转为相对坐标 (减去累积历史的最后一个点)
        future = future - last_pos
        return future

    def get_loaded_models(self) -> list:
        return list(self.models.keys())
