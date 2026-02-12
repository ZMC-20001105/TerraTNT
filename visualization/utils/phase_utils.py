#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase感知推理工具: 根据Phase修改env_map ch17 + 生成候选终点"""
import numpy as np

ENV_COV_KM = 140.0
MAP_SZ = 128


def _pixel_grids(coverage_km=ENV_COV_KM, sz=MAP_SZ):
    cov_m = coverage_km * 1000.0
    res = cov_m / sz
    half = cov_m / 2.0
    px_x = (np.arange(sz) + 0.5) * res - half
    px_y = half - (np.arange(sz) + 0.5) * res
    return px_x, px_y


def make_heatmap_p1(goal_rel_km, sigma_km=1.0):
    """Phase1: 精确终点高斯 (σ=1km)"""
    px_x, px_y = _pixel_grids()
    gx = goal_rel_km[0] * 1000.0
    gy = goal_rel_km[1] * 1000.0
    dx = px_x[None, :] - gx
    dy = px_y[:, None] - gy
    sigma_m = sigma_km * 1000.0
    hm = np.exp(-(dx**2 + dy**2) / (2.0 * sigma_m**2))
    mx = hm.max()
    if mx > 0:
        hm /= mx
    return hm.astype(np.float32)


def make_heatmap_p2(goal_rel_km, env_map, sigma_km=10.0,
                    center_offset_km=None, use_road=True):
    """Phase2: 区域先验宽高斯 (σ=10km), 可选道路掩膜"""
    px_x, px_y = _pixel_grids()
    center = np.array(goal_rel_km, dtype=np.float64).copy()
    if center_offset_km is not None:
        center += np.asarray(center_offset_km, dtype=np.float64)
    cx = center[0] * 1000.0
    cy = center[1] * 1000.0
    dx = px_x[None, :] - cx
    dy = px_y[:, None] - cy
    sigma_m = sigma_km * 1000.0
    hm = np.exp(-(dx**2 + dy**2) / (2.0 * sigma_m**2))
    if use_road and env_map.ndim == 3 and env_map.shape[0] >= 16:
        road = env_map[15]
        hm = hm * (road > 0.5).astype(np.float32)
    mx = hm.max()
    if mx > 0:
        hm /= mx
    return hm.astype(np.float32)


def make_heatmap_p3(history_feat, env_map, angle_deg=60.0):
    """Phase3: 无先验, 运动方向扇形"""
    px_x, px_y = _pixel_grids()
    cov_m = ENV_COV_KM * 1000.0
    # 从历史推断方向
    if history_feat is not None and len(history_feat) >= 10:
        vel = history_feat[-1, :2] - history_feat[-10, :2]
        vel = vel / 10.0
    elif history_feat is not None and len(history_feat) >= 2:
        vel = history_feat[-1, :2] - history_feat[-2, :2]
    else:
        vel = np.array([0.01, 0.0])
    speed = np.linalg.norm(vel)
    heading = np.arctan2(vel[1], vel[0]) if speed > 1e-6 else 0.0
    speed = max(speed, 0.01)
    est_d = np.clip(speed * 360, 10.0, 80.0) * 1000.0
    xx = px_x[None, :]
    yy = px_y[:, None]
    dist = np.sqrt(xx**2 + yy**2)
    dist_ok = (dist >= est_d * 0.3) & (dist <= est_d * 2.0)
    ang = np.arctan2(yy, xx)
    ang_diff = np.abs(np.arctan2(np.sin(ang - heading), np.cos(ang - heading)))
    ang_ok = ang_diff < np.radians(angle_deg)
    road_ok = np.ones_like(dist, dtype=bool)
    if env_map.ndim == 3 and env_map.shape[0] >= 16:
        road_ok = env_map[15] > 0.5
    hm = (dist_ok & ang_ok & road_ok).astype(np.float32)
    if hm.max() < 0.5:
        hm = (dist_ok & ang_ok).astype(np.float32)
    if hm.max() < 0.5:
        hm = dist_ok.astype(np.float32)
    mx = hm.max()
    if mx > 0:
        hm /= mx
    return hm.astype(np.float32)


def make_heatmap_interactive(center_km, sigma_km=10.0, env_map=None, use_road=True):
    """用户交互: 在指定位置生成高斯热力图 (用于P2a交互模式)"""
    px_x, px_y = _pixel_grids()
    cx = center_km[0] * 1000.0
    cy = center_km[1] * 1000.0
    dx = px_x[None, :] - cx
    dy = px_y[:, None] - cy
    sigma_m = sigma_km * 1000.0
    hm = np.exp(-(dx**2 + dy**2) / (2.0 * sigma_m**2))
    if use_road and env_map is not None and env_map.ndim == 3 and env_map.shape[0] >= 16:
        hm = hm * (env_map[15] > 0.5).astype(np.float32)
    mx = hm.max()
    if mx > 0:
        hm /= mx
    return hm.astype(np.float32)


def sample_candidates(heatmap, num=6, include_gt=False, goal_rel_km=None, seed=42):
    """从热力图采样候选终点, 返回 (K,2) km"""
    rng = np.random.default_rng(seed)
    px_x, px_y = _pixel_grids()
    w = heatmap.flatten()
    total = w.sum()
    if total > 1e-8:
        p = w / total
        idx = rng.choice(len(p), size=num, replace=True, p=p)
        rows = idx // MAP_SZ
        cols = idx % MAP_SZ
        cands = np.stack([px_x[cols] / 1000.0, px_y[rows] / 1000.0], axis=1)
    else:
        angles = rng.uniform(0, 2 * np.pi, num)
        dists = rng.uniform(10.0, 50.0, num)
        cands = np.stack([dists * np.cos(angles), dists * np.sin(angles)], axis=1)
    if include_gt and goal_rel_km is not None and num > 0:
        ins = int(rng.integers(0, num))
        cands[ins] = goal_rel_km[:2]
    return cands.astype(np.float32)


def prepare_phase_inputs(sample, phase_key, user_candidates=None,
                         user_prior_center_km=None, user_prior_sigma_km=10.0):
    """根据Phase准备推理输入: 修改env_map ch17 + 生成候选终点

    Args:
        sample: SampleData对象
        phase_key: 'P1a','P1b','P2a','P3a'
        user_candidates: 用户手动放置的候选终点 (N,2) km, 用于P1a/P1b交互
        user_prior_center_km: 用户点击的区域先验中心 (2,) km, 用于P2a交互

    Returns:
        env_map: (18,128,128) 修改后的env_map
        candidates: (K,2) km 候选终点
    """
    env = sample.env_map.copy()
    goal = sample.goal_rel

    if phase_key == 'P1a':
        hm = make_heatmap_p1(goal, sigma_km=1.0)
        env[17] = hm
        if user_candidates is not None and len(user_candidates) > 0:
            cands = np.array(user_candidates, dtype=np.float32)
        else:
            cands = sample_candidates(hm, num=6, include_gt=True, goal_rel_km=goal)

    elif phase_key == 'P1b':
        hm = make_heatmap_p1(goal, sigma_km=1.0)
        env[17] = hm
        if user_candidates is not None and len(user_candidates) > 0:
            cands = np.array(user_candidates, dtype=np.float32)
        else:
            cands = sample_candidates(hm, num=6, include_gt=False, goal_rel_km=goal)

    elif phase_key == 'P2a':
        if user_prior_center_km is not None:
            hm = make_heatmap_interactive(
                user_prior_center_km, sigma_km=user_prior_sigma_km,
                env_map=env, use_road=True)
        else:
            hm = make_heatmap_p2(goal, env, sigma_km=10.0)
        env[17] = hm
        cands = sample_candidates(hm, num=6, include_gt=False, goal_rel_km=goal)

    elif phase_key == 'P3a':
        hist_xy = sample.history_feat[:, :2] if sample.history_feat is not None else None
        hm = make_heatmap_p3(hist_xy, env)
        env[17] = hm
        cands = sample_candidates(hm, num=6, include_gt=False, goal_rel_km=goal)

    else:
        hm = np.zeros((MAP_SZ, MAP_SZ), dtype=np.float32)
        env[17] = hm
        cands = sample.candidates_rel.copy()

    return env, cands
