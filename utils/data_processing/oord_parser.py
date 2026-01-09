"""
OORD 轨迹解析模块（待实现）
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import cfg, get_path

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    region: str
    run_id: str
    timestamps_us: np.ndarray
    positions_xy: np.ndarray
    velocities_xy: np.ndarray
    speed: np.ndarray
    heading_deg: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    meta: Dict[str, Any]


def _find_extracted_runs(extracted_root: Path) -> List[Tuple[str, str, Path]]:
     runs: List[Tuple[str, str, Path]] = []
     if not extracted_root.exists():
         return runs
     for region_dir in sorted(extracted_root.iterdir()):
         if not region_dir.is_dir():
             continue
         for run_dir in sorted(region_dir.iterdir()):
             if not run_dir.is_dir():
                 continue
             run_id = run_dir.name
             runs.append((region_dir.name, run_id, run_dir))
     return runs


def _read_csv(path: Path) -> pd.DataFrame:
     # 部分 CSV 会出现奇怪的空格/换行，pandas 更稳
     return pd.read_csv(path, engine="python")


def _to_bool_series(s: pd.Series) -> pd.Series:
     if s.dtype == bool:
         return s
     return s.astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
     if window <= 1:
         return values
     ser = pd.Series(values)
     return ser.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def _compute_velocity_from_positions(t_us: np.ndarray, xy: np.ndarray) -> np.ndarray:
     t_s = t_us.astype(np.float64) * 1e-6
     dt = np.diff(t_s)
     dt = np.where(dt <= 0, np.nan, dt)
     dxy = np.diff(xy, axis=0)
     v = dxy / dt[:, None]
     # pad to same length
     v = np.vstack([v, v[-1:]])
     v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
     return v


def _heading_from_velocity(vxy: np.ndarray) -> np.ndarray:
     # heading: 0=N, 90=E (与很多导航定义一致)。这里用 atan2(east, north)
     vx = vxy[:, 0]
     vy = vxy[:, 1]
     heading = np.degrees(np.arctan2(vx, vy))
     heading = (heading + 360.0) % 360.0
     return heading


def parse_one_run(region: str, run_id: str, run_dir: Path) -> Optional[Trajectory]:
     mip_dir = run_dir / "MicrostrainMIP"
     gps_path = mip_dir / "gps.csv"
     imu_path = mip_dir / "imu.csv"

     if not gps_path.exists():
         logger.warning(f"GPS CSV 不存在: {gps_path}")
         return None

     gps = _read_csv(gps_path)
     if gps.empty:
         logger.warning(f"GPS CSV 为空: {gps_path}")
         return None

     # 字段名来自 config
     ts_col = cfg.get("oord.gps.fields.timestamp", "timestamp")
     lat_col = cfg.get("oord.gps.fields.latitude", "latitude")
     lon_col = cfg.get("oord.gps.fields.longitude", "longitude")
     v_n_col = cfg.get("oord.gps.fields.velocity_north", "velocity_north")
     v_e_col = cfg.get("oord.gps.fields.velocity_east", "velocity_east")
     v_s_col = cfg.get("oord.gps.fields.velocity_speed", "velocity_speed")
     heading_col = cfg.get("oord.gps.fields.velocity_heading", "velocity_heading")

     required = [ts_col, lat_col, lon_col]
     for col in required:
         if col not in gps.columns:
             logger.error(f"GPS 缺少必要列 {col}: {gps_path}")
             return None

     # 有效性过滤（尽可能温和，避免误删）
     if "lat_and_long_valid" in gps.columns:
         gps = gps[_to_bool_series(gps["lat_and_long_valid"])].copy()
     if "gps_time_valid" in gps.columns:
         gps = gps[_to_bool_series(gps["gps_time_valid"])].copy()

     # 时间戳
     gps[ts_col] = pd.to_numeric(gps[ts_col], errors="coerce")
     gps = gps.dropna(subset=[ts_col])
     gps = gps.sort_values(ts_col).drop_duplicates(subset=[ts_col])
     t_us = gps[ts_col].astype(np.int64).to_numpy()

     # 坐标：优先使用提供的 UTM（更适合后续速度/距离计算）
     if "utm_easting" in gps.columns and "utm_northing" in gps.columns:
         x = pd.to_numeric(gps["utm_easting"], errors="coerce")
         y = pd.to_numeric(gps["utm_northing"], errors="coerce")
     else:
         # 如果没有 UTM，则退化为经纬度（不推荐）
         x = pd.to_numeric(gps[lon_col], errors="coerce")
         y = pd.to_numeric(gps[lat_col], errors="coerce")

     lat = pd.to_numeric(gps[lat_col], errors="coerce").to_numpy()
     lon = pd.to_numeric(gps[lon_col], errors="coerce").to_numpy()

     valid_xy = ~(x.isna() | y.isna())
     gps = gps.loc[valid_xy].copy()
     x = x.loc[valid_xy].to_numpy(dtype=np.float64)
     y = y.loc[valid_xy].to_numpy(dtype=np.float64)
     lat = lat[valid_xy.to_numpy()]
     lon = lon[valid_xy.to_numpy()]
     t_us = gps[ts_col].astype(np.int64).to_numpy()

     xy = np.stack([x, y], axis=1)

     # 速度：优先用 GPS 自带速度分量，否则用差分
     if v_e_col in gps.columns and v_n_col in gps.columns:
         vx = pd.to_numeric(gps[v_e_col], errors="coerce").to_numpy(dtype=np.float64)
         vy = pd.to_numeric(gps[v_n_col], errors="coerce").to_numpy(dtype=np.float64)
         vxy = np.stack([np.nan_to_num(vx), np.nan_to_num(vy)], axis=1)
     else:
         vxy = _compute_velocity_from_positions(t_us, xy)

     if v_s_col in gps.columns:
         speed = pd.to_numeric(gps[v_s_col], errors="coerce").to_numpy(dtype=np.float64)
         speed = np.nan_to_num(speed)
     else:
         speed = np.linalg.norm(vxy, axis=1)

     if heading_col in gps.columns:
         heading = pd.to_numeric(gps[heading_col], errors="coerce").to_numpy(dtype=np.float64)
         heading = np.nan_to_num(heading)
         heading = (heading + 360.0) % 360.0
     else:
         heading = _heading_from_velocity(vxy)

     # 过滤异常速度
     max_speed = float(cfg.get("oord.trajectory.max_speed", 30.0))
     keep = speed <= max_speed
     xy = xy[keep]
     vxy = vxy[keep]
     speed = speed[keep]
     heading = heading[keep]
     t_us = t_us[keep]
     lat = lat[keep]
     lon = lon[keep]

     # 平滑
     window = int(cfg.get("oord.trajectory.smooth_window", 5))
     if window > 1 and len(speed) >= 3:
         xy[:, 0] = _smooth_series(xy[:, 0], window)
         xy[:, 1] = _smooth_series(xy[:, 1], window)
         speed = _smooth_series(speed, window)
         heading = _smooth_series(heading, window)
         vxy = _compute_velocity_from_positions(t_us, xy)

     min_len = int(cfg.get("oord.trajectory.min_length", 100))
     if len(t_us) < min_len:
         logger.warning(f"轨迹过短，跳过: {region}/{run_id} (len={len(t_us)})")
         return None

     meta: Dict[str, Any] = {
         "gps_csv": str(gps_path),
         "imu_csv": str(imu_path) if imu_path.exists() else None,
         "has_imu": imu_path.exists(),
     }

     return Trajectory(
         region=region,
         run_id=run_id,
         timestamps_us=t_us,
         positions_xy=xy,
         velocities_xy=vxy,
         speed=speed,
         heading_deg=heading,
         latitude=lat,
         longitude=lon,
         meta=meta,
     )

def parse_oord_trajectories():

    extracted_root = Path(cfg.get("paths.raw_data.oord_extracted"))
    out_dir = get_path("paths.processed.trajectories")
    out_path = out_dir / "oord_trajectories.pkl"

    runs = _find_extracted_runs(extracted_root)
    if not runs:
        raise FileNotFoundError(f"未找到解压后的 OORD 数据: {extracted_root}")

    logger.info(f"发现 {len(runs)} 个 OORD 运行包")

    trajs: List[Trajectory] = []
    for region, run_id, run_dir in runs:
        try:
            t = parse_one_run(region, run_id, run_dir)
            if t is not None:
                trajs.append(t)
                logger.info(f"✅ 解析完成: {region}/{run_id} (len={len(t.timestamps_us)})")
        except Exception as e:
            logger.error(f"❌ 解析失败: {region}/{run_id} ({e})")

    payload = {
        "trajectories": trajs,
        "schema": {
            "timestamps_us": "int64[ N ]",
            "positions_xy": "float64[ N,2 ] (UTM easting,northing if available)",
            "velocities_xy": "float64[ N,2 ] (vx_east, vy_north)",
            "speed": "float64[ N ] (m/s)",
            "heading_deg": "float64[ N ] (0=N, 90=E)",
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    logger.info(f"保存完成: {out_path} (trajectories={len(trajs)})")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parse_oord_trajectories()
