# 轨迹数据集 PKL 文件格式说明

## 概述

每条轨迹保存为一个 `.pkl` 文件，使用 Python pickle 格式序列化为字典对象。

## 字段说明

### 基础信息

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `trajectory_id` | `int` | 轨迹唯一标识符 |
| `region` | `str` | 区域名称（如 `"bohemian_forest"`） |
| `vehicle_type` | `str` | 车辆类型（`"type1"`, `"type2"`, `"type3"`, `"type4"`） |
| `intent` | `str` | 战术意图（`"intent1"`, `"intent2"`, `"intent3"`） |

### 车辆参数

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `vehicle_params` | `dict` | 车辆参数字典，包含以下子字段： |
| `vehicle_params.max_speed_mps` | `float` | 最大速度（米/秒） |
| `vehicle_params.max_slope_deg` | `float` | 最大可通过坡度（度） |
| `vehicle_params.max_acceleration` | `float` | 最大加速度（米/秒²） |
| `vehicle_params.min_turning_radius` | `float` | 最小转弯半径（米） |

**车辆类型参数表：**

| 车辆类型 | max_speed (m/s) | max_slope (°) | max_accel (m/s²) | min_radius (m) |
|----------|-----------------|---------------|------------------|----------------|
| type1    | 18              | 30            | 2.0              | 8              |
| type2    | 22              | 25            | 2.5              | 6              |
| type3    | 25              | 20            | 3.0              | 5              |
| type4    | 28              | 15            | 3.5              | 10             |

### 轨迹数据

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `path_utm` | `np.ndarray` | `(N, 2)` | UTM 坐标路径，单位：米，列为 `[x, y]` |
| `timestamps_s` | `np.ndarray` | `(N,)` | 时间戳，单位：秒，从 0 开始 |
| `speeds_mps` | `np.ndarray` | `(N,)` | 速度序列，单位：米/秒 |
| `accelerations` | `np.ndarray` | `(N,)` | 加速度序列，单位：米/秒² |
| `curvatures` | `np.ndarray` | `(N,)` | 曲率序列，单位：1/米 |

**注意：**
- 所有数组长度 `N` 相同，对应轨迹的采样点数
- 时间间隔固定为 10 秒（`timestamps_s[i+1] - timestamps_s[i] = 10.0`）
- 路径坐标系为 UTM Zone 33N (EPSG:32633)

### 起终点信息

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `start_utm` | `tuple` | 起点 UTM 坐标 `(x, y)`，单位：米 |
| `goal_utm` | `tuple` | 终点 UTM 坐标 `(x, y)`，单位：米 |
| `straight_line_distance_m` | `float` | 起终点直线距离，单位：米 |

### 统计信息

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `total_length_m` | `float` | 轨迹总长度，单位：米 |
| `total_duration_s` | `float` | 轨迹总时长，单位：秒 |
| `num_points` | `int` | 轨迹点数（10秒采样） |
| `num_samples` | `int` | 可用训练样本数（通常为 `num_points - 1`） |
| `mean_speed_mps` | `float` | 平均速度，单位：米/秒 |
| `detour_ratio` | `float` | 迂回系数（`total_length_m / straight_line_distance_m`） |

### 训练样本

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `samples` | `list[dict]` | 长度 `num_samples` | 训练样本列表，每个样本包含： |

**每个样本字典包含：**

| 子字段 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `sample_id` | `int` | - | 样本索引（0 到 `num_samples-1`） |
| `current_position` | `np.ndarray` | `(2,)` | 当前位置 UTM 坐标 `[x, y]` |
| `current_speed` | `float` | - | 当前速度（米/秒） |
| `history_positions` | `np.ndarray` | `(H, 2)` | 历史位置序列（最多 6 个点） |
| `future_positions` | `np.ndarray` | `(F, 2)` | 未来位置序列（最多 6 个点） |
| `goal_position` | `np.ndarray` | `(2,)` | 目标位置 UTM 坐标 |
| `env_map` | `np.ndarray` | `(18, 128, 128)` | 环境地图（18 通道） |
| `env_features` | `np.ndarray` | `(26,)` | 环境特征向量 |

**环境地图 18 通道说明：**

| 通道索引 | 内容 | 说明 |
|----------|------|------|
| 0 | DEM | 高程（归一化） |
| 1 | Slope | 坡度（归一化） |
| 2 | Aspect (sin) | 坡向正弦分量 |
| 3 | Aspect (cos) | 坡向余弦分量 |
| 4-13 | LULC one-hot | 土地利用类型（10 类：10, 20, 30, 40, 50, 60, 80, 90, 100, 255） |
| 14 | Tree cover | 树木覆盖（LULC=10） |
| 15 | Road | 道路层（从 `road_utm.tif` 读取，1=道路，0=非道路） |
| 16 | History heatmap | 历史轨迹热力图 |
| 17 | Goal map | 目标位置标记 |

**环境特征 26 维说明：**

| 维度 | 内容 | 说明 |
|------|------|------|
| 0 | DEM | 当前位置高程 |
| 1 | Slope | 当前位置坡度 |
| 2-3 | Aspect (sin/cos) | 坡向三角函数分量 |
| 4-14 | LULC one-hot | 土地利用类型（11 类） |
| 15 | Tree cover | 树木覆盖 |
| 16 | Effective slope | 有效坡度（考虑运动方向） |
| 17 | Curvature | 当前曲率 |
| 18 | Past curvature | 过去 10m 平均曲率 |
| 19 | Future curvature | 未来 10m 最大曲率 |
| 20-22 | Velocity (vx, vy, v_norm) | 速度向量和模 |
| 23-24 | Goal direction (sin/cos) | 目标方向角 |
| 25 | On road | 是否在道路上（从 `road_utm.tif` 判断） |

## 使用示例

### 读取轨迹

```python
import pickle
import numpy as np

# 读取轨迹文件
with open("trajectory_000.pkl", "rb") as f:
    traj = pickle.load(f)

# 访问基础信息
print(f"轨迹 ID: {traj['trajectory_id']}")
print(f"车辆类型: {traj['vehicle_type']}")
print(f"意图: {traj['intent']}")

# 访问轨迹数据
path = traj['path_utm']  # (N, 2)
speeds = traj['speeds_mps']  # (N,)
timestamps = traj['timestamps_s']  # (N,)

# 访问统计信息
print(f"轨迹长度: {traj['total_length_m']/1000:.2f} km")
print(f"平均速度: {traj['mean_speed_mps']*3.6:.2f} km/h")
print(f"总时长: {traj['total_duration_s']/60:.2f} 分钟")

# 访问训练样本
samples = traj['samples']
for sample in samples:
    env_map = sample['env_map']  # (18, 128, 128)
    env_features = sample['env_features']  # (26,)
    # ... 训练模型
```

### 计算道路利用率

```python
import rasterio

# 读取道路栅格
with rasterio.open("data/processed/utm_grid/bohemian_forest/road_utm.tif") as src:
    road_raster = src.read(1)
    transform = src.transform

# 计算道路利用率
path = traj['path_utm']
xs, ys = path[:, 0], path[:, 1]
rows, cols = rasterio.transform.rowcol(transform, xs, ys)
rows, cols = np.array(rows), np.array(cols)

valid = (rows >= 0) & (rows < road_raster.shape[0]) & \
        (cols >= 0) & (cols < road_raster.shape[1])
road_vals = road_raster[rows[valid], cols[valid]]
road_utilization = np.mean(road_vals > 0)

print(f"道路利用率: {road_utilization*100:.2f}%")
```

## 数据集统计

使用 `scripts/traj_stats_roadpref.py` 脚本可以批量统计轨迹数据集：

```bash
conda run -n torch-sm120 python scripts/traj_stats_roadpref.py \
    --traj-dir data/processed/complete_dataset_10s/bohemian_forest_roadpref_quick \
    --road-tif data/processed/utm_grid/bohemian_forest/road_utm.tif
```

输出示例：

```
=== Road utilization (road_utm.tif) by vehicle_type ===
vehicle_type    n       road_mean       road_p10        road_p50        road_p90        speed_mean_kmh  length_km       duration_min
type1           6       0.0405          0.0094          0.0382          0.0741          59.36           109.05          112.44
type2           7       0.0779          0.0000          0.0185          0.2127          64.51           113.53          107.68
type3           8       0.0528          0.0221          0.0445          0.1026          69.86           110.58          97.09
type4           6       0.1183          0.0143          0.0705          0.2701          75.24           116.32          94.91
```

## 注意事项

1. **坐标系统**：所有 UTM 坐标使用 EPSG:32633 (UTM Zone 33N)
2. **时间间隔**：固定 10 秒采样间隔
3. **道路定义**：`road_utm.tif` 中值 > 0 的像素为道路
4. **LULC 代码**：
   - 10: 农田
   - 20: 森林
   - 30: 草地
   - 40: 灌木
   - 50: 湿地
   - 60: 水体
   - 80: 人工表面
   - 90: 裸地
   - 100: 冰雪
   - 255: 未知/缺失
5. **速度单位转换**：
   - 米/秒 → 公里/小时：乘以 3.6
   - 公里/小时 → 米/秒：除以 3.6
