# TerraTNT 配置系统完整指南

## 📋 概述

TerraTNT 采用**集中式配置管理**，所有参数统一存放在 `config/config.yaml` 中。修改一次配置，全局生效，避免重复修改代码。

## 🎯 核心设计理念

### 1. **单一配置源**
- 所有可配置参数集中在 `config.yaml`
- 避免硬编码，提高可维护性
- 支持运行时动态修改

### 2. **分层配置结构**
```yaml
project:          # 项目基本信息
paths:            # 所有路径配置
regions:          # 地理区域定义
gee:              # GEE 数据配置
oord:             # OORD 数据配置
environment:      # 环境特征配置
speed_predictor:  # 速度预测模型
trajectory_generation:  # 轨迹生成
terratnt:         # TerraTNT 模型
plotting:         # 绘图配置
gui:              # 界面配置
logging:          # 日志配置
performance:      # 性能配置
network:          # 网络配置
```

### 3. **全局常量定义**
- `config/constants.py` 定义不可变常量
- 包括物理常量、枚举类型、单位转换等
- 提供类型安全和代码提示

## 📂 配置文件结构

### config/config.yaml
主配置文件，包含所有可调参数。

**关键配置项：**

#### 路径配置
```yaml
paths:
  raw_data:
    gee: "data/raw/gee"
    oord: "data/oord"
  processed:
    root: "data/processed"
  models:
    root: "models/saved"
  outputs:
    figures: "outputs/figures"
    logs: "outputs/logs"
```

#### 数据处理配置
```yaml
gee:
  target_resolution: 30  # 统一分辨率（米）
  chunking:
    dem_splits: [4, 4]   # DEM 分块数
    lulc_splits: [8, 8]  # LULC 分块数

oord:
  gps:
    sampling_rate: 4     # GPS 采样率（Hz）
  trajectory:
    min_length: 100      # 最小轨迹长度
    max_speed: 30.0      # 最大速度（m/s）
```

#### 环境特征配置
```yaml
environment:
  num_channels: 18       # 特征通道数
  features:
    - name: "elevation"
      channels: 1
      normalization: "standardize"
    - name: "lulc_onehot"
      channels: 9
      normalization: "none"
```

#### 模型配置
```yaml
speed_predictor:
  model_type: "xgboost"
  xgboost:
    n_estimators: 500
    max_depth: 8
    learning_rate: 0.05

terratnt:
  data:
    history_length: 240
    prediction_length: 240
  architecture:
    env_encoder:
      backbone: "resnet18"
      output_dim: 256
  training:
    batch_size: 32
    learning_rate: 0.001
```

#### 绘图配置
```yaml
plotting:
  style: "seaborn-v0_8-darkgrid"
  font:
    family: "DejaVu Sans"
    size: 12
  figure:
    dpi: 300
    format: "png"
  colors:
    primary: "#2E86AB"
    secondary: "#A23B72"
```

### config/__init__.py
配置加载器，提供统一的配置访问接口。

**核心功能：**
- 单例模式，全局唯一配置实例
- 支持点号分隔的嵌套键访问
- 自动解析相对路径为绝对路径
- 运行时动态修改配置

### config/plot_config.py
绘图全局配置，统一管理所有可视化样式。

**核心功能：**
- 统一颜色方案
- 标准化图形尺寸
- 自动应用样式
- LULC 颜色映射

### config/constants.py
全局常量定义，包含不可变的系统常量。

**包含内容：**
- 地理坐标系统枚举
- LULC 分类常量
- 物理常量（地球半径、重力加速度等）
- 数据处理常量
- 单位转换常量

## 🔧 使用方法

### 1. 读取配置

```python
from config import cfg

# 基本读取
project_name = cfg.get('project.name')

# 嵌套读取（支持点号分隔）
batch_size = cfg.get('terratnt.training.batch_size')

# 带默认值
unknown = cfg.get('unknown.key', default_value)

# 字典式访问
value = cfg['paths.raw_data.gee']
```

### 2. 路径管理

```python
from config import get_path

# 自动创建目录并返回 Path 对象
output_dir = get_path('paths.outputs.figures')
model_dir = get_path('paths.models.root')
```

### 3. 运行时修改

```python
from config import cfg

# 临时修改（不保存到文件）
cfg.set('terratnt.training.batch_size', 64)

# 保存到配置文件
cfg.save()
```

### 4. 使用常量

```python
from config.constants import *

# 使用枚举
coord_sys = CoordinateSystem.WGS84.value  # "EPSG:4326"

# 使用 LULC 常量
forest_cost = LULC_TRAVERSABILITY[LULCClass.TREE_COVER.value]

# 使用物理常量
max_speed = MAX_VEHICLE_SPEED  # 30.0 m/s
```

### 5. 统一绘图

```python
from config.plot_config import create_figure, save_figure, style_axis, plot_cfg

# 创建标准化图形
fig, ax = create_figure(size='large')  # 'default', 'large', 'small'

# 使用预定义颜色
ax.plot(x, y, color=plot_cfg.PRIMARY, linewidth=2)
ax.plot(x2, y2, color=plot_cfg.COLOR_PREDICTED, linestyle='--')

# 应用统一样式
style_axis(ax, 
           title='我的图表',
           xlabel='X 轴',
           ylabel='Y 轴',
           grid=True,
           legend=True)

# 保存到标准路径
save_figure(fig, 'my_plot', subdir='analysis')
# 自动保存到: outputs/figures/analysis/my_plot.png
```

### 6. LULC 颜色映射

```python
from config.plot_config import plot_cfg
import matplotlib.pyplot as plt

# 获取 LULC 颜色映射
cmap, classes = plot_cfg.get_lulc_cmap()

# 显示 LULC 数据
im = ax.imshow(lulc_data, cmap=cmap)
cbar = plt.colorbar(im, ax=ax, ticks=classes)

# 使用单个 LULC 颜色
forest_color = plot_cfg.LULC_COLORS[10]  # 森林颜色
```

## 📝 配置修改示例

### 示例 1: 修改数据路径

```yaml
# 在 config.yaml 中修改
paths:
  raw_data:
    gee: "/new/path/to/gee"
    oord: "/new/path/to/oord"
```

所有使用 `cfg.get('paths.raw_data.gee')` 的代码自动生效。

### 示例 2: 调整模型超参数

```yaml
# 在 config.yaml 中修改
terratnt:
  training:
    batch_size: 64        # 从 32 改为 64
    learning_rate: 0.0005 # 从 0.001 改为 0.0005
    num_epochs: 150       # 从 100 改为 150
```

训练脚本自动使用新参数。

### 示例 3: 更改绘图样式

```yaml
# 在 config.yaml 中修改
plotting:
  colors:
    primary: "#FF5733"    # 更改主色调
  figure:
    dpi: 600              # 提高分辨率
    format: "pdf"         # 改为 PDF 格式
```

所有图表自动应用新样式。

### 示例 4: 添加新区域

```yaml
# 在 config.yaml 中添加
regions:
  new_region:
    name: "New Region"
    bounds:
      lon_min: 10.0
      lon_max: 12.0
      lat_min: 50.0
      lat_max: 52.0
    utm_zone: "32N"
    epsg: 32632
```

代码中可直接访问：`cfg.get('regions.new_region')`

## 🎨 绘图配置详解

### 颜色方案

**预定义颜色：**
- `PRIMARY`: 主色调（默认 #2E86AB）
- `SECONDARY`: 次要色（默认 #A23B72）
- `ACCENT`: 强调色（默认 #F18F01）
- `SUCCESS`: 成功色（默认 #06A77D）
- `WARNING`: 警告色（默认 #F77F00）
- `ERROR`: 错误色（默认 #D62828）

**轨迹颜色：**
- `COLOR_REAL`: 真实轨迹（蓝色）
- `COLOR_PREDICTED`: 预测轨迹（橙色）
- `COLOR_SYNTHETIC`: 合成轨迹（紫色）

**地形颜色：**
- `COLOR_WATER`: 水体（蓝色）
- `COLOR_FOREST`: 森林（深绿）
- `COLOR_GRASSLAND`: 草地（浅绿）
- `COLOR_URBAN`: 城市（灰色）

### 图形尺寸

```python
# 默认尺寸（10x6 英寸）
fig, ax = create_figure(size='default')

# 大图（14x8 英寸）
fig, ax = create_figure(size='large')

# 小图（6x4 英寸）
fig, ax = create_figure(size='small')

# 自定义尺寸
fig, ax = create_figure(size=(12, 7))
```

### 样式模板

```python
# 方法 1: 使用 style_axis 函数
style_axis(ax, 
           title='标题',
           xlabel='X轴',
           ylabel='Y轴',
           grid=True,
           legend=True)

# 方法 2: 手动设置（使用配置的颜色）
ax.set_title('标题', fontweight='bold', color=plot_cfg.PRIMARY)
ax.grid(True, alpha=0.3, linestyle='--')
```

## 🔍 最佳实践

### 1. 避免硬编码

❌ **不好的做法：**
```python
batch_size = 32
learning_rate = 0.001
output_dir = "outputs/figures"
```

✅ **好的做法：**
```python
from config import cfg, get_path

batch_size = cfg.get('terratnt.training.batch_size')
learning_rate = cfg.get('terratnt.training.learning_rate')
output_dir = get_path('paths.outputs.figures')
```

### 2. 统一绘图接口

❌ **不好的做法：**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, color='blue', linewidth=2)
ax.set_title('My Plot')
plt.savefig('my_plot.png', dpi=300)
```

✅ **好的做法：**
```python
from config.plot_config import create_figure, save_figure, style_axis, plot_cfg

fig, ax = create_figure(size='default')
ax.plot(x, y, color=plot_cfg.PRIMARY, linewidth=2)
style_axis(ax, title='My Plot')
save_figure(fig, 'my_plot', subdir='analysis')
```

### 3. 使用常量而非魔法数字

❌ **不好的做法：**
```python
if speed > 30.0:  # 什么是 30.0？
    speed = 30.0

gps_rate = 4  # 什么是 4？
```

✅ **好的做法：**
```python
from config.constants import MAX_VEHICLE_SPEED, GPS_SAMPLING_RATE

if speed > MAX_VEHICLE_SPEED:
    speed = MAX_VEHICLE_SPEED

gps_rate = GPS_SAMPLING_RATE
```

### 4. 路径管理

❌ **不好的做法：**
```python
import os
output_dir = "outputs/figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

✅ **好的做法：**
```python
from config import get_path

output_dir = get_path('paths.outputs.figures')
# 自动创建目录，返回 Path 对象
```

## 🚀 快速参考

### 常用配置项

| 配置项 | 路径 | 默认值 |
|--------|------|--------|
| 批次大小 | `terratnt.training.batch_size` | 32 |
| 学习率 | `terratnt.training.learning_rate` | 0.001 |
| 历史长度 | `terratnt.data.history_length` | 240 |
| 预测长度 | `terratnt.data.prediction_length` | 240 |
| 环境通道数 | `environment.num_channels` | 18 |
| DPI | `plotting.figure.dpi` | 300 |
| 图形格式 | `plotting.figure.format` | "png" |

### 常用函数

| 函数 | 用途 |
|------|------|
| `cfg.get(key, default)` | 读取配置 |
| `cfg.set(key, value)` | 设置配置 |
| `get_path(key)` | 获取路径并创建目录 |
| `create_figure(size)` | 创建标准化图形 |
| `save_figure(fig, name, subdir)` | 保存图形 |
| `style_axis(ax, **kwargs)` | 设置坐标轴样式 |

## 📚 相关文档

- [README_SYSTEM.md](../README_SYSTEM.md) - 系统总体说明
- [config.yaml](../config/config.yaml) - 主配置文件
- [constants.py](../config/constants.py) - 常量定义
- [plot_config.py](../config/plot_config.py) - 绘图配置

## 🐛 故障排除

### 问题 1: 配置读取失败
```python
# 检查配置是否加载
from config import cfg
print(cfg.config)  # 打印完整配置
```

### 问题 2: 路径不存在
```python
# 使用 get_path 自动创建
from config import get_path
path = get_path('paths.outputs.figures')
```

### 问题 3: 绘图样式不生效
```python
# 重新加载绘图配置
from config.plot_config import PlotConfig
plot_cfg = PlotConfig()
```
