# TerraTNT 多星协同观测任务规划系统

## 📋 系统概述

基于深度学习的地面目标轨迹预测系统，用于多星协同观测任务规划。

**核心功能：**
- 多源地理数据管理与处理
- 越野轨迹分析与合成生成
- TerraTNT 模型训练与预测
- 卫星观测任务智能规划
- 可视化分析与结果导出

## 🏗️ 系统架构

```
TerraTNT/
├── config/                 # 配置系统
│   ├── config.yaml        # 主配置文件（所有参数集中管理）
│   ├── __init__.py        # 配置加载器
│   ├── plot_config.py     # 绘图全局配置
│   └── constants.py       # 全局常量定义
│
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   │   ├── gee/          # GEE 遥感数据
│   │   └── oord/         # OORD 轨迹数据
│   ├── processed/         # 处理后数据
│   └── oord_extracted/    # 解压后的 OORD 数据
│
├── gui/                    # Qt 图形界面
│   ├── main_window.py     # 主窗口
│   └── widgets/           # 界面组件
│       ├── data_manager.py
│       ├── map_viewer.py
│       ├── trajectory_analyzer.py
│       ├── model_trainer.py
│       ├── task_planner.py
│       └── result_exporter.py
│
├── models/                 # 模型定义
│   ├── speed_predictor/   # 速度预测模型
│   ├── terratnt/          # TerraTNT 模型
│   └── saved/             # 保存的模型
│
├── utils/                  # 工具模块
│   ├── data_processing/   # 数据处理
│   ├── trajectory/        # 轨迹处理
│   └── visualization/     # 可视化
│
├── scripts/                # 脚本工具
│   ├── gee_chunked_download.py
│   ├── download_oord_gps.py
│   └── install_dependencies.sh
│
├── outputs/                # 输出目录
│   ├── figures/           # 图表
│   ├── logs/              # 日志
│   └── results/           # 结果
│
├── main.py                 # 主程序入口
└── requirements.txt        # 依赖列表
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 激活 conda 环境
conda activate trajectory-prediction

# 安装依赖
bash scripts/install_dependencies.sh

# 或手动安装
pip install -r requirements.txt
```

### 2. 配置修改

**所有配置集中在 `config/config.yaml` 中**，修改一次即可全局生效：

```yaml
# 示例：修改数据路径
paths:
  raw_data:
    gee: "data/raw/gee"
    oord: "data/oord"

# 示例：修改模型参数
terratnt:
  training:
    batch_size: 32
    learning_rate: 0.001

# 示例：修改绘图样式
plotting:
  colors:
    primary: "#2E86AB"
  figure:
    dpi: 300
```

### 3. 启动系统

```bash
# 启动 Qt 图形界面
python main.py
```

## 📊 数据处理流程

### 阶段 1: 数据预处理

```python
# 1. 合并 GEE 分块数据
from utils.data_processing import merge_gee_chunks
merge_gee_chunks(region='scottish_highlands')

# 2. 提取环境特征
from utils.data_processing import extract_environment_features
env_features = extract_environment_features()

# 3. 解析 OORD 轨迹
from utils.trajectory import parse_oord_trajectories
trajectories = parse_oord_trajectories()
```

### 阶段 2: 模型训练

```python
# 1. 训练速度预测模型
from models.speed_predictor import train_speed_model
speed_model = train_speed_model()

# 2. 生成合成轨迹
from utils.trajectory import generate_synthetic_trajectories
synthetic_trajs = generate_synthetic_trajectories(speed_model)

# 3. 训练 TerraTNT
from models.terratnt import train_terratnt
terratnt_model = train_terratnt()
```

### 阶段 3: 预测与评估

```python
# 预测目标轨迹
from models.terratnt import predict_trajectory
prediction = predict_trajectory(history, environment)

# 评估模型性能
from utils.evaluation import evaluate_model
metrics = evaluate_model(predictions, ground_truth)
```

## 🎨 绘图配置

**所有绘图使用统一配置**，避免重复设置：

```python
from config.plot_config import create_figure, save_figure, style_axis

# 创建标准化图形
fig, ax = create_figure(size='large')

# 绘制内容
ax.plot(x, y, color=plot_cfg.PRIMARY)

# 应用统一样式
style_axis(ax, title='标题', xlabel='X轴', ylabel='Y轴', grid=True)

# 保存到标准路径
save_figure(fig, 'my_plot', subdir='trajectory_analysis')
```

**颜色使用：**
```python
from config.plot_config import plot_cfg

# 使用预定义颜色
ax.plot(real_traj, color=plot_cfg.COLOR_REAL, label='真实轨迹')
ax.plot(pred_traj, color=plot_cfg.COLOR_PREDICTED, label='预测轨迹')

# LULC 颜色映射
cmap, classes = plot_cfg.get_lulc_cmap()
```

## ⚙️ 配置系统使用

### 读取配置

```python
from config import cfg

# 获取配置项（支持点号分隔）
batch_size = cfg.get('terratnt.training.batch_size')
gee_path = cfg.get('paths.raw_data.gee')

# 获取路径并自动创建目录
from config import get_path
output_dir = get_path('paths.outputs.figures')
```

### 运行时修改配置

```python
# 临时修改（不保存到文件）
cfg.set('terratnt.training.batch_size', 64)

# 保存配置
cfg.save()
```

## 🖥️ GUI 使用指南

### 主界面布局

- **左侧**：地图视图（支持多图层切换）
- **右侧**：功能标签页
  - 📊 数据管理：加载、合并、预处理数据
  - 📈 轨迹分析：可视化和统计分析
  - 🧠 模型训练：训练和评估模型
  - 🛰️ 任务规划：卫星观测规划
  - 💾 结果导出：导出预测结果

### 快捷键

- `Ctrl+O`: 打开项目
- `Ctrl+S`: 保存项目
- `Ctrl+Q`: 退出
- `F11`: 全屏模式

## 📝 日志系统

日志自动保存到 `outputs/logs/terratnt.log`：

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

## 🔧 常见问题

### Q1: 如何修改数据路径？
**A**: 编辑 `config/config.yaml` 中的 `paths` 部分。

### Q2: 如何调整模型超参数？
**A**: 编辑 `config/config.yaml` 中的 `speed_predictor` 或 `terratnt` 部分。

### Q3: 如何更改绘图样式？
**A**: 编辑 `config/config.yaml` 中的 `plotting` 部分。

### Q4: GUI 启动失败？
**A**: 确保已安装 PyQt6：`pip install PyQt6`

### Q5: 如何添加新的数据区域？
**A**: 在 `config/config.yaml` 的 `regions` 部分添加新区域配置。

## 📚 开发指南

### 添加新功能模块

1. 在相应目录创建模块文件
2. 在 `config/config.yaml` 添加配置项
3. 使用 `cfg.get()` 读取配置
4. 使用统一的绘图和日志接口

### 代码规范

- 使用配置系统，避免硬编码
- 使用全局绘图配置，保持样式一致
- 添加详细的日志记录
- 编写单元测试

## 📄 许可证

本项目用于学术研究。

## 👥 贡献者

- 项目负责人：[您的名字]
- 开发团队：[团队成员]

## 📧 联系方式

如有问题，请联系：[您的邮箱]
