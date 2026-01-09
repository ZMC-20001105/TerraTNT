# 面向遥感任务规划的地面目标位置预测算法

基于多星协同对地观测系统的地面目标轨迹预测与位置预测研究项目。

## 项目概述

本项目实现了一个完整的地面目标位置预测系统，主要解决多卫星协同观测中的观测空窗期目标位置预测问题。系统采用深度学习方法，结合环境约束建模，实现60分钟时域的长期轨迹预测。

### 核心技术

1. **环境约束建模**：融合DEM、LULC、OSM道路数据构建精细化环境代价地图
2. **轨迹数据生成**：基于分层A*路径规划和XGBoost速度预测生成合成轨迹数据集
3. **TerraTNT预测模型**：目标驱动的深度学习轨迹预测框架
4. **多区域验证**：在4个不同地形特征区域进行跨区域泛化测试

### 研究区域

- **波西米亚森林**：捷克-德国-奥地利边境，低山丘陵地形
- **顿巴斯**：乌克兰东部，平原丘陵地形  
- **喀尔巴阡山**：罗马尼亚中部，中高山地地形
- **苏格兰高地**：英国苏格兰北部，高原山地地形

## 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone <repository_url>
cd programwork

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. Google Earth Engine 设置

```bash
# 设置GEE环境
python scripts/setup_gee.py
```

按照提示完成Google Earth Engine认证。

### 3. 数据下载

```bash
# 下载遥感数据
python scripts/gee_data_downloader.py
```

这将下载所有研究区域的DEM、LULC等遥感数据到Google Drive。

### 4. 数据预处理

```bash
# 预处理环境数据
python src/data_processing/preprocess_all.py
```

## 项目结构

```
programwork/
├── src/                          # 源代码
│   ├── config/                   # 配置文件
│   ├── data_processing/          # 数据处理模块
│   │   ├── dem_processor.py      # DEM数据处理
│   │   ├── lulc_processor.py     # LULC数据处理
│   │   └── osm_processor.py      # OSM数据处理
│   ├── environment/              # 环境建模
│   ├── path_planning/            # 路径规划
│   ├── trajectory_generation/    # 轨迹生成
│   ├── models/                   # 深度学习模型
│   ├── training/                 # 训练框架
│   ├── evaluation/               # 评估工具
│   ├── visualization/            # 可视化工具
│   └── utils/                    # 工具函数
├── scripts/                      # 脚本文件
│   ├── setup_gee.py             # GEE环境设置
│   └── gee_data_downloader.py   # 数据下载脚本
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后数据
│   └── osm/                     # OSM数据
├── notebooks/                   # Jupyter notebooks
├── configs/                     # 配置文件
└── requirements.txt             # 依赖列表
```

## 数据获取指南

### Google Earth Engine数据

项目使用Google Earth Engine获取以下数据：

1. **SRTM DEM**：30米分辨率数字高程模型
2. **ESA WorldCover**：10米分辨率土地覆盖数据
3. **地形衍生数据**：坡度、坡向等

### OSM道路数据

通过OSMnx API自动获取道路网络数据。

### OORD轨迹数据

从Oxford Offroad Radar Dataset获取真实车辆轨迹数据用于速度建模。

## 使用说明

### 1. 数据下载与预处理

```python
# 使用GEE下载器
from scripts.gee_data_downloader import GEEDataDownloader

downloader = GEEDataDownloader()
# 下载所有区域数据
tasks = downloader.download_all_regions()
```

### 2. 环境建模

```python
from src.environment.environment_model import EnvironmentModel

# 创建环境模型
env_model = EnvironmentModel()
cost_map = env_model.create_cost_map(region_name='bohemian_forest')
```

### 3. 轨迹生成

```python
from src.trajectory_generation.trajectory_generator import TrajectoryGenerator

# 生成轨迹数据集
generator = TrajectoryGenerator()
trajectories = generator.generate_trajectories(
    region='bohemian_forest',
    num_trajectories=10000,
    vehicle_type='light_vehicle',
    tactical_intent='standard'
)
```

### 4. 模型训练

```python
from src.models.terratnt import TerraTNT
from src.training.trainer import Trainer

# 创建和训练模型
model = TerraTNT()
trainer = Trainer(model)
trainer.train(train_dataset, val_dataset)
```

## 配置说明

主要配置文件位于 `src/config/config.yaml`，包含：

- 数据路径配置
- 坐标系统设置
- 车辆类型参数
- 战术意图配置
- 模型超参数
- 训练配置

## 评估指标

- **ADE (Average Displacement Error)**：平均位移误差
- **FDE (Final Displacement Error)**：最终位移误差
- **EP@K**：终点预测准确率

## 实验结果

在已知目标场景下，TerraTNT模型达到：
- ADE: 1.23 km
- FDE: 1.65 km
- 相比基线模型提升8.9%-74.5%

## 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目维护者：Research Team
- 邮箱：[your-email@example.com]

## 致谢

- Google Earth Engine团队提供的遥感数据平台
- OpenStreetMap社区提供的道路网络数据
- Oxford Robotics Institute提供的OORD数据集
