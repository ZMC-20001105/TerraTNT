# TerraTNT 项目文件夹规范

## 目录结构

```
programwork/
├── config/                  # 配置文件
│   ├── config.yaml          # 全局配置（区域、路径、超参数）
│   └── plot_config.py       # 绘图配置
│
├── data/                    # 数据（不入git）
│   ├── raw/                 # 原始下载数据（GEE导出的tif）
│   ├── processed/
│   │   ├── utm_grid/        # 环境栅格（每区域一个子目录）
│   │   │   ├── bohemian_forest/   # dem_utm.tif, lulc_utm.tif, ...
│   │   │   ├── scottish_highlands/
│   │   │   └── <new_region>/
│   │   ├── final_dataset_v1/      # 轨迹数据集（pkl文件）
│   │   │   └── bohemian_forest/
│   │   └── fas_splits/            # FAS阶段划分
│   │       └── bohemian_forest/
│   └── osm/                 # OSM道路数据（pbf文件）
│
├── models/                  # 模型定义（入git）
│   ├── terratnt.py          # TerraTNT核心模型
│   ├── baseline_models.py   # 基线模型
│   └── baselines/           # PECNet, YNet等
│
├── scripts/                 # 独立运行脚本（入git）
│   ├── train_*.py           # 训练脚本
│   ├── evaluate_*.py        # 评估脚本
│   ├── build_road_*.py      # 道路数据处理
│   ├── download_*.py        # 数据下载
│   ├── generate_*.py        # 数据生成
│   ├── process_*.py         # 数据处理
│   ├── visualize_*.py       # 可视化脚本
│   └── draw_*.py            # 绘图脚本
│
├── training/                # 训练器（入git）
├── utils/                   # 工具模块（入git）
├── src/                     # 数据处理模块（入git）
├── visualization/           # PyQt6 UI系统（入git，主入口）
│
├── runs/                    # 模型权重+训练日志（不入git）
│   └── <experiment_name>/   # 每个实验一个目录
│       ├── best_model.pth
│       ├── config.json      # 实验配置快照
│       └── training.log
│
├── outputs/                 # 所有输出结果（不入git）
│   ├── evaluation/          # 评估结果（JSON + 图表）
│   │   └── <eval_run_name>/
│   ├── dataset_experiments/ # 数据集生成实验
│   │   └── <experiment_name>/
│   ├── figures/             # 论文/报告用图
│   ├── training_curves/     # 训练曲线图
│   └── logs/                # 运行日志
│
├── docs/                    # 文档（入git）
├── cache/                   # 运行时缓存（不入git）
├── _trash/                  # 待删除（不入git）
│
├── .gitignore
├── README.md
├── requirements.txt
└── environment.yml
```

## 命名规范

### 实验命名
- 训练实验: `runs/<model_name>_<variant>/` 如 `runs/incremental_models_v7/`
- 评估实验: `outputs/evaluation/<eval_name>/` 如 `outputs/evaluation/phase_diagnostic_v5/`
- 数据集实验: `outputs/dataset_experiments/<exp_name>/` 如 `outputs/dataset_experiments/D1_optimal_combo/`

### 禁止事项
- ❌ 不在根目录放任何临时文件（脚本、日志、图片、PID）
- ❌ 不在根目录创建 `vis_*`, `viz_*`, `runs_*` 等临时目录
- ❌ 不把大文件（>5MB）提交到git
- ❌ 不把安装包、论文docx放在项目目录

### 新实验流程
1. 训练 → 结果自动保存到 `runs/<name>/`
2. 评估 → 结果自动保存到 `outputs/evaluation/<name>/`
3. 可视化 → 图表保存到 `outputs/figures/` 或对应实验目录
