# TerraTNT 改进版UI使用说明

## 🎯 改进内容

### 1. 修复的问题
- ✅ **字体缺失**：自动检测并设置系统中文字体（Qt + Matplotlib同步）
- ✅ **控件堆叠**：所有Tab页面包裹QScrollArea，窗口缩小不会重叠
- ✅ **虚假图片**：替换为真实数据驱动的可视化
- ✅ **组件设置**：正确的尺寸策略和布局管理

### 2. 新增功能

#### **真实DEM/LULC可视化**
在"数据加载"页面新增两个按钮：
- **显示DEM**：加载并显示真实DEM地形数据（terrain配色）
- **显示LULC**：加载并显示真实土地利用分类数据

数据来源：
- 苏格兰高地：`/home/zmc/文档/programwork/data/processed/utm_grid/scottish_highlands/`
- 波西米亚森林：`/home/zmc/文档/programwork/data/processed/utm_grid/bohemian_forest/`

#### **真实轨迹可视化**
在"轨迹预测"页面：
- 点击"浏览..."选择任意`.pkl`轨迹文件
- 右侧画布自动显示真实轨迹路径（起点绿色、终点蓝色、轨迹红色）

## 📖 使用步骤

### 1. 启动UI
```bash
cd /home/zmc/文档/programwork
python ui/terratnt_professional_ui.py
```

### 2. 查看真实DEM地形
1. 切换到"数据加载"标签页
2. 在"选择区域"下拉框选择区域（苏格兰高地/波西米亚森林）
3. 点击"显示DEM"按钮
4. 右侧画布显示真实DEM地形图（海拔数据，terrain配色）

### 3. 查看真实LULC土地利用
1. 在"数据加载"标签页
2. 选择区域
3. 点击"显示LULC"按钮
4. 右侧画布显示真实LULC分类图（不同土地类型用不同颜色）

### 4. 查看真实轨迹
1. 切换到"轨迹预测"标签页
2. 点击"浏览..."按钮
3. 选择任意轨迹文件（例如：`data/processed/synthetic_trajectories/scottish_highlands/*.pkl`）
4. 右侧画布显示真实轨迹路径

### 5. 加载轨迹数据集
1. 在"数据加载"标签页
2. 修改"数据路径"（或使用默认路径）
3. 点击"加载数据"按钮
4. 右侧画布显示数据集文件大小分布

## 🔧 依赖要求

UI需要GDAL库来读取DEM/LULC的`.tif`文件：

```bash
conda activate torch-sm120
conda install -c conda-forge gdal
```

如果没有安装GDAL，点击"显示DEM/LULC"时会提示安装。

## 📊 数据位置

### DEM数据
- 苏格兰高地：`data/processed/utm_grid/scottish_highlands/dem_utm.tif`
- 波西米亚森林：`data/processed/utm_grid/bohemian_forest/dem_utm.tif`

### LULC数据
- 苏格兰高地：`data/processed/utm_grid/scottish_highlands/lulc_utm.tif`
- 波西米亚森林：`data/processed/utm_grid/bohemian_forest/lulc_utm.tif`

### 轨迹数据
- 苏格兰高地：`data/processed/synthetic_trajectories/scottish_highlands/*.pkl`
- 波西米亚森林：`data/processed/synthetic_trajectories/bohemian_forest/*.pkl`

## 🎨 UI特点

1. **全中文界面**：所有文本统一使用中文，无emoji
2. **标准桌面软件结构**：菜单栏 + 工具栏 + 状态栏
3. **响应式布局**：左右分割可调整，左侧有滚动条
4. **真实数据驱动**：所有可视化都基于真实数据文件
5. **信号联动**：左侧操作自动更新右侧可视化

## 📝 文件位置

- UI程序：`/home/zmc/文档/programwork/ui/terratnt_professional_ui.py`
- 使用说明：`/home/zmc/文档/programwork/docs/UI使用说明.md`
