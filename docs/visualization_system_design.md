# TerraTNT 可视化验证系统 v2 — 软件架构设计文档

> 更新日期: 2025-02-11
> 版本: v2.0 (统一架构)

## 一、系统定位

**目标**：提供一个统一的交互式桌面平台，覆盖完整工作流：
1. **样本分析** — 逐样本检查模型预测质量（轨迹 vs GT），环境特征18通道可视化
2. **多模型对比** — TerraTNT / V3~V7 / baselines 同屏对比，误差曲线
3. **Phase评估** — 一键运行Phase V2评估，结果表格+图表
4. **跨区域泛化** — bohemian_forest → scottish_highlands 等跨域评估与对比
5. **数据管理** — 数据集浏览、轨迹生成、FAS Split生成
6. **模型训练** — GUI驱动训练，支持混合区域联合训练

**技术选型**：Python + PyQt6 + matplotlib（桌面应用），支持离线使用。

**合并来源**：
- `gui/` (旧) — PyQt5 Tab架构、Worker线程模式、MenuBar/ToolBar
- `visualization/` (新) — DataManager、MapCanvas、ModelManager、真实数据集成

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  MainWindow (QMainWindow)                        │
├─────────────────────────────────────────────────────────────────┤
│  MenuBar: 文件 | 工具(重载模型) | 帮助                            │
│  ToolBar: [刷新区域] [推理当前样本]                                │
├─────────────────────────────────────────────────────────────────┤
│  QTabWidget                                                      │
│  ┌───────┬───────┬───────┬───────┐                              │
│  │ 分析  │ 评估  │ 数据  │ 训练  │                              │
│  └───┬───┴───┬───┴───┬───┴───┬───┘                              │
│      │       │       │       │                                   │
│  Tab1: AnalysisTab (核心)                                        │
│  ┌──────────┬────────────────────────────────────┐              │
│  │ 左侧控制  │         右侧可视化 (2x2)            │              │
│  │ ┌──────┐ │  ┌──────────────┐  ┌────────────┐  │              │
│  │ │区域   │ │  │ A. MapView   │  │ B. 轨迹对比 │  │              │
│  │ │Phase  │ │  │  (环境底图+  │  │ (matplotlib │  │              │
│  │ │模型   │ │  │   轨迹叠加)  │  │  多模型)    │  │              │
│  │ │过滤   │ │  └──────────────┘  └────────────┘  │              │
│  │ │样本列表│ │  ┌──────────────┐  ┌────────────┐  │              │
│  │ │指标   │ │  │ C. Env通道   │  │ D. 误差曲线 │  │              │
│  │ └──────┘ │  │  (18ch选择)  │  │  + 样本信息 │  │              │
│  └──────────┴──┴──────────────┴──┴────────────┴──┘              │
│                                                                  │
│  Tab2: EvaluationTab                                             │
│  ┌──────────┬────────────────────────────────────┐              │
│  │ 配置     │  结果 (日志/表格/跨区域对比图)       │              │
│  │ 区域     │                                     │              │
│  │ Phase选择│  ┌─────────────────────────────┐    │              │
│  │ 参数     │  │ 结果表: Phase x Model x ADE │    │              │
│  │ 跨区域   │  │ 跨区域柱状图对比             │    │              │
│  └──────────┴──┴─────────────────────────────┴────┘              │
│                                                                  │
│  Tab3: DataTab                                                   │
│  ┌──────────┬────────────────────────────────────┐              │
│  │ 生成控制  │  数据集列表 / 详情 / 日志           │              │
│  │ FAS Split│                                     │              │
│  └──────────┴────────────────────────────────────┘              │
│                                                                  │
│  Tab4: TrainingTab                                               │
│  ┌──────────┬────────────────────────────────────┐              │
│  │ 超参数   │  训练日志 (实时流)                   │              │
│  │ 模型选择  │                                     │              │
│  │ 混合训练  │                                     │              │
│  └──────────┴────────────────────────────────────┘              │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  StatusBar: [状态] ... [时间] [模型: N个已加载]                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、界面模块详细设计

### 3.1 左侧控制面板 (ControlPanel)

#### 3.1.1 区域选择器
- **下拉框**：bohemian_forest / scottish_highlands / donbas / carpathians
- 切换区域时自动加载对应的环境栅格和数据集
- 显示区域基本信息（覆盖范围、CRS、样本数）

#### 3.1.2 模型选择器
- **多选复选框列表**：
  - TerraTNT (baseline)
  - V3_Waypoint
  - V4_WP_Spatial
  - V6_GoalRefine
  - V6R_RegionPrior
  - V7_ConfGate
  - LSTM_only / LSTM_Env_Goal / Seq2Seq / MLP
  - CV (Constant Velocity)
- 每个模型旁显示颜色标记（与轨迹绘制颜色一致）
- 勾选/取消即时更新可视化

#### 3.1.3 样本浏览器
- **列表视图**：显示所有样本，每行包含：
  - 样本ID
  - 轨迹类型（intent1/2/3）
  - 车辆类型（type1-4）
  - 曲折度（sinuosity）
  - 最佳模型ADE（快速排序用）
- **排序**：按ADE/sinuosity/距离排序
- **过滤**：
  - 按intent过滤
  - 按vehicle_type过滤
  - 按ADE范围过滤（只看"差"的样本）
  - 按sinuosity范围过滤（直行/弯曲）
- **搜索**：按样本ID搜索
- 单击选中 → 中央区域更新

#### 3.1.4 Phase实验选择器
- **单选按钮组**：
  - Phase 1a (精确终点, 域内)
  - Phase 1b (精确终点, OOD)
  - Phase 2a (区域先验, σ=10km)
  - Phase 2b (区域先验, σ=15km)
  - Phase 2c (区域先验, 偏移5km)
  - Phase 3a (无先验, 直行)
  - Phase 3b (无先验, 转弯)
- 切换Phase时自动更新先验热力图和评估指标

#### 3.1.5 快速指标面板
- 当前选中样本的各模型ADE/FDE
- 当前Phase的汇总统计（均值±标准差）
- V7 gate值 α 显示

---

### 3.2 中央可视化区域 (VisualizationArea)

#### 面板 A：地图视图 (MapView) — **主视图**

**内容**：
- **底图层**（可切换）：
  - DEM hillshade（地形阴影）
  - Slope 热力图
  - LULC 分类着色
  - Road 网络叠加
  - Cost map 热力图
  - Passable mask
- **轨迹层**：
  - 历史轨迹（灰色虚线 + 方向箭头）
  - GT未来轨迹（黑色实线 + 终点标记）
  - 各模型预测轨迹（不同颜色实线）
  - 候选终点（彩色圆点，大小表示分类器置信度）
  - GT终点（黑色五角星）
- **先验层**：
  - Phase对应的先验热力图（半透明叠加）
  - 先验质心标记
- **交互**：
  - 鼠标滚轮缩放
  - 拖拽平移
  - 鼠标悬停显示坐标 + 环境特征值
  - 右键菜单：导出当前视图为PNG

**视图范围**：以最后观测点为中心，默认140km×140km，可缩放到10km级别查看细节。

#### 面板 B：轨迹对比视图 (TrajectoryCompareView)

**内容**：
- 相对坐标系（以最后观测点为原点）
- 所有选中模型的预测轨迹 vs GT
- 不带环境底图，纯几何对比
- 每条轨迹旁标注ADE值
- 误差带（沿GT轨迹的逐步误差用颜色编码）

**用途**：快速对比轨迹形状差异，不受底图干扰。

#### 面板 C：环境特征视图 (EnvFeatureView)

**内容**：
- 18通道环境地图的可视化（128×128）
- **通道选择器**（下拉/标签页）：
  - Ch0: DEM (归一化)
  - Ch1: Slope
  - Ch2-3: Aspect sin/cos
  - Ch4-13: LULC one-hot (10类)
  - Ch14: Tree cover
  - Ch15: Road
  - Ch16: History heatmap
  - Ch17: Goal/Prior map ← **关键：显示Phase对应的先验**
- 叠加轨迹（历史+GT+预测）在环境地图上
- V7专用：显示gate网络的attention/activation

**用途**：验证模型"看到"了什么环境信息，诊断环境编码器是否有效。

#### 面板 D：统计/指标视图 (MetricsView)

**内容**（多个子标签页）：

**D1. 当前样本误差曲线**：
- X轴=预测时间步，Y轴=位置误差(m)
- 每个模型一条曲线
- 显示误差随时间增长的模式

**D2. Phase汇总柱状图**：
- 所有模型在当前Phase的ADE/FDE对比
- 柱状图 + 误差条

**D3. 误差分布直方图**：
- 当前Phase下所有样本的ADE分布
- 可叠加多个模型的分布

**D4. V7 Gate分析**（V7专用）：
- Gate值 α 的分布直方图
- α vs ADE 散点图
- α vs 候选终点质量 散点图

---

### 3.3 底部导航栏 (NavigationBar)

- **样本导航**：← 上一个 | 样本 X/N | 下一个 →
- **快捷键**：Left/Right 切换样本，Space 切换底图
- **批量导出**：导出当前Phase所有样本的可视化为PDF/PNG
- **状态信息**：当前区域、Phase、样本数、加载状态

---

## 四、数据流和后端逻辑

### 4.1 数据加载器 (DataManager)

```python
class DataManager:
    """统一数据管理"""
    
    def load_region(self, region: str):
        """加载区域环境栅格"""
        # DEM, Slope, Aspect, LULC, Road, Cost Map, Passable Mask
        # 路径: data/processed/utm_grid/{region}/
    
    def load_dataset(self, region: str, split: str):
        """加载数据集样本"""
        # 路径: data/processed/fas_splits/{region}/
        # 返回: List[Sample] with history, future, env_map, candidates
    
    def load_model_predictions(self, model_name: str, phase: str):
        """加载模型预测结果"""
        # 路径: outputs/evaluation/phase_v2/{phase}/{model}/
        # 或实时推理
    
    def get_phase_prior(self, sample, phase: str):
        """生成Phase对应的先验热力图"""
        # Phase 1: σ=1km高斯 @ GT终点
        # Phase 2: σ=10/15km高斯 @ GT终点(±偏移)
        # Phase 3: 扇形分布
```

### 4.2 模型推理器 (ModelInference)

```python
class ModelInference:
    """实时模型推理（可选，用于交互式探索）"""
    
    def load_model(self, model_name: str, checkpoint: str):
        """加载模型权重"""
    
    def predict(self, sample, phase_prior):
        """对单个样本进行推理"""
        # 返回: predicted_trajectory, goal_logits, (alpha for V7)
    
    def batch_evaluate(self, dataset, phase: str):
        """批量评估，缓存结果"""
```

### 4.3 缓存策略

- 环境栅格：首次加载后常驻内存（~500MB per region）
- 模型预测：预计算并缓存为JSON/PKL
- 环境patch：LRU缓存最近100个样本的128×128 patch

---

## 五、交互流程

### 5.1 典型使用流程

```
1. 启动 → 选择区域(bohemian_forest) → 加载环境数据
2. 选择模型(V6R, V7, LSTM_only) → 加载/计算预测
3. 选择Phase(2a) → 更新先验热力图
4. 浏览样本列表 → 按ADE排序 → 点击感兴趣的样本
5. 地图视图：观察轨迹与地形的关系
6. 环境特征视图：检查Ch17(先验)是否合理
7. 统计视图：对比各模型误差曲线
8. 切换到Phase 3a → 观察无先验时的表现
9. 切换区域到scottish_highlands → 观察跨域泛化
```

### 5.2 关键交互

| 操作 | 效果 |
|---|---|
| 点击样本列表 | 所有面板同步更新到该样本 |
| 切换Phase | 更新先验热力图 + 重新计算/加载预测 |
| 勾选/取消模型 | 轨迹视图即时添加/移除对应轨迹 |
| 鼠标悬停地图 | 底部显示坐标+DEM高程+Slope+LULC类别 |
| 双击地图某点 | 弹出该点的完整环境特征(26维) |
| Ctrl+S | 导出当前视图 |
| 切换底图 | 地图视图底图在DEM/Slope/LULC/Road间切换 |

---

## 六、特色功能

### 6.1 Gate诊断模式（V7专用）
- 在地图视图上用颜色编码显示gate值α沿轨迹的变化
- 红色=高α(信任goal)，蓝色=低α(忽略goal)
- 帮助理解gate在什么地形/情况下做出什么决策

### 6.2 跨区域对比模式
- 左右分屏：左=bohemian_forest，右=scottish_highlands
- 同一模型在两个区域的表现并排对比
- 自动匹配相似特征的样本（相近sinuosity/距离）

### 6.3 误差热力图
- 在地图上用颜色编码显示每个样本的ADE
- 红色=高误差区域，绿色=低误差区域
- 帮助发现"模型在哪些地形上表现差"

### 6.4 候选终点质量分析
- 显示6个候选终点与GT终点的距离
- 分类器选择的终点用高亮标记
- 帮助诊断"是候选生成差还是分类器选择差"

### 6.5 批量报告生成
- 一键生成当前Phase的完整可视化报告(PDF)
- 包含：汇总统计 + Top-10最差样本 + Top-10最好样本 + 分布图

---

## 七、文件结构 (v2 实际实现)

```
visualization/                          # 统一可视化验证系统
├── __init__.py
├── main.py                             # 入口: QApplication + 深色主题 + MainWindow
├── ui/
│   ├── __init__.py
│   ├── main_window.py                  # 统一主窗口: MenuBar + ToolBar + 4 Tabs + StatusBar
│   │                                   #   ModelLoadWorker (后台加载模型)
│   │                                   #   InferenceWorker (后台推理)
│   ├── analysis_tab.py                 # Tab1: 交互式样本分析 (核心)
│   │                                   #   AnalysisTab: 左侧控制 + 右侧2x2可视化
│   │                                   #   TrajectoryView: matplotlib轨迹对比
│   │                                   #   EnvChannelView: 18通道环境特征
│   │                                   #   MetricsView: 误差曲线 + 样本信息
│   ├── evaluation_tab.py               # Tab2: Phase评估 + 跨区域对比
│   │                                   #   EvalWorker: 后台运行evaluate_phases_v2.py
│   │                                   #   结果表格 + 跨区域柱状图
│   ├── data_tab.py                     # Tab3: 数据管理
│   │                                   #   DataGenWorker: 后台轨迹生成
│   │                                   #   FAS Split生成 (内联)
│   │                                   #   数据集浏览 + 统计
│   ├── training_tab.py                 # Tab4: 模型训练
│   │                                   #   TrainWorker: 后台训练
│   │                                   #   混合区域训练支持
│   └── map_view.py                     # MapView + MapCanvas: 环境底图 + 轨迹叠加
│                                       #   支持缩放/平移, DEM/Slope/LULC/Road切换
├── core/
│   ├── __init__.py
│   ├── data_manager.py                 # DataManager: 环境栅格 + 数据集加载
│   │                                   #   RegionData: 区域环境数据 + patch提取
│   │                                   #   SampleData: 单样本数据结构
│   └── model_manager.py                # ModelManager: 统一模型加载 + 推理
│                                       #   自动发现checkpoint
│                                       #   支持: TerraTNT, V3~V7, baselines, CV
├── utils/
│   ├── __init__.py
│   └── colors.py                       # 颜色方案: MODEL_COLORS, LULC颜色
└── config/
    ├── __init__.py
    └── vis_config.yaml                 # 可视化配置 (路径, 模型, Phase定义)

gui/                                    # [旧] 已合并到 visualization/, 保留供参考
├── main_window.py                      # PyQt5 Tab架构 (已被v2替代)
└── panels/                             # 5个面板 (已被v2 Tab替代)
```

---

## 八、实现状态

| 状态 | 模块 | 说明 |
|---|---|---|
| **已完成** | AnalysisTab (MapView + 轨迹 + 环境 + 指标) | 核心交互式分析 |
| **已完成** | ModelManager (自动发现 + 加载 + 推理) | 后台模型管理 |
| **已完成** | EvaluationTab (Phase评估 + 跨区域) | 一键评估 |
| **已完成** | DataTab (数据集浏览 + 生成 + Split) | 数据管理 |
| **已完成** | TrainingTab (GUI训练 + 混合区域) | 训练控制 |
| **已完成** | 统一架构 (MenuBar + ToolBar + StatusBar) | 合并gui/+visualization/ |
| 待完善 | Gate诊断模式 (V7 α可视化) | V7专用分析 |
| 待完善 | 批量报告生成 (PDF导出) | 论文准备 |
| 待完善 | 误差热力图 (地图上的ADE分布) | 高级分析 |
