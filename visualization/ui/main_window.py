#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TerraTNT 可视化验证系统 v2 - 统一主窗口

架构:
  MainWindow
  ├── MenuBar + ToolBar
  ├── QTabWidget
  │   ├── Tab1: AnalysisTab   (交互式样本分析 — 核心)
  │   ├── Tab2: EvaluationTab (Phase评估 + 跨区域对比)
  │   ├── Tab3: DataTab       (数据管理 + 生成)
  │   └── Tab4: TrainingTab   (模型训练)
  └── StatusBar

合并来源:
  - gui/          → Tab架构, Worker线程, MenuBar/ToolBar
  - visualization/ → DataManager, MapCanvas, ModelManager, 真实数据集成
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QApplication, QLabel, QMessageBox, QToolBar,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QShortcut, QKeySequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.core.data_manager import DataManager, SampleData
from visualization.utils.colors import MODEL_COLORS
from visualization.ui.analysis_tab import AnalysisTab
from visualization.ui.evaluation_tab import EvaluationTab
from visualization.ui.data_tab import DataTab
from visualization.ui.training_tab import TrainingTab
from visualization.ui.scenario_tab import ScenarioTab

logger = logging.getLogger(__name__)


# ============================================================
#  后台模型加载 Worker
# ============================================================

class DataLoadWorker(QThread):
    """后台加载区域数据+样本"""
    progress = pyqtSignal(str)
    done = pyqtSignal(str)  # region name
    error = pyqtSignal(str)

    def __init__(self, dm, region):
        super().__init__()
        self.dm = dm
        self.region = region
        self.samples = []  # 结果存在属性上

    def run(self):
        try:
            self.progress.emit(f"加载 {self.region} 环境栅格...")
            self.dm.load_region(self.region)
            self.progress.emit(f"加载 {self.region} 样本数据...")
            self.samples = self.dm.load_dataset(self.region, max_samples=2000)
            self.progress.emit(f"{self.region}: {len(self.samples)} 样本就绪")
            self.done.emit(self.region)
        except Exception as e:
            import traceback
            self.error.emit(f"加载失败: {e}\n{traceback.format_exc()}")


class ModelLoadWorker(QThread):
    """后台加载所有模型"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(int)

    def run(self):
        try:
            from visualization.core.model_manager import ModelManager
            self.mm = ModelManager()
            self.progress.emit("发现模型checkpoint...")
            ckpts = self.mm.discover_checkpoints()
            self.progress.emit(f"找到 {len(ckpts)} 个模型")
            count = 0
            for name, path in ckpts.items():
                self.progress.emit(f"加载 {name}...")
                if self.mm.load_model(name, path):
                    count += 1
            self.finished.emit(count)
        except Exception as e:
            self.progress.emit(f"模型加载失败: {e}")
            self.mm = None
            self.finished.emit(0)


class InferenceWorker(QThread):
    """后台推理 Worker (Phase感知)"""
    progress = pyqtSignal(str)
    result = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, model_manager, sample, model_names,
                 env_map=None, candidates=None):
        super().__init__()
        self.mm = model_manager
        self.sample = sample
        self.model_names = model_names
        self.env_map_override = env_map
        self.candidates_override = candidates

    def run(self):
        predictions = {}
        for name in self.model_names:
            self.progress.emit(f"推理 {name}...")
            hist = (self.sample.history_feat
                    if hasattr(self.sample, 'history_feat') and self.sample.history_feat is not None
                    else np.zeros((90, 26), dtype=np.float32))
            env = self.env_map_override if self.env_map_override is not None else self.sample.env_map
            cands = self.candidates_override
            if cands is None:
                cands = (self.sample.candidates_rel
                         if len(self.sample.candidates_rel) > 0 else None)
            pred = self.mm.predict(name, hist, env, cands)
            if pred is not None:
                predictions[name] = pred
        self.result.emit(predictions)
        self.finished.emit()


# ============================================================
#  主窗口
# ============================================================

class MainWindow(QMainWindow):
    """统一主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TerraTNT 可视化验证系统 v2")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 950)

        # 核心状态
        self.dm = DataManager()
        self.model_manager = None  # 延迟加载
        self.current_phase = 'P1a'
        self.current_sample: SampleData = None
        self.current_predictions = {}
        self.model_visibility = {m: (m in ('V6R_Robust', 'V7_ConfGate', 'V3_Waypoint'))
                                 for m in MODEL_COLORS}
        self._inference_worker = None

        # --- UI ---
        self._build_menu()
        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()
        self._bind_shortcuts()

        # 初始化
        self._init_regions()

        # 后台加载模型
        self._start_model_loading()

        # 交互: 右键放置候选目标点
        self.analysis_tab.map_view.canvas.candidate_placed.connect(
            self._on_candidate_placed)

    # --------------------------------------------------------
    #  UI 构建
    # --------------------------------------------------------

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        exit_act = QAction("退出", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        tools_menu = menubar.addMenu("工具")
        reload_models_act = QAction("重新加载模型", self)
        reload_models_act.triggered.connect(self._start_model_loading)
        tools_menu.addAction(reload_models_act)

        help_menu = menubar.addMenu("帮助")
        about_act = QAction("关于", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self):
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        refresh_act = QAction("刷新区域", self)
        refresh_act.setStatusTip("重新扫描可用区域")
        refresh_act.triggered.connect(self._init_regions)
        toolbar.addAction(refresh_act)

        toolbar.addSeparator()

        infer_act = QAction("推理当前样本", self)
        infer_act.setStatusTip("对当前样本运行所有已加载模型")
        infer_act.triggered.connect(self._run_inference)
        toolbar.addAction(infer_act)

    def _build_tabs(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        self.analysis_tab = AnalysisTab(self)
        self.scenario_tab = ScenarioTab(self)
        self.evaluation_tab = EvaluationTab(self)
        self.data_tab = DataTab(self)
        self.training_tab = TrainingTab(self)

        self.tab_widget.addTab(self.analysis_tab, "分析")
        self.tab_widget.addTab(self.scenario_tab, "仿真")
        self.tab_widget.addTab(self.evaluation_tab, "评估")
        self.tab_widget.addTab(self.data_tab, "数据")
        self.tab_widget.addTab(self.training_tab, "训练")

        layout.addWidget(self.tab_widget)

    def _build_statusbar(self):
        self.status_label = QLabel("就绪")
        self.statusBar().addWidget(self.status_label, stretch=1)
        self.time_label = QLabel("")
        self.statusBar().addPermanentWidget(self.time_label)
        self.model_status_label = QLabel("模型: 加载中...")
        self.statusBar().addPermanentWidget(self.model_status_label)

        self.timer = QTimer()
        self.timer.timeout.connect(
            lambda: self.time_label.setText(datetime.now().strftime("%H:%M:%S")))
        self.timer.start(1000)

    def _bind_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._prev_sample)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._next_sample)

    # --------------------------------------------------------
    #  初始化
    # --------------------------------------------------------

    def _init_regions(self):
        regions = self.dm.available_regions()
        self.analysis_tab.populate_regions(regions)
        self.evaluation_tab.set_regions(regions)
        self.data_tab.set_regions(regions)
        self.training_tab.set_regions(regions)
        self.data_tab._refresh_datasets()
        if regions:
            self.status_label.setText(f"发现 {len(regions)} 个区域: {', '.join(regions)}")

    def _start_model_loading(self):
        self.model_status_label.setText("模型: 加载中...")
        self._model_worker = ModelLoadWorker()
        self._model_worker.progress.connect(
            lambda msg: self.model_status_label.setText(f"模型: {msg}"))
        self._model_worker.finished.connect(self._on_models_loaded)
        self._model_worker.start()

    def _on_models_loaded(self, count):
        if hasattr(self._model_worker, 'mm') and self._model_worker.mm is not None:
            self.model_manager = self._model_worker.mm
            names = self.model_manager.get_loaded_models()
            self.model_status_label.setText(f"模型: {count} 个已加载")
            logger.info(f"已加载模型: {names}")
        else:
            self.model_status_label.setText("模型: 加载失败")

    # --------------------------------------------------------
    #  公共接口 (被各Tab调用)
    # --------------------------------------------------------

    def load_region(self, region: str):
        """异步加载区域数据（后台线程）"""
        if not region:
            return
        self.status_label.setText(f"正在加载 {region}...")
        self._data_worker = DataLoadWorker(self.dm, region)
        self._data_worker.progress.connect(
            lambda msg: (self.status_label.setText(msg),
                         self.analysis_tab.on_load_started(msg)))
        self._data_worker.done.connect(self._on_region_loaded)
        self._data_worker.error.connect(self._on_region_load_error)
        self._data_worker.start()

    def _on_region_loaded(self, region):
        """区域数据加载完成回调"""
        self.analysis_tab.on_load_finished()
        rd = self.dm.get_region()
        if rd and rd.bounds:
            w_km = (rd.bounds.right - rd.bounds.left) / 1000
            h_km = (rd.bounds.top - rd.bounds.bottom) / 1000
            self.analysis_tab.region_info.setText(
                f"{w_km:.0f}x{h_km:.0f}km | {rd.crs} | {rd.shape[0]}x{rd.shape[1]}px")
            # 生成全局环境缩略图 (只在切换区域时执行一次)
            self.analysis_tab.env_view.set_region_data(rd)
            # 传递区域数据给地图视图 (卫星影像需要CRS)
            self.analysis_tab.map_view.set_region_data(rd)
        samples = self.dm.samples
        self.analysis_tab.populate_samples(samples)
        # 同步到仿真标签页
        if rd:
            self.scenario_tab.set_region_data(rd)
        self.scenario_tab.populate_samples(samples)
        self.status_label.setText(f"{region}: {len(samples)} 样本已加载")
        # 自动选择第一个样本
        if self.analysis_tab.sample_list.count() > 0:
            self.analysis_tab.sample_list.setCurrentRow(0)

    def _on_region_load_error(self, msg):
        """区域加载失败"""
        self.analysis_tab.on_load_finished()
        self.status_label.setText(msg)
        self.analysis_tab.region_info.setText(msg)

    def set_phase(self, phase_key: str):
        self.current_phase = phase_key
        self.status_label.setText(f"Phase: {phase_key}")
        if self.current_sample:
            self._update_views()

    def set_model_visibility(self, model_name: str, visible: bool):
        self.model_visibility[model_name] = visible
        self._update_views()

    def select_sample(self, sample_id: str):
        for s in self.dm.samples:
            if s.sample_id == sample_id:
                self.current_sample = s
                break
        else:
            return
        self.status_label.setText(f"样本: {sample_id}")
        self.current_predictions = {}
        self._update_views()

    def refresh_sample_list(self):
        self.analysis_tab.populate_samples(self.dm.samples)

    # --------------------------------------------------------
    #  推理
    # --------------------------------------------------------

    def _run_inference(self):
        if self.current_sample is None or self.model_manager is None:
            return
        visible_models = [m for m, v in self.model_visibility.items()
                          if v and m in self.model_manager.models]
        if not visible_models:
            self.current_predictions = {}
            self._update_views()
            return

        # Phase感知: 根据当前Phase修改env_map ch17和候选终点
        from visualization.utils.phase_utils import prepare_phase_inputs
        env_map, candidates = prepare_phase_inputs(
            self.current_sample, self.current_phase)

        self.status_label.setText(f"推理中 ({self.current_phase})...")
        self._inference_worker = InferenceWorker(
            self.model_manager, self.current_sample, visible_models,
            env_map=env_map, candidates=candidates)
        self._inference_worker.result.connect(self._on_inference_done)
        self._inference_worker.finished.connect(
            lambda: self.status_label.setText(f"推理完成 ({self.current_phase})"))
        self._inference_worker.start()

    def _on_inference_done(self, predictions):
        self.current_predictions = predictions
        self._update_views()

    # --------------------------------------------------------
    #  视图更新
    # --------------------------------------------------------

    def _update_views(self):
        sample = self.current_sample
        if sample is None:
            return
        rd = self.dm.get_region()
        self.analysis_tab.update_views(
            sample, self.current_predictions, self.model_visibility, rd)

        # 快速指标
        lines = []
        for mn, pred in self.current_predictions.items():
            if pred is not None and sample.future_rel is not None and len(sample.future_rel) > 0:
                ade = self.dm.compute_ade(pred, sample.future_rel)
                fde = self.dm.compute_fde(pred, sample.future_rel)
                lines.append(f"{mn:15s} ADE={ade:7.0f}m FDE={fde:7.0f}m")
        self.analysis_tab.metrics_label.setText(
            '\n'.join(lines) if lines else "无预测")

    # --------------------------------------------------------
    #  导航
    # --------------------------------------------------------

    def _prev_sample(self):
        sl = self.analysis_tab.sample_list
        row = sl.currentRow()
        if row > 0:
            sl.setCurrentRow(row - 1)

    def _next_sample(self):
        sl = self.analysis_tab.sample_list
        row = sl.currentRow()
        if row < sl.count() - 1:
            sl.setCurrentRow(row + 1)

    # --------------------------------------------------------
    #  对话框
    # --------------------------------------------------------

    def _on_candidate_placed(self, map_rel_x, map_rel_y):
        """右键在地图上放置候选目标点
        
        map_rel_x, map_rel_y: 相对于地图中心的km坐标
        需要转换为相对于观测点(obs)的km坐标
        """
        if self.current_sample is None:
            return
        from visualization.ui.analysis_tab import _compute_traj_bbox
        cx, cy, half = _compute_traj_bbox(
            self.current_sample, self.current_predictions, padding_km=10.0)
        # map_center偏移 = (cx, cy) km from obs
        # 地图坐标 → obs-relative坐标: obs_rel = map_rel + (cx, cy)
        obs_rel_x = map_rel_x + cx
        obs_rel_y = map_rel_y + cy
        new_cand = np.array([[obs_rel_x, obs_rel_y]], dtype=np.float32)
        # 追加到现有候选点
        if self.current_sample.candidates_rel is not None and len(self.current_sample.candidates_rel) > 0:
            self.current_sample.candidates_rel = np.concatenate(
                [self.current_sample.candidates_rel, new_cand], axis=0)
        else:
            self.current_sample.candidates_rel = new_cand
        self.status_label.setText(
            f"已放置候选点: ({obs_rel_x:.1f}, {obs_rel_y:.1f}) km | "
            f"共 {len(self.current_sample.candidates_rel)} 个候选点")
        self._update_views()

    def _show_about(self):
        QMessageBox.about(self, "关于",
            "<h2>TerraTNT 可视化验证系统 v2</h2>"
            "<p>多星协同观测任务规划 - 地面目标轨迹预测</p>"
            "<p>统一架构: 分析 / 评估 / 数据 / 训练</p>")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, '确认退出', '确定要退出吗？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()
