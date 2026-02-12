#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tab 1: 交互式样本分析 — 核心标签页

设计参考: QGIS/ParaView 面板布局
- 左侧: 控制面板 (区域+加载按钮, 样本列表, 模型选择)
- 右侧: 2x2 可视化网格 (地图/轨迹/环境/指标)
- 显式操作按钮, 进度反馈, 空状态引导
"""
import sys
import numpy as np
from pathlib import Path

# matplotlib 中文字体 — 必须在创建任何 Figure 之前设置
import matplotlib
import matplotlib.font_manager as fm

# 查找可用CJK字体 (matplotlib注册名可能与系统名不同)
_CJK_FONT_NAME = None
for candidate in ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Droid Sans Fallback',
                  'WenQuanYi Micro Hei', 'AR PL UKai CN', 'SimHei']:
    matches = [f for f in fm.fontManager.ttflist if f.name == candidate]
    if matches:
        _CJK_FONT_NAME = candidate
        break

if _CJK_FONT_NAME:
    matplotlib.rcParams['font.sans-serif'] = [_CJK_FONT_NAME, 'DejaVu Sans']
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
_CJK_FONT = fm.FontProperties(family=_CJK_FONT_NAME) if _CJK_FONT_NAME else None

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
    QListWidgetItem, QComboBox, QLabel, QGroupBox, QCheckBox,
    QPushButton, QLineEdit, QScrollArea, QTabWidget, QProgressBar,
    QSizePolicy,
)
from PyQt6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from visualization.ui.map_view import MapView
from visualization.utils.colors import MODEL_COLORS, hex_to_rgb

# ============================================================
#  样式常量
# ============================================================
PANEL_STYLE = """
QGroupBox {
    font-weight: bold; font-size: 12px;
    border: 1px solid #555; border-radius: 4px;
    margin-top: 8px; padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
}
"""

PLACEHOLDER_STYLE = """
    color: #777; font-size: 13px; font-style: italic;
    background: #2a2a2a; border: 1px dashed #555; border-radius: 6px;
"""

BTN_PRIMARY = """
QPushButton {
    background: #2979ff; color: white; border: none;
    border-radius: 4px; padding: 6px 16px; font-weight: bold;
}
QPushButton:hover { background: #448aff; }
QPushButton:pressed { background: #1565c0; }
QPushButton:disabled { background: #555; color: #999; }
"""

BTN_SECONDARY = """
QPushButton {
    background: #424242; color: #ddd; border: 1px solid #666;
    border-radius: 4px; padding: 4px 12px;
}
QPushButton:hover { background: #555; }
QPushButton:disabled { background: #333; color: #666; }
"""


def _make_placeholder(text):
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(PLACEHOLDER_STYLE)
    lbl.setWordWrap(True)
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return lbl


def _set_ax_font(ax, title='', xlabel='', ylabel=''):
    """统一设置坐标轴字体, 确保中文正常显示"""
    fp = _CJK_FONT if _CJK_FONT else {}
    kw = dict(fontproperties=fp) if fp else {}
    if title:
        ax.set_title(title, color='#ddd', fontsize=9, pad=4, **kw)
    if xlabel:
        ax.set_xlabel(xlabel, color='#aaa', fontsize=9, **kw)
    if ylabel:
        ax.set_ylabel(ylabel, color='#aaa', fontsize=9, **kw)


def _compute_traj_bbox(sample, predictions=None, padding_km=10.0, history_steps=30):
    """计算轨迹最小外接正方形 (km), 返回 (center_x, center_y, half_size)
    
    padding_km: 在最小外接正方形每侧额外扩展的距离(km)
    history_steps: 用于计算bbox的历史步数 (None=全部)
    """
    pts = []
    if sample.history_rel is not None and len(sample.history_rel) > 0:
        h = sample.history_rel[-history_steps:] if history_steps else sample.history_rel
        pts.append(h)
    if sample.future_rel is not None and len(sample.future_rel) > 0:
        pts.append(sample.future_rel)
    if sample.candidates_rel is not None and len(sample.candidates_rel) > 0:
        pts.append(sample.candidates_rel)
    if predictions:
        for pred in predictions.values():
            if pred is not None and len(pred) > 0:
                pts.append(pred)
    pts.append(np.array([[0, 0]]))  # 观测原点

    combined = np.concatenate(pts, axis=0)
    xmin, ymin = combined.min(axis=0)
    xmax, ymax = combined.max(axis=0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 + padding_km
    half = max(half, 15.0)  # 至少15km半径
    return cx, cy, half


# ============================================================
#  子视图: 全局环境通道 (区域级栅格缩略图 + 红色局部框)
# ============================================================
class EnvChannelView(QWidget):
    CH_NAMES = [
        '高程(DEM)', '坡度', '土地覆盖', '道路',
    ]
    CH_KEYS = ['dem', 'slope', 'lulc', 'road']

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        header = QHBoxLayout()
        title = QLabel("全局环境总览")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        header.addWidget(title)
        header.addStretch()
        self.ch_combo = QComboBox()
        self.ch_combo.addItems(self.CH_NAMES)
        self.ch_combo.setFixedWidth(120)
        self.ch_combo.currentIndexChanged.connect(self._redraw)
        header.addWidget(self.ch_combo)
        layout.addLayout(header)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(4, 4), dpi=100, facecolor='#2a2a2a')
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.set_facecolor('#2a2a2a')
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        self._placeholder = _make_placeholder("加载区域后\n显示全局环境总览")
        layout.addWidget(self._placeholder)
        self.canvas.hide()

        # 区域级缩略图缓存 (静态, 只在切换区域时更新)
        self._thumbnails = {}  # {key: (256, 256) ndarray}
        self._region_bounds = None  # rasterio BoundingBox
        self._region_name = None
        # 当前样本的局部框 (UTM坐标)
        self._local_bbox_utm = None  # (left, bottom, right, top)

    def set_region_data(self, region_data):
        """加载区域时调用, 生成全局缩略图 (静态不变)"""
        if region_data is None:
            self._thumbnails = {}
            self._region_bounds = None
            self._region_name = None
            self._placeholder.show()
            self.canvas.hide()
            return

        from skimage.transform import resize
        thumb_size = 256
        self._region_bounds = region_data.bounds
        self._region_name = region_data.region

        # DEM缩略图
        if region_data.dem is not None:
            dem = region_data.dem.copy()
            dem = np.where(dem < -500, np.nan, dem)
            self._thumbnails['dem'] = resize(
                np.nan_to_num(dem, nan=float(np.nanmean(dem[np.isfinite(dem)])) if np.any(np.isfinite(dem)) else 0),
                (thumb_size, thumb_size), anti_aliasing=True, preserve_range=True).astype(np.float32)
        # 坡度缩略图
        if region_data.slope is not None:
            self._thumbnails['slope'] = resize(
                region_data.slope, (thumb_size, thumb_size),
                anti_aliasing=True, preserve_range=True).astype(np.float32)
        # LULC缩略图 (最近邻)
        if region_data.lulc is not None:
            self._thumbnails['lulc'] = resize(
                region_data.lulc, (thumb_size, thumb_size),
                order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
        # 道路缩略图
        if region_data.road is not None:
            self._thumbnails['road'] = resize(
                region_data.road, (thumb_size, thumb_size),
                anti_aliasing=True, preserve_range=True).astype(np.float32)

        self._placeholder.hide()
        self.canvas.show()
        self._redraw()

    def update_bbox(self, local_bbox_utm):
        """切换样本时只更新红色框位置 (不重新生成缩略图)
        
        local_bbox_utm: (left, bottom, right, top) UTM坐标
        """
        self._local_bbox_utm = local_bbox_utm
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        if not self._thumbnails or self._region_bounds is None:
            self.canvas.draw_idle()
            return

        ch_idx = self.ch_combo.currentIndex()
        key = self.CH_KEYS[ch_idx]
        data = self._thumbnails.get(key)
        if data is None:
            self.ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                         color='#777', fontsize=11, transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        # 选择colormap
        if key == 'dem':
            cmap = 'terrain'
        elif key == 'slope':
            cmap = 'YlOrRd'
        elif key == 'road':
            cmap = 'gray'
        elif key == 'lulc':
            cmap = 'viridis'
        else:
            cmap = 'viridis'

        # 用区域UTM范围作为extent
        b = self._region_bounds
        extent = [b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000]  # km
        self.ax.imshow(data, cmap=cmap, origin='upper', aspect='equal', extent=extent)

        name = self.CH_NAMES[ch_idx]
        w_km = (b.right - b.left) / 1000
        h_km = (b.top - b.bottom) / 1000
        _set_ax_font(self.ax, title=f"{self._region_name} {name} ({w_km:.0f}×{h_km:.0f}km)")

        # 绘制红色矩形框标注底图局部范围
        if self._local_bbox_utm is not None:
            import matplotlib.patches as mpatches
            left, bottom, right, top = self._local_bbox_utm
            # 转为km
            lk, bk, rk, tk = left / 1000, bottom / 1000, right / 1000, top / 1000
            rect = mpatches.Rectangle((lk, bk), rk - lk, tk - bk,
                linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
            self.ax.add_patch(rect)

        self.ax.set_xlabel('东向 (km)', color='#aaa', fontsize=8)
        self.ax.set_ylabel('北向 (km)', color='#aaa', fontsize=8)
        self.ax.set_aspect('equal')
        self.ax.tick_params(colors='#888', labelsize=6)
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()


# ============================================================
#  子视图: 轨迹对比 (正方形, 固定比例)
# ============================================================
class TrajectoryView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        title = QLabel("轨迹对比")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(5, 5), dpi=100, facecolor='#2a2a2a')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2a2a2a')
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, stretch=1)
        self._placeholder = _make_placeholder("选择样本后\n显示GT与模型预测轨迹对比")
        layout.addWidget(self._placeholder)
        self.canvas.hide()

    def update_plot(self, sample, predictions, model_visibility):
        self.ax.clear()
        if sample is None:
            self._placeholder.show()
            self.canvas.hide()
            return
        self._placeholder.hide()
        self.canvas.show()

        # 只显示最近30步历史 (已经是相对于观测点的坐标)
        if sample.history_rel is not None and len(sample.history_rel) > 0:
            h = sample.history_rel[-30:]
            self.ax.plot(h[:, 0], h[:, 1], '--', color='#50b4ff', lw=1.8,
                         label='历史轨迹', alpha=0.8)

        # GT 未来轨迹
        if sample.future_rel is not None and len(sample.future_rel) > 0:
            f = sample.future_rel
            self.ax.plot(f[:, 0], f[:, 1], '-', color='white', lw=2.5,
                         label='真实未来', zorder=10)
            self.ax.plot(f[-1, 0], f[-1, 1], '*', color='#ffeb3b', ms=14,
                         zorder=11, markeredgecolor='white', markeredgewidth=0.8)

        # 模型预测轨迹
        for mn, pred in predictions.items():
            if not model_visibility.get(mn, True) or pred is None or len(pred) < 2:
                continue
            color = MODEL_COLORS.get(mn, '#888888')
            self.ax.plot(pred[:, 0], pred[:, 1], '-', color=color,
                         lw=1.8, label=mn, alpha=0.9, zorder=8)
            self.ax.plot(pred[-1, 0], pred[-1, 1], 'o', color=color,
                         ms=6, zorder=9, markeredgecolor='white', markeredgewidth=0.5)

        # 候选终点
        if sample.candidates_rel is not None and len(sample.candidates_rel) > 0:
            c = sample.candidates_rel
            self.ax.scatter(c[:, 0], c[:, 1], c='#ff5252', s=60, marker='D',
                            alpha=0.8, zorder=5, edgecolors='white', linewidths=0.8,
                            label='目标点')

        # 观测原点
        self.ax.plot(0, 0, 's', color='#69f0ae', ms=10, zorder=12,
                     markeredgecolor='white', markeredgewidth=1, label='观测点')

        # 计算正方形范围 (轨迹plot用较小padding)
        cx, cy, half = _compute_traj_bbox(sample, predictions, padding_km=3.0)
        self.ax.set_xlim(cx - half, cx + half)
        self.ax.set_ylim(cy - half, cy + half)
        self.ax.set_aspect('equal', adjustable='box')

        _set_ax_font(self.ax, xlabel='东向 (km)', ylabel='北向 (km)')
        sid = sample.sample_id.replace('_', ' ')[:40]
        _set_ax_font(self.ax, title=sid)
        self.ax.legend(fontsize=7, loc='upper left', facecolor='#333',
                       edgecolor='#555', labelcolor='#ddd', framealpha=0.9)
        self.ax.grid(True, alpha=0.15, color='#666', linestyle='--')
        self.ax.tick_params(colors='#888', labelsize=7)
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw_idle()


# ============================================================
#  子视图: 指标
# ============================================================
class MetricsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { padding: 3px 10px; font-size: 11px; }")
        layout.addWidget(self.tabs)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        # Tab 1: Error curve over time
        self.error_fig = Figure(figsize=(5, 3), dpi=100, facecolor='#2a2a2a')
        self.error_ax = self.error_fig.add_subplot(111)
        self.error_ax.set_facecolor('#2a2a2a')
        self.error_canvas = FigureCanvasQTAgg(self.error_fig)
        self.tabs.addTab(self.error_canvas, "误差曲线")

        # Tab 2: ADE/FDE bar chart
        self.bar_fig = Figure(figsize=(5, 3), dpi=100, facecolor='#2a2a2a')
        self.bar_ax = self.bar_fig.add_subplot(111)
        self.bar_ax.set_facecolor('#2a2a2a')
        self.bar_canvas = FigureCanvasQTAgg(self.bar_fig)
        self.tabs.addTab(self.bar_canvas, "ADE/FDE对比")

        # Tab 3: Sample info
        self.info_label = QLabel("Select a sample to view details")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "font-family: 'Consolas','Courier New',monospace; font-size: 11px;"
            "padding: 10px; color: #ccc; line-height: 1.6;")
        scroll = QScrollArea()
        scroll.setWidget(self.info_label)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, "样本信息")

    def update_error_curve(self, sample, predictions, model_visibility):
        self.error_ax.clear()
        if sample is None or sample.future_rel is None or len(sample.future_rel) == 0:
            self.error_ax.text(0.5, 0.5, '选择样本并运行推理\n查看误差曲线',
                               ha='center', va='center', color='#777',
                               fontsize=11, transform=self.error_ax.transAxes)
            self.error_canvas.draw_idle()
            self._update_bar_chart(sample, predictions, model_visibility)
            return
        gt = sample.future_rel
        has_pred = False
        for mn, pred in predictions.items():
            if not model_visibility.get(mn, True) or pred is None or len(pred) < 2:
                continue
            n = min(len(pred), len(gt))
            err = np.linalg.norm(pred[:n] - gt[:n], axis=1) * 1000
            self.error_ax.plot(range(n), err, '-', color=MODEL_COLORS.get(mn, '#888'),
                               label=mn, lw=1.2)
            has_pred = True
        if has_pred:
            self.error_ax.legend(fontsize=7, facecolor='#333', edgecolor='#555',
                                 labelcolor='#ddd')
        else:
            self.error_ax.text(0.5, 0.5, '点击「运行推理」\n查看预测误差',
                               ha='center', va='center', color='#777',
                               fontsize=11, transform=self.error_ax.transAxes)
        self.error_ax.set_xlabel('时间步 (10s)', color='#aaa', fontsize=9)
        self.error_ax.set_ylabel('误差 (m)', color='#aaa', fontsize=9)
        self.error_ax.tick_params(colors='#888', labelsize=7)
        self.error_ax.grid(True, alpha=0.2, color='#666')
        self.error_fig.tight_layout(pad=0.5)
        self.error_canvas.draw_idle()
        self._update_bar_chart(sample, predictions, model_visibility)

    def _update_bar_chart(self, sample, predictions, model_visibility):
        """ADE/FDE bar chart for quick model comparison"""
        self.bar_ax.clear()
        if sample is None or not predictions:
            self.bar_ax.text(0.5, 0.5, '运行推理后对比模型',
                             ha='center', va='center', color='#777',
                             fontsize=11, transform=self.bar_ax.transAxes)
            self.bar_canvas.draw_idle()
            return
        gt = sample.future_rel
        if gt is None or len(gt) == 0:
            self.bar_canvas.draw_idle()
            return

        names, ades, fdes, colors = [], [], [], []
        for mn, pred in predictions.items():
            if not model_visibility.get(mn, True) or pred is None or len(pred) < 2:
                continue
            n = min(len(pred), len(gt))
            ade = float(np.mean(np.linalg.norm(pred[:n] - gt[:n], axis=1)) * 1000)
            fde = float(np.linalg.norm(pred[-1] - gt[-1]) * 1000)
            names.append(mn)
            ades.append(ade)
            fdes.append(fde)
            colors.append(MODEL_COLORS.get(mn, '#888'))

        if not names:
            self.bar_ax.text(0.5, 0.5, '无可见模型预测',
                             ha='center', va='center', color='#777',
                             fontsize=11, transform=self.bar_ax.transAxes)
            self.bar_canvas.draw_idle()
            return

        x = np.arange(len(names))
        w = 0.35
        self.bar_ax.bar(x - w/2, ades, w, color=colors, alpha=0.8, label='ADE')
        self.bar_ax.bar(x + w/2, fdes, w, color=colors, alpha=0.5, label='FDE',
                        edgecolor='white', linewidth=0.5)
        self.bar_ax.set_xticks(x)
        self.bar_ax.set_xticklabels([n.replace('_', '\n') for n in names],
                                     fontsize=7, color='#aaa', rotation=0)
        self.bar_ax.set_ylabel('误差 (m)', color='#aaa', fontsize=9)
        self.bar_ax.tick_params(colors='#888', labelsize=7)
        self.bar_ax.legend(fontsize=8, facecolor='#333', edgecolor='#555',
                           labelcolor='#ddd', loc='upper right')
        self.bar_ax.grid(True, alpha=0.15, axis='y', color='#666')
        # Add value labels on bars
        for i, (a, f) in enumerate(zip(ades, fdes)):
            self.bar_ax.text(i - w/2, a + 20, f'{a:.0f}', ha='center', va='bottom',
                             fontsize=6, color='#ddd')
            self.bar_ax.text(i + w/2, f + 20, f'{f:.0f}', ha='center', va='bottom',
                             fontsize=6, color='#aaa')
        self.bar_fig.tight_layout(pad=0.5)
        self.bar_canvas.draw_idle()

    def update_info(self, sample):
        if sample is None:
            self.info_label.setText("选择样本查看详细信息")
            return
        # 速度统计
        speed_info = ""
        if sample.history_feat is not None and sample.history_feat.shape[1] >= 2:
            dx = sample.history_feat[:, 0]
            dy = sample.history_feat[:, 1]
            speeds_kmh = np.sqrt(dx**2 + dy**2) * 360
            speed_info = (f"<b>速度(历史):</b> 均值={speeds_kmh.mean():.1f} "
                          f"最大={speeds_kmh.max():.1f} km/h")
        # 环境地图统计
        env_info = ""
        if sample.env_map.ndim == 3:
            ch0_range = f"[{sample.env_map[0].min():.0f}, {sample.env_map[0].max():.0f}]"
            road_pct = (sample.env_map[15] > 0).sum() / sample.env_map[15].size * 100
            goal_max = sample.env_map[17].max()
            env_info = (f"<b>高程范围:</b> {ch0_range} m<br>"
                        f"<b>道路覆盖率:</b> {road_pct:.1f}%<br>"
                        f"<b>目标通道峰值:</b> {goal_max:.3f}")

        lines = [
            f"<b style='font-size:12px'>{sample.sample_id}</b>",
            f"<hr style='border-color:#555'>",
            f"<b>意图:</b> {sample.intent} &nbsp;&nbsp; <b>车辆:</b> {sample.vehicle_type}",
            f"<b>曲折度:</b> {sample.sinuosity:.3f} &nbsp;&nbsp; <b>位移:</b> {sample.total_distance_km:.1f} km",
            f"<b>历史:</b> {len(sample.history_rel)}步 (15分钟) &nbsp;&nbsp; <b>未来:</b> {len(sample.future_rel)}步 (60分钟)",
            speed_info,
            f"<hr style='border-color:#555'>",
            f"<b>目标点(相对):</b> ({sample.goal_rel[0]:.1f}, {sample.goal_rel[1]:.1f}) km",
            f"<b>观测点UTM:</b> E={sample.last_obs_utm[0]:.0f} N={sample.last_obs_utm[1]:.0f}",
            f"<b>环境地图:</b> {sample.env_map.shape if sample.env_map.ndim > 1 else '无'}",
            env_info,
        ]
        self.info_label.setText('<br>'.join(l for l in lines if l))


# ============================================================
#  分析标签页 主体
# ============================================================
class AnalysisTab(QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.setStyleSheet(PANEL_STYLE)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ========== 左侧控制面板 ==========
        left = QWidget()
        left.setFixedWidth(260)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        # --- 1. 区域 + 加载 ---
        rg = QGroupBox("1. 区域选择")
        rgl = QVBoxLayout(rg)
        rgl.setSpacing(4)
        self.region_combo = QComboBox()
        self.region_combo.setPlaceholderText("-- 选择区域 --")
        rgl.addWidget(self.region_combo)
        self.load_btn = QPushButton("加载数据")
        self.load_btn.setStyleSheet(BTN_PRIMARY)
        self.load_btn.setFixedHeight(32)
        self.load_btn.clicked.connect(self._on_load_clicked)
        rgl.addWidget(self.load_btn)
        self.load_progress = QProgressBar()
        self.load_progress.setRange(0, 0)
        self.load_progress.setFixedHeight(4)
        self.load_progress.setTextVisible(False)
        self.load_progress.hide()
        rgl.addWidget(self.load_progress)
        self.region_info = QLabel("选择区域并点击加载")
        self.region_info.setStyleSheet("color: #888; font-size: 11px;")
        self.region_info.setWordWrap(True)
        rgl.addWidget(self.region_info)
        ll.addWidget(rg)

        # --- 2. 样本列表 ---
        sg = QGroupBox("2. 样本列表")
        sgl = QVBoxLayout(sg)
        sgl.setSpacing(3)
        # 搜索 + 排序
        filter_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索ID...")
        self.search_edit.setFixedHeight(24)
        self.search_edit.textChanged.connect(lambda: self.mw.refresh_sample_list())
        filter_row.addWidget(self.search_edit, stretch=2)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(['默认', '曲折↑', '曲折↓', '距离↑', '距离↓'])
        self.sort_combo.setFixedWidth(65)
        self.sort_combo.currentTextChanged.connect(lambda: self.mw.refresh_sample_list())
        filter_row.addWidget(self.sort_combo)
        sgl.addLayout(filter_row)
        # 分组筛选: 意图 + 车辆类型 + 曲率分组
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("意图:"))
        self.intent_combo = QComboBox()
        self.intent_combo.addItem("全部")
        self.intent_combo.setFixedHeight(22)
        self.intent_combo.currentTextChanged.connect(lambda: self.mw.refresh_sample_list())
        group_row.addWidget(self.intent_combo, stretch=1)
        group_row.addWidget(QLabel("车辆:"))
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.addItem("全部")
        self.vehicle_combo.setFixedHeight(22)
        self.vehicle_combo.currentTextChanged.connect(lambda: self.mw.refresh_sample_list())
        group_row.addWidget(self.vehicle_combo, stretch=1)
        sgl.addLayout(group_row)
        # 曲率分组筛选
        curve_row = QHBoxLayout()
        curve_row.addWidget(QLabel("曲率:"))
        self.curve_combo = QComboBox()
        self.curve_combo.addItems(['全部', '直行 (<1.2)', '中等 (1.2-1.8)', '弯曲 (>1.8)'])
        self.curve_combo.setFixedHeight(22)
        self.curve_combo.currentTextChanged.connect(lambda: self.mw.refresh_sample_list())
        curve_row.addWidget(self.curve_combo, stretch=1)
        sgl.addLayout(curve_row)

        self.sample_list = QListWidget()
        self.sample_list.setStyleSheet(
            "QListWidget { font-size: 10px; }"
            "QListWidget::item { padding: 2px 3px; }"
            "QListWidget::item:selected { background: #2979ff; }")
        self.sample_list.currentRowChanged.connect(self._on_sample)
        sgl.addWidget(self.sample_list)
        self.sample_count = QLabel("未加载")
        self.sample_count.setStyleSheet("color: #888; font-size: 11px;")
        sgl.addWidget(self.sample_count)
        ll.addWidget(sg, stretch=1)

        # --- 3. 模型 ---
        mg = QGroupBox("3. 模型选择")
        mgl = QVBoxLayout(mg)
        mgl.setSpacing(2)
        self.model_checks = {}
        default_on = {'V6_Autoreg', 'V7_ConfGate', 'V3_Waypoint'}
        for name, chex in MODEL_COLORS.items():
            cb = QCheckBox(name)
            r, g, b = hex_to_rgb(chex)
            cb.setStyleSheet(f"QCheckBox {{ color: rgb({r},{g},{b}); font-size: 11px; }}")
            cb.setChecked(name in default_on)
            cb.stateChanged.connect(
                lambda st, mn=name: self.mw.set_model_visibility(
                    mn, st == Qt.CheckState.Checked.value))
            mgl.addWidget(cb)
            self.model_checks[name] = cb
        # Phase选择
        phase_row = QHBoxLayout()
        phase_row.addWidget(QLabel("Phase:"))
        self.phase_combo = QComboBox()
        self.phase_combo.addItems([
            'P1a 精确终点(域内)',
            'P1b 精确终点(OOD)',
            'P2a 区域先验(σ=10km)',
            'P3a 无先验(直行)',
        ])
        self.phase_combo.setFixedHeight(22)
        self.phase_combo.setToolTip("选择推理时使用的终点先验条件\nP1=精确终点 P2=模糊区域 P3=无先验")
        self.phase_combo.currentTextChanged.connect(
            lambda t: self.mw.set_phase(t.split()[0]))
        phase_row.addWidget(self.phase_combo, stretch=1)
        mgl.addLayout(phase_row)

        self.infer_btn = QPushButton("运行推理")
        self.infer_btn.setStyleSheet(BTN_SECONDARY)
        self.infer_btn.setFixedHeight(28)
        self.infer_btn.clicked.connect(lambda: self.mw._run_inference())
        mgl.addWidget(self.infer_btn)
        ll.addWidget(mg)

        # --- 快速指标 ---
        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet(
            "font-size: 10px; font-family: monospace; color: #4fc3f7; padding: 4px;")
        ll.addWidget(self.metrics_label)
        layout.addWidget(left)

        # ========== 右侧 2x2 可视化 ==========
        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(3)
        top_row = QSplitter(Qt.Orientation.Horizontal)
        top_row.setHandleWidth(3)
        self.map_view = MapView()
        self.traj_view = TrajectoryView()
        top_row.addWidget(self.map_view)
        top_row.addWidget(self.traj_view)
        top_row.setSizes([500, 500])
        bottom_row = QSplitter(Qt.Orientation.Horizontal)
        bottom_row.setHandleWidth(3)
        self.env_view = EnvChannelView()
        self.metrics_view = MetricsView()
        bottom_row.addWidget(self.env_view)
        bottom_row.addWidget(self.metrics_view)
        bottom_row.setSizes([500, 500])
        right.addWidget(top_row)
        right.addWidget(bottom_row)
        right.setSizes([550, 350])
        layout.addWidget(right, stretch=1)

    # --------------------------------------------------------
    #  操作回调
    # --------------------------------------------------------
    def _on_load_clicked(self):
        region = self.region_combo.currentText()
        if not region:
            self.region_info.setText("请先选择一个区域")
            return
        self.load_btn.setEnabled(False)
        self.load_btn.setText("加载中...")
        self.load_progress.show()
        self.region_info.setText(f"Loading {region}...")
        self.mw.load_region(region)

    def on_load_started(self, msg):
        self.region_info.setText(msg)

    def on_load_finished(self):
        self.load_btn.setEnabled(True)
        self.load_btn.setText("加载数据")
        self.load_progress.hide()


    def populate_regions(self, regions):
        self.region_combo.blockSignals(True)
        self.region_combo.clear()
        if regions:
            self.region_combo.addItems(regions)
        self.region_combo.blockSignals(False)

    def populate_samples(self, samples):
        self.sample_list.blockSignals(True)
        self.sample_list.clear()

        # 收集并填充筛选选项 (只在首次或选项为空时)
        if self.intent_combo.count() <= 1:
            intents = sorted(set(s.intent for s in samples if s.intent != 'unknown'))
            self.intent_combo.blockSignals(True)
            for it in intents:
                self.intent_combo.addItem(it)
            self.intent_combo.blockSignals(False)
        if self.vehicle_combo.count() <= 1:
            vtypes = sorted(set(s.vehicle_type for s in samples if s.vehicle_type != 'unknown'))
            self.vehicle_combo.blockSignals(True)
            for vt in vtypes:
                self.vehicle_combo.addItem(vt)
            self.vehicle_combo.blockSignals(False)

        search = self.search_edit.text().strip().lower()
        sort_mode = self.sort_combo.currentText()
        intent_filter = self.intent_combo.currentText()
        vehicle_filter = self.vehicle_combo.currentText()
        curve_filter = self.curve_combo.currentText()

        filtered = list(samples)
        if search:
            filtered = [s for s in filtered if search in s.sample_id.lower()]
        if intent_filter != '全部':
            filtered = [s for s in filtered if s.intent == intent_filter]
        if vehicle_filter != '全部':
            filtered = [s for s in filtered if s.vehicle_type == vehicle_filter]
        if '直行' in curve_filter:
            filtered = [s for s in filtered if s.sinuosity < 1.2]
        elif '中等' in curve_filter:
            filtered = [s for s in filtered if 1.2 <= s.sinuosity <= 1.8]
        elif '弯曲' in curve_filter:
            filtered = [s for s in filtered if s.sinuosity > 1.8]

        if '曲折↑' in sort_mode:
            filtered.sort(key=lambda s: s.sinuosity)
        elif '曲折↓' in sort_mode:
            filtered.sort(key=lambda s: s.sinuosity, reverse=True)
        elif '距离↑' in sort_mode:
            filtered.sort(key=lambda s: s.total_distance_km)
        elif '距离↓' in sort_mode:
            filtered.sort(key=lambda s: s.total_distance_km, reverse=True)

        for s in filtered:
            txt = f"{s.intent}/{s.vehicle_type} d={s.total_distance_km:.0f}km sin={s.sinuosity:.2f}"
            item = QListWidgetItem(txt)
            item.setData(Qt.ItemDataRole.UserRole, s.sample_id)
            item.setToolTip(s.sample_id)
            self.sample_list.addItem(item)
        self.sample_count.setText(f"{len(filtered)} / {len(samples)} 个样本")
        self.sample_list.blockSignals(False)

    def update_views(self, sample, predictions, model_visibility, region_data):
        if sample is None:
            return

        # 计算轨迹最小外接正方形 + 10km padding
        cx, cy, half = _compute_traj_bbox(sample, predictions, padding_km=10.0)
        coverage_km = half * 2  # 正方形边长 = 2 * half
        coverage_km = min(coverage_km, 200.0)  # 上限200km

        # 地图中心 = 轨迹bbox中心的UTM坐标 (不是观测点)
        obs_e, obs_n = sample.last_obs_utm
        map_center = (obs_e + cx * 1000.0, obs_n + cy * 1000.0)

        # 近期历史 (最后30步)
        recent_history = None
        if sample.history_rel is not None and len(sample.history_rel) > 0:
            recent_history = sample.history_rel[-30:]

        if region_data and sample.last_obs_utm != (0.0, 0.0):
            patches = region_data.extract_patch(map_center, coverage_km, 512)
            self.map_view.set_patches(patches, map_center, coverage_km)

        # 轨迹坐标需要从obs-relative转为map_center-relative
        offset = np.array([[-cx, -cy]])  # km
        map_history = recent_history + offset if recent_history is not None else None
        map_future = sample.future_rel + offset if sample.future_rel is not None else None
        map_candidates = sample.candidates_rel + offset if sample.candidates_rel is not None else None
        map_predictions = {}
        for mn, pred in predictions.items():
            if pred is not None:
                map_predictions[mn] = pred + offset
            else:
                map_predictions[mn] = None

        self.map_view.canvas.set_trajectories(
            history_rel=map_history, future_rel=map_future,
            candidates_rel=map_candidates, predictions=map_predictions)
        self.map_view.canvas.reset_view()

        for mn, vis in model_visibility.items():
            self.map_view.canvas.set_model_visibility(mn, vis)
        self.traj_view.update_plot(sample, predictions, model_visibility)

        # 更新全局环境总览的红色框 (UTM坐标)
        half_m = coverage_km * 1000.0 * 0.5
        local_bbox_utm = (
            map_center[0] - half_m, map_center[1] - half_m,
            map_center[0] + half_m, map_center[1] + half_m,
        )
        self.env_view.update_bbox(local_bbox_utm)
        self.metrics_view.update_error_curve(sample, predictions, model_visibility)
        self.metrics_view.update_info(sample)

    def _on_sample(self, row):
        if row < 0:
            return
        item = self.sample_list.item(row)
        if item:
            self.mw.select_sample(item.data(Qt.ItemDataRole.UserRole))
