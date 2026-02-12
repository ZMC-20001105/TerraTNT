#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面板A: 地图视图 - 环境底图 + 轨迹叠加
支持缩放、平移、多种底图模式
"""
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QCheckBox
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QThread
from PyQt6.QtGui import QPainter, QImage, QPixmap, QColor, QPen, QFont, QWheelEvent, QMouseEvent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from visualization.utils.colors import lulc_to_rgb, slope_colormap, terrain_hillshade, hex_to_rgb, MODEL_COLORS, cost_map_colormap


def _compute_hillshade(dem: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """计算山体阴影 (独立函数, 不依赖RegionData)"""
    d = np.nan_to_num(dem, nan=float(np.nanmean(dem[np.isfinite(dem)])) if np.any(np.isfinite(dem)) else 0)
    gy, gx = np.gradient(d)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((shaded + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


class MapCanvas(QWidget):
    """可缩放/平移的地图画布"""

    coord_hover = pyqtSignal(float, float, str)  # easting, northing, info_text
    candidate_placed = pyqtSignal(float, float)  # rel_km_x, rel_km_y (右键放置候选点)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

        # 底图图像
        self._base_image: QImage = None  # 当前底图
        self._base_array: np.ndarray = None  # (H, W, 3) uint8

        # 视图状态
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._dragging = False
        self._drag_start = QPointF()

        # 地理参考
        self._coverage_km = 140.0
        self._center_utm = (0.0, 0.0)
        self._patch_size = 512

        # 轨迹数据
        self._history_rel = None   # (N, 2) km
        self._future_rel = None    # (T, 2) km
        self._candidates_rel = None  # (C, 2) km
        self._goal_rel = None      # (2,) km — GT终点, 始终显示
        self._predictions = {}     # {model_name: (T, 2) km}
        self._model_visibility = {}  # {model_name: bool}

        # 先验热力图
        self._prior_overlay = None  # (H, W) float [0,1]

        # 路网叠加
        self._road_overlay = None  # (H, W) float, >0 表示有道路
        self._road_graded = None   # (H, W) uint8, 1=high 2=medium 3=low
        self._show_road_overlay = False

    def set_base_image(self, rgb_array: np.ndarray):
        """设置底图 (H, W, 3) uint8"""
        self._base_array = rgb_array
        h, w, _ = rgb_array.shape
        self._base_image = QImage(rgb_array.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._patch_size = w
        self.update()

    def set_trajectories(self, history_rel=None, future_rel=None,
                         candidates_rel=None, predictions=None,
                         goal_rel=None):
        """设置轨迹数据 (相对坐标, km)"""
        self._history_rel = history_rel
        self._future_rel = future_rel
        self._candidates_rel = candidates_rel
        self._goal_rel = goal_rel
        if predictions is not None:
            self._predictions = predictions
        self.update()

    def set_model_visibility(self, model_name: str, visible: bool):
        self._model_visibility[model_name] = visible
        self.update()

    def set_road_overlay(self, road: np.ndarray, road_graded: np.ndarray = None):
        """设置路网叠加数据 (H, W) float, >0 表示有道路"""
        self._road_overlay = road
        self._road_graded = road_graded
        self._road_overlay_cache = None  # 清除缓存, 下次绘制时重建
        self.update()

    def set_show_road_overlay(self, show: bool):
        """切换路网叠加显示"""
        self._show_road_overlay = show
        self.update()

    def set_prior_overlay(self, prior: np.ndarray):
        """设置先验热力图叠加 (H, W) float [0,1]"""
        self._prior_overlay = prior
        self.update()

    def set_geo_reference(self, center_utm, coverage_km):
        self._center_utm = center_utm
        self._coverage_km = coverage_km

    def reset_view(self):
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self.update()

    # --- 坐标转换 ---

    def _km_to_pixel(self, rel_km_x: float, rel_km_y: float) -> QPointF:
        """相对坐标(km) -> 画布像素坐标"""
        half_cov = self._coverage_km / 2.0
        px = (rel_km_x + half_cov) / self._coverage_km * self._patch_size
        py = (half_cov - rel_km_y) / self._coverage_km * self._patch_size  # Y轴翻转
        return QPointF(px, py)

    def _pixel_to_widget(self, px: float, py: float) -> QPointF:
        """底图像素 -> widget坐标 (考虑缩放和平移)"""
        w, h = self.width(), self.height()
        scale = min(w, h) / self._patch_size * self._zoom
        ox = (w - self._patch_size * scale) / 2 + self._pan_offset.x()
        oy = (h - self._patch_size * scale) / 2 + self._pan_offset.y()
        return QPointF(px * scale + ox, py * scale + oy)

    def _widget_to_pixel(self, wx: float, wy: float) -> QPointF:
        """widget坐标 -> 底图像素"""
        w, h = self.width(), self.height()
        scale = min(w, h) / self._patch_size * self._zoom
        ox = (w - self._patch_size * scale) / 2 + self._pan_offset.x()
        oy = (h - self._patch_size * scale) / 2 + self._pan_offset.y()
        return QPointF((wx - ox) / scale, (wy - oy) / scale)

    # --- 绘制 ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._base_image is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "加载区域数据...")
            painter.end()
            return

        # 计算变换
        w, h = self.width(), self.height()
        scale = min(w, h) / self._patch_size * self._zoom
        ox = (w - self._patch_size * scale) / 2 + self._pan_offset.x()
        oy = (h - self._patch_size * scale) / 2 + self._pan_offset.y()

        painter.save()
        painter.translate(ox, oy)
        painter.scale(scale, scale)

        # 绘制底图
        painter.drawImage(0, 0, self._base_image)

        # 绘制路网叠加
        if self._show_road_overlay and self._road_overlay is not None:
            self._draw_road_overlay(painter)

        # 绘制先验叠加
        if self._prior_overlay is not None:
            self._draw_prior_overlay(painter)

        # 绘制轨迹
        self._draw_trajectories(painter, scale)

        painter.restore()
        painter.end()

    def _draw_prior_overlay(self, painter: QPainter):
        """绘制半透明先验热力图"""
        if self._prior_overlay is None:
            return
        prior = self._prior_overlay
        h, w = prior.shape
        # 创建半透明橙色叠加
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        intensity = (np.clip(prior, 0, 1) * 180).astype(np.uint8)
        overlay[:, :, 0] = 255  # R
        overlay[:, :, 1] = 165  # G
        overlay[:, :, 2] = 0    # B
        overlay[:, :, 3] = intensity  # A
        img = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        painter.drawImage(0, 0, img.scaled(self._patch_size, self._patch_size))

    def _draw_road_overlay(self, painter: QPainter):
        """绘制路网叠加 — 仅高/中等级道路, 使用缓存避免重复计算"""
        if not hasattr(self, '_road_overlay_cache') or self._road_overlay_cache is None:
            self._road_overlay_cache = self._build_road_overlay_image()
        if self._road_overlay_cache is not None:
            painter.drawImage(0, 0, self._road_overlay_cache.scaled(
                self._patch_size, self._patch_size))

    def _build_road_overlay_image(self):
        """构建路网叠加QImage — 骨架化+膨胀渲染为连续线条"""
        from skimage.morphology import skeletonize
        from scipy.ndimage import binary_dilation, generate_binary_structure

        graded = self._road_graded
        if graded is None:
            road = self._road_overlay
            if road is None:
                return None
            h, w = road.shape
            mask = road > 0
            if not np.any(mask):
                return None
            # 骨架化 → 膨胀为2px线条
            skel = skeletonize(mask)
            struct = generate_binary_structure(2, 2)  # 8-连通
            line = binary_dilation(skel, structure=struct, iterations=1)
            overlay = np.ascontiguousarray(np.zeros((h, w, 4), dtype=np.uint8))
            overlay[line, 0] = 255
            overlay[line, 1] = 220
            overlay[line, 2] = 50
            overlay[line, 3] = 200
        else:
            h, w = graded.shape
            overlay = np.ascontiguousarray(np.zeros((h, w, 4), dtype=np.uint8))
            struct = generate_binary_structure(2, 2)
            # 高等级道路: 骨架化 → 膨胀3px, 亮黄色
            high_mask = (graded == 1)
            if np.any(high_mask):
                skel_h = skeletonize(high_mask)
                line_h = binary_dilation(skel_h, structure=struct, iterations=2)
                overlay[line_h, 0] = 255
                overlay[line_h, 1] = 235
                overlay[line_h, 2] = 59
                overlay[line_h, 3] = 230
            # 中等级道路: 骨架化 → 膨胀2px, 橙色
            med_mask = (graded == 2)
            if np.any(med_mask):
                skel_m = skeletonize(med_mask)
                line_m = binary_dilation(skel_m, structure=struct, iterations=1)
                # 不覆盖高等级
                new_pixels = line_m & (overlay[:, :, 3] == 0)
                overlay[new_pixels, 0] = 255
                overlay[new_pixels, 1] = 180
                overlay[new_pixels, 2] = 50
                overlay[new_pixels, 3] = 190
            if not np.any(overlay[:, :, 3] > 0):
                return None
        self._road_overlay_data = overlay
        return QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)

    def _draw_trajectories(self, painter: QPainter, view_scale: float):
        """绘制所有轨迹"""
        pen_width = max(1.5, 3.0 / view_scale)

        # 历史轨迹 (蓝色虚线, 高对比度)
        if self._history_rel is not None and len(self._history_rel) > 1:
            pen = QPen(QColor(80, 180, 255, 220))
            pen.setWidthF(pen_width * 1.2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            pts = [self._km_to_pixel(x, y) for x, y in self._history_rel]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        # GT未来轨迹 (白色实线, 高对比度)
        if self._future_rel is not None and len(self._future_rel) > 1:
            # 白色描边
            pen_bg = QPen(QColor(0, 0, 0, 160))
            pen_bg.setWidthF(pen_width * 2.2)
            painter.setPen(pen_bg)
            pts = [self._km_to_pixel(x, y) for x, y in self._future_rel]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            # 白色前景
            pen = QPen(QColor(255, 255, 255, 240))
            pen.setWidthF(pen_width * 1.5)
            painter.setPen(pen)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            # GT终点标记 (黄色星)
            ep = pts[-1]
            painter.setBrush(QColor(255, 235, 59))
            painter.setPen(QPen(QColor(255, 255, 255), 1.0 / view_scale))
            painter.drawEllipse(ep, 6 / view_scale, 6 / view_scale)

        # 模型预测轨迹
        for model_name, pred_rel in self._predictions.items():
            if not self._model_visibility.get(model_name, True):
                continue
            if pred_rel is None or len(pred_rel) < 2:
                continue
            color_hex = MODEL_COLORS.get(model_name, '#ffffff')
            r, g, b = hex_to_rgb(color_hex)
            pen = QPen(QColor(r, g, b, 200))
            pen.setWidthF(pen_width)
            painter.setPen(pen)
            pts = [self._km_to_pixel(x, y) for x, y in pred_rel]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        # 候选终点
        if self._candidates_rel is not None:
            for i, (cx, cy) in enumerate(self._candidates_rel):
                pt = self._km_to_pixel(cx, cy)
                painter.setBrush(QColor(255, 100, 100, 180))
                painter.setPen(QPen(QColor(200, 0, 0), 1.0 / view_scale))
                r = 4.0 / view_scale
                painter.drawEllipse(pt, r, r)

        # GT终点标记 (黄色星, 始终可见)
        if self._goal_rel is not None:
            gp = self._km_to_pixel(self._goal_rel[0], self._goal_rel[1])
            # 外圈光晕
            painter.setBrush(QColor(255, 235, 59, 60))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(gp, 12 / view_scale, 12 / view_scale)
            # 实心星
            painter.setBrush(QColor(255, 235, 59))
            painter.setPen(QPen(QColor(255, 255, 255), 1.2 / view_scale))
            painter.drawEllipse(gp, 5 / view_scale, 5 / view_scale)
            # 标签
            painter.setPen(QColor(255, 235, 59, 200))
            font = QFont("sans-serif", max(7, int(9 / view_scale)))
            painter.setFont(font)
            painter.drawText(int(gp.x() + 8 / view_scale), int(gp.y() - 4 / view_scale), "GT")

        # 观测原点 (绿色方块, 带白色边框)
        origin = self._km_to_pixel(0, 0)
        painter.setBrush(QColor(105, 240, 174))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5 / view_scale))
        s = 6.0 / view_scale
        painter.drawRect(int(origin.x() - s), int(origin.y() - s), int(s * 2), int(s * 2))

        # 绘制图例
        self._draw_legend(painter, view_scale)

    def _draw_legend(self, painter: QPainter, view_scale: float):
        """在左下角绘制紧凑半透明图例, 不遮挡轨迹"""
        entries = []
        if self._history_rel is not None and len(self._history_rel) > 1:
            entries.append(('历史', QColor(80, 180, 255), True))
        if self._goal_rel is not None:
            entries.append(('GT终点', QColor(255, 235, 59), False))
        if self._future_rel is not None and len(self._future_rel) > 1:
            entries.append(('GT轨迹', QColor(255, 255, 255), False))
        for mn, pred in self._predictions.items():
            if not self._model_visibility.get(mn, True) or pred is None or len(pred) < 2:
                continue
            c = MODEL_COLORS.get(mn, '#888')
            r, g, b = hex_to_rgb(c)
            entries.append((mn, QColor(r, g, b), False))
        if self._candidates_rel is not None and len(self._candidates_rel) > 0:
            entries.append(('候选', QColor(255, 100, 100), False))

        if not entries:
            return

        fs = max(6, int(9 / view_scale))
        font = QFont("sans-serif", fs)
        painter.setFont(font)
        lh = fs + 3
        lw = max(len(e[0]) for e in entries) * fs * 0.65 + 22
        lh_total = len(entries) * lh + 4
        # 左下角
        lx = 4
        ly = self._patch_size - lh_total - 4

        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(int(lx), int(ly), int(lw), int(lh_total), 3, 3)

        for i, (label, color, dashed) in enumerate(entries):
            y = ly + 2 + i * lh + lh // 2
            pen = QPen(color, max(1.0, 1.5 / view_scale))
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(lx + 3), int(y), int(lx + 14), int(y))
            painter.setPen(QColor(200, 200, 200, 180))
            painter.drawText(int(lx + 17), int(y + fs // 3), label)

    # --- 交互 ---

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom = max(0.2, min(20.0, self._zoom * factor))
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.position()
        elif event.button() == Qt.MouseButton.RightButton:
            # 右键放置候选目标点
            pix = self._widget_to_pixel(event.position().x(), event.position().y())
            if 0 <= pix.x() < self._patch_size and 0 <= pix.y() < self._patch_size:
                rel_x = (pix.x() / self._patch_size - 0.5) * self._coverage_km
                rel_y = (0.5 - pix.y() / self._patch_size) * self._coverage_km
                self.candidate_placed.emit(rel_x, rel_y)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.position() - self._drag_start
            self._pan_offset += delta
            self._drag_start = event.position()
            self.update()
        else:
            # 悬停信息
            pix = self._widget_to_pixel(event.position().x(), event.position().y())
            if 0 <= pix.x() < self._patch_size and 0 <= pix.y() < self._patch_size:
                rel_x = (pix.x() / self._patch_size - 0.5) * self._coverage_km
                rel_y = (0.5 - pix.y() / self._patch_size) * self._coverage_km
                utm_e = self._center_utm[0] + rel_x * 1000
                utm_n = self._center_utm[1] + rel_y * 1000
                info = f"E={utm_e:.0f} N={utm_n:.0f} ({rel_x:.1f}km, {rel_y:.1f}km)"
                self.coord_hover.emit(utm_e, utm_n, info)


class MapView(QWidget):
    """地图视图面板 (底图选择 + MapCanvas)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # 底图选择器
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("底图:"))
        self.basemap_combo = QComboBox()
        self.basemap_combo.addItems(['地形晕渲', '坡度', '土地覆盖', '道路', '代价地图',
                                      '卫星影像', '街道地图', '地形地图'])
        self.basemap_combo.currentTextChanged.connect(self._on_basemap_changed)
        top_bar.addWidget(self.basemap_combo)

        self.road_overlay_cb = QCheckBox("叠加路网")
        self.road_overlay_cb.setStyleSheet("color: #ffd740; font-size: 11px;")
        self.road_overlay_cb.setChecked(False)
        self.road_overlay_cb.stateChanged.connect(self._on_road_overlay_toggled)
        top_bar.addWidget(self.road_overlay_cb)

        top_bar.addStretch()

        self.coord_label = QLabel("坐标: -")
        self.coord_label.setStyleSheet("color: #aaa; font-size: 11px;")
        top_bar.addWidget(self.coord_label)

        layout.addLayout(top_bar)

        # 地图画布
        self.canvas = MapCanvas()
        self.canvas.coord_hover.connect(self._on_coord_hover)
        layout.addWidget(self.canvas)

        # 缓存的patch数据
        self._patches = {}
        self._region_data = None
        self._center_utm = None
        self._coverage_km = None
        self._tile_worker = None  # 瓦片下载线程

    def set_region_data(self, region_data):
        """设置区域数据引用"""
        self._region_data = region_data

    def set_patches(self, patches: dict, center_utm, coverage_km=140.0):
        """设置环境patch数据"""
        self._patches = patches
        self._center_utm = center_utm
        self._coverage_km = coverage_km
        self.canvas.set_geo_reference(center_utm, coverage_km)
        # 更新路网叠加数据
        road = patches.get('road')
        road_graded = patches.get('road_graded')
        if road is not None:
            self.canvas.set_road_overlay(road, road_graded)
        self._update_basemap()

    def _update_basemap(self):
        mode = self.basemap_combo.currentText()
        if not self._patches:
            return

        if mode == '地形晕渲':
            dem = self._patches.get('dem')
            if dem is not None:
                hs = _compute_hillshade(dem)
                rgb = terrain_hillshade(dem, hs)
                self.canvas.set_base_image(rgb)
        elif mode == '坡度':
            slope = self._patches.get('slope')
            if slope is not None:
                rgb = slope_colormap(slope)
                self.canvas.set_base_image(rgb)
        elif mode == '土地覆盖':
            lulc = self._patches.get('lulc')
            if lulc is not None:
                rgb = lulc_to_rgb(lulc.astype(np.uint8))
                self.canvas.set_base_image(rgb)
        elif mode == '道路':
            graded = self._patches.get('road_graded')
            road = self._patches.get('road')
            from skimage.morphology import skeletonize
            from scipy.ndimage import binary_dilation, generate_binary_structure
            struct = generate_binary_structure(2, 2)
            if graded is not None:
                rgb = np.full((*graded.shape, 3), 30, dtype=np.uint8)
                # 高等级: 骨架化+膨胀3px
                hm = (graded == 1)
                if np.any(hm):
                    lh = binary_dilation(skeletonize(hm), structure=struct, iterations=2)
                    rgb[lh] = [255, 235, 59]
                # 中等级: 骨架化+膨胀2px
                mm = (graded == 2)
                if np.any(mm):
                    lm = binary_dilation(skeletonize(mm), structure=struct, iterations=1)
                    new = lm & (rgb[:,:,0] == 30)
                    rgb[new] = [255, 180, 50]
                # 低等级: 骨架化+1px, 暗色
                lo = (graded == 3)
                if np.any(lo):
                    ll = skeletonize(lo)
                    new = ll & (rgb[:,:,0] == 30)
                    rgb[new] = [80, 70, 50]
                self.canvas.set_base_image(rgb)
            elif road is not None:
                rgb = np.full((*road.shape, 3), 30, dtype=np.uint8)
                mask = road > 0
                if np.any(mask):
                    line = binary_dilation(skeletonize(mask), structure=struct, iterations=1)
                    rgb[line] = [255, 200, 50]
                self.canvas.set_base_image(rgb)
        elif mode == '代价地图':
            slope = self._patches.get('slope')
            lulc = self._patches.get('lulc')
            road = self._patches.get('road')
            if slope is not None and lulc is not None:
                rgb = cost_map_colormap(slope, lulc.astype(np.uint8), road)
                self.canvas.set_base_image(rgb)
        elif mode in ('卫星影像', '街道地图', '地形地图'):
            self._fetch_tile_basemap(mode)

    def _fetch_tile_basemap(self, source_name: str):
        """后台下载瓦片底图"""
        if self._center_utm is None or self._coverage_km is None:
            return
        crs = self._region_data.crs if self._region_data else None
        if crs is None:
            return
        self.coord_label.setText(f"正在加载{source_name}...")
        # 后台线程
        self._tile_worker = _TileFetchWorker(
            self._center_utm, self._coverage_km, 512, crs, source_name)
        self._tile_worker.done.connect(self._on_tile_fetched)
        self._tile_worker.start()

    def _on_tile_fetched(self, rgb_list):
        """瓦片下载完成回调"""
        if rgb_list is not None and len(rgb_list) > 0:
            # rgb_list is a list wrapping the numpy array (signal can't send ndarray directly)
            self.canvas.set_base_image(rgb_list[0])
            self.coord_label.setText("坐标: -")
        else:
            self.coord_label.setText("瓦片加载失败 (网络问题?)")

    def _on_basemap_changed(self, text):
        self._update_basemap()

    def _on_road_overlay_toggled(self, state):
        self.canvas.set_show_road_overlay(state == Qt.CheckState.Checked.value)

    def _on_coord_hover(self, easting, northing, info):
        self.coord_label.setText(f"坐标: {info}")


class _TileFetchWorker(QThread):
    """后台瓦片下载线程"""
    done = pyqtSignal(list)  # [np.ndarray] or []

    def __init__(self, center_utm, coverage_km, out_size, crs, source_name):
        super().__init__()
        self.center_utm = center_utm
        self.coverage_km = coverage_km
        self.out_size = out_size
        self.crs = crs
        self.source_name = source_name

    def run(self):
        try:
            from visualization.utils.tile_fetcher import fetch_satellite_image
            rgb = fetch_satellite_image(
                self.center_utm, self.coverage_km, self.out_size,
                self.crs, self.source_name)
            if rgb is not None:
                self.done.emit([rgb])
            else:
                self.done.emit([])
        except Exception as e:
            print(f"瓦片获取失败: {e}")
            self.done.emit([])
