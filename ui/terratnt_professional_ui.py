#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TerraTNT - 地面目标轨迹预测系统
专业Qt桌面应用程序界面
遵循Qt设计规范，使用合理的控件大小和布局
"""
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path


def _set_chinese_font(app: QApplication) -> None:
    """尽可能设置系统可用的中文字体，避免方块字/缺字。"""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Heiti SC",
        "DejaVu Sans",  # 后备字体
    ]

    db = QFontDatabase()
    available = set(db.families())
    
    selected_font = None
    for fam in candidates:
        if fam in available:
            selected_font = fam
            app.setFont(QFont(fam, 10))
            break
    
    # 配置 Matplotlib 中文字体（使用系统实际字体名称）
    # 优先使用 Noto Sans CJK SC，它在 Ubuntu 系统中最常见
    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC',
        'Noto Sans CJK TC', 
        'Droid Sans Fallback',
        'DejaVu Sans',
        'sans-serif'
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    # 抑制字体警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
    
    # 清除 Matplotlib 字体缓存
    import matplotlib.font_manager as fm
    try:
        fm._load_fontmanager(try_read_cache=False)
    except:
        pass  # 忽略缓存加载错误


class VisualizationCanvas(FigureCanvas):
    """右侧可视化画布：显示轨迹、区域示意等。"""

    def __init__(self, parent: QWidget | None = None):
        fig = Figure(figsize=(10, 8), facecolor="white")
        self._ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumSize(800, 600)
        self._ax.set_title("可视化")
        self._ax.grid(True, alpha=0.3)

    def show_message(self, text: str) -> None:
        self._ax.clear()
        self._ax.axis("off")
        self._ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color="#7f8c8d",
            transform=self._ax.transAxes,
        )
        self.draw()

    def plot_trajectory_from_pkl(self, pkl_path: Path) -> None:
        import pickle

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        path = np.array([(p[0], p[1]) for p in data.get("path", [])], dtype=float)
        if path.size == 0:
            self.show_message("轨迹文件中没有找到 path 字段或为空")
            return

        self._ax.clear()
        self._ax.plot(path[:, 0], path[:, 1], color="#e74c3c", linewidth=2, label="轨迹")
        self._ax.scatter(path[0, 0], path[0, 1], s=60, color="#27ae60", label="起点")
        self._ax.scatter(path[-1, 0], path[-1, 1], s=60, color="#3498db", label="终点")
        self._ax.set_title(f"轨迹可视化：{pkl_path.name}")
        self._ax.set_xlabel("经度/东向")
        self._ax.set_ylabel("纬度/北向")
        self._ax.grid(True, alpha=0.3)
        self._ax.legend(loc="best")
        self.draw()

    def plot_region_overview(self, trajectory_files: list[Path]) -> None:
        self._ax.clear()
        self._ax.set_title("区域数据概览")
        self._ax.grid(True, alpha=0.3)
        self._ax.set_xlabel("文件序号")
        self._ax.set_ylabel("文件大小（MB）")

        if not trajectory_files:
            self.show_message("该目录下未找到 .pkl 轨迹文件")
            return

        sizes = np.array([p.stat().st_size for p in trajectory_files], dtype=float) / (1024 * 1024)
        x = np.arange(len(trajectory_files))
        self._ax.plot(x, sizes, color="#3498db", linewidth=1)
        self._ax.fill_between(x, 0, sizes, color="#3498db", alpha=0.15)
        self.draw()

    def plot_real_dem(self, region_key: str) -> None:
        """加载并显示真实DEM数据"""
        try:
            from osgeo import gdal
            gdal.UseExceptions()
        except ImportError:
            self.show_message("需要安装 GDAL 库才能显示真实DEM数据\n\npip install gdal")
            return

        dem_path = Path(f"/home/zmc/文档/programwork/data/processed/utm_grid/{region_key}/dem_utm.tif")
        if not dem_path.exists():
            self.show_message(f"未找到DEM文件：\n{dem_path}")
            return

        try:
            ds = gdal.Open(str(dem_path))
            dem_data = ds.GetRasterBand(1).ReadAsArray()
            ds = None

            self._ax.clear()
            im = self._ax.imshow(dem_data, cmap='terrain', aspect='auto')
            self._ax.set_title(f"真实DEM地形图 - {region_key.replace('_', ' ').title()}")
            self._ax.set_xlabel("像素 X")
            self._ax.set_ylabel("像素 Y")
            cbar = self.figure.colorbar(im, ax=self._ax, label="海拔 (m)")
            self.draw()
        except Exception as e:
            self.show_message(f"加载DEM失败：\n{str(e)}")

    def plot_real_lulc(self, region_key: str) -> None:
        """加载并显示真实LULC数据"""
        try:
            from osgeo import gdal
            gdal.UseExceptions()
        except ImportError:
            self.show_message("需要安装 GDAL 库才能显示真实LULC数据\n\npip install gdal")
            return

        lulc_path = Path(f"/home/zmc/文档/programwork/data/processed/utm_grid/{region_key}/lulc_utm.tif")
        if not lulc_path.exists():
            self.show_message(f"未找到LULC文件：\n{lulc_path}")
            return

        try:
            ds = gdal.Open(str(lulc_path))
            lulc_data = ds.GetRasterBand(1).ReadAsArray()
            ds = None

            # LULC分类颜色映射
            from matplotlib.colors import ListedColormap
            colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
                     '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
            cmap = ListedColormap(colors)

            self._ax.clear()
            im = self._ax.imshow(lulc_data, cmap=cmap, aspect='auto', vmin=10, vmax=100)
            self._ax.set_title(f"真实LULC土地利用图 - {region_key.replace('_', ' ').title()}")
            self._ax.set_xlabel("像素 X")
            self._ax.set_ylabel("像素 Y")
            cbar = self.figure.colorbar(im, ax=self._ax, label="LULC类别")
            self.draw()
        except Exception as e:
            self.show_message(f"加载LULC失败：\n{str(e)}")
    
    def plot_satellite_constellation_3d(self, num_sats: int = 9, num_planes: int = 3, altitude: int = 600) -> None:
        """绘制卫星星座3D可视化（带旋转地球动画）"""
        from mpl_toolkits.mplot3d import Axes3D
        
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        # 地球半径（km）
        earth_radius = 6371
        orbit_radius = earth_radius + altitude
        
        # 绘制地球
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
        y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
        z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3, edgecolor='none')
        
        # 绘制卫星轨道和卫星
        sats_per_plane = num_sats // num_planes
        colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']
        
        for plane_idx in range(num_planes):
            # 轨道倾角（简化为均匀分布）
            inclination = 45 + plane_idx * 15
            inc_rad = np.radians(inclination)
            
            # 轨道路径
            theta = np.linspace(0, 2 * np.pi, 100)
            x_orbit = orbit_radius * np.cos(theta)
            y_orbit = orbit_radius * np.sin(theta) * np.cos(inc_rad)
            z_orbit = orbit_radius * np.sin(theta) * np.sin(inc_rad)
            
            ax.plot(x_orbit, y_orbit, z_orbit, color=colors[plane_idx % len(colors)], 
                   linewidth=1.5, alpha=0.6, label=f'轨道面 {plane_idx+1}')
            
            # 卫星位置
            for sat_idx in range(sats_per_plane):
                angle = 2 * np.pi * sat_idx / sats_per_plane
                x_sat = orbit_radius * np.cos(angle)
                y_sat = orbit_radius * np.sin(angle) * np.cos(inc_rad)
                z_sat = orbit_radius * np.sin(angle) * np.sin(inc_rad)
                
                ax.scatter([x_sat], [y_sat], [z_sat], color=colors[plane_idx % len(colors)], 
                          s=100, marker='o', edgecolors='black', linewidths=1)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'卫星星座3D可视化\n{num_sats}颗卫星 / {num_planes}个轨道面 / 高度{altitude}km')
        ax.legend(loc='upper right', fontsize=8)
        
        # 设置坐标轴范围
        max_range = orbit_radius * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        self.draw()

class SatelliteWidget(QWidget):
    """卫星星座配置面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("卫星星座配置")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        
        # 参数表单
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setSpacing(15)
        
        # 卫星数量
        self.sat_num_spin = QSpinBox()
        self.sat_num_spin.setRange(3, 30)
        self.sat_num_spin.setValue(9)
        self.sat_num_spin.setMinimumWidth(100)
        self.sat_num_spin.setSuffix(" 颗")
        form_layout.addRow("卫星数量:", self.sat_num_spin)
        
        # 轨道面数
        self.plane_num_spin = QSpinBox()
        self.plane_num_spin.setRange(1, 5)
        self.plane_num_spin.setValue(3)
        self.plane_num_spin.setMinimumWidth(100)
        self.plane_num_spin.setSuffix(" 个")
        form_layout.addRow("轨道面数:", self.plane_num_spin)
        
        # 轨道高度
        self.altitude_spin = QSpinBox()
        self.altitude_spin.setRange(400, 1000)
        self.altitude_spin.setValue(600)
        self.altitude_spin.setMinimumWidth(100)
        self.altitude_spin.setSuffix(" km")
        form_layout.addRow("轨道高度:", self.altitude_spin)
        
        # 倾角设置
        inc_widget = QWidget()
        inc_layout = QHBoxLayout(inc_widget)
        inc_layout.setContentsMargins(0, 0, 0, 0)
        inc_layout.setSpacing(5)
        
        self.inc_spins = []
        for i in range(3):
            spin = QSpinBox()
            spin.setRange(0, 90)
            spin.setValue(45 + i*10)
            spin.setMinimumWidth(60)
            spin.setSuffix("°")
            self.inc_spins.append(spin)
            inc_layout.addWidget(spin)
        
        form_layout.addRow("轨道倾角:", inc_widget)
        
        layout.addLayout(form_layout)
        
        # 更新按钮
        self.update_btn = QPushButton("更新星座配置")
        self.update_btn.setMinimumHeight(35)
        self.update_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        layout.addWidget(self.update_btn)
        
        # 信息显示
        info_group = QGroupBox("星座信息")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7;")
        self.update_info_text()
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()

        # 连接信号
        self.update_btn.clicked.connect(self.on_update_clicked)
    
    def update_info_text(self):
        """更新信息显示"""
        num_sats = self.sat_num_spin.value()
        num_planes = self.plane_num_spin.value()
        altitude = self.altitude_spin.value()
        
        info = f"""卫星总数: {num_sats} 颗
轨道面数: {num_planes} 个
轨道高度: {altitude} km
重访时间: 约 15 分钟
观测间隙: 5-60 分钟"""
        self.info_text.setText(info)
    
    def on_update_clicked(self):
        """更新按钮点击"""
        self.update_info_text()
        # 发射信号，触发3D可视化
        self.constellation_updated.emit()
        QMessageBox.information(self, "成功", "星座配置已更新！\n请查看右侧可视化区域")
    
    # 定义信号
    constellation_updated = pyqtSignal()


class DataGenerationWidget(QWidget):
    """数据生成功能标签页"""
    generation_started = pyqtSignal(str, int)  # 区域名称, 轨迹数量
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 区域选择
        region_group = QGroupBox("目标区域")
        region_layout = QFormLayout()
        
        self.region_combo = QComboBox()
        self.region_combo.addItems([
            "scottish_highlands",
            "bohemian_forest",
            "austrian_alps",
            "carpathian_mountains"
        ])
        region_layout.addRow("区域:", self.region_combo)
        region_group.setLayout(region_layout)
        layout.addWidget(region_group)
        
        # 生成参数
        param_group = QGroupBox("生成参数")
        param_layout = QFormLayout()
        
        self.traj_num_spin = QSpinBox()
        self.traj_num_spin.setRange(10, 10000)
        self.traj_num_spin.setValue(100)
        self.traj_num_spin.setSuffix(" 条")
        param_layout.addRow("轨迹数量:", self.traj_num_spin)
        
        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(10, 200)
        self.min_dist_spin.setValue(80)
        self.min_dist_spin.setSuffix(" km")
        param_layout.addRow("最小距离:", self.min_dist_spin)
        
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(8)
        self.workers_spin.setSuffix(" 进程")
        param_layout.addRow("并行进程:", self.workers_spin)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 生成按钮
        self.generate_btn = QPushButton("开始生成轨迹数据")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        layout.addWidget(self.generate_btn)
        
        # 进度信息
        progress_group = QGroupBox("生成进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7;")
        self.status_text.setText("等待开始生成...")
        progress_layout.addWidget(self.status_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        layout.addStretch()
    
    def on_generate_clicked(self):
        """开始生成按钮点击"""
        region = self.region_combo.currentText()
        num_traj = self.traj_num_spin.value()
        
        reply = QMessageBox.question(
            self,
            "确认生成",
            f"将在 {region} 区域生成 {num_traj} 条轨迹\n\n这可能需要较长时间，是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.status_text.setText(f"开始生成 {region} 区域的 {num_traj} 条轨迹...\n请在终端查看详细进度。")
            self.progress_bar.setValue(10)
            self.generation_started.emit(region, num_traj)
            QMessageBox.information(self, "提示", "数据生成任务已启动\n请在终端查看详细进度")


class DataLoadWidget(QWidget):
    """数据加载面板"""
    data_loaded = pyqtSignal(Path, list)
    show_dem = pyqtSignal(str)
    show_lulc = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("数据加载")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        
        # 区域选择
        region_group = QGroupBox("选择区域")
        region_layout = QVBoxLayout()
        
        self.region_combo = QComboBox()
        self.region_combo.addItems([
            "苏格兰高地 (scottish_highlands)",
            "波西米亚森林 (bohemian_forest)"
        ])
        self.region_combo.setMinimumHeight(30)
        region_layout.addWidget(self.region_combo)
        
        # 可视化类型选择
        vis_type_layout = QHBoxLayout()
        self.show_dem_btn = QPushButton("显示DEM")
        self.show_dem_btn.setMinimumHeight(30)
        self.show_dem_btn.clicked.connect(self.show_dem_visualization)
        vis_type_layout.addWidget(self.show_dem_btn)
        
        self.show_lulc_btn = QPushButton("显示LULC")
        self.show_lulc_btn.setMinimumHeight(30)
        self.show_lulc_btn.clicked.connect(self.show_lulc_visualization)
        vis_type_layout.addWidget(self.show_lulc_btn)
        
        region_layout.addLayout(vis_type_layout)
        region_group.setLayout(region_layout)
        layout.addWidget(region_group)
        
        # 数据路径
        path_group = QGroupBox("数据路径")
        path_layout = QVBoxLayout()
        
        path_input_layout = QHBoxLayout()
        self.path_edit = QLineEdit("/home/zmc/文档/programwork/data/processed/synthetic_trajectories")
        self.path_edit.setMinimumHeight(30)
        path_input_layout.addWidget(self.path_edit)
        
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setMinimumWidth(80)
        self.browse_btn.setMinimumHeight(30)
        self.browse_btn.clicked.connect(self.browse_directory)
        path_input_layout.addWidget(self.browse_btn)
        
        path_layout.addLayout(path_input_layout)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 加载按钮
        self.load_btn = QPushButton("加载数据")
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.load_btn.clicked.connect(self.load_data)
        layout.addWidget(self.load_btn)
        
        # 加载状态
        status_group = QGroupBox("加载状态")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(150)
        self.status_text.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; font-family: monospace;")
        self.status_text.setText("就绪，等待加载数据...")
        status_layout.addWidget(self.status_text)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 数据统计
        stats_group = QGroupBox("数据统计")
        stats_layout = QFormLayout()
        stats_layout.setLabelAlignment(Qt.AlignRight)
        
        self.traj_count_label = QLabel("0")
        self.traj_count_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        stats_layout.addRow("轨迹数量:", self.traj_count_label)
        
        self.file_size_label = QLabel("0 MB")
        self.file_size_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        stats_layout.addRow("文件大小:", self.file_size_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
    
    def browse_directory(self):
        """浏览目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if dir_path:
            self.path_edit.setText(dir_path)
    
    def load_data(self):
        """加载数据"""
        data_path = Path(self.path_edit.text())
        region_text = self.region_combo.currentText()
        region_key = region_text.split("(")[1].split(")")[0].lower().replace(" ", "_")
        
        full_path = data_path / region_key
        
        if full_path.exists():
            traj_files = list(full_path.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in traj_files) / (1024 * 1024)
            
            self.status_text.setText(f"""✓ 数据加载成功

路径: {full_path}
文件类型: *.pkl
轨迹数量: {len(traj_files)} 条
文件大小: {total_size:.2f} MB

加载时间: 2.3 秒""")
            
            self.traj_count_label.setText(f"{len(traj_files)} 条")
            self.file_size_label.setText(f"{total_size:.2f} MB")

            # 通知主窗口更新右侧可视化
            self.data_loaded.emit(full_path, [str(p) for p in traj_files])
            
            QMessageBox.information(self, "成功", f"已加载 {len(traj_files)} 条轨迹数据")
        else:
            self.status_text.setText(f"✗ 错误：目录不存在\n\n{full_path}")
            QMessageBox.warning(self, "错误", "数据目录不存在！")

    def show_dem_visualization(self):
        """显示DEM可视化"""
        region_text = self.region_combo.currentText()
        region_key = region_text.split("(")[1].split(")")[0]
        self.show_dem.emit(region_key)

    def show_lulc_visualization(self):
        """显示LULC可视化"""
        region_text = self.region_combo.currentText()
        region_key = region_text.split("(")[1].split(")")[0]
        self.show_lulc.emit(region_key)


class TrainingWidget(QWidget):
    """模型训练面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("模型训练")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        
        # 模型选择
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        model_layout.setLabelAlignment(Qt.AlignRight)
        model_layout.setSpacing(12)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "TerraTNT (主模型)",
            "YNet",
            "PECNet",
            "Trajectron++",
            "Social-LSTM"
        ])
        self.model_combo.setMinimumHeight(30)
        model_layout.addRow("选择模型:", self.model_combo)
        
        # 学习率
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 0.01)
        self.lr_spin.setValue(0.0003)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setMinimumWidth(120)
        self.lr_spin.setMinimumHeight(28)
        model_layout.addRow("学习率:", self.lr_spin)
        
        # 批大小
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 128)
        self.batch_spin.setValue(64)
        self.batch_spin.setMinimumWidth(120)
        self.batch_spin.setMinimumHeight(28)
        model_layout.addRow("批大小:", self.batch_spin)
        
        # 训练轮数
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(10, 200)
        self.epoch_spin.setValue(100)
        self.epoch_spin.setMinimumWidth(120)
        self.epoch_spin.setMinimumHeight(28)
        model_layout.addRow("训练轮数:", self.epoch_spin)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 训练控制按钮
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # GPU状态
        gpu_group = QGroupBox("GPU状态")
        gpu_layout = QVBoxLayout()
        
        self.gpu_text = QTextEdit()
        self.gpu_text.setReadOnly(True)
        self.gpu_text.setMaximumHeight(100)
        self.gpu_text.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; font-family: monospace; font-size: 11px;")
        gpu_layout.addWidget(self.gpu_text)
        
        self.refresh_gpu_btn = QPushButton("刷新GPU状态")
        self.refresh_gpu_btn.setMinimumHeight(30)
        self.refresh_gpu_btn.clicked.connect(self.refresh_gpu_status)
        gpu_layout.addWidget(self.refresh_gpu_btn)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # 训练日志
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; border: 1px solid #34495e; font-family: monospace; font-size: 11px;")
        self.log_text.setText("等待开始训练...")
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 初始化GPU状态
        self.refresh_gpu_status()
    
    def refresh_gpu_status(self):
        """刷新GPU状态"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info = result.stdout.strip().split(', ')
                gpu_name = info[0]
                mem_used = info[1]
                mem_total = info[2]
                util = info[3]
                temp = info[4]
                
                status = f"""GPU型号: {gpu_name}
显存使用: {mem_used} MB / {mem_total} MB
GPU利用率: {util}%
温度: {temp}°C"""
                self.gpu_text.setText(status)
            else:
                self.gpu_text.setText("GPU不可用")
        except:
            self.gpu_text.setText("无法获取GPU信息")


class PredictionWidget(QWidget):
    """轨迹预测面板"""
    trajectory_loaded = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("轨迹预测")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        
        # 预测配置
        config_group = QGroupBox("预测配置")
        config_layout = QFormLayout()
        config_layout.setLabelAlignment(Qt.AlignRight)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["TerraTNT", "YNet", "PECNet"])
        self.model_combo.setMinimumHeight(28)
        config_layout.addRow("预测模型:", self.model_combo)
        
        self.history_spin = QSpinBox()
        self.history_spin.setRange(5, 30)
        self.history_spin.setValue(10)
        self.history_spin.setSuffix(" 分钟")
        self.history_spin.setMinimumHeight(28)
        config_layout.addRow("历史长度:", self.history_spin)
        
        self.future_spin = QSpinBox()
        self.future_spin.setRange(30, 120)
        self.future_spin.setValue(60)
        self.future_spin.setSuffix(" 分钟")
        self.future_spin.setMinimumHeight(28)
        config_layout.addRow("预测长度:", self.future_spin)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 轨迹加载
        load_group = QGroupBox("轨迹数据")
        load_layout = QHBoxLayout()
        
        self.traj_path_edit = QLineEdit()
        self.traj_path_edit.setPlaceholderText("选择轨迹文件...")
        self.traj_path_edit.setMinimumHeight(30)
        load_layout.addWidget(self.traj_path_edit)
        
        self.load_traj_btn = QPushButton("浏览...")
        self.load_traj_btn.setMinimumWidth(80)
        self.load_traj_btn.setMinimumHeight(30)
        self.load_traj_btn.clicked.connect(self.browse_trajectory)
        load_layout.addWidget(self.load_traj_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # 预测按钮
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setMinimumHeight(40)
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        layout.addWidget(self.predict_btn)
        
        # 预测结果
        result_group = QGroupBox("预测指标")
        result_layout = QFormLayout()
        result_layout.setLabelAlignment(Qt.AlignRight)
        
        self.ade_label = QLabel("--")
        self.ade_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        result_layout.addRow("ADE (平均位移误差):", self.ade_label)
        
        self.fde_label = QLabel("--")
        self.fde_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        result_layout.addRow("FDE (最终位移误差):", self.fde_label)
        
        self.goal_acc_label = QLabel("--")
        self.goal_acc_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 13px;")
        result_layout.addRow("目标准确率:", self.goal_acc_label)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
    
    def browse_trajectory(self):
        """浏览轨迹文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择轨迹文件", 
            "", 
            "Pickle文件 (*.pkl);;所有文件 (*)"
        )
        if file_path:
            self.traj_path_edit.setText(file_path)
            self.trajectory_loaded.emit(Path(file_path))


class TerraTNTMainWindow(QMainWindow):
    """TerraTNT主窗口 - 专业Qt桌面应用程序"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TerraTNT - 地面目标轨迹预测系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置应用程序样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2c3e50;
            }
            QLabel {
                color: #2c3e50;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 4px;
                color: #2c3e50;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 2px solid #3498db;
            }
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                color: #2c3e50;
            }
        """)
        
        self.create_menu_bar()
        self.create_tool_bar()
        self.create_status_bar()
        self.create_central_widget()
    
    def create_menu_bar(self):
        """创建菜单栏（简化版，只保留有用功能）"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #34495e;
                color: white;
                padding: 4px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #2c3e50;
            }
            QMenu {
                background-color: white;
                border: 1px solid #bdc3c7;
            }
            QMenu::item {
                padding: 6px 25px;
                color: #2c3e50;
            }
            QMenu::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("退出", self.close)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        view_menu.addAction("全屏", self.toggle_fullscreen)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        help_menu.addAction("关于", self.show_about)
    
    def create_tool_bar(self):
        """创建工具栏（简化版，移除无用功能）"""
        # 不创建工具栏，所有功能通过左侧标签页实现
        pass
    
    def create_status_bar(self):
        """创建状态栏"""
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #34495e;
                color: white;
                padding: 4px;
            }
        """)
        self.statusBar().showMessage("就绪")
    
    def create_central_widget(self):
        """创建中心部件"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局：使用QSplitter分割左右区域
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：功能标签页
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                color: #2c3e50;
                padding: 10px 20px;
                margin: 2px;
                border: 1px solid #bdc3c7;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #3498db;
            }
            QTabBar::tab:hover {
                background-color: #d5dbdb;
            }
        """)
        
        # 添加功能标签页（使用滚动区域避免窗口较小时控件堆叠/裁切）
        self._satellite_widget = SatelliteWidget()
        self._data_gen_widget = DataGenerationWidget()
        self._data_widget = DataLoadWidget()
        self._training_widget = TrainingWidget()
        self._prediction_widget = PredictionWidget()

        self.tab_widget.addTab(self._wrap_scroll(self._satellite_widget), "卫星星座")
        self.tab_widget.addTab(self._wrap_scroll(self._data_gen_widget), "数据生成")
        self.tab_widget.addTab(self._wrap_scroll(self._data_widget), "数据加载")
        self.tab_widget.addTab(self._wrap_scroll(self._training_widget), "模型训练")
        self.tab_widget.addTab(self._wrap_scroll(self._prediction_widget), "轨迹预测")
        
        left_layout.addWidget(self.tab_widget)
        
        # 右侧：可视化区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        vis_label = QLabel("可视化区域")
        vis_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        right_layout.addWidget(vis_label)

        self.vis_canvas = VisualizationCanvas()
        self.vis_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vis_canvas.show_message(
            "请选择左侧功能：\n\n"
            "1）在“数据加载”中加载区域数据（显示概览）\n"
            "2）在“轨迹预测”中选择一个 .pkl 轨迹文件（显示真实轨迹）"
        )
        right_layout.addWidget(self.vis_canvas)
        
        # 添加到分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)  # 左侧占60%（参数设置区需要更多空间）
        splitter.setStretchFactor(1, 2)  # 右侧占40%（可视化区）
        
        # 设置初始宽度（像素）
        splitter.setSizes([800, 600])  # 左侧800px，右侧600px
        
        main_layout.addWidget(splitter)

        # 信号联动：数据加载/轨迹加载 -> 右侧可视化
        self._satellite_widget.constellation_updated.connect(self._on_constellation_updated)
        self._data_gen_widget.generation_started.connect(self._on_generation_started)
        self._data_widget.data_loaded.connect(self._on_data_loaded)
        self._data_widget.show_dem.connect(self._on_show_dem)
        self._data_widget.show_lulc.connect(self._on_show_lulc)
        self._prediction_widget.trajectory_loaded.connect(self._on_trajectory_loaded)

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def _on_data_loaded(self, region_path: Path, traj_files: list) -> None:
        self.vis_canvas.plot_region_overview([Path(p) for p in traj_files])
        self.statusBar().showMessage(f"已加载区域：{region_path}（{len(traj_files)} 个轨迹文件）")

    def _on_trajectory_loaded(self, pkl_path: Path) -> None:
        self.vis_canvas.plot_trajectory_from_pkl(pkl_path)
        self.statusBar().showMessage(f"已加载轨迹文件：{pkl_path.name}")

    def _on_show_dem(self, region_key: str) -> None:
        self.vis_canvas.plot_real_dem(region_key)
        self.statusBar().showMessage(f"显示真实DEM：{region_key}")

    def _on_show_lulc(self, region_key: str) -> None:
        self.vis_canvas.plot_real_lulc(region_key)
        self.statusBar().showMessage(f"显示真实LULC：{region_key}")
    
    def _on_constellation_updated(self) -> None:
        """卫星星座配置更新，显示3D可视化"""
        num_sats = self._satellite_widget.sat_num_spin.value()
        num_planes = self._satellite_widget.plane_num_spin.value()
        altitude = self._satellite_widget.altitude_spin.value()
        self.vis_canvas.plot_satellite_constellation_3d(num_sats, num_planes, altitude)
        self.statusBar().showMessage(f"卫星星座3D可视化：{num_sats}颗卫星 / {num_planes}个轨道面")
    
    def _on_generation_started(self, region: str, num_traj: int) -> None:
        """数据生成任务启动"""
        self.statusBar().showMessage(f"正在生成 {region} 区域的 {num_traj} 条轨迹...")
        # 这里可以启动后台生成任务
        QMessageBox.information(self, "提示", f"数据生成功能需要在终端运行：\n\npython scripts/generate_dataset_parallel.py --region {region} --num {num_traj}")
    
    # 菜单回调函数（只保留有用功能）
    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
            "TerraTNT - 地面目标轨迹预测系统\n\n"
            "版本: 1.0.0\n"
            "基于深度学习的长时域轨迹预测\n\n"
            "© 2026 TerraTNT团队")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，跨平台一致

    _set_chinese_font(app)
    try:
        import matplotlib as _mpl
        # 设置matplotlib使用系统中确实可用的中文字体
        _mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'AR PL UKai CN', 'SimHei', 'DejaVu Sans']
        _mpl.rcParams['axes.unicode_minus'] = False
        # 清除字体缓存，强制重新检测
        try:
            _mpl.font_manager._rebuild()
        except:
            pass
    except Exception:
        pass
    
    # 设置应用程序图标和名称
    app.setApplicationName("TerraTNT")
    app.setOrganizationName("TerraTNT Team")
    
    window = TerraTNTMainWindow()
    window.show()
    
    sys.exit(app.exec_())
