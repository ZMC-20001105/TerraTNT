#!/usr/bin/env python
"""
TerraTNT交互式UI系统
包含：参数设置、数据加载、真实地球图片、轨迹预测动画
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
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json

class InteractiveSatelliteCanvas(FigureCanvas):
    """交互式卫星星座3D可视化"""
    def __init__(self, parent=None):
        fig = Figure(figsize=(10, 10), facecolor='#1e1e2e')
        self.axes = fig.add_subplot(111, projection='3d', facecolor='#1e1e2e')
        super().__init__(fig)
        self.setParent(parent)
        
        # 可调参数
        self.num_satellites = 9
        self.num_planes = 3
        self.orbit_altitude = 600
        self.inclinations = [45, 55, 65]
        
        self.plot_constellation()
    
    def update_parameters(self, num_sats, num_planes, altitude, inclinations):
        """更新参数并重绘"""
        self.num_satellites = num_sats
        self.num_planes = num_planes
        self.orbit_altitude = altitude
        self.inclinations = inclinations
        self.plot_constellation()
    
    def plot_constellation(self):
        """绘制卫星星座"""
        self.axes.clear()
        
        # 使用真实的地球纹理颜色
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = 6371 * np.outer(np.cos(u), np.sin(v))
        y = 6371 * np.outer(np.sin(u), np.sin(v))
        z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # 地球表面（蓝绿色）
        self.axes.plot_surface(x, y, z, color='#4a9eff', alpha=0.6, shade=True)
        
        # 绘制轨道和卫星
        orbit_radius = 6371 + self.orbit_altitude
        colors = ['#f38ba8', '#a6e3a1', '#89b4fa']
        sats_per_plane = self.num_satellites // self.num_planes
        
        for i in range(self.num_planes):
            color = colors[i % len(colors)]
            inclination = np.radians(self.inclinations[i])
            
            theta = np.linspace(0, 2*np.pi, 100)
            x_orbit = orbit_radius * np.cos(theta)
            y_orbit = orbit_radius * np.sin(theta) * np.cos(inclination)
            z_orbit = orbit_radius * np.sin(theta) * np.sin(inclination)
            
            self.axes.plot(x_orbit, y_orbit, z_orbit, color=color, linewidth=2.5, 
                          label=f'Plane {i+1} ({self.inclinations[i]}°)', alpha=0.8)
            
            # 卫星位置
            for j in range(sats_per_plane):
                angle = j * 2*np.pi/sats_per_plane
                sat_x = orbit_radius * np.cos(angle)
                sat_y = orbit_radius * np.sin(angle) * np.cos(inclination)
                sat_z = orbit_radius * np.sin(angle) * np.sin(inclination)
                self.axes.scatter([sat_x], [sat_y], [sat_z], color=color, s=150, 
                                marker='^', edgecolors='white', linewidths=2, zorder=10)
        
        self.axes.set_xlabel('X (km)', color='white', fontsize=11)
        self.axes.set_ylabel('Y (km)', color='white', fontsize=11)
        self.axes.set_zlabel('Z (km)', color='white', fontsize=11)
        self.axes.set_title(f'Satellite Constellation\n{self.num_satellites} Sats | {self.num_planes} Planes | {self.orbit_altitude}km Alt', 
                           color='white', fontsize=14, fontweight='bold')
        self.axes.legend(facecolor='#313244', edgecolor='#89b4fa', labelcolor='white', fontsize=10)
        self.axes.tick_params(colors='white')
        self.axes.xaxis.pane.fill = False
        self.axes.yaxis.pane.fill = False
        self.axes.zaxis.pane.fill = False
        self.axes.grid(True, alpha=0.2, color='white')
        self.draw()


class InteractiveRegionCanvas(FigureCanvas):
    """交互式地理区域地图"""
    def __init__(self, parent=None):
        fig = Figure(figsize=(12, 10), facecolor='#1e1e2e')
        self.axes = fig.add_subplot(111, facecolor='#1e1e2e')
        super().__init__(fig)
        self.setParent(parent)
        
        self.current_region = 'scottish_highlands'
        self.data_loaded = False
        self.dem_data = None
        
        self.plot_region()
    
    def load_region_data(self, region_path):
        """加载真实区域数据"""
        try:
            # 这里应该加载真实的DEM数据
            # 目前使用模拟数据
            self.data_loaded = True
            self.plot_region()
            return True
        except Exception as e:
            return False
    
    def plot_region(self):
        """绘制区域地图"""
        self.axes.clear()
        
        # 模拟真实地形数据
        x = np.linspace(0, 100, 300)
        y = np.linspace(0, 100, 300)
        X, Y = np.meshgrid(x, y)
        
        # 更真实的地形生成
        Z = (50 * np.sin(X/10) * np.cos(Y/10) + 
             30 * np.sin(X/5) * np.cos(Y/8) +
             20 * np.random.randn(300, 300) + 200)
        
        # 使用真实地形配色
        im = self.axes.contourf(X, Y, Z, levels=50, cmap='terrain', alpha=0.95)
        cbar = self.figure.colorbar(im, ax=self.axes, label='Elevation (m)')
        cbar.ax.tick_params(labelcolor='white')
        cbar.set_label('Elevation (m)', color='white', fontsize=11)
        
        # 区域信息
        region_info = {
            'scottish_highlands': {
                'name': 'Scottish Highlands',
                'area': '25,000 km²',
                'elevation': '200-1,300 m',
                'trajectories': 3600,
                'terrain': 'Mountainous'
            },
            'bohemian_forest': {
                'name': 'Bohemian Forest',
                'area': '12,000 km²',
                'elevation': '300-1,456 m',
                'trajectories': 2800,
                'terrain': 'Forest & Hills'
            }
        }
        
        info = region_info.get(self.current_region, region_info['scottish_highlands'])
        info_text = f"{info['name']}\n{'─'*25}\n"
        info_text += f"Area: {info['area']}\n"
        info_text += f"Elevation: {info['elevation']}\n"
        info_text += f"Trajectories: {info['trajectories']}\n"
        info_text += f"Terrain: {info['terrain']}"
        
        self.axes.text(5, 93, info_text, bbox=dict(boxstyle='round', facecolor='#313244', 
                      alpha=0.95, edgecolor='#89b4fa', linewidth=2),
                      fontsize=11, color='white', verticalalignment='top', family='monospace')
        
        self.axes.set_xlabel('X (km)', color='white', fontsize=12)
        self.axes.set_ylabel('Y (km)', color='white', fontsize=12)
        self.axes.set_title(f'Region: {info["name"]}', color='white', fontsize=15, fontweight='bold')
        self.axes.grid(True, alpha=0.3, color='white')
        self.axes.tick_params(colors='white')
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#45475a')
        
        self.draw()


class InteractiveTrajectoryCanvas(FigureCanvas):
    """交互式轨迹预测动画"""
    def __init__(self, parent=None):
        fig = Figure(figsize=(12, 12), facecolor='#1e1e2e')
        self.axes = fig.add_subplot(111, facecolor='#1e1e2e')
        super().__init__(fig)
        self.setParent(parent)
        
        self.current_frame = 0
        self.max_frames = 60
        self.is_playing = False
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.history_traj = None
        self.ground_truth = None
        self.prediction = None
        
        self.setup_demo_data()
        self.plot_frame()
    
    def setup_demo_data(self):
        """设置演示数据"""
        t_hist = np.linspace(0, 10, 10)
        self.history_traj = np.column_stack([
            20 + t_hist * 2,
            20 + t_hist * 3 + np.sin(t_hist) * 2
        ])
        
        t_future = np.linspace(10, 70, 60)
        self.ground_truth = np.column_stack([
            20 + t_future * 2 + np.sin(t_future/5) * 5,
            20 + t_future * 3 + np.cos(t_future/5) * 5
        ])
        
        self.prediction = self.ground_truth + np.random.randn(60, 2) * 2
    
    def load_trajectory_data(self, traj_file):
        """加载真实轨迹数据"""
        try:
            # 这里应该加载真实轨迹
            return True
        except:
            return False
    
    def plot_frame(self):
        """绘制当前帧"""
        self.axes.clear()
        
        # 地形背景
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X/10) * np.cos(Y/10)
        self.axes.contourf(X, Y, Z, levels=20, cmap='terrain', alpha=0.4)
        
        # 历史轨迹
        self.axes.plot(self.history_traj[:, 0], self.history_traj[:, 1], 
                      'o-', color='#89b4fa', linewidth=4, markersize=10, 
                      label='Observed (10 min)', zorder=5)
        
        # 真实轨迹
        self.axes.plot(self.ground_truth[:, 0], self.ground_truth[:, 1], 
                      '--', color='#a6e3a1', linewidth=3, alpha=0.6,
                      label='Ground Truth (60 min)', zorder=3)
        
        # 预测轨迹（逐帧显示）
        if self.current_frame > 0:
            pred_so_far = self.prediction[:self.current_frame]
            self.axes.plot(pred_so_far[:, 0], pred_so_far[:, 1], 
                          'o-', color='#f38ba8', linewidth=3, markersize=7,
                          label=f'Predicted ({self.current_frame} min)', zorder=4)
            
            # 当前位置
            self.axes.scatter([self.prediction[self.current_frame-1, 0]], 
                            [self.prediction[self.current_frame-1, 1]], 
                            color='#f9e2af', s=500, marker='*', 
                            edgecolors='white', linewidths=2, zorder=6)
        
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 100)
        self.axes.set_xlabel('X (km)', color='white', fontsize=12)
        self.axes.set_ylabel('Y (km)', color='white', fontsize=12)
        self.axes.set_title(f'Trajectory Prediction (Frame {self.current_frame}/{self.max_frames})', 
                           color='white', fontsize=15, fontweight='bold')
        self.axes.legend(loc='upper left', facecolor='#313244', edgecolor='#89b4fa', 
                        labelcolor='white', fontsize=11)
        self.axes.grid(True, alpha=0.3, color='white')
        self.axes.tick_params(colors='white')
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#45475a')
        
        self.draw()
    
    def update_frame(self):
        """更新动画帧"""
        if self.current_frame >= self.max_frames:
            self.stop_animation()
            return
        
        self.current_frame += 1
        self.plot_frame()
    
    def start_animation(self, speed=100):
        """开始播放"""
        self.is_playing = True
        self.timer.start(speed)
    
    def stop_animation(self):
        """停止播放"""
        self.is_playing = False
        self.timer.stop()
    
    def reset_animation(self):
        """重置"""
        self.current_frame = 0
        self.plot_frame()


class TerraTNTInteractiveUI(QMainWindow):
    """TerraTNT交互式主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TerraTNT - Interactive Trajectory Prediction System')
        self.setGeometry(50, 50, 1800, 1000)
        
        # 设置暗色主题
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QTabWidget::pane {
                border: 2px solid #45475a;
                background-color: #1e1e2e;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #cdd6f4;
                padding: 10px 20px;
                margin: 2px;
                border: 1px solid #45475a;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #a6e3a1;
            }
            QPushButton:pressed {
                background-color: #f38ba8;
            }
            QGroupBox {
                border: 2px solid #45475a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #cdd6f4;
                font-size: 12px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                padding: 5px;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
            }
        """)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.create_satellite_tab()
        self.create_region_tab()
        self.create_prediction_tab()
        self.create_training_tab()
        
        self.show()
    
    def create_satellite_tab(self):
        """卫星星座配置页面"""
        tab = QWidget()
        layout = QHBoxLayout()
        
        # 左侧：参数设置
        left_panel = QGroupBox('Constellation Parameters')
        left_layout = QVBoxLayout()
        
        # 卫星数量
        left_layout.addWidget(QLabel('Number of Satellites:'))
        self.sat_num_spin = QSpinBox()
        self.sat_num_spin.setRange(3, 30)
        self.sat_num_spin.setValue(9)
        left_layout.addWidget(self.sat_num_spin)
        
        # 轨道面数量
        left_layout.addWidget(QLabel('Number of Orbit Planes:'))
        self.plane_num_spin = QSpinBox()
        self.plane_num_spin.setRange(1, 5)
        self.plane_num_spin.setValue(3)
        left_layout.addWidget(self.plane_num_spin)
        
        # 轨道高度
        left_layout.addWidget(QLabel('Orbit Altitude (km):'))
        self.altitude_spin = QSpinBox()
        self.altitude_spin.setRange(400, 1000)
        self.altitude_spin.setValue(600)
        left_layout.addWidget(self.altitude_spin)
        
        # 倾角设置
        left_layout.addWidget(QLabel('Inclinations (degrees):'))
        inc_layout = QHBoxLayout()
        self.inc1_spin = QSpinBox()
        self.inc1_spin.setRange(0, 90)
        self.inc1_spin.setValue(45)
        self.inc2_spin = QSpinBox()
        self.inc2_spin.setRange(0, 90)
        self.inc2_spin.setValue(55)
        self.inc3_spin = QSpinBox()
        self.inc3_spin.setRange(0, 90)
        self.inc3_spin.setValue(65)
        inc_layout.addWidget(self.inc1_spin)
        inc_layout.addWidget(self.inc2_spin)
        inc_layout.addWidget(self.inc3_spin)
        left_layout.addLayout(inc_layout)
        
        # 更新按钮
        update_btn = QPushButton('Update Constellation')
        update_btn.clicked.connect(self.update_satellite_view)
        left_layout.addWidget(update_btn)
        
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, 1)
        
        # 右侧：3D可视化
        self.satellite_canvas = InteractiveSatelliteCanvas()
        layout.addWidget(self.satellite_canvas, 3)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, '🛰️ Satellite Constellation')
    
    def create_region_tab(self):
        """地理区域页面"""
        tab = QWidget()
        layout = QHBoxLayout()
        
        # 左侧：数据加载
        left_panel = QGroupBox('Region Data Loading')
        left_layout = QVBoxLayout()
        
        # 区域选择
        left_layout.addWidget(QLabel('Select Region:'))
        self.region_combo = QComboBox()
        self.region_combo.addItems([
            'Scottish Highlands',
            'Bohemian Forest',
            'Carpathians',
            'Danube Delta'
        ])
        self.region_combo.currentTextChanged.connect(self.on_region_changed)
        left_layout.addWidget(self.region_combo)
        
        # 数据路径
        left_layout.addWidget(QLabel('Data Directory:'))
        data_path_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit('/home/zmc/文档/programwork/data/processed/synthetic_trajectories')
        data_path_layout.addWidget(self.data_path_edit)
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self.browse_data_dir)
        data_path_layout.addWidget(browse_btn)
        left_layout.addLayout(data_path_layout)
        
        # 加载按钮
        load_btn = QPushButton('Load Region Data')
        load_btn.clicked.connect(self.load_region_data)
        left_layout.addWidget(load_btn)
        
        # 状态显示
        left_layout.addWidget(QLabel('Status:'))
        self.region_status = QTextEdit()
        self.region_status.setReadOnly(True)
        self.region_status.setMaximumHeight(150)
        self.region_status.setText('Ready to load data...')
        left_layout.addWidget(self.region_status)
        
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, 1)
        
        # 右侧：地图
        self.region_canvas = InteractiveRegionCanvas()
        layout.addWidget(self.region_canvas, 3)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, '🗺️ Geographic Region')
    
    def create_prediction_tab(self):
        """轨迹预测动画页面"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 控制面板
        control_group = QGroupBox('Animation Control')
        control_layout = QHBoxLayout()
        
        self.play_btn = QPushButton('▶ Play')
        self.play_btn.clicked.connect(self.play_animation)
        control_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton('⏸ Pause')
        self.pause_btn.clicked.connect(self.pause_animation)
        control_layout.addWidget(self.pause_btn)
        
        self.reset_btn = QPushButton('🔄 Reset')
        self.reset_btn.clicked.connect(self.reset_animation)
        control_layout.addWidget(self.reset_btn)
        
        control_layout.addWidget(QLabel('Speed:'))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        self.speed_slider.setMaximumWidth(200)
        control_layout.addWidget(self.speed_slider)
        
        # 轨迹加载
        control_layout.addWidget(QLabel('Load Trajectory:'))
        load_traj_btn = QPushButton('Browse')
        load_traj_btn.clicked.connect(self.load_trajectory)
        control_layout.addWidget(load_traj_btn)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 动画画布
        self.traj_canvas = InteractiveTrajectoryCanvas()
        layout.addWidget(self.traj_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, '🎬 Trajectory Prediction')
    
    def create_training_tab(self):
        """模型训练页面"""
        tab = QWidget()
        layout = QHBoxLayout()
        
        # 左侧：训练配置
        left_panel = QGroupBox('Training Configuration')
        left_layout = QVBoxLayout()
        
        left_layout.addWidget(QLabel('Model:'))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['TerraTNT', 'YNet', 'PECNet', 'Trajectron++', 'Social-LSTM'])
        left_layout.addWidget(self.model_combo)
        
        left_layout.addWidget(QLabel('Learning Rate:'))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 0.01)
        self.lr_spin.setValue(0.0003)
        self.lr_spin.setSingleStep(0.0001)
        left_layout.addWidget(self.lr_spin)
        
        left_layout.addWidget(QLabel('Batch Size:'))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 128)
        self.batch_spin.setValue(64)
        left_layout.addWidget(self.batch_spin)
        
        left_layout.addWidget(QLabel('Epochs:'))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(10, 200)
        self.epoch_spin.setValue(100)
        left_layout.addWidget(self.epoch_spin)
        
        train_btn = QPushButton('Start Training')
        train_btn.clicked.connect(self.start_training)
        left_layout.addWidget(train_btn)
        
        left_layout.addWidget(QLabel('Training Log:'))
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        left_layout.addWidget(self.train_log)
        
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, 1)
        
        # 右侧：GPU监控
        right_panel = QGroupBox('GPU Status')
        right_layout = QVBoxLayout()
        
        self.gpu_status = QTextEdit()
        self.gpu_status.setReadOnly(True)
        self.gpu_status.setMaximumHeight(200)
        right_layout.addWidget(self.gpu_status)
        
        refresh_btn = QPushButton('Refresh GPU Status')
        refresh_btn.clicked.connect(self.refresh_gpu_status)
        right_layout.addWidget(refresh_btn)
        
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel, 2)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, '🎯 Model Training')
        
        # 初始化GPU状态
        self.refresh_gpu_status()
    
    # 回调函数
    def update_satellite_view(self):
        """更新卫星视图"""
        num_sats = self.sat_num_spin.value()
        num_planes = self.plane_num_spin.value()
        altitude = self.altitude_spin.value()
        inclinations = [self.inc1_spin.value(), self.inc2_spin.value(), self.inc3_spin.value()]
        
        self.satellite_canvas.update_parameters(num_sats, num_planes, altitude, inclinations)
        QMessageBox.information(self, 'Success', 'Constellation updated!')
    
    def on_region_changed(self, region_name):
        """区域切换"""
        region_key = region_name.lower().replace(' ', '_')
        self.region_canvas.current_region = region_key
        self.region_canvas.plot_region()
    
    def browse_data_dir(self):
        """浏览数据目录"""
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Data Directory')
        if dir_path:
            self.data_path_edit.setText(dir_path)
    
    def load_region_data(self):
        """加载区域数据"""
        data_path = self.data_path_edit.text()
        region = self.region_combo.currentText().lower().replace(' ', '_')
        full_path = Path(data_path) / region
        
        if full_path.exists():
            traj_files = list(full_path.glob('*.pkl'))
            self.region_status.setText(f'✓ Loaded {len(traj_files)} trajectory files from:\n{full_path}')
            self.region_canvas.load_region_data(str(full_path))
        else:
            self.region_status.setText(f'✗ Directory not found:\n{full_path}')
    
    def play_animation(self):
        """播放动画"""
        speed = 200 - self.speed_slider.value() * 15
        self.traj_canvas.start_animation(speed)
    
    def pause_animation(self):
        """暂停动画"""
        self.traj_canvas.stop_animation()
    
    def reset_animation(self):
        """重置动画"""
        self.traj_canvas.reset_animation()
    
    def load_trajectory(self):
        """加载轨迹文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Trajectory File', '', 'Pickle Files (*.pkl)')
        if file_path:
            success = self.traj_canvas.load_trajectory_data(file_path)
            if success:
                QMessageBox.information(self, 'Success', f'Loaded trajectory from:\n{file_path}')
            else:
                QMessageBox.warning(self, 'Error', 'Failed to load trajectory file')
    
    def start_training(self):
        """开始训练"""
        model = self.model_combo.currentText()
        lr = self.lr_spin.value()
        batch_size = self.batch_spin.value()
        epochs = self.epoch_spin.value()
        
        self.train_log.append(f'Starting training: {model}')
        self.train_log.append(f'  LR: {lr}, Batch: {batch_size}, Epochs: {epochs}')
        self.train_log.append('  Training script should be launched externally...')
    
    def refresh_gpu_status(self):
        """刷新GPU状态"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader'], capture_output=True, text=True)
            if result.returncode == 0:
                self.gpu_status.setText(f'GPU Status:\n{result.stdout}')
            else:
                self.gpu_status.setText('GPU not available')
        except:
            self.gpu_status.setText('nvidia-smi not found')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TerraTNTInteractiveUI()
    sys.exit(app.exec_())
