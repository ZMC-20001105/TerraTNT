"""
地图视图组件
使用 matplotlib 嵌入 Qt 实现地图可视化
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class MapViewerWidget(QWidget):
    """地图视图组件"""
    
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.trajectories = []
        self.predictions = []
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 控制栏
        control_layout = QHBoxLayout()
        
        # 图层选择
        control_layout.addWidget(QLabel("图层:"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["卫星影像", "DEM", "坡度", "坡向", "LULC"])
        self.layer_combo.currentTextChanged.connect(self.change_layer)
        control_layout.addWidget(self.layer_combo)
        
        control_layout.addStretch()
        
        # 控制按钮
        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(self.reset_view_btn)
        
        layout.addLayout(control_layout)
        
        # 创建 matplotlib 画布
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # 添加工具栏
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # 初始化地图
        self.init_map()
    
    def init_map(self):
        """初始化地图"""
        self.ax.clear()
        self.ax.set_title("地图视图", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("经度")
        self.ax.set_ylabel("纬度")
        self.ax.grid(True, alpha=0.3)
        self.ax.text(0.5, 0.5, '请加载数据', 
                    ha='center', va='center', 
                    transform=self.ax.transAxes,
                    fontsize=16, color='gray')
        self.canvas.draw()
    
    def load_data(self, data: dict):
        """加载数据"""
        self.current_data = data
        self.update_map()
    
    def change_layer(self, layer_name: str):
        """切换图层"""
        if self.current_data is None:
            return
        self.update_map()
    
    def update_map(self):
        """更新地图显示"""
        if self.current_data is None:
            return
        
        self.ax.clear()
        
        layer = self.layer_combo.currentText()
        
        # TODO: 根据选择的图层显示不同数据
        self.ax.set_title(f"地图视图 - {layer}", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("经度")
        self.ax.set_ylabel("纬度")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def show_trajectory(self, trajectory_data: dict):
        """显示轨迹"""
        if 'positions' in trajectory_data:
            positions = trajectory_data['positions']
            self.ax.plot(positions[:, 0], positions[:, 1], 
                        'b-', linewidth=2, label='真实轨迹')
            self.ax.legend()
            self.canvas.draw()
    
    def show_prediction(self, prediction_data: dict):
        """显示预测结果"""
        if 'predicted_positions' in prediction_data:
            positions = prediction_data['predicted_positions']
            self.ax.plot(positions[:, 0], positions[:, 1], 
                        'r--', linewidth=2, label='预测轨迹', alpha=0.7)
            self.ax.legend()
            self.canvas.draw()
    
    def reset_view(self):
        """重置视图"""
        self.ax.autoscale()
        self.canvas.draw()
