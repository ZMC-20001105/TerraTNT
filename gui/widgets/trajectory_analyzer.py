"""
轨迹分析组件
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal


class TrajectoryAnalyzerWidget(QWidget):
    """轨迹分析组件"""
    
    trajectory_selected = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("轨迹分析功能开发中..."))
        layout.addStretch()
