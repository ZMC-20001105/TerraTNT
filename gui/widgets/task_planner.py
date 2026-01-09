"""
任务规划组件
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal


class TaskPlannerWidget(QWidget):
    """任务规划组件"""
    
    prediction_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("任务规划功能开发中..."))
        layout.addStretch()
