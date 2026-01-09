"""
结果导出组件
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal


class ResultExporterWidget(QWidget):
    """结果导出组件"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("结果导出功能开发中..."))
        layout.addStretch()
