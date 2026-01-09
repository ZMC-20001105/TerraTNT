"""
日志查看器组件
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QTextCursor


class LogViewerWidget(QWidget):
    """日志查看器组件"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)
    
    def append_log(self, message: str):
        """添加日志消息"""
        self.log_text.append(message)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
