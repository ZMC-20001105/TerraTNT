import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFrame, QLabel, QPushButton, 
                             QProgressBar, QScrollArea, QGraphicsDropShadowEffect,
                             QStackedWidget)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QFont, QPalette

# QSS 现代深色样式表
MODERN_STYLE = """
QMainWindow {
    background-color: #0F172A;
}

QWidget#Sidebar {
    background-color: #020617;
    min-width: 200px;
    max-width: 200px;
}

QWidget#MainContent {
    background-color: #0F172A;
}

QFrame#Card {
    background-color: #1E293B;
    border-radius: 12px;
    border: 1px solid #334155;
}

QLabel#CardTitle {
    color: #F1F5F9;
    font-size: 14px;
    font-weight: bold;
}

QLabel#CardValue {
    color: #38BDF8;
    font-size: 24px;
    font-weight: bold;
}

QPushButton#NavBtn {
    background-color: transparent;
    color: #94A3B8;
    text-align: left;
    padding: 12px;
    border: none;
    font-size: 13px;
    border-radius: 6px;
}

QPushButton#NavBtn:hover {
    background-color: #1E293B;
    color: #F1F5F9;
}

QPushButton#NavBtn[active="true"] {
    background-color: rgba(56, 189, 248, 0.1);
    color: #38BDF8;
    border-left: 3px solid #38BDF8;
}

QPushButton#ActionBtn {
    background-color: #0EA5E9;
    color: white;
    border-radius: 6px;
    padding: 10px;
    font-weight: bold;
}

QPushButton#ActionBtn:hover {
    background-color: #38BDF8;
}

QProgressBar {
    background-color: #334155;
    border-radius: 4px;
    text-align: center;
    color: white;
}

QProgressBar::chunk {
    background-color: #38BDF8;
    border-radius: 4px;
}
"""

class ModernCard(QFrame):
    def __init__(self, title, value, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        
        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setObjectName("CardValue")
        layout.addWidget(value_label)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)

class ModernDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TerraTNT Workstation v2.0")
        self.resize(1400, 900)
        self.setStyleSheet(MODERN_STYLE)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- Sidebar ---
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        
        logo = QLabel("TerraTNT")
        logo.setStyleSheet("color: #38BDF8; font-size: 24px; font-weight: bold; margin-bottom: 30px;")
        sidebar_layout.addWidget(logo)
        
        self.nav_buttons = []
        modules = [
            ("概览 Overview", self.show_overview),
            ("环境工作站 Environment", self.show_environment),
            ("数据集工厂 Dataset", self.show_dataset),
            ("模型实验室 Model Lab", self.show_model_lab),
            ("智能预测中心 Inference", self.show_inference),
            ("实验评估看板 Analytics", self.show_analytics)
        ]
        
        for i, (name, slot) in enumerate(modules):
            btn = QPushButton(name)
            btn.setObjectName("NavBtn")
            btn.clicked.connect(slot)
            btn.clicked.connect(lambda _, b=btn: self.update_nav_style(b))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)
        
        self.nav_buttons[0].setProperty("active", "true")
        
        sidebar_layout.addStretch()
        main_layout.addWidget(self.sidebar)
        
        # --- Main Content Area (Stacked Widget) ---
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack, 1)
        
        self.init_pages()
        self.show_overview()

    def update_nav_style(self, active_btn):
        for btn in self.nav_buttons:
            btn.setProperty("active", "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        active_btn.setProperty("active", "true")
        active_btn.style().unpolish(active_btn)
        active_btn.style().polish(active_btn)

    def init_pages(self):
        # 1. Overview Page
        self.page_overview = QWidget()
        layout = QVBoxLayout(self.page_overview)
        layout.setContentsMargins(30, 30, 30, 30)
        
        header = QLabel("系统概览 / System Overview")
        header.setStyleSheet("color: white; font-size: 22px; font-weight: bold;")
        layout.addWidget(header)
        
        stats = QHBoxLayout()
        stats.addWidget(ModernCard("活跃区域 / Regions", "4 Areas"))
        stats.addWidget(ModernCard("样本总量 / Total Tracks", "14,400"))
        stats.addWidget(ModernCard("训练状态 / Training", "2 Active"))
        stats.addWidget(ModernCard("系统负载 / CPU", "45%"))
        layout.addLayout(stats)
        
        chart_area = QFrame()
        chart_area.setObjectName("Card")
        chart_layout = QVBoxLayout(chart_area)
        chart_layout.addWidget(QLabel("近期任务进度 / Recent Tasks"))
        # Placeholder for progress list
        for task in ["Scottish Highlands Gen - 50%", "Bohemian Forest Gen - 34%", "YNet Baseline Training - Epoch 12"]:
            item = QLabel(f"• {task}")
            item.setStyleSheet("color: #94A3B8; margin-left: 10px;")
            chart_layout.addWidget(item)
        layout.addWidget(chart_area, 1)
        self.content_stack.addWidget(self.page_overview)

        # 2. Environment Page
        self.page_env = QWidget()
        env_layout = QVBoxLayout(self.page_env)
        env_layout.addWidget(QLabel("环境工作站 / Environment Station").setStyleSheet("color: white; font-size: 22px; font-weight: bold;"))
        # ... Add map viewer mockup
        self.content_stack.addWidget(self.page_env)

        # 3. Dataset Page
        self.page_dataset = QWidget()
        # ... Add task manager mockup
        self.content_stack.addWidget(self.page_dataset)

        # 4. Model Lab Page
        self.page_model = QWidget()
        # ... Add model repo mockup
        self.content_stack.addWidget(self.page_model)

        # 5. Inference Page
        self.page_inference = QWidget()
        # ... Add interactive prediction mockup
        self.content_stack.addWidget(self.page_inference)

        # 6. Analytics Page
        self.page_analytics = QWidget()
        # ... Add metrics matrix mockup
        self.content_stack.addWidget(self.page_analytics)

    def show_overview(self): self.content_stack.setCurrentIndex(0)
    def show_environment(self): self.content_stack.setCurrentIndex(1)
    def show_dataset(self): self.content_stack.setCurrentIndex(2)
    def show_model_lab(self): self.content_stack.setCurrentIndex(3)
    def show_inference(self): self.content_stack.setCurrentIndex(4)
    def show_analytics(self): self.content_stack.setCurrentIndex(5)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernDashboard()
    window.show()
    sys.exit(app.exec_())
