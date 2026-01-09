"""
主窗口界面
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QStatusBar, QMenuBar, QToolBar,
    QLabel, QSplitter, QDockWidget
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QAction

from config import cfg
from gui.widgets.data_manager import DataManagerWidget
from gui.widgets.map_viewer import MapViewerWidget
from gui.widgets.trajectory_analyzer import TrajectoryAnalyzerWidget
from gui.widgets.model_trainer import ModelTrainerWidget
from gui.widgets.task_planner import TaskPlannerWidget
from gui.widgets.result_exporter import ResultExporterWidget


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 加载配置
        self.window_config = cfg.get('gui.window', {})
        self.theme_config = cfg.get('gui.theme', {})
        
        # 初始化UI
        self.init_ui()
        self.create_menu_bar()
        self.create_tool_bar()
        self.create_status_bar()
        self.create_dock_widgets()
        
        # 应用主题
        self.apply_theme()
    
    def init_ui(self):
        """初始化UI布局"""
        # 设置窗口属性
        self.setWindowTitle(self.window_config.get('title', 'TerraTNT'))
        self.setGeometry(100, 100, 
                        self.window_config.get('width', 1600),
                        self.window_config.get('height', 900))
        self.setMinimumSize(self.window_config.get('min_width', 1200),
                           self.window_config.get('min_height', 700))
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建分割器（左右布局）
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：地图视图
        self.map_viewer = MapViewerWidget()
        splitter.addWidget(self.map_viewer)
        
        # 右侧：功能标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # 添加各功能模块
        self.data_manager = DataManagerWidget()
        self.trajectory_analyzer = TrajectoryAnalyzerWidget()
        self.model_trainer = ModelTrainerWidget()
        self.task_planner = TaskPlannerWidget()
        self.result_exporter = ResultExporterWidget()
        
        self.tab_widget.addTab(self.data_manager, "📊 数据管理")
        self.tab_widget.addTab(self.trajectory_analyzer, "📈 轨迹分析")
        self.tab_widget.addTab(self.model_trainer, "🧠 模型训练")
        self.tab_widget.addTab(self.task_planner, "🛰️ 任务规划")
        self.tab_widget.addTab(self.result_exporter, "💾 结果导出")
        
        splitter.addWidget(self.tab_widget)
        
        # 设置分割比例（地图:功能面板 = 6:4）
        splitter.setSizes([960, 640])
        
        main_layout.addWidget(splitter)
        
        # 连接信号
        self.connect_signals()
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        open_action = QAction('打开项目', self)
        open_action.setShortcut('Ctrl+O')
        file_menu.addAction(open_action)
        
        save_action = QAction('保存项目', self)
        save_action.setShortcut('Ctrl+S')
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu('编辑(&E)')
        
        settings_action = QAction('设置', self)
        settings_action.setShortcut('Ctrl+,')
        edit_menu.addAction(settings_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图(&V)')
        
        fullscreen_action = QAction('全屏', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具(&T)')
        
        data_process_action = QAction('数据预处理', self)
        tools_menu.addAction(data_process_action)
        
        trajectory_gen_action = QAction('轨迹生成', self)
        tools_menu.addAction(trajectory_gen_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        doc_action = QAction('文档', self)
        help_menu.addAction(doc_action)
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # 添加工具按钮
        load_data_action = QAction("加载数据", self)
        load_data_action.setStatusTip("加载GEE和OORD数据")
        toolbar.addAction(load_data_action)
        
        toolbar.addSeparator()
        
        train_model_action = QAction("训练模型", self)
        train_model_action.setStatusTip("训练TerraTNT模型")
        toolbar.addAction(train_model_action)
        
        predict_action = QAction("预测轨迹", self)
        predict_action.setStatusTip("预测目标轨迹")
        toolbar.addAction(predict_action)
        
        toolbar.addSeparator()
        
        export_action = QAction("导出结果", self)
        export_action.setStatusTip("导出预测结果")
        toolbar.addAction(export_action)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 添加状态信息
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
        # 添加进度信息（右侧）
        self.progress_label = QLabel("")
        self.status_bar.addPermanentWidget(self.progress_label)
    
    def create_dock_widgets(self):
        """创建停靠窗口"""
        # 日志窗口
        log_dock = QDockWidget("日志", self)
        log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        
        from gui.widgets.log_viewer import LogViewerWidget
        log_widget = LogViewerWidget()
        log_dock.setWidget(log_widget)
        
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)
        
        # 默认隐藏
        log_dock.hide()
    
    def apply_theme(self):
        """应用主题样式"""
        theme_name = self.theme_config.get('name', 'fusion')
        dark_mode = self.theme_config.get('dark_mode', False)
        accent_color = self.theme_config.get('accent_color', '#2E86AB')
        
        # 设置样式表
        if dark_mode:
            stylesheet = f"""
            QMainWindow {{
                background-color: #2b2b2b;
                color: #ffffff;
            }}
            QTabWidget::pane {{
                border: 1px solid #3d3d3d;
                background-color: #2b2b2b;
            }}
            QTabBar::tab {{
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {accent_color};
            }}
            QPushButton {{
                background-color: {accent_color};
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #3a9fc4;
            }}
            """
        else:
            stylesheet = f"""
            QMainWindow {{
                background-color: #f5f5f5;
            }}
            QTabWidget::pane {{
                border: 1px solid #ddd;
                background-color: #ffffff;
            }}
            QTabBar::tab {{
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {accent_color};
                color: #ffffff;
            }}
            QPushButton {{
                background-color: {accent_color};
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #3a9fc4;
            }}
            QPushButton:pressed {{
                background-color: #1e6a8a;
            }}
            """
        
        self.setStyleSheet(stylesheet)
    
    def connect_signals(self):
        """连接信号和槽"""
        # 数据管理 -> 地图视图
        self.data_manager.data_loaded.connect(self.map_viewer.load_data)
        
        # 轨迹分析 -> 地图视图
        self.trajectory_analyzer.trajectory_selected.connect(
            self.map_viewer.show_trajectory
        )
        
        # 任务规划 -> 地图视图
        self.task_planner.prediction_updated.connect(
            self.map_viewer.show_prediction
        )
    
    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def show_about(self):
        """显示关于对话框"""
        from PyQt6.QtWidgets import QMessageBox
        
        QMessageBox.about(
            self,
            "关于 TerraTNT",
            f"""
            <h2>TerraTNT 多星协同观测任务规划系统</h2>
            <p>版本: {cfg.get('project.version', '1.0.0')}</p>
            <p>基于深度学习的地面目标轨迹预测系统</p>
            <br>
            <p><b>主要功能：</b></p>
            <ul>
                <li>多源地理数据管理</li>
                <li>越野轨迹分析与生成</li>
                <li>TerraTNT 模型训练与预测</li>
                <li>卫星观测任务规划</li>
            </ul>
            """
        )
    
    def update_status(self, message: str):
        """更新状态栏消息"""
        self.status_label.setText(message)
    
    def update_progress(self, message: str):
        """更新进度信息"""
        self.progress_label.setText(message)
