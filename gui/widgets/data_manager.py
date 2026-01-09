"""
数据管理组件
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QProgressBar, QTreeWidget,
    QTreeWidgetItem, QFileDialog, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, QThread
from pathlib import Path

from config import cfg


class DataManagerWidget(QWidget):
    """数据管理组件"""
    
    data_loaded = pyqtSignal(dict)  # 数据加载完成信号
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # GEE 数据组
        gee_group = QGroupBox("GEE 遥感数据")
        gee_layout = QVBoxLayout()
        
        self.gee_tree = QTreeWidget()
        self.gee_tree.setHeaderLabels(["数据类型", "状态", "文件数"])
        self.gee_tree.setMaximumHeight(200)
        gee_layout.addWidget(self.gee_tree)
        
        gee_btn_layout = QHBoxLayout()
        self.load_gee_btn = QPushButton("加载 GEE 数据")
        self.load_gee_btn.clicked.connect(self.load_gee_data)
        self.merge_gee_btn = QPushButton("合并分块")
        self.merge_gee_btn.clicked.connect(self.merge_gee_chunks)
        gee_btn_layout.addWidget(self.load_gee_btn)
        gee_btn_layout.addWidget(self.merge_gee_btn)
        gee_layout.addLayout(gee_btn_layout)
        
        gee_group.setLayout(gee_layout)
        layout.addWidget(gee_group)
        
        # OORD 数据组
        oord_group = QGroupBox("OORD 轨迹数据")
        oord_layout = QVBoxLayout()
        
        self.oord_tree = QTreeWidget()
        self.oord_tree.setHeaderLabels(["区域", "状态", "轨迹数"])
        self.oord_tree.setMaximumHeight(150)
        oord_layout.addWidget(self.oord_tree)
        
        oord_btn_layout = QHBoxLayout()
        self.load_oord_btn = QPushButton("加载 OORD 数据")
        self.load_oord_btn.clicked.connect(self.load_oord_data)
        self.extract_oord_btn = QPushButton("解压数据")
        self.extract_oord_btn.clicked.connect(self.extract_oord_data)
        oord_btn_layout.addWidget(self.load_oord_btn)
        oord_btn_layout.addWidget(self.extract_oord_btn)
        oord_layout.addLayout(oord_btn_layout)
        
        oord_group.setLayout(oord_layout)
        layout.addWidget(oord_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("未加载数据")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # 初始化数据树
        self.refresh_data_status()
    
    def refresh_data_status(self):
        """刷新数据状态"""
        # 检查 GEE 数据
        self.gee_tree.clear()
        gee_path = Path(cfg.get('paths.raw_data.gee'))
        
        if gee_path.exists():
            for region in gee_path.iterdir():
                if region.is_dir():
                    region_item = QTreeWidgetItem([region.name, "", ""])
                    
                    for data_type in ['dem', 'slope', 'aspect', 'lulc']:
                        type_path = region / data_type
                        if type_path.exists():
                            count = len(list(type_path.glob('*.tif')))
                            status = "✓ 完成" if count > 0 else "✗ 缺失"
                            child = QTreeWidgetItem([data_type.upper(), status, str(count)])
                            region_item.addChild(child)
                    
                    self.gee_tree.addTopLevelItem(region_item)
                    region_item.setExpanded(True)
        
        # 检查 OORD 数据
        self.oord_tree.clear()
        oord_path = Path(cfg.get('paths.raw_data.oord'))
        
        if oord_path.exists():
            for region in oord_path.iterdir():
                if region.is_dir():
                    count = len(list(region.glob('*.zip')))
                    status = "✓ 完成" if count > 0 else "✗ 缺失"
                    item = QTreeWidgetItem([region.name, status, str(count)])
                    self.oord_tree.addTopLevelItem(item)
    
    def load_gee_data(self):
        """加载 GEE 数据"""
        self.status_label.setText("正在加载 GEE 数据...")
        self.progress_bar.setVisible(True)
        # TODO: 实现数据加载逻辑
        QMessageBox.information(self, "提示", "GEE 数据加载功能开发中")
    
    def merge_gee_chunks(self):
        """合并 GEE 分块"""
        self.status_label.setText("正在合并 GEE 分块...")
        # TODO: 实现分块合并逻辑
        QMessageBox.information(self, "提示", "分块合并功能开发中")
    
    def load_oord_data(self):
        """加载 OORD 数据"""
        self.status_label.setText("正在加载 OORD 数据...")
        # TODO: 实现数据加载逻辑
        QMessageBox.information(self, "提示", "OORD 数据加载功能开发中")
    
    def extract_oord_data(self):
        """解压 OORD 数据"""
        self.status_label.setText("正在解压 OORD 数据...")
        # TODO: 实现数据解压逻辑
        QMessageBox.information(self, "提示", "数据解压功能开发中")
