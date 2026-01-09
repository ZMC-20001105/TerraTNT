"""
TerraTNT系统Qt界面原型
包含主界面、数据管理、模型训练、预测、评估等模块
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                             QTableWidget, QTableWidgetItem, QGroupBox, 
                             QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
                             QProgressBar, QSplitter, QListWidget, QCheckBox,
                             QLineEdit, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QPen
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MapCanvas(FigureCanvas):
    """地图显示画布"""
    def __init__(self, parent=None, width=8, height=6):
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        
        # 绘制示例地图
        self.plot_demo_map()
    
    def plot_demo_map(self):
        """绘制演示地图"""
        self.axes.clear()
        
        # 生成地形数据
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X/10) * np.cos(Y/10) * 50 + 200
        
        # 绘制地形
        im = self.axes.contourf(X, Y, Z, levels=20, cmap='terrain', alpha=0.7)
        
        # 绘制示例轨迹
        # 历史轨迹（蓝色）
        history_x = [20, 25, 30, 35, 40]
        history_y = [20, 25, 28, 32, 35]
        self.axes.plot(history_x, history_y, 'b-o', linewidth=2, 
                      markersize=6, label='历史轨迹')
        
        # 预测轨迹（红色）
        pred_x = [40, 45, 50, 55, 60, 65, 70]
        pred_y = [35, 40, 43, 47, 50, 55, 58]
        self.axes.plot(pred_x, pred_y, 'r--o', linewidth=2, 
                      markersize=6, label='预测轨迹')
        
        # 真实轨迹（绿色）
        true_x = [40, 44, 49, 54, 59, 64, 69]
        true_y = [35, 39, 44, 48, 51, 56, 59]
        self.axes.plot(true_x, true_y, 'g-.o', linewidth=2, 
                      markersize=6, label='真实轨迹')
        
        # 候选目标（黄色点）
        goal_x = [65, 68, 70, 72, 75]
        goal_y = [55, 58, 60, 57, 62]
        self.axes.scatter(goal_x, goal_y, c='yellow', s=100, 
                         marker='*', edgecolors='black', label='候选目标')
        
        self.axes.set_xlabel('X (km)', fontsize=10)
        self.axes.set_ylabel('Y (km)', fontsize=10)
        self.axes.set_title('轨迹预测可视化', fontsize=12, fontweight='bold')
        self.axes.legend(loc='upper left', fontsize=9)
        self.axes.grid(True, alpha=0.3)
        
        self.draw()

class TerraTNTMainWindow(QMainWindow):
    """TerraTNT主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TerraTNT - 地面目标轨迹预测系统')
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # 添加各个功能页面
        self.create_overview_tab()
        self.create_dataset_tab()
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_evaluation_tab()
        self.create_settings_tab()
        
        # 状态栏
        self.statusBar().showMessage('就绪')
    
    def create_overview_tab(self):
        """创建概览页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 标题
        title = QLabel('TerraTNT 系统概览')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 系统状态
        status_group = QGroupBox('系统状态')
        status_layout = QVBoxLayout()
        
        status_info = [
            ('数据集', 'Scottish Highlands: 3,600条 | Bohemian Forest: 3,600条'),
            ('模型状态', 'TerraTNT已训练 | 6个基线模型已加载'),
            ('GPU状态', 'NVIDIA RTX 5060 (8GB) - 可用'),
            ('最近训练', '2026-01-08 09:30 | Epoch 45 | Val Loss: 0.0234'),
        ]
        
        for label, value in status_info:
            row = QHBoxLayout()
            row.addWidget(QLabel(f'<b>{label}:</b>'))
            row.addWidget(QLabel(value))
            row.addStretch()
            status_layout.addLayout(row)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 快速操作
        quick_group = QGroupBox('快速操作')
        quick_layout = QHBoxLayout()
        
        buttons = [
            ('加载数据集', self.load_dataset),
            ('开始训练', self.start_training),
            ('轨迹预测', self.predict_trajectory),
            ('模型评估', self.evaluate_model),
        ]
        
        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setMinimumHeight(40)
            btn.clicked.connect(callback)
            quick_layout.addWidget(btn)
        
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        # 最近活动
        activity_group = QGroupBox('最近活动')
        activity_layout = QVBoxLayout()
        
        self.activity_list = QListWidget()
        activities = [
            '✓ 2026-01-08 10:15 - Scottish Highlands数据集生成完成',
            '✓ 2026-01-08 09:30 - TerraTNT模型训练完成 (Epoch 45)',
            '✓ 2026-01-08 08:45 - Bohemian Forest数据预处理完成',
            '✓ 2026-01-08 07:20 - 基线模型YNet训练完成',
            '✓ 2026-01-08 06:10 - 数据增强完成 (18通道地图)',
        ]
        for activity in activities:
            self.activity_list.addItem(activity)
        
        activity_layout.addWidget(self.activity_list)
        activity_group.setLayout(activity_layout)
        layout.addWidget(activity_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, '概览')
    
    def create_dataset_tab(self):
        """创建数据集管理页面"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 左侧：数据集列表
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel('<b>数据集列表</b>'))
        
        self.dataset_list = QListWidget()
        datasets = [
            'Scottish Highlands (3,600条)',
            'Bohemian Forest (3,600条)',
            'Donbas (待生成)',
            'Carpathians (待生成)',
        ]
        for ds in datasets:
            self.dataset_list.addItem(ds)
        self.dataset_list.setCurrentRow(0)
        left_layout.addWidget(self.dataset_list)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton('导入'))
        btn_layout.addWidget(QPushButton('导出'))
        btn_layout.addWidget(QPushButton('删除'))
        left_layout.addLayout(btn_layout)
        
        layout.addWidget(left_panel, 1)
        
        # 右侧：数据集详情
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel('<b>数据集详情</b>'))
        
        # 统计信息
        stats_group = QGroupBox('统计信息')
        stats_layout = QVBoxLayout()
        
        stats_table = QTableWidget(8, 2)
        stats_table.setHorizontalHeaderLabels(['指标', '数值'])
        stats_data = [
            ('区域', 'Scottish Highlands'),
            ('轨迹总数', '3,600'),
            ('平均长度', '125.3 km'),
            ('平均时长', '358.7 分钟'),
            ('车辆类型', '4种 (Type1-4)'),
            ('战术意图', '3种 (Intent1-3)'),
            ('数据大小', '252 MB'),
            ('生成时间', '2026-01-08 10:15'),
        ]
        
        for i, (key, value) in enumerate(stats_data):
            stats_table.setItem(i, 0, QTableWidgetItem(key))
            stats_table.setItem(i, 1, QTableWidgetItem(value))
        
        stats_table.resizeColumnsToContents()
        stats_layout.addWidget(stats_table)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # 数据分布
        dist_group = QGroupBox('数据分布')
        dist_layout = QVBoxLayout()
        
        # 创建简单的分布图
        canvas = FigureCanvas(Figure(figsize=(6, 3)))
        ax = canvas.figure.add_subplot(111)
        
        # 轨迹长度分布
        lengths = np.random.normal(125, 20, 1000)
        ax.hist(lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('轨迹长度 (km)')
        ax.set_ylabel('频数')
        ax.set_title('轨迹长度分布')
        ax.grid(True, alpha=0.3)
        
        dist_layout.addWidget(canvas)
        dist_group.setLayout(dist_layout)
        right_layout.addWidget(dist_group)
        
        layout.addWidget(right_panel, 2)
        
        self.tabs.addTab(tab, '数据集')
    
    def create_training_tab(self):
        """创建模型训练页面"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 左侧：训练配置
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel('<b>训练配置</b>'))
        
        # 模型选择
        model_group = QGroupBox('模型选择')
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        models = ['TerraTNT (主模型)', 'YNet', 'PECNet', 'Trajectron++', 
                 'AgentFormer', 'Social-LSTM']
        self.model_combo.addItems(models)
        model_layout.addWidget(QLabel('选择模型:'))
        model_layout.addWidget(self.model_combo)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # 超参数
        hyper_group = QGroupBox('超参数设置')
        hyper_layout = QVBoxLayout()
        
        params = [
            ('学习率', 0.0001, 0.0001, 0.01, 4),
            ('批大小', 32, 8, 128, 0),
            ('训练轮数', 100, 10, 500, 0),
        ]
        
        for label, default, min_val, max_val, decimals in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(f'{label}:'))
            if decimals > 0:
                spin = QDoubleSpinBox()
                spin.setDecimals(decimals)
            else:
                spin = QSpinBox()
            spin.setMinimum(min_val)
            spin.setMaximum(max_val)
            spin.setValue(default)
            row.addWidget(spin)
            hyper_layout.addLayout(row)
        
        hyper_group.setLayout(hyper_layout)
        left_layout.addWidget(hyper_group)
        
        # 训练控制
        control_layout = QHBoxLayout()
        self.train_btn = QPushButton('开始训练')
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_btn)
        
        self.stop_btn = QPushButton('停止')
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(control_layout)
        left_layout.addStretch()
        
        layout.addWidget(left_panel, 1)
        
        # 右侧：训练监控
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel('<b>训练监控</b>'))
        
        # 进度
        progress_group = QGroupBox('训练进度')
        progress_layout = QVBoxLayout()
        
        self.epoch_label = QLabel('Epoch: 0/100')
        progress_layout.addWidget(self.epoch_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)
        
        # 损失曲线
        loss_group = QGroupBox('损失曲线')
        loss_layout = QVBoxLayout()
        
        self.loss_canvas = FigureCanvas(Figure(figsize=(7, 4)))
        self.loss_ax = self.loss_canvas.figure.add_subplot(111)
        
        # 绘制示例损失曲线
        epochs = np.arange(1, 46)
        train_loss = 0.5 * np.exp(-epochs/15) + 0.02
        val_loss = 0.55 * np.exp(-epochs/15) + 0.023
        
        self.loss_ax.plot(epochs, train_loss, 'b-', label='训练损失', linewidth=2)
        self.loss_ax.plot(epochs, val_loss, 'r-', label='验证损失', linewidth=2)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('训练/验证损失曲线')
        self.loss_ax.legend()
        self.loss_ax.grid(True, alpha=0.3)
        
        loss_layout.addWidget(self.loss_canvas)
        loss_group.setLayout(loss_layout)
        right_layout.addWidget(loss_group)
        
        # 训练日志
        log_group = QGroupBox('训练日志')
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_text_sample = """
[2026-01-08 09:30:15] Epoch 45/100
[2026-01-08 09:30:15] Train Loss: 0.0234 | Val Loss: 0.0256
[2026-01-08 09:30:15] ADE: 15.3m | FDE: 38.6m
[2026-01-08 09:30:15] Goal Accuracy: 73.5%
[2026-01-08 09:30:15] ✓ 保存最佳模型
        """
        self.log_text.setText(log_text_sample)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        layout.addWidget(right_panel, 2)
        
        self.tabs.addTab(tab, '模型训练')
    
    def create_prediction_tab(self):
        """创建轨迹预测页面"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 左侧：预测控制
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel('<b>预测配置</b>'))
        
        # 输入配置
        input_group = QGroupBox('输入配置')
        input_layout = QVBoxLayout()
        
        input_layout.addWidget(QLabel('区域:'))
        region_combo = QComboBox()
        region_combo.addItems(['Scottish Highlands', 'Bohemian Forest'])
        input_layout.addWidget(region_combo)
        
        input_layout.addWidget(QLabel('历史长度:'))
        history_spin = QSpinBox()
        history_spin.setValue(10)
        history_spin.setSuffix(' 分钟')
        input_layout.addWidget(history_spin)
        
        input_layout.addWidget(QLabel('预测长度:'))
        future_spin = QSpinBox()
        future_spin.setValue(60)
        future_spin.setSuffix(' 分钟')
        input_layout.addWidget(future_spin)
        
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)
        
        # 模型选择
        pred_model_group = QGroupBox('模型选择')
        pred_model_layout = QVBoxLayout()
        
        for model in ['TerraTNT', 'YNet', 'PECNet', 'Trajectron++']:
            cb = QCheckBox(model)
            if model == 'TerraTNT':
                cb.setChecked(True)
            pred_model_layout.addWidget(cb)
        
        pred_model_group.setLayout(pred_model_layout)
        left_layout.addWidget(pred_model_group)
        
        # 预测按钮
        predict_btn = QPushButton('开始预测')
        predict_btn.setMinimumHeight(40)
        predict_btn.clicked.connect(self.predict_trajectory)
        left_layout.addWidget(predict_btn)
        
        # 性能指标
        metrics_group = QGroupBox('性能指标')
        metrics_layout = QVBoxLayout()
        
        metrics_table = QTableWidget(4, 2)
        metrics_table.setHorizontalHeaderLabels(['指标', '数值'])
        metrics_data = [
            ('ADE', '15.3 m'),
            ('FDE', '38.6 m'),
            ('Goal Acc', '73.5%'),
            ('推理时间', '45 ms'),
        ]
        
        for i, (key, value) in enumerate(metrics_data):
            metrics_table.setItem(i, 0, QTableWidgetItem(key))
            metrics_table.setItem(i, 1, QTableWidgetItem(value))
        
        metrics_table.resizeColumnsToContents()
        metrics_layout.addWidget(metrics_table)
        metrics_group.setLayout(metrics_layout)
        left_layout.addWidget(metrics_group)
        
        left_layout.addStretch()
        layout.addWidget(left_panel, 1)
        
        # 右侧：地图显示
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel('<b>轨迹可视化</b>'))
        
        # 地图画布
        self.map_canvas = MapCanvas(self, width=9, height=7)
        right_layout.addWidget(self.map_canvas)
        
        # 图例说明
        legend_layout = QHBoxLayout()
        legend_items = [
            ('蓝色', '历史轨迹'),
            ('红色', '预测轨迹'),
            ('绿色', '真实轨迹'),
            ('黄色', '候选目标'),
        ]
        
        for color, text in legend_items:
            label = QLabel(f'● {text}')
            if color == '蓝色':
                label.setStyleSheet('color: blue;')
            elif color == '红色':
                label.setStyleSheet('color: red;')
            elif color == '绿色':
                label.setStyleSheet('color: green;')
            elif color == '黄色':
                label.setStyleSheet('color: orange;')
            legend_layout.addWidget(label)
        
        legend_layout.addStretch()
        right_layout.addLayout(legend_layout)
        
        layout.addWidget(right_panel, 2)
        
        self.tabs.addTab(tab, '轨迹预测')
    
    def create_evaluation_tab(self):
        """创建模型评估页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel('<b>模型对比评估</b>'))
        
        # 对比表格
        compare_group = QGroupBox('模型性能对比')
        compare_layout = QVBoxLayout()
        
        compare_table = QTableWidget(7, 6)
        compare_table.setHorizontalHeaderLabels(
            ['模型', 'ADE (m)', 'FDE (m)', 'Goal Acc (%)', '推理时间 (ms)', '状态'])
        
        models_data = [
            ('TerraTNT', '15.3', '38.6', '73.5', '45', '✓ 已训练'),
            ('YNet', '25.4', '62.8', '45.7', '35', '✓ 已训练'),
            ('PECNet', '22.1', '58.3', '52.3', '42', '✓ 已训练'),
            ('Trajectron++', '20.8', '54.7', '56.8', '68', '✓ 已训练'),
            ('AgentFormer', '19.5', '51.2', '61.2', '85', '○ 训练中'),
            ('Social-LSTM', '38.7', '95.3', '28.5', '15', '○ 待训练'),
            ('Constant Vel', '45.2', '128.5', '12.3', '0.1', '✓ 已实现'),
        ]
        
        for i, row_data in enumerate(models_data):
            for j, value in enumerate(row_data):
                item = QTableWidgetItem(value)
                if i == 0:  # TerraTNT行高亮
                    item.setBackground(QColor(200, 255, 200))
                compare_table.setItem(i, j, item)
        
        compare_table.resizeColumnsToContents()
        compare_layout.addWidget(compare_table)
        compare_group.setLayout(compare_layout)
        layout.addWidget(compare_group)
        
        # 可视化对比
        vis_group = QGroupBox('可视化对比')
        vis_layout = QHBoxLayout()
        
        # ADE对比柱状图
        ade_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        ade_ax = ade_canvas.figure.add_subplot(111)
        
        models = ['TerraTNT', 'YNet', 'PECNet', 'Traj++', 'Agent\nFormer', 'Social\nLSTM', 'CV']
        ade_values = [15.3, 25.4, 22.1, 20.8, 19.5, 38.7, 45.2]
        colors = ['green' if i == 0 else 'skyblue' for i in range(len(models))]
        
        ade_ax.bar(models, ade_values, color=colors, edgecolor='black')
        ade_ax.set_ylabel('ADE (m)')
        ade_ax.set_title('平均位移误差对比')
        ade_ax.grid(True, alpha=0.3, axis='y')
        
        vis_layout.addWidget(ade_canvas)
        
        # FDE对比柱状图
        fde_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        fde_ax = fde_canvas.figure.add_subplot(111)
        
        fde_values = [38.6, 62.8, 58.3, 54.7, 51.2, 95.3, 128.5]
        
        fde_ax.bar(models, fde_values, color=colors, edgecolor='black')
        fde_ax.set_ylabel('FDE (m)')
        fde_ax.set_title('最终位移误差对比')
        fde_ax.grid(True, alpha=0.3, axis='y')
        
        vis_layout.addWidget(fde_canvas)
        
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QPushButton('导出报告'))
        btn_layout.addWidget(QPushButton('生成图表'))
        btn_layout.addWidget(QPushButton('详细分析'))
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.tabs.addTab(tab, '模型评估')
    
    def create_settings_tab(self):
        """创建设置页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel('<b>系统设置</b>'))
        
        # GPU设置
        gpu_group = QGroupBox('GPU设置')
        gpu_layout = QVBoxLayout()
        
        gpu_layout.addWidget(QLabel('GPU设备: NVIDIA RTX 5060 (8GB)'))
        gpu_layout.addWidget(QLabel('CUDA版本: 13.0'))
        gpu_layout.addWidget(QLabel('显存使用: 1.2GB / 8.0GB'))
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # 路径设置
        path_group = QGroupBox('路径设置')
        path_layout = QVBoxLayout()
        
        paths = [
            ('数据目录', '/home/zmc/文档/programwork/data'),
            ('模型目录', '/home/zmc/文档/programwork/models/saved'),
            ('日志目录', '/home/zmc/文档/programwork/runs'),
        ]
        
        for label, path in paths:
            row = QHBoxLayout()
            row.addWidget(QLabel(f'{label}:'))
            line_edit = QLineEdit(path)
            row.addWidget(line_edit)
            row.addWidget(QPushButton('浏览'))
            path_layout.addLayout(row)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        layout.addStretch()
        
        self.tabs.addTab(tab, '设置')
    
    # 回调函数
    def load_dataset(self):
        self.statusBar().showMessage('加载数据集...')
        QMessageBox.information(self, '提示', '数据集加载成功！')
    
    def start_training(self):
        self.statusBar().showMessage('开始训练...')
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        QMessageBox.information(self, '提示', '训练已开始！')
    
    def predict_trajectory(self):
        self.statusBar().showMessage('预测中...')
        QMessageBox.information(self, '提示', '轨迹预测完成！')
    
    def evaluate_model(self):
        self.statusBar().showMessage('评估中...')
        QMessageBox.information(self, '提示', '模型评估完成！')

def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = TerraTNTMainWindow()
    window.show()
    
    # 保存截图
    print("正在生成界面截图...")
    
    # 等待界面渲染
    app.processEvents()
    
    # 保存每个标签页的截图
    for i in range(window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        app.processEvents()
        
        # 截图
        pixmap = window.grab()
        tab_name = window.tabs.tabText(i)
        filename = f'/home/zmc/文档/programwork/docs/ui_screenshot_{i+1}_{tab_name}.png'
        pixmap.save(filename)
        print(f"✓ 保存截图: {filename}")
    
    print("✅ 所有截图已保存！")
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
