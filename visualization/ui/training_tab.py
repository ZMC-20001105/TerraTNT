#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tab 4: 模型训练"""
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QComboBox, QLabel, QPushButton, QTextEdit, QProgressBar,
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QFileDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TrainWorker(QThread):
    """训练工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, cmd_args):
        super().__init__()
        self.cmd_args = cmd_args
        self.process = None

    def run(self):
        try:
            self.progress.emit(f"$ {' '.join(self.cmd_args)}\n")
            self.process = subprocess.Popen(
                self.cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(PROJECT_ROOT),
            )
            for line in self.process.stdout:
                self.progress.emit(line.rstrip())
            self.process.wait()
            if self.process.returncode == 0:
                self.finished.emit(True, "训练完成")
            else:
                self.finished.emit(False, f"退出码: {self.process.returncode}")
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        if self.process:
            self.process.terminate()


class TrainingTab(QWidget):
    """模型训练标签页"""

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- 左: 训练配置 ---
        left = QWidget()
        ll = QVBoxLayout(left)

        # 数据
        dg = QGroupBox("数据")
        dl = QFormLayout(dg)
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("自动检测...")
        dl.addRow("数据目录:", self.data_dir_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_data)
        dl.addRow("", browse_btn)
        self.region_combo = QComboBox()
        dl.addRow("区域:", self.region_combo)
        ll.addWidget(dg)

        # 模型
        mg = QGroupBox("模型")
        ml = QFormLayout(mg)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'TerraTNT', 'V3_Waypoint', 'V4_WP_Spatial',
            'V6_Autoreg', 'V6R_Robust', 'V7_ConfGate',
            'LSTM_only', 'LSTM_Env_Goal', 'Seq2Seq_Attn', 'MLP',
        ])
        ml.addRow("模型:", self.model_combo)
        ll.addWidget(mg)

        # 超参数
        pg = QGroupBox("超参数")
        pl = QFormLayout(pg)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        pl.addRow("Batch size:", self.batch_spin)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000)
        self.epoch_spin.setValue(100)
        pl.addRow("Epochs:", self.epoch_spin)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        pl.addRow("Learning rate:", self.lr_spin)
        self.gpu_check = QCheckBox("使用GPU")
        self.gpu_check.setChecked(True)
        pl.addRow("", self.gpu_check)
        ll.addWidget(pg)

        # 混合区域训练
        mix_g = QGroupBox("混合区域训练")
        mix_l = QVBoxLayout(mix_g)
        self.mix_check = QCheckBox("启用多区域联合训练")
        mix_l.addWidget(self.mix_check)
        self.mix_regions = QLineEdit()
        self.mix_regions.setPlaceholderText("bohemian_forest,scottish_highlands")
        mix_l.addWidget(self.mix_regions)
        ll.addWidget(mix_g)

        # 进度
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        ll.addWidget(self.progress_bar)

        # 按钮
        btn = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self._start_train)
        btn.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_train)
        btn.addWidget(self.stop_btn)
        ll.addLayout(btn)

        ll.addStretch()
        splitter.addWidget(left)

        # --- 右: 日志 ---
        right = QWidget()
        rl = QVBoxLayout(right)
        log_g = QGroupBox("训练日志")
        log_l = QVBoxLayout(log_g)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        log_l.addWidget(self.log_text)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.log_text.clear)
        log_l.addWidget(clear_btn)
        log_g.setLayout(log_l)
        rl.addWidget(log_g)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    def set_regions(self, regions):
        self.region_combo.clear()
        self.region_combo.addItems(regions)

    def _browse_data(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据目录", str(PROJECT_ROOT / 'data'))
        if d:
            self.data_dir_edit.setText(d)

    def _start_train(self):
        model = self.model_combo.currentText()
        region = self.region_combo.currentText()

        # 确定数据目录
        data_dir = self.data_dir_edit.text().strip()
        if not data_dir:
            for base in ['final_dataset_v1', 'complete_dataset_10s']:
                d = PROJECT_ROOT / 'data' / 'processed' / base / region
                if d.exists():
                    data_dir = str(d)
                    break

        if not data_dir:
            self.log_text.append("未找到数据目录")
            return

        # 根据模型类型选择训练脚本
        if model in ('LSTM_only', 'LSTM_Env_Goal', 'Seq2Seq_Attn', 'MLP'):
            script = str(PROJECT_ROOT / 'scripts' / 'train_eval_all_baselines.py')
        elif model.startswith('V'):
            script = str(PROJECT_ROOT / 'scripts' / 'train_incremental_models.py')
        else:
            script = str(PROJECT_ROOT / 'scripts' / 'train_terratnt_10s.py')

        cmd = [
            'conda', 'run', '-n', 'torch-sm120', 'python', script,
            '--data-dir', data_dir,
            '--batch-size', str(self.batch_spin.value()),
            '--epochs', str(self.epoch_spin.value()),
            '--lr', str(self.lr_spin.value()),
        ]
        if self.gpu_check.isChecked():
            cmd.extend(['--device', 'cuda'])

        self.log_text.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.worker = TrainWorker(cmd)
        self.worker.progress.connect(self.log_text.append)
        self.worker.finished.connect(self._on_done)
        self.worker.start()

    def _stop_train(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self._on_done(False, "用户停止")

    def _on_done(self, success, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        status = "OK" if success else "FAIL"
        self.log_text.append(f"\n[{status}] {msg}")
