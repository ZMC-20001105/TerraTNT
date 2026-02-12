#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tab 2: Phase评估 + 跨区域对比"""
import sys
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QComboBox, QLabel, QPushButton, QCheckBox, QTextEdit,
    QProgressBar, QFormLayout, QTableWidget, QTableWidgetItem,
    QFileDialog, QLineEdit, QTabWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import numpy as np
import matplotlib
import matplotlib.font_manager as _fm
_cjk = None
for _c in ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Droid Sans Fallback', 'WenQuanYi Micro Hei']:
    if any(f.name == _c for f in _fm.fontManager.ttflist):
        _cjk = _c
        break
matplotlib.rcParams['font.sans-serif'] = [_cjk, 'DejaVu Sans'] if _cjk else ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class EvalWorker(QThread):
    """后台评估线程"""
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
                self.finished.emit(True, "评估完成")
            else:
                self.finished.emit(False, f"退出码: {self.process.returncode}")
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        if self.process:
            self.process.terminate()


class EvaluationTab(QWidget):
    """Phase评估 + 跨区域对比"""

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- 左: 配置 ---
        left = QWidget()
        ll = QVBoxLayout(left)

        # 评估区域
        dg = QGroupBox("评估数据")
        dl = QFormLayout(dg)
        self.region_combo = QComboBox()
        dl.addRow("区域:", self.region_combo)

        self.traj_dir_edit = QLineEdit()
        self.traj_dir_edit.setPlaceholderText("自动检测...")
        dl.addRow("轨迹目录:", self.traj_dir_edit)

        self.split_edit = QLineEdit()
        self.split_edit.setPlaceholderText("自动检测...")
        dl.addRow("Split文件:", self.split_edit)

        browse_btn = QPushButton("浏览Split...")
        browse_btn.clicked.connect(self._browse_split)
        dl.addRow("", browse_btn)
        ll.addWidget(dg)

        # Phase选择
        pg = QGroupBox("Phase选择")
        pgl = QVBoxLayout(pg)
        self.phase_checks = {}
        for pid in ['P1a', 'P1b', 'P2a', 'P2b', 'P2c', 'P3a', 'P3b']:
            cb = QCheckBox(pid)
            cb.setChecked(pid in ('P1a', 'P3a'))
            pgl.addWidget(cb)
            self.phase_checks[pid] = cb
        ll.addWidget(pg)

        # 参数
        param_g = QGroupBox("参数")
        param_l = QFormLayout(param_g)
        self.batch_spin = QLineEdit("16")
        param_l.addRow("Batch size:", self.batch_spin)
        self.fraction_edit = QLineEdit("1.0")
        param_l.addRow("Sample fraction:", self.fraction_edit)
        ll.addWidget(param_g)

        # 进度
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        ll.addWidget(self.progress_bar)

        # 按钮
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("开始评估")
        self.run_btn.clicked.connect(self._start_eval)
        btn_layout.addWidget(self.run_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_eval)
        btn_layout.addWidget(self.stop_btn)
        self.load_results_btn = QPushButton("加载已有结果")
        self.load_results_btn.clicked.connect(self._load_results)
        btn_layout.addWidget(self.load_results_btn)
        ll.addLayout(btn_layout)

        # 快速跨区域评估
        cross_g = QGroupBox("快速跨区域评估")
        cross_l = QVBoxLayout(cross_g)
        cross_l.addWidget(QLabel("训练区域 → 测试区域"))
        self.cross_train_combo = QComboBox()
        self.cross_test_combo = QComboBox()
        row = QHBoxLayout()
        row.addWidget(self.cross_train_combo)
        row.addWidget(QLabel("→"))
        row.addWidget(self.cross_test_combo)
        cross_l.addLayout(row)
        self.cross_btn = QPushButton("运行跨区域评估")
        self.cross_btn.clicked.connect(self._start_cross_eval)
        cross_l.addWidget(self.cross_btn)
        ll.addWidget(cross_g)

        ll.addStretch()
        splitter.addWidget(left)

        # --- 右: 结果 ---
        right = QWidget()
        rl = QVBoxLayout(right)

        self.result_tabs = QTabWidget()

        # 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.result_tabs.addTab(self.log_text, "日志")

        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(
            ["Phase", "模型", "ADE(m)", "FDE(m)", "样本数"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_tabs.addTab(self.result_table, "结果表")

        # 跨区域对比图
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.cross_fig = Figure(figsize=(8, 5), dpi=100)
        self.cross_ax = self.cross_fig.add_subplot(111)
        self.cross_canvas = FigureCanvasQTAgg(self.cross_fig)
        self.result_tabs.addTab(self.cross_canvas, "跨区域对比")

        # 使用说明
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h3 style="color:#42a5f5">📖 评估模块使用说明</h3>
        <p style="color:#ccc">本模块用于对已训练的轨迹预测模型进行系统性Phase评估，
        测试模型在不同终点先验条件下的预测性能。</p>

        <h4 style="color:#69f0ae">Phase评估体系</h4>
        <table style="color:#ddd; border-collapse:collapse; width:100%">
            <tr style="background:#3e3e42">
                <td style="padding:4px"><b>P1a</b></td>
                <td style="padding:4px">精确终点(域内) — σ=1km高斯先验，GT终点在候选集中</td></tr>
            <tr><td style="padding:4px"><b>P1b</b></td>
                <td style="padding:4px">精确终点(OOD) — 同上，但终点为域外(未见过的目标)</td></tr>
            <tr style="background:#3e3e42">
                <td style="padding:4px"><b>P2a</b></td>
                <td style="padding:4px">区域先验(σ=10km) — 模糊终点，覆盖~20km区域</td></tr>
            <tr><td style="padding:4px"><b>P2b</b></td>
                <td style="padding:4px">区域先验(σ=15km) — 更模糊的终点先验</td></tr>
            <tr style="background:#3e3e42">
                <td style="padding:4px"><b>P2c</b></td>
                <td style="padding:4px">区域先验(偏移5km) — σ=10km + 中心偏移5km</td></tr>
            <tr><td style="padding:4px"><b>P3a</b></td>
                <td style="padding:4px">无先验(直行) — 沿运动方向扇形分布，历史方向与终点夹角<30°</td></tr>
            <tr style="background:#3e3e42">
                <td style="padding:4px"><b>P3b</b></td>
                <td style="padding:4px">无先验(转弯) — 同上，但夹角>60°的转弯样本</td></tr>
        </table>

        <h4 style="color:#69f0ae">使用流程</h4>
        <ol style="color:#ddd">
            <li>选择区域 → 轨迹目录和Split文件路径自动填充</li>
            <li>勾选要评估的Phase (建议先选P1a快速验证)</li>
            <li>调整参数: Sample fraction=0.1可快速预览，=1.0为完整评估</li>
            <li>点击「开始评估」→ 在日志标签中查看实时进度</li>
            <li>完成后在「结果表」中查看各模型的ADE/FDE指标</li>
        </ol>

        <h4 style="color:#69f0ae">⚠️ 注意事项</h4>
        <ul style="color:#ddd">
            <li>首次运行需要扫描所有pkl文件，<b>可能需要1-2分钟</b>加载</li>
            <li>完整评估(fraction=1.0)在GPU上约需10-30分钟/Phase</li>
            <li>建议先用 sample_fraction=0.1 快速验证流程</li>
            <li>评估结果保存在 outputs/evaluation/gui_区域名/ 目录</li>
        </ul>

        <h4 style="color:#69f0ae">跨区域评估</h4>
        <p style="color:#ddd">用于测试模型泛化能力: 用训练区域的模型权重在测试区域的数据上评估。
        选择训练区域和测试区域后点击运行即可。</p>
        """)
        self.result_tabs.addTab(help_text, "📖 使用说明")

        rl.addWidget(self.result_tabs)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    def set_regions(self, regions):
        for combo in [self.region_combo, self.cross_train_combo, self.cross_test_combo]:
            combo.clear()
            combo.addItems(regions)
        if len(regions) >= 2:
            self.cross_train_combo.setCurrentIndex(0)
            self.cross_test_combo.setCurrentIndex(1)
        # 自动检测路径
        self.region_combo.currentTextChanged.connect(self._auto_detect_paths)
        if regions:
            self._auto_detect_paths(regions[0])

    def _auto_detect_paths(self, region):
        """根据区域自动填充轨迹目录和split文件路径"""
        if not region:
            return
        # 轨迹目录
        traj_candidates = [
            PROJECT_ROOT / 'data' / 'processed' / 'final_dataset_v1' / region,
            PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / region,
        ]
        for c in traj_candidates:
            if c.exists():
                self.traj_dir_edit.setText(str(c))
                self.traj_dir_edit.setStyleSheet("color: #4fc3f7;")
                break
        else:
            self.traj_dir_edit.clear()
            self.traj_dir_edit.setPlaceholderText(f"未找到 {region} 轨迹数据")
        # split文件
        split_candidates = [
            PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region / 'fas_splits.json',
            PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region / 'fas_splits_trajlevel.json',
        ]
        for c in split_candidates:
            if c.exists():
                self.split_edit.setText(str(c))
                self.split_edit.setStyleSheet("color: #4fc3f7;")
                break
        else:
            self.split_edit.clear()
            self.split_edit.setPlaceholderText(f"未找到 {region} split文件")

    def _browse_split(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择Split文件", str(PROJECT_ROOT / 'data'), "JSON (*.json)")
        if path:
            self.split_edit.setText(path)

    def _build_eval_cmd(self, region=None):
        """构建评估命令"""
        phases = [pid for pid, cb in self.phase_checks.items() if cb.isChecked()]
        if not phases:
            return None

        region = region or self.region_combo.currentText()
        if not region:
            return None

        # 自动检测路径
        traj_dir = self.traj_dir_edit.text().strip()
        if not traj_dir:
            candidates = [
                PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s' / region,
                PROJECT_ROOT / 'data' / 'processed' / 'final_dataset_v1' / region,
                PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo',
            ]
            for c in candidates:
                if c.exists():
                    traj_dir = str(c)
                    break
            if not traj_dir:
                self.log_text.append(f"未找到 {region} 的轨迹数据")
                return None

        split_file = self.split_edit.text().strip()
        if not split_file:
            candidates = [
                PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region / 'fas_splits_trajlevel.json',
                PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region / 'fas_splits.json',
                PROJECT_ROOT / 'outputs' / 'dataset_experiments' / 'D1_optimal_combo' / 'fas_splits_full_phases.json',
            ]
            for c in candidates:
                if c.exists():
                    split_file = str(c)
                    break
            if not split_file:
                self.log_text.append(f"未找到 {region} 的split文件")
                return None

        output_dir = str(PROJECT_ROOT / 'outputs' / 'evaluation' / f'gui_{region}')

        cmd = [
            'conda', 'run', '-n', 'torch-sm120', 'python',
            str(PROJECT_ROOT / 'scripts' / 'evaluate_phases_v2.py'),
            '--traj_dir', traj_dir,
            '--fas_split_file', split_file,
            '--output_dir', output_dir,
            '--phases', *phases,
            '--batch_size', self.batch_spin.text(),
            '--sample_fraction', self.fraction_edit.text(),
        ]
        return cmd

    def _start_eval(self):
        cmd = self._build_eval_cmd()
        if cmd is None:
            return
        self.log_text.clear()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.worker = EvalWorker(cmd)
        self.worker.progress.connect(self.log_text.append)
        self.worker.finished.connect(self._on_eval_done)
        self.worker.start()

    def _stop_eval(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self._on_eval_done(False, "用户停止")

    def _on_eval_done(self, success, msg):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        status = "OK" if success else "FAIL"
        self.log_text.append(f"\n[{status}] {msg}")
        if success:
            self._load_results()

    def _start_cross_eval(self):
        test_region = self.cross_test_combo.currentText()
        if not test_region:
            return
        cmd = self._build_eval_cmd(region=test_region)
        if cmd is None:
            return
        # 修改输出目录
        train_region = self.cross_train_combo.currentText()
        output_dir = str(PROJECT_ROOT / 'outputs' / 'evaluation' / f'cross_{train_region}_to_{test_region}')
        # 替换output_dir
        for i, arg in enumerate(cmd):
            if arg == '--output_dir' and i + 1 < len(cmd):
                cmd[i + 1] = output_dir
        self.log_text.clear()
        self.log_text.append(f"跨区域: {train_region} → {test_region}")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.worker = EvalWorker(cmd)
        self.worker.progress.connect(self.log_text.append)
        self.worker.finished.connect(self._on_eval_done)
        self.worker.start()

    def _find_results_file(self, region=None):
        """搜索评估结果文件"""
        region = region or self.region_combo.currentText()
        candidates = [
            PROJECT_ROOT / 'outputs' / 'evaluation' / f'gui_{region}' / 'phase_v2_results.json',
            PROJECT_ROOT / 'outputs' / 'evaluation' / 'phase_v2' / 'phase_v2_results.json',
        ]
        for c in candidates:
            if c.exists():
                import json
                with open(c) as f:
                    data = json.load(f)
                if data:  # 非空
                    return c, data
        return None, None

    def _load_results(self):
        """加载评估结果到表格 + 跨Phase对比图"""
        path, data = self._find_results_file()
        if data is None:
            self.log_text.append("未找到有效的评估结果文件")
            return
        try:
            self.log_text.append(f"加载结果: {path}")
            self.result_table.setRowCount(0)
            row = 0
            for phase_id, pdata in sorted(data.items()):
                pname = pdata.get('name', phase_id)
                for mname, mdata in sorted(pdata.get('models', {}).items(),
                                            key=lambda x: x[1].get('ade_mean', 1e9)):
                    self.result_table.insertRow(row)
                    self.result_table.setItem(row, 0, QTableWidgetItem(pname))
                    self.result_table.setItem(row, 1, QTableWidgetItem(mname))
                    self.result_table.setItem(row, 2, QTableWidgetItem(
                        f"{mdata.get('ade_mean', 0):.0f}"))
                    self.result_table.setItem(row, 3, QTableWidgetItem(
                        f"{mdata.get('fde_mean', 0):.0f}"))
                    self.result_table.setItem(row, 4, QTableWidgetItem(
                        str(mdata.get('n_samples', 0))))
                    row += 1
            self.result_tabs.setCurrentIndex(1)
            # 绘制跨Phase对比图
            self._plot_phase_comparison(data)
        except Exception as e:
            self.log_text.append(f"加载结果失败: {e}")

    def _plot_phase_comparison(self, data):
        """绘制跨Phase ADE对比热力图/柱状图"""
        self.cross_ax.clear()
        if not data:
            self.cross_canvas.draw()
            return

        # 收集所有Phase和模型
        phases = sorted(data.keys())
        all_models = set()
        for pid in phases:
            all_models.update(data[pid].get('models', {}).keys())
        models = sorted(all_models)

        if not phases or not models:
            self.cross_ax.text(0.5, 0.5, '无评估数据', ha='center', va='center',
                               color='#777', fontsize=12, transform=self.cross_ax.transAxes)
            self.cross_canvas.draw()
            return

        # 构建ADE矩阵 (models x phases)
        ade_matrix = np.full((len(models), len(phases)), np.nan)
        for j, pid in enumerate(phases):
            for i, mn in enumerate(models):
                mdata = data[pid].get('models', {}).get(mn)
                if mdata:
                    ade_matrix[i, j] = mdata.get('ade_mean', np.nan)

        # 分组柱状图: 每个Phase一组, 每个模型一个柱子
        x = np.arange(len(phases))
        n_models = len(models)
        bar_w = 0.8 / max(n_models, 1)
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0',
                  '#00BCD4', '#FF5722', '#795548', '#607D8B', '#CDDC39',
                  '#F44336', '#3F51B5']

        for i, mn in enumerate(models):
            vals = ade_matrix[i, :]
            offset = (i - n_models / 2 + 0.5) * bar_w
            mask = ~np.isnan(vals)
            c = colors[i % len(colors)]
            self.cross_ax.bar(x[mask] + offset, vals[mask], bar_w * 0.9,
                              label=mn, color=c, alpha=0.85)

        phase_labels = []
        for pid in phases:
            pname = data[pid].get('name', pid)
            # 缩短名称
            short = pname.replace('Phase', 'P').replace('精确终点', '精确')
            short = short.replace('区域先验', '区域').replace('无先验', '无先验')
            phase_labels.append(short[:15])

        self.cross_ax.set_xticks(x)
        self.cross_ax.set_xticklabels(phase_labels, fontsize=7, rotation=15, ha='right')
        self.cross_ax.set_ylabel('ADE (m)', fontsize=9)
        self.cross_ax.set_title('跨Phase模型性能对比', fontsize=11)
        self.cross_ax.legend(fontsize=6, loc='upper left', ncol=2)
        self.cross_ax.grid(True, alpha=0.2, axis='y')
        self.cross_fig.tight_layout()
        self.cross_canvas.draw()

    def plot_cross_comparison(self, in_domain_results, cross_results):
        """绘制跨区域对比柱状图"""
        self.cross_ax.clear()
        if not in_domain_results or not cross_results:
            self.cross_canvas.draw()
            return
        models = sorted(set(in_domain_results.keys()) & set(cross_results.keys()))
        x = np.arange(len(models))
        w = 0.35
        in_vals = [in_domain_results[m] for m in models]
        cross_vals = [cross_results[m] for m in models]
        self.cross_ax.bar(x - w / 2, in_vals, w, label='域内', color='#4CAF50')
        self.cross_ax.bar(x + w / 2, cross_vals, w, label='跨区域', color='#FF5722')
        self.cross_ax.set_xticks(x)
        self.cross_ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        self.cross_ax.set_ylabel('ADE (m)')
        self.cross_ax.legend()
        self.cross_ax.set_title('域内 vs 跨区域对比')
        self.cross_fig.tight_layout()
        self.cross_canvas.draw()
