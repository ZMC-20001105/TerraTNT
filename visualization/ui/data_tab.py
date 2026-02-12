#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tab 3: 数据管理 — 数据集浏览、生成、统计"""
import sys
import json
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QComboBox, QLabel, QPushButton, QTextEdit, QProgressBar,
    QFormLayout, QSpinBox, QListWidget, QFileDialog, QTabWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DataGenWorker(QThread):
    """轨迹数据生成工作线程"""
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
                self.finished.emit(True, "数据生成完成")
            else:
                self.finished.emit(False, f"退出码: {self.process.returncode}")
        except Exception as e:
            self.finished.emit(False, str(e))

    def stop(self):
        if self.process:
            self.process.terminate()


class DataTab(QWidget):
    """数据管理标签页"""

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- 左: 数据生成 ---
        left = QWidget()
        ll = QVBoxLayout(left)

        # 区域选择
        rg = QGroupBox("区域配置")
        rl = QFormLayout(rg)
        self.region_combo = QComboBox()
        rl.addRow("区域:", self.region_combo)
        self.num_traj_spin = QSpinBox()
        self.num_traj_spin.setRange(1, 100000)
        self.num_traj_spin.setValue(1000)
        self.num_traj_spin.setSuffix(" 条")
        rl.addRow("轨迹数:", self.num_traj_spin)
        ll.addWidget(rg)

        # 生成控制
        gen_g = QGroupBox("轨迹生成")
        gen_l = QVBoxLayout(gen_g)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        gen_l.addWidget(self.progress_bar)
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gen_l.addWidget(self.status_label)
        btn_row = QHBoxLayout()
        self.gen_btn = QPushButton("生成轨迹")
        self.gen_btn.clicked.connect(self._start_gen)
        btn_row.addWidget(self.gen_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_gen)
        btn_row.addWidget(self.stop_btn)
        gen_l.addLayout(btn_row)
        ll.addWidget(gen_g)

        # FAS Split生成
        split_g = QGroupBox("FAS Split")
        split_l = QVBoxLayout(split_g)
        self.split_btn = QPushButton("生成 fas_splits.json")
        self.split_btn.clicked.connect(self._gen_splits)
        split_l.addWidget(self.split_btn)
        self.trajlevel_btn = QPushButton("生成 trajlevel split")
        self.trajlevel_btn.clicked.connect(self._gen_trajlevel)
        split_l.addWidget(self.trajlevel_btn)
        ll.addWidget(split_g)

        # 环境数据检查 & 下载
        env_g = QGroupBox("环境数据管理")
        env_l = QVBoxLayout(env_g)
        self.env_status = QTextEdit()
        self.env_status.setReadOnly(True)
        self.env_status.setMaximumHeight(120)
        self.env_status.setStyleSheet("font-size: 10px; font-family: monospace; background: #1e1e1e; color: #ccc;")
        env_l.addWidget(self.env_status)
        check_btn = QPushButton("检查环境数据完整性")
        check_btn.clicked.connect(self._check_env_data)
        env_l.addWidget(check_btn)
        dl_row = QHBoxLayout()
        self.dl_road_btn = QPushButton("下载路网 (OSM)")
        self.dl_road_btn.setToolTip("从OpenStreetMap下载道路网络并栅格化")
        self.dl_road_btn.clicked.connect(self._download_road)
        dl_row.addWidget(self.dl_road_btn)
        self.dl_gee_btn = QPushButton("下载DEM/LULC (GEE)")
        self.dl_gee_btn.setToolTip("通过Google Earth Engine下载DEM、坡度、坡向、土地覆盖")
        self.dl_gee_btn.clicked.connect(self._download_gee)
        dl_row.addWidget(self.dl_gee_btn)
        env_l.addLayout(dl_row)
        ll.addWidget(env_g)

        ll.addStretch()
        splitter.addWidget(left)

        # --- 右: 数据集信息 ---
        right = QWidget()
        rl2 = QVBoxLayout(right)

        self.info_tabs = QTabWidget()

        # 数据集列表
        list_w = QWidget()
        list_l = QVBoxLayout(list_w)
        self.dataset_list = QListWidget()
        self.dataset_list.currentRowChanged.connect(self._on_dataset_selected)
        list_l.addWidget(self.dataset_list)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_datasets)
        list_l.addWidget(refresh_btn)
        self.info_tabs.addTab(list_w, "数据集")

        # 数据集概览
        overview = QTextEdit()
        overview.setReadOnly(True)
        overview.setHtml("""
        <h3 style="color:#42a5f5">📊 数据集概览</h3>
        <h4 style="color:#69f0ae">项目背景</h4>
        <p style="color:#ddd">本项目面向多星协同对地观测系统，预测地面目标在观测空窗期的未来位置。
        预测时域60分钟，空间范围数十公里。</p>

        <h4 style="color:#69f0ae">数据来源</h4>
        <table style="color:#ddd; border-collapse:collapse; width:100%">
            <tr style="background:#3e3e42"><td style="padding:4px"><b>环境数据</b></td><td style="padding:4px">DEM, Slope, Aspect, LULC (ESA WorldCover), OSM Road</td></tr>
            <tr><td style="padding:4px"><b>CRS</b></td><td style="padding:4px">UTM (EPSG:32633 BF / EPSG:32630 SH)</td></tr>
            <tr style="background:#3e3e42"><td style="padding:4px"><b>环境地图</b></td><td style="padding:4px">18通道 128×128 像素, 覆盖100km×100km</td></tr>
            <tr><td style="padding:4px"><b>轨迹生成</b></td><td style="padding:4px">分层A*路径规划 + XGBoost速度模型</td></tr>
        </table>

        <h4 style="color:#69f0ae">区域说明</h4>
        <table style="color:#ddd; border-collapse:collapse; width:100%">
            <tr style="background:#3e3e42"><td style="padding:4px"><b>bohemian_forest</b></td><td style="padding:4px">波西米亚森林 (捷克/德国/奥地利边境), 山地森林地形</td></tr>
            <tr><td style="padding:4px"><b>scottish_highlands</b></td><td style="padding:4px">苏格兰高地 (英国), 丘陵草地地形, 用于跨域泛化测试</td></tr>
        </table>

        <h4 style="color:#69f0ae">样本结构</h4>
        <table style="color:#ddd; border-collapse:collapse; width:100%">
            <tr style="background:#3e3e42"><td style="padding:4px"><b>history_feat_26d</b></td><td style="padding:4px">(90, 26) 历史特征: dx/dy + 环境特征, 15分钟</td></tr>
            <tr><td style="padding:4px"><b>future_rel</b></td><td style="padding:4px">(360, 2) 未来轨迹相对坐标 (km), 60分钟</td></tr>
            <tr style="background:#3e3e42"><td style="padding:4px"><b>env_map_100km</b></td><td style="padding:4px">(18, 128, 128) 环境栅格地图</td></tr>
            <tr><td style="padding:4px"><b>goal_rel</b></td><td style="padding:4px">(2,) 目标点相对坐标 (km)</td></tr>
            <tr style="background:#3e3e42"><td style="padding:4px"><b>current_pos_abs</b></td><td style="padding:4px">(2,) 当前位置UTM绝对坐标</td></tr>
        </table>

        <h4 style="color:#69f0ae">车辆类型 (4种)</h4>
        <ul style="color:#ddd">
            <li><b>type1</b>: 轻型越野 (v_max=18m/s, slope_max=30°)</li>
            <li><b>type2</b>: 中型车辆 (v_max=22m/s, slope_max=25°)</li>
            <li><b>type3</b>: 重型车辆 (v_max=25m/s, slope_max=20°)</li>
            <li><b>type4</b>: 公路车辆 (v_max=28m/s, slope_max=15°)</li>
        </ul>

        <h4 style="color:#69f0ae">战术意图 (3种)</h4>
        <ul style="color:#ddd">
            <li><b>intent1</b>: 快速机动 — 优先最短路径</li>
            <li><b>intent2</b>: 隐蔽行进 — 优先植被遮蔽</li>
            <li><b>intent3</b>: 地形利用 — 优先有利地形</li>
        </ul>
        """)
        self.info_tabs.insertTab(0, overview, "📊 概览")
        self.info_tabs.setCurrentIndex(0)

        # 详情
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.info_tabs.addTab(self.detail_text, "详情")

        # 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.info_tabs.addTab(self.log_text, "日志")

        rl2.addWidget(self.info_tabs)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def set_regions(self, regions):
        self.region_combo.clear()
        self.region_combo.addItems(regions)

    def _refresh_datasets(self):
        self.dataset_list.clear()
        data_dirs = [
            PROJECT_ROOT / 'data' / 'processed' / 'final_dataset_v1',
            PROJECT_ROOT / 'data' / 'processed' / 'complete_dataset_10s',
        ]
        for base in data_dirs:
            if not base.exists():
                continue
            for region_dir in sorted(base.iterdir()):
                if region_dir.is_dir():
                    pkls = list(region_dir.glob('*.pkl'))
                    if pkls:
                        self.dataset_list.addItem(
                            f"{base.name}/{region_dir.name} ({len(pkls)} files)")

    def _on_dataset_selected(self, row):
        if row < 0:
            return
        text = self.dataset_list.item(row).text()
        parts = text.split('/')
        if len(parts) < 2:
            return
        base_name = parts[0]
        region = parts[1].split(' ')[0]

        data_dir = PROJECT_ROOT / 'data' / 'processed' / base_name / region
        lines = [f"数据集: {data_dir}\n"]

        pkls = sorted(data_dir.glob('*.pkl'))
        lines.append(f"PKL文件: {len(pkls)}")

        # 检查fas_splits
        splits_dir = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region
        for sf in ['fas_splits.json', 'fas_splits_trajlevel.json']:
            sp = splits_dir / sf
            if sp.exists():
                try:
                    with open(sp) as f:
                        sd = json.load(f)
                    meta = sd.get('metadata', {})
                    lines.append(f"\n{sf}:")
                    lines.append(f"  总轨迹: {meta.get('total_trajectories', '?')}")
                    lines.append(f"  总样本: {meta.get('total_samples', '?')}")
                    for key in ['fas1', 'fas2', 'fas3']:
                        if key in sd:
                            lines.append(f"  {key}: {sd[key].get('num_samples', '?')} samples")
                except Exception as e:
                    lines.append(f"  读取失败: {e}")

        # 统计文件
        stats_path = data_dir / 'dataset_stats.json'
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                lines.append(f"\n统计:")
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        lines.append(f"  {k}: {v}")
            except:
                pass

        self.detail_text.setPlainText('\n'.join(lines))
        self.info_tabs.setCurrentIndex(1)

    def _start_gen(self):
        region = self.region_combo.currentText()
        if not region:
            return
        num = self.num_traj_spin.value()
        cmd = [
            'conda', 'run', '-n', 'torch-sm120', 'python',
            str(PROJECT_ROOT / 'utils' / 'trajectory_generation' / 'trajectory_generator_v2.py'),
            '--region', region, '--num_trajectories', str(num),
        ]
        self.log_text.clear()
        self.gen_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"生成中: {region} x {num}...")
        self.worker = DataGenWorker(cmd)
        self.worker.progress.connect(self.log_text.append)
        self.worker.finished.connect(self._on_gen_done)
        self.worker.start()

    def _stop_gen(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self._on_gen_done(False, "用户停止")

    def _on_gen_done(self, success, msg):
        self.gen_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"{'OK' if success else 'FAIL'}: {msg}")
        if success:
            self._refresh_datasets()

    def _gen_splits(self):
        region = self.region_combo.currentText()
        if not region:
            return
        self.log_text.clear()
        self.log_text.append(f"生成 fas_splits for {region}...")
        # 内联执行
        try:
            import pickle
            import numpy as np
            traj_dir = None
            for base in ['complete_dataset_10s', 'final_dataset_v1']:
                d = PROJECT_ROOT / 'data' / 'processed' / base / region
                if d.exists() and list(d.glob('*.pkl')):
                    traj_dir = d
                    break
            if traj_dir is None:
                self.log_text.append("未找到数据")
                return

            out_dir = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region
            out_dir.mkdir(parents=True, exist_ok=True)

            pkl_files = sorted([f.name for f in traj_dir.glob('*.pkl')
                                if f.name != 'dataset_stats.json'])
            total = 0
            info = []
            for fn in pkl_files:
                with open(traj_dir / fn, 'rb') as f:
                    data = pickle.load(f)
                n = len(data.get('samples', []))
                total += n
                info.append({'file': fn, 'n_samples': n})

            rng = np.random.default_rng(42)
            idx = rng.permutation(len(pkl_files))
            n1 = int(len(pkl_files) * 0.70)
            n2 = int(len(pkl_files) * 0.85)

            splits = {}
            for key, start, end in [('fas1', 0, n1), ('fas2', n1, n2), ('fas3', n2, len(idx))]:
                files = sorted([pkl_files[i] for i in idx[start:end]])
                ns = sum(info[i]['n_samples'] for i in idx[start:end])
                splits[key] = {'files': files, 'num_samples': ns}

            splits['metadata'] = {
                'region': region, 'total_trajectories': len(pkl_files),
                'total_samples': total, 'split_seed': 42,
            }
            out_path = out_dir / 'fas_splits.json'
            with open(out_path, 'w') as f:
                json.dump(splits, f, indent=2, ensure_ascii=False)
            self.log_text.append(f"已保存: {out_path}")
            for k in ['fas1', 'fas2', 'fas3']:
                self.log_text.append(f"  {k}: {len(splits[k]['files'])} trajs, "
                                     f"{splits[k]['num_samples']} samples")
        except Exception as e:
            self.log_text.append(f"失败: {e}")

    def _gen_trajlevel(self):
        region = self.region_combo.currentText()
        if not region:
            return
        fas_path = PROJECT_ROOT / 'data' / 'processed' / 'fas_splits' / region / 'fas_splits.json'
        if not fas_path.exists():
            self.log_text.append("请先生成 fas_splits.json")
            return
        traj_dir = None
        for base in ['complete_dataset_10s', 'final_dataset_v1']:
            d = PROJECT_ROOT / 'data' / 'processed' / base / region
            if d.exists():
                traj_dir = str(d)
                break
        if not traj_dir:
            return
        out = str(fas_path.parent / 'fas_splits_trajlevel.json')
        cmd = [
            'conda', 'run', '-n', 'torch-sm120', 'python',
            str(PROJECT_ROOT / 'scripts' / 'generate_traj_level_split.py'),
            '--traj_dir', traj_dir, '--fas_split_file', str(fas_path),
            '--output', out, '--val_ratio', '0.2', '--seed', '42',
        ]
        self.log_text.clear()
        self.worker = DataGenWorker(cmd)
        self.worker.progress.connect(self.log_text.append)
        self.worker.finished.connect(lambda ok, m: self.log_text.append(f"{'OK' if ok else 'FAIL'}: {m}"))
        self.worker.start()

    # --- 环境数据管理 ---

    def _check_env_data(self):
        """检查所有区域的环境数据完整性"""
        self.env_status.clear()
        utm_dir = PROJECT_ROOT / 'data' / 'processed' / 'utm_grid'
        if not utm_dir.exists():
            self.env_status.append("未找到 utm_grid 目录")
            return

        required = ['dem_utm.tif', 'slope_utm.tif', 'aspect_utm.tif', 'lulc_utm.tif', 'road_utm.tif']
        regions = sorted(d.name for d in utm_dir.iterdir() if d.is_dir())

        if not regions:
            self.env_status.append("未找到任何区域数据")
            return

        all_ok = True
        for region in regions:
            rdir = utm_dir / region
            missing = []
            present = []
            for f in required:
                fpath = rdir / f
                if fpath.exists():
                    # 检查文件大小
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    present.append(f"{f} ({size_mb:.1f}MB)")
                else:
                    missing.append(f)
                    all_ok = False

            if missing:
                self.env_status.append(f"[缺失] {region}: 缺少 {', '.join(missing)}")
            else:
                self.env_status.append(f"[完整] {region}: {len(present)} 个文件")

        # 检查OSM原始数据
        osm_dir = PROJECT_ROOT / 'data' / 'osm'
        if osm_dir.exists():
            osm_regions = sorted(d.name for d in osm_dir.iterdir() if d.is_dir())
            self.env_status.append(f"\nOSM数据: {', '.join(osm_regions) if osm_regions else '无'}")

        if all_ok:
            self.env_status.append("\n所有区域环境数据完整")

    def _download_road(self):
        """下载路网数据 (OSMnx)"""
        region = self.region_combo.currentText()
        if not region:
            self.env_status.append("请先选择区域")
            return

        self.env_status.append(f"\n开始下载 {region} 路网数据...")
        self.dl_road_btn.setEnabled(False)

        cmd = [
            'conda', 'run', '-n', 'torch-sm120', 'python',
            str(PROJECT_ROOT / 'scripts' / 'download_osm_data.py'),
            '--region', region,
        ]
        self._env_worker = DataGenWorker(cmd)
        self._env_worker.progress.connect(self.env_status.append)
        self._env_worker.finished.connect(self._on_road_download_done)
        self._env_worker.start()

    def _on_road_download_done(self, success, msg):
        self.dl_road_btn.setEnabled(True)
        if success:
            self.env_status.append(f"路网下载完成: {msg}")
            self.env_status.append("提示: 需要运行栅格化处理将GeoJSON转为road_utm.tif")
        else:
            self.env_status.append(f"路网下载失败: {msg}")

    def _download_gee(self):
        """下载DEM/LULC (Google Earth Engine)"""
        region = self.region_combo.currentText()
        if not region:
            self.env_status.append("请先选择区域")
            return

        self.env_status.append(f"\n启动GEE下载 {region}...")
        self.env_status.append("注意: GEE导出为异步任务, 文件将导出到Google Drive")
        self.env_status.append("完成后需手动下载并放到 data/raw/gee/{region}/ 目录")
        self.dl_gee_btn.setEnabled(False)

        cmd = [
            'conda', 'run', '-n', 'torch-sm120', 'python',
            str(PROJECT_ROOT / 'scripts' / 'download_new_regions.py'),
            '--regions', region,
        ]
        self._env_worker = DataGenWorker(cmd)
        self._env_worker.progress.connect(self.env_status.append)
        self._env_worker.finished.connect(self._on_gee_download_done)
        self._env_worker.start()

    def _on_gee_download_done(self, success, msg):
        self.dl_gee_btn.setEnabled(True)
        if success:
            self.env_status.append(f"GEE任务已提交: {msg}")
            self.env_status.append("请在 https://code.earthengine.google.com/tasks 监控进度")
        else:
            self.env_status.append(f"GEE下载失败: {msg}")
            self.env_status.append("可能原因: GEE凭证未配置或区域未在脚本中定义")
