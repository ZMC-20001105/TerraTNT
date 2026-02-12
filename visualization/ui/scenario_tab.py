#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tab 5: 任务场景仿真"""
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QLabel, QPushButton, QSlider, QRadioButton, QButtonGroup,
    QListWidget, QListWidgetItem, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer
from visualization.ui.map_view import MapView
from visualization.utils.colors import MODEL_COLORS, hex_to_rgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STEP_DT = 10; HIST_N = 90; FUTURE_N = 360; TOTAL_N = 450

INTEL_PHASE = {'精确终点坐标':'P1a','候选终点列表(含真实)':'P1a',
    '候选终点列表(不含真实)':'P1b','大致方向(区域先验σ=10km)':'P2a','无任何先验情报':'P3a'}
PHASE_DESC = {'P1a':'精确终点(域内)','P1b':'精确终点(OOD)',
    'P2a':'区域先验(σ=10km)','P3a':'无先验'}

def _fmt(step):
    s=step*STEP_DT; return f"{s//60:02d}:{s%60:02d}"

def _bbox(sample, pad=15.0):
    pts=[]
    if sample.history_rel is not None and len(sample.history_rel)>0: pts.append(sample.history_rel)
    if sample.future_rel is not None and len(sample.future_rel)>0: pts.append(sample.future_rel)
    if not pts: return 0.,0.,70.
    a=np.concatenate(pts); cx=(a[:,0].min()+a[:,0].max())/2; cy=(a[:,1].min()+a[:,1].max())/2
    h=max((a[:,0].max()-a[:,0].min())/2,(a[:,1].max()-a[:,1].min())/2)+pad
    return cx,cy,max(h*2,30.)


class ScenarioTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._sample = None
        self._rd = None
        self._phase = 'idle'
        self._step = 0
        self._speed = 5.0
        self._preds = {}
        self._offset = np.zeros((1, 2))
        self._cov = 140.0
        self._center = (0.0, 0.0)
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        # 交互状态: 用户手动放置的候选终点/区域先验
        self._user_cands = []           # [(x_km, y_km), ...] 相对坐标
        self._user_prior_center = None  # (x_km, y_km) 或 None
        self._user_prior_sigma = 10.0   # km
        self._build_ui()

    def _build_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        left = QWidget(); left.setFixedWidth(270)
        ll = QVBoxLayout(left); ll.setContentsMargins(4,4,4,4); ll.setSpacing(6)

        self.sg = sg = QGroupBox("\u2776 选择目标样本"); sgl = QVBoxLayout(sg)
        self.sample_list = QListWidget(); self.sample_list.setMaximumHeight(140)
        self.sample_list.setStyleSheet("QListWidget{font-size:10px}QListWidget::item{padding:2px}QListWidget::item:selected{background:#2979ff}")
        self.sample_list.currentRowChanged.connect(self._on_sample)
        sgl.addWidget(self.sample_list)
        self.sample_info = QLabel("未选择"); self.sample_info.setStyleSheet("color:#888;font-size:10px;"); self.sample_info.setWordWrap(True)
        sgl.addWidget(self.sample_info); ll.addWidget(sg)

        self.ig = ig = QGroupBox("\u2777 可用情报 → Phase"); igl = QVBoxLayout(ig)
        self.intel_grp = QButtonGroup(self)
        for i,(lab,ph) in enumerate(INTEL_PHASE.items()):
            rb = QRadioButton(lab); rb.setStyleSheet("font-size:11px;"); rb.setToolTip(f"→ {ph}: {PHASE_DESC[ph]}")
            self.intel_grp.addButton(rb, i); igl.addWidget(rb)
        self.intel_grp.button(0).setChecked(True)
        self.phase_lbl = QLabel(""); self.phase_lbl.setStyleSheet("color:#4fc3f7;font-size:12px;font-weight:bold;padding:4px;")
        igl.addWidget(self.phase_lbl)
        # 交互提示
        self.interact_lbl = QLabel("右键地图放置候选终点/区域中心")
        self.interact_lbl.setStyleSheet("color:#ffd740;font-size:10px;")
        self.interact_lbl.setWordWrap(True)
        igl.addWidget(self.interact_lbl)
        # 清除按钮 + sigma滑块
        ir = QHBoxLayout()
        self.clear_cands_btn = QPushButton("清除标记")
        self.clear_cands_btn.setStyleSheet("font-size:10px;padding:2px 6px;")
        self.clear_cands_btn.clicked.connect(self._clear_user_marks)
        ir.addWidget(self.clear_cands_btn)
        ir.addWidget(QLabel("σ:"))
        self.sigma_combo = QComboBox()
        self.sigma_combo.addItems(['5km','10km','15km','20km'])
        self.sigma_combo.setCurrentText('10km')
        self.sigma_combo.currentTextChanged.connect(
            lambda t: setattr(self, '_user_prior_sigma', float(t.replace('km',''))))
        ir.addWidget(self.sigma_combo)
        igl.addLayout(ir)
        self.cand_info = QLabel("候选终点: 0 个")
        self.cand_info.setStyleSheet("color:#aaa;font-size:10px;")
        igl.addWidget(self.cand_info)
        self.intel_grp.buttonClicked.connect(self._upd_phase); ll.addWidget(ig)

        self.mg = mg = QGroupBox("\u2778 预测模型"); mgl = QVBoxLayout(mg); mgl.setSpacing(2)
        self.mcbs = {}; default_on = {'V6_Autoreg','V7_ConfGate','LSTM_Env_Goal'}
        for nm,hx in MODEL_COLORS.items():
            cb = QCheckBox(nm); r,g,b = hex_to_rgb(hx)
            cb.setStyleSheet(f"QCheckBox{{color:rgb({r},{g},{b});font-size:10px;}}"); cb.setChecked(nm in default_on)
            mgl.addWidget(cb); self.mcbs[nm] = cb
        ll.addWidget(mg)

        self.pg = pg = QGroupBox("\u2779 播放控制"); pgl = QVBoxLayout(pg)
        sr = QHBoxLayout(); sr.addWidget(QLabel("倍速:"))
        self.spd = QComboBox(); self.spd.addItems(['1x','2x','5x','10x','20x','50x']); self.spd.setCurrentText('5x')
        self.spd.currentTextChanged.connect(self._chg_spd); sr.addWidget(self.spd); pgl.addLayout(sr)
        br = QHBoxLayout()
        self.play_btn = QPushButton("▶ 开始仿真"); self.play_btn.setStyleSheet("QPushButton{background:#2979ff;color:white;font-weight:bold;padding:6px;}")
        self.play_btn.clicked.connect(self._on_play); br.addWidget(self.play_btn)
        self.pause_btn = QPushButton("⏸ 暂停"); self.pause_btn.setEnabled(False); self.pause_btn.clicked.connect(self._on_pause); br.addWidget(self.pause_btn)
        self.rst_btn = QPushButton("⏹ 重置"); self.rst_btn.clicked.connect(self._on_reset); br.addWidget(self.rst_btn)
        pgl.addLayout(br)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, TOTAL_N); self.slider.setValue(0)
        self.slider.sliderMoved.connect(self._on_slider); pgl.addWidget(self.slider)
        self.time_lbl = QLabel(f"T = 00:00 / {_fmt(TOTAL_N)}"); self.time_lbl.setStyleSheet("font-size:14px;font-weight:bold;font-family:monospace;color:#fff;")
        self.time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); pgl.addWidget(self.time_lbl)
        self.stage_lbl = QLabel("就绪"); self.stage_lbl.setStyleSheet("font-size:11px;color:#ffd740;padding:2px;")
        self.stage_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); pgl.addWidget(self.stage_lbl)
        ll.addWidget(pg)
        rg = QGroupBox("⑤ 预测结果"); rgl = QVBoxLayout(rg); rgl.setSpacing(2)
        self.met_lbl = QLabel(""); self.met_lbl.setWordWrap(True)
        self.met_lbl.setStyleSheet("font-size:9px;font-family:monospace;color:#4fc3f7;padding:2px;")
        rgl.addWidget(self.met_lbl)
        # Phase性能参考
        self.ref_lbl = QLabel(
            "<b>Phase最佳模型参考:</b><br>"
            "P1a(精确终点): V6 973m<br>"
            "P1b(OOD终点): LSTM_EG 2027m<br>"
            "P2a(区域σ=10): LSTM_EG 1440m<br>"
            "P3a(无先验): V6R 1208m"
        )
        self.ref_lbl.setStyleSheet("color:#888;font-size:9px;padding:2px;")
        self.ref_lbl.setWordWrap(True)
        rgl.addWidget(self.ref_lbl)
        ll.addWidget(rg)
        ll.addStretch(); lay.addWidget(left)

        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0)
        self.map_view = MapView(); rl.addWidget(self.map_view)
        # 连接右键放置信号
        self.map_view.canvas.candidate_placed.connect(self._on_map_right_click)
        self.ade_lbl = QLabel(""); self.ade_lbl.setStyleSheet("color:#4fc3f7;font-size:11px;font-weight:bold;padding:2px;"); rl.addWidget(self.ade_lbl)
        lay.addWidget(right, stretch=1)
        self._upd_phase(); self._chg_spd()
        self._sync_ui_state()

    # ---- 引导式UI状态同步 ----
    _ACTIVE = "QGroupBox{border:2px solid #2979ff;border-radius:4px;margin-top:6px;padding-top:14px;font-weight:bold;color:#fff;}" \
              "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;color:#2979ff;}"
    _DONE   = "QGroupBox{border:1px solid #4caf50;border-radius:4px;margin-top:6px;padding-top:14px;color:#aaa;}" \
              "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;color:#4caf50;}"
    _WAIT   = "QGroupBox{border:1px solid #555;border-radius:4px;margin-top:6px;padding-top:14px;color:#666;}" \
              "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;color:#666;}"

    def _sync_ui_state(self):
        """根据当前流程阶段启用/禁用控件, 高亮当前步骤"""
        has_sample = self._sample is not None
        ph = self._get_phase()
        # P1a: 精确终点——直接用数据集GT终点, 不需要用户交互
        # P1b: OOD候选终点——需要用户放置候选点
        # P2a: 区域先验——需要用户设置区域中心
        # P3a: 无先验——不需要交互
        needs_interact = ph in ('P1b', 'P2a')
        interact_done = True
        if ph == 'P1b':
            interact_done = len(self._user_cands) > 0
        elif ph == 'P2a':
            interact_done = self._user_prior_center is not None
        can_play = has_sample and (not needs_interact or interact_done)
        is_running = self._phase in ('obs', 'pred', 'infer')

        # 步骤1: 样本选择
        if not has_sample:
            self.sg.setStyleSheet(self._ACTIVE)
            self.ig.setStyleSheet(self._WAIT)
            self.mg.setStyleSheet(self._WAIT)
            self.pg.setStyleSheet(self._WAIT)
        # 步骤2: 情报+交互
        elif needs_interact and not interact_done:
            self.sg.setStyleSheet(self._DONE)
            self.ig.setStyleSheet(self._ACTIVE)
            self.mg.setStyleSheet(self._WAIT)
            self.pg.setStyleSheet(self._WAIT)
        # 步骤3+4: 可以开始
        elif not is_running:
            self.sg.setStyleSheet(self._DONE)
            self.ig.setStyleSheet(self._DONE)
            self.mg.setStyleSheet(self._ACTIVE if self._phase == 'idle' else self._DONE)
            self.pg.setStyleSheet(self._ACTIVE)
        else:
            self.sg.setStyleSheet(self._DONE)
            self.ig.setStyleSheet(self._DONE)
            self.mg.setStyleSheet(self._DONE)
            self.pg.setStyleSheet(self._ACTIVE)

        # 控件启用/禁用
        for btn in self.intel_grp.buttons():
            btn.setEnabled(not is_running)
        self.clear_cands_btn.setEnabled(not is_running and has_sample)
        self.sigma_combo.setEnabled(not is_running and ph == 'P2a')
        for cb in self.mcbs.values():
            cb.setEnabled(not is_running)
        self.play_btn.setEnabled(can_play and not is_running)
        self.slider.setEnabled(has_sample)

        # 交互提示更新
        if not has_sample:
            self.interact_lbl.setText("← 请先在左侧列表选择一个样本")
            self.interact_lbl.setStyleSheet("color:#ff9800;font-size:10px;font-weight:bold;")
        elif needs_interact and not interact_done:
            if ph == 'P1b':
                self.interact_lbl.setText("⚠ 请在地图上右键放置候选终点(不含GT)，然后点击「开始仿真」")
            elif ph == 'P2a':
                self.interact_lbl.setText("⚠ 请在地图上右键点击设置区域先验中心，然后点击「开始仿真」")
            self.interact_lbl.setStyleSheet("color:#ff9800;font-size:10px;font-weight:bold;")
        elif can_play and self._phase == 'idle':
            self.interact_lbl.setText("✓ 准备就绪，点击「开始仿真」")
            self.interact_lbl.setStyleSheet("color:#4caf50;font-size:10px;font-weight:bold;")
        else:
            self.interact_lbl.setText("")

    # ---- data ----
    def populate_samples(self, samples):
        self.sample_list.blockSignals(True)
        self.sample_list.clear()
        for s in samples[:500]:
            t = f"{s.intent}/{s.vehicle_type} d={s.total_distance_km:.0f}km"
            it = QListWidgetItem(t)
            it.setData(Qt.ItemDataRole.UserRole, s.sample_id)
            self.sample_list.addItem(it)
        self.sample_list.blockSignals(False)

    def set_region_data(self, rd):
        self._rd = rd
        self.map_view.set_region_data(rd)

    # ---- events ----
    def _on_sample(self, row):
        if row < 0:
            return
        it = self.sample_list.item(row)
        if not it:
            return
        sid = it.data(Qt.ItemDataRole.UserRole)
        for s in self.mw.dm.samples:
            if s.sample_id == sid:
                self._sample = s
                self.sample_info.setText(
                    f"ID: {sid}\n{s.intent}/{s.vehicle_type} "
                    f"d={s.total_distance_km:.1f}km sin={s.sinuosity:.2f}")
                self._setup_map()
                self._on_reset()
                self._sync_ui_state()
                return

    def _upd_phase(self, btn=None):
        cb = self.intel_grp.checkedButton()
        if cb:
            ph = INTEL_PHASE.get(cb.text(), 'P1a')
            self.phase_lbl.setText(f"→ {ph}: {PHASE_DESC.get(ph,'')}")
            if ph in ('P1a', 'P1b'):
                self.interact_lbl.setText("右键地图放置候选终点 (至少1个)")
            elif ph == 'P2a':
                self.interact_lbl.setText("右键地图设置区域先验中心, 调节σ")
            else:
                self.interact_lbl.setText("P3a无先验: 模型仅依赖历史轨迹")
            self._clear_user_marks()
            self._sync_ui_state()

    def _chg_spd(self, t=None):
        self._speed = float(self.spd.currentText().replace('x', ''))
        if self._timer.isActive():
            self._timer.setInterval(max(int(STEP_DT * 1000 / self._speed), 20))

    def _get_phase(self):
        cb = self.intel_grp.checkedButton()
        return INTEL_PHASE.get(cb.text(), 'P1a') if cb else 'P1a'

    # ---- map setup ----
    def _setup_map(self):
        s = self._sample
        if s is None or self._rd is None:
            return
        cx, cy, cov = _bbox(s, pad=15.0)
        self._cov = min(cov, 200.0)
        oe, on = s.last_obs_utm
        self._center = (oe + cx * 1000.0, on + cy * 1000.0)
        self._offset = np.array([[-cx, -cy]])
        patches = self._rd.extract_patch(self._center, self._cov, 512)
        self.map_view.set_patches(patches, self._center, self._cov)

    # ---- playback controls ----
    def _on_play(self):
        if self._sample is None:
            self.stage_lbl.setText("请先选择样本!")
            return
        if self._phase in ('idle', 'done'):
            self._step = 0
            self._preds = {}
            self._phase = 'obs'
            self.stage_lbl.setText("▶ 卫星观测中...")
            self.stage_lbl.setStyleSheet("font-size:11px;color:#66bb6a;padding:2px;")
        iv = max(int(STEP_DT * 1000 / self._speed), 20)
        self._timer.start(iv)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self._sync_ui_state()

    def _on_pause(self):
        self._timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stage_lbl.setText("⏸ 已暂停")
        self._sync_ui_state()

    def _on_reset(self):
        self._timer.stop()
        self._step = 0
        self._phase = 'idle'
        self._preds = {}
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.slider.setValue(0)
        self.time_lbl.setText(f"T = 00:00 / {_fmt(TOTAL_N)}")
        self.stage_lbl.setText("就绪")
        self.stage_lbl.setStyleSheet("font-size:11px;color:#ffd740;padding:2px;")
        self.met_lbl.setText("")
        self.ade_lbl.setText("")
        self.map_view.canvas.set_prior_overlay(None)
        self._draw(0)
        self._sync_ui_state()

    # ---- interactive map ----
    def _on_map_right_click(self, map_rel_x, map_rel_y):
        """右键在地图上放置候选终点或区域先验中心
        map_rel_x/y 是相对于地图中心的km坐标, 需转为观测点相对坐标"""
        # 地图中心相对坐标 → 观测点相对坐标
        # _draw中: vis = obs_rel + offset, 所以 obs_rel = map_rel - offset
        obs_rel_x = map_rel_x - self._offset[0, 0]
        obs_rel_y = map_rel_y - self._offset[0, 1]
        ph = self._get_phase()
        if ph in ('P1a', 'P1b'):
            # 放置候选终点
            self._user_cands.append([obs_rel_x, obs_rel_y])
            self.cand_info.setText(f"候选终点: {len(self._user_cands)} 个")
            self.stage_lbl.setText(f"已放置候选终点 #{len(self._user_cands)}")
        elif ph == 'P2a':
            # 设置区域先验中心
            self._user_prior_center = [obs_rel_x, obs_rel_y]
            self.cand_info.setText(f"区域先验中心: ({obs_rel_x:.1f}, {obs_rel_y:.1f})km")
            self.stage_lbl.setText(f"已设置区域先验中心 σ={self._user_prior_sigma}km")
            # 立即显示先验热力图预览
            from visualization.utils.phase_utils import make_heatmap_interactive
            hm = make_heatmap_interactive(
                self._user_prior_center, self._user_prior_sigma,
                self._sample.env_map if self._sample else None)
            self.map_view.canvas.set_prior_overlay(hm)
        else:
            # P3a 无先验, 不允许交互
            self.stage_lbl.setText("P3a无先验模式, 不需要放置标记")
            return
        # 刷新地图显示
        self._draw(self._step)
        self._sync_ui_state()

    def _clear_user_marks(self):
        """清除所有用户放置的标记"""
        self._user_cands = []
        self._user_prior_center = None
        self.cand_info.setText("候选终点: 0 个")
        self.map_view.canvas.set_prior_overlay(None)
        self.stage_lbl.setText("已清除所有标记")
        self._draw(self._step)
        self._sync_ui_state()

    def _on_slider(self, val):
        was_running = self._timer.isActive()
        self._timer.stop()
        self._step = val
        if val < HIST_N:
            self._phase = 'obs'
        elif val >= HIST_N and not self._preds:
            self._phase = 'obs'
            self._step = HIST_N - 1
        else:
            self._phase = 'pred'
        self._draw(self._step)
        self._upd_time()
        if was_running:
            self._timer.start(max(int(STEP_DT * 1000 / self._speed), 20))

    # ---- tick ----
    def _tick(self):
        if self._phase == 'obs':
            self._step += 1
            if self._step >= HIST_N:
                self._step = HIST_N
                self._phase = 'infer'
                self.stage_lbl.setText("⚡ 观测结束, 启动模型推理...")
                self.stage_lbl.setStyleSheet("font-size:11px;color:#ff9800;padding:2px;")
                self._timer.stop()
                self._draw(self._step)
                self._upd_time()
                self._run_inference()
                return
        elif self._phase == 'pred':
            self._step += 1
            if self._step >= TOTAL_N:
                self._step = TOTAL_N
                self._phase = 'done'
                self._timer.stop()
                self.play_btn.setEnabled(True)
                self.pause_btn.setEnabled(False)
                self.stage_lbl.setText("✓ 仿真完成")
                self.stage_lbl.setStyleSheet("font-size:11px;color:#4caf50;padding:2px;")
        self._draw(self._step)
        self._upd_time()

    def _upd_time(self):
        self.slider.blockSignals(True)
        self.slider.setValue(self._step)
        self.slider.blockSignals(False)
        self.time_lbl.setText(f"T = {_fmt(self._step)} / {_fmt(TOTAL_N)}")
        if self._phase == 'obs':
            self.stage_lbl.setText(f"▶ 卫星观测中 ({self._step}/{HIST_N})")
        elif self._phase == 'pred':
            pred_step = self._step - HIST_N
            self.stage_lbl.setText(f"📡 预测展开中 ({pred_step}/{FUTURE_N})")

    # ---- draw ----
    def _draw(self, step):
        s = self._sample
        if s is None:
            return
        off = self._offset
        hist = s.history_rel  # (90,2), index 0=oldest, 89=present(0,0)
        fut = s.future_rel    # (360,2)

        # GT终点始终可见 (黄色星标记)
        vis_goal = s.goal_rel + off.flatten() if s.goal_rel is not None and np.linalg.norm(s.goal_rel) > 0.01 else None

        # 构建当前可见的候选终点 (在所有阶段都可见)
        ph = self._get_phase()
        vis_cands = None
        if ph in ('P1a', 'P1b') and self._user_cands:
            vis_cands = np.array(self._user_cands, dtype=np.float32) + off
        elif ph == 'P2a' and self._user_prior_center is not None:
            vis_cands = np.array([self._user_prior_center], dtype=np.float32) + off

        # 观测阶段: 从历史起点开始逐步延伸到当前观测点
        if step <= HIST_N and step > 0:
            end_idx = int(step * len(hist) / HIST_N)
            end_idx = max(2, min(end_idx, len(hist)))
            vis_hist = hist[:end_idx] + off
            self.map_view.canvas.set_trajectories(
                history_rel=vis_hist, future_rel=None,
                candidates_rel=vis_cands, predictions={},
                goal_rel=vis_goal)
        elif step == 0:
            # 空闲/重置: 显示完整历史 + 候选终点, 方便用户交互定位
            vis_hist = hist + off if hist is not None and len(hist) > 1 else None
            self.map_view.canvas.set_trajectories(
                history_rel=vis_hist, future_rel=None,
                candidates_rel=vis_cands, predictions={},
                goal_rel=vis_goal)
        else:
            # 预测阶段: 显示全部历史 + 逐步展开GT和预测
            pred_step = step - HIST_N
            vis_hist = hist + off if hist is not None else None
            vis_fut = (fut[:pred_step] + off) if fut is not None and pred_step > 0 else None
            # vis_cands 已在上方计算; 若用户未放置则用数据集默认候选
            if vis_cands is None and ph in ('P1a', 'P1b'):
                vis_cands = s.candidates_rel + off
            # 逐步展开预测轨迹
            vis_preds = {}
            for mn, pred in self._preds.items():
                if pred is not None and len(pred) > 0:
                    show_n = min(pred_step, len(pred))
                    vis_preds[mn] = pred[:show_n] + off
            self.map_view.canvas.set_trajectories(
                history_rel=vis_hist, future_rel=vis_fut,
                candidates_rel=vis_cands, predictions=vis_preds,
                goal_rel=vis_goal)
            # 模型可见性
            for mn, cb in self.mcbs.items():
                self.map_view.canvas.set_model_visibility(mn, cb.isChecked())
            # 实时ADE
            self._update_metrics(pred_step)
        self.map_view.canvas.repaint()  # 强制立即重绘

    def _update_metrics(self, pred_step):
        if pred_step <= 0:
            self.met_lbl.setText("")
            self.ade_lbl.setText("")
            return
        s = self._sample
        lines = []
        best_ade = float('inf')
        best_name = ""
        for mn, pred in self._preds.items():
            if not self.mcbs.get(mn, QCheckBox()).isChecked():
                continue
            if pred is None or s.future_rel is None:
                continue
            n = min(pred_step, len(pred), len(s.future_rel))
            if n <= 0:
                continue
            diff = pred[:n] - s.future_rel[:n]
            ade = float(np.mean(np.linalg.norm(diff, axis=1)) * 1000)
            fde = float(np.linalg.norm(pred[n-1] - s.future_rel[n-1]) * 1000)
            lines.append(f"{mn:15s} ADE={ade:6.0f}m FDE={fde:6.0f}m")
            if ade < best_ade:
                best_ade = ade
                best_name = mn
        self.met_lbl.setText('\n'.join(lines) if lines else "")
        if best_name:
            t_min = pred_step * STEP_DT / 60
            self.ade_lbl.setText(f"最佳: {best_name} ADE={best_ade:.0f}m (T+{t_min:.0f}min)")
        else:
            self.ade_lbl.setText("")

    # ---- inference (Phase感知) ----
    def _run_inference(self):
        if self._sample is None or self.mw.model_manager is None:
            self._phase = 'pred'
            self._timer.start(max(int(STEP_DT * 1000 / self._speed), 20))
            return
        vis = [n for n, cb in self.mcbs.items()
               if cb.isChecked() and n in self.mw.model_manager.models]
        if not vis:
            self._phase = 'pred'
            self._timer.start(max(int(STEP_DT * 1000 / self._speed), 20))
            return

        # Phase感知: 根据情报选择修改env_map ch17 + 候选终点
        from visualization.utils.phase_utils import prepare_phase_inputs
        phase_key = self._get_phase()
        user_cands = np.array(self._user_cands, dtype=np.float32) if self._user_cands else None
        user_prior = np.array(self._user_prior_center, dtype=np.float32) if self._user_prior_center else None
        env_map, candidates = prepare_phase_inputs(
            self._sample, phase_key,
            user_candidates=user_cands,
            user_prior_center_km=user_prior,
            user_prior_sigma_km=self._user_prior_sigma)

        # 显示先验热力图叠加
        self.map_view.canvas.set_prior_overlay(env_map[17])

        self.stage_lbl.setText(f"⚡ 推理中 ({phase_key}, {len(vis)}模型)...")
        from PyQt6.QtCore import QThread
        from PyQt6.QtCore import pyqtSignal as Signal

        class _W(QThread):
            done = Signal(dict)
            def __init__(self, mm, sample, names, env, cands):
                super().__init__()
                self.mm = mm; self.sample = sample; self.names = names
                self.env = env; self.cands = cands
            def run(self):
                preds = {}
                for nm in self.names:
                    h = (self.sample.history_feat
                         if hasattr(self.sample, 'history_feat') and self.sample.history_feat is not None
                         else np.zeros((90, 26), dtype=np.float32))
                    p = self.mm.predict(nm, h, self.env, self.cands)
                    if p is not None:
                        preds[nm] = p
                self.done.emit(preds)

        self._infer_w = _W(self.mw.model_manager, self._sample, vis, env_map, candidates)
        self._infer_w.done.connect(self._on_infer_done)
        self._infer_w.start()

    def _on_infer_done(self, preds):
        self._preds = preds
        n_ok = len(preds)
        self.stage_lbl.setText(f"📡 推理完成 ({n_ok}模型), 展开预测...")
        self.stage_lbl.setStyleSheet("font-size:11px;color:#42a5f5;padding:2px;")
        self._phase = 'pred'
        iv = max(int(STEP_DT * 1000 / self._speed), 20)
        self._timer.start(iv)
        self._sync_ui_state()
