#!/usr/bin/env python3
"""绘制 TerraTNT 系统架构设计图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(_font_path)
_fp = fm.FontProperties(fname=_font_path)
_font_name = _fp.get_name()
plt.rcParams['font.sans-serif'] = [_font_name, 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

C_UI = ('#E3F2FD','#1565C0','#BBDEFB')
C_APP = ('#E8F5E9','#2E7D32','#C8E6C9')
C_MOD = ('#FFF3E0','#E65100','#FFE0B2')
C_DAT = ('#FCE4EC','#C62828','#F8BBD0')
C_INF = ('#F3E5F5','#6A1B9A','#E1BEE7')
C_PROC = ('#ECEFF1','#546E7A','#CFD8DC')

def rbox(ax, x, y, w, h, fc, ec, label='', fs=9, lw=1.5, al=0.85):
    p = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",facecolor=fc,edgecolor=ec,lw=lw,alpha=al,zorder=2)
    ax.add_patch(p)
    if label:
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color='#212121',fontweight='bold',zorder=3)

def layer(ax, x, y, w, h, bg, ec, title, fs=11):
    p = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03",facecolor=bg,edgecolor=ec,lw=2.5,alpha=0.35,zorder=0)
    ax.add_patch(p)
    ax.text(x+0.08,y+h-0.06,title,ha='left',va='top',fontsize=fs,color=ec,fontweight='bold',zorder=1)

def sub(ax, x, y, t, fs=7.5):
    ax.text(x,y,t,ha='center',va='center',fontsize=fs,color='#424242',zorder=3,linespacing=1.3)

def arr(ax, x1, y1, x2, y2, s='->'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle=s,color='#455A64',lw=1.5),zorder=5)

W, H = 20, 15
fig, ax = plt.subplots(1, 1, figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')

# ── 标题 ──
ax.text(W/2, 14.65, 'TerraTNT 系统架构设计', ha='center', fontsize=22, fontweight='bold', color='#212121')
ax.text(W/2, 14.25, '面向遥感任务规划的地面目标位置预测系统', ha='center', fontsize=12, color='#757575')

# ================================================================
# L1: 交互层
# ================================================================
L1_y, L1_h = 11.8, 2.15
layer(ax, 0.3, L1_y, W-0.6, L1_h, *C_UI[:2], '交互层 (PyQt5 用户界面)', fs=12)

tabs = [
    ('分析\nAnalysis',   '样本检视 | 阶段推理'),
    ('仿真\nScenario',   '交互仿真 | 时间轴回放'),
    ('评估\nEvaluation', '批量评估 | 跨阶段对比'),
    ('数据\nData',       '数据集管理 | 环境数据'),
    ('训练\nTraining',   '模型训练 | 进度监控'),
]
tw, th, tgap = 3.2, 0.95, 0.45
tx0 = (W - 5*tw - 4*tgap) / 2
for i, (n, d) in enumerate(tabs):
    x = tx0 + i*(tw+tgap)
    rbox(ax, x, L1_y+0.7, tw, th, C_UI[2], C_UI[1], n, fs=10)
    sub(ax, x+tw/2, L1_y+0.35, d, fs=8)

# ================================================================
# L2: 应用层
# ================================================================
L2_y, L2_h = 8.9, 2.6
layer(ax, 0.3, L2_y, W-0.6, L2_h, *C_APP[:2], '应用层 (业务逻辑与编排)', fs=12)

apps = [
    ('数据生成引擎',  '分层A*路径规划\n代价地图 | XGBoost速度\n环境地图提取(18ch)',  0.8,  4.2),
    ('阶段管理器',    'P1a精确 | P1b候选\nP2a区域 | P3a无先验\n热力图/候选终点生成',  5.4,  4.0),
    ('评估框架',      'ADE/FDE/MR指标\n早/中/晚分段评估\n跨阶段·跨模型对比',         9.8,  4.0),
    ('模型编排器',    '模型加载/切换\n推理调度 | 多模型并行\n热力图goal提取',          14.2, 5.2),
]
for n, d, x, w in apps:
    rbox(ax, x, L2_y+1.0, w, 1.0, C_APP[2], C_APP[1], n, fs=10)
    sub(ax, x+w/2, L2_y+0.42, d, fs=7.5)

# ================================================================
# L3 左: 模型层
# ================================================================
L3_y, L3_h = 5.2, 3.4
layer(ax, 0.3, L3_y, 13.0, L3_h, *C_MOD[:2], '模型层 (深度学习)', fs=12)

# TerraTNT 核心 (大框)
rbox(ax, 0.7, L3_y+0.3, 7.2, 2.5, '#FFE0B2', C_MOD[1], '', lw=2.0)
ax.text(4.3, L3_y+2.55, 'TerraTNT 核心模型', ha='center', fontsize=11,
        fontweight='bold', color=C_MOD[1], zorder=3)

# 上排子模块
row1_y = L3_y + 1.55
mw, mh = 2.15, 0.55
subs_r1 = [
    ('CNN环境编码器\n18ch → 128d', 0.9),
    ('LSTM历史编码器\n26d → 128d',  3.2),
    ('目标分类器\nN候选 → 概率',    5.5),
]
for lb, x in subs_r1:
    rbox(ax, x, row1_y, mw, mh, '#FFF8E1', '#F57C00', lb, fs=7.5, lw=1)

# 下排子模块
row2_y = L3_y + 0.7
subs_r2 = [
    ('层次化解码器\nWaypoint + LSTM', 0.9),
    ('空间环境采样\nGrid Sample',     3.2),
    ('置信度门控\nV7 Gate',           5.5),
]
for lb, x in subs_r2:
    rbox(ax, x, row2_y, mw, mh, '#FFF8E1', '#F57C00', lb, fs=7.5, lw=1)

# 增量模型
ax.text(9.3, L3_y+2.55, '增量模型', ha='center', fontsize=10,
        fontweight='bold', color=C_MOD[1], zorder=3)
inc = [('V3 Waypoint', 8.2, row1_y), ('V4 Spatial', 9.8, row1_y),
       ('V6 Autoreg',  8.2, row2_y), ('V7 ConfGate', 9.8, row2_y)]
for lb, x, y in inc:
    rbox(ax, x, y, 1.45, mh, '#FFF8E1', '#F57C00', lb, fs=7.5, lw=1)

# 基线模型
ax.text(12.1, L3_y+2.55, '基线模型', ha='center', fontsize=10,
        fontweight='bold', color=C_MOD[1], zorder=3)
bsl = [('LSTM_Env_Goal', row1_y), ('LSTM_Only', row2_y)]
for lb, y in bsl:
    rbox(ax, 11.4, y, 1.6, mh, '#FFF8E1', '#F57C00', lb, fs=7.5, lw=1)

# ================================================================
# L3 右: 数据持久化层
# ================================================================
layer(ax, 13.5, L3_y, 6.2, L3_h, *C_DAT[:2], '数据持久化层', fs=12)

dw, dh = 2.6, 1.1
rbox(ax, 13.8, L3_y+1.6, dw, dh, C_DAT[2], C_DAT[1], 'GIS环境资产', fs=9)
sub(ax, 15.1, L3_y+1.15, 'DEM / Slope / Aspect\nLULC / Road(分级)\nUTM投影 30m分辨率', fs=7.5)

rbox(ax, 16.8, L3_y+1.6, dw, dh, C_DAT[2], C_DAT[1], '轨迹数据集', fs=9)
sub(ax, 18.1, L3_y+1.15, '10s采样 PKL格式\nFAS阶段划分\n4车型 × 3意图', fs=7.5)

rbox(ax, 13.8, L3_y+0.2, dw, 0.8, C_DAT[2], C_DAT[1], '模型权重', fs=9)
sub(ax, 15.1, L3_y+0.05, 'runs/*_best.pth', fs=7.5)

rbox(ax, 16.8, L3_y+0.2, dw, 0.8, C_DAT[2], C_DAT[1], '评估结果', fs=9)
sub(ax, 18.1, L3_y+0.05, 'JSON / CSV统计', fs=7.5)

# ================================================================
# L4: 数据处理层
# ================================================================
L4_y, L4_h = 2.5, 2.35
layer(ax, 0.3, L4_y, W-0.6, L4_h, *C_INF[:2], '数据处理层', fs=12)

pw, ph = 3.4, 1.0
pgap = 0.35
px0 = (W - 5*pw - 4*pgap) / 2
procs = [
    ('GEE数据下载\n& UTM重投影',  'DEM / LULC / Slope / Aspect\nGoogle Earth Engine API'),
    ('OSM道路处理',               'PBF解析 | 分级栅格化\nosmium + rasterio'),
    ('代价地图生成',              '可通行性分析 | 速度建模\n4车型 × 3意图 = 12张'),
    ('轨迹生成 & 预处理',        '分层A* | 速度采样\n环境地图提取(18ch 128×128)'),
    ('OORD真实数据',             '苏格兰高地GPS轨迹\nXGBoost速度模型训练'),
]
for i, (n, d) in enumerate(procs):
    x = px0 + i*(pw+pgap)
    rbox(ax, x, L4_y+0.85, pw, ph, C_INF[2], C_INF[1], n, fs=8.5)
    sub(ax, x+pw/2, L4_y+0.32, d, fs=7)

# ================================================================
# L5: 基础设施
# ================================================================
L5_y, L5_h = 0.7, 1.5
rbox(ax, 0.3, L5_y, W-0.6, L5_h, '#ECEFF1', '#546E7A', '', al=0.4, lw=2.5)
ax.text(0.55, L5_y+L5_h-0.12, '基础设施层', ha='left', va='top',
        fontsize=12, fontweight='bold', color='#546E7A', zorder=3)
infra = ['Ubuntu 24.04 LTS', 'RTX 5060 (8GB)', 'CUDA 12.8',
         'PyTorch 2.9', 'GDAL / rasterio', 'Anaconda']
iw = 2.6; igap = 0.3
ix0 = (W - 6*iw - 5*igap) / 2
for i, t in enumerate(infra):
    rbox(ax, ix0+i*(iw+igap), L5_y+0.25, iw, 0.6, C_PROC[2], '#78909C', t, fs=8, lw=1)

# ================================================================
# 层间箭头
# ================================================================
# UI → 应用层
for xp in [3.5, 8.5, 16.0]:
    arr(ax, xp, L1_y, xp, L2_y+L2_h)

# 应用层 → 模型层
for xp in [3.0, 8.0]:
    arr(ax, xp, L2_y, xp, L3_y+L3_h)
# 模型编排器 → 模型层
arr(ax, 14.5, L2_y, 10.0, L3_y+L3_h)
# 应用层 → 数据层
arr(ax, 17.0, L2_y, 17.0, L3_y+L3_h)

# 模型层 ↔ 数据层
arr(ax, 13.1, L3_y+1.8, 13.5, L3_y+1.8, s='<->')

# 数据处理层 → 数据持久化层
arr(ax, 15.0, L4_y+L4_h, 15.0, L3_y)
arr(ax, 18.0, L4_y+L4_h, 18.0, L3_y)

# ================================================================
# 数据流标注
# ================================================================
ax.text(W/2, 0.3, '数据流: 原始GIS数据 → 数据处理层 → 数据持久化层 → 模型层 → 应用层 → 交互层 → 用户',
        ha='center', fontsize=9, color='#757575', style='italic')

# ================================================================
# 图例
# ================================================================
legs = [(C_UI[2], C_UI[1], '交互层: UI'),
        (C_APP[2], C_APP[1], '应用层: 业务逻辑'),
        (C_MOD[2], C_MOD[1], '模型层: AI/ML'),
        (C_DAT[2], C_DAT[1], '数据层: 持久化'),
        (C_INF[2], C_INF[1], '处理层: ETL'),
        (C_PROC[2], '#78909C', '基础设施')]
lx0 = (W - 6*3.1) / 2
for i, (fc, ec, lb) in enumerate(legs):
    x = lx0 + i*3.1
    rbox(ax, x, 0.0, 0.35, 0.22, fc, ec, '', fs=6, lw=1)
    ax.text(x+0.45, 0.11, lb, ha='left', va='center', fontsize=7.5, color=ec, fontweight='bold')

plt.tight_layout(pad=0.3)
out = '/home/zmc/文档/programwork/docs/architecture_diagram_cn.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'✓ 已保存: {out}')
plt.close()
