#!/usr/bin/env python3
"""Draw V6R model architecture diagram for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

try:
    import matplotlib.font_manager as fm
    _fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    fm.fontManager.addfont(_fp)
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=_fp).get_name(), 'DejaVu Sans']
except Exception:
    pass
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

C_IN   = ('#FFF8E1', '#F57F17')
C_ENC  = ('#E3F2FD', '#1565C0')
C_GOAL = ('#FFF3E0', '#E65100')
C_FUS  = ('#E8F5E9', '#2E7D32')
C_DEC  = ('#F3E5F5', '#6A1B9A')
C_WP   = ('#FCE4EC', '#C62828')
C_OUT  = ('#ECEFF1', '#37474F')

def box(ax, cx, cy, w, h, fc, ec, label, fs=8.5, lw=1.5):
    """Draw centered box at (cx, cy)."""
    x, y = cx - w/2, cy - h/2
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                       facecolor=fc, edgecolor=ec, lw=lw, alpha=0.92, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fs, color='#212121', fontweight='bold', zorder=3, linespacing=1.25)
    return (cx, cy, w, h)

def arr(ax, x1, y1, x2, y2, c='#455A64', lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=5)

def larr(ax, x1, y1, x2, y2, label, c='#455A64', side='right'):
    """Arrow with label."""
    arr(ax, x1, y1, x2, y2, c=c)
    mx, my = (x1+x2)/2, (y1+y2)/2
    ha = 'left' if side == 'right' else 'right'
    dx = 0.08 if side == 'right' else -0.08
    ax.text(mx+dx, my, label, fontsize=6.5, color=c, ha=ha, va='center', zorder=6)

# ── Layout: 3 columns, top-to-bottom ──
# Col1 (x=2.5): Inputs + Encoders
# Col2 (x=7.5): Goal + Fusion (center)
# Col3 (x=12.5): Waypoint + Spatial
# Bottom span: Decoder + Output

W, H = 17, 13.5
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')

# Title
ax.text(W/2, 13.1, 'TerraTNT V6R Model Architecture', ha='center',
        fontsize=17, fontweight='bold', color='#212121')
ax.text(W/2, 12.7, 'Autoregressive Decoder with Waypoint Milestones, Spatial Env Sampling & Goal Dropout',
        ha='center', fontsize=9, color='#757575')

# ════════════════════════════════════════
# ROW 1 (y=11.5): INPUTS
# ════════════════════════════════════════
r1 = 11.5
bw, bh = 2.8, 0.65
box(ax, 2.5, r1, bw, bh, *C_IN, 'Environment Map\n18ch × 128×128')
box(ax, 7.5, r1, bw, bh, *C_IN, 'History Trajectory\n90 steps × 26d')
box(ax, 12.5, r1, bw, bh, *C_IN, 'Goal Candidates\n6 × (x, y) km')

# ════════════════════════════════════════
# ROW 2 (y=10.2): ENCODERS + CLASSIFIER
# ════════════════════════════════════════
r2 = 10.2
box(ax, 2.5, r2, bw, 0.75, *C_ENC, 'CNN Env Encoder\n→ 128d global + spatial map')
box(ax, 7.5, r2, bw, 0.75, *C_ENC, 'LSTM History Encoder\n2-layer → 128d')
box(ax, 12.5, r2, bw, 0.75, *C_GOAL, 'Goal Classifier\nenv⊕hist → 6-class logits')

# Row1 → Row2
arr(ax, 2.5, r1-bh/2, 2.5, r2+0.75/2, c=C_ENC[1])
arr(ax, 7.5, r1-bh/2, 7.5, r2+0.75/2, c=C_ENC[1])
arr(ax, 12.5, r1-bh/2, 12.5, r2+0.75/2, c=C_GOAL[1])
# env,hist → classifier
arr(ax, 2.5+bw/2, r2+0.1, 12.5-bw/2, r2+0.3, c='#90CAF9', lw=1.0)
arr(ax, 7.5+bw/2, r2+0.1, 12.5-bw/2, r2+0.1, c='#90CAF9', lw=1.0)

# ════════════════════════════════════════
# ROW 3 (y=8.8): GOAL SELECTION + DROPOUT
# ════════════════════════════════════════
r3 = 8.85
box(ax, 12.5, r3, bw, 0.65, *C_GOAL, 'Goal Selection\nargmax → selected goal (x,y)')
arr(ax, 12.5, r2-0.75/2, 12.5, r3+0.65/2, c=C_GOAL[1])

# Goal Dropout badge
ax.text(14.6, r3, 'Goal Dropout\np=0.3 (V6R)', fontsize=8,
        color='#C62828', fontweight='bold', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.15', fc='#FFEBEE', ec='#C62828', lw=1.2))

# ════════════════════════════════════════
# ROW 4 (y=7.6): CONTEXT FUSION (full width)
# ════════════════════════════════════════
r4 = 7.6
box(ax, 7.5, r4, 11.0, 0.7, *C_FUS,
    'Context Fusion:  ReLU( W · [ hist_128d  ||  env_128d  ||  goal_64d ] )  →  256d context', fs=9)

# env → fusion
arr(ax, 2.5, r2-0.75/2, 3.0, r4+0.7/2, c=C_ENC[1])
# hist → fusion
arr(ax, 7.5, r2-0.75/2, 7.5, r4+0.7/2, c=C_ENC[1])
# goal → fusion
arr(ax, 12.5, r3-0.65/2, 12.0, r4+0.7/2, c=C_GOAL[1])

# ════════════════════════════════════════
# ROW 5 (y=6.3): WAYPOINT PREDICTOR
# ════════════════════════════════════════
r5 = 6.3
box(ax, 5.0, r5, 3.5, 0.7, *C_WP,
    'Waypoint Predictor\ncontext → 10 waypoints (base + residual)')
arr(ax, 5.0, r4-0.7/2, 5.0, r5+0.7/2, c=C_FUS[1])

# ════════════════════════════════════════
# ROW 5b (y=6.3): SPATIAL ENV SAMPLING (right)
# ════════════════════════════════════════
box(ax, 11.5, r5, 3.5, 0.7, *C_WP,
    'Spatial Env Sampling\ngrid_sample at waypoints → local features')
# waypoints → spatial
arr(ax, 5.0+3.5/2, r5, 11.5-3.5/2, r5, c=C_WP[1])
# env spatial → spatial sampling
ax.annotate('', xy=(11.5+1.0, r5+0.7/2), xytext=(2.5+bw/2, r2-0.2),
            arrowprops=dict(arrowstyle='->', color='#90CAF9', lw=1.0,
                           connectionstyle='arc3,rad=0.25'), zorder=4)
ax.text(13.8, r5+0.55, 'spatial feat.', fontsize=7, color='#1976D2', ha='center',
        fontweight='bold')

# ════════════════════════════════════════
# ROW 6 (y=5.0): SEGMENT CONDITIONING
# ════════════════════════════════════════
r6 = 5.0
box(ax, 8.0, r6, 5.0, 0.65, *C_WP,
    'Segment Conditioning:  [wp_start, wp_end, local_env]  →  per-segment 256d')
arr(ax, 5.0, r5-0.7/2, 6.5, r6+0.65/2, c=C_WP[1])
arr(ax, 11.5, r5-0.7/2, 9.5, r6+0.65/2, c=C_WP[1])

# ════════════════════════════════════════
# ROW 7 (y=2.5-4.0): AUTOREGRESSIVE DECODER
# ════════════════════════════════════════
dy_top, dy_bot = 4.0, 2.2
p = FancyBboxPatch((1.5, dy_bot), 13.0, dy_top-dy_bot,
                   boxstyle="round,pad=0.02", facecolor=C_DEC[0],
                   edgecolor=C_DEC[1], lw=2.5, alpha=0.35, zorder=0)
ax.add_patch(p)
ax.text(1.7, dy_top-0.15, 'Autoregressive LSTM Decoder  (t = 1 ... 360)', ha='left',
        fontsize=10, fontweight='bold', color=C_DEC[1], zorder=1)

# Decoder internal boxes
dc = (dy_top + dy_bot) / 2 - 0.1
box(ax, 3.5, dc, 2.2, 0.7, C_DEC[0], C_DEC[1], 'LSTM Cell\n2-layer, 256d')
box(ax, 7.0, dc, 2.0, 0.7, C_DEC[0], C_DEC[1], 'Output FC\n256 → 2')
box(ax, 10.5, dc, 2.2, 0.7, C_DEC[0], C_DEC[1], 'pos += delta\n(cumulative)')

arr(ax, 3.5+2.2/2, dc, 7.0-2.0/2, dc, c=C_DEC[1])
arr(ax, 7.0+2.0/2, dc, 10.5-2.2/2, dc, c=C_DEC[1])

# Feedback loop (curved arrow from pos back to LSTM input)
ax.annotate('', xy=(3.5-2.2/2, dc-0.15), xytext=(10.5+2.2/2, dc-0.15),
            arrowprops=dict(arrowstyle='->', color=C_DEC[1], lw=1.2,
                           connectionstyle='arc3,rad=-0.35'), zorder=5)
ax.text(7.0, dy_bot+0.05, 'feedback: prev_pos(t-1)', fontsize=7,
        color=C_DEC[1], ha='center', fontstyle='italic', zorder=6)

# context → decoder (from fusion)
arr(ax, 7.5, r4-0.7/2, 3.5-0.3, dy_top, c=C_FUS[1])
ax.text(4.8, 4.65, 'context (256d)', fontsize=8, color=C_FUS[1],
        fontweight='bold', ha='center', zorder=6)

# seg_cond → decoder
arr(ax, 8.0, r6-0.65/2, 10.5+0.3, dy_top, c=C_WP[1])
ax.text(10.2, 4.65, 'seg_cond(t)', fontsize=8, color=C_WP[1],
        fontweight='bold', ha='center', zorder=6)

# ════════════════════════════════════════
# ROW 8 (y=1.0): OUTPUTS
# ════════════════════════════════════════
r8 = 1.1
box(ax, 5.5, r8, 5.5, 0.65, *C_OUT,
    'Predicted Trajectory:  360 × (x, y) positions in km')
arr(ax, 10.5, dy_bot, 7.0, r8+0.65/2, c=C_OUT[1])

box(ax, 12.5, r8, 2.8, 0.65, *C_OUT, 'Goal Logits\n(training loss)')
arr(ax, 12.5, r3-0.65/2, 12.5, r8+0.65/2, c=C_GOAL[1], lw=1.0)

# ════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════
items = [(C_IN, 'Input'), (C_ENC, 'Encoder'), (C_GOAL, 'Goal'),
         (C_FUS, 'Fusion'), (C_WP, 'Waypoint/Spatial'), (C_DEC, 'Decoder'), (C_OUT, 'Output')]
lx = 2.0
for (fc, ec), lb in items:
    p = FancyBboxPatch((lx, 0.15), 0.3, 0.2, boxstyle="round,pad=0.01",
                       facecolor=fc, edgecolor=ec, lw=1, alpha=0.9, zorder=2)
    ax.add_patch(p)
    ax.text(lx+0.4, 0.25, lb, fontsize=7, color=ec, fontweight='bold', va='center')
    lx += 1.9

plt.tight_layout(pad=0.3)
out = '/home/zmc/文档/programwork/outputs/paper_figures/fig_v6r_architecture'
fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}.pdf/.png')
plt.close()
