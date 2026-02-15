#!/usr/bin/env python3
"""生成图4.11(K敏感性)和图4.12(观测长度敏感性) — 使用真实实验数据"""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / 'outputs' / 'paper_final'
FIGS.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.savefig(FIGS / f'{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {name}')

# === 图4.11: K敏感性 ===
fp = ROOT / 'outputs/evaluation/control_variables/candidate_k_sensitivity.json'
with open(fp) as f: data = json.load(f)
ks = sorted(data.keys(), key=int)
k_vals = [int(k) for k in ks]
ades = [data[k]['ade']/1000 for k in ks]
fdes = [data[k]['fde']/1000 for k in ks]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_vals, ades, 'o-', color='#1565C0', lw=2, ms=8, label='ADE')
ax.plot(k_vals, fdes, 's--', color='#E91E63', lw=2, ms=8, label='FDE')
for k, a in zip(k_vals, ades):
    ax.annotate(f'{a:.2f}', (k, a), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='#1565C0')
ax.set_xlabel('候选目标数量 K')
ax.set_ylabel('误差 (km)')
ax.set_title('候选目标数量K对预测性能的影响', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
r = max(ades)-min(ades)
ax.text(0.95, 0.05, f'ADE变化范围: {r*1000:.0f}m (<1%)', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, style='italic', color='gray')
plt.tight_layout()
save(fig, 'fig4_11_k_sensitivity')

# === 图4.12: 观测长度敏感性 ===
fp2 = ROOT / 'outputs/evaluation/control_variables/observation_length_sensitivity.json'
with open(fp2) as f: data2 = json.load(f)
keys2 = sorted(data2.keys(), key=int)
mins = [data2[k]['obs_minutes'] for k in keys2]
ades2 = [data2[k]['ade']/1000 for k in keys2]
fdes2 = [data2[k]['fde']/1000 for k in keys2]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mins, ades2, 'o-', color='#1565C0', lw=2, ms=8, label='ADE')
ax.plot(mins, fdes2, 's--', color='#E91E63', lw=2, ms=8, label='FDE')
for m, a in zip(mins, ades2):
    ax.annotate(f'{a:.2f}', (m, a), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='#1565C0')
ax.set_xlabel('观测时长 (分钟)')
ax.set_ylabel('误差 (km)')
ax.set_title('观测长度对预测性能的影响', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
r2 = max(ades2)-min(ades2)
ax.text(0.95, 0.05, f'ADE变化范围: {r2*1000:.0f}m', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, style='italic', color='gray')
plt.tight_layout()
save(fig, 'fig4_12_obs_length_sensitivity')

print('\n完成!')
