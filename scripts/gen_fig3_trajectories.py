#!/usr/bin/env python3
"""生成第3章轨迹可视化图: 图3.4(速度着色), 图3.5(车辆对比), 图3.6a(意图对比)"""
import pickle, glob, numpy as np
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
TRAJ_DIR = ROOT / 'data/processed/complete_dataset_10s/bohemian_forest'

def save(fig, name):
    fig.savefig(FIGS / f'{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {name}')

def load_traj(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

def path_to_km(traj):
    path = np.array(traj['path'])
    return (path - path[0]) / 1000.0

def path_length_km(path_km):
    return np.sum(np.sqrt(np.sum(np.diff(path_km, axis=0)**2, axis=1)))

# === 图3.4: 轨迹生成结果(速度着色) ===
print('\n--- 图3.4: 轨迹生成结果 ---')
# Pick a long trajectory
files = sorted(glob.glob(str(TRAJ_DIR / 'traj_*_intent1_type1.pkl')))
best = None
for fp in files[:100]:
    t = load_traj(fp)
    if len(t['path']) > 400:
        best = t
        break
if best is None and files:
    best = load_traj(files[0])

if best:
    pk = path_to_km(best)
    sp = np.array(best['speeds']) * 3.6
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(pk[:,0], pk[:,1], c=sp, cmap='RdYlGn_r', s=4,
                    vmin=0, vmax=min(sp.max()*1.1, 100))
    ax.plot(pk[0,0], pk[0,1], 'k*', ms=15, zorder=5, label='起点')
    ax.plot(pk[-1,0], pk[-1,1], 'r*', ms=15, zorder=5, label='终点')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('速度 (km/h)')
    dist = path_length_km(pk)
    dur = best['duration'] / 60
    ax.set_xlabel('东向距离 (km)')
    ax.set_ylabel('北向距离 (km)')
    ax.set_title(f'轨迹生成结果 (总长{dist:.1f}km, 时长{dur:.0f}min)', fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save(fig, 'fig3_4_trajectory_result')
else:
    print('  ⚠️ 无轨迹数据')

# === 图3.5: 不同车辆类型对比 ===
print('\n--- 图3.5: 不同车辆类型对比 ---')
# Check what types exist
all_files = sorted(glob.glob(str(TRAJ_DIR / 'traj_*.pkl')))
available_types = set()
for fp in all_files[:500]:
    name = Path(fp).stem
    parts = name.split('_')
    for p in parts:
        if p.startswith('type'):
            available_types.add(p)
print(f'  可用车辆类型: {sorted(available_types)}')

colors = {'type1': '#1565C0', 'type2': '#E91E63', 'type3': '#4CAF50', 'type4': '#FF9800'}
type_names = {'type1': '车辆类型1', 'type2': '车辆类型2', 'type3': '车辆类型3', 'type4': '车辆类型4'}

# Find same trajectory index with different types
fig, ax = plt.subplots(figsize=(10, 8))
found = 0
ref_start = None
for vt in sorted(available_types):
    # Find first trajectory of this type
    pattern = str(TRAJ_DIR / f'traj_000000_intent1_{vt}.pkl')
    matches = glob.glob(pattern)
    if not matches:
        # Try other indices
        pattern = str(TRAJ_DIR / f'traj_*_intent1_{vt}.pkl')
        matches = sorted(glob.glob(pattern))[:1]
    if matches:
        t = load_traj(matches[0])
        path = np.array(t['path'])
        if ref_start is None:
            ref_start = path[0]
        pk = (path - ref_start) / 1000.0
        dist = path_length_km(pk)
        avg_sp = np.mean(t['speeds']) * 3.6
        ax.plot(pk[:,0], pk[:,1], '-', color=colors.get(vt, '#999'), lw=1.5,
                label=f'{type_names.get(vt, vt)} ({dist:.1f}km, {avg_sp:.0f}km/h)')
        found += 1

if found > 0:
    ax.plot(0, 0, 'k*', ms=15, zorder=5)
    ax.set_xlabel('东向距离 (km)')
    ax.set_ylabel('北向距离 (km)')
    ax.set_title('不同车辆类型轨迹对比', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save(fig, 'fig3_5_vehicle_comparison')
else:
    print('  ⚠️ 无匹配轨迹')
    plt.close(fig)

# === 图3.6a: 不同战术意图对比 ===
print('\n--- 图3.6a: 不同战术意图对比 ---')
available_intents = set()
for fp in all_files[:500]:
    name = Path(fp).stem
    parts = name.split('_')
    for p in parts:
        if p.startswith('intent'):
            available_intents.add(p)
print(f'  可用意图: {sorted(available_intents)}')

int_colors = {'intent1': '#1565C0', 'intent2': '#E91E63', 'intent3': '#4CAF50'}
int_names = {'intent1': '意图1 (标准)', 'intent2': '意图2 (隐蔽优先)', 'intent3': '意图3 (高机动)'}

fig, ax = plt.subplots(figsize=(10, 8))
found = 0
ref_start = None
for it in sorted(available_intents):
    pattern = str(TRAJ_DIR / f'traj_000000_{it}_type1.pkl')
    matches = glob.glob(pattern)
    if not matches:
        pattern = str(TRAJ_DIR / f'traj_*_{it}_type1.pkl')
        matches = sorted(glob.glob(pattern))[:1]
    if matches:
        t = load_traj(matches[0])
        path = np.array(t['path'])
        if ref_start is None:
            ref_start = path[0]
        pk = (path - ref_start) / 1000.0
        dist = path_length_km(pk)
        disp = np.sqrt(np.sum((pk[-1] - pk[0])**2))
        sin = dist / (disp + 1e-6)
        ax.plot(pk[:,0], pk[:,1], '-', color=int_colors.get(it, '#999'), lw=1.5,
                label=f'{int_names.get(it, it)} ({dist:.1f}km, 弯曲度{sin:.2f})')
        found += 1

if found > 0:
    ax.plot(0, 0, 'k*', ms=15, zorder=5)
    ax.set_xlabel('东向距离 (km)')
    ax.set_ylabel('北向距离 (km)')
    ax.set_title('不同战术意图路径对比 (车辆类型1)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save(fig, 'fig3_6a_intent_comparison')
else:
    print('  ⚠️ 无匹配轨迹')
    plt.close(fig)

print('\n完成!')
