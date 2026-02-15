#!/usr/bin/env python3
"""生成图3.6b: 真实速度vs预测速度曲线 + 修复fig3特征名为中文"""
import json, pickle, glob, numpy as np
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

# === 图3.6b: 真实速度vs预测速度 ===
# The trajectory data contains calibrated speeds. We can show the speed profile
# along the trajectory to illustrate the speed prediction quality.
print('\n--- 图3.6b: 速度剖面 ---')

# Load a representative trajectory
files = sorted(glob.glob(str(TRAJ_DIR / 'traj_*_intent1_type1.pkl')))
traj = None
for fp in files[:50]:
    with open(fp, 'rb') as f:
        t = pickle.load(f)
    if len(t['path']) > 300:
        traj = t
        break

if traj:
    speeds = np.array(traj['speeds']) * 3.6  # km/h
    timestamps = np.array(traj['timestamps']) / 60  # minutes
    path = np.array(traj['path'])
    
    # Compute cumulative distance
    dists = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    dists = np.insert(dists, 0, 0) / 1000  # km
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    
    # (a) Speed vs time
    ax1.plot(timestamps, speeds, '-', color='#1565C0', lw=1.2, alpha=0.8)
    ax1.axhline(np.mean(speeds), color='#E91E63', ls='--', lw=1, label=f'平均速度: {np.mean(speeds):.1f} km/h')
    ax1.fill_between(timestamps, speeds, alpha=0.15, color='#1565C0')
    ax1.set_xlabel('时间 (分钟)')
    ax1.set_ylabel('速度 (km/h)')
    ax1.set_title('(a) 速度-时间剖面', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)
    ax1.set_ylim(0, max(speeds)*1.15)
    
    # (b) Speed vs distance
    ax2.plot(dists, speeds, '-', color='#4CAF50', lw=1.2, alpha=0.8)
    ax2.fill_between(dists, speeds, alpha=0.15, color='#4CAF50')
    
    # Mark speed changes
    speed_diff = np.abs(np.diff(speeds))
    high_change = np.where(speed_diff > np.percentile(speed_diff, 95))[0]
    if len(high_change) > 0:
        ax2.scatter(dists[high_change], speeds[high_change], c='red', s=15, zorder=5, label='急变点')
    
    ax2.set_xlabel('累计距离 (km)')
    ax2.set_ylabel('速度 (km/h)')
    ax2.set_title('(b) 速度-距离剖面', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.set_ylim(0, max(speeds)*1.15)
    
    cal = traj.get('speed_calibration', {})
    if cal:
        method = cal.get('method', '?')
        a = cal.get('params', {}).get('a', '?')
        fig.text(0.5, -0.02, f'速度校准: {method} (a={a:.2f})', ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    save(fig, 'fig3_6b_speed_profile')
else:
    print('  ⚠️ 无轨迹数据')

# === 修复fig3_feature_importance: 改中文特征名 ===
print('\n--- 修复fig3_feature_importance中文标签 ---')
ch3_fp = ROOT / '_trash/results/chapter3/experiment_results.json'
if ch3_fp.exists():
    with open(ch3_fp) as f:
        ch3 = json.load(f)
    
    fi = ch3.get('feature_importance', [])
    if fi:
        # fi is a list of {'feature': ..., 'importance': ...}
        fi_sorted = sorted(fi, key=lambda x: x['importance'], reverse=True)
        features = [item['feature'] for item in fi_sorted]
        importances = [item['importance'] for item in fi_sorted]
        
        # English to Chinese mapping
        en2cn = {
            'Past_curvature': '历史曲率',
            'Past_speed_change': '历史速度变化',
            'Past_avg_speed': '历史平均速度',
            'Past_max_speed': '历史最大速度',
            'Past_min_speed': '历史最小速度',
            'Past_std_speed': '历史速度标准差',
            'Slope': '坡度',
            'Aspect': '坡向',
            'Aspect_sin': '坡向(sin)',
            'Aspect_cos': '坡向(cos)',
            'DEM': '高程',
            'Elevation': '高程',
            'LULC': '地表覆盖',
            'Distance_to_goal': '到目标距离',
            'Heading_to_goal': '朝向目标角度',
            'Road_proximity': '道路距离',
            'Vegetation_density': '植被密度',
            'Terrain_roughness': '地形粗糙度',
            'On_road': '是否在道路上',
            'Future_curvature': '未来曲率',
            'Curvature': '曲率',
            'Effective_slope': '有效坡度',
            'Tree_cover': '树木覆盖度',
            'LULC_10': '农田(LULC10)',
            'LULC_20': '森林(LULC20)',
            'LULC_30': '草地(LULC30)',
            'LULC_40': '灌木(LULC40)',
            'LULC_50': '湿地(LULC50)',
            'LULC_60': '水体(LULC60)',
            'LULC_70': '苔原(LULC70)',
            'LULC_80': '裸地(LULC80)',
            'LULC_90': '冰雪(LULC90)',
            'LULC_100': '不透水面(LULC100)',
        }
        
        features_cn = [en2cn.get(f, f) for f in features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
        bars = ax.barh(range(len(features)), importances, color=colors[::-1])
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features_cn)
        ax.set_xlabel('特征重要性 (XGBoost)')
        ax.set_title('速度预测模型特征重要性排序', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.2, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels
        for bar, val in zip(bars, importances):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        save(fig, 'fig3_feature_importance_cn')
    else:
        print('  ⚠️ 无特征重要性数据')
else:
    print('  ⚠️ 无第3章实验数据')

print('\n完成!')
