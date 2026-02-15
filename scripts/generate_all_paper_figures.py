#!/usr/bin/env python3
"""
论文第3章+第4章所有图表统一生成脚本 (唯一权威版本)
输出目录: outputs/paper_final/
数据源:
  - 第3章: _trash/results/chapter3/experiment_results.json (速度预测K-fold/LOOCV/SHAP)
  - 第4章: phase_v2_with_faithful (含真正YNet/PECNet), ablation, waypoint_ablation
使用正确的中文字体 (Noto Sans CJK JP)
"""

import re, json, sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ============================================================
#  字体设置 (全局)
# ============================================================
_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(_FONT_PATH)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / 'outputs' / 'paper_final'
FIGS.mkdir(parents=True, exist_ok=True)


# ============================================================
#  辅助函数
# ============================================================
def save_fig(fig, name):
    fig.savefig(FIGS / f'{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {name}')


def parse_training_log(log_path):
    """解析 'epoch X/Y train_loss=... ADE=... | val_loss=... ADE=...' 格式"""
    epochs = []
    pattern = re.compile(
        r'epoch\s+(\d+)/(\d+)\s+train_loss=([\d.]+)\s+ADE=(\d+)m\s+FDE=(\d+)m\s+\|\s+'
        r'val_loss=([\d.]+)\s+ADE=(\d+)m\s+FDE=(\d+)m'
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append({
                    'epoch': int(m.group(1)),
                    'train_loss': float(m.group(3)),
                    'train_ade': int(m.group(4)),
                    'train_fde': int(m.group(5)),
                    'val_loss': float(m.group(6)),
                    'val_ade': int(m.group(7)),
                    'val_fde': int(m.group(8)),
                })
    return epochs


# ============================================================
#  第3章: 速度预测模型验证
# ============================================================
CH3_DATA = ROOT / '_trash' / 'results' / 'chapter3' / 'experiment_results.json'


def fig_ch3_kfold():
    """图3.x: K-fold交叉验证结果"""
    print('\n--- 第3章: K-fold交叉验证 ---')
    if not CH3_DATA.exists():
        print('  ⚠️ 无第3章实验数据，跳过')
        return

    with open(CH3_DATA) as f:
        data = json.load(f)

    folds = data['kfold_results']
    summary = data['kfold_summary']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) R² per fold
    train_r2 = [f['train_r2'] for f in folds]
    val_r2 = [f['val_r2'] for f in folds]
    x = np.arange(1, 6)
    w = 0.35
    axes[0].bar(x - w/2, train_r2, w, label='训练集', color='#1565C0', alpha=0.8)
    axes[0].bar(x + w/2, val_r2, w, label='验证集', color='#E91E63', alpha=0.8)
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('R²')
    axes[0].set_title('(a) R²')
    axes[0].set_xticks(x)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim(0.6, 1.0)

    # (b) RMSE per fold
    train_rmse = [f['train_rmse'] for f in folds]
    val_rmse = [f['val_rmse'] for f in folds]
    axes[1].bar(x - w/2, train_rmse, w, label='训练集', color='#1565C0', alpha=0.8)
    axes[1].bar(x + w/2, val_rmse, w, label='验证集', color='#E91E63', alpha=0.8)
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('RMSE (m/s)')
    axes[1].set_title('(b) RMSE')
    axes[1].set_xticks(x)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    # (c) MAPE per fold
    train_mape = [f['train_mape'] for f in folds]
    val_mape = [f['val_mape'] for f in folds]
    axes[2].bar(x - w/2, train_mape, w, label='训练集', color='#1565C0', alpha=0.8)
    axes[2].bar(x + w/2, val_mape, w, label='验证集', color='#E91E63', alpha=0.8)
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].set_title('(c) MAPE')
    axes[2].set_xticks(x)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('5-fold交叉验证结果', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig3_kfold_validation')


def fig_ch3_feature_importance():
    """图3.x: 特征重要性"""
    print('\n--- 第3章: 特征重要性 ---')
    if not CH3_DATA.exists():
        print('  ⚠️ 无第3章实验数据，跳过')
        return

    with open(CH3_DATA) as f:
        data = json.load(f)

    fi = data['feature_importance']
    names = [f['feature'] for f in fi]
    imps = [f['importance'] for f in fi]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(names))
    colors = ['#1565C0' if imp > 0.1 else '#42A5F5' if imp > 0.01 else '#90CAF9'
              for imp in imps]
    ax.barh(y, imps, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('特征重要性 (XGBoost gain)')
    ax.set_title('速度预测模型特征重要性排序', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig3_feature_importance')


def fig_ch3_loocv():
    """图3.x: LOOCV结果"""
    print('\n--- 第3章: LOOCV ---')
    if not CH3_DATA.exists():
        print('  ⚠️ 无第3章实验数据，跳过')
        return

    with open(CH3_DATA) as f:
        data = json.load(f)

    loocv = data['loocv_results']
    avg_r2 = data['loocv_avg_r2']
    human_r2 = data.get('human_consistency_r2', None)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(loocv) + 1)
    ax.bar(x, loocv, color='#42A5F5', alpha=0.8, label='LOOCV R²')
    ax.axhline(avg_r2, color='#1565C0', ls='--', lw=2,
               label=f'平均 R²={avg_r2:.3f}')
    if human_r2 is not None:
        ax.axhline(human_r2, color='#E91E63', ls=':', lw=2,
                   label=f'人类一致性 R²={human_r2:.3f}')
    ax.set_xlabel('轨迹编号')
    ax.set_ylabel('R²')
    ax.set_title('留一轨迹交叉验证 (LOOCV)', fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.3, 0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig3_loocv')


def print_ch3_tables():
    """打印第3章数值表格"""
    if not CH3_DATA.exists():
        return

    with open(CH3_DATA) as f:
        data = json.load(f)

    print('\n' + '=' * 60)
    print('论文第3章数值结果汇总 (真实实验数据)')
    print('=' * 60)

    s = data['kfold_summary']
    print(f'\n--- 表3.8: K-fold交叉验证 ---')
    print(f'{"":>8} {"R²":>8} {"RMSE":>10} {"MAPE(%)":>10}')
    print('-' * 38)
    print(f'  {"训练集":<6} {s["avg_train_r2"]:>7.4f} {s["avg_train_rmse"]:>9.3f} {s["avg_train_mape"]:>9.2f}')
    print(f'  {"验证集":<6} {s["avg_val_r2"]:>7.4f} {s["avg_val_rmse"]:>9.3f} {s["avg_val_mape"]:>9.2f}')

    print(f'\n  LOOCV 平均 R²: {data["loocv_avg_r2"]:.4f}')
    if 'human_consistency_r2' in data:
        print(f'  人类一致性 R²: {data["human_consistency_r2"]}')

    print(f'\n--- 特征重要性 Top 5 ---')
    for fi in data['feature_importance'][:5]:
        print(f'  {fi["feature"]:<20} {fi["importance"]:.4f}')


# ============================================================
#  第4章开始
# ============================================================

# ============================================================
#  图4.6: 训练收敛曲线
# ============================================================
def fig_training_curves():
    print('\n--- 图4.6: 训练收敛曲线 ---')
    logs = {}

    # YNet / PECNet 日志
    for name, path in [('YNet', 'runs/ynet_d1_training.log'),
                        ('PECNet', 'runs/pecnet_d1_training.log')]:
        p = ROOT / path
        if p.exists():
            data = parse_training_log(p)
            if data:
                logs[name] = data
                print(f'  {name}: {len(data)} epochs')

    # V6R 日志 (JSON格式)
    v6r_log = ROOT / 'runs/cross_region/single_reg_V6R_drop0.15_bohemian_forest/training_log.json'
    if v6r_log.exists():
        with open(v6r_log) as f:
            v6r_data = json.load(f)
        logs['TerraTNT(本文)'] = [{
            'epoch': d['epoch'],
            'train_loss': d['train_loss'],
            'val_ade': d['bohemian_forest_ade'],
            'val_fde': d['bohemian_forest_fde'],
        } for d in v6r_data]
        print(f'  TerraTNT(本文): {len(v6r_data)} epochs')

    if not logs:
        print('  ⚠️ 无训练日志，跳过')
        return

    colors = {'TerraTNT(本文)': '#1565C0', 'YNet': '#E91E63', 'PECNet': '#9C27B0'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, data in logs.items():
        ep = [d['epoch'] for d in data]
        val_ade = [d['val_ade'] / 1000 for d in data]
        c = colors.get(name, '#666')
        axes[0].plot(ep, val_ade, '-', color=c, linewidth=2, label=name)

        if 'train_loss' in data[0]:
            train_loss = [d['train_loss'] for d in data]
            axes[1].plot(ep, train_loss, '-', color=c, linewidth=2, label=name)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('验证集 ADE (km)')
    axes[0].set_title('(a) 验证集ADE收敛曲线')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('训练损失')
    axes[1].set_title('(b) 训练损失收敛曲线')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    save_fig(fig, 'fig4_6_training_curves')


# ============================================================
#  图4.7: Phase V2 模型对比柱状图
# ============================================================
def fig_phase_comparison():
    print('\n--- 图4.7: Phase V2 模型对比 ---')
    rp = ROOT / 'outputs/evaluation/phase_v2_with_faithful/phase_v2_results.json'
    if not rp.exists():
        print('  ⚠️ 无评估结果，跳过')
        return

    with open(rp) as f:
        results = json.load(f)

    models = ['V6R_Robust', 'YNet', 'PECNet', 'LSTM_only']
    names  = {'V6R_Robust': 'TerraTNT(本文)', 'YNet': 'YNet',
              'PECNet': 'PECNet', 'LSTM_only': 'SimpleLSTM'}
    clrs   = {'V6R_Robust': '#1565C0', 'YNet': '#E91E63',
              'PECNet': '#9C27B0', 'LSTM_only': '#FF9800'}

    phases = ['P1a', 'P2a', 'P3a']
    phase_labels = ['Phase 1\n(精确终点)', 'Phase 2\n(区域先验)', 'Phase 3\n(无先验)']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(phases))
    w = 0.18

    for i, mk in enumerate(models):
        ades = []
        for pid in phases:
            if pid in results and mk in results[pid]['models']:
                ades.append(results[pid]['models'][mk]['ade_mean'] / 1000)
            else:
                ades.append(0)
        off = (i - len(models) / 2 + 0.5) * w
        bars = ax.bar(x + off, ades, w, label=names[mk], color=clrs[mk], alpha=0.85)
        for bar, v in zip(bars, ades):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('评估阶段')
    ax.set_ylabel('ADE (km)')
    ax.set_title('不同先验条件下的模型性能对比', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig4_7_phase_comparison')


# ============================================================
#  图4.8: 消融实验柱状图
# ============================================================
def fig_ablation():
    print('\n--- 图4.8: 消融实验 ---')
    ap = ROOT / 'outputs/evaluation/ablation/ablation_results.json'
    if not ap.exists():
        print('  ⚠️ 无消融结果，跳过')
        return

    with open(ap) as f:
        abl = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 模块消融
    mod = abl['module_ablation']
    keys_a = ['full', 'no_env', 'no_history', 'no_goal_cls']
    if 'mlp_decoder' in mod:
        keys_a.append('mlp_decoder')
    names_a = [mod[k]['name'] for k in keys_a]
    ades_a  = [mod[k]['ade_mean'] / 1000 for k in keys_a]
    clrs_a  = ['#1565C0', '#E53935', '#FB8C00', '#43A047', '#7B1FA2'][:len(keys_a)]

    bars = axes[0].bar(range(len(names_a)), ades_a, color=clrs_a, alpha=0.85)
    for bar, v in zip(bars, ades_a):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    axes[0].set_xticks(range(len(names_a)))
    axes[0].set_xticklabels(names_a, fontsize=9, rotation=15)
    axes[0].set_ylabel('ADE (km)')
    axes[0].set_title('(a) 模块消融')
    axes[0].grid(axis='y', alpha=0.3)

    # (b) 环境通道消融
    ch = abl['channel_ablation']
    keys_b = ['full', 'no_dem', 'no_lulc', 'no_osm']
    names_b = [ch[k]['name'] for k in keys_b]
    ades_b  = [ch[k]['ade_mean'] / 1000 for k in keys_b]
    clrs_b  = ['#1565C0', '#E53935', '#FB8C00', '#43A047']

    bars = axes[1].bar(range(len(names_b)), ades_b, color=clrs_b, alpha=0.85)
    for bar, v in zip(bars, ades_b):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    axes[1].set_xticks(range(len(names_b)))
    axes[1].set_xticklabels(names_b, fontsize=9, rotation=15)
    axes[1].set_ylabel('ADE (km)')
    axes[1].set_title('(b) 环境通道消融')
    axes[1].grid(axis='y', alpha=0.3)

    # (c) 运动学/历史特征消融
    if 'kinematics_ablation' in abl:
        kin = abl['kinematics_ablation']
        keys_c = [k for k in kin.keys()]
        names_c = [kin[k]['name'] for k in keys_c]
        ades_c  = [kin[k]['ade_mean'] / 1000 for k in keys_c]
        clrs_c  = ['#1565C0'] + ['#7B1FA2', '#AB47BC', '#CE93D8',
                                  '#E1BEE7', '#F44336', '#FF7043'][:len(keys_c)-1]

        bars = axes[2].bar(range(len(names_c)), ades_c, color=clrs_c, alpha=0.85)
        for bar, v in zip(bars, ades_c):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        axes[2].set_xticks(range(len(names_c)))
        axes[2].set_xticklabels(names_c, fontsize=8, rotation=25, ha='right')
        axes[2].set_ylabel('ADE (km)')
        axes[2].set_title('(c) 运动学/历史特征消融')
        axes[2].grid(axis='y', alpha=0.3)
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig4_8_ablation')


# ============================================================
#  图4.9 / 4.10: 控制变量
# ============================================================
def fig_control_variables():
    print('\n--- 图4.9/4.10: 控制变量 ---')

    # Waypoint消融 (唯一有真实数据的控制变量实验)
    wp_path = ROOT / 'outputs/evaluation/control_variables/waypoint_ablation_results.json'
    if not wp_path.exists():
        print('  ⚠️ 无waypoint消融结果，跳过')
        return

    with open(wp_path) as f:
        wp_data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    wps = sorted(wp_data.keys(), key=int)
    w_ade = [wp_data[w]['ade_mean'] / 1000 for w in wps]
    w_fde = [wp_data[w]['fde_mean'] / 1000 for w in wps]

    ax.plot([int(w) for w in wps], w_ade, 'o-', color='#1565C0', lw=2, ms=8, label='ADE')
    ax.plot([int(w) for w in wps], w_fde, 's--', color='#E91E63', lw=2, ms=8, label='FDE')
    ax.set_xlabel('Waypoint数量')
    ax.set_ylabel('误差 (km)')
    ax.set_title('Waypoint数量对预测性能的影响', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig4_9_waypoint_ablation')


# ============================================================
#  表格汇总 (打印 + 保存 LaTeX)
# ============================================================
def print_all_tables():
    print('\n' + '=' * 60)
    print('论文第4章数值结果汇总')
    print('=' * 60)

    # 表4.6-4.8: Phase V2 结果
    rp = ROOT / 'outputs/evaluation/phase_v2_with_faithful/phase_v2_results.json'
    if rp.exists():
        with open(rp) as f:
            results = json.load(f)

        paper_models = ['V6R_Robust', 'YNet', 'PECNet', 'LSTM_only']
        paper_names = {'V6R_Robust': 'TerraTNT(本文)', 'YNet': 'YNet',
                       'PECNet': 'PECNet', 'LSTM_only': 'SimpleLSTM'}
        phase_map = {
            'P1a': 'Phase 1 (精确终点)',
            'P2a': 'Phase 2 (区域先验σ=10km)',
            'P3a': 'Phase 3 (无先验)',
        }

        for pid, label in phase_map.items():
            print(f'\n--- 表: {label} ---')
            print(f'{"模型":<20} {"ADE(km)":>10} {"FDE(km)":>10}')
            print('-' * 42)
            if pid in results:
                for mk in paper_models:
                    if mk in results[pid]['models']:
                        ms = results[pid]['models'][mk]
                        print(f'  {paper_names[mk]:<18} {ms["ade_mean"]/1000:>9.2f} {ms["fde_mean"]/1000:>9.2f}')

    # 消融结果
    ap = ROOT / 'outputs/evaluation/ablation/ablation_results.json'
    if ap.exists():
        with open(ap) as f:
            abl = json.load(f)

        base_ade = abl['module_ablation']['full']['ade_mean']

        print(f'\n--- 表4.13: 模块消融 ---')
        print(f'{"配置":<20} {"ADE(km)":>10} {"ΔADE":>10} {"变化%":>10}')
        print('-' * 52)
        mod_keys = ['full', 'no_env', 'no_history', 'no_goal_cls']
        if 'mlp_decoder' in abl['module_ablation']:
            mod_keys.append('mlp_decoder')
        for k in mod_keys:
            r = abl['module_ablation'][k]
            delta = (r['ade_mean'] - base_ade) / 1000
            pct = (r['ade_mean'] - base_ade) / base_ade * 100
            print(f'  {r["name"]:<18} {r["ade_mean"]/1000:>9.2f} {delta:>+9.2f} {pct:>+9.1f}%')

        if 'kinematics_ablation' in abl:
            print(f'\n--- 表4.15: 运动学/历史特征消融 ---')
            print(f'{"配置":<22} {"ADE(km)":>10} {"ΔADE":>10} {"变化%":>10}')
            print('-' * 54)
            for k, r in abl['kinematics_ablation'].items():
                delta = (r['ade_mean'] - base_ade) / 1000
                pct = (r['ade_mean'] - base_ade) / base_ade * 100
                print(f'  {r["name"]:<20} {r["ade_mean"]/1000:>9.2f} {delta:>+9.2f} {pct:>+9.1f}%')

        print(f'\n--- 表4.14: 环境通道消融 ---')
        print(f'{"配置":<20} {"ADE(km)":>10} {"ΔADE":>10} {"变化%":>10}')
        print('-' * 52)
        for k in ['full', 'no_dem', 'no_lulc', 'no_osm']:
            r = abl['channel_ablation'][k]
            delta = (r['ade_mean'] - base_ade) / 1000
            pct = (r['ade_mean'] - base_ade) / base_ade * 100
            print(f'  {r["name"]:<18} {r["ade_mean"]/1000:>9.2f} {delta:>+9.2f} {pct:>+9.1f}%')

    # Waypoint消融结果
    wp_path = ROOT / 'outputs/evaluation/control_variables/waypoint_ablation_results.json'
    if wp_path.exists():
        with open(wp_path) as f:
            wp_data = json.load(f)
        print(f'\n--- 表4.11: Waypoint数量 ---')
        print(f'{"WP数":>5} {"ADE(km)":>10} {"FDE(km)":>10}')
        print('-' * 27)
        for w, r in sorted(wp_data.items(), key=lambda x: int(x[0])):
            print(f'{w:>5} {r["ade_mean"]/1000:>9.2f} {r["fde_mean"]/1000:>9.2f}')


# ============================================================
#  图4.10: 三阶段热力图 (phase_v2_with_faithful, 真实YNet/PECNet)
# ============================================================
def fig_phase_heatmap():
    print('\n--- 图4.10: 三阶段评估热力图 ---')
    rp = ROOT / 'outputs/evaluation/phase_v2_with_faithful/phase_v2_results.json'
    if not rp.exists():
        print('  ⚠️ 无评估结果，跳过')
        return

    with open(rp) as f:
        results = json.load(f)

    models = ['V6R_Robust', 'YNet', 'PECNet', 'LSTM_only']
    model_labels = ['TerraTNT(本文)', 'YNet', 'PECNet', 'SimpleLSTM']
    phases = ['P1a', 'P2a', 'P3a']
    phase_labels = ['Phase 1\n(精确终点)', 'Phase 2\n(区域先验)', 'Phase 3\n(无先验)']

    matrix = np.zeros((len(models), len(phases)))
    for j, pid in enumerate(phases):
        if pid not in results:
            continue
        for i, mk in enumerate(models):
            if mk in results[pid]['models']:
                matrix[i, j] = results[pid]['models'][mk]['ade_mean'] / 1000
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phase_labels, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=10)

    for i in range(len(models)):
        for j in range(len(phases)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = 'white' if v > np.nanmax(matrix) * 0.6 else 'black'
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                       fontsize=12, fontweight='bold', color=color)

    ax.set_title('三阶段评估ADE对比 (km)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='ADE (km)', shrink=0.8)
    plt.tight_layout()
    save_fig(fig, 'fig4_10_phase_heatmap')


# ============================================================
#  箱线图 / 时间趋势图 (v5 per-sample 数据, 正确模型标签)
#  注意: v5 中 LSTM_Env_Goal ≠ YNet, Seq2Seq_Attn ≠ PECNet
#  这里使用真实模型名称，不伪装
# ============================================================
V5_DIR = ROOT / 'outputs' / 'evaluation' / 'phase_diagnostic_v5'
V5_MODELS = ['V6R_Robust', 'LSTM_Env_Goal', 'Seq2Seq_Attn', 'LSTM_only']
V5_LABELS = ['TerraTNT(本文)', 'LSTM+Env+Goal', 'Seq2Seq+Attn', 'SimpleLSTM']
V5_COLORS = ['#1565C0', '#FF9800', '#4CAF50', '#9E9E9E']


def load_v5_phase(phase):
    fpath = V5_DIR / f'results_{phase}.json'
    if not fpath.exists():
        return None
    with open(fpath) as f:
        return json.load(f)


def get_model_ades(samples, model_key):
    return np.array([s['models'][model_key]['ade_m']
                     for s in samples if model_key in s['models']])


def get_model_temporal(samples, model_key):
    early, mid, late = [], [], []
    for s in samples:
        if model_key in s['models']:
            m = s['models'][model_key]
            early.append(m['early_ade_m'])
            mid.append(m['mid_ade_m'])
            late.append(m['late_ade_m'])
    return np.array(early), np.array(mid), np.array(late)


def fig_boxplot(phase, phase_label, out_name):
    samples = load_v5_phase(phase)
    if not samples:
        print(f'  [SKIP] {out_name}: no data for {phase}')
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []
    for mk, ml, mc in zip(V5_MODELS, V5_LABELS, V5_COLORS):
        ades = get_model_ades(samples, mk) / 1000
        if len(ades) > 0:
            data.append(ades)
            labels.append(ml)
            colors.append(mc)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=2, alpha=0.3))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('ADE (km)')
    ax.set_title(f'{phase_label}条件下各模型误差分布', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig(fig, out_name)


def fig_temporal_trend(phase, phase_label, out_name):
    samples = load_v5_phase(phase)
    if not samples:
        print(f'  [SKIP] {out_name}: no data for {phase}')
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    time_labels = ['0-15min\n(Early)', '15-40min\n(Mid)', '40-60min\n(Late)']
    x = np.arange(3)
    for mk, ml, mc in zip(V5_MODELS, V5_LABELS, V5_COLORS):
        early, mid, late = get_model_temporal(samples, mk)
        if len(early) > 0:
            means = [np.mean(early)/1000, np.mean(mid)/1000, np.mean(late)/1000]
            stds = [np.std(early)/1000, np.std(mid)/1000, np.std(late)/1000]
            ax.errorbar(x, means, yerr=stds, label=ml, color=mc,
                       marker='o', markersize=8, linewidth=2, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(time_labels, fontsize=10)
    ax.set_ylabel('ADE (km)')
    ax.set_title(f'{phase_label}条件下预测误差随时间变化', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig(fig, out_name)


# ============================================================
#  跨区域图表占位 (需要真实实验数据后替换)
# ============================================================
def fig_cross_region_placeholder():
    print('\n--- 跨区域图表 (占位) ---')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5,
            '跨区域实验尚未完成\n需要完成4区域轨迹生成+训练+评估后\n用真实数据替换此图',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=16, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('图4.25-4.27: 跨区域泛化实验 (待完成)', fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(fig, 'fig4_25_26_27_PLACEHOLDER')


# ============================================================
#  主函数
# ============================================================
def main():
    import shutil
    print(f'输出目录: {FIGS}')
    print(f'字体: {plt.rcParams["font.sans-serif"]}')

    # === 第3章图表 (速度预测) ===
    fig_ch3_kfold()
    fig_ch3_feature_importance()
    fig_ch3_loocv()
    print_ch3_tables()

    # 复制已有的SHAP图
    shap_src = ROOT / '_trash' / 'results' / 'chapter3' / 'fig3_11_shap_importance.png'
    if shap_src.exists():
        shutil.copy2(shap_src, FIGS / 'fig3_shap_importance.png')
        print('  ✅ fig3_shap_importance (copied from _trash)')

    # === 第4章图表 ===
    fig_training_curves()                                    # 训练日志
    fig_phase_comparison()                                   # phase_v2_with_faithful
    fig_ablation()                                           # ablation_results.json
    fig_control_variables()                                  # waypoint_ablation (唯一真实数据)
    fig_phase_heatmap()                                      # phase_v2_with_faithful

    # === v5 per-sample 图表 (正确标注模型名) ===
    fig_boxplot('fas1', 'Phase 1', 'fig4_box_phase1')
    fig_temporal_trend('fas1', 'Phase 1', 'fig4_time_phase1')
    fig_boxplot('fas2', 'Phase 2', 'fig4_box_phase2')
    fig_temporal_trend('fas2', 'Phase 2', 'fig4_time_phase2')
    fig_boxplot('fas3', 'Phase 3', 'fig4_box_phase3')
    fig_temporal_trend('fas3', 'Phase 3', 'fig4_time_phase3')

    # === 跨区域占位 ===
    fig_cross_region_placeholder()

    # === 表格数据 ===
    print_all_tables()

    # === 复制架构图和预测示例图 ===
    for src_dir_name in ['paper_figures', 'paper_ch4_figures']:
        src_dir = ROOT / 'outputs' / src_dir_name
        if not src_dir.exists():
            continue
        for f in src_dir.iterdir():
            if f.name.startswith('fig_v6r_architecture') or f.name.startswith('fig_example_'):
                shutil.copy2(f, FIGS / f.name)
    print('  ✅ 架构图 + 预测示例图 (copied)')

    print(f'\n{"="*60}')
    print(f'所有论文图表已保存到: {FIGS}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
