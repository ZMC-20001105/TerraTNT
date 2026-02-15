#!/usr/bin/env python3
"""Generate paper figures from Phase V2 evaluation results."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / 'outputs' / 'evaluation' / 'phase_v2' / 'phase_v2_results.json'
FIGS = ROOT / 'outputs' / 'paper_figures'
FIGS.mkdir(parents=True, exist_ok=True)


def load():
    with open(RESULTS) as f:
        return json.load(f)


def fig1_bars(R):
    """Grouped bar: ADE across phases for key models."""
    phases = ['P1a','P1b','P2a','P2c','P3a']
    plabels = ['P1a\nPrecise','P1b\nOOD','P2a\nRegion\nσ=10km','P2c\nOffset\n5km','P3a\nNo Prior']
    models = [
        ('V6R_Robust','V6R (Ours)','#1565C0'),
        ('V6_Autoreg','V6','#42A5F5'),
        ('V4_WP_Spatial','V4','#90CAF9'),
        ('LSTM_Env_Goal','LSTM+Env+Goal','#FF9800'),
        ('V7_ConfGate','V7 ConfGate','#4CAF50'),
        ('Seq2Seq_Attn','Seq2Seq','#9E9E9E'),
    ]
    fig, ax = plt.subplots(figsize=(13,5.5))
    n = len(phases); nm = len(models); w = 0.12
    x = np.arange(n)
    for i,(k,lb,c) in enumerate(models):
        vals = [R[p]['models'].get(k,{}).get('ade_mean',0)/1000 for p in phases]
        off = (i - nm/2 + 0.5)*w
        bars = ax.bar(x+off, vals, w, label=lb, color=c, edgecolor='white', lw=0.5)
        for b,v in zip(bars,vals):
            if v>0: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(plabels, fontsize=9)
    ax.set_ylabel('ADE (km)'); ax.set_title('Model Performance Across Phases', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper left'); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGS/'fig1_phase_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS/'fig1_phase_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print('  fig1 done')


def fig2_lines(R):
    """Line chart: degradation across phases."""
    phases = ['P1a','P1b','P2a','P2c','P3a']
    models = [
        ('V6R_Robust','V6R (Ours)','#1565C0','-','o'),
        ('V6_Autoreg','V6','#42A5F5','--','s'),
        ('V4_WP_Spatial','V4','#90CAF9','-.','D'),
        ('LSTM_Env_Goal','LSTM+EG','#FF9800','-','^'),
        ('V7_ConfGate','V7','#4CAF50','-','v'),
        ('Seq2Seq_Attn','Seq2Seq','#9E9E9E',':','x'),
    ]
    fig, ax = plt.subplots(figsize=(10,5.5))
    x = np.arange(len(phases))
    for k,lb,c,ls,mk in models:
        vals = [R[p]['models'].get(k,{}).get('ade_mean',np.nan)/1000 for p in phases]
        ax.plot(x, vals, label=lb, color=c, ls=ls, marker=mk, ms=8, lw=2)
    ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=10)
    ax.set_ylabel('ADE (km)'); ax.set_xlabel('Phase (decreasing prior quality →)')
    ax.set_title('Performance Degradation with Prior Quality', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGS/'fig2_degradation.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS/'fig2_degradation.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print('  fig2 done')


def fig3_ablation(R):
    """Horizontal bar: ablation V2→V6R on P1a and P2a."""
    models = [
        ('LSTM_Env_Goal','V2: LSTM+Env+Goal'),
        ('V3_Waypoint','V3: +Waypoints'),
        ('V4_WP_Spatial','V4: +Spatial Env'),
        ('TerraTNT','V5: TerraTNT'),
        ('V6_Autoreg','V6: +Autoreg'),
        ('V6R_Robust','V6R: +GoalDrop'),
    ]
    colors = ['#FFB74D','#FFA726','#FF9800','#FB8C00','#2196F3','#1565C0']
    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    for ax, ph, ti in zip(axes, ['P1a','P2a'], ['P1a: Precise Goal','P2a: Region Prior']):
        names, vals, errs = [],[],[]
        for k,lb in models:
            if ph in R and k in R[ph]['models']:
                names.append(lb)
                vals.append(R[ph]['models'][k]['ade_mean']/1000)
                errs.append(R[ph]['models'][k]['ade_std']/1000)
        bars = ax.barh(range(len(names)), vals, xerr=errs, color=colors[:len(names)], edgecolor='white', capsize=3)
        for i,(b,v) in enumerate(zip(bars,vals)):
            ax.text(b.get_width()+0.1, i, f'{v:.1f}km', va='center', fontsize=9)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('ADE (km)'); ax.set_title(ti, fontsize=11, fontweight='bold')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGS/'fig3_ablation.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS/'fig3_ablation.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print('  fig3 done')


def fig4_temporal(R):
    """Early/Mid/Late ADE breakdown."""
    models = [('V6R_Robust','V6R'),('V6_Autoreg','V6'),('LSTM_Env_Goal','LSTM+EG'),
              ('V4_WP_Spatial','V4'),('Seq2Seq_Attn','Seq2Seq')]
    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    for ax, ph, ti in zip(axes, ['P1a','P3a'], ['P1a: Precise Goal','P3a: No Prior']):
        labels, early, mid, late = [],[],[],[]
        for k,lb in models:
            if ph in R and k in R[ph]['models']:
                m = R[ph]['models'][k]
                labels.append(lb); early.append(m['early_ade']/1000)
                mid.append(m['mid_ade']/1000); late.append(m['late_ade']/1000)
        x = np.arange(len(labels)); w = 0.25
        ax.bar(x-w, early, w, label='Early (0-2h)', color='#81C784')
        ax.bar(x, mid, w, label='Mid (2-7h)', color='#FFB74D')
        ax.bar(x+w, late, w, label='Late (7-10h)', color='#E57373')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('ADE (km)'); ax.set_title(ti, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGS/'fig4_temporal.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS/'fig4_temporal.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print('  fig4 done')


def print_latex_table(R):
    """Print LaTeX table."""
    phases = ['P1a','P1b','P2a','P2c','P3a']
    models = [
        ('ConstantVelocity','Const. Vel.'),('MLP','MLP'),('LSTM_only','LSTM'),
        ('Seq2Seq_Attn','Seq2Seq+Attn'),('LSTM_Env_Goal','LSTM+Env+Goal'),
        ('V3_Waypoint','V3: +Waypoints'),('V4_WP_Spatial','V4: +Spatial'),
        ('TerraTNT','TerraTNT'),('V6_Autoreg','V6: +Autoreg'),
        ('V6R_Robust','V6R: +GoalDrop'),('V7_ConfGate','V7: ConfGate'),
    ]
    # Find best per phase
    best = {}
    for p in phases:
        bv = float('inf')
        for k,_ in models:
            if p in R and k in R[p]['models']:
                v = R[p]['models'][k]['ade_mean']
                if v < bv: bv = v
        best[p] = bv

    print('\n% LaTeX Table')
    print(r'\begin{tabular}{l' + 'r'*len(phases) + '}')
    print(r'\toprule')
    print('Model & ' + ' & '.join(phases) + r' \\')
    print(r'\midrule')
    for k, lb in models:
        row = lb
        for p in phases:
            if p in R and k in R[p]['models']:
                v = R[p]['models'][k]['ade_mean']/1000
                is_best = abs(R[p]['models'][k]['ade_mean'] - best[p]) < 1
                if is_best:
                    row += f' & \\textbf{{{v:.1f}}}'
                else:
                    row += f' & {v:.1f}'
            else:
                row += ' & --'
        row += r' \\'
        print(row)
    print(r'\bottomrule')
    print(r'\end{tabular}')


def fig5_cross_region(R_bh, R_sh):
    """Cross-region comparison: Bohemian Forest vs Scottish Highlands."""
    models = [
        ('V6R_Robust','V6R','#1565C0'),
        ('V6_Autoreg','V6','#42A5F5'),
        ('V3_Waypoint','V3','#FF9800'),
        ('LSTM_Env_Goal','LSTM+EG','#FFA726'),
        ('V4_WP_Spatial','V4','#90CAF9'),
        ('TerraTNT','TerraTNT','#4CAF50'),
        ('Seq2Seq_Attn','Seq2Seq','#9E9E9E'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ph, ti in zip(axes, ['P1a','P3a'], ['P1a: Precise Goal','P3a: No Prior']):
        labels, bh_vals, sh_vals = [], [], []
        for k, lb, _ in models:
            bh_v = R_bh.get(ph,{}).get('models',{}).get(k,{}).get('ade_mean', np.nan)
            sh_v = R_sh.get(ph,{}).get('models',{}).get(k,{}).get('ade_mean', np.nan)
            if not np.isnan(bh_v) and not np.isnan(sh_v):
                labels.append(lb); bh_vals.append(bh_v/1000); sh_vals.append(sh_v/1000)
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w/2, bh_vals, w, label='Bohemian Forest (train)', color='#2196F3', alpha=0.8)
        ax.bar(x + w/2, sh_vals, w, label='Scottish Highlands (test)', color='#FF5722', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_ylabel('ADE (km)'); ax.set_title(ti, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGS/'fig5_cross_region.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS/'fig5_cross_region.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print('  fig5 done')


def fig6_v5_legacy(v5_data):
    """Legacy Phase system (v5) results: FAS1-FAS4 with GT goal for LSTM_EG."""
    phases_order = ['fas1','fas2','fas3','fas3b_gaussian','fas4']
    phase_labels = ['FAS1\nGT Goal','FAS2\nOOD','FAS3\nNo GT\ncand','FAS3b\nGauss\nσ=10km','FAS4\nNo prior\nr=40km']
    models = [
        ('V6R_Robust','V6R','#1565C0','-','o'),
        ('V6_Autoreg','V6','#42A5F5','--','s'),
        ('V4_WP_Spatial','V4','#90CAF9','-.','D'),
        ('LSTM_Env_Goal','LSTM+EG','#FF9800','-','^'),
        ('TerraTNT','TerraTNT','#4CAF50','-','v'),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(phases_order))
    for k, lb, c, ls, mk in models:
        vals = []
        for ph_data in v5_data:
            phase = ph_data['phase']
            if phase in phases_order:
                v = ph_data['global'].get(k,{}).get('ade_m', np.nan)
                vals.append((phases_order.index(phase), v/1000))
        vals.sort()
        if vals:
            xs, ys = zip(*vals)
            ax.plot(list(xs), list(ys), label=lb, color=c, ls=ls, marker=mk, ms=8, lw=2)
    ax.set_xticks(x); ax.set_xticklabels(phase_labels, fontsize=9)
    ax.set_ylabel('ADE (km)'); ax.set_xlabel('Phase (decreasing prior quality →)')
    ax.set_title('Legacy Phase System: Performance with GT Goal Input', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGS/'fig6_v5_legacy.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS/'fig6_v5_legacy.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print('  fig6 done')


if __name__ == '__main__':
    print('Loading results...')
    R = load()
    print('Generating figures...')
    fig1_bars(R)
    fig2_lines(R)
    fig3_ablation(R)
    fig4_temporal(R)
    print_latex_table(R)

    # Cross-region
    sh_path = ROOT / 'outputs' / 'evaluation' / 'cross_region_sh' / 'phase_v2_results.json'
    if sh_path.exists():
        with open(sh_path) as f:
            R_sh = json.load(f)
        fig5_cross_region(R, R_sh)

    # Legacy v5 results
    v5_path = ROOT / 'outputs' / 'evaluation' / 'phase_diagnostic_v5' / 'summary_all_phases.json'
    if v5_path.exists():
        with open(v5_path) as f:
            v5_data = json.load(f)
        fig6_v5_legacy(v5_data)

    print(f'\nAll figures saved to: {FIGS}')
