"""
Refined Functional UI Mockup for TerraTNT
Based on UX_DESIGN_LOGIC.md: Environment, Dataset, Model, Inference, Analytics
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.font_manager import FontProperties
import numpy as np

# Load Chinese font
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
cn_font = FontProperties(fname=font_path)

# Modern Dark Theme Palette
COLORS = {
    'bg': '#0F172A',         # Deep dark blue-gray
    'card': '#1E293B',       # Dark blue-gray
    'sidebar': '#020617',    # Near black
    'accent': '#38BDF8',     # Bright cyan
    'text': '#F1F5F9',       # Off-white
    'text_dim': '#94A3B8',   # Muted gray-blue
    'success': '#10B981',    # Emerald
    'warning': '#F59E0B',    # Amber
    'border': '#334155'      # Subtle border
}

def setup_canvas():
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['bg'])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    return fig, ax

def draw_sidebar(ax, active_module):
    # Sidebar background
    sidebar = Rectangle((0, 0), 2.8, 10, facecolor=COLORS['sidebar'], zorder=1)
    ax.add_patch(sidebar)
    
    # Logo
    ax.text(1.4, 9.4, 'TerraTNT v2.0', ha='center', va='center', 
            fontsize=16, fontweight='bold', color=COLORS['accent'], zorder=2)
    
    modules = [
        ('Environment', '环境工作站'),
        ('Dataset', '数据集工厂'),
        ('Model', '模型实验室'),
        ('Inference', '智能预测中心'),
        ('Analytics', '实验评估看板'),
        ('Settings', '系统设置')
    ]
    
    for i, (en, cn) in enumerate(modules):
        y = 8.0 - i * 0.8
        is_active = (cn == active_module)
        
        if is_active:
            bg = Rectangle((0.2, y - 0.3), 2.4, 0.6, facecolor=COLORS['accent'], alpha=0.1, zorder=2)
            ax.add_patch(bg)
            border = Rectangle((0, y - 0.3), 0.1, 0.6, facecolor=COLORS['accent'], zorder=3)
            ax.add_patch(border)
            text_color = COLORS['accent']
        else:
            text_color = COLORS['text_dim']
            
        ax.text(0.5, y + 0.1, f'{cn}', va='center', fontsize=10, 
               color=text_color, fontproperties=cn_font, zorder=3)
        ax.text(0.5, y - 0.15, f'{en} Module', va='center', fontsize=7, 
               color=text_color, alpha=0.7, zorder=3)

def create_card(ax, x, y, w, h, title="", zorder=2):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=COLORS['card'], edgecolor=COLORS['border'], 
                          linewidth=1, zorder=zorder)
    ax.add_patch(card)
    if title:
        ax.text(x + 0.1, y + h - 0.2, title, fontsize=11, fontweight='bold', 
                color=COLORS['text'], fontproperties=cn_font, zorder=zorder+1)
    return card

def draw_inference_module():
    """Module D: Inference Center - The most interactive part"""
    fig, ax = setup_canvas()
    draw_sidebar(ax, '智能预测中心')
    
    # Title
    ax.text(3.2, 9.4, '智能预测中心 / Inference Center', fontsize=18, fontweight='bold', 
            color=COLORS['text'], fontproperties=cn_font, zorder=2)
    
    # 1. Main Map (The central stage)
    create_card(ax, 3.2, 0.5, 9.0, 8.5, title="交互式地图预览 / Interactive Map Viewer")
    # Draw a stylized map
    for i in range(12):
        for j in range(10):
            val = (np.sin(i/3) * np.cos(j/3) + 1) / 2
            rect = Rectangle((3.5 + i*0.7, 1.0 + j*0.7), 0.65, 0.65, 
                            facecolor=plt.cm.viridis(val), alpha=0.15, zorder=3)
            ax.add_patch(rect)
    
    # History Path (Drawn by user)
    hist_x = [4, 5, 6, 7]
    hist_y = [2, 3, 3.5, 4.5]
    px, py = zip(*[(3.5 + x*0.7, 1.0 + y*0.7) for x, y in zip(hist_x, hist_y)])
    ax.plot(px, py, color='#60A5FA', linewidth=3, marker='o', markersize=5, label='User Input', zorder=5)
    ax.text(px[0], py[0]-0.2, 'Start', color=COLORS['text'], fontsize=8, ha='center', zorder=5)
    
    # Prediction Results (Multiple models)
    # TerraTNT (Cyan)
    pred_x = [7, 8, 9, 10]
    pred_y = [4.5, 5.5, 6.5, 8]
    ppx, ppy = zip(*[(3.5 + x*0.7, 1.0 + y*0.7) for x, y in zip(pred_x, pred_y)])
    ax.plot(ppx, ppy, color=COLORS['accent'], linewidth=4, alpha=0.9, zorder=6)
    
    # YNet (Amber)
    ax.plot(ppx, [y-0.3 for y in ppy], color=COLORS['warning'], linewidth=2, linestyle='--', alpha=0.6, zorder=6)
    
    # Goal Heatmap
    circle = Circle((ppx[-1], ppy[-1]), 0.6, color=COLORS['accent'], alpha=0.2, zorder=4)
    ax.add_patch(circle)

    # 2. Control Console (Right Sidebar)
    create_card(ax, 12.5, 4.5, 3.2, 4.5, title="预测控制面板 / Controls")
    
    controls = [
        ('模型选择', 'TerraTNT (Primary)'),
        ('基线对比', 'YNet, PECNet'),
        ('历史跨度', '10 Minutes'),
        ('预测跨度', '60 Minutes')
    ]
    for i, (label, val) in enumerate(controls):
        y = 7.5 - i * 0.9
        ax.text(12.7, y, label, color=COLORS['text_dim'], fontsize=9, fontproperties=cn_font, zorder=4)
        input_bg = FancyBboxPatch((12.7, y-0.5), 2.8, 0.4, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['bg'], edgecolor=COLORS['border'], zorder=4)
        ax.add_patch(input_bg)
        ax.text(12.8, y-0.3, val, color=COLORS['text'], fontsize=8, zorder=5)

    # Run Button
    btn = FancyBboxPatch((12.7, 4.8), 2.8, 0.7, boxstyle="round,pad=0.05",
                        facecolor=COLORS['accent'], zorder=4)
    ax.add_patch(btn)
    ax.text(14.1, 5.15, '执行预测 / RUN INFERENCE', ha='center', va='center', 
            color=COLORS['bg'], fontweight='bold', fontsize=9, fontproperties=cn_font, zorder=5)

    # 3. Metrics Panel (Bottom Right)
    create_card(ax, 12.5, 0.5, 3.2, 3.7, title="性能指标 / Metrics")
    metrics = [
        ('ADE', '12.4m', '-2.1m'),
        ('FDE', '35.8m', '-5.2m'),
        ('Goal Acc', '82.4%', '+4.5%')
    ]
    for i, (name, val, diff) in enumerate(metrics):
        y = 2.8 - i * 0.8
        ax.text(12.7, y, name, color=COLORS['text_dim'], fontsize=9, fontproperties=cn_font, zorder=4)
        ax.text(14.0, y, val, color=COLORS['text'], fontsize=11, fontweight='bold', zorder=4)
        ax.text(15.2, y, diff, color=COLORS['success'], fontsize=8, ha='right', zorder=4)

    plt.tight_layout()
    return fig

def draw_dataset_factory():
    """Module B: Dataset Factory - Task management focus"""
    fig, ax = setup_canvas()
    draw_sidebar(ax, '数据集工厂')
    
    # Title
    ax.text(3.2, 9.4, '数据集工厂 / Dataset Factory', fontsize=18, fontweight='bold', 
            color=COLORS['text'], fontproperties=cn_font, zorder=2)
    
    # 1. Task Queue
    create_card(ax, 3.2, 5.0, 12.3, 4.0, title="任务队列 / Task Queue")
    headers = ['任务 ID', '区域', '状态', '进度', '预计完成']
    x_cols = [3.5, 5.5, 7.5, 9.5, 13.0]
    for x, h in zip(x_cols, headers):
        ax.text(x, 8.5, h, color=COLORS['accent'], fontsize=10, fontweight='bold', fontproperties=cn_font, zorder=4)
    
    tasks = [
        ['TASK_08', 'Scottish', 'Running', '50%', '35m'],
        ['TASK_09', 'Bohemian', 'Running', '34%', '55m'],
        ['TASK_07', 'Donbas', 'Completed', '100%', '-'],
    ]
    for i, row in enumerate(tasks):
        y = 7.8 - i * 0.8
        color = COLORS['success'] if row[2] == 'Completed' else COLORS['text']
        for x, val in zip(x_cols, row):
            ax.text(x, y, val, color=color, fontsize=9, zorder=4)
        # Simple progress bar for running tasks
        if row[2] == 'Running':
            prog_bg = Rectangle((9.5, y-0.1), 3.0, 0.15, facecolor=COLORS['bg'], zorder=4)
            ax.add_patch(prog_bg)
            w = 1.5 if row[3] == '50%' else 1.02
            prog_fg = Rectangle((9.5, y-0.1), w, 0.15, facecolor=COLORS['accent'], zorder=5)
            ax.add_patch(prog_fg)

    # 2. Quality Analysis
    create_card(ax, 3.2, 0.5, 6.0, 4.2, title="数据质量分析 / Quality QA")
    # Mock data distribution
    x_dist = np.linspace(3.5, 8.8, 10)
    h_dist = [0.2, 0.5, 1.5, 2.5, 3.0, 2.0, 1.5, 0.8, 0.4, 0.2]
    for x, h in zip(x_dist, h_dist):
        ax.add_patch(Rectangle((x, 1.2), 0.4, h/1.5, facecolor=COLORS['success'], alpha=0.7, zorder=4))
    ax.text(6.2, 0.8, '轨迹连贯性评分 (Consistency Score)', color=COLORS['text_dim'], fontsize=8, ha='center', fontproperties=cn_font)

    # 3. Storage Info
    create_card(ax, 9.5, 0.5, 6.0, 4.2, title="存储与配额 / Storage")
    labels = ['Scottish Highlands', 'Bohemian Forest', 'Donbas Tiles', 'System Logs']
    sizes = [45, 32, 120, 5] # GB
    total = sum(sizes)
    start_x = 10.0
    for i, s in enumerate(sizes):
        w = (s/total) * 5.0
        ax.add_patch(Rectangle((start_x, 2.5), w, 0.6, facecolor=plt.cm.Blues(0.3 + i*0.2), zorder=4))
        ax.text(start_x, 2.3, f'{labels[i][:5]}..', color=COLORS['text_dim'], fontsize=7)
        start_x += w
    ax.text(10.0, 3.4, f'已用存储: {total} GB / 500 GB', color=COLORS['text'], fontsize=10, fontweight='bold', fontproperties=cn_font, zorder=4)

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("Generating Refined Functional Mockups...")
    
    # 1. Dataset Factory (Module B)
    fig_b = draw_dataset_factory()
    fig_b.savefig('/home/zmc/文档/programwork/docs/ui_module_dataset.png', dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig_b)
    
    # 2. Inference Center (Module D)
    fig_d = draw_inference_module()
    fig_d.savefig('/home/zmc/文档/programwork/docs/ui_module_inference.png', dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig_d)
    
    print("✅ Refined Mockups Generated: ui_module_dataset.png, ui_module_inference.png")
