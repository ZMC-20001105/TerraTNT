"""
Modern UI Mockup Generator for TerraTNT
Design Principles: Dark Theme, Card-based, Dashboard Layout, Professional Accents
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, PathPatch
from matplotlib.font_manager import FontProperties
import matplotlib.path as mpath
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
    'accent_dark': '#0EA5E9',
    'text': '#F1F5F9',       # Off-white
    'text_dim': '#94A3B8',   # Muted gray-blue
    'success': '#10B981',    # Emerald
    'warning': '#F59E0B',    # Amber
    'danger': '#EF4444',     # Rose
    'border': '#334155'      # Subtle border
}

def setup_canvas():
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=COLORS['bg'])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    return fig, ax

def draw_sidebar(ax, active_tab):
    # Sidebar background
    sidebar = Rectangle((0, 0), 2.5, 10, facecolor=COLORS['sidebar'], zorder=1)
    ax.add_patch(sidebar)
    
    # Logo
    ax.text(1.25, 9.5, 'TerraTNT', ha='center', va='center', 
            fontsize=18, fontweight='bold', color=COLORS['accent'], zorder=2)
    ax.text(1.25, 9.2, 'Trajectory Intelligence', ha='center', va='center', 
            fontsize=8, color=COLORS['text_dim'], zorder=2)
    
    tabs = [
        ('Overview', '概览'),
        ('Datasets', '数据集'),
        ('Training', '模型训练'),
        ('Prediction', '轨迹预测'),
        ('Evaluation', '模型评估'),
        ('Settings', '系统设置')
    ]
    
    for i, (en, cn) in enumerate(tabs):
        y = 8.2 - i * 0.7
        is_active = (cn == active_tab)
        
        if is_active:
            # Active indicator
            bg = Rectangle((0.2, y - 0.25), 2.1, 0.5, facecolor=COLORS['accent'], alpha=0.1, zorder=2)
            ax.add_patch(bg)
            border = Rectangle((0, y - 0.25), 0.1, 0.5, facecolor=COLORS['accent'], zorder=3)
            ax.add_patch(border)
            text_color = COLORS['accent']
        else:
            text_color = COLORS['text_dim']
            
        ax.text(0.4, y, f'{cn}', va='center', fontsize=10, 
               color=text_color, fontproperties=cn_font, zorder=3)
        ax.text(0.4, y - 0.15, f'{en}', va='center', fontsize=7, 
               color=text_color, alpha=0.7, zorder=3)

def draw_header(ax, title):
    # Header bar
    ax.text(2.8, 9.5, title, fontsize=16, fontweight='bold', color=COLORS['text'], 
            fontproperties=cn_font, zorder=2)
    
    # Search bar mockup
    search_bg = FancyBboxPatch((11, 9.3), 3.5, 0.4, boxstyle="round,pad=0.05",
                              facecolor=COLORS['card'], edgecolor=COLORS['border'], zorder=2)
    ax.add_patch(search_bg)
    ax.text(11.2, 9.5, '🔍 Search...', color=COLORS['text_dim'], fontsize=9, va='center', zorder=3)
    
    # User profile
    user_circle = Circle((15.2, 9.5), 0.2, facecolor=COLORS['accent'], zorder=2)
    ax.add_patch(user_circle)
    ax.text(15.2, 9.5, 'U', ha='center', va='center', color=COLORS['bg'], fontweight='bold', zorder=3)

def create_card(ax, x, y, w, h, title="", zorder=2):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=COLORS['card'], edgecolor=COLORS['border'], 
                          linewidth=1, zorder=zorder)
    ax.add_patch(card)
    if title:
        ax.text(x + 0.1, y + h - 0.15, title, fontsize=11, fontweight='bold', 
                color=COLORS['text'], fontproperties=cn_font, zorder=zorder+1)
    return card

def draw_overview():
    fig, ax = setup_canvas()
    draw_sidebar(ax, '概览')
    draw_header(ax, '系统概览 / Overview')
    
    # 1. Status Cards
    card_w = 3.0
    titles = ['Generated Tracks', 'Models Ready', 'GPU Status', 'Uptime']
    values = ['14,400', '7 Active', '92% Free', '42h 15m']
    icons = ['📈', '🧠', '⚙️', '🕒']
    
    for i in range(4):
        x = 2.8 + i * 3.3
        create_card(ax, x, 7.8, card_w, 1.2)
        ax.text(x + 0.2, 8.6, icons[i], fontsize=14, zorder=4)
        ax.text(x + 0.2, 8.2, titles[i], color=COLORS['text_dim'], fontsize=9, zorder=4)
        ax.text(x + 0.2, 7.95, values[i], color=COLORS['text'], fontsize=14, fontweight='bold', zorder=4)

    # 2. Main Chart Card
    create_card(ax, 2.8, 3.5, 8.0, 4.0, title="Trajectory Generation Progress")
    # Draw simple area chart
    x_chart = np.linspace(3.0, 10.5, 20)
    y_chart = 3.8 + 2.5 * (1 - np.exp(-(x_chart-3.0)/2))
    ax.fill_between(x_chart, 3.8, y_chart, color=COLORS['accent'], alpha=0.2, zorder=3)
    ax.plot(x_chart, y_chart, color=COLORS['accent'], linewidth=2, zorder=4)
    ax.text(3.0, 3.7, 'Time (h)', color=COLORS['text_dim'], fontsize=8)
    
    # 3. Recent Activity Card
    create_card(ax, 11.1, 0.5, 4.4, 7.0, title="Recent Activity")
    activities = [
        ('✓', 'Generation Complete', 'Scottish Highlands', '10m ago'),
        ('⚙', 'Training Epoch 45', 'TerraTNT Main', '25m ago'),
        ('✓', 'Preprocessing Done', 'Bohemian Forest', '1h ago'),
        ('⚠️', 'Resource Warning', 'Memory usage 85%', '2h ago'),
        ('⚙', 'Baseline Sync', 'YNet, PECNet', '3h ago'),
        ('✓', 'Data Validated', 'All 14.4k records', '5h ago')
    ]
    for i, (icon, msg, sub, time) in enumerate(activities):
        y = 6.8 - i * 1.0
        color = COLORS['success'] if icon == '✓' else COLORS['accent'] if icon == '⚙' else COLORS['warning']
        ax.text(11.3, y, icon, color=color, fontweight='bold', fontsize=10, zorder=4)
        ax.text(11.6, y, msg, color=COLORS['text'], fontsize=9, zorder=4)
        ax.text(11.6, y - 0.25, sub, color=COLORS['text_dim'], fontsize=8, zorder=4)
        ax.text(15.3, y, time, ha='right', color=COLORS['text_dim'], fontsize=7, zorder=4)

    # 4. Quick Actions Card
    create_card(ax, 2.8, 0.5, 8.0, 2.7, title="Quick Actions")
    actions = [
        ('Start Gen', COLORS['accent']),
        ('Retrain', COLORS['success']),
        ('Export PDF', COLORS['card']),
        ('System Log', COLORS['card'])
    ]
    for i, (name, color) in enumerate(actions):
        x = 3.2 + i * 1.8
        btn = FancyBboxPatch((x, 1.2), 1.6, 1.2, boxstyle="round,pad=0.05",
                            facecolor=color if color != COLORS['card'] else COLORS['bg'],
                            edgecolor=COLORS['border'] if color == COLORS['card'] else color,
                            zorder=4)
        ax.add_patch(btn)
        ax.text(x + 0.8, 1.8, name, ha='center', va='center', 
                color=COLORS['text'] if color == COLORS['card'] else COLORS['bg'], 
                fontsize=9, fontweight='bold', zorder=5)

    plt.tight_layout()
    return fig

def draw_dataset():
    fig, ax = setup_canvas()
    draw_sidebar(ax, '数据集')
    draw_header(ax, '数据集管理 / Dataset Management')
    
    # 1. Dataset List Card (Left)
    create_card(ax, 2.8, 0.5, 4.0, 8.5, title="Regions")
    regions = [
        ('Scottish Highlands', '3,600 tracks', 'Complete', COLORS['success']),
        ('Bohemian Forest', '3,600 tracks', 'Complete', COLORS['success']),
        ('Donbas', '1,752/3,600', 'Processing', COLORS['accent']),
        ('Carpathians', '0/3,600', 'Pending', COLORS['text_dim'])
    ]
    for i, (name, count, status, color) in enumerate(regions):
        y = 7.8 - i * 1.8
        is_sel = (i == 0)
        bg_col = COLORS['bg'] if not is_sel else COLORS['accent']
        rect = FancyBboxPatch((3.0, y - 0.6), 3.6, 1.3, boxstyle="round,pad=0.05",
                             facecolor=bg_col, alpha=0.1 if is_sel else 0.5, zorder=3)
        ax.add_patch(rect)
        ax.text(3.2, y + 0.3, name, color=COLORS['text'], fontsize=10, fontweight='bold', zorder=4)
        ax.text(3.2, y, count, color=COLORS['text_dim'], fontsize=8, zorder=4)
        ax.text(3.2, y - 0.3, f'● {status}', color=color, fontsize=8, zorder=4)

    # 2. Statistics Card (Top Right)
    create_card(ax, 7.0, 4.5, 8.5, 4.5, title="Distribution: Scottish Highlands")
    # Mock Histogram
    x_hist = np.linspace(7.5, 15, 15)
    h_hist = [0.2, 0.5, 1.2, 2.5, 3.2, 2.8, 2.1, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1]
    for x, h in zip(x_hist, h_hist):
        rect = Rectangle((x, 5.0), 0.4, h, facecolor=COLORS['accent'], alpha=0.8, zorder=4)
        ax.add_patch(rect)
    ax.text(11.25, 4.8, 'Trajectory Length (km)', color=COLORS['text_dim'], ha='center', fontsize=8)

    # 3. Details Card (Bottom Right)
    create_card(ax, 7.0, 0.5, 8.5, 3.7, title="Selected Data Sample")
    cols = ['ID', 'Vehicle', 'Intent', 'Length', 'Duration']
    row_data = [
        ['TR_001', 'Type 1', 'Goal-oriented', '125.4 km', '350 min'],
        ['TR_002', 'Type 4', 'Tactical', '88.2 km', '210 min'],
        ['TR_003', 'Type 2', 'Covert', '142.1 km', '420 min']
    ]
    for i, col in enumerate(cols):
        ax.text(7.5 + i*1.7, 3.4, col, color=COLORS['accent'], fontsize=9, fontweight='bold', zorder=4)
    
    for r, data in enumerate(row_data):
        y = 2.9 - r * 0.6
        for c, val in enumerate(data):
            ax.text(7.5 + c*1.7, y, val, color=COLORS['text'], fontsize=8, zorder=4)
            
    plt.tight_layout()
    return fig

def draw_prediction():
    fig, ax = setup_canvas()
    draw_sidebar(ax, '轨迹预测')
    draw_header(ax, '实时轨迹预测 / Real-time Prediction')
    
    # 1. Map Panel (Main)
    create_card(ax, 2.8, 0.5, 10.0, 8.5, title="Environmental View")
    # Draw complex looking map grid
    for i in range(12):
        for j in range(10):
            val = (np.sin(i/2) * np.cos(j/2) + 1) / 2
            rect = Rectangle((3.2 + i*0.8, 1.0 + j*0.7), 0.75, 0.65, 
                            facecolor=plt.cm.magma(val), alpha=0.2, zorder=3)
            ax.add_patch(rect)
    
    # Draw a sophisticated path
    path_data = [
        (4, 2), (5, 3), (6, 3.5), (7, 4), (8, 4.2), # Past
        (9, 5), (10, 6), (11, 7.5), (12, 8.5)      # Future
    ]
    px, py = zip(*[(3.2 + x*0.8, 1.0 + y*0.7) for x, y in path_data])
    ax.plot(px[:5], py[:5], color='#60A5FA', linewidth=3, marker='o', markersize=4, label='History', zorder=5)
    ax.plot(px[4:], py[4:], color='#F87171', linewidth=3, linestyle='--', marker='o', markersize=4, label='Predicted', zorder=5)
    
    # Add Heatmap points (Candidate goals)
    goals = [(11.5, 8.2), (12.2, 7.8), (11.8, 8.5)]
    for gx, gy in goals:
        circle = Circle((3.2 + gx*0.8, 1.0 + gy*0.7), 0.3, color=COLORS['warning'], alpha=0.4, zorder=4)
        ax.add_patch(circle)
        
    # Legend
    leg_box = FancyBboxPatch((10.0, 1.2), 2.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor=COLORS['sidebar'], alpha=0.8, zorder=6)
    ax.add_patch(leg_box)
    ax.text(10.2, 2.4, '● History', color='#60A5FA', fontsize=9, zorder=7)
    ax.text(10.2, 2.1, '● Predicted', color='#F87171', fontsize=9, zorder=7)
    ax.text(10.2, 1.8, '★ Goal Cand.', color=COLORS['warning'], fontsize=9, zorder=7)

    # 2. Control Panel (Right)
    create_card(ax, 13.0, 0.5, 2.5, 8.5, title="Controls")
    controls = ['Model: TerraTNT', 'History: 10m', 'Future: 60m', 'Intent: Covert']
    for i, ctrl in enumerate(controls):
        y = 7.5 - i * 1.0
        create_card(ax, 13.2, y - 0.4, 2.1, 0.6, zorder=3)
        ax.text(13.4, y, ctrl, color=COLORS['text'], fontsize=8, zorder=4)
        
    btn = FancyBboxPatch((13.2, 1.0), 2.1, 0.8, boxstyle="round,pad=0.05",
                        facecolor=COLORS['accent'], zorder=4)
    ax.add_patch(btn)
    ax.text(14.25, 1.4, 'RUN INFERENCE', ha='center', va='center', 
            color=COLORS['bg'], fontweight='bold', fontsize=9, zorder=5)

    plt.tight_layout()
    return fig

def main():
    print("Generating Modern UI Mockups...")
    
    pages = [
        ('Overview', draw_overview),
        ('Datasets', draw_dataset),
        ('Prediction', draw_prediction)
    ]
    
    for i, (name, draw_func) in enumerate(pages, 1):
        print(f"  Generating {name}...")
        fig = draw_func()
        filename = f'/home/zmc/文档/programwork/docs/ui_modern_{i}_{name}.png'
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        plt.close(fig)
        print(f"  ✓ Saved: {filename}")
    
    print("\n✅ Modern UI Mockups Generated!")

if __name__ == '__main__':
    main()
