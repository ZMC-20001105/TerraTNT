"""
系统架构图生成脚本 - 修复版
明确指定中文字体路径
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
from pathlib import Path

# 查找并设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    # 尝试查找系统中的中文字体
    chinese_fonts = [
        'Noto Sans CJK SC',
        'Noto Sans CJK TC', 
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Droid Sans Fallback',
        'AR PL UMing CN',
        'AR PL UKai CN'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 使用字体: {font}")
            return font
    
    # 如果没有找到，使用默认并警告
    print("⚠ 未找到中文字体，文本可能无法正常显示")
    return None

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'data': '#E3F2FD',
        'process': '#FFF3E0',
        'model': '#F3E5F5',
        'app': '#E8F5E9',
        'border_data': '#1976D2',
        'border_process': '#F57C00',
        'border_model': '#7B1FA2',
        'border_app': '#388E3C'
    }
    
    # 标题
    ax.text(8, 11.5, 'TerraTNT: Environment-Constrained Ground Target Trajectory Prediction', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # ============ 第1层：数据层 ============
    y_data = 9.5
    
    ax.text(0.5, y_data + 0.8, 'Data Layer', 
            fontsize=14, fontweight='bold', color=colors['border_data'])
    
    data_boxes = [
        {'x': 0.5, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'DEM Data\n(SRTM 30m)'},
        {'x': 3.2, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'LULC Data\n(ESA WorldCover)'},
        {'x': 5.9, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'OSM Roads\n(6 Countries)'},
        {'x': 8.6, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'OORD Tracks\n(Real Data)'},
        {'x': 11.3, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'Synthetic\n(14,400 tracks)'},
    ]
    
    for box in data_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['data'],
                              edgecolor=colors['border_data'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=9)
    
    # ============ 第2层：处理层 ============
    y_process = 7.5
    
    ax.text(0.5, y_process + 0.8, 'Processing Layer', 
            fontsize=14, fontweight='bold', color=colors['border_process'])
    
    process_boxes = [
        {'x': 0.5, 'y': y_process, 'w': 3.5, 'h': 0.6, 
         'text': 'Env Preprocessing\n• Projection (UTM)\n• Terrain Features'},
        {'x': 4.2, 'y': y_process, 'w': 3.5, 'h': 0.6, 
         'text': 'Cost Map Generation\n• Passable Analysis\n• Multi-intent Costs'},
        {'x': 8.0, 'y': y_process, 'w': 3.5, 'h': 0.6, 
         'text': 'Trajectory Generation\n• Hierarchical A*\n• XGBoost Speed'},
        {'x': 11.8, 'y': y_process, 'w': 3.5, 'h': 0.6, 
         'text': 'Data Augmentation\n• 18-channel Maps\n• Train/Val/Test Split'},
    ]
    
    for box in process_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['process'],
                              edgecolor=colors['border_process'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=8)
    
    # ============ 第3层：模型层 ============
    y_model = 5.0
    
    ax.text(0.5, y_model + 0.8, 'Model Layer', 
            fontsize=14, fontweight='bold', color=colors['border_model'])
    
    model_boxes = [
        {'x': 1.0, 'y': y_model, 'w': 3.0, 'h': 0.6, 'text': 'CNN Env Encoder\n(ResNet-18)'},
        {'x': 4.5, 'y': y_model, 'w': 3.0, 'h': 0.6, 'text': 'LSTM History\n(2 Layers)'},
        {'x': 8.0, 'y': y_model, 'w': 3.0, 'h': 0.6, 'text': 'Goal Classifier\n(Candidates)'},
        {'x': 11.5, 'y': y_model, 'w': 3.0, 'h': 0.6, 'text': 'LSTM Decoder\n(Hierarchical)'},
    ]
    
    for box in model_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['model'],
                              edgecolor=colors['border_model'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=9)
    
    # 训练框架
    train_box = FancyBboxPatch((1.0, y_model - 1.2), 13.5, 0.8,
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['model'],
                               edgecolor=colors['border_model'], linewidth=2, linestyle='--')
    ax.add_patch(train_box)
    ax.text(7.75, y_model - 0.8, 
            'Training: PyTorch | Adam | Loss: NLL+ADE+FDE | Early Stop | TensorBoard', 
            ha='center', va='center', fontsize=9, style='italic')
    
    # ============ 第4层：应用层 ============
    y_app = 2.0
    
    ax.text(0.5, y_app + 0.8, 'Application Layer', 
            fontsize=14, fontweight='bold', color=colors['border_app'])
    
    app_boxes = [
        {'x': 1.5, 'y': y_app, 'w': 3.5, 'h': 0.6, 
         'text': 'Prediction Service\n• Real-time\n• Batch'},
        {'x': 5.5, 'y': y_app, 'w': 3.5, 'h': 0.6, 
         'text': 'Visualization\n• Map Display\n• Comparison'},
        {'x': 9.5, 'y': y_app, 'w': 3.5, 'h': 0.6, 
         'text': 'Evaluation\n• ADE/FDE\n• Ablation'},
    ]
    
    for box in app_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['app'],
                              edgecolor=colors['border_app'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=9)
    
    # ============ 第5层：用户界面 ============
    y_ui = 0.3
    
    ui_box = FancyBboxPatch((2.0, y_ui), 11.0, 0.5,
                            boxstyle="round,pad=0.05", 
                            facecolor='#FFECB3',
                            edgecolor='#FF6F00', linewidth=2)
    ax.add_patch(ui_box)
    ax.text(7.5, y_ui + 0.25, 
            'User Interface: Web Dashboard | REST API | CLI Tools', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ============ 绘制箭头 ============
    arrow_style = "Simple,tail_width=0.5,head_width=8,head_length=8"
    
    # 数据层 -> 处理层
    for x in [2.25, 4.45, 7.15, 9.85, 12.55]:
        arrow = FancyArrowPatch((x, y_data - 0.1), (x, y_process + 0.7),
                               arrowstyle=arrow_style, color=colors['border_process'],
                               linewidth=1.5, alpha=0.6)
        ax.add_patch(arrow)
    
    # 处理层 -> 模型层
    for x in [2.25, 5.95, 9.75, 13.55]:
        arrow = FancyArrowPatch((x, y_process - 0.1), (x, y_model + 0.7),
                               arrowstyle=arrow_style, color=colors['border_model'],
                               linewidth=1.5, alpha=0.6)
        ax.add_patch(arrow)
    
    # 模型层 -> 应用层
    for x in [3.25, 7.25, 11.25]:
        arrow = FancyArrowPatch((x, y_model - 1.3), (x, y_app + 0.7),
                               arrowstyle=arrow_style, color=colors['border_app'],
                               linewidth=1.5, alpha=0.6)
        ax.add_patch(arrow)
    
    # 应用层 -> 用户界面
    for x in [3.25, 7.25, 11.25]:
        arrow = FancyArrowPatch((x, y_app - 0.1), (x, y_ui + 0.6),
                               arrowstyle=arrow_style, color='#FF6F00',
                               linewidth=1.5, alpha=0.6)
        ax.add_patch(arrow)
    
    # ============ 图例 ============
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor=colors['border_data'], 
                      label='Data Layer'),
        mpatches.Patch(facecolor=colors['process'], edgecolor=colors['border_process'], 
                      label='Processing Layer'),
        mpatches.Patch(facecolor=colors['model'], edgecolor=colors['border_model'], 
                      label='Model Layer'),
        mpatches.Patch(facecolor=colors['app'], edgecolor=colors['border_app'], 
                      label='Application Layer'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
             framealpha=0.9, edgecolor='black')
    
    # ============ 系统特性 ============
    ax.text(15.5, 10.5, 'Features', fontsize=11, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    features = [
        '✓ Multi-region',
        '✓ Parallel',
        '✓ GPU Accelerated',
        '✓ Real-time',
        '✓ Scalable'
    ]
    for i, feature in enumerate(features):
        ax.text(15.5, 10.0 - i*0.4, feature, fontsize=8)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # 设置字体
    font_name = setup_chinese_font()
    
    # 绘制架构图
    fig = draw_system_architecture()
    
    # 保存为PNG
    output_path = '/home/zmc/文档/programwork/docs/system_architecture.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 系统架构图已保存: {output_path}")
    
    # 保存为PDF
    pdf_path = '/home/zmc/文档/programwork/docs/system_architecture.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF版本已保存: {pdf_path}")
    
    plt.close()
