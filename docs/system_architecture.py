"""
系统架构图生成脚本
使用matplotlib绘制多层系统架构图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'data': '#E3F2FD',      # 浅蓝
        'process': '#FFF3E0',   # 浅橙
        'model': '#F3E5F5',     # 浅紫
        'app': '#E8F5E9',       # 浅绿
        'border_data': '#1976D2',
        'border_process': '#F57C00',
        'border_model': '#7B1FA2',
        'border_app': '#388E3C'
    }
    
    # ============ 标题 ============
    ax.text(8, 11.5, 'TerraTNT: 基于环境约束的地面目标轨迹预测系统', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # ============ 第1层：数据层 (Data Layer) ============
    y_data = 9.5
    
    # 数据层标题
    ax.text(0.5, y_data + 0.8, '数据层 (Data Layer)', 
            fontsize=14, fontweight='bold', color=colors['border_data'])
    
    # 多源数据
    data_boxes = [
        {'x': 0.5, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'DEM数据\n(SRTM 30m)', 'icon': '🗻'},
        {'x': 3.2, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'LULC数据\n(ESA WorldCover)', 'icon': '🌍'},
        {'x': 5.9, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'OSM道路\n(6国数据)', 'icon': '🛣️'},
        {'x': 8.6, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': 'OORD轨迹\n(真实数据)', 'icon': '📍'},
        {'x': 11.3, 'y': y_data, 'w': 2.5, 'h': 0.6, 'text': '合成轨迹\n(14,400条)', 'icon': '🎯'},
    ]
    
    for box in data_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['data'],
                              edgecolor=colors['border_data'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                f"{box['icon']}\n{box['text']}", 
                ha='center', va='center', fontsize=9)
    
    # ============ 第2层：数据处理层 (Processing Layer) ============
    y_process = 7.5
    
    # 处理层标题
    ax.text(0.5, y_process + 0.8, '数据处理层 (Processing Layer)', 
            fontsize=14, fontweight='bold', color=colors['border_process'])
    
    # 数据处理模块
    process_boxes = [
        {'x': 0.5, 'y': y_process, 'w': 3.5, 'h': 0.6, 'text': '环境数据预处理\n• 投影转换 (UTM)\n• 地形特征提取', 'icon': '⚙️'},
        {'x': 4.2, 'y': y_process, 'w': 3.5, 'h': 0.6, 'text': '代价图生成\n• 可通行域分析\n• 多意图代价计算', 'icon': '🗺️'},
        {'x': 8.0, 'y': y_process, 'w': 3.5, 'h': 0.6, 'text': '轨迹生成\n• 分层A*规划\n• XGBoost速度预测', 'icon': '🚗'},
        {'x': 11.8, 'y': y_process, 'w': 3.5, 'h': 0.6, 'text': '数据增强\n• 18通道地图\n• 训练集划分', 'icon': '📊'},
    ]
    
    for box in process_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['process'],
                              edgecolor=colors['border_process'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                f"{box['icon']} {box['text']}", 
                ha='center', va='center', fontsize=8)
    
    # ============ 第3层：模型层 (Model Layer) ============
    y_model = 5.0
    
    # 模型层标题
    ax.text(0.5, y_model + 0.8, '模型层 (Model Layer)', 
            fontsize=14, fontweight='bold', color=colors['border_model'])
    
    # TerraTNT模型架构
    model_boxes = [
        {'x': 1.0, 'y': y_model, 'w': 3.0, 'h': 0.6, 'text': 'CNN环境编码器\n(ResNet-18)', 'icon': '🖼️'},
        {'x': 4.5, 'y': y_model, 'w': 3.0, 'h': 0.6, 'text': 'LSTM历史编码器\n(双层)', 'icon': '🔄'},
        {'x': 8.0, 'y': y_model, 'w': 3.0, 'h': 0.6, 'text': '目标分类器\n(候选终点)', 'icon': '🎯'},
        {'x': 11.5, 'y': y_model, 'w': 3.0, 'h': 0.6, 'text': 'LSTM解码器\n(层次化)', 'icon': '📝'},
    ]
    
    for box in model_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['model'],
                              edgecolor=colors['border_model'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                f"{box['icon']}\n{box['text']}", 
                ha='center', va='center', fontsize=9)
    
    # 模型训练框架
    train_box = FancyBboxPatch((1.0, y_model - 1.2), 13.5, 0.8,
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['model'],
                               edgecolor=colors['border_model'], linewidth=2, linestyle='--')
    ax.add_patch(train_box)
    ax.text(7.75, y_model - 0.8, 
            '🔧 训练框架: PyTorch | 优化器: Adam | 损失: NLL + ADE | 早停机制 | TensorBoard监控', 
            ha='center', va='center', fontsize=9, style='italic')
    
    # ============ 第4层：应用层 (Application Layer) ============
    y_app = 2.0
    
    # 应用层标题
    ax.text(0.5, y_app + 0.8, '应用层 (Application Layer)', 
            fontsize=14, fontweight='bold', color=colors['border_app'])
    
    # 应用模块
    app_boxes = [
        {'x': 1.5, 'y': y_app, 'w': 3.5, 'h': 0.6, 'text': '轨迹预测服务\n• 实时推理\n• 批量预测', 'icon': '🚀'},
        {'x': 5.5, 'y': y_app, 'w': 3.5, 'h': 0.6, 'text': '可视化界面\n• 地图展示\n• 轨迹对比', 'icon': '📱'},
        {'x': 9.5, 'y': y_app, 'w': 3.5, 'h': 0.6, 'text': '评估系统\n• ADE/FDE指标\n• 消融实验', 'icon': '📈'},
    ]
    
    for box in app_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['app'],
                              edgecolor=colors['border_app'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                f"{box['icon']}\n{box['text']}", 
                ha='center', va='center', fontsize=9)
    
    # ============ 第5层：用户交互层 (User Interface) ============
    y_ui = 0.3
    
    ui_box = FancyBboxPatch((2.0, y_ui), 11.0, 0.5,
                            boxstyle="round,pad=0.05", 
                            facecolor='#FFECB3',
                            edgecolor='#FF6F00', linewidth=2)
    ax.add_patch(ui_box)
    ax.text(7.5, y_ui + 0.25, 
            '👤 用户界面: Web Dashboard | REST API | 命令行工具', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ============ 绘制数据流箭头 ============
    arrow_style = "Simple,tail_width=0.5,head_width=8,head_length=8"
    
    # 数据层 -> 处理层
    for i, x in enumerate([2.25, 4.45, 7.15, 9.85, 12.55]):
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
    
    # ============ 添加图例 ============
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor=colors['border_data'], 
                      label='数据层 - 多源地理数据'),
        mpatches.Patch(facecolor=colors['process'], edgecolor=colors['border_process'], 
                      label='处理层 - 数据预处理与生成'),
        mpatches.Patch(facecolor=colors['model'], edgecolor=colors['border_model'], 
                      label='模型层 - TerraTNT深度学习模型'),
        mpatches.Patch(facecolor=colors['app'], edgecolor=colors['border_app'], 
                      label='应用层 - 预测服务与评估'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
             framealpha=0.9, edgecolor='black')
    
    # ============ 添加系统特性标注 ============
    ax.text(15.5, 10.5, '系统特性', fontsize=11, fontweight='bold', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    features = [
        '✓ 多区域支持',
        '✓ 并行计算',
        '✓ GPU加速',
        '✓ 实时推理',
        '✓ 可扩展架构'
    ]
    for i, feature in enumerate(features):
        ax.text(15.5, 10.0 - i*0.4, feature, fontsize=8)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    fig = draw_system_architecture()
    
    # 保存为高清图片
    output_path = '/home/zmc/文档/programwork/docs/system_architecture.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 系统架构图已保存: {output_path}")
    
    # 同时保存为PDF（论文用）
    pdf_path = '/home/zmc/文档/programwork/docs/system_architecture.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF版本已保存: {pdf_path}")
