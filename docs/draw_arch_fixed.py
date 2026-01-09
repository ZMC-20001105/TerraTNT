"""
系统架构图 - 修复中文字体显示
关键：在每个text()调用上都使用fontproperties参数
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

# 加载中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
cn_font = FontProperties(fname=font_path)

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
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
    
    # 标题 - 使用fontproperties
    ax.text(9, 12.3, 'TerraTNT: 基于环境约束的地面目标轨迹预测系统', 
            ha='center', va='top', fontsize=18, fontweight='bold',
            fontproperties=cn_font)
    
    # ============ 数据层 ============
    y_data = 10.2
    
    ax.text(0.8, y_data + 0.7, '数据层 (Data Layer)', 
            fontsize=15, fontweight='bold', color=colors['border_data'],
            fontproperties=cn_font)
    
    data_boxes = [
        {'x': 0.8, 'y': y_data, 'w': 2.8, 'h': 0.7, 'text': 'DEM数据\nSRTM 30m'},
        {'x': 4.0, 'y': y_data, 'w': 2.8, 'h': 0.7, 'text': 'LULC数据\nESA WorldCover'},
        {'x': 7.2, 'y': y_data, 'w': 2.8, 'h': 0.7, 'text': 'OSM道路\n6国数据'},
        {'x': 10.4, 'y': y_data, 'w': 2.8, 'h': 0.7, 'text': 'OORD轨迹\n真实数据'},
        {'x': 13.6, 'y': y_data, 'w': 2.8, 'h': 0.7, 'text': '合成轨迹\n14,400条'},
    ]
    
    for box in data_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['data'],
                              edgecolor=colors['border_data'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=10, fontweight='bold',
                fontproperties=cn_font)
    
    # ============ 处理层 ============
    y_process = 8.2
    
    ax.text(0.8, y_process + 0.7, '处理层 (Processing Layer)', 
            fontsize=15, fontweight='bold', color=colors['border_process'],
            fontproperties=cn_font)
    
    process_boxes = [
        {'x': 0.8, 'y': y_process, 'w': 3.8, 'h': 0.7, 
         'text': '环境数据预处理\n投影转换 | 地形特征提取'},
        {'x': 5.0, 'y': y_process, 'w': 3.8, 'h': 0.7, 
         'text': '代价图生成\n可通行域分析 | 多意图代价'},
        {'x': 9.2, 'y': y_process, 'w': 3.8, 'h': 0.7, 
         'text': '轨迹生成\n分层A* | XGBoost速度预测'},
        {'x': 13.4, 'y': y_process, 'w': 3.8, 'h': 0.7, 
         'text': '数据增强\n18通道地图 | 训练集划分'},
    ]
    
    for box in process_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['process'],
                              edgecolor=colors['border_process'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=9.5,
                fontproperties=cn_font)
    
    # ============ 模型层 ============
    y_model = 5.8
    
    ax.text(0.8, y_model + 0.7, '模型层 (TerraTNT + 对比模型)', 
            fontsize=15, fontweight='bold', color=colors['border_model'],
            fontproperties=cn_font)
    
    model_boxes = [
        {'x': 1.5, 'y': y_model, 'w': 3.3, 'h': 0.7, 
         'text': 'CNN环境编码器\n(ResNet-18)'},
        {'x': 5.3, 'y': y_model, 'w': 3.3, 'h': 0.7, 
         'text': 'LSTM历史编码器\n(双层)'},
        {'x': 9.1, 'y': y_model, 'w': 3.3, 'h': 0.7, 
         'text': '目标分类器\n(候选终点评分)'},
        {'x': 12.9, 'y': y_model, 'w': 3.3, 'h': 0.7, 
         'text': 'LSTM解码器\n(层次化)'},
    ]
    
    for box in model_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['model'],
                              edgecolor=colors['border_model'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=10, fontweight='bold',
                fontproperties=cn_font)
    
    # 训练框架
    train_box = FancyBboxPatch((1.5, y_model - 1.3), 14.7, 0.9,
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['model'],
                               edgecolor=colors['border_model'], linewidth=2, linestyle='--')
    ax.add_patch(train_box)
    ax.text(8.85, y_model - 0.85, 
            '训练框架: PyTorch | Adam优化器 | 损失: NLL+ADE+FDE | 早停 | TensorBoard', 
            ha='center', va='center', fontsize=9.5, style='italic',
            fontproperties=cn_font)
    
    # 对比模型框
    baseline_box = FancyBboxPatch((1.5, y_model - 2.4), 14.7, 0.9,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='#FFF9C4',
                                 edgecolor=colors['border_model'], linewidth=2, linestyle=':')
    ax.add_patch(baseline_box)
    ax.text(8.85, y_model - 1.95, 
            '对比基线: Constant Velocity | Social-LSTM | YNet | PECNet | Trajectron++ | AgentFormer', 
            ha='center', va='center', fontsize=9.5, style='italic',
            fontproperties=cn_font)
    
    # ============ 应用层 ============
    y_app = 2.2
    
    ax.text(0.8, y_app + 0.7, '应用层 (Application Layer)', 
            fontsize=15, fontweight='bold', color=colors['border_app'],
            fontproperties=cn_font)
    
    app_boxes = [
        {'x': 2.2, 'y': y_app, 'w': 3.8, 'h': 0.7, 
         'text': '轨迹预测服务\n实时推理 | 批量处理'},
        {'x': 6.6, 'y': y_app, 'w': 3.8, 'h': 0.7, 
         'text': '可视化界面\n地图展示 | 对比分析'},
        {'x': 11.0, 'y': y_app, 'w': 3.8, 'h': 0.7, 
         'text': '评估系统\nADE/FDE | 消融实验'},
    ]
    
    for box in app_boxes:
        rect = FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['app'],
                              edgecolor=colors['border_app'], linewidth=2.5)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', fontsize=10, fontweight='bold',
                fontproperties=cn_font)
    
    # ============ 用户界面 ============
    y_ui = 0.6
    
    ui_box = FancyBboxPatch((3.0, y_ui), 12.0, 0.6,
                            boxstyle="round,pad=0.05", 
                            facecolor='#FFECB3',
                            edgecolor='#FF6F00', linewidth=2.5)
    ax.add_patch(ui_box)
    ax.text(9.0, y_ui + 0.3, 
            '用户界面: Web仪表盘 | REST API | 命令行工具', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            fontproperties=cn_font)
    
    # ============ 绘制箭头 ============
    arrow_style = "Simple,tail_width=0.6,head_width=10,head_length=10"
    
    for x in [2.2, 5.4, 8.6, 11.8, 15.0]:
        arrow = FancyArrowPatch((x, y_data - 0.1), (x, y_process + 0.8),
                               arrowstyle=arrow_style, color=colors['border_process'],
                               linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    for x in [2.7, 6.9, 11.1, 15.3]:
        arrow = FancyArrowPatch((x, y_process - 0.1), (x, y_model + 0.8),
                               arrowstyle=arrow_style, color=colors['border_model'],
                               linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    for x in [4.1, 9.0, 12.9]:
        arrow = FancyArrowPatch((x, y_model - 2.5), (x, y_app + 0.8),
                               arrowstyle=arrow_style, color=colors['border_app'],
                               linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    for x in [4.1, 8.5, 12.9]:
        arrow = FancyArrowPatch((x, y_app - 0.1), (x, y_ui + 0.7),
                               arrowstyle=arrow_style, color='#FF6F00',
                               linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    # ============ 图例 ============
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor=colors['border_data'], 
                      label='数据层'),
        mpatches.Patch(facecolor=colors['process'], edgecolor=colors['border_process'], 
                      label='处理层'),
        mpatches.Patch(facecolor=colors['model'], edgecolor=colors['border_model'], 
                      label='模型层'),
        mpatches.Patch(facecolor=colors['app'], edgecolor=colors['border_app'], 
                      label='应用层'),
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                      framealpha=0.95, edgecolor='black', fancybox=True,
                      prop=cn_font)
    
    # ============ 系统特性 ============
    feature_box = FancyBboxPatch((16.8, 9.0), 1.0, 2.5,
                                boxstyle="round,pad=0.08", 
                                facecolor='wheat', alpha=0.8,
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(feature_box)
    
    ax.text(17.3, 11.2, '系统特性', fontsize=10, fontweight='bold', ha='center',
            fontproperties=cn_font)
    
    features = ['多区域支持', '并行计算', 'GPU加速', '实时推理', '可扩展架构']
    for i, feature in enumerate(features):
        ax.text(17.3, 10.8 - i*0.35, f'✓ {feature}', fontsize=8.5, ha='center',
                fontproperties=cn_font)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("正在生成中文版系统架构图...")
    
    fig = draw_system_architecture()
    
    output_path = '/home/zmc/文档/programwork/docs/system_architecture_cn.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PNG已保存: {output_path}")
    
    pdf_path = '/home/zmc/文档/programwork/docs/system_architecture_cn.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF已保存: {pdf_path}")
    
    plt.close()
    print("✅ 完成！中文字体应该正常显示了")
