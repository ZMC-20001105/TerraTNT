"""
生成TerraTNT系统界面原型截图
使用matplotlib绘制6个主要界面
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.font_manager import FontProperties
import numpy as np

# 加载中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
cn_font = FontProperties(fname=font_path)

def create_header(ax, title):
    """创建统一的页面头部"""
    # 标题栏
    header = Rectangle((0, 9.5), 14, 0.5, facecolor='#2196F3', edgecolor='black', linewidth=2)
    ax.add_patch(header)
    ax.text(7, 9.75, 'TerraTNT - 地面目标轨迹预测系统', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white', fontproperties=cn_font)
    
    # 标签页
    tabs = ['概览', '数据集', '模型训练', '轨迹预测', '模型评估', '设置']
    tab_width = 14 / len(tabs)
    active_idx = tabs.index(title)
    
    for i, tab in enumerate(tabs):
        x = i * tab_width
        color = '#E3F2FD' if i == active_idx else '#F5F5F5'
        tab_box = Rectangle((x, 9.0), tab_width, 0.4, facecolor=color, 
                           edgecolor='black', linewidth=1)
        ax.add_patch(tab_box)
        weight = 'bold' if i == active_idx else 'normal'
        ax.text(x + tab_width/2, 9.2, tab, ha='center', va='center', 
               fontsize=9, fontweight=weight, fontproperties=cn_font)

def draw_overview():
    """界面1: 概览页面"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    create_header(ax, '概览')
    
    # 系统状态
    status_box = FancyBboxPatch((0.5, 7.0), 13, 1.7, boxstyle="round,pad=0.1",
                               facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(status_box)
    ax.text(1, 8.4, '系统状态', fontsize=11, fontweight='bold', fontproperties=cn_font)
    
    status_items = [
        '• 数据集: Scottish Highlands (3,600条) | Bohemian Forest (3,600条)',
        '• 模型状态: TerraTNT已训练 | 6个基线模型已加载',
        '• GPU状态: NVIDIA RTX 5060 (8GB) - 可用',
        '• 最近训练: 2026-01-08 09:30 | Epoch 45 | Val Loss: 0.0234'
    ]
    
    for i, item in enumerate(status_items):
        ax.text(1.2, 8.0 - i*0.3, item, fontsize=8.5, fontproperties=cn_font)
    
    # 快速操作
    quick_box = FancyBboxPatch((0.5, 5.3), 13, 1.4, boxstyle="round,pad=0.1",
                              facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
    ax.add_patch(quick_box)
    ax.text(1, 6.4, '快速操作', fontsize=11, fontweight='bold', fontproperties=cn_font)
    
    buttons = ['加载数据集', '开始训练', '轨迹预测', '模型评估']
    for i, btn in enumerate(buttons):
        x = 1 + i * 3.2
        btn_box = FancyBboxPatch((x, 5.6), 2.8, 0.5, boxstyle="round,pad=0.05",
                                facecolor='#4CAF50', edgecolor='black', linewidth=1.5)
        ax.add_patch(btn_box)
        ax.text(x + 1.4, 5.85, btn, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white', fontproperties=cn_font)
    
    # 最近活动
    activity_box = FancyBboxPatch((0.5, 0.5), 13, 4.5, boxstyle="round,pad=0.1",
                                 facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(activity_box)
    ax.text(1, 4.7, '最近活动', fontsize=11, fontweight='bold', fontproperties=cn_font)
    
    activities = [
        '✓ 2026-01-08 10:15 - Scottish Highlands数据集生成完成',
        '✓ 2026-01-08 09:30 - TerraTNT模型训练完成 (Epoch 45)',
        '✓ 2026-01-08 08:45 - Bohemian Forest数据预处理完成',
        '✓ 2026-01-08 07:20 - 基线模型YNet训练完成',
        '✓ 2026-01-08 06:10 - 数据增强完成 (18通道地图)',
        '✓ 2026-01-08 05:30 - 代价图生成完成',
        '✓ 2026-01-08 04:15 - 环境数据投影到UTM完成',
    ]
    
    for i, activity in enumerate(activities):
        ax.text(1.2, 4.3 - i*0.5, activity, fontsize=8, fontproperties=cn_font)
    
    plt.tight_layout()
    return fig

def draw_dataset():
    """界面2: 数据集管理"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    create_header(ax, '数据集')
    
    # 左侧：数据集列表
    list_box = FancyBboxPatch((0.5, 0.5), 4, 8, boxstyle="round,pad=0.1",
                             facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(list_box)
    ax.text(2.5, 8.2, '数据集列表', fontsize=10, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    datasets = [
        ('Scottish Highlands', '3,600条', True),
        ('Bohemian Forest', '3,600条', True),
        ('Donbas', '待生成', False),
        ('Carpathians', '待生成', False),
    ]
    
    for i, (name, count, active) in enumerate(datasets):
        y = 7.5 - i * 0.8
        color = '#BBDEFB' if active else '#F5F5F5'
        item_box = Rectangle((0.8, y), 3.4, 0.6, facecolor=color, 
                            edgecolor='black', linewidth=1)
        ax.add_patch(item_box)
        ax.text(1.0, y + 0.3, f'{name}\n({count})', va='center',
               fontsize=8, fontproperties=cn_font)
    
    # 右侧：数据集详情
    detail_box = FancyBboxPatch((5, 4.5), 8.5, 4, boxstyle="round,pad=0.1",
                               facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(detail_box)
    ax.text(9.25, 8.2, '数据集详情 - Scottish Highlands', fontsize=10, 
           fontweight='bold', ha='center', fontproperties=cn_font)
    
    stats = [
        ('区域', 'Scottish Highlands'),
        ('轨迹总数', '3,600'),
        ('平均长度', '125.3 km'),
        ('平均时长', '358.7 分钟'),
        ('车辆类型', '4种 (Type1-4)'),
        ('战术意图', '3种 (Intent1-3)'),
        ('数据大小', '252 MB'),
    ]
    
    for i, (key, value) in enumerate(stats):
        y = 7.7 - i * 0.4
        ax.text(5.5, y, f'{key}:', fontsize=8, fontproperties=cn_font)
        ax.text(8.0, y, value, fontsize=8, fontweight='bold', fontproperties=cn_font)
    
    # 数据分布图
    dist_box = FancyBboxPatch((5, 0.5), 8.5, 3.7, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(dist_box)
    ax.text(9.25, 4.0, '轨迹长度分布', fontsize=9, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    # 简单的柱状图
    x_pos = np.linspace(5.5, 13, 10)
    heights = [0.3, 0.6, 1.2, 1.8, 2.0, 1.7, 1.3, 0.8, 0.4, 0.2]
    for x, h in zip(x_pos, heights):
        bar = Rectangle((x, 0.8), 0.6, h, facecolor='#42A5F5', edgecolor='black')
        ax.add_patch(bar)
    
    ax.text(9.25, 0.6, '80   100   120   140   160 (km)', fontsize=7, ha='center')
    
    plt.tight_layout()
    return fig

def draw_training():
    """界面3: 模型训练"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    create_header(ax, '模型训练')
    
    # 左侧：训练配置
    config_box = FancyBboxPatch((0.5, 0.5), 4.5, 8, boxstyle="round,pad=0.1",
                               facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
    ax.add_patch(config_box)
    ax.text(2.75, 8.2, '训练配置', fontsize=10, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    # 模型选择
    ax.text(1, 7.7, '模型选择:', fontsize=9, fontproperties=cn_font)
    model_box = Rectangle((1, 7.2), 3.5, 0.4, facecolor='white', edgecolor='black')
    ax.add_patch(model_box)
    ax.text(1.2, 7.4, 'TerraTNT (主模型)', fontsize=8, va='center', fontproperties=cn_font)
    
    # 超参数
    params = [
        ('学习率', '0.0001'),
        ('批大小', '32'),
        ('训练轮数', '100'),
        ('优化器', 'Adam'),
    ]
    
    y_start = 6.5
    for i, (key, value) in enumerate(params):
        y = y_start - i * 0.6
        ax.text(1, y, f'{key}:', fontsize=8, fontproperties=cn_font)
        param_box = Rectangle((2.5, y - 0.15), 1.8, 0.3, facecolor='white', edgecolor='black')
        ax.add_patch(param_box)
        ax.text(2.6, y, value, fontsize=8, va='center')
    
    # 训练按钮
    train_btn = FancyBboxPatch((1, 2.5), 3.5, 0.6, boxstyle="round,pad=0.05",
                              facecolor='#4CAF50', edgecolor='black', linewidth=2)
    ax.add_patch(train_btn)
    ax.text(2.75, 2.8, '开始训练', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white', fontproperties=cn_font)
    
    # 右侧：训练监控
    monitor_box = FancyBboxPatch((5.5, 4.5), 8, 4, boxstyle="round,pad=0.1",
                                facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(monitor_box)
    ax.text(9.5, 8.2, '训练进度', fontsize=10, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    ax.text(6, 7.7, 'Epoch: 45/100', fontsize=9, fontproperties=cn_font)
    
    # 进度条
    progress_bg = Rectangle((6, 7.3), 7, 0.3, facecolor='#E0E0E0', edgecolor='black')
    ax.add_patch(progress_bg)
    progress_fg = Rectangle((6, 7.3), 3.15, 0.3, facecolor='#4CAF50', edgecolor='black')
    ax.add_patch(progress_fg)
    ax.text(9.5, 7.45, '45%', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # 损失曲线
    loss_box = FancyBboxPatch((5.5, 0.5), 8, 3.7, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(9.5, 4.0, '损失曲线', fontsize=9, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    # 绘制损失曲线
    epochs = np.linspace(6, 13, 50)
    train_loss = 3.5 - 2.8 * (epochs - 6) / 7
    val_loss = 3.6 - 2.7 * (epochs - 6) / 7
    
    for i in range(len(epochs)-1):
        ax.plot([epochs[i], epochs[i+1]], [train_loss[i], train_loss[i+1]], 
               'b-', linewidth=2)
        ax.plot([epochs[i], epochs[i+1]], [val_loss[i], val_loss[i+1]], 
               'r--', linewidth=2)
    
    ax.text(6.5, 3.5, '训练损失', fontsize=7, color='blue', fontproperties=cn_font)
    ax.text(6.5, 3.2, '验证损失', fontsize=7, color='red', fontproperties=cn_font)
    
    plt.tight_layout()
    return fig

def draw_prediction():
    """界面4: 轨迹预测"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    create_header(ax, '轨迹预测')
    
    # 左侧：预测配置
    config_box = FancyBboxPatch((0.5, 0.5), 4, 8, boxstyle="round,pad=0.1",
                               facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(config_box)
    ax.text(2.5, 8.2, '预测配置', fontsize=10, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    # 配置项
    configs = [
        ('区域', 'Scottish Highlands'),
        ('历史长度', '10 分钟'),
        ('预测长度', '60 分钟'),
    ]
    
    y_start = 7.5
    for i, (key, value) in enumerate(configs):
        y = y_start - i * 0.7
        ax.text(1, y, f'{key}:', fontsize=8, fontproperties=cn_font)
        config_item = Rectangle((1, y - 0.4), 3, 0.3, facecolor='white', edgecolor='black')
        ax.add_patch(config_item)
        ax.text(1.2, y - 0.25, value, fontsize=8, va='center', fontproperties=cn_font)
    
    # 模型选择
    ax.text(1, 5.0, '模型选择:', fontsize=8, fontproperties=cn_font)
    models = ['TerraTNT', 'YNet', 'PECNet']
    for i, model in enumerate(models):
        y = 4.5 - i * 0.4
        # 复选框
        cb = Rectangle((1, y), 0.25, 0.25, facecolor='white', edgecolor='black')
        ax.add_patch(cb)
        if i == 0:  # TerraTNT选中
            ax.text(1.125, y + 0.125, '✓', ha='center', va='center', fontsize=10)
        ax.text(1.4, y + 0.125, model, fontsize=8, va='center')
    
    # 预测按钮
    pred_btn = FancyBboxPatch((1, 2.5), 3, 0.6, boxstyle="round,pad=0.05",
                             facecolor='#2196F3', edgecolor='black', linewidth=2)
    ax.add_patch(pred_btn)
    ax.text(2.5, 2.8, '开始预测', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white', fontproperties=cn_font)
    
    # 右侧：地图显示
    map_box = FancyBboxPatch((5, 0.5), 8.5, 8, boxstyle="round,pad=0.1",
                            facecolor='#F5F5F5', edgecolor='#388E3C', linewidth=2)
    ax.add_patch(map_box)
    ax.text(9.25, 8.2, '轨迹可视化', fontsize=10, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    # 绘制地形背景
    for i in range(10):
        for j in range(10):
            color_val = 0.7 + 0.2 * np.sin(i/2) * np.cos(j/2)
            terrain = Rectangle((5.5 + i*0.8, 1 + j*0.7), 0.8, 0.7,
                              facecolor=plt.cm.terrain(color_val), alpha=0.3)
            ax.add_patch(terrain)
    
    # 绘制轨迹
    # 历史轨迹（蓝色）
    hist_x = [6, 6.5, 7, 7.5, 8]
    hist_y = [2, 2.5, 3, 3.5, 4]
    for i in range(len(hist_x)-1):
        ax.plot([hist_x[i], hist_x[i+1]], [hist_y[i], hist_y[i+1]], 
               'b-', linewidth=3)
        ax.plot(hist_x[i], hist_y[i], 'bo', markersize=6)
    
    # 预测轨迹（红色）
    pred_x = [8, 8.5, 9, 9.5, 10, 10.5, 11]
    pred_y = [4, 4.5, 5, 5.5, 6, 6.5, 7]
    for i in range(len(pred_x)-1):
        ax.plot([pred_x[i], pred_x[i+1]], [pred_y[i], pred_y[i+1]], 
               'r--', linewidth=3)
        ax.plot(pred_x[i], pred_y[i], 'ro', markersize=6)
    
    # 图例
    ax.text(6, 1.3, '● 历史轨迹', fontsize=7, color='blue', fontproperties=cn_font)
    ax.text(7.5, 1.3, '● 预测轨迹', fontsize=7, color='red', fontproperties=cn_font)
    
    plt.tight_layout()
    return fig

def draw_evaluation():
    """界面5: 模型评估"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    create_header(ax, '模型评估')
    
    # 对比表格
    table_box = FancyBboxPatch((0.5, 5.5), 13, 3, boxstyle="round,pad=0.1",
                              facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(table_box)
    ax.text(7, 8.2, '模型性能对比', fontsize=10, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    # 表头
    headers = ['模型', 'ADE (m)', 'FDE (m)', 'Goal Acc (%)', '时间 (ms)']
    x_positions = [1, 3.5, 5.5, 8, 11]
    for x, header in zip(x_positions, headers):
        ax.text(x, 7.8, header, fontsize=8, fontweight='bold', fontproperties=cn_font)
    
    # 数据行
    models_data = [
        ('TerraTNT', '15.3', '38.6', '73.5', '45'),
        ('YNet', '25.4', '62.8', '45.7', '35'),
        ('PECNet', '22.1', '58.3', '52.3', '42'),
        ('Trajectron++', '20.8', '54.7', '56.8', '68'),
        ('AgentFormer', '19.5', '51.2', '61.2', '85'),
        ('Social-LSTM', '38.7', '95.3', '28.5', '15'),
    ]
    
    for i, row_data in enumerate(models_data):
        y = 7.2 - i * 0.4
        color = '#C8E6C9' if i == 0 else 'white'
        row_bg = Rectangle((0.8, y - 0.15), 12.4, 0.3, facecolor=color, 
                          edgecolor='black', linewidth=0.5)
        ax.add_patch(row_bg)
        
        for j, (x, value) in enumerate(zip(x_positions, row_data)):
            weight = 'bold' if i == 0 and j > 0 else 'normal'
            ax.text(x, y, value, fontsize=7, fontweight=weight, va='center',
                   fontproperties=cn_font if j == 0 else None)
    
    # 可视化对比
    chart_box = FancyBboxPatch((0.5, 0.5), 13, 4.7, boxstyle="round,pad=0.1",
                              facecolor='white', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(chart_box)
    ax.text(7, 5.0, 'ADE/FDE对比', fontsize=10, fontweight='bold', 
           ha='center', fontproperties=cn_font)
    
    # ADE柱状图
    models_short = ['Terra\nTNT', 'YNet', 'PEC\nNet', 'Traj++', 'Agent\nFormer', 'Social\nLSTM']
    ade_values = [15.3, 25.4, 22.1, 20.8, 19.5, 38.7]
    
    x_pos = np.linspace(1.5, 6, 6)
    for i, (x, val) in enumerate(zip(x_pos, ade_values)):
        color = '#4CAF50' if i == 0 else '#42A5F5'
        height = val / 10
        bar = Rectangle((x, 0.8), 0.6, height, facecolor=color, edgecolor='black')
        ax.add_patch(bar)
        ax.text(x + 0.3, 0.8 + height + 0.1, f'{val}', ha='center', fontsize=6)
        ax.text(x + 0.3, 0.6, models_short[i], ha='center', fontsize=6)
    
    ax.text(3.75, 4.7, 'ADE (m)', fontsize=8, fontweight='bold')
    
    # FDE柱状图
    fde_values = [38.6, 62.8, 58.3, 54.7, 51.2, 95.3]
    x_pos2 = np.linspace(8, 12.5, 6)
    for i, (x, val) in enumerate(zip(x_pos2, fde_values)):
        color = '#4CAF50' if i == 0 else '#FF7043'
        height = val / 25
        bar = Rectangle((x, 0.8), 0.6, height, facecolor=color, edgecolor='black')
        ax.add_patch(bar)
        ax.text(x + 0.3, 0.8 + height + 0.1, f'{val}', ha='center', fontsize=6)
        ax.text(x + 0.3, 0.6, models_short[i], ha='center', fontsize=6)
    
    ax.text(10.25, 4.7, 'FDE (m)', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    return fig

def draw_settings():
    """界面6: 系统设置"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    create_header(ax, '设置')
    
    # GPU设置
    gpu_box = FancyBboxPatch((0.5, 6.5), 13, 2, boxstyle="round,pad=0.1",
                            facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
    ax.add_patch(gpu_box)
    ax.text(1, 8.2, 'GPU设置', fontsize=10, fontweight='bold', fontproperties=cn_font)
    
    gpu_info = [
        'GPU设备: NVIDIA RTX 5060 (8GB)',
        'CUDA版本: 13.0',
        '显存使用: 1.2GB / 8.0GB (15%)',
        '计算能力: 8.9',
    ]
    
    for i, info in enumerate(gpu_info):
        ax.text(1.5, 7.7 - i * 0.4, f'• {info}', fontsize=8.5)
    
    # 路径设置
    path_box = FancyBboxPatch((0.5, 3.5), 13, 2.7, boxstyle="round,pad=0.1",
                             facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(path_box)
    ax.text(1, 6.0, '路径设置', fontsize=10, fontweight='bold', fontproperties=cn_font)
    
    paths = [
        ('数据目录', '/home/zmc/文档/programwork/data'),
        ('模型目录', '/home/zmc/文档/programwork/models/saved'),
        ('日志目录', '/home/zmc/文档/programwork/runs'),
    ]
    
    for i, (label, path) in enumerate(paths):
        y = 5.5 - i * 0.7
        ax.text(1.5, y, f'{label}:', fontsize=8.5, fontproperties=cn_font)
        path_field = Rectangle((1.5, y - 0.4), 10, 0.3, facecolor='white', edgecolor='black')
        ax.add_patch(path_field)
        ax.text(1.7, y - 0.25, path, fontsize=7, va='center')
        
        # 浏览按钮
        browse_btn = Rectangle((11.7, y - 0.4), 0.8, 0.3, facecolor='#2196F3', edgecolor='black')
        ax.add_patch(browse_btn)
        ax.text(12.1, y - 0.25, '浏览', ha='center', va='center', 
               fontsize=7, color='white', fontproperties=cn_font)
    
    # 训练设置
    train_box = FancyBboxPatch((0.5, 0.5), 13, 2.7, boxstyle="round,pad=0.1",
                              facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(train_box)
    ax.text(1, 3.0, '训练设置', fontsize=10, fontweight='bold', fontproperties=cn_font)
    
    train_settings = [
        ('默认批大小', '32'),
        ('默认学习率', '0.0001'),
        ('早停轮数', '10'),
        ('检查点保存', '每5个epoch'),
    ]
    
    for i, (label, value) in enumerate(train_settings):
        y = 2.5 - i * 0.5
        ax.text(1.5, y, f'{label}:', fontsize=8.5, fontproperties=cn_font)
        ax.text(5, y, value, fontsize=8.5, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    print("正在生成TerraTNT系统界面原型截图...")
    
    pages = [
        ('概览', draw_overview),
        ('数据集', draw_dataset),
        ('模型训练', draw_training),
        ('轨迹预测', draw_prediction),
        ('模型评估', draw_evaluation),
        ('设置', draw_settings),
    ]
    
    for i, (name, draw_func) in enumerate(pages, 1):
        print(f"  生成界面 {i}/6: {name}...")
        fig = draw_func()
        filename = f'/home/zmc/文档/programwork/docs/ui_screenshot_{i}_{name}.png'
        fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ 已保存: {filename}")
    
    print("\n✅ 所有界面截图已生成完成！")
    print(f"\n生成的文件:")
    for i, (name, _) in enumerate(pages, 1):
        print(f"  {i}. ui_screenshot_{i}_{name}.png")

if __name__ == '__main__':
    main()
