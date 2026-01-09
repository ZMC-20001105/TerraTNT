"""
轨迹预测结果可视化工具
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
cn_font = FontProperties(fname=font_path)

# 颜色方案
COLORS = {
    'history': '#3B82F6',      # Blue
    'ground_truth': '#10B981', # Green
    'TerraTNT': '#EF4444',     # Red
    'YNet': '#F59E0B',         # Amber
    'PECNet': '#8B5CF6',       # Purple
    'Trajectron++': '#EC4899', # Pink
    'Social-LSTM': '#06B6D4',  # Cyan
    'CV': '#6B7280'            # Gray
}


def plot_trajectory_comparison(
    history: np.ndarray,
    ground_truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
    env_map: Optional[np.ndarray] = None,
    title: str = "轨迹预测对比",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    绘制多模型轨迹预测对比图
    
    Args:
        history: 历史轨迹 (T_hist, 2)
        ground_truth: 真实未来轨迹 (T_fut, 2)
        predictions: 各模型预测结果 {model_name: (T_fut, 2)}
        env_map: 环境地图背景 (H, W) 或 (H, W, 3)
        title: 图标题
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制环境背景
    if env_map is not None:
        ax.imshow(env_map, cmap='terrain', alpha=0.3, 
                 extent=[history[:, 0].min() - 10, ground_truth[:, 0].max() + 10,
                        history[:, 1].min() - 10, ground_truth[:, 1].max() + 10])
    
    # 绘制历史轨迹
    ax.plot(history[:, 0], history[:, 1], 
           color=COLORS['history'], linewidth=2, marker='o', markersize=4,
           label='历史轨迹 (History)')
    
    # 绘制真实轨迹
    ax.plot(ground_truth[:, 0], ground_truth[:, 1],
           color=COLORS['ground_truth'], linewidth=2, marker='s', markersize=4,
           label='真实轨迹 (Ground Truth)')
    
    # 绘制各模型预测
    for model_name, pred in predictions.items():
        color = COLORS.get(model_name, '#000000')
        ax.plot(pred[:, 0], pred[:, 1],
               color=color, linewidth=2, linestyle='--', marker='^', markersize=3,
               label=f'{model_name}', alpha=0.8)
    
    # 标记起点和终点
    ax.scatter(history[0, 0], history[0, 1], c='blue', s=100, marker='*', 
              zorder=10, label='起点')
    ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c='green', s=100, marker='*',
              zorder=10, label='真实终点')
    
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', fontproperties=cn_font)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    绘制模型性能对比柱状图
    
    Args:
        results: {model_name: {metric_name: value}}
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.6
    
    # ADE
    ade_values = [results[m]['ade'] for m in models]
    colors = [COLORS.get(m, '#6B7280') for m in models]
    axes[0].bar(x, ade_values, width, color=colors, edgecolor='black')
    axes[0].set_ylabel('ADE (m)', fontsize=12)
    axes[0].set_title('平均位移误差 (ADE)', fontsize=12, fontweight='bold', fontproperties=cn_font)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # FDE
    fde_values = [results[m]['fde'] for m in models]
    axes[1].bar(x, fde_values, width, color=colors, edgecolor='black')
    axes[1].set_ylabel('FDE (m)', fontsize=12)
    axes[1].set_title('最终位移误差 (FDE)', fontsize=12, fontweight='bold', fontproperties=cn_font)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Goal Accuracy
    goal_values = [results[m]['goal_accuracy'] * 100 for m in models]
    axes[2].bar(x, goal_values, width, color=colors, edgecolor='black')
    axes[2].set_ylabel('Goal Accuracy (%)', fontsize=12)
    axes[2].set_title('目标预测准确率', fontsize=12, fontweight='bold', fontproperties=cn_font)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
):
    """
    绘制消融实验结果
    
    Args:
        ablation_results: {variant_name: {metric: value}}
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = list(ablation_results.keys())
    metrics = ['ade', 'fde']
    x = np.arange(len(variants))
    width = 0.35
    
    ade_values = [ablation_results[v]['ade'] for v in variants]
    fde_values = [ablation_results[v]['fde'] for v in variants]
    
    bars1 = ax.bar(x - width/2, ade_values, width, label='ADE (m)', color='#3B82F6')
    bars2 = ax.bar(x + width/2, fde_values, width, label='FDE (m)', color='#EF4444')
    
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title('消融实验结果 / Ablation Study', fontsize=14, fontweight='bold', fontproperties=cn_font)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def generate_paper_figures(
    results: Dict[str, Dict[str, float]],
    output_dir: Path
):
    """
    生成论文所需的所有图表
    
    Args:
        results: 评估结果
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 性能对比柱状图
    plot_metrics_comparison(results, output_dir / 'fig_metrics_comparison.png')
    print(f"✓ Generated: fig_metrics_comparison.png")
    
    # 2. 示例轨迹对比图 (需要实际数据)
    # plot_trajectory_comparison(...)
    
    print(f"✅ All figures saved to {output_dir}")


if __name__ == '__main__':
    # 示例数据
    example_results = {
        'TerraTNT': {'ade': 15.3, 'fde': 38.6, 'goal_accuracy': 0.735},
        'YNet': {'ade': 25.4, 'fde': 62.8, 'goal_accuracy': 0.457},
        'PECNet': {'ade': 22.1, 'fde': 58.3, 'goal_accuracy': 0.523},
        'Trajectron++': {'ade': 20.8, 'fde': 54.7, 'goal_accuracy': 0.568},
        'Social-LSTM': {'ade': 38.7, 'fde': 95.3, 'goal_accuracy': 0.285},
        'CV': {'ade': 45.2, 'fde': 128.5, 'goal_accuracy': 0.123}
    }
    
    generate_paper_figures(example_results, Path('/home/zmc/文档/programwork/docs/figures'))
