"""
轨迹可视化模块

支持：
- 中文字体
- 1:1比例（轨迹平面图）
- 地图背景
- 论文适配的尺寸和字体
"""
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
import contextily as ctx

from config import cfg

logger = logging.getLogger(__name__)


class TrajectoryPlotter:
    """轨迹可视化器"""
    
    def __init__(self):
        # 加载中文字体
        self.setup_chinese_font()
        
        # 加载配置
        self.font_sizes = cfg.get('visualization.font_sizes', {
            'title': 10,
            'label': 9,
            'tick': 8,
            'legend': 8
        })
        
        self.line_width = cfg.get('visualization.line_width', 1.0)
        self.marker_size = cfg.get('visualization.marker_size', 3)
        
        logger.info("✓ 轨迹可视化器初始化完成")
    
    def setup_chinese_font(self):
        """设置中文字体 - 使用Noto Sans CJK"""
        import os
        
        # 使用Noto Sans CJK字体（最可靠的中文字体）
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        
        if os.path.exists(font_path):
            try:
                # 创建FontProperties对象，指定字体文件和索引（TTC文件包含多个字体）
                # 索引0是简体中文
                from matplotlib.font_manager import FontProperties
                self.chinese_font = FontProperties(fname=font_path)
                
                # 不修改全局rcParams，而是在每个文本元素上显式使用FontProperties
                logger.info(f"✓ 中文字体已加载: {font_path}")
            except Exception as e:
                logger.error(f"字体加载失败: {e}")
                self.chinese_font = None
        else:
            logger.error(f"字体文件不存在: {font_path}")
            self.chinese_font = None
    
    def plot_trajectory_paper_style(
        self,
        trajectory: Dict,
        df: pd.DataFrame,
        output_path: Path,
        add_basemap: bool = True
    ):
        """
        生成论文风格的轨迹可视化（2×2子图）
        
        Args:
            trajectory: 轨迹字典
            df: 轨迹DataFrame
            output_path: 输出路径
            add_basemap: 是否添加地图背景
        """
        # 强制设置中文字体（在绘图前）
        if hasattr(self, 'chinese_font') and self.chinese_font:
            plt.rcParams['font.family'] = self.chinese_font.get_name()
        
        # 使用双栏尺寸（增加高度以容纳标题和标签）
        fig_width, fig_height = cfg.get('visualization.paper_figure_size.double_column', [7.0, 5.25])
        fig_height = 6.0  # 增加高度避免文字遮挡
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        
        # 创建2×2子图（增加间距避免文字遮挡）
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35, 
                             left=0.08, right=0.96, top=0.94, bottom=0.08)
        
        # 1. 轨迹平面图（左上，1:1比例）
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_trajectory_map(ax1, trajectory, df, add_basemap)
        
        # 2. 速度-时间曲线（右上）
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_speed_time(ax2, df)
        
        # 3. 加速度-时间曲线（左下）
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_acceleration_time(ax3, df)
        
        # 4. 曲率-时间曲线（右下）
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_curvature_time(ax4, df)
        
        # 保存（不使用bbox_inches='tight'以保持1:1比例）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white')
        plt.close(fig)
        
        logger.info(f"✓ 论文风格可视化已保存: {output_path}")
    
    def _plot_trajectory_map(self, ax, trajectory: Dict, df: pd.DataFrame, add_basemap: bool):
        """绘制轨迹平面图（1:1比例）"""
        import rasterio
        from rasterio.plot import show
        import numpy as np
        
        utm_x = df['utm_x'].values
        utm_y = df['utm_y'].values
        speeds = df['speed'].values
        
        # 添加DEM背景
        if add_basemap:
            try:
                # 加载DEM栅格
                dem_path = cfg.get('paths.processed.utm_grid') + '/scottish_highlands/dem_utm.tif'
                
                with rasterio.open(dem_path) as dem_src:
                    # 计算轨迹范围
                    x_min, x_max = utm_x.min(), utm_x.max()
                    y_min, y_max = utm_y.min(), utm_y.max()
                    
                    # 添加边距（10%）
                    x_margin = (x_max - x_min) * 0.1
                    y_margin = (y_max - y_min) * 0.1
                    
                    # 读取DEM窗口
                    window = rasterio.windows.from_bounds(
                        x_min - x_margin, y_min - y_margin,
                        x_max + x_margin, y_max + y_margin,
                        dem_src.transform
                    )
                    
                    dem_data = dem_src.read(1, window=window)
                    dem_transform = dem_src.window_transform(window)
                    
                    # 处理NoData值（-32768）
                    dem_data = dem_data.astype(float)
                    dem_data[dem_data == -32768] = np.nan
                    
                    # 绘制DEM作为背景（使用地形色带）
                    extent = [
                        dem_transform[2],  # left
                        dem_transform[2] + dem_transform[0] * dem_data.shape[1],  # right
                        dem_transform[5] + dem_transform[4] * dem_data.shape[0],  # bottom
                        dem_transform[5]   # top
                    ]
                    
                    # 使用terrain色带显示DEM，提高透明度使其更明显
                    # 注意：不在imshow中设置aspect，而是通过ax.set_aspect统一控制
                    im = ax.imshow(dem_data, extent=extent, cmap='terrain', 
                                  alpha=0.6, zorder=1, interpolation='bilinear')
                    
                logger.debug("✓ DEM背景已添加")
            except Exception as e:
                logger.warning(f"DEM背景添加失败: {e}")
        
        # 绘制轨迹，用速度着色
        scatter = ax.scatter(utm_x, utm_y, c=speeds, cmap='viridis', 
                            s=self.marker_size, alpha=0.8, zorder=3)
        
        # 标记起终点
        ax.plot(utm_x[0], utm_y[0], 'go', markersize=6, 
               label='起点', zorder=4, markeredgecolor='white', markeredgewidth=0.5)
        ax.plot(utm_x[-1], utm_y[-1], 'ro', markersize=6, 
               label='终点', zorder=4, markeredgecolor='white', markeredgewidth=0.5)
        
        # 设置1:1比例（确保X和Y轴刻度相同，DEM背景也会按此比例显示）
        ax.set_aspect('equal', adjustable='box')
        
        # 标签和标题（显式指定字体）
        if self.chinese_font:
            ax.set_xlabel('UTM X (m)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
            ax.set_ylabel('UTM Y (m)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
            ax.set_title(f'轨迹平面图 ({trajectory["intent"]}, {trajectory["vehicle_type"]})', 
                        fontsize=self.font_sizes['title'], fontweight='bold', pad=10, fontproperties=self.chinese_font)
        else:
            ax.set_xlabel('UTM X (m)', fontsize=self.font_sizes['label'])
            ax.set_ylabel('UTM Y (m)', fontsize=self.font_sizes['label'])
            ax.set_title(f'轨迹平面图 ({trajectory["intent"]}, {trajectory["vehicle_type"]})', 
                        fontsize=self.font_sizes['title'], fontweight='bold', pad=10)
        
        # 图例（显式指定字体）
        if self.chinese_font:
            legend = ax.legend(loc='upper right', fontsize=self.font_sizes['legend'], 
                              framealpha=0.9, edgecolor='gray', fancybox=False, prop=self.chinese_font)
        else:
            legend = ax.legend(loc='upper right', fontsize=self.font_sizes['legend'], 
                              framealpha=0.9, edgecolor='gray', fancybox=False)
        
        # 网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 刻度标签 - 自动格式化X轴刻度，避免数字重叠
        ax.tick_params(labelsize=self.font_sizes['tick'])
        
        # 使用科学计数法或自动旋转X轴标签
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f'{y/1e6:.2f}M'))
        
        # 色标
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        if self.chinese_font:
            cbar.set_label('速度 (m/s)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
        else:
            cbar.set_label('速度 (m/s)', fontsize=self.font_sizes['label'])
        cbar.ax.tick_params(labelsize=self.font_sizes['tick'])
    
    def _plot_speed_time(self, ax, df: pd.DataFrame):
        """绘制速度-时间曲线"""
        time_min = df['timestamp'].values / 60
        speeds = df['speed'].values
        
        ax.plot(time_min, speeds, 'b-', linewidth=self.line_width, alpha=0.8)
        ax.axhline(y=speeds.mean(), color='r', linestyle='--', linewidth=self.line_width*0.8,
                  label=f'平均: {speeds.mean():.2f} m/s', alpha=0.7)
        
        if self.chinese_font:
            ax.set_xlabel('时间 (分钟)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
            ax.set_ylabel('速度 (m/s)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
            ax.set_title('速度-时间曲线', fontsize=self.font_sizes['title'], fontweight='bold', pad=10, fontproperties=self.chinese_font)
            ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9, edgecolor='gray', prop=self.chinese_font)
        else:
            ax.set_xlabel('时间 (分钟)', fontsize=self.font_sizes['label'])
            ax.set_ylabel('速度 (m/s)', fontsize=self.font_sizes['label'])
            ax.set_title('速度-时间曲线', fontsize=self.font_sizes['title'], fontweight='bold', pad=10)
            ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=self.font_sizes['tick'])
    
    def _plot_acceleration_time(self, ax, df: pd.DataFrame):
        """绘制加速度-时间曲线"""
        time_min = df['timestamp'].values / 60
        accel = df['acceleration'].values
        
        ax.plot(time_min, accel, 'g-', linewidth=self.line_width, alpha=0.8)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=self.line_width*0.5, alpha=0.5)
        
        if self.chinese_font:
            ax.set_xlabel('时间 (分钟)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
            ax.set_ylabel('加速度 (m/s²)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
            ax.set_title('加速度-时间曲线', fontsize=self.font_sizes['title'], fontweight='bold', pad=10, fontproperties=self.chinese_font)
        else:
            ax.set_xlabel('时间 (分钟)', fontsize=self.font_sizes['label'])
            ax.set_ylabel('加速度 (m/s²)', fontsize=self.font_sizes['label'])
            ax.set_title('加速度-时间曲线', fontsize=self.font_sizes['title'], fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=self.font_sizes['tick'])
    
    def _plot_curvature_time(self, ax, df: pd.DataFrame):
        """绘制曲率-时间曲线"""
        time_min = df['timestamp'].values / 60
        curvature = df['curvature'].values
        
        ax.plot(time_min, curvature, color='orange', linewidth=self.line_width, alpha=0.8)
        
        if self.chinese_font:
            ax.set_xlabel('时间 (分钟)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
            ax.set_ylabel('曲率 (1/m)', fontsize=self.font_sizes['label'], fontproperties=self.chinese_font)
            ax.set_title('曲率-时间曲线', fontsize=self.font_sizes['title'], fontweight='bold', pad=10, fontproperties=self.chinese_font)
        else:
            ax.set_xlabel('时间 (分钟)', fontsize=self.font_sizes['label'])
            ax.set_ylabel('曲率 (1/m)', fontsize=self.font_sizes['label'])
            ax.set_title('曲率-时间曲线', fontsize=self.font_sizes['title'], fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=self.font_sizes['tick'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 测试
    import pickle
    
    test_pkl = Path('/home/zmc/文档/programwork/data/processed/synthetic_trajectories/test/test_trajectory.pkl')
    test_csv = Path('/home/zmc/文档/programwork/data/processed/synthetic_trajectories/test/test_trajectory.csv')
    
    if test_pkl.exists() and test_csv.exists():
        with open(test_pkl, 'rb') as f:
            trajectory = pickle.load(f)
        
        df = pd.read_csv(test_csv)
        
        plotter = TrajectoryPlotter()
        output_path = test_pkl.parent / 'trajectory_paper_style.png'
        
        plotter.plot_trajectory_paper_style(trajectory, df, output_path, add_basemap=True)
        
        print(f"\n✓ 测试完成，输出: {output_path}")
    else:
        print("✗ 测试文件不存在")
