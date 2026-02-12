"""
全局绘图配置
统一管理所有可视化相关的样式和参数
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional
import warnings

from config import cfg

warnings.filterwarnings('ignore', category=UserWarning)


class PlotConfig:
    """绘图配置管理类"""
    
    def __init__(self):
        self.setup_style()
        self.setup_colors()
        self.setup_fonts()
        self.setup_figure()
    
    def setup_style(self):
        """设置绘图风格"""
        style = cfg.get('plotting.style', 'seaborn-v0_8-darkgrid')
        context = cfg.get('plotting.context', 'paper')
        
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        
        sns.set_context(context)
        
        # 设置 seaborn 调色板
        sns.set_palette("husl")
    
    def setup_colors(self):
        """设置颜色方案"""
        self.colors = cfg.get('plotting.colors', {})
        
        # 定义常用颜色
        self.PRIMARY = self.colors.get('primary', '#2E86AB')
        self.SECONDARY = self.colors.get('secondary', '#A23B72')
        self.ACCENT = self.colors.get('accent', '#F18F01')
        self.SUCCESS = self.colors.get('success', '#06A77D')
        self.WARNING = self.colors.get('warning', '#F77F00')
        self.ERROR = self.colors.get('error', '#D62828')
        
        # 轨迹颜色
        traj_colors = self.colors.get('trajectory', {})
        self.COLOR_REAL = traj_colors.get('real', '#2E86AB')
        self.COLOR_PREDICTED = traj_colors.get('predicted', '#F18F01')
        self.COLOR_SYNTHETIC = traj_colors.get('synthetic', '#A23B72')
        
        # 地形颜色
        terrain_colors = self.colors.get('terrain', {})
        self.COLOR_WATER = terrain_colors.get('water', '#4A90E2')
        self.COLOR_FOREST = terrain_colors.get('forest', '#2D5016')
        self.COLOR_GRASSLAND = terrain_colors.get('grassland', '#8BC34A')
        self.COLOR_URBAN = terrain_colors.get('urban', '#9E9E9E')
        
        # LULC 颜色映射
        self.LULC_COLORS = {
            10: self.COLOR_FOREST,      # Tree cover
            20: '#6B8E23',               # Shrubland
            30: self.COLOR_GRASSLAND,    # Grassland
            40: '#FFD700',               # Cropland
            50: self.COLOR_URBAN,        # Built-up
            60: '#D2B48C',               # Bare/sparse vegetation
            80: self.COLOR_WATER,        # Water
            90: '#20B2AA',               # Wetland
            100: '#8FBC8F',              # Moss and lichen
        }
    
    def setup_fonts(self):
        """设置字体"""
        font_config = cfg.get('plotting.font', {})
        
        rcParams['font.family'] = font_config.get('family', 'DejaVu Sans')
        rcParams['font.size'] = font_config.get('size', 12)
        rcParams['axes.titlesize'] = font_config.get('title_size', 14)
        rcParams['axes.labelsize'] = font_config.get('label_size', 11)
        rcParams['legend.fontsize'] = font_config.get('legend_size', 10)
        rcParams['xtick.labelsize'] = font_config.get('label_size', 11)
        rcParams['ytick.labelsize'] = font_config.get('label_size', 11)
        
        # 支持中文显示 - 使用Noto Sans CJK字体
        chinese_font_cfg = cfg.get('plotting.chinese_font', {})
        enabled = chinese_font_cfg.get('enabled', True)
        families = chinese_font_cfg.get('families')
        if not families:
            family = chinese_font_cfg.get('family')
            families = [family] if family else []
        fallback_families = chinese_font_cfg.get(
            'fallback_families',
            ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', 'AR PL UKai CN', 'AR PL UMing CN', 'Droid Sans Fallback', 'DejaVu Sans', 'Arial']
        )
        sans_serif_list = [f for f in (families + fallback_families) if f]
        if enabled and sans_serif_list:
            rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = sans_serif_list if sans_serif_list else ['DejaVu Sans', 'Arial']
        rcParams['axes.unicode_minus'] = False
        
        # 加载中文字体
        from matplotlib.font_manager import FontProperties
        from matplotlib import font_manager
        font_paths = chinese_font_cfg.get('paths')
        if not font_paths:
            font_path = chinese_font_cfg.get('path')
            font_paths = [font_path] if font_path else []
        # 常见字体路径兜底
        font_paths = [p for p in font_paths if p] + [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
            '/usr/share/fonts/truetype/arphic/ukai.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc',
        ]
        selected_path = None
        if enabled:
            for p in font_paths:
                try:
                    if Path(p).exists():
                        font_manager.fontManager.addfont(p)
                        if selected_path is None:
                            selected_path = p
                except Exception:
                    continue
        self.chinese_font = FontProperties(fname=selected_path) if selected_path else None
    
    def setup_figure(self):
        """设置图形参数"""
        fig_config = cfg.get('plotting.figure', {})
        layout_config = cfg.get('plotting.layout', {})
        traj_plot_config = cfg.get('plotting.trajectory_plot', {})
        
        self.DPI = fig_config.get('dpi', 300)
        self.FORMAT = fig_config.get('format', 'png')
        self.DEFAULT_SIZE = tuple(fig_config.get('default_size', [10, 6]))
        self.LARGE_SIZE = tuple(fig_config.get('large_size', [14, 8]))
        self.SMALL_SIZE = tuple(fig_config.get('small_size', [6, 4]))

        self.SQUARE_PANELS = bool(layout_config.get('square_panels', True))
        self.PANEL_SIZE_INCH = float(layout_config.get('panel_size_inch', 4.5))

        self.TRAJ_PLOT = traj_plot_config
        
        # 设置默认 DPI
        rcParams['figure.dpi'] = self.DPI
        rcParams['savefig.dpi'] = self.DPI
        rcParams['savefig.format'] = self.FORMAT
        rcParams['savefig.bbox'] = 'tight'
        rcParams['savefig.pad_inches'] = 0.1
        
        # 设置网格
        rcParams['grid.alpha'] = 0.3
        rcParams['grid.linewidth'] = 0.5
        
        # 设置线条
        rcParams['lines.linewidth'] = 2
        rcParams['lines.markersize'] = 6
        
        # 设置图例
        rcParams['legend.frameon'] = True
        rcParams['legend.framealpha'] = 0.8
        rcParams['legend.fancybox'] = True
    
    def create_figure(self, size: str = 'default', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        创建标准化的图形
        
        Args:
            size: 'default', 'large', 'small' 或自定义 (width, height)
            **kwargs: 传递给 plt.subplots 的其他参数
        
        Returns:
            fig, ax
        """
        if size == 'default':
            figsize = self.DEFAULT_SIZE
        elif size == 'large':
            figsize = self.LARGE_SIZE
        elif size == 'small':
            figsize = self.SMALL_SIZE
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            figsize = size
        else:
            figsize = self.DEFAULT_SIZE
        
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        return fig, ax
    
    def save_figure(self, fig: plt.Figure, filename: str, subdir: str = ''):
        """
        保存图形到标准路径
        
        Args:
            fig: matplotlib figure 对象
            filename: 文件名（不含扩展名）
            subdir: 子目录名
        """
        output_dir = Path(cfg.get('paths.outputs.figures'))
        if subdir:
            output_dir = output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / f"{filename}.{self.FORMAT}"
        fig.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        print(f"图形已保存: {filepath}")
    
    def get_lulc_cmap(self):
        """获取 LULC 颜色映射"""
        from matplotlib.colors import ListedColormap
        
        lulc_classes = sorted(self.LULC_COLORS.keys())
        colors = [self.LULC_COLORS[c] for c in lulc_classes]
        return ListedColormap(colors), lulc_classes
    
    def get_terrain_cmap(self, name: str = 'terrain'):
        """获取地形颜色映射"""
        if name == 'terrain':
            return plt.cm.terrain
        elif name == 'elevation':
            return plt.cm.gist_earth
        elif name == 'slope':
            return plt.cm.YlOrRd
        else:
            return plt.cm.viridis
    
    def style_axis(self, ax: plt.Axes, 
                   title: Optional[str] = None,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   grid: bool = True,
                   legend: bool = False):
        """
        统一设置坐标轴样式
        
        Args:
            ax: matplotlib axes 对象
            title: 标题
            xlabel: x 轴标签
            ylabel: y 轴标签
            grid: 是否显示网格
            legend: 是否显示图例
        """
        if title:
            ax.set_title(title, fontweight='bold', pad=10)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        if legend:
            ax.legend(frameon=True, framealpha=0.8, loc='best')
        
        # 设置脊柱样式
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)


# 全局绘图配置实例
plot_cfg = PlotConfig()


def get_plot_config() -> PlotConfig:
    """获取全局绘图配置实例"""
    return plot_cfg


# 便捷函数
def create_figure(size: str = 'default', **kwargs):
    """创建标准化图形"""
    return plot_cfg.create_figure(size, **kwargs)


def save_figure(fig: plt.Figure, filename: str, subdir: str = ''):
    """保存图形"""
    plot_cfg.save_figure(fig, filename, subdir)


def style_axis(ax: plt.Axes, **kwargs):
    """设置坐标轴样式"""
    plot_cfg.style_axis(ax, **kwargs)
