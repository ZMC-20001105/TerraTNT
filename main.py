"""
TerraTNT 主程序入口
多星协同观测任务规划系统
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow
from config import cfg
import logging


def setup_logging():
    """配置日志系统"""
    log_config = cfg.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')
    
    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format
    )
    
    # 文件日志
    if log_config.get('file', {}).get('enabled', True):
        from logging.handlers import RotatingFileHandler
        log_dir = Path(cfg.get('paths.outputs.logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_dir / 'terratnt.log',
            maxBytes=log_config.get('file', {}).get('max_bytes', 10485760),
            backupCount=log_config.get('file', {}).get('backup_count', 5)
        )
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"TerraTNT v{cfg.get('project.version', '1.0.0')} 启动")
    logger.info("=" * 60)


def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 创建 Qt 应用
        app = QApplication(sys.argv)
        app.setApplicationName(cfg.get('project.name', 'TerraTNT'))
        app.setApplicationVersion(cfg.get('project.version', '1.0.0'))
        
        # 设置应用样式
        theme = cfg.get('gui.theme.name', 'Fusion')
        app.setStyle(theme)
        
        # 启用高DPI支持
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
        
        # 创建主窗口
        logger.info("初始化主窗口...")
        main_window = MainWindow()
        main_window.show()
        
        logger.info("系统就绪")
        
        # 运行应用
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"程序异常退出: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
