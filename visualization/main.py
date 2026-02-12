#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TerraTNT 可视化验证系统 - 入口
用法: python -m visualization.main
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from visualization.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TerraTNT Visualizer")
    app.setStyle("Fusion")

    # 深色主题 Palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Text, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Button, QColor(55, 55, 58))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 100, 100))
    palette.setColor(QPalette.ColorRole.Link, QColor(66, 165, 245))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(41, 121, 255))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # 全局样式表
    app.setStyleSheet("""
        QMainWindow { background: #2d2d30; }
        QTabWidget::pane { border: 1px solid #3f3f46; background: #2d2d30; }
        QTabBar::tab {
            background: #3e3e42; color: #ccc; padding: 6px 18px;
            border: 1px solid #3f3f46; border-bottom: none; border-radius: 3px 3px 0 0;
            margin-right: 2px; min-width: 60px;
        }
        QTabBar::tab:selected { background: #2d2d30; color: #fff; border-bottom: 2px solid #2979ff; }
        QTabBar::tab:hover { background: #505054; }
        QToolBar { background: #333337; border-bottom: 1px solid #3f3f46; spacing: 6px; padding: 3px; }
        QToolBar QToolButton { color: #ccc; padding: 4px 10px; border-radius: 3px; }
        QToolBar QToolButton:hover { background: #505054; }
        QStatusBar { background: #007acc; color: white; font-size: 12px; }
        QStatusBar QLabel { color: white; padding: 0 8px; }
        QMenuBar { background: #333337; color: #ccc; }
        QMenuBar::item:selected { background: #505054; }
        QMenu { background: #2d2d30; color: #ccc; border: 1px solid #3f3f46; }
        QMenu::item:selected { background: #2979ff; }
        QGroupBox {
            font-weight: bold; font-size: 12px; color: #ddd;
            border: 1px solid #3f3f46; border-radius: 4px;
            margin-top: 10px; padding-top: 16px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #42a5f5; }
        QComboBox { background: #3e3e42; border: 1px solid #555; border-radius: 3px; padding: 4px 8px; color: #ddd; }
        QComboBox:hover { border-color: #2979ff; }
        QComboBox::drop-down { border: none; }
        QLineEdit { background: #3e3e42; border: 1px solid #555; border-radius: 3px; padding: 4px 8px; color: #ddd; }
        QLineEdit:focus { border-color: #2979ff; }
        QListWidget { background: #1e1e1e; border: 1px solid #3f3f46; border-radius: 3px; }
        QListWidget::item { padding: 4px 6px; }
        QListWidget::item:selected { background: #2979ff; color: white; }
        QListWidget::item:hover { background: #3e3e42; }
        QTextEdit { background: #1e1e1e; border: 1px solid #3f3f46; color: #ddd; font-family: monospace; }
        QProgressBar { background: #3e3e42; border: none; border-radius: 2px; }
        QProgressBar::chunk { background: #2979ff; border-radius: 2px; }
        QSplitter::handle { background: #3f3f46; }
        QSplitter::handle:horizontal { width: 2px; }
        QSplitter::handle:vertical { height: 2px; }
        QScrollBar:vertical { background: #2d2d30; width: 10px; border: none; }
        QScrollBar::handle:vertical { background: #555; border-radius: 4px; min-height: 20px; }
        QScrollBar::handle:vertical:hover { background: #777; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QTableWidget { background: #1e1e1e; gridline-color: #3f3f46; }
        QHeaderView::section { background: #3e3e42; color: #ddd; padding: 4px; border: 1px solid #3f3f46; }
    """)

    # matplotlib 中文字体
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _cjk = None
    for c in ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Droid Sans Fallback', 'WenQuanYi Micro Hei']:
        if any(f.name == c for f in fm.fontManager.ttflist):
            _cjk = c
            break
    plt.rcParams['font.sans-serif'] = [_cjk, 'DejaVu Sans'] if _cjk else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
