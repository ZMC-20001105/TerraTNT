#!/bin/bash
# TerraTNT 依赖安装脚本

set -e

echo "============================================================"
echo "TerraTNT 依赖安装脚本"
echo "============================================================"

# 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "错误: 请先激活 conda 环境"
    echo "运行: conda activate trajectory-prediction"
    exit 1
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 安装 PyQt6
echo "📦 安装 PyQt6..."
pip install PyQt6 PyQt6-Qt6

# 安装其他依赖
echo "📦 安装其他依赖..."
pip install -r requirements.txt

echo ""
echo "✅ 依赖安装完成！"
echo ""
echo "运行以下命令启动系统:"
echo "  python main.py"
