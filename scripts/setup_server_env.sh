#!/bin/bash

# 服务器端环境设置脚本
# 适用于GPU服务器的Anaconda环境配置

set -e  # 遇到错误立即退出

echo "🚀 开始设置服务器端深度学习环境"
echo "=================================="

# 检查是否已安装Anaconda/Miniconda
if ! command -v conda &> /dev/null; then
    echo "❌ 未检测到Conda，开始安装Miniconda..."
    
    # 下载Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # 安装Miniconda
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # 初始化conda
    $HOME/miniconda3/bin/conda init bash
    
    # 重新加载bashrc
    source ~/.bashrc
    
    echo "✅ Miniconda安装完成"
else
    echo "✅ 检测到Conda环境"
fi

# 检查CUDA版本
echo "🔍 检查CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ CUDA环境正常"
else
    echo "⚠️  未检测到NVIDIA GPU"
fi

# 创建conda环境
echo "📦 创建项目conda环境..."
if conda env list | grep -q "trajectory-prediction"; then
    echo "环境已存在，是否重新创建？(y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n trajectory-prediction -y
        conda env create -f environment.yml
    fi
else
    conda env create -f environment.yml
fi

echo "✅ Conda环境创建完成"

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate trajectory-prediction

# 验证PyTorch CUDA支持
echo "🧪 验证PyTorch CUDA支持..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
"

# 设置Google Earth Engine
echo "🌍 设置Google Earth Engine..."
echo "注意：GEE认证需要浏览器，在服务器环境下我们将使用服务账号方式"

# 创建GEE配置目录
mkdir -p ~/.config/earthengine

echo "📋 环境设置完成！"
echo "=================================="
echo "下一步操作："
echo "1. 激活环境: conda activate trajectory-prediction"
echo "2. 配置GEE认证: python scripts/setup_gee_server.py"
echo "3. 下载数据: python scripts/gee_data_downloader.py"
echo "=================================="
