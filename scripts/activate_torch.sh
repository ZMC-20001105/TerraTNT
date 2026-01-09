#!/bin/bash
# PyTorch环境激活脚本
# 使用方法: source scripts/activate_torch.sh

export LD_PRELOAD=/home/zmc/文档/programwork/fix_torch/libittnotify_stub.so
export CONDA_ENV=/home/zmc/miniconda3/envs/torch-clean

echo "✓ PyTorch环境已激活"
echo "  Python: $CONDA_ENV/bin/python"
echo "  LD_PRELOAD: $LD_PRELOAD"
echo ""
echo "验证PyTorch:"
$CONDA_ENV/bin/python -c "import torch; print('  版本:', torch.__version__); print('  CUDA:', torch.cuda.is_available())"
