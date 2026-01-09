#!/bin/bash
# 项目清理脚本
# 删除旧的/错误的模型checkpoints和临时文件
# 节省约1.1GB空间

set -e

echo "=========================================="
echo "TerraTNT 项目清理"
echo "=========================================="

cd /home/zmc/文档/programwork

# 统计清理前的大小
echo ""
echo "清理前大小:"
du -sh runs/ logs/ scripts/

# 1. 删除旧的模型checkpoints
echo ""
echo "【1. 清理旧模型checkpoints】"

# 23秒间隔的旧版本
if [ -d "runs/terratnt_fas1" ]; then
    echo "  删除: terratnt_fas1 (23秒版本)"
    rm -rf runs/terratnt_fas1
fi

if [ -d "runs/terratnt_fas2" ]; then
    echo "  删除: terratnt_fas2 (23秒版本)"
    rm -rf runs/terratnt_fas2
fi

if [ -d "runs/terratnt_fas3" ]; then
    echo "  删除: terratnt_fas3 (23秒版本)"
    rm -rf runs/terratnt_fas3
fi

# optimized版本
for dir in runs/terratnt_fas*_optimized; do
    if [ -d "$dir" ]; then
        echo "  删除: $(basename $dir)"
        rm -rf "$dir"
    fi
done

# fixed版本
for dir in runs/terratnt_fas*_fixed; do
    if [ -d "$dir" ]; then
        echo "  删除: $(basename $dir)"
        rm -rf "$dir"
    fi
done

# real_env旧版本（保留最新的3个）
if [ -d "runs/terratnt_fas1_real_env" ]; then
    echo "  删除: terratnt_fas1_real_env (旧版本)"
    rm -rf runs/terratnt_fas1_real_env
fi

# 失败的ynet训练
for dir in runs/ynet*; do
    if [ -d "$dir" ]; then
        echo "  删除: $(basename $dir) (训练失败)"
        rm -rf "$dir"
    fi
done

# 旧的pecnet
for dir in runs/pecnet_2026*; do
    if [ -d "$dir" ]; then
        echo "  删除: $(basename $dir) (旧版本)"
        rm -rf "$dir"
    fi
done

# 空目录
if [ -d "runs/constant_velocity" ]; then
    echo "  删除: constant_velocity (空目录)"
    rm -rf runs/constant_velocity
fi

# 2. 删除临时脚本
echo ""
echo "【2. 清理临时脚本】"

if [ -f "scripts/fix_trajectory_speeds.py" ]; then
    echo "  删除: fix_trajectory_speeds.py"
    rm -f scripts/fix_trajectory_speeds.py
fi

if [ -f "scripts/calibrate_xgboost_speeds.py" ]; then
    echo "  删除: calibrate_xgboost_speeds.py"
    rm -f scripts/calibrate_xgboost_speeds.py
fi

if [ -f "scripts/auto_download_garmisch_hohenfels.py" ]; then
    echo "  删除: auto_download_garmisch_hohenfels.py"
    rm -f scripts/auto_download_garmisch_hohenfels.py
fi

if [ -f "scripts/cleanup_redundant_files.py" ]; then
    echo "  删除: cleanup_redundant_files.py"
    rm -f scripts/cleanup_redundant_files.py
fi

if [ -f "scripts/download_oord_gps.py" ]; then
    echo "  删除: download_oord_gps.py (空文件)"
    rm -f scripts/download_oord_gps.py
fi

# 3. 删除超过24小时的日志 (可选)
echo ""
echo "【3. 清理旧日志】"
old_logs=$(find logs/ -name "*.log" -mtime +1 2>/dev/null | wc -l)
if [ "$old_logs" -gt 0 ]; then
    echo "  删除 $old_logs 个超过24小时的日志"
    find logs/ -name "*.log" -mtime +1 -delete
else
    echo "  无需清理（没有旧日志）"
fi

# 统计清理后的大小
echo ""
echo "清理后大小:"
du -sh runs/ logs/ scripts/

echo ""
echo "✓ 清理完成！"
echo ""
echo "保留的模型:"
ls -lh runs/ | grep "^d" | tail -5

echo ""
echo "保留的核心脚本:"
ls scripts/*.py | grep -E "train_terratnt_10s|visualize_results|compare_baselines|evaluate_all"
