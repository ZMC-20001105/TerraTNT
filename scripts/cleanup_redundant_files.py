#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理冗余脚本和文件
识别并列出可以删除的重复、过时或测试文件
"""
import os
from pathlib import Path
from collections import defaultdict

# 定义冗余文件模式
REDUNDANT_PATTERNS = {
    'train': [
        'train_terratnt_fixed.py',  # 已有train_terratnt_10s.py
        'train_terratnt_optimized.py',  # 已有train_terratnt_10s.py
        'train_terratnt_fas.py',  # 已有train_terratnt_all_phases.py
        'train_terratnt_real_env.py',  # 测试脚本
        'train_parallel.py',  # 已有train_parallel_v2.py
        'train_now.py',  # 临时脚本
        'quick_train_baseline.py',  # 已有train_all_baselines.py
        'train_single_model.py',  # 已有train_all_baselines.py
    ],
    'visualize': [
        'visualize_results_fixed.py',  # 已有visualize_results.py
        'draw_architecture.py',  # 重复
        'draw_architecture_fixed.py',  # 重复
        'draw_architecture_cn.py',  # 重复
        'draw_architecture_cn_fixed.py',  # 重复
        'draw_architecture_cn_v2.py',  # 重复
        'draw_architecture_en.py',  # 重复
    ],
    'download': [
        'download_new_regions.py',  # 已有download_garmisch_hohenfels.py
        'download_new_regions_no_proxy.py',  # 已有download_garmisch_hohenfels.py
        'direct_download_gee_data.py',  # 已有auto_download_garmisch_hohenfels.py
        'gee_chunked_download.py',  # 已有auto_download_garmisch_hohenfels.py
        'gee_tiled_download.py',  # 已有auto_download_garmisch_hohenfels.py
        'gee_export_to_gcs.py',  # 不使用GCS
    ],
    'test': [
        'test_complete_pipeline.py',  # 测试脚本
        'test_config_system.py',  # 测试脚本
        'test_models.py',  # 测试脚本
        'test_trajectory_generation.py',  # 测试脚本
        'test_ui_fonts.py',  # 测试脚本
    ],
    'ui': [
        # UI相关的所有文件都可以删除（不需要GUI）
    ],
    'docs': [
        'CLEANUP_REPORT.md',  # 临时报告
        'PROGRESS_REPORT.md',  # 临时报告
        'PROJECT_STATUS.md',  # 临时报告
        'README_SYSTEM.md',  # 临时文档
    ]
}

# 保留的核心脚本
KEEP_SCRIPTS = {
    'train_terratnt_10s.py',  # 核心训练脚本
    'train_terratnt_all_phases.py',  # 多阶段训练
    'train_all_baselines.py',  # 基线模型训练
    'train_parallel_v2.py',  # 并行训练
    'evaluate_all_models.py',  # 评估脚本
    'compare_baselines.py',  # 对比脚本
    'visualize_results.py',  # 可视化脚本
    'download_garmisch_hohenfels.py',  # GEE下载
    'auto_download_garmisch_hohenfels.py',  # 自动下载
    'download_osm_data.py',  # OSM下载
    'generate_synthetic_dataset.py',  # 数据生成
    'prepare_fas_datasets.py',  # 数据准备
    'chapter3_experiments.py',  # 第三章实验
}

def analyze_redundancy():
    """分析冗余文件"""
    base_dir = Path('/home/zmc/文档/programwork')
    
    print("="*60)
    print("冗余文件分析报告")
    print("="*60)
    
    to_delete = []
    
    # 检查scripts目录
    scripts_dir = base_dir / 'scripts'
    for category, files in REDUNDANT_PATTERNS.items():
        if files:
            print(f"\n【{category}】类别:")
            for filename in files:
                filepath = scripts_dir / filename
                if filepath.exists():
                    size = filepath.stat().st_size / 1024
                    print(f"  - {filename} ({size:.1f}KB)")
                    to_delete.append(str(filepath))
    
    # 检查UI目录（整个目录可删除）
    ui_dir = base_dir / 'ui'
    if ui_dir.exists():
        ui_files = list(ui_dir.glob('*.py'))
        if ui_files:
            print(f"\n【UI】目录 (可完全删除):")
            for f in ui_files:
                size = f.stat().st_size / 1024
                print(f"  - {f.name} ({size:.1f}KB)")
                to_delete.append(str(f))
    
    # 检查GUI目录
    gui_dir = base_dir / 'gui'
    if gui_dir.exists():
        print(f"\n【GUI】目录 (可完全删除)")
        to_delete.append(str(gui_dir))
    
    # 检查临时文档
    for doc in REDUNDANT_PATTERNS['docs']:
        filepath = base_dir / doc
        if filepath.exists():
            size = filepath.stat().st_size / 1024
            print(f"\n【文档】{doc} ({size:.1f}KB)")
            to_delete.append(str(filepath))
    
    # 统计
    print(f"\n{'='*60}")
    print(f"总计: {len(to_delete)} 个文件/目录可删除")
    print(f"{'='*60}")
    
    # 生成删除脚本
    delete_script = base_dir / 'scripts' / 'delete_redundant.sh'
    with open(delete_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# 删除冗余文件脚本\n")
        f.write("# 执行前请仔细检查\n\n")
        for path in to_delete:
            if Path(path).is_dir():
                f.write(f"rm -rf '{path}'\n")
            else:
                f.write(f"rm -f '{path}'\n")
    
    delete_script.chmod(0o755)
    print(f"\n删除脚本已生成: {delete_script}")
    print("请检查后执行: bash scripts/delete_redundant.sh")
    
    return to_delete

if __name__ == '__main__':
    analyze_redundancy()
