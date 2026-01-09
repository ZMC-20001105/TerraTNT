# 项目文件清理报告

## 📊 当前文件结构分析

### 1. Scripts目录 (33个Python文件)

#### ✅ 保留的核心脚本 (7个)
- `generate_dataset_parallel.py` - 并行轨迹生成 (核心)
- `generate_synthetic_dataset.py` - 单进程轨迹生成 (备用)
- `download_osm_data.py` - OSM数据下载
- `process_bohemian_forest.py` - Bohemian Forest数据处理
- `prepare_bohemian_forest.py` - Bohemian Forest数据准备
- `generate_slope_aspect.py` - 地形数据生成
- `extract_oord_archives.py` - OORD数据解压

#### ❌ 冗余的GEE下载脚本 (12个) - 功能重复
- `gee_data_downloader.py`
- `gee_downloader_robust.py`
- `gee_tiled_download.py`
- `gee_chunked_download.py`
- `gee_drive_export.py`
- `gee_export_to_gcs.py`
- `gee_with_proxy.py`
- `direct_download_gee_data.py`
- `download_new_regions.py`
- `download_new_regions_no_proxy.py`
- `run_all_downloads_with_proxy.py`
- `check_gee_tasks.py`

#### ❌ 冗余的GEE设置脚本 (3个)
- `setup_gee.py`
- `setup_gee_server.py`
- `setup_gee_simple.py`

#### ❌ 冗余的测试脚本 (4个)
- `test_gee_connection.py`
- `test_gee_download.py`
- `test_trajectory_generation.py`
- `test_complete_pipeline.py`
- `test_config_system.py`

#### ❌ 冗余的数据处理脚本 (3个)
- `process_offline_data.py` - 已被新脚本替代
- `process_new_regions.py` - 已被新脚本替代
- `download_from_drive.py` - 不再使用

#### ❌ 冗余的OORD下载脚本 (2个)
- `download_oord_dataset.py` - 已完成下载
- `download_oord_gps.py` - 已完成下载

### 2. 数据目录

#### ✅ 保留的数据 (必需)
- `data/processed/utm_grid/` - 1.8GB (UTM投影的环境数据)
- `data/processed/synthetic_trajectories/scottish_highlands/` - 125MB (生成中)
- `data/processed/synthetic_trajectories/bohemian_forest/` - 18MB (生成中)
- `data/osm/*.osm.pbf` - 3.0GB (OSM道路数据)
- `data/oord_extracted/` - 722MB (真实轨迹数据)
- `data/processed/speed_training/` - 2.5MB (速度模型训练数据)

#### ❌ 可删除的数据 (冗余/临时)
- `data/processed/synthetic_trajectories/test/` - 888KB (测试数据)
- `data/processed/synthetic_trajectories/test_complete/` - 3.9MB (测试数据)
- `data/raw/gee/bohemian_forest_*_tiles/` - 22MB (已合并的tiles)
- `data/raw/gee/temp/` - 19MB (临时文件)
- `data/processed/merged_gee/` - 334MB (已投影到UTM，可删除)
- `data/oord/*.zip` - 257MB (已解压的压缩包)
- `data/processed/trajectories/` - 7.4MB (旧版本轨迹数据)

#### ⚠️ 谨慎处理
- `data/raw/gee/scottish_highlands/` - 227MB (原始GEE数据，已合并但保留备份)

### 3. 其他冗余文件
- `venv/` - Python虚拟环境 (如果使用conda，可删除)

## 💾 清理后预计释放空间

| 类别 | 大小 | 说明 |
|------|------|------|
| 冗余脚本 | ~100KB | 24个Python文件 |
| 测试数据 | ~5MB | test/test_complete目录 |
| GEE tiles | ~41MB | 已合并的原始tiles |
| 临时文件 | ~19MB | temp目录 |
| 已合并数据 | ~334MB | merged_gee目录 |
| OORD压缩包 | ~257MB | 已解压的zip文件 |
| 旧版轨迹 | ~7.4MB | 旧版本数据 |
| **总计** | **~664MB** | |

## 🎯 清理建议

### 立即删除 (安全)
1. 测试数据和临时文件
2. 冗余的脚本文件
3. 已合并的GEE tiles
4. OORD压缩包

### 可选删除 (释放更多空间)
1. `data/processed/merged_gee/` - 已有UTM版本
2. `data/raw/gee/scottish_highlands/` - 已合并，保留备份可选

### 不建议删除
1. `data/processed/utm_grid/` - 核心环境数据
2. `data/osm/*.osm.pbf` - OSM道路数据
3. `data/oord_extracted/` - 真实轨迹数据
4. 正在生成的轨迹数据
