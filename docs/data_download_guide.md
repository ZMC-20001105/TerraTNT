# 数据下载指南

由于服务器网络限制，需要手动下载数据。以下是详细的下载指南：

## 🌍 需要下载的数据

### 1. SRTM DEM数据 (30m分辨率)

**下载地址**：
- NASA官网：https://earthdata.nasa.gov/
- USGS Earth Explorer：https://earthexplorer.usgs.gov/

**下载区域**：
- 波西米亚森林：12.5°E-14.0°E, 48.5°N-49.5°N
- 顿巴斯：37.0°E-39.5°E, 47.5°N-49.0°N  
- 喀尔巴阡山：23.0°E-26.0°E, 45.0°N-47.5°N
- 苏格兰高地：5.5°W-3.5°W, 56.5°N-58.5°N

**文件命名**：
- `bohemian_forest_dem.tif`
- `donbas_dem.tif`
- `carpathians_dem.tif`
- `scottish_highlands_dem.tif`

### 2. 土地覆盖数据

**ESA WorldCover (推荐)**：
- 下载地址：https://worldcover2021.esa.int/
- 分辨率：10m
- 年份：2021

**Copernicus Land Cover (备选)**：
- 下载地址：https://land.copernicus.eu/
- 分辨率：100m

**文件命名**：
- `bohemian_forest_lulc.tif`
- `donbas_lulc.tif`
- `carpathians_lulc.tif`
- `scottish_highlands_lulc.tif`

### 3. OSM道路数据

**下载方式1：Overpass API**
```bash
# 使用curl下载（需要代理）
curl -o bohemian_forest_roads.osm "https://overpass-api.de/api/interpreter?data=[out:xml][timeout:300];(way[highway](bbox:12.5,48.5,14.0,49.5););out geom;"
```

**下载方式2：Geofabrik**
- 网址：https://download.geofabrik.de/
- 下载对应国家/地区的OSM数据

**文件命名**：
- `bohemian_forest_roads.osm` 或 `.pbf`
- `donbas_roads.osm` 或 `.pbf`
- `carpathians_roads.osm` 或 `.pbf`
- `scottish_highlands_roads.osm` 或 `.pbf`

## 📁 目录结构

请将下载的数据放在以下目录：

```
data/
├── raw/
│   ├── dem/
│   │   ├── bohemian_forest_dem.tif
│   │   ├── donbas_dem.tif
│   │   ├── carpathians_dem.tif
│   │   └── scottish_highlands_dem.tif
│   ├── lulc/
│   │   ├── bohemian_forest_lulc.tif
│   │   ├── donbas_lulc.tif
│   │   ├── carpathians_lulc.tif
│   │   └── scottish_highlands_lulc.tif
│   └── osm/
│       ├── bohemian_forest_roads.osm
│       ├── donbas_roads.osm
│       ├── carpathians_roads.osm
│       └── scottish_highlands_roads.osm
└── processed/
    └── (处理后的数据将保存在这里)
```

## 🔧 数据处理

数据下载完成后，运行以下命令进行处理：

```bash
# 激活环境
conda activate trajectory-prediction

# 处理所有数据
python scripts/process_offline_data.py

# 或分别处理
python scripts/process_dem_data.py
python scripts/process_lulc_data.py
python scripts/process_osm_data.py
```

## 📋 检查清单

- [ ] 下载所有4个区域的DEM数据
- [ ] 下载所有4个区域的LULC数据  
- [ ] 下载所有4个区域的OSM道路数据
- [ ] 创建正确的目录结构
- [ ] 运行数据处理脚本
- [ ] 验证处理结果

## 💡 替代方案

如果无法下载某些数据：

1. **使用公开数据集**：
   - Natural Earth：https://www.naturalearthdata.com/
   - OpenTopography：https://www.opentopography.org/

2. **简化研究区域**：
   - 先处理1-2个区域
   - 使用较低分辨率数据

3. **模拟数据**：
   - 生成合成地形数据
   - 使用简化的环境模型

## 🆘 需要帮助？

如果在数据下载过程中遇到问题：

1. 检查网络连接和代理设置
2. 尝试使用不同的下载源
3. 联系数据提供方获取帮助
4. 考虑使用替代数据源
