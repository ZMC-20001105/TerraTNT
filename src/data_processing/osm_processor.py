"""
OpenStreetMap数据处理器
"""
import os
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import osmnx as ox
from shapely.geometry import Point, LineString, Polygon
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from ..utils.coordinate_transform import CoordinateTransformer
from ..utils.config_loader import config

logger = logging.getLogger(__name__)


class OSMProcessor:
    """OpenStreetMap数据处理器"""
    
    def __init__(self, transformer: CoordinateTransformer = None):
        """
        初始化OSM处理器
        
        Args:
            transformer: 坐标转换器
        """
        self.transformer = transformer or CoordinateTransformer()
        self.data_config = config.get_data_config()
        
        # OSM道路类型权重配置
        self.road_weights = {
            'motorway': 0.3,      # 高速公路
            'trunk': 0.4,         # 国道
            'primary': 0.5,       # 省道
            'secondary': 0.6,     # 县道
            'tertiary': 0.7,      # 乡道
            'residential': 0.8,   # 住宅区道路
            'service': 0.9,       # 服务道路
            'track': 1.2,         # 土路
            'path': 1.5,          # 小径
            'footway': 2.0,       # 人行道
            'cycleway': 1.8,      # 自行车道
        }
    
    def load_osm_data(self, region_name: str) -> Optional[gpd.GeoDataFrame]:
        """
        加载OSM数据文件
        
        Args:
            region_name: 区域名称（如 'germany', 'austria' 等）
            
        Returns:
            GeoDataFrame或None
        """
        osm_file = Path(self.data_config['osm_dir']) / f"{region_name}.osm.pbf"
        
        if not osm_file.exists():
            logger.warning(f"OSM文件不存在: {osm_file}")
            return None
        
        try:
            # 使用osmnx读取OSM数据
            logger.info(f"加载OSM数据: {osm_file}")
            
            # 这里需要先将PBF文件转换为可读格式
            # 实际项目中可能需要使用osmium或其他工具预处理
            # 暂时返回None，后续实现具体的PBF读取逻辑
            logger.warning("PBF文件读取功能待实现")
            return None
            
        except Exception as e:
            logger.error(f"加载OSM数据失败: {e}")
            return None
    
    def extract_road_network(self, bounds: Tuple[float, float, float, float], 
                           network_type: str = 'drive') -> gpd.GeoDataFrame:
        """
        从OSM提取道路网络
        
        Args:
            bounds: 边界范围 (min_lon, min_lat, max_lon, max_lat)
            network_type: 网络类型 ('drive', 'walk', 'bike', 'all')
            
        Returns:
            道路网络GeoDataFrame
        """
        try:
            logger.info(f"从OSM提取道路网络: {bounds}")
            
            # 使用osmnx从OSM API获取道路网络
            G = ox.graph_from_bbox(
                bounds[3], bounds[1], bounds[2], bounds[0],  # north, south, east, west
                network_type=network_type,
                simplify=True,
                retain_all=False
            )
            
            # 转换为GeoDataFrame
            edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
            
            # 坐标转换
            if self.transformer:
                edges_gdf = edges_gdf.to_crs(self.transformer.target_crs.to_string())
            
            # 添加道路权重
            edges_gdf['road_weight'] = edges_gdf['highway'].apply(self._get_road_weight)
            
            logger.info(f"提取到 {len(edges_gdf)} 条道路")
            return edges_gdf
            
        except Exception as e:
            logger.error(f"提取道路网络失败: {e}")
            return gpd.GeoDataFrame()
    
    def _get_road_weight(self, highway_type) -> float:
        """
        获取道路类型对应的权重
        
        Args:
            highway_type: 道路类型
            
        Returns:
            权重值
        """
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else 'residential'
        
        return self.road_weights.get(highway_type, 1.0)
    
    def create_road_raster(self, roads_gdf: gpd.GeoDataFrame, 
                          bounds: Tuple[float, float, float, float],
                          resolution: float = 30) -> np.ndarray:
        """
        创建道路栅格图
        
        Args:
            roads_gdf: 道路GeoDataFrame
            bounds: 边界范围 (min_x, min_y, max_x, max_y) - 投影坐标
            resolution: 栅格分辨率（米）
            
        Returns:
            道路栅格数组
        """
        min_x, min_y, max_x, max_y = bounds
        
        # 计算栅格尺寸
        width = int((max_x - min_x) / resolution)
        height = int((max_y - min_y) / resolution)
        
        # 创建仿射变换
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        
        # 准备栅格化数据
        shapes = []
        for idx, row in roads_gdf.iterrows():
            if row.geometry.geom_type == 'LineString':
                # 为线要素创建缓冲区
                buffered = row.geometry.buffer(resolution / 2)
                shapes.append((buffered, row['road_weight']))
        
        # 栅格化
        if shapes:
            road_raster = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,  # 无道路区域填充值
                default_value=1,  # 默认道路权重
                dtype=np.float32
            )
        else:
            road_raster = np.zeros((height, width), dtype=np.float32)
        
        logger.info(f"创建道路栅格: {width}x{height}, 分辨率: {resolution}m")
        return road_raster
    
    def extract_poi(self, bounds: Tuple[float, float, float, float], 
                   poi_types: List[str] = None) -> gpd.GeoDataFrame:
        """
        提取兴趣点(POI)
        
        Args:
            bounds: 边界范围 (min_lon, min_lat, max_lon, max_lat)
            poi_types: POI类型列表
            
        Returns:
            POI GeoDataFrame
        """
        if poi_types is None:
            poi_types = ['amenity', 'shop', 'tourism', 'military']
        
        try:
            logger.info(f"提取POI: {bounds}")
            
            pois = []
            for poi_type in poi_types:
                try:
                    poi_gdf = ox.geometries_from_bbox(
                        bounds[3], bounds[1], bounds[2], bounds[0],  # north, south, east, west
                        tags={poi_type: True}
                    )
                    
                    if not poi_gdf.empty:
                        poi_gdf['poi_type'] = poi_type
                        pois.append(poi_gdf)
                        
                except Exception as e:
                    logger.warning(f"提取POI类型 {poi_type} 失败: {e}")
                    continue
            
            if pois:
                all_pois = pd.concat(pois, ignore_index=True)
                
                # 坐标转换
                if self.transformer:
                    all_pois = all_pois.to_crs(self.transformer.target_crs.to_string())
                
                logger.info(f"提取到 {len(all_pois)} 个POI")
                return all_pois
            else:
                return gpd.GeoDataFrame()
                
        except Exception as e:
            logger.error(f"提取POI失败: {e}")
            return gpd.GeoDataFrame()
    
    def create_settlement_raster(self, bounds: Tuple[float, float, float, float],
                               resolution: float = 30) -> np.ndarray:
        """
        创建居民点栅格图
        
        Args:
            bounds: 边界范围 (min_lon, min_lat, max_lon, max_lat)
            resolution: 栅格分辨率（米）
            
        Returns:
            居民点栅格数组
        """
        try:
            # 提取建筑物和居民点
            buildings = ox.geometries_from_bbox(
                bounds[3], bounds[1], bounds[2], bounds[0],
                tags={'building': True}
            )
            
            if buildings.empty:
                # 如果没有建筑物数据，返回空栅格
                min_x, min_y, max_x, max_y = self.transformer.transform_points(
                    [bounds[0], bounds[2]], [bounds[1], bounds[3]]
                )
                width = int((max_x - min_x) / resolution)
                height = int((max_y - min_y) / resolution)
                return np.zeros((height, width), dtype=np.float32)
            
            # 坐标转换
            buildings = buildings.to_crs(self.transformer.target_crs.to_string())
            
            # 获取投影边界
            min_x, min_y, max_x, max_y = self.transformer.transform_points(
                [bounds[0], bounds[2]], [bounds[1], bounds[3]]
            )
            
            # 创建栅格
            width = int((max_x - min_x) / resolution)
            height = int((max_y - min_y) / resolution)
            transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
            
            # 栅格化建筑物
            shapes = [(geom, 1) for geom in buildings.geometry if geom.is_valid]
            
            if shapes:
                settlement_raster = rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.float32
                )
            else:
                settlement_raster = np.zeros((height, width), dtype=np.float32)
            
            logger.info(f"创建居民点栅格: {width}x{height}")
            return settlement_raster
            
        except Exception as e:
            logger.error(f"创建居民点栅格失败: {e}")
            # 返回空栅格
            min_x, min_y, max_x, max_y = self.transformer.transform_points(
                [bounds[0], bounds[2]], [bounds[1], bounds[3]]
            )
            width = int((max_x - min_x) / resolution)
            height = int((max_y - min_y) / resolution)
            return np.zeros((height, width), dtype=np.float32)
    
    def save_processed_data(self, data: gpd.GeoDataFrame, output_path: str):
        """
        保存处理后的数据
        
        Args:
            data: 要保存的GeoDataFrame
            output_path: 输出路径
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 根据文件扩展名选择保存格式
            if output_path.suffix.lower() == '.geojson':
                data.to_file(output_path, driver='GeoJSON')
            elif output_path.suffix.lower() == '.shp':
                data.to_file(output_path, driver='ESRI Shapefile')
            else:
                # 默认保存为GeoJSON
                output_path = output_path.with_suffix('.geojson')
                data.to_file(output_path, driver='GeoJSON')
            
            logger.info(f"数据已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")


def process_region_osm(region_name: str, bounds: Tuple[float, float, float, float],
                      output_dir: str = None) -> Dict[str, np.ndarray]:
    """
    处理指定区域的OSM数据
    
    Args:
        region_name: 区域名称
        bounds: 边界范围 (min_lon, min_lat, max_lon, max_lat)
        output_dir: 输出目录
        
    Returns:
        包含各种栅格数据的字典
    """
    if output_dir is None:
        output_dir = config.get('data.output_dir', 'data/processed')
    
    processor = OSMProcessor()
    
    # 提取道路网络
    roads = processor.extract_road_network(bounds)
    
    # 转换边界到投影坐标
    transformer = processor.transformer
    min_x, min_y = transformer.transform_point(bounds[0], bounds[1])
    max_x, max_y = transformer.transform_point(bounds[2], bounds[3])
    proj_bounds = (min_x, min_y, max_x, max_y)
    
    # 创建道路栅格
    road_raster = processor.create_road_raster(roads, proj_bounds)
    
    # 创建居民点栅格
    settlement_raster = processor.create_settlement_raster(bounds)
    
    # 保存结果
    output_path = Path(output_dir) / region_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存道路网络
    if not roads.empty:
        processor.save_processed_data(roads, output_path / 'roads.geojson')
    
    # 保存栅格数据
    np.save(output_path / 'road_raster.npy', road_raster)
    np.save(output_path / 'settlement_raster.npy', settlement_raster)
    
    logger.info(f"区域 {region_name} OSM数据处理完成")
    
    return {
        'road_raster': road_raster,
        'settlement_raster': settlement_raster,
        'roads_vector': roads
    }
