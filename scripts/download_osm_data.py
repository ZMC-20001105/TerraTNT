"""
OSM道路数据下载脚本
使用OSMnx从OpenStreetMap获取道路网络数据
"""
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import sys
import os
from typing import Dict

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.coordinate_transform import CoordinateTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSMDataDownloader:
    """OSM数据下载器"""
    
    def __init__(self, output_dir: str = "data/osm"):
        """初始化下载器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 研究区域
        self.regions = {
            'bohemian_forest': {
                'name': '波西米亚森林',
                'bounds': [12.5, 48.5, 14.0, 49.5],  # [min_lon, min_lat, max_lon, max_lat]
                'description': '捷克-德国-奥地利边境'
            },
            'donbas': {
                'name': '顿巴斯',
                'bounds': [37.0, 47.5, 39.5, 49.0],
                'description': '乌克兰东部'
            },
            'carpathians': {
                'name': '喀尔巴阡山',
                'bounds': [23.0, 45.0, 26.0, 47.5],
                'description': '罗马尼亚中部'
            },
            'scottish_highlands': {
                'name': '苏格兰高地',
                'bounds': [-5.5, 56.5, -3.5, 58.5],
                'description': '英国苏格兰北部'
            }
        }
        
        # 配置OSMnx
        ox.settings.log_console = True
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(self.output_dir / "cache")
    
    def download_road_network(self, region_name: str, network_type: str = 'drive') -> bool:
        """
        下载道路网络数据
        
        Args:
            region_name: 区域名称
            network_type: 网络类型 ('drive', 'walk', 'bike', 'all')
            
        Returns:
            是否成功
        """
        if region_name not in self.regions:
            logger.error(f"未知区域: {region_name}")
            return False
        
        region_info = self.regions[region_name]
        bounds = region_info['bounds']
        
        logger.info(f"🛣️  下载 {region_info['name']} 道路网络数据")
        logger.info(f"📍 边界: {bounds}")
        
        try:
            # 从OSM获取道路网络
            min_lon, min_lat, max_lon, max_lat = bounds
            
            logger.info(f"正在从OSM获取 {network_type} 网络...")
            G = ox.graph_from_bbox(
                max_lat, min_lat, max_lon, min_lon,  # north, south, east, west
                network_type=network_type,
                simplify=True,
                retain_all=False
            )
            
            logger.info(f"获取到 {len(G.nodes)} 个节点, {len(G.edges)} 条边")
            
            # 转换为GeoDataFrame
            logger.info("转换为GeoDataFrame...")
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
            
            # 保存数据
            region_dir = self.output_dir / region_name
            region_dir.mkdir(exist_ok=True)
            
            # 保存节点
            nodes_file = region_dir / f"nodes_{network_type}.geojson"
            nodes_gdf.to_file(nodes_file, driver='GeoJSON')
            logger.info(f"节点数据已保存: {nodes_file}")
            
            # 保存边
            edges_file = region_dir / f"edges_{network_type}.geojson"
            edges_gdf.to_file(edges_file, driver='GeoJSON')
            logger.info(f"边数据已保存: {edges_file}")
            
            # 保存GraphML格式（用于后续路径规划）
            graphml_file = region_dir / f"network_{network_type}.graphml"
            ox.save_graphml(G, graphml_file)
            logger.info(f"网络图已保存: {graphml_file}")
            
            # 保存统计信息
            stats = {
                'region': region_name,
                'network_type': network_type,
                'bounds': bounds,
                'num_nodes': len(G.nodes),
                'num_edges': len(G.edges),
                'total_length_km': edges_gdf['length'].sum() / 1000,
                'avg_edge_length_m': edges_gdf['length'].mean(),
                'download_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            stats_file = region_dir / f"stats_{network_type}.json"
            import json
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ {region_info['name']} 道路网络下载完成")
            logger.info(f"   总长度: {stats['total_length_km']:.1f} km")
            logger.info(f"   平均边长: {stats['avg_edge_length_m']:.1f} m")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 下载 {region_name} 道路网络失败: {e}")
            return False
    
    def download_poi_data(self, region_name: str) -> bool:
        """
        下载兴趣点数据
        
        Args:
            region_name: 区域名称
            
        Returns:
            是否成功
        """
        if region_name not in self.regions:
            logger.error(f"未知区域: {region_name}")
            return False
        
        region_info = self.regions[region_name]
        bounds = region_info['bounds']
        
        logger.info(f"📍 下载 {region_info['name']} POI数据")
        
        try:
            min_lon, min_lat, max_lon, max_lat = bounds
            
            # 定义要下载的POI类型
            poi_types = {
                'amenity': ['hospital', 'school', 'police', 'fire_station'],
                'military': True,
                'tourism': ['attraction', 'viewpoint'],
                'landuse': ['military', 'industrial']
            }
            
            all_pois = []
            
            for poi_category, poi_values in poi_types.items():
                try:
                    logger.info(f"获取 {poi_category} POI...")
                    
                    if poi_values is True:
                        # 获取所有该类别的POI
                        tags = {poi_category: True}
                    else:
                        # 获取特定值的POI
                        tags = {poi_category: poi_values}
                    
                    pois = ox.features_from_bbox(
                        max_lat, min_lat, max_lon, min_lon,
                        tags=tags
                    )
                    
                    if not pois.empty:
                        pois['poi_category'] = poi_category
                        all_pois.append(pois)
                        logger.info(f"  获取到 {len(pois)} 个 {poi_category} POI")
                    
                except Exception as e:
                    logger.warning(f"获取 {poi_category} POI失败: {e}")
                    continue
            
            if all_pois:
                # 合并所有POI
                combined_pois = pd.concat(all_pois, ignore_index=True)
                
                # 保存POI数据
                region_dir = self.output_dir / region_name
                region_dir.mkdir(exist_ok=True)
                
                poi_file = region_dir / "pois.geojson"
                combined_pois.to_file(poi_file, driver='GeoJSON')
                
                logger.info(f"✅ POI数据已保存: {poi_file}")
                logger.info(f"   总计: {len(combined_pois)} 个POI")
                
                return True
            else:
                logger.warning(f"未获取到 {region_name} 的POI数据")
                return False
                
        except Exception as e:
            logger.error(f"❌ 下载 {region_name} POI数据失败: {e}")
            return False
    
    def download_region_data(self, region_name: str) -> bool:
        """下载指定区域的所有OSM数据"""
        logger.info(f"\n{'='*50}")
        logger.info(f"处理区域: {region_name}")
        logger.info(f"{'='*50}")
        
        success = True
        
        # 下载道路网络（驾驶）
        if not self.download_road_network(region_name, 'drive'):
            success = False
        
        # 稍作停顿
        time.sleep(2)
        
        # 下载POI数据
        if not self.download_poi_data(region_name):
            logger.warning(f"POI下载失败，但继续处理")
        
        return success
    
    def download_all_regions(self) -> Dict[str, bool]:
        """下载所有区域的OSM数据"""
        results = {}
        
        for region_name in self.regions.keys():
            try:
                results[region_name] = self.download_region_data(region_name)
                
                # 在区域之间稍作停顿，避免过于频繁的请求
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"处理区域 {region_name} 时出错: {e}")
                results[region_name] = False
        
        return results


def main():
    """主函数"""
    logger.info("🗺️  启动OSM数据下载器")
    logger.info("=" * 50)
    
    downloader = OSMDataDownloader()
    
    # 下载所有区域数据
    results = downloader.download_all_regions()
    
    # 统计结果
    logger.info("\n📊 下载结果统计:")
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    for region_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"  {region_name}: {status}")
    
    logger.info(f"\n总结: {successful}/{total} 个区域下载成功")
    
    if successful == total:
        logger.info("🎉 所有OSM数据下载完成！")
    else:
        logger.warning("⚠️  部分区域下载失败，请检查网络连接")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️  用户中断下载")
    except Exception as e:
        logger.error(f"\n❌ 程序异常: {e}")
        sys.exit(1)
