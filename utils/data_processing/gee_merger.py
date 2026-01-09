"""
GEE 数据分块合并模块
将下载的 GeoTIFF 分块合并为完整的栅格数据
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum

from config import cfg, get_path

logger = logging.getLogger(__name__)


class GEEMerger:
    """GEE 数据合并器"""
    
    def __init__(self, region: str = 'scottish_highlands'):
        """
        初始化合并器
        
        Args:
            region: 区域名称
        """
        self.region = region
        self.input_dir = Path(cfg.get('paths.raw_data.gee')) / region
        self.output_dir = Path(get_path('paths.processed.merged_gee')) / region
        self.output_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        self.target_resolution = cfg.get('gee.target_resolution', 30)
        
        logger.info(f"初始化 GEE 合并器 - 区域: {region}")
        logger.info(f"  输入目录: {self.input_dir}")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info(f"  目标分辨率: {self.target_resolution}m")
    
    def merge_data_type(self, data_type: str) -> Path:
        """
        合并指定类型的数据
        
        Args:
            data_type: 数据类型 (dem, slope, aspect, lulc)
        
        Returns:
            合并后文件路径
        """
        logger.info(f"=" * 60)
        logger.info(f"开始合并 {data_type.upper()} 数据")
        logger.info(f"=" * 60)
        
        # 查找所有分块文件
        chunk_dir = self.input_dir / data_type
        if not chunk_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {chunk_dir}")
        
        chunk_files = sorted(chunk_dir.glob('chunk_*.tif'))
        if not chunk_files:
            raise FileNotFoundError(f"未找到分块文件: {chunk_dir}/chunk_*.tif")
        
        logger.info(f"找到 {len(chunk_files)} 个分块文件")
        
        # 合并分块
        logger.info("正在合并分块...")
        merged_data, merged_transform = self._merge_chunks(chunk_files)
        
        # 获取参考元数据
        with rasterio.open(chunk_files[0]) as src:
            profile = src.profile.copy()
        
        # 更新元数据
        profile.update({
            'height': merged_data.shape[1],
            'width': merged_data.shape[2],
            'transform': merged_transform,
            'compress': 'lzw'
        })
        
        # 保存合并结果
        output_file = self.output_dir / f"{data_type}_merged.tif"
        logger.info(f"保存合并结果: {output_file}")
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(merged_data)
        
        # 如果需要重采样
        if data_type == 'lulc':
            logger.info(f"重采样 LULC 到 {self.target_resolution}m 分辨率...")
            output_file = self._resample_to_target(output_file, data_type)
        
        logger.info(f"✅ {data_type.upper()} 合并完成: {output_file}")
        
        # 打印统计信息
        self._print_statistics(output_file, data_type)
        
        return output_file
    
    def _merge_chunks(self, chunk_files: List[Path]) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        合并分块文件
        
        Args:
            chunk_files: 分块文件列表
        
        Returns:
            (合并后数据, 仿射变换矩阵)
        """
        # 打开所有分块
        src_files = [rasterio.open(f) for f in chunk_files]
        
        try:
            # 使用 rasterio.merge 合并
            merged_data, merged_transform = merge(
                src_files,
                resampling=ResamplingEnum.nearest
            )
            
            return merged_data, merged_transform
            
        finally:
            # 关闭所有文件
            for src in src_files:
                src.close()
    
    def _resample_to_target(self, input_file: Path, data_type: str) -> Path:
        """
        重采样到目标分辨率
        
        Args:
            input_file: 输入文件
            data_type: 数据类型
        
        Returns:
            重采样后文件路径
        """
        output_file = input_file.parent / f"{data_type}_merged_resampled.tif"
        
        with rasterio.open(input_file) as src:
            # 计算新的变换矩阵和尺寸
            transform, width, height = calculate_default_transform(
                src.crs,
                src.crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=self.target_resolution
            )
            
            # 更新元数据
            profile = src.profile.copy()
            profile.update({
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # 重采样方法（LULC 使用最近邻）
            resampling = ResamplingEnum.nearest if data_type == 'lulc' else ResamplingEnum.bilinear
            
            # 执行重采样
            with rasterio.open(output_file, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=resampling
                    )
        
        logger.info(f"  重采样完成: {width} x {height} @ {self.target_resolution}m")
        
        return output_file
    
    def _print_statistics(self, file_path: Path, data_type: str):
        """打印数据统计信息"""
        with rasterio.open(file_path) as src:
            data = src.read(1)
            
            logger.info(f"\n数据统计:")
            logger.info(f"  尺寸: {src.width} x {src.height}")
            logger.info(f"  分辨率: {src.res[0]:.2f}m x {src.res[1]:.2f}m")
            logger.info(f"  范围: {src.bounds}")
            logger.info(f"  数据类型: {src.dtypes[0]}")
            logger.info(f"  值域: {data.min():.2f} ~ {data.max():.2f}")
            logger.info(f"  均值: {data.mean():.2f}")
            
            if data_type == 'lulc':
                unique_values = np.unique(data)
                logger.info(f"  LULC 类别: {unique_values.tolist()}")
    
    def merge_all(self) -> dict:
        """
        合并所有数据类型
        
        Returns:
            合并后文件路径字典
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"开始合并 {self.region} 的所有 GEE 数据")
        logger.info("=" * 60 + "\n")
        
        results = {}
        data_types = ['dem', 'slope', 'aspect', 'lulc']
        
        for data_type in data_types:
            try:
                output_file = self.merge_data_type(data_type)
                results[data_type] = str(output_file)
                logger.info("")
            except Exception as e:
                logger.error(f"❌ {data_type.upper()} 合并失败: {e}")
                results[data_type] = None
        
        # 总结
        logger.info("=" * 60)
        logger.info("合并完成总结")
        logger.info("=" * 60)
        for data_type, file_path in results.items():
            status = "✅" if file_path else "❌"
            logger.info(f"{status} {data_type.upper()}: {file_path}")
        
        return results


def merge_gee_chunks(region: str = 'scottish_highlands') -> dict:
    """
    合并 GEE 分块数据（便捷函数）
    
    Args:
        region: 区域名称
    
    Returns:
        合并后文件路径字典
    """
    merger = GEEMerger(region)
    return merger.merge_all()


if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 执行合并
    results = merge_gee_chunks('scottish_highlands')
    
    print("\n" + "=" * 60)
    print("合并完成！")
    print("=" * 60)
    for data_type, file_path in results.items():
        if file_path:
            print(f"✅ {data_type.upper()}: {file_path}")
