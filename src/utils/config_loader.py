"""
配置文件加载器
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认使用项目根目录下的config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "src" / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键，支持点分割的嵌套键，如 'data.osm_dir'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_config(self) -> Dict[str, str]:
        """获取数据配置"""
        return self.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get('training', {})
    
    def get_vehicle_types(self) -> Dict[str, Dict[str, Any]]:
        """获取车辆类型配置"""
        return self.get('vehicle_types', {})
    
    def get_tactical_intents(self) -> Dict[str, Dict[str, Any]]:
        """获取战术意图配置"""
        return self.get('tactical_intents', {})
    
    def get_lulc_classes(self) -> Dict[int, Dict[str, Any]]:
        """获取土地覆盖分类配置"""
        return self.get('environment.lulc_classes', {})
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            updates: 要更新的配置项
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def save_config(self, output_path: str = None):
        """
        保存配置文件
        
        Args:
            output_path: 输出路径，默认覆盖原文件
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)


# 全局配置实例
config = ConfigLoader()
